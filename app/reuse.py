"""旧草案复用：换起点重扫时的上界折算与平移直认。

背景
====
对同一纹样换一个起点重新扫描残片，可能多认出几个完好色点。若手上已有一份
**已证最优**的旧草案，在满足下列条件时可避免重新求解：

1. 新旧重复单元尺寸相同（``unit_w``/``unit_h`` 一致）；
2. 新旧参考图的 2×2 图块**种类相同**，且每种图块的频次成**同一比例**
   ``r = new_count / old_count``（正有理数）；
3. 存在一个环绕平移 ``d1``，使旧草案的每个固定色点平移后都落在新残片
   投影的同色点上（旧约束被新约束覆盖）。

折算上界
========
条件 3 意味着新问题的任一可行纹版在平移回旧坐标后仍满足旧固定约束，因而
其旧频次得分不超过旧最优分 ``S_old``；由条件 2，新频次得分恒为旧得分的
``r`` 倍，故本次最优分不超过 ``r × S_old``。取整数上界
``floor(r × S_old)``，并与通用上界 ``W×H×最高新频次`` 取小。

平移直认
========
若还存在（允许与 ``d1`` **不同位移**的）环绕平移 ``d2``，使旧纹版平移后
符合全部新色点，则平移纹版是新问题的可行解；环绕平移只置换 2×2 图块的
位置，其新得分恰为 ``r × S_old``，达到折算上界，可直接标记为已证最优、
跳过求解。两个平移都找不到时，旧草案不可用，交由常规求解流程处理。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from math import gcd
from typing import Optional

from .images import RGB, TileKey
from .solver import (
    STATUS_OPTIMAL,
    SolveResult,
    _max_freq,
    _tile_usage,
)

# 复用判定受阻原因（机器可读码）
REASON_ELIGIBLE = "eligible"
REASON_OLD_NOT_OPTIMAL = "old_not_optimal"
REASON_SIZE_MISMATCH = "size_mismatch"
REASON_TILE_KIND_MISMATCH = "tile_kind_mismatch"
REASON_RATIO_MISMATCH = "frequency_ratio_mismatch"
REASON_NO_COVERING_SHIFT = "no_covering_shift"

_REASON_MESSAGES = {
    REASON_ELIGIBLE: "满足折算条件，可使用旧草案折算上界",
    REASON_OLD_NOT_OPTIMAL: "旧草案未证最优或缺少候选纹版，不能折算",
    REASON_SIZE_MISMATCH: "新旧重复单元尺寸不同，不能折算",
    REASON_TILE_KIND_MISMATCH: "新旧参考图 2×2 图块种类不同，不能折算",
    REASON_RATIO_MISMATCH: "图块频次不成同一比例，不能折算",
    REASON_NO_COVERING_SHIFT: "旧色点不存在被新残片同色覆盖的环绕平移，不能折算",
}


@dataclass
class ReuseAssessment:
    """一次旧草案复用评估的结果。"""

    eligible: bool
    reason: str
    message: str
    # 频次比例 r = numerator / denominator = new_count / old_count
    ratio: Optional[tuple[int, int]] = None
    # 旧色点覆盖平移 d1
    point_shift: Optional[tuple[int, int]] = None
    # 折算上界（已与通用上界取小）
    scaled_bound: Optional[int] = None
    generic_bound: Optional[int] = None
    # 旧纹版直认平移 d2 及平移后的纹版/得分
    grid_shift: Optional[tuple[int, int]] = None
    shifted_grid: Optional[tuple[tuple[RGB, ...], ...]] = None
    shifted_score: Optional[int] = None
    elapsed_seconds: float = 0.0

    @property
    def can_skip_solver(self) -> bool:
        return self.eligible and self.shifted_grid is not None


# --------------------------------------------------------------------------- #
# 纯函数：比例、平移
# --------------------------------------------------------------------------- #

def proportional_ratio(
    old_counts: dict[TileKey, int],
    new_counts: dict[TileKey, int],
) -> Optional[tuple[int, int]]:
    """若两图块表种类相同且频次成同一比例，返回约分后的 (分子, 分母)。

    比例 ``r = new_count / old_count``；种类不同或比例不一时返回 ``None``。
    """
    if set(old_counts) != set(new_counts) or not old_counts:
        return None
    first = next(iter(old_counts))
    num = new_counts[first]
    den = old_counts[first]
    if den <= 0 or num <= 0:
        return None
    common = gcd(num, den)
    num, den = num // common, den // common
    for key in old_counts:
        # new_count * den == old_count * num，全程整数避免浮点误差
        if new_counts[key] * den != old_counts[key] * num:
            return None
    return num, den


def shift_points(
    points: dict[tuple[int, int], RGB],
    dx: int,
    dy: int,
    unit_w: int,
    unit_h: int,
) -> dict[tuple[int, int], RGB]:
    """把固定色点整体做环绕平移。"""
    return {
        ((x + dx) % unit_w, (y + dy) % unit_h): color
        for (x, y), color in points.items()
    }


def find_point_shift(
    old_fixed: dict[tuple[int, int], RGB],
    new_fixed: dict[tuple[int, int], RGB],
    unit_w: int,
    unit_h: int,
) -> Optional[tuple[int, int]]:
    """寻找 d1：旧色点平移后全部被新残片投影的同色点覆盖。

    旧色点为空时任意平移都满足，确定性地返回 (0, 0)。
    """
    if not old_fixed:
        return 0, 0
    for dy in range(unit_h):
        for dx in range(unit_w):
            moved = shift_points(old_fixed, dx, dy, unit_w, unit_h)
            if all(
                cell in new_fixed and new_fixed[cell] == color
                for cell, color in moved.items()
            ):
                return dx, dy
    return None


def shift_grid(
    grid: tuple[tuple[RGB, ...], ...],
    dx: int,
    dy: int,
    unit_w: int,
    unit_h: int,
) -> tuple[tuple[RGB, ...], ...]:
    """把纹版整体做环绕平移：新格 (x,y) 取旧格 (x-dx, y-dy)。"""
    return tuple(
        tuple(
            grid[(y - dy) % unit_h][(x - dx) % unit_w]
            for x in range(unit_w)
        )
        for y in range(unit_h)
    )


def grid_matches_points(
    grid: tuple[tuple[RGB, ...], ...],
    fixed: dict[tuple[int, int], RGB],
) -> bool:
    return all(grid[y][x] == color for (x, y), color in fixed.items())


def find_grid_shift(
    grid: tuple[tuple[RGB, ...], ...],
    fixed: dict[tuple[int, int], RGB],
    unit_w: int,
    unit_h: int,
) -> Optional[tuple[int, int]]:
    """寻找 d2：纹版平移后符合全部新色点。"""
    for dy in range(unit_h):
        for dx in range(unit_w):
            moved = shift_grid(grid, dx, dy, unit_w, unit_h)
            if grid_matches_points(moved, fixed):
                return dx, dy
    return None


# --------------------------------------------------------------------------- #
# 评估入口
# --------------------------------------------------------------------------- #

def assess_reuse(
    *,
    old_status: str,
    old_score: Optional[int],
    old_grid: Optional[tuple[tuple[RGB, ...], ...]],
    old_fixed: dict[tuple[int, int], RGB],
    old_tile_counts: dict[TileKey, int],
    unit_w: int,
    unit_h: int,
    new_tile_counts: dict[TileKey, int],
    new_fixed: dict[tuple[int, int], RGB],
) -> ReuseAssessment:
    """评估旧草案能否用于本次重扫。"""
    started = time.perf_counter()

    def finish(assessment: ReuseAssessment) -> ReuseAssessment:
        assessment.elapsed_seconds = time.perf_counter() - started
        return assessment

    def reject(reason: str) -> ReuseAssessment:
        return finish(ReuseAssessment(
            eligible=False, reason=reason,
            message=_REASON_MESSAGES[reason]))

    # 1. 旧草案必须已证最优且留有候选纹版
    if old_status != STATUS_OPTIMAL or old_grid is None or old_score is None:
        return reject(REASON_OLD_NOT_OPTIMAL)

    # 2. 新旧单元必须同尺寸（grid 的行列尺寸即旧单元尺寸）
    if (len(old_grid[0]), len(old_grid)) != (unit_w, unit_h):
        return reject(REASON_SIZE_MISMATCH)

    # 3. 图块种类相同、频次成同一比例
    ratio = proportional_ratio(old_tile_counts, new_tile_counts)
    if ratio is None:
        if set(old_tile_counts) != set(new_tile_counts):
            return reject(REASON_TILE_KIND_MISMATCH)
        return reject(REASON_RATIO_MISMATCH)
    num, den = ratio

    # 4. 旧色点可经环绕平移被新残片同色覆盖
    point_shift = find_point_shift(old_fixed, new_fixed, unit_w, unit_h)
    if point_shift is None:
        return reject(REASON_NO_COVERING_SHIFT)

    # 折算上界：floor(r × S_old)，与通用上界取小
    scaled_bound = (num * old_score) // den
    generic_bound = unit_w * unit_h * _max_freq(new_tile_counts)
    scaled_bound = min(scaled_bound, generic_bound)

    assessment = ReuseAssessment(
        eligible=True,
        reason=REASON_ELIGIBLE,
        message=_REASON_MESSAGES[REASON_ELIGIBLE],
        ratio=ratio,
        point_shift=point_shift,
        scaled_bound=scaled_bound,
        generic_bound=generic_bound,
    )

    # 5. 旧纹版能否（用另一个允许不同的位移）直接符合全部新色点
    grid_shift = find_grid_shift(old_grid, new_fixed, unit_w, unit_h)
    if grid_shift is not None:
        moved = shift_grid(old_grid, grid_shift[0], grid_shift[1],
                           unit_w, unit_h)
        from .solver import _score_rgb_grid
        moved_score = _score_rgb_grid(
            new_tile_counts, moved, unit_w, unit_h)
        assessment.grid_shift = grid_shift
        assessment.shifted_grid = moved
        assessment.shifted_score = moved_score
    return finish(assessment)


def build_skip_result(
    assessment: ReuseAssessment,
    new_tile_counts: dict[TileKey, int],
    unit_w: int,
    unit_h: int,
) -> SolveResult:
    """把平移直认的纹版封装成「已证最优」求解结果（不调用求解器）。"""
    assert assessment.shifted_grid is not None
    assert assessment.scaled_bound is not None
    grid = assessment.shifted_grid
    result = SolveResult(
        status=STATUS_OPTIMAL,
        grid=grid,
        score=assessment.shifted_score,
        upper_bound=assessment.scaled_bound,
        elapsed_seconds=assessment.elapsed_seconds,
        solver_name="reuse-toroidal-shift",
    )
    result.tile_usage = _tile_usage(
        new_tile_counts, grid, unit_w, unit_h)
    return result
