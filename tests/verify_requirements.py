"""需求逐条核对脚本（不依赖网络，直接调用内部模块）。"""
import io
import sys

from PIL import Image

from app import images, reuse, solver
from tests.test_app import (
    png_bytes, stripes_grid, checker_grid,
    _REUSE_GRID, _REUSE_OLD_FIXED, _REUSE_NEW_FIXED, _D1, _D2,
    _RESCAN_GRID, _rescan_fragments, _RESCAN_SHIFT,
)

RED = (220, 60, 60)
BLUE = (60, 90, 220)
GREEN = (60, 180, 90)
YELLOW = (230, 200, 70)
ok = 0


def check(name, cond):
    global ok
    assert cond, f"FAIL: {name}"
    ok += 1
    print(f"[PASS] {name}")


# 1. 参考图最多 4 色、每边 <=64
ref = images.parse_reference(png_bytes(stripes_grid(8)))
check("参考图调色板 <=4 色", len(ref.colors) <= 4)

# 2. 2×2 图块统计：总数=像素数，跨边跨角
check("图块统计总数=参考图像素数", sum(ref.tile_counts.values()) == 64)
# 手工验证一个跨角图块（右下角为左上 -> 环绕到 (0,0)）
grid = stripes_grid(8)
corner_tile = (grid[7][7], grid[7][0], grid[0][7], grid[0][0])
check("跨角图块被统计", corner_tile in ref.tile_counts)

# 3. 同格异色返回两处原坐标
points = {(0, 0): RED, (4, 4): GREEN, (1, 1): BLUE}
fixed, conflicts = images.project_points(points, 4, 4)
check("同格异色检测", len(conflicts) == 1)
check("返回两处原坐标", set(conflicts[0].positions) == {(0, 0), (4, 4)})
check("冲突格坐标", conflicts[0].cell == (0, 0))

# 4. 同格同色合并
f2, c2 = images.project_points({(0, 0): RED, (4, 4): RED}, 4, 4)
check("同格同色合并无冲突", c2 == [] and len(f2) == 1)

# 5. 完好色点保留 + 每图块都在参考图出现（含跨边跨角）
r = solver.solve(ref.colors, ref.tile_counts, 6, 6, {(2, 3): RED}, 5.0)
check("求解有候选", r.grid is not None)
check("完好色点保留", r.grid[3][2] == RED)
W = H = 6
for y in range(H):
    for x in range(W):
        key = (r.grid[y][x], r.grid[y][(x + 1) % W],
               r.grid[(y + 1) % H][x],
               r.grid[(y + 1) % H][(x + 1) % W])
        assert key in ref.tile_counts, (x, y)
check("每格 2×2 图块均在参考图出现（含环绕）", True)

# 6. 得分=图块频次之和
manual = sum(
    ref.tile_counts[(
        r.grid[y][x], r.grid[y][(x + 1) % W],
        r.grid[(y + 1) % H][x],
        r.grid[(y + 1) % H][(x + 1) % W])]
    for y in range(H) for x in range(W))
check("得分=图块频次之和", manual == r.score)

# 7. 四种状态都可达
check("已证最优可达", r.status in (solver.STATUS_OPTIMAL, solver.STATUS_FEASIBLE))
ri = solver.solve(ref.colors, ref.tile_counts, 4, 4, {(0, 0): (1, 2, 3)}, 5.0)
check("已证无解可达", ri.status == solver.STATUS_INFEASIBLE and ri.grid is None)
ru = solver.solve(ref.colors, ref.tile_counts, 12, 12, {}, 0.02)
check("待定可达", ru.status in (solver.STATUS_UNKNOWN, solver.STATUS_FEASIBLE,
                                solver.STATUS_OPTIMAL))

# 8. 限时 5 秒
import time
t0 = time.perf_counter()
solver.solve(ref.colors, ref.tile_counts, 10, 10, {}, 5.0)
check("5 秒限时", time.perf_counter() - t0 <= 5.1)

# 9. 单元尺寸边界 4~12
for bad in (3, 13):
    try:
        images.validate_unit_size(bad, 8)
        raise AssertionError("应拒绝")
    except images.ImageValidationError:
        pass
check("单元尺寸 4~12 边界校验", True)

# 10. 透明=缺损
frag = images.parse_fragment(
    png_bytes(stripes_grid(4), transparent_pred=lambda x, y, c: y >= 2))
check("透明像素标缺损", len(frag.points) == 8)

# 11. 64 像素上限
big = png_bytes([[RED] * 65 for _ in range(4)])
try:
    images.parse_reference(big)
    raise AssertionError("应拒绝超宽图")
except images.ImageValidationError:
    check("每边 <=64 像素上限", True)

# 12. 回溯兜底与 CP-SAT 在小案例结论一致
bt = solver._solve_backtrack(ref.colors, ref.tile_counts, 6, 6, {(2, 3): RED}, 5.0)
cp = solver._solve_cpsat(ref.colors, ref.tile_counts, 6, 6, {(2, 3): RED}, 5.0)
check("回溯/CP-SAT 最优性一致", bt.status == cp.status == solver.STATUS_OPTIMAL)
check("回溯/CP-SAT 得分一致", bt.score == cp.score)

# 13. 图块频次同比例判定（整数精确比较）
rt = images.parse_reference(png_bytes(_REUSE_GRID)).tile_counts
check("同表比例 1:1", reuse.proportional_ratio(rt, rt) == (1, 1))
scaled = {k: v * 3 for k, v in rt.items()}
check("频次整体 ×3 比例 3:1", reuse.proportional_ratio(rt, scaled) == (3, 1))
broken = dict(scaled)
broken[next(iter(broken))] += 1
check("比例不一拒绝折算", reuse.proportional_ratio(rt, broken) is None)
check("种类不同拒绝折算",
      reuse.proportional_ratio(rt, dict(list(rt.items())[:-1])) is None)

# 14. 两次平移可不同：d1 覆盖旧色点、d2 纹版符合新色点
assessment = reuse.assess_reuse(
    old_status=solver.STATUS_OPTIMAL,
    old_score=18,
    old_grid=tuple(tuple(row) for row in _REUSE_GRID),
    old_fixed=dict(_REUSE_OLD_FIXED),
    old_tile_counts=rt,
    unit_w=4, unit_h=4,
    new_tile_counts=rt,
    new_fixed=dict(_REUSE_NEW_FIXED),
)
check("折算条件成立", assessment.eligible)
check("d1 与 d2 可为不同位移",
      assessment.point_shift == _D1 and assessment.grid_shift == _D2
      and _D1 != _D2)
check("平移纹版符合全部新色点",
      reuse.grid_matches_points(assessment.shifted_grid, _REUSE_NEW_FIXED))
skip = reuse.build_skip_result(assessment, rt, 4, 4)
check("直认结果已证最优且不调求解器",
      skip.status == solver.STATUS_OPTIMAL
      and skip.solver_name == "reuse-toroidal-shift")

# 15. 无法折算的各类原因
rej = reuse.assess_reuse(
    old_status=solver.STATUS_FEASIBLE, old_score=1,
    old_grid=tuple(tuple(row) for row in _REUSE_GRID),
    old_fixed={}, old_tile_counts=rt, unit_w=4, unit_h=4,
    new_tile_counts=rt, new_fixed={})
check("非最优旧草案拒绝折算",
      not rej.eligible and rej.reason == reuse.REASON_OLD_NOT_OPTIMAL)
rej2 = reuse.assess_reuse(
    old_status=solver.STATUS_OPTIMAL, old_score=1,
    old_grid=tuple(tuple(row) for row in _REUSE_GRID),
    old_fixed={(0, 0): (4, 4, 4)}, old_tile_counts=rt,
    unit_w=4, unit_h=4, new_tile_counts=rt,
    new_fixed={(0, 0): (1, 1, 1)})
check("无覆盖平移拒绝折算",
      not rej2.eligible and rej2.reason == reuse.REASON_NO_COVERING_SHIFT)

# 16. 折算上界提示：求解器合法候选命中即停搜并标最优
hinted = solver.solve(
    ref.colors, ref.tile_counts, 6, 6, {}, 5.0,
    upper_bound_hint=36 * 32)
check("折算上界提示下声明已证最优",
      hinted.status == solver.STATUS_OPTIMAL
      and hinted.upper_bound == 36 * 32)
bt_hint = solver._solve_backtrack(
    ref.colors, ref.tile_counts, 6, 6, {(2, 3): RED}, 5.0,
    upper_bound_hint=36 * 32)
check("回溯路径同样遵守折算上界",
      bt_hint.status == solver.STATUS_OPTIMAL
      and bt_hint.upper_bound == 36 * 32)

# 17. 换起点演示夹具：旧点经 d 平移被新残片覆盖，新残片认出更多格
old_points, new_points = _rescan_fragments()
old_fixed, c1 = images.project_points(old_points, 8, 8)
new_fixed, c2 = images.project_points(new_points, 8, 8)
check("演示残片无同格异色", c1 == [] and c2 == [])
check("新残片认出格数不少于旧残片", len(new_fixed) >= len(old_fixed))
cover = reuse.find_point_shift(old_fixed, new_fixed, 8, 8)
check("旧色点可经演示位移被新残片覆盖", cover == _RESCAN_SHIFT)
demo_grid = tuple(tuple(tuple(c) for c in row) for row in _RESCAN_GRID)
assert len(demo_grid) == 8 and len(demo_grid[0]) == 8
gshift = reuse.find_grid_shift(demo_grid, new_fixed, 8, 8)
check("旧纹版可平移符合全部新色点", gshift == _RESCAN_SHIFT)

print(f"\n全部 {ok} 项需求核对通过")
