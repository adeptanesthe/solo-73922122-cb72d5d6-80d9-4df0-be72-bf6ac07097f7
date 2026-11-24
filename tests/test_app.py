"""端到端与单元测试。

运行：``PYTHONPATH=/tmp/pylibs:/workspace python3 -m pytest tests/ -v``
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import images, reuse, solver, storage
from app.main import app, SOLVE_TIME_LIMIT

RED = (220, 60, 60)
BLUE = (60, 90, 220)
GREEN = (60, 180, 90)
YELLOW = (230, 200, 70)


# --------------------------------------------------------------------------- #
# 构造工具
# --------------------------------------------------------------------------- #

def png_bytes(grid_xy, size=None, transparent_pred=None) -> bytes:
    height = len(grid_xy)
    width = len(grid_xy[0])
    img = Image.new("RGBA", (width, height))
    for y, row in enumerate(grid_xy):
        for x, color in enumerate(row):
            if transparent_pred and transparent_pred(x, y, color):
                img.putpixel((x, y), (0, 0, 0, 0))
            else:
                img.putpixel((x, y), (*color, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def stripes_grid(size=8):
    return [[RED if x % 2 == 0 else BLUE for x in range(size)]
            for _ in range(size)]


def checker_grid(size=8):
    return [[GREEN if (x + y) % 2 == 0 else YELLOW for x in range(size)]
            for y in range(size)]


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db = tmp_path / "test_drafts.db"
    monkeypatch.setattr(storage, "DEFAULT_DB_PATH", db)
    # main 模块在 lifespan 中读取 storage.DEFAULT_DB_PATH
    with TestClient(app) as c:
        yield c


# --------------------------------------------------------------------------- #
# images 单元测试
# --------------------------------------------------------------------------- #

def test_reference_rejects_transparent():
    data = png_bytes(stripes_grid(4),
                     transparent_pred=lambda x, y, c: (x, y) == (0, 0))
    with pytest.raises(images.ImageValidationError):
        images.parse_reference(data)


def test_reference_too_many_colors():
    grid = [[RED, BLUE], [GREEN, YELLOW]]
    # 4 色合法；再造第 5 色
    grid5 = [row + [(1, 2, 3)] for row in grid]
    with pytest.raises(images.ImageValidationError):
        images.parse_reference(png_bytes(grid5))


def test_reference_four_colors_ok():
    grid = [[RED, BLUE], [GREEN, YELLOW]]
    ref = images.parse_reference(png_bytes(grid))
    assert len(ref.colors) == 4


def test_size_limit_enforced():
    grid = [[RED] * 65 for _ in range(8)]
    with pytest.raises(images.ImageValidationError):
        images.parse_reference(png_bytes(grid))


def test_unit_size_validation():
    with pytest.raises(images.ImageValidationError):
        images.validate_unit_size(3, 8)
    with pytest.raises(images.ImageValidationError):
        images.validate_unit_size(8, 13)
    images.validate_unit_size(4, 12)


def test_tile_counts_toroidal_and_total():
    """8×8 竖条纹：共 64 个图块位置（含跨边跨角），仅 2 种图块。"""
    ref = images.parse_reference(png_bytes(stripes_grid(8)))
    assert sum(ref.tile_counts.values()) == 8 * 8
    assert len(ref.tile_counts) == 2
    for key, count in ref.tile_counts.items():
        assert count == 32


def test_project_points_conflict():
    points = {(0, 0): RED, (4, 0): GREEN, (1, 1): BLUE}
    fixed, conflicts = images.project_points(points, 4, 4)
    assert len(conflicts) == 1
    c = conflicts[0]
    assert c.cell == (0, 0)
    assert c.positions == ((0, 0), (4, 0))
    assert set(c.colors) == {RED, GREEN}
    # 无冲突格仍然正常投影
    assert fixed[(1, 1)] == BLUE


def test_project_points_same_color_merges():
    points = {(0, 0): RED, (4, 4): RED}
    fixed, conflicts = images.project_points(points, 4, 4)
    assert conflicts == []
    assert fixed == {(0, 0): RED}


def test_fragment_transparent_means_missing():
    grid = stripes_grid(4)
    data = png_bytes(grid, transparent_pred=lambda x, y, c: y >= 2)
    frag = images.parse_fragment(data)
    assert all(y < 2 for (x, y) in frag.points)
    assert len(frag.points) == 8


# --------------------------------------------------------------------------- #
# solver 单元测试
# --------------------------------------------------------------------------- #

def test_solver_stripes_optimal_and_preserves_points():
    ref = images.parse_reference(png_bytes(stripes_grid(8)))
    fixed = {(0, 0): RED, (1, 0): BLUE, (0, 1): RED}
    result = solver.solve(ref.colors, ref.tile_counts, 6, 6, fixed, 5.0)
    assert result.status == solver.STATUS_OPTIMAL
    assert result.grid is not None
    # 固定色点保留
    for (x, y), color in fixed.items():
        assert result.grid[y][x] == color
    # 每个 2×2 图块都必须在参考图出现
    keys = set(ref.tile_counts)
    w = h = 6
    for y in range(h):
        for x in range(w):
            key = (
                result.grid[y][x],
                result.grid[y][(x + 1) % w],
                result.grid[(y + 1) % h][x],
                result.grid[(y + 1) % h][(x + 1) % w],
            )
            assert key in keys
    # 满分：所有图块频次之和（条纹每种图块频次 32）
    assert result.score == 6 * 6 * 32


def test_solver_infeasible_on_foreign_color():
    ref = images.parse_reference(png_bytes(stripes_grid(8)))
    fixed = {(0, 0): (1, 2, 3)}  # 调色板外颜色
    result = solver.solve(ref.colors, ref.tile_counts, 4, 4, fixed, 5.0)
    assert result.status == solver.STATUS_INFEASIBLE
    assert result.grid is None


def test_solver_conflicting_fixed_points_infeasible():
    ref = images.parse_reference(png_bytes(checker_grid(8)))
    # 棋盘格约束下强制相邻同色 -> 无合法图块
    fixed = {(0, 0): GREEN, (1, 0): GREEN, (0, 1): GREEN}
    result = solver.solve(ref.colors, ref.tile_counts, 4, 4, fixed, 5.0)
    assert result.status == solver.STATUS_INFEASIBLE


def test_solver_score_is_maximum_frequency_sum():
    """目标函数取图块频次之和最高：构造两种频次不同的合法图块并验证。"""
    # 参考图 4×4：3 个 AABB 图块 + 1 个 ABAB 图块（跨角位置）
    A, B = RED, BLUE
    grid = [
        [A, A, A, A],
        [A, B, B, A],
        [A, B, B, A],
        [A, A, A, A],
    ]
    ref = images.parse_reference(png_bytes(grid))
    result = solver.solve(ref.colors, ref.tile_counts, 4, 4, {}, 5.0)
    assert result.status == solver.STATUS_OPTIMAL
    # 最优解分数应等于「单元格数 × 最高频图块频次」上界，若该上界可达
    max_freq = max(ref.tile_counts.values())
    assert result.score is not None and result.score > 0
    assert result.score <= 16 * max_freq
    # 全 A 板的每个图块是四角 A，检查其频次确实被计入目标
    all_a = (A, A, A, A)
    if all_a in ref.tile_counts:
        # 全 A 纹版可行且应是最优之一
        uniform = solver.solve(ref.colors, ref.tile_counts, 4, 4,
                               {(x, y): A for y in range(4) for x in range(4)},
                               5.0)
        assert uniform.status == solver.STATUS_OPTIMAL
        assert uniform.score == 16 * ref.tile_counts[all_a]


def test_solver_time_limit_respected():
    ref = images.parse_reference(png_bytes(checker_grid(8)))
    result = solver.solve(ref.colors, ref.tile_counts, 12, 12, {}, 5.0)
    assert result.elapsed_seconds <= SOLVE_TIME_LIMIT + 2.0  # 含调度余量
    assert result.status in {
        solver.STATUS_OPTIMAL, solver.STATUS_FEASIBLE,
        solver.STATUS_INFEASIBLE, solver.STATUS_UNKNOWN}


def test_odd_cycle_stripes_is_infeasible():
    """奇宽环形下二色交替无法闭合：5×5 条纹纹版应已证无解。"""
    ref = images.parse_reference(png_bytes(stripes_grid(8)))
    result = solver.solve(ref.colors, ref.tile_counts, 5, 5, {}, 5.0)
    assert result.status == solver.STATUS_INFEASIBLE
    assert result.grid is None


def test_backtrack_matches_cpsat_on_small_case():
    ref = images.parse_reference(png_bytes(stripes_grid(8)))
    fixed = {(2, 3): RED}
    bt = solver._solve_backtrack(ref.colors, ref.tile_counts, 6, 6, fixed, 5.0)
    cp = solver._solve_cpsat(ref.colors, ref.tile_counts, 6, 6, fixed, 5.0)
    assert bt.status == cp.status == solver.STATUS_OPTIMAL
    assert bt.score == cp.score
    for (x, y), color in fixed.items():
        assert bt.grid[y][x] == color


# --------------------------------------------------------------------------- #
# API 端到端
# --------------------------------------------------------------------------- #

def _files(ref_grid, frag_grid=None, frag_transparent=None):
    files = {
        "reference": ("ref.png", png_bytes(ref_grid), "image/png"),
    }
    if frag_grid is not None:
        files["fragment"] = (
            "frag.png",
            png_bytes(frag_grid, transparent_pred=frag_transparent),
            "image/png",
        )
    return files


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_create_draft_optimal_flow(client):
    ref_grid = stripes_grid(8)
    frag_grid = [row[:] for row in ref_grid]
    resp = client.post(
        "/api/drafts",
        files=_files(ref_grid, frag_grid,
                     frag_transparent=lambda x, y, c: (x >= 4 and y >= 4)),
        data={"unit_w": 6, "unit_h": 6},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "optimal"
    assert body["score"] == 36 * 32
    assert "draft_id" in body
    # 图块统计总和等于参考图像素数
    assert sum(t["count"] for t in body["tile_counts"]) == 64

    draft_id = body["draft_id"]

    # 查看草案
    got = client.get(f"/api/drafts/{draft_id}")
    assert got.status_code == 200
    assert got.json()["status"] == "optimal"

    # 领取正式纹版
    png = client.get(f"/api/drafts/{draft_id}/pattern.png")
    assert png.status_code == 200
    assert png.headers["content-type"] == "image/png"
    im = Image.open(io.BytesIO(png.content))
    assert im.size == (6, 6)
    assert im.getchannel("A").getextrema() == (255, 255)


def test_create_draft_conflict_returns_coordinates(client):
    size = 6
    ref_grid = stripes_grid(size)
    frag_grid = [[(0, 0, 0) for _ in range(size)] for _ in range(size)]
    transparent = [[True] * size for _ in range(size)]
    # 手工铺像素
    frag_img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    frag_img.putpixel((0, 0), (*RED, 255))
    frag_img.putpixel((4, 0), (*GREEN, 255))
    buf = io.BytesIO()
    frag_img.save(buf, format="PNG")

    resp = client.post(
        "/api/drafts",
        files={
            "reference": ("ref.png", png_bytes(ref_grid), "image/png"),
            "fragment": ("frag.png", buf.getvalue(), "image/png"),
        },
        data={"unit_w": 4, "unit_h": 4},
    )
    assert resp.status_code == 409
    body = resp.json()["detail"]
    assert body["error"] == "cell_conflict"
    conflict = body["conflicts"][0]
    assert conflict["cell"] == {"x": 0, "y": 0}
    positions = {(p["x"], p["y"]) for p in conflict["fragment_positions"]}
    assert positions == {(0, 0), (4, 0)}


def test_add_point_then_download(client):
    ref_grid = stripes_grid(8)
    frag_grid = [row[:] for row in ref_grid]
    resp = client.post(
        "/api/drafts",
        files=_files(ref_grid, frag_grid,
                     frag_transparent=lambda x, y, c: True),  # 残片全空
        data={"unit_w": 6, "unit_h": 6},
    )
    assert resp.status_code == 200
    draft_id = resp.json()["draft_id"]

    # 补录一个完好色点（用格坐标）
    added = client.post(
        f"/api/drafts/{draft_id}/points",
        data={"cell_x": 0, "cell_y": 0, "color": "220,60,60"},
    )
    assert added.status_code == 200, added.text
    body = added.json()
    assert body["added"]["cell"] == {"x": 0, "y": 0}
    assert body["candidate_grid"][0][0] == list(RED)

    # 用残片原坐标补录（投影到同格同色，应幂等合并）
    same = client.post(
        f"/api/drafts/{draft_id}/points",
        data={"x": 6, "y": 6, "color": "220,60,60"},  # 6%6=0
    )
    assert same.status_code == 200
    assert same.json()["fixed_point_count"] == 1

    # 正式纹版仍可领取且包含该色点
    png = client.get(f"/api/drafts/{draft_id}/pattern.png")
    assert png.status_code == 200
    im = Image.open(io.BytesIO(png.content)).convert("RGB")
    assert im.getpixel((0, 0)) == RED


def test_add_point_conflict_keeps_draft(client):
    ref_grid = stripes_grid(8)
    resp = client.post(
        "/api/drafts",
        files=_files(ref_grid, ref_grid,
                     frag_transparent=lambda x, y, c: (x, y) != (0, 0)),
        data={"unit_w": 4, "unit_h": 4},
    )
    draft_id = resp.json()["draft_id"]

    conflict = client.post(
        f"/api/drafts/{draft_id}/points",
        data={"cell_x": 0, "cell_y": 0, "color": "60,90,220"},  # 格内已是 RED
    )
    assert conflict.status_code == 409
    detail = conflict.json()["detail"]
    assert detail["conflict"]["cell"] == {"x": 0, "y": 0}
    assert detail["conflict"]["existing_color"] == list(RED)
    assert detail["conflict"]["incoming_color"] == list(BLUE)

    # 草案保留：色点集合未变
    draft = client.get(f"/api/drafts/{draft_id}").json()
    assert draft["fixed_points"]["0,0"] == list(RED)
    assert len(draft["fixed_points"]) == 1


def test_infeasible_draft_not_saved(client):
    # 残片只给一个调色板外颜色点
    frag = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    frag.putpixel((0, 0), (1, 2, 3, 255))
    buf = io.BytesIO()
    frag.save(buf, format="PNG")
    resp = client.post(
        "/api/drafts",
        files={
            "reference": ("ref.png", png_bytes(stripes_grid(8)), "image/png"),
            "fragment": ("frag.png", buf.getvalue(), "image/png"),
        },
        data={"unit_w": 4, "unit_h": 4},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "infeasible"
    assert "draft_id" not in body


def test_pattern_download_requires_optimal(client):
    ref_grid = checker_grid(8)
    resp = client.post(
        "/api/drafts",
        files=_files(ref_grid, ref_grid,
                     frag_transparent=lambda x, y, c: True),
        data={"unit_w": 4, "unit_h": 4},
    )
    draft_id = resp.json()["draft_id"]
    # 手动把库内状态改成 feasible 模拟未证最优
    from app.main import _store
    with _store._lock:
        _store._conn.execute(
            "UPDATE drafts SET status='feasible' WHERE id=?", (draft_id,))
        _store._conn.commit()
    blocked = client.get(f"/api/drafts/{draft_id}/pattern.png")
    assert blocked.status_code == 409
    assert blocked.json()["detail"]["error"] == "not_optimal"


def test_bad_unit_size_rejected(client):
    resp = client.post(
        "/api/drafts",
        files=_files(stripes_grid(8), stripes_grid(8)),
        data={"unit_w": 3, "unit_h": 4},
    )
    assert resp.status_code == 400


def test_unknown_draft_404(client):
    assert client.get("/api/drafts/9999").status_code == 404
    assert client.get("/api/drafts/9999/pattern.png").status_code == 404


def test_add_point_out_of_palette_rejected(client):
    resp = client.post(
        "/api/drafts",
        files=_files(stripes_grid(8), stripes_grid(8),
                     frag_transparent=lambda x, y, c: True),
        data={"unit_w": 4, "unit_h": 4},
    )
    draft_id = resp.json()["draft_id"]
    bad = client.post(
        f"/api/drafts/{draft_id}/points",
        data={"cell_x": 0, "cell_y": 0, "color": "1,2,3"},
    )
    assert bad.status_code == 400


# --------------------------------------------------------------------------- #
# 旧草案复用（换起点重扫）
# --------------------------------------------------------------------------- #

# 非对称 4×4 模式：单元测试中 d1 与 d2 唯一且不同的确定性夹具
_REUSE_GRID = [
    [(4, 4, 4), (2, 2, 2), (4, 4, 4), (3, 3, 3)],
    [(2, 2, 2), (4, 4, 4), (2, 2, 2), (2, 2, 2)],
    [(2, 2, 2), (1, 1, 1), (4, 4, 4), (2, 2, 2)],
    [(4, 4, 4), (4, 4, 4), (2, 2, 2), (3, 3, 3)],
]
_REUSE_OLD_FIXED = {
    (0, 0): (4, 4, 4), (0, 1): (2, 2, 2), (1, 0): (2, 2, 2),
}
_REUSE_NEW_FIXED = {
    (1, 0): (4, 4, 4), (1, 1): (2, 2, 2), (1, 3): (1, 1, 1),
    (2, 0): (2, 2, 2),
}
_D1 = (1, 0)
_D2 = (0, 1)

# 端到端换起点夹具：非对称 8×8 模式（random.seed(11) 冻结）
_RESCAN_GRID = [
    [[230, 200, 70], [230, 200, 70], [230, 200, 70], [60, 90, 220],
     [60, 90, 220], [230, 200, 70], [60, 90, 220], [220, 60, 60]],
    [[230, 200, 70], [60, 180, 90], [60, 90, 220], [220, 60, 60],
     [220, 60, 60], [230, 200, 70], [230, 200, 70], [60, 90, 220]],
    [[220, 60, 60], [220, 60, 60], [220, 60, 60], [220, 60, 60],
     [60, 90, 220], [60, 90, 220], [220, 60, 60], [230, 200, 70]],
    [[60, 180, 90], [230, 200, 70], [60, 90, 220], [60, 90, 220],
     [60, 180, 90], [230, 200, 70], [220, 60, 60], [220, 60, 60]],
    [[230, 200, 70], [60, 180, 90], [230, 200, 70], [220, 60, 60],
     [60, 180, 90], [60, 180, 90], [60, 90, 220], [60, 180, 90]],
    [[220, 60, 60], [220, 60, 60], [220, 60, 60], [230, 200, 70],
     [220, 60, 60], [60, 180, 90], [230, 200, 70], [220, 60, 60]],
    [[220, 60, 60], [220, 60, 60], [60, 90, 220], [60, 90, 220],
     [220, 60, 60], [230, 200, 70], [230, 200, 70], [230, 200, 70]],
    [[230, 200, 70], [220, 60, 60], [60, 90, 220], [60, 180, 90],
     [60, 180, 90], [220, 60, 60], [60, 180, 90], [60, 180, 90]],
]
_RESCAN_SHIFT = (3, 2)
_RESCAN_CANVAS = 10  # 10×10 残片画布，单元 8×8


def _rescan_fragments():
    """返回 (旧残片色点, 新残片色点)，画布 10×10、透明处缺损。

    旧扫描保留上 6 行；新扫描窗口平移 (3,2) 且保留上 8 行（多认出色点）。
    """
    w = h = _RESCAN_CANVAS
    u = len(_RESCAN_GRID)
    dx, dy = _RESCAN_SHIFT
    old_points = {
        (x, y): tuple(_RESCAN_GRID[y % u][x % u])
        for y in range(6) for x in range(w)}
    new_points = {
        (xp, yp): tuple(
            _RESCAN_GRID[(yp - dy) % u][(xp - dx) % u])
        for yp in range(8) for xp in range(w)}
    return old_points, new_points


def _all_cells(grid):
    return {
        (x, y): tuple(grid[y][x])
        for y in range(len(grid)) for x in range(len(grid[0]))
    }


def _fragment_from_cells(cells, size):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    for (x, y), color in cells.items():
        img.putpixel((x, y), (*color, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _tile_counts_of(grid):
    ref = images.parse_reference(png_bytes(grid))
    return ref.tile_counts


# ---- reuse 纯函数 ---------------------------------------------------------- #

def test_proportional_ratio_same_and_scaled():
    tc = _tile_counts_of(_REUSE_GRID)
    # 同图：比例 1/1
    assert reuse.proportional_ratio(tc, tc) == (1, 1)
    # 全部频次乘 3：比例 3/1
    scaled = {k: v * 3 for k, v in tc.items()}
    assert reuse.proportional_ratio(tc, scaled) == (3, 1)
    # 比例不一 -> None
    broken = dict(scaled)
    first = next(iter(broken))
    broken[first] += 1
    assert reuse.proportional_ratio(tc, broken) is None
    # 种类不同 -> None
    other = dict(list(tc.items())[:-1])
    assert reuse.proportional_ratio(tc, other) is None


def test_point_shift_and_grid_shift_distinct():
    tc = _tile_counts_of(_REUSE_GRID)
    assessment = reuse.assess_reuse(
        old_status=solver.STATUS_OPTIMAL,
        old_score=sum(tc.values()),
        old_grid=tuple(tuple(row) for row in _REUSE_GRID),
        old_fixed=dict(_REUSE_OLD_FIXED),
        old_tile_counts=tc,
        unit_w=4, unit_h=4,
        new_tile_counts=tc,
        new_fixed=dict(_REUSE_NEW_FIXED),
    )
    assert assessment.eligible
    assert assessment.point_shift == _D1
    assert assessment.grid_shift == _D2
    assert _D1 != _D2
    assert assessment.can_skip_solver
    # 平移纹版确实符合全部新色点
    assert reuse.grid_matches_points(
        assessment.shifted_grid, _REUSE_NEW_FIXED)


def test_assess_reuse_rejects_non_optimal_old_draft():
    tc = _tile_counts_of(_REUSE_GRID)
    assessment = reuse.assess_reuse(
        old_status=solver.STATUS_FEASIBLE,
        old_score=10,
        old_grid=tuple(tuple(row) for row in _REUSE_GRID),
        old_fixed=dict(_REUSE_OLD_FIXED),
        old_tile_counts=tc,
        unit_w=4, unit_h=4,
        new_tile_counts=tc,
        new_fixed=dict(_REUSE_NEW_FIXED),
    )
    assert not assessment.eligible
    assert assessment.reason == reuse.REASON_OLD_NOT_OPTIMAL


def test_assess_reuse_rejects_size_mismatch():
    tc = _tile_counts_of(_REUSE_GRID)
    assessment = reuse.assess_reuse(
        old_status=solver.STATUS_OPTIMAL,
        old_score=10,
        old_grid=tuple(tuple(row) for row in _REUSE_GRID),
        old_fixed=dict(_REUSE_OLD_FIXED),
        old_tile_counts=tc,
        unit_w=5, unit_h=4,  # 旧 grid 是 4×4
        new_tile_counts=tc,
        new_fixed=dict(_REUSE_NEW_FIXED),
    )
    assert not assessment.eligible
    assert assessment.reason == reuse.REASON_SIZE_MISMATCH


def test_assess_reuse_rejects_when_no_covering_shift():
    tc = _tile_counts_of(_REUSE_GRID)
    # 新残片放一个与旧点任何平移都冲突的颜色
    new_fixed = {(0, 0): (1, 1, 1)}
    assessment = reuse.assess_reuse(
        old_status=solver.STATUS_OPTIMAL,
        old_score=sum(tc.values()),
        old_grid=tuple(tuple(row) for row in _REUSE_GRID),
        old_fixed={(0, 0): (4, 4, 4)},
        old_tile_counts=tc,
        unit_w=4, unit_h=4,
        new_tile_counts=tc,
        new_fixed=new_fixed,
    )
    assert not assessment.eligible
    assert assessment.reason == reuse.REASON_NO_COVERING_SHIFT


def test_eligible_but_no_grid_shift_keeps_bound_only():
    """覆盖平移存在但旧纹版无法符合全部新色点：只折算上界，不跳过求解。"""
    tc = _tile_counts_of(_REUSE_GRID)
    # 新色点：覆盖旧点（d1=(1,0)），但加入一个旧纹版任何平移都给不出的格
    new_fixed = {
        ((x + 1) % 4, y): c for (x, y), c in _REUSE_OLD_FIXED.items()}
    new_fixed[(3, 3)] = (1, 1, 1)
    assessment = reuse.assess_reuse(
        old_status=solver.STATUS_OPTIMAL,
        old_score=sum(tc.values()),
        old_grid=tuple(tuple(row) for row in _REUSE_GRID),
        old_fixed=dict(_REUSE_OLD_FIXED),
        old_tile_counts=tc,
        unit_w=4, unit_h=4,
        new_tile_counts=tc,
        new_fixed=new_fixed,
    )
    assert assessment.eligible
    assert assessment.point_shift == (1, 0)
    assert assessment.grid_shift is None
    assert not assessment.can_skip_solver
    assert assessment.scaled_bound == sum(tc.values())


def test_scaled_bound_uses_frequency_ratio():
    tc = _tile_counts_of(_REUSE_GRID)
    grid = tuple(tuple(row) for row in _REUSE_GRID)
    old_score = 18  # 该纹版在原参考图下的实际得分（含一个频次 2 的图块）
    scaled_counts = {k: v * 2 for k, v in tc.items()}
    assessment = reuse.assess_reuse(
        old_status=solver.STATUS_OPTIMAL,
        old_score=old_score,
        old_grid=grid,
        old_fixed={},
        old_tile_counts=tc,
        unit_w=4, unit_h=4,
        new_tile_counts=scaled_counts,
        new_fixed={},
    )
    assert assessment.eligible
    assert assessment.ratio == (2, 1)
    # 全空新残片：零位移即可直认，新得分严格按新频次折算
    assert assessment.can_skip_solver
    assert assessment.shifted_score == 2 * old_score
    assert assessment.scaled_bound == 2 * old_score


def test_shift_grid_toroidal_roundtrip():
    grid = tuple(tuple(row) for row in _REUSE_GRID)
    moved = reuse.shift_grid(grid, 3, 2, 4, 4)
    back = reuse.shift_grid(moved, -3, -2, 4, 4)
    assert back == grid


# ---- solver 的折算上界提示 ------------------------------------------------- #

def test_solver_upper_bound_hint_marks_optimal_early():
    ref = images.parse_reference(png_bytes(stripes_grid(8)))
    # 给一个与真实最优分相等的提示，应直接判最优且上界取提示值
    result = solver.solve(ref.colors, ref.tile_counts, 6, 6, {},
                          5.0, upper_bound_hint=36 * 32)
    assert result.status == solver.STATUS_OPTIMAL
    assert result.upper_bound == 36 * 32


def test_solver_hint_does_not_lower_true_optimum_claim():
    ref = images.parse_reference(png_bytes(stripes_grid(8)))
    # 提示高于通用上界时，有效上界仍是通用上界
    result = solver.solve(ref.colors, ref.tile_counts, 6, 6, {},
                          5.0, upper_bound_hint=10 ** 9)
    assert result.status == solver.STATUS_OPTIMAL
    assert result.upper_bound == 36 * 32


def test_backtrack_honors_upper_bound_hint():
    ref = images.parse_reference(png_bytes(stripes_grid(8)))
    bt = solver._solve_backtrack(
        ref.colors, ref.tile_counts, 6, 6, {(2, 3): RED}, 5.0,
        upper_bound_hint=36 * 32)
    assert bt.status == solver.STATUS_OPTIMAL
    assert bt.upper_bound == 36 * 32


# ---- API 端到端：换起点重扫 ------------------------------------------------ #

def _create_reuse_draft(client, fixed_cells, unit=4, grid=None):
    grid = grid or _REUSE_GRID
    resp = client.post(
        "/api/drafts",
        files={
            "reference": ("ref.png", png_bytes(grid), "image/png"),
            "fragment": ("frag.png", _fragment_from_cells(fixed_cells, unit),
                         "image/png"),
        },
        data={"unit_w": unit, "unit_h": unit},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_rescan_skips_solver_with_shifted_pattern(client):
    old_points, new_points = _rescan_fragments()
    first = client.post(
        "/api/drafts",
        files={
            "reference": ("ref.png", png_bytes(_RESCAN_GRID), "image/png"),
            "fragment": (
                "old.png",
                _fragment_from_cells(old_points, _RESCAN_CANVAS),
                "image/png"),
        },
        data={"unit_w": 8, "unit_h": 8},
    )
    assert first.status_code == 200, first.text
    first_body = first.json()
    assert first_body["status"] == "optimal"
    # 旧求解器确实运行过（非复用直认）
    assert first_body["solver"] != "reuse-toroidal-shift"
    old_id = first_body["draft_id"]

    resp = client.post(
        "/api/drafts",
        files={
            "reference": ("ref.png", png_bytes(_RESCAN_GRID), "image/png"),
            "fragment": (
                "new.png",
                _fragment_from_cells(new_points, _RESCAN_CANVAS),
                "image/png"),
        },
        data={"unit_w": 8, "unit_h": 8,
              "previous_draft_id": old_id},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["reuse"]["eligible"] is True
    assert body["reuse"]["solver_skipped"] is True
    assert body["reuse"]["point_shift"] == {
        "dx": _RESCAN_SHIFT[0], "dy": _RESCAN_SHIFT[1]}
    assert body["reuse"]["grid_shift"] == {
        "dx": _RESCAN_SHIFT[0], "dy": _RESCAN_SHIFT[1]}
    assert body["reuse"]["frequency_ratio"] == {
        "numerator": 1, "denominator": 1}
    # 跳过求解：求解器名是平移直认，状态已证最优
    assert body["solver"] == "reuse-toroidal-shift"
    assert body["status"] == "optimal"
    # 生成的是独立新草案
    assert body["draft_id"] != old_id
    # 平移纹版符合全部新色点
    new_fixed, conflicts = images.project_points(new_points, 8, 8)
    assert conflicts == []
    for (x, y), color in new_fixed.items():
        assert body["candidate_grid"][y][x] == list(color)
    # 新残片认出的格数不少于旧残片
    old_fixed, _ = images.project_points(old_points, 8, 8)
    assert len(new_fixed) >= len(old_fixed)
    # 两份草案各自可领取 PNG
    for did in (old_id, body["draft_id"]):
        png = client.get(f"/api/drafts/{did}/pattern.png")
        assert png.status_code == 200


def test_rescan_without_previous_id_runs_normal_solver(client):
    _, new_points = _rescan_fragments()
    resp = client.post(
        "/api/drafts",
        files={
            "reference": ("ref.png", png_bytes(_RESCAN_GRID), "image/png"),
            "fragment": ("new.png",
                         _fragment_from_cells(new_points, _RESCAN_CANVAS),
                         "image/png"),
        },
        data={"unit_w": 8, "unit_h": 8},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "reuse" not in body
    assert body["status"] in ("optimal", "feasible")


def test_rescan_unknown_previous_draft_404(client):
    _, new_points = _rescan_fragments()
    resp = client.post(
        "/api/drafts",
        files={
            "reference": ("ref.png", png_bytes(_RESCAN_GRID), "image/png"),
            "fragment": ("new.png",
                         _fragment_from_cells(new_points, _RESCAN_CANVAS),
                         "image/png"),
        },
        data={"unit_w": 8, "unit_h": 8, "previous_draft_id": 9999},
    )
    assert resp.status_code == 404
    assert resp.json()["detail"]["error"] == "not_found"


def test_rescan_ineligible_falls_back_to_solver(client):
    # 旧草案基于竖条纹参考图；新残片配另一参考图 -> 图块种类不同，照常求解
    stripes = stripes_grid(8)
    old = client.post(
        "/api/drafts",
        files=_files(stripes, stripes,
                     frag_transparent=lambda x, y, c: (x, y) != (0, 0)),
        data={"unit_w": 4, "unit_h": 4},
    )
    old_id = old.json()["draft_id"]

    resp = client.post(
        "/api/drafts",
        files={
            "reference": ("ref.png", png_bytes(_REUSE_GRID), "image/png"),
            "fragment": ("frag.png",
                         _fragment_from_cells(_REUSE_NEW_FIXED, 4),
                         "image/png"),
        },
        data={"unit_w": 4, "unit_h": 4,
              "previous_draft_id": old_id},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["reuse"]["eligible"] is False
    assert body["reuse"]["reason"] == reuse.REASON_TILE_KIND_MISMATCH
    assert body["reuse"]["solver_skipped"] is False
    assert body["solver"] != "reuse-toroidal-shift"
    assert "draft_id" in body


def test_rescan_eligible_bound_but_no_grid_shift_invokes_solver(client):
    tc = _tile_counts_of(_REUSE_GRID)
    first = _create_reuse_draft(client, _REUSE_OLD_FIXED)
    old_id = first["draft_id"]

    new_fixed = {
        ((x + 1) % 4, y): c for (x, y), c in _REUSE_OLD_FIXED.items()}
    new_fixed[(3, 3)] = (1, 1, 1)
    resp = client.post(
        "/api/drafts",
        files={
            "reference": ("ref.png", png_bytes(_REUSE_GRID), "image/png"),
            "fragment": ("frag.png", _fragment_from_cells(new_fixed, 4),
                         "image/png"),
        },
        data={"unit_w": 4, "unit_h": 4,
              "previous_draft_id": old_id},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["reuse"]["eligible"] is True
    assert body["reuse"]["solver_skipped"] is False
    # 折算上界 = min(旧最优分, 通用上界)；该参考图通用上界为 16×最高频次 2
    assert body["reuse"]["scaled_upper_bound"] == min(
        first["score"], 16 * max(tc.values()))
    assert body["solver"] != "reuse-toroidal-shift"


def test_rescan_scaled_ratio_through_api(client):
    """新参考图由 4×4 模式双向平铺成 8×8：同图块种类、整体频次 ×4。"""
    first = _create_reuse_draft(client, _all_cells(_REUSE_GRID))
    assert first["status"] == "optimal"
    old_id = first["draft_id"]
    old_score = first["score"]

    # 8×8 平铺参考图：每个 4×4 位置的图块各出现 4 次
    big = [
        [_REUSE_GRID[y % 4][x % 4] for x in range(8)]
        for y in range(8)
    ]
    # 新残片沿用旧草案全部色点（零位移覆盖），保证折算条件成立
    resp = client.post(
        "/api/drafts",
        files={
            "reference": ("ref8.png", png_bytes(big), "image/png"),
            "fragment": (
                "frag.png",
                _fragment_from_cells(_all_cells(_REUSE_GRID), 4),
                "image/png"),
        },
        data={"unit_w": 4, "unit_h": 4,
              "previous_draft_id": old_id},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["reuse"]["eligible"] is True
    assert body["reuse"]["frequency_ratio"] == {
        "numerator": 4, "denominator": 1}
    assert body["reuse"]["solver_skipped"] is True
    assert body["score"] == 4 * old_score
    assert body["upper_bound"] == 4 * old_score


def test_two_drafts_serve_their_own_patterns(client):
    old_points, new_points = _rescan_fragments()
    first = client.post(
        "/api/drafts",
        files={
            "reference": ("ref.png", png_bytes(_RESCAN_GRID), "image/png"),
            "fragment": ("old.png",
                         _fragment_from_cells(old_points, _RESCAN_CANVAS),
                         "image/png"),
        },
        data={"unit_w": 8, "unit_h": 8},
    ).json()
    old_id = first["draft_id"]
    second = client.post(
        "/api/drafts",
        files={
            "reference": ("ref.png", png_bytes(_RESCAN_GRID), "image/png"),
            "fragment": ("new.png",
                         _fragment_from_cells(new_points, _RESCAN_CANVAS),
                         "image/png"),
        },
        data={"unit_w": 8, "unit_h": 8,
              "previous_draft_id": old_id},
    ).json()
    assert second["reuse"]["solver_skipped"] is True
    new_id = second["draft_id"]

    old_png = client.get(f"/api/drafts/{old_id}/pattern.png").content
    new_png = client.get(f"/api/drafts/{new_id}/pattern.png").content
    old_im = Image.open(io.BytesIO(old_png)).convert("RGB")
    new_im = Image.open(io.BytesIO(new_png)).convert("RGB")
    old_pixels = [old_im.getpixel((x, y)) for y in range(8) for x in range(8)]
    new_pixels = [new_im.getpixel((x, y)) for y in range(8) for x in range(8)]
    # 两份 PNG 不同（新纹版是旧纹版经 (3,2) 环绕平移的结果）
    assert old_pixels != new_pixels
    # 且恰为环绕平移关系：new(x,y) == old(x-3, y-2)
    dx, dy = _RESCAN_SHIFT
    for y in range(8):
        for x in range(8):
            assert new_im.getpixel((x, y)) == \
                old_im.getpixel(((x - dx) % 8, (y - dy) % 8))
    # 新 PNG 逐格符合新色点
    new_fixed, _ = images.project_points(new_points, 8, 8)
    for (x, y), color in new_fixed.items():
        assert new_im.getpixel((x, y)) == color

