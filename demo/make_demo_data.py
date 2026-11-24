"""生成演示用的参考 PNG 与残片 PNG。

运行后在 demo/ 目录生成：

* ``reference_stripes.png``  8×8 双色竖条纹参考图（图块种类少、容易最优）
* ``fragment_stripes.png``   同尺寸残片，部分像素透明表示缺损
* ``reference_checker.png``  8×8 棋盘格参考图（含跨边/跨角图块）
* ``fragment_checker.png``   棋盘格残片
* ``reference_conflict.png`` 6×6 参考图，用于制造同格异色冲突
* ``fragment_conflict.png``  周期投影后同格异色的残片
* ``reference_rescan.png``   8×8 非对称四色参考图（换起点重扫演示）
* ``fragment_rescan_old.png`` 首次扫描残片（10×10 画布，上 6 行完好）
* ``fragment_rescan_new.png`` 换起点后重扫残片（窗口平移 (3,2)，上 8 行完好，
  多认出若干色点，配合 ``previous_draft_id`` 可跳过求解）

直接执行：``python demo/make_demo_data.py``
"""

from __future__ import annotations

import os
import random
from io import BytesIO
from pathlib import Path

from PIL import Image

OUT_DIR = Path(__file__).resolve().parent

RED = (220, 60, 60, 255)
BLUE = (60, 90, 220, 255)
GREEN = (60, 180, 90, 255)
YELLOW = (230, 200, 70, 255)
TRANSPARENT = (0, 0, 0, 0)

# 换起点演示参数（与 tests/test_app.py 的 _RESCAN_* 夹具保持一致）
RESCAN_UNIT = 8
RESCAN_CANVAS = 10
RESCAN_SHIFT = (3, 2)
RESCAN_OLD_ROWS = 6
RESCAN_NEW_ROWS = 8
_RESCAN_SEED = 11


def save(img: Image.Image, name: str) -> Path:
    path = OUT_DIR / name
    img.save(path, format="PNG")
    return path


def stripes_reference(size: int = 8) -> Image.Image:
    img = Image.new("RGBA", (size, size))
    for y in range(size):
        for x in range(size):
            img.putpixel((x, y), RED if x % 2 == 0 else BLUE)
    return img


def stripes_fragment(size: int = 8) -> Image.Image:
    img = stripes_reference(size)
    # 右下半区挖空为透明缺损
    for y in range(size // 2, size):
        for x in range(size // 2, size):
            img.putpixel((x, y), TRANSPARENT)
    # 再挖几个零散格
    for x, y in [(1, 1), (3, 2), (6, 1)]:
        img.putpixel((x, y), TRANSPARENT)
    return img


def checker_reference(size: int = 8) -> Image.Image:
    img = Image.new("RGBA", (size, size))
    for y in range(size):
        for x in range(size):
            img.putpixel((x, y), GREEN if (x + y) % 2 == 0 else YELLOW)
    return img


def checker_fragment(size: int = 8) -> Image.Image:
    img = checker_reference(size)
    for y in range(size):
        for x in range(size):
            # 保留两条对角带，其余缺损
            if not (abs(x - y) <= 1 or abs(x + y - (size - 1)) <= 1):
                img.putpixel((x, y), TRANSPARENT)
    return img


def conflict_artifacts(size: int = 6) -> tuple[Image.Image, Image.Image]:
    """参考图为纯色竖条；残片在同一投影格放两种颜色制造冲突。"""
    ref = Image.new("RGBA", (size, size))
    for y in range(size):
        for x in range(size):
            ref.putpixel((x, y), RED if x % 2 == 0 else BLUE)

    frag = Image.new("RGBA", (size, size), TRANSPARENT)
    # 单元宽 4：像素 (0,0) 与 (4,0) 投影到同一格 (0,0)，故意异色
    frag.putpixel((0, 0), RED)
    frag.putpixel((4, 0), GREEN)  # GREEN 也在参考调色板之外，先触发同格冲突
    # 其余给一些正常完好色点
    for y in range(1, size):
        frag.putpixel((0, y), RED)
        frag.putpixel((1, y), BLUE)
    return ref, frag


def rescan_artifacts() -> tuple[Image.Image, Image.Image, Image.Image]:
    """换起点重扫演示：非对称参考图 + 两次扫描残片。

    参考图是 8×8 非对称四色模式（固定随机种子，保证可复现）；
    两张残片均为 10×10 画布，透明像素标缺损：

    * 旧残片只保留上 6 行（投影覆盖 48 个不同纹版格）；
    * 新残片对应扫描窗口平移 (3,2)，保留上 8 行（覆盖全部 64 格），
      旧残片的完好色点经平移 (3,2) 后全部被新残片同色覆盖。
    """
    rng = random.Random(_RESCAN_SEED)
    palette = [RED[:3], BLUE[:3], GREEN[:3], YELLOW[:3]]
    u = RESCAN_UNIT
    while True:
        grid = [
            [rng.choice(palette) for _ in range(u)] for _ in range(u)]
        symmetric = any(
            (dx, dy) != (0, 0)
            and all(
                grid[y][x] == grid[(y - dy) % u][(x - dx) % u]
                for y in range(u) for x in range(u))
            for dy in range(u) for dx in range(u))
        if not symmetric:
            break

    reference = Image.new("RGBA", (u, u))
    for y in range(u):
        for x in range(u):
            reference.putpixel((x, y), (*grid[y][x], 255))

    dx, dy = RESCAN_SHIFT
    old_fragment = Image.new(
        "RGBA", (RESCAN_CANVAS, RESCAN_CANVAS), TRANSPARENT)
    for y in range(RESCAN_OLD_ROWS):
        for x in range(RESCAN_CANVAS):
            old_fragment.putpixel(
                (x, y), (*grid[y % u][x % u], 255))

    # 新扫描窗口相对旧窗口平移 (dx,dy)：新画布坐标 (xp,yp) 对应
    # 模式格 (xp-dx, yp-dy)
    new_fragment = Image.new(
        "RGBA", (RESCAN_CANVAS, RESCAN_CANVAS), TRANSPARENT)
    for yp in range(RESCAN_NEW_ROWS):
        for xp in range(RESCAN_CANVAS):
            color = grid[(yp - dy) % u][(xp - dx) % u]
            new_fragment.putpixel((xp, yp), (*color, 255))
    return reference, old_fragment, new_fragment


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    artifacts = [
        (stripes_reference(), "reference_stripes.png"),
        (stripes_fragment(), "fragment_stripes.png"),
        (checker_reference(), "reference_checker.png"),
        (checker_fragment(), "fragment_checker.png"),
    ]
    ref_conflict, frag_conflict = conflict_artifacts()
    artifacts.append((ref_conflict, "reference_conflict.png"))
    artifacts.append((frag_conflict, "fragment_conflict.png"))
    rescan_ref, rescan_old, rescan_new = rescan_artifacts()
    artifacts.extend([
        (rescan_ref, "reference_rescan.png"),
        (rescan_old, "fragment_rescan_old.png"),
        (rescan_new, "fragment_rescan_new.png"),
    ])

    for img, name in artifacts:
        path = save(img, name)
        print(f"wrote {path} ({img.size[0]}x{img.size[1]})")

    # 简单自检：确认参考图不透明、残片含透明像素
    for _, name in artifacts[:4]:
        im = Image.open(OUT_DIR / name)
        alpha = im.getchannel("A")
        lo, hi = alpha.getextrema()
        print(f"{name}: alpha range=({lo},{hi})")


if __name__ == "__main__":
    main()
