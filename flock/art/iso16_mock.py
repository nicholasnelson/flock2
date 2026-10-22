"""Isometric sprite mock at 16x8 px diamond tiles: a scene and a sprite sheet for reviewing the
art. The art itself lives in ``flock.sprites_iso`` (signed off from this mock, 2026-10-23).

Referent set for the toy world (owner, 2026-10-23): salt lick (a sharp-edged cube), pine, star
flower, fox. Rock, bush and stump are dropped: they served no purpose and their silhouettes were
indistinguishable blobs. Sheep and fox here are the signed-off 8x4 art enlarged 2x with an
outline added; the three objects are redrawn for distinct silhouettes.

    python -m flock.art.iso16_mock <out_dir>
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from flock.palette import CHAR_TO_INDEX, TRANSPARENT, to_rgb

from flock import sprites_iso as S

TW, TH = S.TILE_W, S.TILE_H
C = CHAR_TO_INDEX
T_ = TRANSPARENT
GROUND_NAMES = S.GROUND_NAMES
GROUND = S.GROUND_VARIANTS
tile_variant = S.tile_variant
OBJECTS = {n: fs[0] for n, fs in S.OBJECTS.items() if n != "fox"}
ANIMALS = S.ANIMALS


# --- rendering ---------------------------------------------------------------------------


def paste(img, spr, x0, y0):
    h, w = spr.shape
    for y in range(h):
        for x in range(w):
            if spr[y, x] != T_ and 0 <= y0 + y < img.shape[0] and 0 <= x0 + x < img.shape[1]:
                img[y0 + y, x0 + x] = spr[y, x]


def box_xy(r, c):
    """Top-left of tile (r, c)'s 16x8 box at rotation 0 (world-screen coordinates)."""
    return (c - r) * (TW // 2) - TW // 2, (c + r) * (TH // 2)


def render(tiles: np.ndarray, items: list[tuple]) -> np.ndarray:
    n = tiles.shape[0]
    pad = 40
    W, H = n * TW + 2 * pad, n * TH + 2 * pad
    ox, oy = W // 2, pad
    img = np.zeros((H, W), np.uint8)
    for y in range(H):
        for x in range(W):
            dx, dy = x + 0.5 - ox, y + 0.5 - oy - TH / 2
            u, v = dx / (TW / 2) + dy / (TH / 2), dy / (TH / 2) - dx / (TW / 2)
            c, r = int(np.floor((u + 1) / 2)), int(np.floor((v + 1) / 2))
            if 0 <= r < n and 0 <= c < n:
                bx, by = box_xy(r, c)
                tex = GROUND[GROUND_NAMES[tiles[r, c]]][tile_variant(r, c)]
                img[y, x] = tex[np.clip(y - oy - by, 0, TH - 1), np.clip(x - ox - bx, 0, TW - 1)]
    for r, c, spr in sorted(items, key=lambda it: it[0] + it[1]):
        bx, by = box_xy(r, c)
        h, w = spr.shape
        paste(img, spr, ox + bx + TW // 2 - w // 2, oy + by + TH - h)
    return img


def sheet() -> tuple[np.ndarray, list[str]]:
    cells = [(n, s) for n, s in OBJECTS.items()]
    cells += [(f"sheep{f}", s) for f, s in enumerate(ANIMALS["sheep"])]
    cells += [(f"fox{f}", s) for f, s in enumerate(ANIMALS["fox"])]
    cw, ch = 36, 40
    img = np.full((ch, cw * len(cells)), C["s"], np.uint8)
    for i, (_, s) in enumerate(cells):
        paste(img, GROUND["short"][0], i * cw + cw // 2 - TW // 2, ch - TH - 2)
        h, w = s.shape
        paste(img, s, i * cw + cw // 2 - w // 2, ch - 2 - h)
    gcells = [(n, v) for n in GROUND_NAMES for v in range(4)]
    g = np.full((14, 20 * len(gcells)), C["s"], np.uint8)
    for i, (n, v) in enumerate(gcells):
        paste(g, GROUND[n][v], i * 20 + 2, 3)
    return img, g, [c[0] for c in cells]


if __name__ == "__main__":
    import sys

    out = sys.argv[1]
    rng = np.random.default_rng(0)
    n = 14
    tiles = rng.choice(4, size=(n, n), p=[0.08, 0.42, 0.3, 0.2])
    tiles[5:8, 4:7] = 4  # pond
    tiles[6, 7] = 4
    items = [
        (2, 9, OBJECTS["tree"]), (1, 3, OBJECTS["tree"]), (9, 2, OBJECTS["tree"]),
        (10, 10, OBJECTS["salt"]), (7, 11, OBJECTS["flower"]), (3, 6, OBJECTS["flower"]), (11, 6, OBJECTS["flower"]),
        (6, 9, ANIMALS["sheep"][1]), (8, 9, ANIMALS["sheep"][2]), (9, 8, ANIMALS["sheep"][0]), (4, 10, ANIMALS["sheep"][3]),
        (12, 12, ANIMALS["fox"][2]), (3, 12, ANIMALS["fox"][0]),
    ]
    scene = render(tiles, items)
    Image.fromarray(to_rgb(scene)).resize((scene.shape[1] * 4, scene.shape[0] * 4), Image.NEAREST).save(f"{out}/iso16_scene.png")
    sh, g, names = sheet()
    Image.fromarray(to_rgb(sh)).resize((sh.shape[1] * 5, sh.shape[0] * 5), Image.NEAREST).save(f"{out}/iso16_sheet.png")
    Image.fromarray(to_rgb(g)).resize((g.shape[1] * 5, g.shape[0] * 5), Image.NEAREST).save(f"{out}/iso16_ground.png")
    print(names, {k: v.shape for k, v in OBJECTS.items()}, {k: v[0].shape for k, v in ANIMALS.items()})
