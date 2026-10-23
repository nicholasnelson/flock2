"""The human display: an isometric window of the meadow drawn from the state (numpy, not part of
training), with one sheep's first-person strip and a status line beneath it.

Continuous positions are projected with the same isometric formula as ``flock.iso`` (tile point
(r, c) -> screen ((c - r) W/2, (c + r) H/2)), so sheep and objects sit between tiles where they
are. Night is shown by darkening the frame (the sheep themselves see a shorter range).
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from flock import fpv, sprites_iso as S
from flock.palette import RGB, TRANSPARENT

TW, TH = S.TILE_W, S.TILE_H


def _ground(state: dict, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Palette index of the ground at float world points (rows, cols)."""
    n = state["water"].shape[0]
    ri, ci = np.floor(rows).astype(int), np.floor(cols).astype(int)
    inside = (ri >= 0) & (ri < n) & (ci >= 0) & (ci < n)
    rc, cc = np.clip(ri, 0, n - 1), np.clip(ci, 0, n - 1)
    g = np.clip(np.floor(state["grass"][rc, cc]), 0, 3).astype(int)
    st = np.where(state["water"][rc, cc], S.WATER, g)
    fu, fv = rows - ri, cols - ci
    bx = np.clip(np.floor((fv - fu) * (TW / 2) + TW / 2), 0, TW - 1).astype(int)
    by = np.clip(np.floor((fv + fu) * (TH / 2)), 0, TH - 1).astype(int)
    tex = S.GROUND_STACK[st, S.tile_variant(rc, cc), by, bx]
    tex = np.where(tex == TRANSPARENT, S.GROUND_STACK[st, 0, TH // 2, TW // 2], tex)
    return np.where(inside, tex, 0).astype(np.uint8)


def iso_window(state: dict, centre: np.ndarray, half: int = 20) -> np.ndarray:
    """(H, W) palette image of the tiles within ``half`` of ``centre`` (rows and cols),
    isometric at rotation 0, with objects and sheep as sprites."""
    r0, c0 = centre[0] - half, centre[1] - half
    r1, c1 = centre[0] + half, centre[1] + half
    X0, X1 = (c0 - r1) * TW / 2, (c1 - r0) * TW / 2
    Y0, Y1 = (c0 + r0) * TH / 2 - S.BOX_H, (c1 + r1) * TH / 2
    W, H = int(X1 - X0), int(Y1 - Y0)
    X = X0 + np.arange(W)[None, :] + 0.5
    Y = Y0 + np.arange(H)[:, None] + 0.5
    cols = (X / (TW / 2) + Y / (TH / 2)) / 2
    rows = (Y / (TH / 2) - X / (TW / 2)) / 2
    img = _ground(state, rows, cols)
    img = np.where((rows >= r0) & (rows < r1) & (cols >= c0) & (cols < c1), img, 0).astype(np.uint8)

    items = []
    for k, v, p, pr in zip(state["obj_kind"], state["obj_variant"], state["obj_pos"], state["obj_present"]):
        if pr and r0 - 2 <= p[0] < r1 + 2 and c0 - 2 <= p[1] < c1 + 2:
            items.append((p[0] + p[1], S.OBJECT_STACK[k, v, 1], p))
    for p, h in zip(state["pos"], state["heading"]):
        if r0 - 2 <= p[0] < r1 + 2 and c0 - 2 <= p[1] < c1 + 2:
            face = int(np.round(h / (np.pi / 2))) % 4  # world heading -> isometric facing at rotation 0
            items.append((p[0] + p[1], S.ANIMAL_STACK[S.SHEEP, face], p))
    for _, spr, p in sorted(items, key=lambda it: it[0]):
        x = int(round((p[1] - p[0]) * TW / 2 - X0)) - S.BOX_W // 2
        y = int(round((p[1] + p[0]) * TH / 2 - Y0)) + TH // 2 - S.BOX_H
        h, w = spr.shape
        ys, xs = slice(max(y, 0), min(y + h, H)), slice(max(x, 0), min(x + w, W))
        sub = spr[ys.start - y : ys.stop - y, xs.start - x : xs.stop - x]
        m = sub != TRANSPARENT
        img[ys, xs][m] = sub[m]
    return img


def frame(state: dict, view: np.ndarray, sheep: int, night: bool, text: str, half: int = 20) -> Image.Image:
    """One display frame: the isometric window following the flock, the chosen sheep's strip, a
    status line."""
    centre = state["pos"].mean(0)
    iso_rgb = RGB[iso_window(state, centre, half)].astype(np.float32)
    strip = np.kron(RGB[view], np.ones((2, 2, 1), np.uint8)).astype(np.float32)
    if night:
        iso_rgb *= 0.45
    W = max(iso_rgb.shape[1], strip.shape[1])
    out = np.zeros((iso_rgb.shape[0] + strip.shape[0] + 22, W, 3), np.uint8)
    out[: iso_rgb.shape[0], : iso_rgb.shape[1]] = iso_rgb.astype(np.uint8)
    out[iso_rgb.shape[0] : iso_rgb.shape[0] + strip.shape[0], : strip.shape[1]] = strip.astype(np.uint8)
    im = Image.fromarray(out)
    ImageDraw.Draw(im).text((6, out.shape[0] - 18), text, fill=(235, 235, 235))
    # mark the chosen sheep
    p = state["pos"][sheep]
    r0, c0 = centre[0] - half, centre[1] - half
    r1 = centre[0] + half
    X0 = (c0 - r1) * TW / 2
    Y0 = (c0 + r0) * TH / 2 - S.BOX_H
    x = (p[1] - p[0]) * TW / 2 - X0
    y = (p[1] + p[0]) * TH / 2 - Y0 - S.BOX_H
    ImageDraw.Draw(im).polygon([(x - 4, y - 8), (x + 4, y - 8), (x, y - 2)], fill=(250, 220, 60))
    return im
