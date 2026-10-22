"""Isometric sprite mock at 16x8 px diamond tiles, for sign-off before the pipeline moves to a
128x128 view. Self-contained: nothing in the training code imports this.

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

TW, TH = 16, 8
C = CHAR_TO_INDEX
T_ = TRANSPARENT


def P(rows: list[str]) -> np.ndarray:
    w = max(len(r) for r in rows)
    out = np.full((len(rows), w), T_, np.uint8)
    for y, r in enumerate(rows):
        for x, ch in enumerate(r):
            if ch != ".":
                out[y, x] = C[ch]
    return out


def up2(s: np.ndarray) -> np.ndarray:
    return np.kron(s, np.ones((2, 2), np.uint8))


def outline(s: np.ndarray, colour: str = "D") -> np.ndarray:
    """Add a 1-px outline (4-neighbour) around the opaque pixels, growing the box by 1 px."""
    p = np.pad(s, 1, constant_values=T_)
    op = p != T_
    nb = np.zeros_like(op)
    nb[1:, :] |= op[:-1, :]
    nb[:-1, :] |= op[1:, :]
    nb[:, 1:] |= op[:, :-1]
    nb[:, :-1] |= op[:, 1:]
    p[nb & ~op] = C[colour]
    return p


# --- ground: 16x8 boxes, four variants per state --------------------------------------------
# Textures cover the whole box; the renderer's inverse projection only ever samples a tile's
# own diamond, so the corners are never seen.


def _ground(state: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if state == "water":
        g = np.full((TH, TW), C["B"], np.uint8)
        for _ in range(2):  # short horizontal ripples
            y, x = rng.integers(1, TH - 1), rng.integers(2, TW - 5)
            g[y, x : x + 3] = C["b"]
        return g
    if state == "bare":
        g = np.full((TH, TW), C["t"], np.uint8)
        for _ in range(3):
            g[rng.integers(0, TH), rng.integers(0, TW)] = C["G"]
        g[rng.integers(0, TH), rng.integers(0, TW)] = C["T"]
        return g
    g = np.full((TH, TW), C["G"], np.uint8)
    n, h = {"short": (1, 1), "medium": (3, 2), "long": (6, 3)}[state]
    for i in range(n):  # stalks: short vertical (sometimes leaning) strokes
        x, y = rng.integers(1, TW - 2), rng.integers(h, TH)
        lean = rng.integers(-1, 2) if h > 1 else 0
        col = C["l"] if (state != "long" or i % 2 == 0) else C["g"]
        for k in range(h):
            g[y - k, np.clip(x + (lean if k == h - 1 else 0), 0, TW - 1)] = col
    return g


GROUND_NAMES = ["bare", "short", "medium", "long", "water"]
GROUND = {n: [_ground(n, 100 * i + j) for j in range(4)] for i, n in enumerate(GROUND_NAMES)}


def tile_variant(r, c):
    return ((r * 73856093) ^ (c * 19349663) ^ ((r + c) * 83492791)) % 4


# --- objects -----------------------------------------------------------------------------


def _salt_cube(h_side: int = 10) -> np.ndarray:
    """An isometric cube whose base is exactly one tile diamond: lit white top, light-grey left
    face, dark-grey right face, white highlight on the near vertical edge and the top's front
    edges, dark outline."""
    H = TH + h_side
    s = np.full((H, TW), T_, np.uint8)
    for y in range(H):
        for x in range(TW):
            xc, yc = x + 0.5 - TW / 2, y + 0.5
            top = abs(xc) / (TW / 2) + abs(yc - TH / 2) / (TH / 2) <= 1
            ye = TH / 2 + (TH / 2) * (1 - abs(xc) / (TW / 2))  # lower edge of the top face
            side = (yc >= ye) and (yc <= ye + h_side)
            if top:
                s[y, x] = C["W"]
            elif side:
                s[y, x] = C["S"] if xc < 0 else C["s"]
    # highlights: near vertical edge, and the top face's two front edges
    for y in range(H):
        if s[y, TW // 2 - 1] in (C["S"], C["s"]):
            s[y, TW // 2 - 1] = C["W"]
    for x in range(TW):
        col = s[:, x]
        sides = np.where((col == C["S"]) | (col == C["s"]))[0]
        if len(sides):
            s[sides[0], x] = C["W"]
    return outline(s)


def _pine() -> np.ndarray:
    """Three stacked tiers, lit from the left; dark line under each upper tier."""
    tiers = [(0, 10, 2, 10), (7, 17, 4, 16), (13, 25, 6, 20)]  # (top, bottom, w_top, w_bottom)
    W, H = 20, 30
    s = np.full((H, W), T_, np.uint8)
    for i, (t, b, w0, w1) in enumerate(tiers):
        for y in range(t, b):
            w = w0 + (w1 - w0) * (y - t) / max(b - 1 - t, 1)
            x0, x1 = int(round(W / 2 - w / 2)), int(round(W / 2 + w / 2))
            for x in range(x0, x1):
                if x < W / 2 - 1 and x <= x0 + 1:
                    c = "l"
                elif x < W / 2:
                    c = "g"
                else:
                    c = "G"
                s[y, x] = C[c]
        if i < len(tiers) - 1:  # tier edge
            y = b - 1
            row = np.where(s[y] != T_)[0]
            s[y, row] = C["D"]
    s[25:30, 8:12] = C["T"]
    s[25:30, 8] = C["t"]
    return outline(s)


def _flower() -> np.ndarray:
    """A five-pointed red star with a yellow centre, on a stem with two leaves."""
    R, N = 6.5, 14
    s = np.full((24, N), T_, np.uint8)
    cx = cy = N / 2
    for y in range(N):
        for x in range(N):
            dx, dy = x + 0.5 - cx, y + 0.5 - cy
            r, th = np.hypot(dx, dy), np.arctan2(dy, dx)
            if r <= 2.0:
                s[y, x] = C["Y"]
            elif r <= R * (0.55 + 0.45 * np.cos(5 * (th + np.pi / 2))):
                s[y, x] = C["R"]
    s[N - 1 : 24, 6:8] = C["g"]
    for k in range(3):  # leaves
        s[17 - k, 8 + k] = C["l"]
        s[17 - k, 9 + k] = C["g"]
        s[20 - k, 5 - k] = C["g"]
        s[20 - k, 4 - k] = C["l"]
    return outline(s)


OBJECTS = {"salt": _salt_cube(), "tree": _pine(), "flower": _flower()}

# --- animals: signed-off 8x4 art, enlarged and outlined -----------------------------------

SHEEP_FRONT_R = P([
    "....WWWW....",
    "..WWWWWWWW..",
    ".WWWWWWWWWW.",
    ".WWWWWWWWWW.",
    "WWWWWWWWWWPs",
    "WWWWWWWWWsss",
    ".SWWWWWWWsKs",
    "..SSSSSSSsss",
    "...K...K.SS.",
    "...K...K....",
])
SHEEP_BACK_R = P([
    "....WWWWWsPs",
    "..WWWWWWWWss",
    ".WWWWWWWWWWs",
    ".WWWWWWWWWW.",
    "WWWWWWWWWWW.",
    "WWWWWWWWWW..",
    "sSWWWWWWWW..",
    "..SSSSSSSS..",
    "...K...K.K..",
    "...K...K....",
])
FOX_FRONT_R = P([
    "........O.O.",
    "WWO....OOOO.",
    "OOOO..OOOKOO",
    ".OOOOOOOOOOW",
    "..OOOOOOOO..",
    "..K..K.K.K..",
])
FOX_BACK_R = P([
    "........O.O.",
    ".......OOOO.",
    "....OOOOOOO.",
    ".WOOOOOOOO..",
    "WOOOOOOOO...",
    "OO.K..K.K...",
])


def _facings(front_r, back_r):
    f, b = outline(up2(front_r)), outline(up2(back_r))
    return [b, f, f[:, ::-1], b[:, ::-1]]


ANIMALS = {"sheep": _facings(SHEEP_FRONT_R, SHEEP_BACK_R), "fox": _facings(FOX_FRONT_R, FOX_BACK_R)}


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
        g[3:11, i * 20 + 2 : i * 20 + 18] = GROUND[n][v]
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
