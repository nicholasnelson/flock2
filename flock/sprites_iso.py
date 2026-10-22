"""Isometric pixel-art sprites at 16x8 px diamond tiles (signed off 2026-10-23; the mock is
``flock.art.iso16_mock``).

Every sprite is camera-facing (billboarded, always drawn upright), so the tile layout can be
rotated under a fixed camera: that is what makes heading-aligned isometric views possible.
Ground states are drawn per position (never pixel-rotated), so grass texture is also
rotation-safe. Water is a ground state (a pond is a cluster of water tiles), not an object.

Referents (objects the toy game asks about): a sharp-edged salt-lick cube, a pine, a star
flower, and the fox. Each has a distinct silhouette; the earlier rock, bush and stump were
dropped because as silhouettes they were indistinguishable blobs with no role in the world.
The fox is both an animal and a referent, so objects have four facings like animals (static
objects repeat one sprite).

Sprites are anchored at the bottom-centre of their tile's diamond. Outlines are generated, not
drawn: a 1-px outline-colour border around every sprite's opaque pixels.
"""

from __future__ import annotations

import numpy as np

from flock.palette import CHAR_TO_INDEX, TRANSPARENT

TILE_W, TILE_H = 16, 8  # diamond bounding box
C = CHAR_TO_INDEX


def _parse(rows: list[str]) -> np.ndarray:
    w = max(len(r) for r in rows)
    out = np.full((len(rows), w), TRANSPARENT, dtype=np.uint8)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch != ".":
                out[y, x] = CHAR_TO_INDEX[ch]
    return out


def _up2(s: np.ndarray) -> np.ndarray:
    return np.kron(s, np.ones((2, 2), np.uint8))


def _outline(s: np.ndarray, colour: str = "D") -> np.ndarray:
    """Add a 1-px outline (4-neighbour) around the opaque pixels, growing the box by 1 px."""
    p = np.pad(s, 1, constant_values=TRANSPARENT)
    op = p != TRANSPARENT
    nb = np.zeros_like(op)
    nb[1:, :] |= op[:-1, :]
    nb[:-1, :] |= op[1:, :]
    nb[:, 1:] |= op[:, :-1]
    nb[:, :-1] |= op[:, 1:]
    p[nb & ~op] = C[colour]
    return p


# --- ground states: 16x8 diamonds, four texture variants each ------------------------------
# Which pixels of the box belong to the tile, by the renderer's own rule (``iso.render_view``):
# pixel centre (dx, dy) from the diamond centre is in the tile iff -1 <= u < 1 and -1 <= v < 1
# with u = dx/(W/2) + dy/(H/2), v = dy/(H/2) - dx/(W/2).
_yy, _xx = np.mgrid[0:TILE_H, 0:TILE_W]
_dx, _dy = _xx + 0.5 - TILE_W / 2, _yy + 0.5 - TILE_H / 2
_u, _v = _dx / (TILE_W / 2) + _dy / (TILE_H / 2), _dy / (TILE_H / 2) - _dx / (TILE_W / 2)
DIAMOND = (_u >= -1) & (_u < 1) & (_v >= -1) & (_v < 1)


def _ground(state: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if state == "water":
        g = np.full((TILE_H, TILE_W), C["B"], np.uint8)
        for _ in range(2):  # short horizontal ripples
            y, x = rng.integers(1, TILE_H - 1), rng.integers(2, TILE_W - 5)
            g[y, x : x + 3] = C["b"]
    elif state == "bare":
        g = np.full((TILE_H, TILE_W), C["t"], np.uint8)
        for _ in range(3):
            g[rng.integers(0, TILE_H), rng.integers(0, TILE_W)] = C["G"]
    else:
        g = np.full((TILE_H, TILE_W), C["G"], np.uint8)
        n, h = {"short": (1, 1), "medium": (3, 2), "long": (6, 3)}[state]
        for i in range(n):  # stalks: short vertical (sometimes leaning) strokes
            x, y = rng.integers(1, TILE_W - 2), rng.integers(h, TILE_H)
            lean = rng.integers(-1, 2) if h > 1 else 0
            col = C["l"] if (state != "long" or i % 2 == 0) else C["g"]
            for k in range(h):
                g[y - k, np.clip(x + (lean if k == h - 1 else 0), 0, TILE_W - 1)] = col
    return np.where(DIAMOND, g, TRANSPARENT).astype(np.uint8)


GROUND_NAMES = ["bare", "short", "medium", "long", "water"]
N_VARIANTS = 4
GROUND_VARIANTS = {n: [_ground(n, 100 * i + j) for j in range(N_VARIANTS)] for i, n in enumerate(GROUND_NAMES)}
GROUND = {n: v[0] for n, v in GROUND_VARIANTS.items()}  # first variant, for tests/thumbnails
GROUND_STACK = np.stack([np.stack(GROUND_VARIANTS[n]) for n in GROUND_NAMES])  # (n_ground, 4, 8, 16)
N_GROUND = len(GROUND_NAMES)
GRASS_STATES = [GROUND_NAMES.index(n) for n in ("bare", "short", "medium", "long")]
WATER = GROUND_NAMES.index("water")


def tile_variant(r, c):
    """Per-tile texture variant index (0..3) from a scrambled hash of the tile position.
    Works on ints and on integer arrays (numpy or jax)."""
    return ((r * 73856093) ^ (c * 19349663) ^ ((r + c) * 83492791)) % N_VARIANTS


# --- objects -------------------------------------------------------------------------------


def _salt_cube(h_side: int = 10) -> np.ndarray:
    """An isometric cube whose base is exactly one tile diamond: lit white top, light-grey left
    face, dark-grey right face, white highlight on the near vertical edge and on the top's two
    front edges."""
    H = TILE_H + h_side
    s = np.full((H, TILE_W), TRANSPARENT, np.uint8)
    for y in range(H):
        for x in range(TILE_W):
            xc, yc = x + 0.5 - TILE_W / 2, y + 0.5
            top = abs(xc) / (TILE_W / 2) + abs(yc - TILE_H / 2) / (TILE_H / 2) <= 1
            ye = TILE_H / 2 + (TILE_H / 2) * (1 - abs(xc) / (TILE_W / 2))  # lower edge of the top
            if top:
                s[y, x] = C["W"]
            elif ye <= yc <= ye + h_side:
                s[y, x] = C["S"] if xc < 0 else C["s"]
    for y in range(H):
        if s[y, TILE_W // 2 - 1] in (C["S"], C["s"]):
            s[y, TILE_W // 2 - 1] = C["W"]
    for x in range(TILE_W):
        sides = np.where((s[:, x] == C["S"]) | (s[:, x] == C["s"]))[0]
        if len(sides):
            s[sides[0], x] = C["W"]
    return _outline(s)


def _pine() -> np.ndarray:
    """Three stacked tiers, lit from the left; dark line under each upper tier."""
    tiers = [(0, 10, 2, 10), (7, 17, 4, 16), (13, 25, 6, 20)]  # (top, bottom, w_top, w_bottom)
    W, H = 20, 30
    s = np.full((H, W), TRANSPARENT, np.uint8)
    for i, (t, b, w0, w1) in enumerate(tiers):
        for y in range(t, b):
            w = w0 + (w1 - w0) * (y - t) / max(b - 1 - t, 1)
            x0, x1 = int(round(W / 2 - w / 2)), int(round(W / 2 + w / 2))
            for x in range(x0, x1):
                s[y, x] = C["l"] if (x < W / 2 - 1 and x <= x0 + 1) else C["g"] if x < W / 2 else C["G"]
        if i < len(tiers) - 1:
            row = np.where(s[b - 1] != TRANSPARENT)[0]
            s[b - 1, row] = C["D"]
    s[25:30, 8:12] = C["T"]
    s[25:30, 8] = C["t"]
    return _outline(s)


def _flower() -> np.ndarray:
    """A five-pointed red star with a yellow centre on a short stem with a pair of leaves;
    shorter than a sheep."""
    R, N, H = 5.8, 12, 17
    s = np.full((H, N), TRANSPARENT, np.uint8)
    cx = cy = N / 2
    for y in range(N):
        for x in range(N):
            dx, dy = x + 0.5 - cx, y + 0.5 - cy
            r, th = np.hypot(dx, dy), np.arctan2(dy, dx)
            if r <= 1.8:
                s[y, x] = C["Y"]
            elif r <= R * (0.55 + 0.45 * np.cos(5 * (th + np.pi / 2))):
                s[y, x] = C["R"]
    s[N - 1 : H, 5:7] = C["g"]
    for k in range(2):  # leaves
        s[15 - k, 7 + k] = C["l"]
        s[15 - k, 8 + k] = C["g"]
        s[15 - k, 4 - k] = C["g"]
        s[15 - k, 3 - k] = C["l"]
    return _outline(s)


# --- animals: the signed-off 8x4 art enlarged 2x and outlined -----------------------------
# Facing index = (world heading - view heading) % 4, i.e. the heading in the rotated frame:
#   0 = N: away, up-right     1 = E: toward, down-right
#   2 = S: toward, down-left  3 = W: away, up-left
# Front views show the face; back views show wool/tail and the ears only. Left-facing sprites
# are mirrors of right-facing ones.

SHEEP_FRONT_R = _parse(
    [
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
    ]
)
SHEEP_BACK_R = _parse(
    [
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
    ]
)
FOX_FRONT_R = _parse(
    [
        "........O.O.",
        "WWO....OOOO.",
        "OOOO..OOOKOO",
        ".OOOOOOOOOOW",
        "..OOOOOOOO..",
        "..K..K.K.K..",
    ]
)
FOX_BACK_R = _parse(
    [
        "........O.O.",
        ".......OOOO.",
        "....OOOOOOO.",
        ".WOOOOOOOO..",
        "WOOOOOOOO...",
        "OO.K..K.K...",
    ]
)


def _facings(front_r: np.ndarray, back_r: np.ndarray) -> list[np.ndarray]:
    f, b = _outline(_up2(front_r)), _outline(_up2(back_r))
    return [b, f, f[:, ::-1], b[:, ::-1]]


ANIMALS = {"sheep": _facings(SHEEP_FRONT_R, SHEEP_BACK_R), "fox": _facings(FOX_FRONT_R, FOX_BACK_R)}
ANIMAL_NAMES = list(ANIMALS)
SHEEP = ANIMAL_NAMES.index("sheep")
FOX = ANIMAL_NAMES.index("fox")
N_FACINGS = 4

# Referents, each as four facings (static objects repeat one sprite).
OBJECTS = {
    "salt": [_salt_cube()] * N_FACINGS,
    "tree": [_pine()] * N_FACINGS,
    "flower": [_flower()] * N_FACINGS,
    "fox": ANIMALS["fox"],
}
OBJECT_NAMES = list(OBJECTS)
N_OBJECT_TYPES = len(OBJECT_NAMES)
FOX_OBJECT = OBJECT_NAMES.index("fox")


def _pad(s: np.ndarray, h: int, w: int) -> np.ndarray:
    """Bottom-centre-anchored padding to a common (h, w) box."""
    out = np.full((h, w), TRANSPARENT, np.uint8)
    sh, sw = s.shape
    x0 = (w - sw) // 2
    out[h - sh :, x0 : x0 + sw] = s
    return out


# Common box for everything that is pasted.
_all = [f for fs in OBJECTS.values() for f in fs] + [f for fs in ANIMALS.values() for f in fs]
BOX_H = max(f.shape[0] for f in _all)
BOX_W = max(f.shape[1] for f in _all)
# (n_obj, 4, H, W) and (n_animal, 4, H, W): facing index as documented above
OBJECT_STACK = np.stack([np.stack([_pad(f, BOX_H, BOX_W) for f in OBJECTS[n]]) for n in OBJECT_NAMES])
ANIMAL_STACK = np.stack([np.stack([_pad(f, BOX_H, BOX_W) for f in ANIMALS[n]]) for n in ANIMAL_NAMES])
