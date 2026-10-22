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


# --- objects: parameterised generators --------------------------------------------------
# Each referent type has several variants that share what defines the type and differ in what
# is incidental (size, proportions, petal count, colour, species), so a drawing has to capture
# the type rather than one remembered example. The last ``HELD_OUT`` variants of each type are
# never shown in training; evaluation, galleries and the quiz use them.


def _salt_cube(w: int = TILE_W, h_side: int = 10) -> np.ndarray:
    """An isometric cube of base width ``w`` (``w`` = TILE_W fills the tile diamond): lit white
    top, light-grey left face, dark-grey right face, white highlight on the near vertical edge
    and on the top's two front edges."""
    th = w // 2
    H = th + h_side
    s = np.full((H, w), TRANSPARENT, np.uint8)
    for y in range(H):
        for x in range(w):
            xc, yc = x + 0.5 - w / 2, y + 0.5
            top = abs(xc) / (w / 2) + abs(yc - th / 2) / (th / 2) <= 1
            ye = th / 2 + (th / 2) * (1 - abs(xc) / (w / 2))  # lower edge of the top
            if top:
                s[y, x] = C["W"]
            elif ye <= yc <= ye + h_side:
                s[y, x] = C["S"] if xc < 0 else C["s"]
    for y in range(H):
        if s[y, w // 2 - 1] in (C["S"], C["s"]):
            s[y, w // 2 - 1] = C["W"]
    for x in range(w):
        sides = np.where((s[:, x] == C["S"]) | (s[:, x] == C["s"]))[0]
        if len(sides):
            s[sides[0], x] = C["W"]
    return _outline(s)


def _pine(tiers: list[tuple[int, int, int, int]] | None = None, trunk_h: int = 5) -> np.ndarray:
    """Stacked triangular tiers (top, bottom, w_top, w_bottom), lit from the left; dark line
    under each upper tier; bark trunk below."""
    tiers = tiers or [(0, 10, 2, 10), (7, 17, 4, 16), (13, 25, 6, 20)]
    W = max(t[3] for t in tiers)
    top_h = max(t[1] for t in tiers)
    H = top_h + trunk_h
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
    c = W // 2
    s[top_h:H, c - 2 : c + 2] = C["T"]
    s[top_h:H, c - 2] = C["t"]
    return _outline(s)


def _round_tree(r: float = 8.5, trunk_h: int = 8, squash: float = 1.0) -> np.ndarray:
    """A broadleaf tree: a round canopy (lit upper-left, shaded lower-right, a few highlight
    leaves) on a bark trunk."""
    W = int(2 * r) + 2
    ch = int(2 * r * squash) + 2
    H = ch + trunk_h - 2
    s = np.full((H, W), TRANSPARENT, np.uint8)
    cx, cy = W / 2, ch / 2
    c = W // 2
    s[ch - 3 : H, c - 2 : c + 2] = C["T"]
    s[ch - 3 : H, c - 2] = C["t"]
    for y in range(ch):
        for x in range(W):
            dx, dy = (x + 0.5 - cx) / r, (y + 0.5 - cy) / (r * squash)
            d = dx * dx + dy * dy
            if d <= 1.0:
                shade = dx + dy
                s[y, x] = C["l"] if shade < -0.9 else C["g"] if shade < 0.4 else C["G"]
                if (x * 7 + y * 3) % 11 == 0 and shade < 0.2:
                    s[y, x] = C["l"]
    return _outline(s)


def _flower(petals: int = 5, petal: str = "R", centre: str = "Y", R: float = 5.8, stem: int = 5) -> np.ndarray:
    """A star flower: ``petals`` points of colour ``petal`` round a ``centre`` disc, on a short
    stem with a pair of leaves; shorter than a sheep."""
    N = int(np.ceil(2 * R)) + 1
    H = N + stem
    s = np.full((H, N), TRANSPARENT, np.uint8)
    cx = cy = N / 2
    for y in range(N):
        for x in range(N):
            dx, dy = x + 0.5 - cx, y + 0.5 - cy
            r, th = np.hypot(dx, dy), np.arctan2(dy, dx)
            if r <= 0.31 * R:
                s[y, x] = C[centre]
            elif r <= R * (0.55 + 0.45 * np.cos(petals * (th + np.pi / 2))):
                s[y, x] = C[petal]
    m = N // 2
    s[N - 1 : H, m - 1 : m + 1] = C["g"]
    ly = H - 2
    for k in range(2):  # leaves
        s[ly - k, m + 1 + k] = C["l"]
        if m + 2 + k < N:
            s[ly - k, m + 2 + k] = C["g"]
        s[ly - k, m - 2 - k] = C["g"]
        if m - 3 - k >= 0:
            s[ly - k, m - 3 - k] = C["l"]
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


def _facings(front_r: np.ndarray, back_r: np.ndarray, tail_tip: bool = True) -> list[np.ndarray]:
    if not tail_tip:  # the white tail tip coloured like the body
        front_r, back_r = _recolour(front_r, "W", "O"), _recolour(back_r, "W", "O")
    f, b = _outline(_up2(front_r)), _outline(_up2(back_r))
    return [b, f, f[:, ::-1], b[:, ::-1]]


def _recolour(s: np.ndarray, a: str, b: str) -> np.ndarray:
    return np.where(s == C[a], C[b], s).astype(np.uint8)


def _small_fox_facings(body: str = "O") -> list[np.ndarray]:
    """The fox at its original 1x pixel scale (half size), for size variants."""
    f, b = _recolour(FOX_FRONT_R, "O", body), _recolour(FOX_BACK_R, "O", body)
    return [b, f, f[:, ::-1], b[:, ::-1]]


ANIMALS = {"sheep": _facings(SHEEP_FRONT_R, SHEEP_BACK_R), "fox": _facings(FOX_FRONT_R, FOX_BACK_R)}
ANIMAL_NAMES = list(ANIMALS)
SHEEP = ANIMAL_NAMES.index("sheep")
FOX = ANIMAL_NAMES.index("fox")
N_FACINGS = 4

# Referent variants. Each entry is a list of four facings (static objects repeat one sprite).
# Order matters: the last HELD_OUT of each type are evaluation-only.
_static = lambda spr: [spr] * N_FACINGS
OBJECT_VARIANTS = {
    "salt": [
        _static(_salt_cube(16, 10)),  # the signed-off cube
        _static(_salt_cube(16, 6)),  # a flat slab
        _static(_salt_cube(12, 10)),
        _static(_salt_cube(14, 14)),  # tall
        _static(_salt_cube(10, 6)),  # held out: small
        _static(_salt_cube(16, 14)),  # held out: big
    ],
    "tree": [
        _static(_pine()),  # the signed-off pine
        _static(_pine([(0, 9, 2, 10), (7, 18, 4, 16)], trunk_h=5)),  # two tiers
        _static(_pine([(0, 8, 2, 8), (6, 14, 4, 12), (11, 20, 4, 14), (17, 26, 6, 16)], trunk_h=4)),  # tall, narrow
        _static(_round_tree(8.5, 8)),  # broadleaf
        _static(_pine([(0, 9, 2, 12), (7, 17, 4, 18), (13, 23, 6, 22)], trunk_h=4)),  # held out: wide pine
        _static(_round_tree(7.0, 10, squash=1.2)),  # held out: tall-crowned broadleaf
    ],
    "flower": [
        _static(_flower(5, "R", "Y")),  # the signed-off star flower
        _static(_flower(4, "Y", "D", R=5.2)),
        _static(_flower(6, "P", "Y", R=6.2, stem=4)),
        _static(_flower(5, "W", "Y", R=5.0, stem=6)),
        _static(_flower(6, "R", "Y", R=5.5, stem=6)),  # held out
        _static(_flower(4, "b", "Y", R=6.0, stem=4)),  # held out: blue
    ],
    "fox": [
        ANIMALS["fox"],  # the signed-off fox
        _facings(_recolour(FOX_FRONT_R, "O", "T"), _recolour(FOX_BACK_R, "O", "T")),  # brown fox
        [_outline(f) for f in _small_fox_facings()],  # small fox (1x art)
        _facings(FOX_FRONT_R, FOX_BACK_R, tail_tip=False),
        _facings(_recolour(FOX_FRONT_R, "O", "R"), _recolour(FOX_BACK_R, "O", "R")),  # held out: red
        [_outline(f) for f in _small_fox_facings("T")],  # held out: small brown
    ],
}
HELD_OUT = 2
OBJECT_NAMES = list(OBJECT_VARIANTS)
N_OBJECT_TYPES = len(OBJECT_NAMES)
FOX_OBJECT = OBJECT_NAMES.index("fox")
N_OBJECT_VARIANTS = np.array([len(OBJECT_VARIANTS[n]) for n in OBJECT_NAMES])
N_TRAIN_VARIANTS = N_OBJECT_VARIANTS - HELD_OUT
OBJECTS = {n: v[0] for n, v in OBJECT_VARIANTS.items()}  # first variant, as signed off


def _pad(s: np.ndarray, h: int, w: int) -> np.ndarray:
    """Bottom-centre-anchored padding to a common (h, w) box."""
    out = np.full((h, w), TRANSPARENT, np.uint8)
    sh, sw = s.shape
    x0 = (w - sw) // 2
    out[h - sh :, x0 : x0 + sw] = s
    return out


# Common box for everything that is pasted.
_all = [f for vs in OBJECT_VARIANTS.values() for fs in vs for f in fs] + [f for fs in ANIMALS.values() for f in fs]
BOX_H = max(f.shape[0] for f in _all)
BOX_W = max(f.shape[1] for f in _all)
_MAXV = int(N_OBJECT_VARIANTS.max())
# (n_obj, n_variants, 4, H, W): a type with fewer variants repeats its last one
OBJECT_STACK = np.stack(
    [np.stack([np.stack([_pad(f, BOX_H, BOX_W) for f in OBJECT_VARIANTS[n][min(v, len(OBJECT_VARIANTS[n]) - 1)]]) for v in range(_MAXV)]) for n in OBJECT_NAMES]
)
# (n_animal, 4, H, W): facing index as documented above
ANIMAL_STACK = np.stack([np.stack([_pad(f, BOX_H, BOX_W) for f in ANIMALS[n]]) for n in ANIMAL_NAMES])
