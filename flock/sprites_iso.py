"""Isometric pixel-art sprites: 8x4 diamond ground tiles and single-tile, camera-facing sprites.

Every sprite is one tile and billboarded (always drawn upright, facing the camera), so the tile
layout can be rotated under a fixed camera: that is what makes heading-aligned isometric views
possible. Ground states are drawn per position (never pixel-rotated), so grass texture is also
rotation-safe. Water is a ground state (a pond is a cluster of water tiles), not an object.

Sprites are anchored at the bottom-centre of their tile's diamond.
"""

from __future__ import annotations

import numpy as np

from flock.palette import CHAR_TO_INDEX, TRANSPARENT

TILE_W, TILE_H = 8, 4  # diamond bounding box


def _parse(rows: list[str]) -> np.ndarray:
    w = max(len(r) for r in rows)
    out = np.full((len(rows), w), TRANSPARENT, dtype=np.uint8)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch != ".":
                out[y, x] = CHAR_TO_INDEX[ch]
    return out


# --- ground states (8x4 diamonds, transparent corners), four texture variants each ---------
# Stalks are 2-px lines (vertical or leaning) with gaps between them; a per-tile hash picks the
# variant so the tiling does not repeat. "short" is plain.

GROUND_VARIANTS = {
    "bare": [
        _parse(["...tt...", ".ttGttt.", ".tttttt.", "...tt..."]),
        _parse(["...tt...", ".tttttt.", ".tGtttt.", "...tt..."]),
        _parse(["...tt...", ".tttttG.", ".tttttt.", "...tt..."]),
        _parse(["...tt...", ".tttttt.", ".ttttGt.", "...tt..."]),
    ],
    "short": [_parse(["...GG...", ".GGGGGG.", ".GGGGGG.", "...GG..."])] * 4,
    "medium": [
        _parse(["...GG...", ".GlGGGG.", ".GlGGGG.", "...GG..."]),
        _parse(["...GG...", ".GGGGlG.", ".GGGGlG.", "...GG..."]),
        _parse(["...GG...", ".GGGlGG.", ".GGGGlG.", "...GG..."]),
        _parse(["...GG...", ".GGGGGG.", ".GGlGGG.", "...GG..."]),
    ],
    "long": [
        _parse(["...lG...", ".GlGGlG.", ".GlGGlG.", "...GG..."]),
        _parse(["...GG...", ".GGlGGl.", ".GlGGGl.", "...GG..."]),
        _parse(["...Gl...", ".lGGGlG.", ".lGGGGl.", "...GG..."]),
        _parse(["...GG...", ".GGlGlG.", ".GlGGlG.", "...lG..."]),
    ],
    "water": [
        _parse(["...BB...", ".BBbBBB.", ".BBBBBB.", "...BB..."]),
        _parse(["...BB...", ".BBBBBB.", ".BBBBbB.", "...BB..."]),
        _parse(["...BB...", ".BBBBBB.", ".BbBBBB.", "...BB..."]),
        _parse(["...BB...", ".BBBBbB.", ".BBBBBB.", "...BB..."]),
    ],
}
GROUND_NAMES = list(GROUND_VARIANTS)
N_VARIANTS = 4
GROUND = {n: v[0] for n, v in GROUND_VARIANTS.items()}  # first variant, for tests/thumbnails
GROUND_STACK = np.stack([np.stack(GROUND_VARIANTS[n]) for n in GROUND_NAMES])  # (n_ground, 4, 4, 8)
N_GROUND = len(GROUND_NAMES)
GRASS_STATES = [GROUND_NAMES.index(n) for n in ("bare", "short", "medium", "long")]
WATER = GROUND_NAMES.index("water")

# Diamond mask shared by all ground tiles (which pixels of the 8x4 box belong to the tile).
DIAMOND = GROUND["short"] != TRANSPARENT


def tile_variant(r, c):
    """Per-tile texture variant index (0..3) from a scrambled hash of the tile position.
    Works on ints and on integer arrays (numpy or jax)."""
    return ((r * 73856093) ^ (c * 19349663) ^ ((r + c) * 83492791)) % N_VARIANTS


# --- objects (single tile, camera-facing) -----------------------------------------------------

OBJECTS = {
    "salt": _parse(
        [
            "...WW...",
            ".WWWWWW.",
            "WWWWWWWW",
            "WWSWWWSW",
            "WWSWWWSW",
            ".SSSWSS.",
            "...SS...",
        ]
    ),
    "tree": _parse(
        [
            "...DDDD...",
            ".DDggggDD.",
            "DgggglgggD",
            "DgglggglgD",
            ".DgggggDD.",
            "..DDgggD..",
            "....TT....",
            "....TT....",
            "...DTTD...",
        ]
    ),
    "flower": _parse(
        [
            "..RR..",
            ".RYYR.",
            "..RR..",
            "..g...",
            ".gg.g.",
            "..gg..",
        ]
    ),
    "rock": _parse(
        [
            "..SSS...",
            ".SSSSSs.",
            "SSSSssss",
            ".Sssssss",
            "..ssss..",
        ]
    ),
    "bush": _parse(
        [
            "..gggg..",
            ".glgggg.",
            "gggglggg",
            ".DggggDg",
            "..DDDD..",
        ]
    ),
    "stump": _parse(
        [
            "..TTTT..",
            ".TtttttT",
            ".TTTTTTT",
            ".DTTTTD.",
            "..DDDD..",
        ]
    ),
}
OBJECT_NAMES = list(OBJECTS)
N_OBJECT_TYPES = len(OBJECT_NAMES)
SPRITE_H = max(s.shape[0] for s in OBJECTS.values())
SPRITE_W = max(s.shape[1] for s in OBJECTS.values())


def _pad(s: np.ndarray, h: int, w: int) -> np.ndarray:
    """Bottom-centre-anchored padding to a common (h, w) box."""
    out = np.full((h, w), TRANSPARENT, np.uint8)
    sh, sw = s.shape
    x0 = (w - sw) // 2
    out[h - sh :, x0 : x0 + sw] = s
    return out


# --- animals (single tile, facing right; mirrored for left) ----------------------------------

SHEEP_R = _parse(
    [
        "..WWWW....",
        ".WWWWWWss.",
        "WWWWWWWsPs",
        "WWWWWWWWss",
        ".WWWWWWW..",
        "..K.K.K...",
        "..K...K...",
    ]
)
FOX_R = _parse(
    [
        "......OO..",
        "WO...OOKO.",
        "OOO.OOOOOO",
        ".OOOOOOOO.",
        "..OOOOOO..",
        "..K..K.K..",
    ]
)
ANIMALS = {"sheep": SHEEP_R, "fox": FOX_R}
ANIMAL_NAMES = list(ANIMALS)
SHEEP = ANIMAL_NAMES.index("sheep")
FOX = ANIMAL_NAMES.index("fox")

# Common box for everything that is pasted: objects and animals, two facings for animals.
BOX_H = max(SPRITE_H, max(s.shape[0] for s in ANIMALS.values()))
BOX_W = max(SPRITE_W, max(s.shape[1] for s in ANIMALS.values()))
OBJECT_STACK = np.stack([_pad(OBJECTS[n], BOX_H, BOX_W) for n in OBJECT_NAMES])  # (n_obj, H, W)
# (n_animal, 2, H, W): facing 0 = right, 1 = left
ANIMAL_STACK = np.stack([np.stack([_pad(ANIMALS[n], BOX_H, BOX_W), _pad(ANIMALS[n][:, ::-1], BOX_H, BOX_W)]) for n in ANIMAL_NAMES])
