"""Pixel-art sprites, drawn in source with the palette character codes.

Scale: 4 px per tile. Ground tiles are 4x4. Objects and animals are 8x8 (2x2 tiles) so that
they have enough shape to be recognisable, and enough to be *drawn* on a 16x16 canvas.
These sprites are the referents the sheep will learn to depict, so their legibility matters.
"""

from __future__ import annotations

import numpy as np

from flock.palette import CHAR_TO_INDEX, TRANSPARENT

SPRITE = 8  # object sprite size in px
TILE = 4  # px per tile


def _parse(rows: list[str]) -> np.ndarray:
    h = len(rows)
    w = len(rows[0])
    out = np.full((h, w), TRANSPARENT, dtype=np.uint8)
    for y, row in enumerate(rows):
        assert len(row) == w, f"ragged sprite row {y}: {row!r}"
        for x, ch in enumerate(row):
            if ch == ".":
                continue
            out[y, x] = CHAR_TO_INDEX[ch]
    return out


# --- ground tiles (4x4, no transparency) --------------------------------------------------

GROUND_TILES = {
    "plain": _parse(["GGGG", "GGGG", "GGGG", "GGGG"]),
    "speck": _parse(["GGGG", "GgGG", "GGGG", "GGGg"]),
    "speck2": _parse(["GGGg", "GGGG", "gGGG", "GGGG"]),
    "earth": _parse(["tttt", "tGtt", "tttt", "ttGt"]),
}
GROUND_NAMES = list(GROUND_TILES)
GROUND_STACK = np.stack([GROUND_TILES[n] for n in GROUND_NAMES])  # (n_ground, 4, 4)

# --- objects (8x8, transparent background) ------------------------------------------------

OBJECT_SPRITES = {
    "grass": _parse(
        [
            "........",
            "..l..l..",
            ".l.l.l..",
            ".l.ll.l.",
            "..lll.l.",
            "..ll.l..",
            "...ll...",
            "...gg...",
        ]
    ),
    "pond": _parse(
        [
            "........",
            "..BBBB..",
            ".BBbBBB.",
            "BBBBBBBB",
            "BBBBBBbB",
            ".BbBBBB.",
            "..BBBB..",
            "........",
        ]
    ),
    "salt": _parse(
        [
            "........",
            ".WWWWWW.",
            ".WWWWWS.",
            ".WWWWWS.",
            ".WWWWWS.",
            ".WSSSSS.",
            "........",
            "........",
        ]
    ),
    "tree": _parse(
        [
            "..DDDD..",
            ".DggggD.",
            "DggglggD",
            "DgglgggD",
            ".DggggD.",
            "..DggD..",
            "...TT...",
            "...TT...",
        ]
    ),
    "flower": _parse(
        [
            "........",
            "...RR...",
            "..RYYR..",
            "..RYYR..",
            "...RR...",
            "...gg...",
            "..g.g...",
            "...g....",
        ]
    ),
    "rock": _parse(
        [
            "........",
            "...SS...",
            "..SSSS..",
            ".SSSsSS.",
            ".SSSsss.",
            ".Ssssss.",
            "..ssss..",
            "........",
        ]
    ),
}
OBJECT_NAMES = list(OBJECT_SPRITES)
OBJECT_STACK = np.stack([OBJECT_SPRITES[n] for n in OBJECT_NAMES])  # (n_obj, 8, 8)
N_OBJECT_TYPES = len(OBJECT_NAMES)

# --- animals (8x8, facing up / north; rotate with np.rot90 for other headings) ------------

ANIMAL_SPRITES = {
    "sheep": _parse(
        [
            "...ss...",
            "..sPPs..",
            ".WWssWW.",
            "WWWWWWWW",
            "WWWWWWWW",
            ".WWWWWW.",
            ".K.KK.K.",
            "........",
        ]
    ),
    "fox": _parse(
        [
            ".O....O.",
            ".OO..OO.",
            "..OKKO..",
            "..OOOO..",
            ".OOOOOO.",
            ".OOOOOO.",
            "..O..O..",
            "...WW...",
        ]
    ),
}
ANIMAL_NAMES = list(ANIMAL_SPRITES)
ANIMAL_STACK = np.stack([ANIMAL_SPRITES[n] for n in ANIMAL_NAMES])


def rotated_stack(stack: np.ndarray) -> np.ndarray:
    """(n, 8, 8) -> (n, 4, 8, 8): sprite for each of 4 headings (0=N, 1=E, 2=S, 3=W).

    Heading k means the sprite is rotated clockwise by 90k degrees.
    """
    return np.stack([np.rot90(stack, k=-k, axes=(1, 2)) for k in range(4)], axis=1)
