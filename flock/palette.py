"""The 16-colour palette shared by the world view and the drawing canvas.

Index 0 is reserved for "void" (outside the world / unlit at night). Everything a sheep sees or
draws is one of these 16 indices; RGB is only for humans.
"""

from __future__ import annotations

import numpy as np

PALETTE: list[tuple[str, tuple[int, int, int]]] = [
    ("void", (0, 0, 0)),
    ("grass_dark", (46, 102, 46)),
    ("grass", (76, 150, 60)),
    ("grass_light", (140, 200, 90)),
    ("water", (50, 90, 200)),
    ("water_light", (120, 170, 240)),
    ("bark", (120, 80, 40)),
    ("earth", (180, 150, 100)),
    ("white", (245, 245, 240)),
    ("stone", (150, 150, 150)),
    ("stone_dark", (90, 90, 95)),
    ("fox", (220, 120, 40)),
    ("red", (210, 50, 50)),
    ("yellow", (240, 210, 60)),
    ("pink", (235, 170, 180)),
    ("outline", (40, 30, 20)),
]

N_COLOURS = len(PALETTE)
assert N_COLOURS == 16

NAMES = [name for name, _ in PALETTE]
RGB = np.array([rgb for _, rgb in PALETTE], dtype=np.uint8)

# Single-character codes used to draw sprites in source.
CHAR_TO_INDEX = {
    "K": 0,  # void / black
    "G": 1,  # grass dark (ground base)
    "g": 2,  # grass
    "l": 3,  # grass light
    "B": 4,  # water
    "b": 5,  # water light
    "T": 6,  # bark
    "t": 7,  # earth
    "W": 8,  # white
    "S": 9,  # stone
    "s": 10,  # stone dark
    "O": 11,  # fox orange
    "R": 12,  # red
    "Y": 13,  # yellow
    "P": 14,  # pink
    "D": 15,  # outline / dark brown
}
TRANSPARENT = 255  # marker in sprite arrays: the pixel underneath shows through


def to_rgb(indices: np.ndarray) -> np.ndarray:
    """Palette-index image (..., H, W) -> uint8 RGB image (..., H, W, 3)."""
    return RGB[np.asarray(indices, dtype=np.int64)]
