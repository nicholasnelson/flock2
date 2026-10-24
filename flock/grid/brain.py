"""The sheep's brain sized for the grid world: the same world model (``flock.wm``) and
actor-critic (``flock.ac``) as the meadow, with a small configuration."""

from __future__ import annotations

from flock import wm
from flock.grid import world as Wd

GRID = wm.Config(
    h=Wd.V,
    w=Wd.V,
    n_classes=Wd.N_CLASSES,
    n_actions=Wd.N_ACTIONS,
    enc_ch=(32, 64),  # 8x8 -> 4x4 -> 2x2
    bottom=(2, 2),
    deter=128,
    hidden=128,
    stoch=16,
    classes=8,
    embed=128,
    body_dims=(0, 1),  # food, water
    intero_weight=10.0,  # the view's cross-entropy sums only 64 cells
)


GRID_FULL = GRID._replace(h=Wd.G, w=Wd.G, n_classes=Wd.N_CLASSES_FULL, enc_ch=(32, 64, 64), bottom=(2, 2))  # 16 -> 8 -> 4 -> 2


def config(view: str) -> wm.Config:
    return GRID if view == "local" else GRID_FULL
