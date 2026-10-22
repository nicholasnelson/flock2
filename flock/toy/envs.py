"""Select the episode generator by config: the top-down baseline or the isometric renderer.

Both modules expose the same surface: ``Episode``, ``make_batch``, ``object_views``,
``plain_ground_view``, ``PROBE_CANVAS_PX``, ``N_TYPES``, ``OBJECT_NAMES``, ``object_thumbnail``,
``CELL`` (px per receiver cell) and ``CANVAS`` (px side of the drawing in the view).
"""

from __future__ import annotations

from types import ModuleType

import numpy as np

from flock.toy import env as env_topdown
from flock.toy import env_iso


def get(cfg) -> ModuleType:
    r = getattr(cfg, "renderer", "topdown")
    if r == "topdown":
        return env_topdown
    if r == "iso":
        return env_iso
    raise ValueError(f"unknown renderer {r!r}")


def object_thumbnail(cfg, t: int) -> np.ndarray:
    if getattr(cfg, "renderer", "topdown") == "iso":
        return env_iso.object_thumbnail(t)
    from flock import sprites
    from flock.palette import TRANSPARENT

    cell = np.full((16, 16), 1, np.uint8)
    s = sprites.OBJECT_STACK[t]
    m = s != TRANSPARENT
    cell[4:12, 4:12][m] = s[m]
    return cell
