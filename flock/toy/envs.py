"""Select the episode generator by config: the top-down baseline or the isometric renderer.

Both modules expose the same surface: ``Episode``, ``make_batch``, ``object_views``,
``plain_ground_view``, ``PROBE_CANVAS_PX``, ``N_TYPES``, ``OBJECT_NAMES``, ``object_thumbnail``,
``CELL`` (px per receiver cell) and ``CANVAS`` (px side of the drawing in the view).
"""

from __future__ import annotations

from types import ModuleType

import jax.numpy as jnp
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


def make_batch(cfg, key, batch: int, held_out: bool = False):
    """A batch of episodes for ``cfg``: instance variation (iso only) when ``cfg.variation``;
    ``held_out`` draws variants from the evaluation-only set."""
    E = get(cfg)
    if getattr(cfg, "renderer", "topdown") == "iso":
        v = bool(getattr(cfg, "variation", False))
        return E.make_batch(key, batch, cfg.n_candidates, cfg.mode, cfg.random_heading, v, held_out and v)
    return E.make_batch(key, batch, cfg.n_candidates, cfg.mode, cfg.random_heading)


def object_views(cfg, key, n: int):
    """Single-object views (types) for fitting probes; training variants if ``cfg.variation``."""
    E = get(cfg)
    if getattr(cfg, "renderer", "topdown") == "iso":
        return E.object_views(key, n, bool(getattr(cfg, "variation", False)))
    return E.object_views(key, n)


def probe_px(cfg, ep):
    """Per-episode top-left of the canvas laid over the sender's referent, (B, 2)."""
    if getattr(ep, "probe_px", None) is not None:
        return ep.probe_px
    return jnp.broadcast_to(get(cfg).PROBE_CANVAS_PX, (ep.target_type.shape[0], 2))
