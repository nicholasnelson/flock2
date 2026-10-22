"""PNG galleries and a human-readability quiz for Milestone 0 drawings."""

from __future__ import annotations

import json
import os

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from flock.palette import to_rgb
from flock.raster import canvas_to_indices
from flock.toy import envs, game


def _scale(img_rgb: np.ndarray, s: int) -> Image.Image:
    im = Image.fromarray(img_rgb)
    return im.resize((im.width * s, im.height * s), Image.NEAREST)


def drawings_grid(cfg, canvases_idx: np.ndarray, types: np.ndarray, per_type: int = 10, gap: int = 1) -> np.ndarray:
    """Rows = object types (sprite first), columns = examples. Returns a palette-index image."""
    n_types = envs.get(cfg).N_TYPES
    cell = 16 + gap
    grid = np.full((n_types * cell + gap, (per_type + 2) * cell + gap), 10, np.uint8)
    for t in range(n_types):
        r = gap + t * cell
        grid[r : r + 16, gap : gap + 16] = envs.object_thumbnail(cfg, t)
        idx = np.where(types == t)[0][:per_type]
        for j, i in enumerate(idx):
            c = gap + (j + 2) * cell
            grid[r : r + 16, c : c + 16] = canvases_idx[i]
    return grid


def export_run_gallery(model, cfg, key: jax.Array, n: int = 192) -> None:
    out = cfg.out_dir
    E = envs.get(cfg)
    k_ep, k_s = jax.random.split(key)
    ep = E.make_batch(k_ep, n, cfg.n_candidates, cfg.mode, cfg.random_heading)
    fwd = game.forward(model, ep, k_s, cfg.opaque, hard=True)
    bg = 8 if cfg.opaque else 1
    idx = np.asarray(jax.vmap(lambda c, a: canvas_to_indices(type(fwd.canvas)(c, a, 0.0), background=bg))(fwd.canvas.colour, fwd.canvas.alpha))
    types = np.asarray(ep.target_type)
    grid = drawings_grid(cfg, idx, types)
    _scale(to_rgb(grid), 6).save(os.path.join(out, "gallery.png"))

    # a few receiver views with the canvas composited, and the sender views
    rv = np.asarray(jnp.argmax(fwd.receiver_view[:6], -1)).astype(np.uint8)
    sv = np.asarray(ep.sender_view[:6]).astype(np.uint8)
    row = np.concatenate([np.concatenate([sv[i], rv[i]], axis=1) for i in range(6)], axis=0)
    _scale(to_rgb(row), 3).save(os.path.join(out, "views.png"))

    # quiz: shuffled drawings, answer key kept separately
    rng = np.random.default_rng(0)
    order = rng.permutation(min(60, n))
    cell = 18
    cols = 10
    rows = (len(order) + cols - 1) // cols
    quiz = np.full((rows * cell + 2, cols * cell + 2), 10, np.uint8)
    for j, i in enumerate(order):
        r, c = divmod(j, cols)
        quiz[2 + r * cell : 2 + r * cell + 16, 2 + c * cell : 2 + c * cell + 16] = idx[i]
    _scale(to_rgb(quiz), 6).save(os.path.join(out, "quiz.png"))
    key_ = {"order": [int(i) for i in order], "types": [E.OBJECT_NAMES[int(types[i])] for i in order], "layout": f"{rows}x{cols}, row-major, numbered from 0"}
    with open(os.path.join(out, "quiz_key.json"), "w") as f:
        json.dump(key_, f, indent=2)
