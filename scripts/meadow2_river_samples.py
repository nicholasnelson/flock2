#!/usr/bin/env python
"""What the river meadow looks like (spec section 21.8): four meadows top-down (water, the mud band,
the willows along the banks, the other objects, the flock's start), then first-person frames from a
sheep 6 tiles from the river facing it and facing away, at midday and at night, with the skyline.

    FLOCK2_TILES=72 JAX_PLATFORMS=cpu python scripts/meadow2_river_samples.py --out runs/meadow2/river_samples.png
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from flock.meadow2 import render as R
from flock.meadow2 import world as W


def top_down(st, px=4):
    """(N*px, N*px, 3) map: grass green scaled by the regrowth map (mud dark), water blue, objects as dots."""
    m = st.meadow
    n = W.N_TILES
    g = np.asarray(m.grass.reshape(n, W.GRASS_RES, n, W.GRASS_RES).mean((1, 3))) / 3.0
    rm = np.asarray(m.regrow_map.reshape(n, W.GRASS_RES, n, W.GRASS_RES).mean((1, 3)))
    img = np.stack([0.35 + 0.1 * g, 0.45 + 0.4 * g, 0.2 + 0.1 * g], -1) * (0.55 + 0.45 * rm)[..., None]
    img[np.asarray(m.water)] = (0.2, 0.45, 0.85)
    img = np.kron(img, np.ones((px, px, 1)))
    colours = {W.WILLOW: (0.85, 0.95, 0.6), W.TREE: (0.05, 0.3, 0.05), W.HUT: (0.6, 0.3, 0.1)}
    for kind, pos, present in zip(np.asarray(m.obj_kind), np.asarray(m.obj_pos), np.asarray(m.obj_present)):
        if present and int(kind) in colours:
            r, c = (pos * px).astype(int)
            img[max(r - 2, 0):r + 3, max(c - 2, 0):c + 3] = colours[int(kind)]
    for pos in np.asarray(st.flock.pos):
        r, c = (pos * px).astype(int)
        img[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2] = (1.0, 1.0, 1.0)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/meadow2/river_samples.png")
    a = p.parse_args()
    maps = []
    for seed in range(4):
        st = W.random_start(jax.random.PRNGKey(seed), willows=3, river=True, skyline=True)
        st = W.with_terrain(st, bank=10.0, mud=True, inner=8.0)
        maps.append(top_down(st))
        maps.append(np.ones((maps[-1].shape[0], 6, 3)))
    top = np.concatenate(maps[:-1], 1)

    # first-person: sheep 0 on the far side of the mud, facing the river and facing away, midday and night
    st = W.random_start(jax.random.PRNGKey(1), willows=3, river=True, skyline=True)
    st = W.with_terrain(st, bank=10.0, mud=True, inner=8.0)
    c, normal = W.river_geometry(st.meadow.water)
    side = jnp.sign(jnp.dot(st.flock.pos[0] - c, normal))
    pos = c + side * normal * (W.RIVER_WIDTH / 2 + 6.0)
    toward = jnp.arctan2(-side * normal[1], side * normal[0])  # heading h has direction (-cos h, sin h) in (row, col)
    strips = []
    for t in (int(0.3 * W.DAY), int(0.8 * W.DAY)):
        for h in (toward, toward + jnp.pi):
            f = st.flock
            s = st._replace(flock=f._replace(pos=f.pos.at[0].set(pos), heading=f.heading.at[0].set(h)), t=jnp.int32(t))
            strips.append(np.asarray(R.render(s, jnp.int32(0))))
            strips.append(np.ones((2, R.W, 3)))
    fpv = np.concatenate(strips, 0)
    fpv = np.kron(fpv, np.ones((2, 2, 1)))
    w = max(top.shape[1], fpv.shape[1])
    pad = lambda x: np.concatenate([x, np.ones((x.shape[0], w - x.shape[1], 3))], 1)
    img = (np.clip(np.concatenate([pad(top), np.ones((8, w, 3)), pad(fpv)], 0), 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img).save(a.out)
    print("wrote", a.out, img.shape)


if __name__ == "__main__":
    main()
