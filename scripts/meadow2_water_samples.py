#!/usr/bin/env python
"""What water looks like to a sheep: sheep 0 stands on dry ground facing the nearest shore of a pond
from several distances, by day and by night; the strips are stacked into one PNG with labels.

    JAX_PLATFORMS=cpu python scripts/meadow2_water_samples.py --out runs/meadow2/water_samples.png
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw

from flock.meadow2 import render as R
from flock.meadow2.world import DAY, N_TILES, random_start, water_at


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--willows", type=int, default=0)
    p.add_argument("--distances", default="4,8,12,16")
    p.add_argument("--out", default="runs/meadow2/water_samples.png")
    a = p.parse_args()
    st = random_start(jax.random.PRNGKey(a.seed), willows=a.willows)
    water = np.asarray(st.meadow.water)
    yy, xx = np.nonzero(water)
    tiles = np.stack([yy + 0.5, xx + 0.5], -1)
    centre = tiles.mean(0)  # of all water; fine for picking an outward direction from one pond
    # the water tile nearest the meadow's centre, and the outward direction from its pond
    pond = tiles[np.argmin(np.linalg.norm(tiles - centre, axis=-1))]
    near = tiles[np.linalg.norm(tiles - pond, axis=-1) < 6]
    out = pond - near.mean(0)
    out = out / max(np.linalg.norm(out), 1e-6)
    strips, labels = [], []
    for t, name in ((int(0.3 * DAY), "day"), (int(0.8 * DAY), "night")):
        for dist in (float(x) for x in a.distances.split(",")):
            pos = np.clip(pond + dist * out, 1.0, N_TILES - 1.0)
            if bool(water_at(st.meadow.water, jnp.asarray(pos)[None])[0]):
                continue
            f = st.flock
            heading = jnp.arctan2(pond[1] - pos[1], -(pond[0] - pos[0]))
            s = st._replace(flock=f._replace(pos=f.pos.at[0].set(jnp.asarray(pos)), heading=f.heading.at[0].set(heading)), t=jnp.int32(t))
            strips.append(np.asarray(R.render(s, jnp.int32(0))))
            strips.append(np.ones((2, R.W, 3)))
            labels.append(f"{name}, water {dist:.0f} tiles ahead")
    img = (np.concatenate(strips, 0) * 255).astype(np.uint8)
    im = Image.fromarray(img).resize((R.W * 3, img.shape[0] * 3), Image.NEAREST)
    d = ImageDraw.Draw(im)
    for i, lab in enumerate(labels):
        d.text((6, i * (R.H + 2) * 3 + 4), lab, fill=(255, 255, 255))
    im.save(a.out)
    print("wrote", a.out, labels)


if __name__ == "__main__":
    main()
