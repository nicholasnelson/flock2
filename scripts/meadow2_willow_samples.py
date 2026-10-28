#!/usr/bin/env python
"""What a willow looks like to a sheep (stage 1, spec § 20): sheep 0 faces a willow from several
distances by day and by night; the strips are stacked into one PNG.

    JAX_PLATFORMS=cpu python scripts/meadow2_willow_samples.py --out runs/meadow2/willow_samples.png
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from flock.meadow2 import render as R
from flock.meadow2.world import DAY, WILLOW, random_start


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--out", default="runs/meadow2/willow_samples.png")
    a = p.parse_args()
    st = random_start(jax.random.PRNGKey(a.seed), willows=3)
    w = np.asarray(st.meadow.obj_kind) == WILLOW
    target = np.asarray(st.meadow.obj_pos)[w][0]
    strips = []
    for t in (int(0.3 * DAY), int(0.8 * DAY)):  # midday, night
        for dist in (3.0, 8.0, 12.0, 16.0):
            f = st.flock
            pos = jnp.asarray(target) + jnp.array([dist, 0.0])
            heading = jnp.arctan2(target[1] - pos[1], -(target[0] - pos[0]))
            s = st._replace(flock=f._replace(pos=f.pos.at[0].set(pos), heading=f.heading.at[0].set(heading)), t=jnp.int32(t))
            strips.append(np.asarray(R.render(s, jnp.int32(0))))
            strips.append(np.ones((2, R.W, 3)))
    img = (np.concatenate(strips, 0) * 255).astype(np.uint8)
    Image.fromarray(img).resize((R.W * 2, img.shape[0] * 2), Image.NEAREST).save(a.out)
    print("wrote", a.out, img.shape)


if __name__ == "__main__":
    main()
