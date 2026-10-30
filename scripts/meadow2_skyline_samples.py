#!/usr/bin/env python
"""What the skyline looks like (spec section 21.7): sheep 0 at the meadow's centre turning through eight
headings, at midday, at dusk, at night and on a fog day; strips stacked into one PNG.

    JAX_PLATFORMS=cpu python scripts/meadow2_skyline_samples.py --out runs/meadow2/skyline_samples.png
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from flock.meadow2 import render as R
from flock.meadow2.world import DAY, MAX_DAYS, random_start


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--out", default="runs/meadow2/skyline_samples.png")
    a = p.parse_args()
    st = random_start(jax.random.PRNGKey(a.seed), willows=3, skyline=True)
    strips = []
    for t, fog in ((int(0.3 * DAY), False), (int(0.58 * DAY), False), (int(0.8 * DAY), False), (int(0.3 * DAY), True)):
        m = st.meadow._replace(fog_days=jnp.full((MAX_DAYS,), fog))
        for k in range(8):
            f = st.flock
            heading = jnp.float32(k * np.pi / 4)
            s = st._replace(meadow=m, flock=f._replace(pos=f.pos.at[0].set(jnp.array([24.0, 24.0])), heading=f.heading.at[0].set(heading)), t=jnp.int32(t))
            strips.append(np.asarray(R.render(s, jnp.int32(0))))
            strips.append(np.ones((2, R.W, 3)))
        strips.append(np.zeros((6, R.W, 3)))
    img = (np.concatenate(strips, 0) * 255).astype(np.uint8)
    Image.fromarray(img).resize((R.W * 2, img.shape[0] * 2), Image.NEAREST).save(a.out)
    print("wrote", a.out, img.shape)


if __name__ == "__main__":
    main()
