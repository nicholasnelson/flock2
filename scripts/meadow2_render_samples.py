#!/usr/bin/env python
"""Sample views from renderer v2 (``flock.meadow2.render``), for checking by eye.

    python scripts/meadow2_render_samples.py --out docs/flock-2/meadow2-results/render_samples.png

One meadow, sheep 0, at several moments: a strip per row, enlarged 4 x with nearest-neighbour
pixels, labelled with the time and what the sheep faces. The reticle (mouth point) is drawn as
a small ring.
"""

from __future__ import annotations

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw

from flock.meadow2 import render as R, sim
from flock.meadow2.world import DAY, random_start


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--out", default="docs/flock-2/meadow2-results/render_samples.png")
    a = p.parse_args()
    st = random_start(jax.random.PRNGKey(a.seed))
    m = st.meadow
    # face the nearest water from where the flock is, and stand 2.5 tiles from it
    w = np.argwhere(np.asarray(m.water)) + 0.5
    me = np.asarray(st.flock.pos[0])
    tgt = w[np.argmin(np.linalg.norm(w - me, axis=1))]
    v = tgt - me
    head_w = float(np.arctan2(v[1], -v[0]))
    near_w = tgt - 2.5 * v / np.linalg.norm(v)
    rend = jax.jit(R.render, static_argnums=1)
    ret = jax.jit(R.reticle_row, static_argnums=1)
    cases = [
        ("midday, facing the flock's nearest pond", 0.3 * DAY, me, head_w),
        ("midday, 2.5 tiles from the pond", 0.3 * DAY, near_w, head_w),
        ("midday, facing away from the pond", 0.3 * DAY, me, head_w + np.pi),
        ("dusk", 0.6 * DAY, me, head_w),
        ("night (hut light if in view)", 0.8 * DAY, me, head_w),
        ("dawn", 0.02 * DAY, me, head_w),
    ]
    S = 4
    rows = []
    for label, t, pos, h in cases:
        s2 = st._replace(t=jnp.int32(int(t)), flock=st.flock._replace(pos=st.flock.pos.at[0].set(jnp.asarray(pos, jnp.float32)), heading=st.flock.heading.at[0].set(h)))
        img = (np.asarray(rend(s2, 0)) * 255).astype(np.uint8)
        big = Image.fromarray(img).resize((R.W * S, R.H * S), Image.NEAREST)
        d = ImageDraw.Draw(big)
        ry = float(ret(s2, 0)) * S
        d.ellipse([R.W * S / 2 - 7, ry - 7, R.W * S / 2 + 7, ry + 7], outline=(255, 235, 120), width=2)
        canvas = Image.new("RGB", (R.W * S, R.H * S + 20), (20, 24, 28))
        canvas.paste(big, (0, 20))
        ImageDraw.Draw(canvas).text((6, 4), label, fill=(230, 230, 230))
        rows.append(canvas)
    out = Image.new("RGB", (R.W * S, sum(r.height for r in rows)))
    y = 0
    for r in rows:
        out.paste(r, (0, y))
        y += r.height
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out.save(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
