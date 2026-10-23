#!/usr/bin/env python
"""SPIKE benchmark: first-person strip view (flock/fpv.py) vs the isometric view (iso.render_view).

    python scripts/bench_fpv.py                 # throughput table
    python scripts/bench_fpv.py --preview out   # also save a preview image of both views

Same world for both: a square meadow of random grass states with N referent objects scattered
on it; B agents per batch, each at a random place and heading. Reports views per second
(batched, jitted, after warm-up).
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from flock import fpv, iso, sprites_iso as S


def world(key, n_tiles: int, n_obj: int):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    tm = jax.random.choice(k1, 4, (n_tiles, n_tiles), p=jnp.array([0.05, 0.55, 0.25, 0.15])).astype(jnp.int32)
    tm = tm.at[n_tiles // 3 : n_tiles // 3 + 3, n_tiles // 3 : n_tiles // 3 + 4].set(S.WATER)
    kind = jax.random.randint(k2, (n_obj,), 0, S.N_OBJECT_TYPES)
    pos = jax.random.uniform(k3, (n_obj, 2), minval=1.0, maxval=n_tiles - 1.0)
    variant = jax.random.randint(k4, (n_obj,), 0, 4)
    return tm, kind, pos, variant


def bench(fn, args, n=20):
    jax.block_until_ready(fn(*args))
    t = time.time()
    for _ in range(n):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.time() - t) / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--preview", default=None)
    a = p.parse_args()
    print(jax.devices())
    rows = []
    for n_tiles, n_obj in [(24, 8), (64, 40), (128, 120)]:
        tm, kind, pos, variant = world(jax.random.PRNGKey(0), n_tiles, n_obj)
        B = a.batch
        ke, kh = jax.random.split(jax.random.PRNGKey(1))
        eye = jax.random.uniform(ke, (B, 2), minval=2.0, maxval=n_tiles - 2.0)
        head4 = jax.random.randint(kh, (B,), 0, 4)
        sprites = fpv.object_billboards(kind, variant, jnp.zeros_like(kind))
        present = jnp.ones((n_obj,), bool)

        f_fpv = jax.jit(jax.vmap(lambda e, h: fpv.render_fpv(tm, sprites, pos, present, e, h)))
        objs = iso.Objects(kind=kind, pos=jnp.floor(pos).astype(jnp.int32), present=present, variant=variant)
        ag = iso.Agents(kind=jnp.array([S.SHEEP]), pos=jnp.zeros((1, 2), jnp.int32), heading=jnp.zeros((1,), jnp.int32), present=jnp.array([False]))
        f_iso = jax.jit(jax.vmap(lambda e, h: iso.render_view(tm, objs, ag, jnp.floor(e).astype(jnp.int32), h)))
        t_fpv = bench(f_fpv, (eye, head4.astype(jnp.float32) * (jnp.pi / 2)))
        t_iso = bench(f_iso, (eye, head4))
        rows.append((n_tiles, n_obj, B / t_iso, B / t_fpv, t_fpv / t_iso))
        if a.preview and n_tiles == 64:
            from PIL import Image

            from flock.palette import to_rgb

            v_f = np.asarray(f_fpv(eye[:4], head4[:4].astype(jnp.float32) * (jnp.pi / 2)))
            v_i = np.asarray(f_iso(eye[:4], head4[:4]))
            strip = np.concatenate([np.pad(v, ((2, 2), (0, 0)), constant_values=10) for v in v_f], 0)
            Image.fromarray(to_rgb(strip)).resize((strip.shape[1] * 3, strip.shape[0] * 3), Image.NEAREST).save(f"{a.preview}/fpv_strips.png")
            iv = np.concatenate(list(v_i), 1)
            Image.fromarray(to_rgb(iv)).resize((iv.shape[1] * 2, iv.shape[0] * 2), Image.NEAREST).save(f"{a.preview}/iso_views.png")
    print(f"batch {a.batch}")
    print(f"{'world':>10s} {'objects':>8s} {'iso views/s':>12s} {'fpv views/s':>12s} {'fpv/iso cost':>13s}")
    for n_tiles, n_obj, vi, vf, ratio in rows:
        print(f"{n_tiles:>8d}^2 {n_obj:>8d} {vi:12.0f} {vf:12.0f} {ratio:12.2f}x")


if __name__ == "__main__":
    main()
