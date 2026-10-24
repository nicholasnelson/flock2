#!/usr/bin/env python
"""The third review's S5 diagnostic: where is the direction to water lost in the meadow? If the
world model's decoder draws water in the right columns of the strip (the right azimuth), the
direction is in the state and the probes failed to read it; if not, the model's loss ignores it.

    python scripts/m1_decoder_water.py --wm ~/flock2_data/m1_agent/fix_s0/wm.pkl

Fresh meadows (``world.random_start``, seeds 30,000 + i) run the scripted flock; ``--record``
sheep per meadow have their posterior state filtered from their strips and felt numbers, as a
learning sheep's, and decoded back to a strip (the most likely colour per pixel). Water pixels
are palette 4 and 5 below the horizon (row >= ``fpv.HORIZON``; colour 5 is also the sky above
it). For frames with water in the true strip: the column IoU of water (true vs decoded), the
share of frames where the decoder draws any water, and the error in the water's azimuth (the
circular mean of its columns), against a baseline of the true columns rotated by a random angle.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from flock import ac, fpv, wm
from flock.meadow import drives as D, scripted, sim
from flock.meadow.views import view_one
from flock.meadow.world import ground_state, random_start

WATER = (4, 5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wm", required=True)
    p.add_argument("--meadows", type=int, default=6)
    p.add_argument("--sheep", type=int, default=40)
    p.add_argument("--record", type=int, default=8)
    p.add_argument("--warm", type=int, default=100)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    world = wm.WorldModel(nnx.Rngs(0), n_intero=D.N_FELT, reward=True)
    wm.load_into(world, pickle.load(open(os.path.expanduser(a.wm), "rb")))
    rec = jnp.arange(a.record)

    def one(w, st, mem, k):
        h, z, a_prev = mem
        k1, k2 = jax.random.split(k)
        tiles = ground_state(st.meadow)
        views = jax.vmap(lambda i: view_one(st, i, tiles))(jnp.arange(a.sheep))
        embed = w.encoder(jax.nn.one_hot(views, 16)) + w.intero_in(D.felt(st.flock.levels, st.flock.taste))
        h = w.transition(h, z, jax.nn.one_hot(a_prev, wm.N_ACTIONS))
        z = wm._sample(w.posterior(h, embed), k1)
        act = scripted.policy(st, k2)
        return sim.step(st, act), (h, z, act), (views[rec], ac.feat_of(h, z)[rec])

    @nnx.jit
    def go(w, key):
        states = jax.vmap(lambda k: random_start(k, a.sheep))(jnp.stack([jax.random.PRNGKey(30_000 + i) for i in range(a.meadows)]))
        mems = (jnp.zeros((a.meadows, a.sheep, wm.DETER)), jnp.zeros((a.meadows, a.sheep, wm.STOCH * wm.CLASSES)), jnp.zeros((a.meadows, a.sheep), jnp.int32))

        def body(c, k):
            st, mm = c
            st, mm, out = jax.vmap(lambda s, m, kk: one(w, s, m, kk))(st, mm, jax.random.split(k, a.meadows))
            return (st, mm), out

        _, (views, feats) = jax.lax.scan(body, (states, mems), jax.random.split(key, a.warm + a.steps))
        return views[a.warm :], feats[a.warm :]

    @nnx.jit
    def decode(w, feats):
        return jnp.argmax(w.decoder(feats), -1).astype(jnp.uint8)

    views, feats = go(world, jax.random.PRNGKey(3))
    views = np.asarray(views).reshape(-1, fpv.H, fpv.W)
    feats = np.asarray(feats).reshape(-1, feats.shape[-1])
    dec = np.concatenate([np.asarray(decode(world, feats[i : i + 256])) for i in range(0, len(feats), 256)])
    below = slice(fpv.HORIZON, fpv.H)
    tw = np.isin(views[:, below], WATER).any(1)  # (N, W) water columns
    dw = np.isin(dec[:, below], WATER).any(1)
    has = tw.any(1)
    tw, dw = tw[has], dw[has]
    iou = (tw & dw).sum(1) / np.maximum(1, (tw | dw).sum(1))
    rng = np.random.default_rng(0)
    shifted = np.stack([np.roll(t, rng.integers(fpv.W)) for t in tw])
    iou_base = (shifted & dw).sum(1) / np.maximum(1, (shifted | dw).sum(1))
    ang = 2 * np.pi * (np.arange(fpv.W) + 0.5) / fpv.W

    def az(m):
        return np.arctan2((np.sin(ang) * m).sum(1), (np.cos(ang) * m).sum(1))

    both = dw.any(1)
    err = np.abs((az(tw[both]) - az(dw[both]) + np.pi) % (2 * np.pi) - np.pi)
    res = {
        "wm": a.wm,
        "frames": int(len(views)),
        "frames_with_water": int(has.sum()),
        "decoder_draws_water": round(float(both.mean()), 3),
        "column_iou": round(float(iou.mean()), 3),
        "column_iou_random_rotation": round(float(iou_base.mean()), 3),
        "azimuth_error_deg_median": round(float(np.degrees(np.median(err))), 1) if both.any() else None,
        "azimuth_error_deg_random": 90.0,
        "true_water_columns_mean": round(float(tw.sum(1).mean()), 1),
        "decoded_water_columns_mean": round(float(dw.sum(1).mean()), 1),
    }
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
