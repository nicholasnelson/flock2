#!/usr/bin/env python
"""The third review's S4 probe: does a grid sheep's world-model state remember where the pond is
when the pond is out of its 8x8 view?

    python scripts/grid_probe.py --agent ~/flock2_data/grid/plain_s0

``--worlds`` fixed worlds (seeds 80,000 + i) are walked by a random policy for ``--steps``
steps; the world model filters each sheep's state as a learning sheep's. At every step after
``--warm``, the target is the pond's centre relative to the sheep (rows, cols; also the
Manhattan distance), and whether the pond is in view. Ridge probes from (h, z) are fitted on the
sheep of the first 75 % of worlds and scored (R²) on the rest, on all steps and on the steps with
the pond out of view (where only memory can know it).
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

from flock import ac, wm
from flock.grid import brain, world as Wd


def ridge(x, y, lam=1.0):
    mu, sd = x.mean(0), x.std(0) + 1e-6
    xs = np.concatenate([(x - mu) / sd, np.ones((x.shape[0], 1))], 1)
    w = np.linalg.solve(xs.T @ xs + lam * np.eye(xs.shape[1]), xs.T @ y)
    return lambda q: np.concatenate([(q - mu) / sd, np.ones((q.shape[0], 1))], 1) @ w


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", required=True)
    p.add_argument("--worlds", type=int, default=128)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--warm", type=int, default=30)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    d = os.path.expanduser(a.agent)
    cfgj = json.load(open(os.path.join(d, "config.json")))
    view = cfgj.get("view", "local")
    cfg = brain.config(view)
    world = wm.WorldModel(nnx.Rngs(0), n_intero=Wd.N_FELT, reward=True, ensemble=cfgj.get("ensemble", 0), cfg=cfg)
    wm.load_into(world, pickle.load(open(os.path.join(d, "wm.pkl"), "rb")))

    def one(w, s, mem, k):
        h, z, a_prev = mem
        k1, k2 = jax.random.split(k)
        v = Wd.observe(s, view)
        embed = w.encoder(jax.nn.one_hot(v, cfg.n_classes)) + w.intero_in(Wd.felt(s))
        h = w.transition(h, z, jax.nn.one_hot(a_prev, Wd.N_ACTIONS))
        z = wm._sample(w.posterior(h, embed), k1)
        act = jax.random.randint(k2, (), 0, Wd.N_ACTIONS)
        rr, cc = jnp.nonzero(s.water, size=9, fill_value=-1)
        n = jnp.maximum((rr >= 0).sum(), 1)
        centre = jnp.stack([jnp.where(rr >= 0, rr, 0).sum() / n, jnp.where(cc >= 0, cc, 0).sum() / n])
        rel = centre - s.pos
        in_view = (Wd.view(s) == Wd.WATER).any()
        return Wd.step(s, act), (h, z, act), (ac.feat_of(h, z), rel, in_view)

    @nnx.jit
    def go(w, key):
        states = jax.vmap(lambda k: Wd.generate(k))(jnp.stack([jax.random.PRNGKey(80_000 + i) for i in range(a.worlds)]))
        mems = (jnp.zeros((a.worlds, cfg.deter)), jnp.zeros((a.worlds, cfg.stoch * cfg.classes)), jnp.zeros(a.worlds, jnp.int32))

        def body(c, k):
            st, mm = c
            st, mm, out = jax.vmap(lambda s, m, kk: one(w, s, m, kk))(st, mm, jax.random.split(k, a.worlds))
            return (st, mm), out

        return jax.lax.scan(body, (states, mems), jax.random.split(key, a.steps))[1]

    feat, rel, in_view = (np.asarray(x)[a.warm :] for x in go(world, jax.random.PRNGKey(2)))  # (T, B, ...)
    split = int(a.worlds * 0.75)
    tr = lambda x: x[:, :split].reshape((-1,) + x.shape[2:])
    te = lambda x: x[:, split:].reshape((-1,) + x.shape[2:])
    res = {"agent": a.agent, "view": view, "pond_in_view_share": round(float(in_view.mean()), 3)}
    targets = {"pond_drow": rel[..., 0], "pond_dcol": rel[..., 1], "pond_dist": np.abs(rel).sum(-1)}
    for name, y in targets.items():
        for subset, mtr, mte in (("all", np.ones(tr(in_view).shape, bool), np.ones(te(in_view).shape, bool)), ("out_of_view", ~tr(in_view), ~te(in_view))):
            f = ridge(tr(feat)[mtr], tr(y)[mtr])
            yt = te(y)[mte]
            pred = f(te(feat)[mte])
            res[f"{name}_{subset}"] = round(float(1 - ((pred - yt) ** 2).sum() / max(1e-9, ((yt - yt.mean()) ** 2).sum())), 3)
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
