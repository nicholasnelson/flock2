#!/usr/bin/env python
"""Linear probes of a world model's state (Milestone 1; second review of 2026-10-24, § 3): can
the things a sheep needs to act well be read from (h, z) at all?

    python scripts/m1_probe_latent.py --wm ~/flock2_data/m1_agent/fix_s0/wm.pkl --out runs/m1/probe_fix_s0.json

Fresh meadows (``world.random_start``, seeds 20,000 + i) are run with the scripted flock (which
visits water, grass and cover, so every target has positives); after a warm-up, each sheep's
posterior state is filtered from its own strips and felt numbers exactly as a learning sheep's
is, and paired with the privileged features of ``flock.meadow.privileged`` at the same step.
Ridge regressions from (h, z) are fitted on the sheep of the first 75 % of meadows and scored
on the rest (no meadow in both). Reported: for binary targets (near water, grass to bite
underfoot, near cover, night) the balanced accuracy at threshold 0.5 against 0.5 for chance; for
continuous ones (the five levels, the egocentric direction and distance to water, grass, cover)
R² on the held-out meadows. Direction and distance targets are also scored only on samples where
that target is within the daytime view range (12 tiles) and inside the 300° field of view
(``*_in_view``), since a sheep cannot know where water it cannot see is. ``--mlp`` replaces the
ridge regressions with a small nonlinear probe (896 -> 256 -> 256 -> 1, ReLU, Adam 1e-3, squared
error, 8 epochs), for information the state holds in a form a straight-line readout cannot use.
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
from flock.meadow import drives as D, privileged as P, scripted, sim
from flock.meadow.views import view_one
from flock.meadow.world import ground_state, random_start

# privileged.observe layout: felt 0-5, time 6-8, resting 9, water 10-12, grass 13-15, cover 16-18, flags 19-21
BINARY = {"near_water": 19, "grass_underfoot": 20, "near_cover": 21, "night": 8}
CONTINUOUS = {**{f"level_{n}": i for i, n in enumerate(D.NAMES)}, "water_fwd": 10, "water_left": 11, "water_dist": 12, "grass_fwd": 13, "grass_left": 14, "grass_dist": 15, "cover_fwd": 16, "cover_left": 17, "cover_dist": 18}


def run(world, meadows, sheep, warm, steps, seed):
    def one(st, mem, k):
        h, z, a_prev = mem
        k1, k2 = jax.random.split(k)
        tiles = ground_state(st.meadow)
        views = jax.vmap(lambda i: view_one(st, i, tiles))(jnp.arange(sheep))
        embed = world.encoder(jax.nn.one_hot(views, 16)) + world.intero_in(D.felt(st.flock.levels, st.flock.taste))
        h = world.transition(h, z, jax.nn.one_hot(a_prev, wm.N_ACTIONS))
        z = wm._sample(world.posterior(h, embed), k1)
        priv = P.observe(st)
        a = scripted.policy(st, k2)
        return sim.step(st, a), (h, z, a), (ac.feat_of(h, z), priv)

    @nnx.jit
    def go(w, states, mems, key):
        def body(c, k):
            st, mem = c
            st, mem, out = jax.vmap(one)(st, mem, jax.random.split(k, meadows))
            return (st, mem), out

        return jax.lax.scan(body, (states, mems), jax.random.split(key, warm + steps))[1]

    states = jax.vmap(lambda k: random_start(k, sheep))(jnp.stack([jax.random.PRNGKey(20_000 + i) for i in range(meadows)]))
    mems = (jnp.zeros((meadows, sheep, wm.DETER)), jnp.zeros((meadows, sheep, wm.STOCH * wm.CLASSES)), jnp.zeros((meadows, sheep), jnp.int32))
    feat, priv = go(world, states, mems, jax.random.PRNGKey(seed))
    return np.asarray(feat[warm:]), np.asarray(priv[warm:])  # (T, B, S, ...)


def ridge(x, y, lam=1.0):
    mu, sd = x.mean(0), x.std(0) + 1e-6
    xs = np.concatenate([(x - mu) / sd, np.ones((x.shape[0], 1))], 1)
    w = np.linalg.solve(xs.T @ xs + lam * np.eye(xs.shape[1]), xs.T @ y)
    return lambda q: np.concatenate([(q - mu) / sd, np.ones((q.shape[0], 1))], 1) @ w


def mlp_probe(x, y, epochs=8, seed=0):
    import optax

    mu, sd = x.mean(0), x.std(0) + 1e-6
    ym, ys = y.mean(), y.std() + 1e-6
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
    init = lambda k, i, o: (jax.random.normal(k, (i, o)) * np.sqrt(2.0 / i), jnp.zeros(o))
    params = [init(k1, x.shape[1], 256), init(k2, 256, 256), init(k3, 256, 1)]

    def f(ps, q):
        for w, b in ps[:-1]:
            q = jax.nn.relu(q @ w + b)
        return (q @ ps[-1][0] + ps[-1][1])[:, 0]

    opt = optax.adam(1e-3)
    st = opt.init(params)

    @jax.jit
    def step(ps, st, xb, yb):
        l, g = jax.value_and_grad(lambda q: jnp.mean((f(q, xb) - yb) ** 2))(ps)
        u, st = opt.update(g, st)
        return optax.apply_updates(ps, u), st

    xs, yn = ((x - mu) / sd).astype(np.float32), ((y - ym) / ys).astype(np.float32)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        for mb in np.array_split(rng.permutation(len(xs)), max(1, len(xs) // 512)):
            params, st = step(params, st, xs[mb], yn[mb])
    fj = jax.jit(f)
    return lambda q: np.asarray(fj(params, ((q - mu) / sd).astype(np.float32))) * ys + ym


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wm", required=True)
    p.add_argument("--meadows", type=int, default=8)
    p.add_argument("--sheep", type=int, default=40)
    p.add_argument("--warm", type=int, default=200)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mlp", action="store_true", help="nonlinear probes instead of ridge regressions")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    world = wm.WorldModel(nnx.Rngs(0), n_intero=D.N_FELT, reward=True)
    wm.load_into(world, pickle.load(open(os.path.expanduser(a.wm), "rb")))
    feat, priv = run(world, a.meadows, a.sheep, a.warm, a.steps, a.seed)
    split = int(a.meadows * 0.75)
    tr = lambda x: x[:, :split].reshape((-1,) + x.shape[3:])
    te = lambda x: x[:, split:].reshape((-1,) + x.shape[3:])
    xtr, xte = tr(feat), te(feat)
    fit = mlp_probe if a.mlp else ridge
    res = {"wm": a.wm, "probe": "mlp" if a.mlp else "ridge", "train_samples": int(xtr.shape[0]), "test_samples": int(xte.shape[0])}
    for name, i in BINARY.items():
        ytr, yte = tr(priv)[:, i] > 0.5, te(priv)[:, i] > 0.5
        if ytr.all() or not ytr.any() or yte.all() or not yte.any():
            res[name] = {"balanced_acc": None, "positive_rate": float(yte.mean())}
            continue
        pred = fit(xtr, ytr.astype(np.float64))(xte) > 0.5
        bal = 0.5 * ((pred & yte).sum() / yte.sum() + (~pred & ~yte).sum() / (~yte).sum())
        res[name] = {"balanced_acc": round(float(bal), 3), "positive_rate": round(float(yte.mean()), 3)}
    def r2_of(mtr, mte, i):
        ytr, yte = tr(priv)[mtr, i], te(priv)[mte, i]
        if len(ytr) < 50 or len(yte) < 20:
            return None
        pred = fit(xtr[mtr], ytr)(xte[mte])
        return round(float(1 - ((pred - yte) ** 2).sum() / max(1e-9, ((yte - yte.mean()) ** 2).sum())), 3)

    every_tr, every_te = np.ones(len(xtr), bool), np.ones(len(xte), bool)
    for name, i in CONTINUOUS.items():
        res[name] = {"r2": r2_of(every_tr, every_te, i)}
    # in view: distance (scaled by 1/32) within 12 tiles, and the bearing inside the 300 degree field
    for target, base in (("water", 10), ("grass", 13), ("cover", 16)):
        def in_view(x):
            fwd, left, dist = x[:, base], x[:, base + 1], x[:, base + 2]
            return (dist < 12 / 32) & (np.abs(np.arctan2(left, fwd)) < np.deg2rad(150))
        mtr, mte = in_view(tr(priv)), in_view(te(priv))
        res[f"{target}_in_view_share"] = round(float(mte.mean()), 3)
        for k, off in (("fwd", 0), ("left", 1), ("dist", 2)):
            res[f"{target}_{k}_in_view"] = {"r2": r2_of(mtr, mte, base + off)}
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
