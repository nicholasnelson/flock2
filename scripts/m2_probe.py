#!/usr/bin/env python
"""What does the meadow v2 hybrid's world-model state hold? Probes from (h, z) to the true
situation (``flock.meadow2.privileged``), evaluation only.

    python scripts/m2_probe.py --agent ~/flock2_data/m2_hybrid/s0 --out runs/meadow2/probes/s0.json
    python scripts/m2_probe.py --untrained --out runs/meadow2/probes/untrained.json   # the floor

``--meadows`` fresh meadows (seeds 70,000 + i; neither training nor evaluation meadows) are run with
the scripted flock, which visits water, grass and cover, so every target has examples. Each sheep's
posterior state is filtered from its own view and felt numbers exactly as a learning sheep's is.
After ``--warm`` steps, its state is paired with the true features at every step for ``--steps``
steps.

Probes are fitted on the sheep of the first 75 % of meadows and scored on the rest (no meadow in
both): ridge regressions, and small MLPs (896 -> 256 -> 256 -> 1, as ``m1_probe_latent.py``), with
inputs standardised by a floored spread (``_stats``). Scores:
balanced accuracy for flags, R^2 for numbers. Direction and distance targets (forward, left,
distance, for water, good grass and cover) are scored on all samples, on samples where the target is
**in view** (inside the 300-degree field and within ``IN_VIEW`` tiles), and **out of view** (behind
the sheep or farther), which only memory can answer.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from flock import ac, wm
from flock.meadow2 import drives as D, privileged as P, scripted, sim
from flock.meadow2.world import random_start


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(__file__), f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


IN_VIEW = 16.0  # tiles
SD_FLOOR = 0.05  # inputs are standardised by max(sd, this); see _stats
BINARY = {"mouth_on_water": D.N_FELT + 4 + 9, "mouth_biteable": D.N_FELT + 4 + 10, "near_cover": D.N_FELT + 4 + 11}
LEVELS = {f"felt_{n}": i for i, n in enumerate(D.NAMES + ("fat",))}
TARGETS = {"water": D.N_FELT + 4, "grass": D.N_FELT + 7, "cover": D.N_FELT + 10}


def _stats(x):
    """Mean and spread for standardising probe inputs. The spread is floored: a latent unit that
    never varies in the training meadows but does in the test meadows would otherwise be divided by
    ~1e-6, and its predictions explode. ``m1_probe_latent.py`` divides by sd + 1e-6, so its R^2 values
    can be distorted by this (noticed 2026-10-26; that script is left as it was, for reproducibility)."""
    return x.mean(0), np.maximum(x.std(0), SD_FLOOR)


def ridge(x, y, lam=1.0):
    mu, sd = _stats(x)
    xs = np.concatenate([(x - mu) / sd, np.ones((x.shape[0], 1))], 1)
    w = np.linalg.solve(xs.T @ xs + lam * np.eye(xs.shape[1]), xs.T @ y)
    return lambda q: np.concatenate([(q - mu) / sd, np.ones((q.shape[0], 1))], 1) @ w


def mlp_probe(x, y, epochs=8, seed=0):
    """896 -> 256 -> 256 -> 1, ReLU, Adam 1e-3, squared error, as ``m1_probe_latent.mlp_probe``,
    with floored standardisation."""
    import optax

    mu, sd = _stats(x)
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


def collect(world, hy, n_meadows, warm, steps, seed):
    states = jax.vmap(random_start)(jnp.stack([jax.random.PRNGKey(70_000 + i) for i in range(n_meadows)]))
    S = states.flock.pos.shape[1]
    mems = (jnp.zeros((n_meadows, S, hy.CFG.deter)), jnp.zeros((n_meadows, S, hy.CFG.stoch * hy.CFG.classes)), jnp.zeros((n_meadows, S), jnp.int32))

    def body(carry, k):
        st, (h, z, a_prev) = carry
        k1, k2 = jax.random.split(k)
        views = hy.to_bytes(hy.R.render_meadows(st))
        f = st.flock
        embed = world.embed_obs(views, D.felt(f.levels, f.fat, f.taste))
        h = world.transition(h, z, jax.nn.one_hot(a_prev, hy.CFG.n_actions))
        z = wm._sample(world.posterior(h, embed), k1)
        priv = jax.vmap(P.observe)(st)
        a = jax.vmap(scripted.policy)(st, jax.random.split(k2, n_meadows))
        st2 = jax.vmap(sim.step)(st, a)
        return (st2, (h, z, a)), (ac.feat_of(h, z), priv, f.alive)

    run = jax.jit(lambda s, m, k: jax.lax.scan(body, (s, m), jax.random.split(k, warm + steps))[1])
    feat, priv, alive = run(states, mems, jax.random.PRNGKey(seed))
    return np.asarray(feat[warm:]), np.asarray(priv[warm:]), np.asarray(alive[warm:])  # (T, B, S, ...)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default=None)
    p.add_argument("--untrained", action="store_true")
    p.add_argument("--meadows", type=int, default=32)
    p.add_argument("--warm", type=int, default=150)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    hy = _load("m2_hybrid")
    world = hy.make_world(0)
    if not a.untrained:
        wm.load_into(world, pickle.load(open(os.path.join(os.path.expanduser(a.agent), "wm.pkl"), "rb")))
    feat, priv, alive = collect(world, hy, a.meadows, a.warm, a.steps, a.seed)
    split = int(a.meadows * 0.75)
    tr = lambda x: x[:, :split].reshape((-1,) + x.shape[3:])
    te = lambda x: x[:, split:].reshape((-1,) + x.shape[3:])
    ok_tr, ok_te = tr(alive), te(alive)
    xtr, xte, ptr, pte = tr(feat)[ok_tr], te(feat)[ok_te], tr(priv)[ok_tr], te(priv)[ok_te]
    res = {"agent": "untrained" if a.untrained else a.agent, "train_samples": int(len(xtr)), "test_samples": int(len(xte))}

    def r2(fit, mtr, mte, i):
        if mtr.sum() < 200 or mte.sum() < 50:
            return None
        pred = fit(xtr[mtr], ptr[mtr, i])(xte[mte])
        y = pte[mte, i]
        return round(float(1 - ((pred - y) ** 2).sum() / max(1e-9, ((y - y.mean()) ** 2).sum())), 3)

    for name_fit, fit in (("ridge", ridge), ("mlp", mlp_probe)):
        out = {}
        for name, i in BINARY.items():
            ytr, yte = ptr[:, i] > 0.5, pte[:, i] > 0.5
            if ytr.all() or not ytr.any() or yte.all() or not yte.any():
                out[name] = None
                continue
            pred = fit(xtr, ytr.astype(np.float64))(xte) > 0.5
            out[name] = {"balanced_acc": round(float(0.5 * ((pred & yte).sum() / yte.sum() + (~pred & ~yte).sum() / (~yte).sum())), 3), "positive_rate": round(float(yte.mean()), 3)}
        all_tr, all_te = np.ones(len(xtr), bool), np.ones(len(xte), bool)
        for name, i in LEVELS.items():
            out[name] = r2(fit, all_tr, all_te, i)
        for target, base in TARGETS.items():
            def in_view(x):
                fwd, left, dist = x[:, base], x[:, base + 1], x[:, base + 2] * 32.0
                return (dist <= IN_VIEW) & (np.abs(np.arctan2(left, fwd)) < np.deg2rad(150))
            vtr, vte = in_view(ptr), in_view(pte)
            out[f"{target}_in_view_share"] = round(float(vte.mean()), 3)
            for k, off in (("fwd", 0), ("left", 1), ("dist", 2)):
                out[f"{target}_{k}"] = {"all": r2(fit, all_tr, all_te, base + off), "in_view": r2(fit, vtr, vte, base + off), "out_of_view": r2(fit, ~vtr, ~vte, base + off)}
        res[name_fit] = out
        print(name_fit, json.dumps({k: out[k] for k in out if k.startswith("water")}), flush=True)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
