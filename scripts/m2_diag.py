#!/usr/bin/env python
"""The diagnosis gate's measurements (spec § 15): D1, water reconstruction, and D2/D3, a better probe.
Evaluation only; nothing here trains the sheep.

    python scripts/m2_diag.py --agent ~/flock2_data/m2_hybrid/s0 --out runs/meadow2/diag/s0.json

``--meadows`` fresh meadows (seeds 70,000 + i) are run with the scripted flock. Each sheep's posterior
state is filtered from its own view and felt numbers, as a learning sheep's is. After ``--warm``
steps, ``--steps`` steps are recorded.

**D1** (every ``--decode_every``-th step, every sheep): the view is decoded from the posterior (h, z).
A pixel that truly shows water (``render.render_aux``'s mask) counts as **redrawn** if the decoded
colour is nearer its true colour than the colour that pixel shows when the meadow is rendered again
with its water removed. Recall is reported by the pixel's distance (< 4, 4-8, 8-16, >= 16 tiles) and by
day (light >= 0.5) or night.

**D2/D3** (every step, every living sheep): targets come from the renderer and the true state.
- *Visible water*: frames with at least ``MIN_PX`` water pixels. Targets are the bearing of the water's
  centre in the image (degrees, from its mean column) and its distance (the nearest water pixel's
  depth). Results are binned by that distance; frames with water within 1.5 tiles are left out.
- *Just out of view*: no water pixels now, but some within the last ``RECENT`` steps. Targets are the
  sine and cosine of the true egocentric bearing to the nearest water tile.

Probed inputs: the encoder's embedding, h, z, and (h, z). Probes are fitted on the first 75 % of
meadows and scored on the rest. Ridge (standardised by a floored spread) is the headline; an MLP
with weight decay, early-stopped on held-out training meadows, is reported beside it.

**Stage 1 (spec § 20).** A reactive agent (``config.json`` kind ``m2_reactive``) is probed on its
policy's embedding alone (no D1, no D3). **D2_true**, for every agent: the sine and cosine of the true
bearing of the nearest water tile inside the field of view, binned by its true distance, on frames
where there is one within 32 tiles, whether or not water pixels show. This is the measure of "can the
eyes read where water is at this distance", which with willows should hold where water pixels do not
(the willows mark it). The meadows carry the agent's willows.
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax

from flock import wm
from flock.meadow2 import drives as D, privileged as P, render as R, scripted, sim
from flock.meadow2.world import random_start

MIN_PX = 5
RECENT = 30
NEAR = 1.5
BINS = ((0, 4), (4, 8), (8, 16), (16, 1e9))
SD_FLOOR = 0.05


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(__file__), f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def render_all(states, fn):
    B, S = states.flock.pos.shape[:2]
    idx = (jnp.repeat(jnp.arange(B), S), jnp.tile(jnp.arange(S), B))
    out = jax.lax.map(lambda mi: fn(jax.tree.map(lambda a: a[mi[0]], states), mi[1]), idx)
    return jax.tree.map(lambda x: x.reshape((B, S) + x.shape[1:]), out)


FAR = 32.0


def true_water(st):
    """(bearing, distance, found) of the nearest water tile inside the field of view within FAR tiles."""
    vec, dist, found = P.visible_water(st, FAR)
    return P.bearing(vec, st.flock.heading), dist, found


def _targets(st, water, depth):
    """The probe targets shared by every agent, from the renderer and the true state."""
    f = st.flock
    priv = jax.vmap(P.observe)(st)
    n_px = water.sum((-2, -1))
    cols = jnp.arange(R.W)[None, None, None, :]
    col_c = jnp.sum(jnp.where(water, cols, 0), (-2, -1)) / jnp.maximum(n_px, 1)
    bearing = ((col_c + 0.5) / R.W - 0.5) * 300.0
    near_px = jnp.min(jnp.where(water, depth, jnp.inf), (-2, -1))
    light = sim.light(st.t)
    tb, td, tf = jax.vmap(true_water)(st)
    return dict(n_px=n_px, bearing=bearing, near_px=near_px,
                wbear=jnp.arctan2(priv[..., D.N_FELT + 5], priv[..., D.N_FELT + 4]), wdist=priv[..., D.N_FELT + 6] * 32.0, alive=f.alive,
                light=jnp.broadcast_to(light[:, None], n_px.shape), heading=f.heading, wbear_v=tb, wdist_v=td, found_v=tf)


def _starts(n_meadows, willows):
    return jax.vmap(functools.partial(random_start, willows=willows))(jnp.stack([jax.random.PRNGKey(70_000 + i) for i in range(n_meadows)]))


def collect_reactive(policy, ra, n_meadows, warm, steps, seed, willows=0):
    """The reactive sheep's embedding (stage 1) under the scripted flock's behaviour."""
    states = _starts(n_meadows, willows)

    def body(st, xs):
        k, t = xs
        rgb, water, depth = render_all(st, R.render_aux)
        f = st.flock
        embed = policy.embed(ra.to_bytes(rgb), D.felt(f.levels, f.fat, f.taste))
        a = jax.vmap(scripted.policy)(st, jax.random.split(k, n_meadows))
        st2 = jax.vmap(sim.step)(st, a)
        return st2, dict(embed=embed.astype(jnp.float16), **_targets(st, water, depth))

    T = warm + steps
    run = jax.jit(lambda s, k: jax.lax.scan(body, s, (jax.random.split(k, T), jnp.arange(T)))[1])
    out = run(states, jax.random.PRNGKey(seed))
    return {k: np.asarray(v[warm:]) for k, v in out.items()}


def collect(world, hy, n_meadows, warm, steps, decode_every, seed, willows=0):
    states = _starts(n_meadows, willows)
    S = states.flock.pos.shape[1]
    mems = hy.zero_mems(world, n_meadows, S)

    def body(carry, xs):
        st, (h, z, a_prev) = carry
        k, t = xs
        k1, k2 = jax.random.split(k)
        rgb, water, depth = render_all(st, R.render_aux)
        views = hy.to_bytes(rgb)
        f = st.flock
        embed = world.embed_obs(views, D.felt(f.levels, f.fat, f.taste))
        h = world.transition(h, z, jax.nn.one_hot(a_prev, world.cfg.n_actions))
        z = wm._sample(world.posterior(h, embed), k1)
        def d1(_):
            dec = world.decode(wm.Step(h, z, None, None)) + 0.5  # (B, S, H, W, 3)
            dry = st._replace(meadow=st.meadow._replace(water=jnp.zeros_like(st.meadow.water)))
            alt = render_all(dry, R.render)
            hit = jnp.linalg.norm(dec - rgb, axis=-1) < jnp.linalg.norm(dec - alt, axis=-1)
            counts = []
            for lo, hi in BINS:
                m = water & (depth >= lo) & (depth < hi)
                counts.append(jnp.stack([(m & hit).sum((1, 2, 3)), m.sum((1, 2, 3))], -1))
            return jnp.stack(counts, 1)  # (B, bins, 2): per meadow, since each has its own time of day

        rec_d1 = jax.lax.cond(t % decode_every == 0, d1, lambda _: jnp.zeros((n_meadows, len(BINS), 2), jnp.int32), None)
        a = jax.vmap(scripted.policy)(st, jax.random.split(k2, n_meadows))
        st2 = jax.vmap(sim.step)(st, a)
        out = dict(embed=embed.astype(jnp.float16), h=h.astype(jnp.float16), z=z.astype(jnp.float16), d1=rec_d1, **_targets(st, water, depth))
        return (st2, (h, z, a)), out

    T = warm + steps
    run = jax.jit(lambda s, m, k: jax.lax.scan(body, (s, m), (jax.random.split(k, T), jnp.arange(T)))[1])
    out = run(states, mems, jax.random.PRNGKey(seed))
    return {k: np.asarray(v[warm:]) for k, v in out.items()}


def _stats(x):
    return x.mean(0), np.maximum(x.std(0), SD_FLOOR)


def ridge(xtr, ytr, xte, lam=1.0):
    mu, sd = _stats(xtr)
    a = np.concatenate([(xtr - mu) / sd, np.ones((len(xtr), 1))], 1)
    w = np.linalg.solve(a.T @ a + lam * np.eye(a.shape[1]), a.T @ ytr)
    return np.concatenate([(xte - mu) / sd, np.ones((len(xte), 1))], 1) @ w


def mlp_es(xtr, ytr, xte, xval, yval, epochs=20, seed=0):
    """896-ish -> 256 -> 256 -> 1, AdamW (weight decay 1e-4), early-stopped on validation loss."""
    mu, sd = _stats(xtr)
    ym, ys = ytr.mean(), ytr.std() + 1e-6
    n = lambda x: ((x - mu) / sd).astype(np.float32)
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
    init = lambda k, i, o: (jax.random.normal(k, (i, o)) * np.sqrt(2.0 / i), jnp.zeros(o))
    params = [init(k1, xtr.shape[1], 256), init(k2, 256, 256), init(k3, 256, 1)]

    def f(ps, q):
        for w, b in ps[:-1]:
            q = jax.nn.relu(q @ w + b)
        return (q @ ps[-1][0] + ps[-1][1])[:, 0]

    opt = optax.adamw(1e-3, weight_decay=1e-4)
    st = opt.init(params)

    @jax.jit
    def step(ps, st, xb, yb):
        g = jax.grad(lambda q: jnp.mean((f(q, xb) - yb) ** 2))(ps)
        u, st = opt.update(g, st, ps)
        return optax.apply_updates(ps, u), st

    fj = jax.jit(f)
    xs, yn, xv, yv = n(xtr), ((ytr - ym) / ys).astype(np.float32), n(xval), ((yval - ym) / ys).astype(np.float32)
    rng = np.random.default_rng(seed)
    best, best_ps = np.inf, params
    for _ in range(epochs):
        for mb in np.array_split(rng.permutation(len(xs)), max(1, len(xs) // 512)):
            params, st = step(params, st, xs[mb], yn[mb])
        v = float(np.mean((np.asarray(fj(params, xv)) - yv) ** 2))
        if v < best:
            best, best_ps = v, params
    return np.asarray(fj(best_ps, n(xte))) * ys + ym


def r2(pred, y):
    return round(float(1 - ((pred - y) ** 2).sum() / max(1e-9, ((y - y.mean()) ** 2).sum())), 3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", required=True)
    p.add_argument("--meadows", type=int, default=32)
    p.add_argument("--warm", type=int, default=150)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--decode_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_mlp", action="store_true")
    p.add_argument("--validate_d3", action="store_true", help="only D3, on the model state and on inputs of known content (spec § 17)")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    d = os.path.expanduser(a.agent)
    cfg = json.load(open(os.path.join(d, "config.json")))
    willows = int(cfg.get("willows", 0))
    reactive = cfg.get("kind") == "m2_reactive"
    world = None
    if reactive:
        ra = _load("m2_reactive")
        policy = ra.make_policy(0, cfg.get("hidden", 256))
        wm.load_into(policy, pickle.load(open(os.path.join(d, "policy.pkl"), "rb")))
        rec = collect_reactive(policy, ra, a.meadows, a.warm, a.steps, a.seed, willows)
    else:
        hy = _load("m2_hybrid")
        world = hy.make_world(0, cfg.get("model", "vector"))
        wm.load_into(world, pickle.load(open(os.path.join(d, "wm.pkl"), "rb")))
        rec = collect(world, hy, a.meadows, a.warm, a.steps, 10**9 if a.validate_d3 else a.decode_every, a.seed, willows)
    res = {"agent": a.agent, "meadows": a.meadows, "steps": a.steps, "willows": willows}

    day = rec["light"][:, :, 0] >= 0.5  # (T, B), each meadow by its own light
    if not reactive:
        # D1: recall of water pixels by distance and day/night (light of the decoded step)
        d1 = rec["d1"]  # (T, B, bins, 2)
        res["D1"] = {}
        for name, m in (("day", day), ("night", ~day), ("all", np.ones_like(day))):
            s = d1[m].sum(0)
            res["D1"][name] = {f"{lo}-{hi if hi < 1e8 else 'inf'}": {"recall": round(float(s[i, 0] / max(1, s[i, 1])), 3), "water_px": int(s[i, 1])} for i, (lo, hi) in enumerate(BINS)}
        s = d1[day].sum(0)
        res["D1"]["day_within_8_recall"] = round(float((s[0, 0] + s[1, 0]) / max(1, s[0, 1] + s[1, 1])), 3)

    # D2/D3 probes
    T, B, S = rec["n_px"].shape
    split, val = int(B * 0.75), int(B * 0.6)
    meadow = np.broadcast_to(np.arange(B)[None, :, None], (T, B, S))
    alive = rec["alive"]
    vis = (rec["n_px"] >= MIN_PX) & alive & (rec["wdist"] > NEAR)
    seen = rec["n_px"] >= MIN_PX
    last_seen = np.full((B, S), -10_000)
    recent = np.zeros_like(seen)
    for t in range(T):
        recent[t] = (t - last_seen <= RECENT) & ~seen[t]
        last_seen = np.where(seen[t], t, last_seen)
    recent &= alive & (rec["wdist"] > NEAR)
    inputs = {"embed": rec["embed"]}
    if not reactive:
        inputs.update({"h": rec["h"], "z": rec["z"], "hz": np.concatenate([rec["h"], rec["z"]], -1)})
    if world is not None and world.cfg.rot_bins:  # rung 4: the ring and the GRU state, probed apart as well
        inputs["ring"], inputs["gru"] = rec["h"][..., world.cfg.deter :], rec["h"][..., : world.cfg.deter]
    day_full = rec["light"] >= 0.5

    def probe(mask, target, label):
        out = {}
        for nm, X in inputs.items():
            tr = mask & (meadow < split)
            te = mask & (meadow >= split)
            fit = mask & (meadow < val)
            vv = mask & (meadow >= val) & (meadow < split)
            if tr.sum() < 200 or te.sum() < 50:
                out[nm] = None
                continue
            Xtr, ytr, Xte, yte = X[tr].astype(np.float32), target[tr], X[te].astype(np.float32), target[te]
            row = {"ridge": r2(ridge(Xtr, ytr, Xte), yte), "n_train": int(tr.sum()), "n_test": int(te.sum())}
            if not a.no_mlp:
                row["mlp_es"] = r2(mlp_es(X[fit].astype(np.float32), target[fit], Xte, X[vv].astype(np.float32), target[vv]), yte)
            out[nm] = row
        print(label, json.dumps(out), flush=True)
        return out

    if a.validate_d3:
        # inputs of known content on the D3 frames: the oracle (the target itself), an ideal memory
        # (the true bearing when water was last seen, turned by the heading change since; left-positive,
        # as the target), its raw parts, and noise of the same width
        last_b = np.zeros((B, S))
        last_h = np.zeros((B, S))
        last_t = np.full((B, S), -10_000)
        ideal = np.zeros((T, B, S, 2))
        parts = np.zeros((T, B, S, 5))
        for t in range(T):
            dpsi = (rec["heading"][t] - last_h + np.pi) % (2 * np.pi) - np.pi
            est = last_b + dpsi
            ideal[t] = np.stack([np.sin(est), np.cos(est)], -1)
            parts[t] = np.stack([np.sin(last_b), np.cos(last_b), np.sin(dpsi), np.cos(dpsi), np.clip((t - last_t) / RECENT, 0, 10)], -1)
            last_b = np.where(seen[t], rec["wbear"][t], last_b)
            last_h = np.where(seen[t], rec["heading"][t], last_h)
            last_t = np.where(seen[t], t, last_t)
        oracle = np.stack([np.sin(rec["wbear"]), np.cos(rec["wbear"])], -1)
        noise = np.random.default_rng(0).normal(size=(T, B, S, 5))
        inputs = {"oracle": oracle, "ideal_memory": ideal, "memory_parts": parts, "noise": noise, "hz": inputs["hz"]}
        res["D3_validation"] = {"frames": int(recent.sum()),
                                "sin_bearing": probe(recent, np.sin(rec["wbear"]), "validate sin"), "cos_bearing": probe(recent, np.cos(rec["wbear"]), "validate cos")}
        if a.out:
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            json.dump(res, open(a.out, "w"), indent=1)
        return
    res["D2_visible"] = {}
    for lo, hi in BINS:
        m = vis & (rec["near_px"] >= lo) & (rec["near_px"] < hi)
        key = f"{lo}-{hi if hi < 1e8 else 'inf'}"
        res["D2_visible"][key] = {"frames": int(m.sum()), "bearing": probe(m, rec["bearing"], f"bearing {key}"), "distance": probe(m, rec["near_px"], f"distance {key}")}
    m8 = vis & (rec["near_px"] < 8) & day_full
    res["D2_visible"]["day_within_8"] = {"frames": int(m8.sum()), "bearing": probe(m8, rec["bearing"], "bearing day<8")}
    # D2_true: the true bearing of the nearest water tile in the field of view, by its true distance,
    # by day and by night, whether or not water pixels show (stage 1: the willows should carry it)
    res["D2_true"] = {}
    for dn, dm in (("day", day_full), ("night", ~day_full)):
        for lo, hi in BINS[:-1]:
            m = alive & rec["found_v"] & (rec["wdist_v"] >= max(lo, NEAR)) & (rec["wdist_v"] < hi) & dm
            key = f"{dn}_{lo}-{hi}"
            res["D2_true"][key] = {"frames": int(m.sum()), "sin": probe(m, np.sin(rec["wbear_v"]), f"true sin {key}"), "cos": probe(m, np.cos(rec["wbear_v"]), f"true cos {key}")}
    if not reactive:
        if world.cfg.rot_bins:  # spec § 18: noise as wide as h and z, as the D3 floor
            inputs = {**inputs, "noise_wide": np.random.default_rng(0).normal(size=inputs["hz"].shape).astype(np.float16)}
        res["D3_recent"] = {"frames": int(recent.sum()),
                            "sin_bearing": probe(recent, np.sin(rec["wbear"]), "recent sin"), "cos_bearing": probe(recent, np.cos(rec["wbear"]), "recent cos")}
    print(json.dumps({"D1": res.get("D1"), "D2_true": res["D2_true"]}, indent=1))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
