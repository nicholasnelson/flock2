#!/usr/bin/env python
"""Probe what a world model believes an action does, against what it really does (Milestone 1).

    python scripts/m1_wm_probe.py --wm ~/flock2_data/m1_wm/run4_negdrive/model.pkl

Fresh meadows are run with the scripted flock (warm-up, then ``CONTEXT`` recorded steps for every
sheep). Each sheep's model state is inferred from its last ``CONTEXT`` strips and levels; then,
for each probed action, the model imagines ``K`` steps of repeating it, and the simulator really
does it from the same state. Reported per action, split by whether the sheep stands by water
(``sim.near_water``): the predicted and true change in hydration and energy over ``K`` steps and
the predicted and true mean reward.

The question it answers: does the model think drinking raises hydration away from water (where it
does nothing)? The scripted flock only ever drinks at the water's edge, so a model pretrained on
it never sees the contrary.
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

from flock import wm
from flock.meadow import drives as D, scripted, sim, views
from flock.meadow.world import generate, ground_state

CONTEXT, K = 16, 8
PROBED = ("drink", "graze", "stop", "walk", "left", "head_up", "rest")


def warm_and_record(key, meadows, sheep, warm):
    """Scripted flock: warm up, then record CONTEXT steps for every sheep."""

    def one_meadow(k):
        k1, k2, k3, k4 = jax.random.split(k, 4)
        st = generate(k1, sheep)
        st = st._replace(t=jax.random.randint(k2, (), 0, int(sim.DAY * sim.DAY_FRACTION) - warm - CONTEXT - K))  # daylight
        st = jax.lax.scan(lambda s, kk: (sim.step(s, scripted.policy(s, kk)), None), st, jax.random.split(k3, warm))[0]

        def rec(s, kk):
            v = jax.vmap(lambda i: views.view_one(s, i, ground_state(s.meadow)))(jnp.arange(sheep))
            a = scripted.policy(s, kk)
            return sim.step(s, a), (v.astype(jnp.uint8), a, D.felt(s.flock.levels, s.flock.taste))

        st, (v, a, lv) = jax.lax.scan(rec, st, jax.random.split(k4, CONTEXT))
        return st, v, a, lv

    return jax.vmap(one_meadow)(jax.random.split(key, meadows))


def truth(states, action):
    """Every sheep repeats ``action`` for K steps; returns (levels before, levels after, mean reward)."""

    def one(st):
        acts = jnp.full(st.flock.pos.shape[:1], action, jnp.int32)
        def body(s, _):
            n = sim.step(s, acts)
            return n, n.flock.reward

        end, r = jax.lax.scan(body, st, None, length=K)
        return st.flock.levels, end.flock.levels, r.mean(0)

    return jax.vmap(one)(states)


@nnx.jit
def believe(world, views_, actions, levels, probe, key):
    """views_ (N, CONTEXT, H, W) etc. -> predicted (levels at the last observed step, levels after
    K imagined steps of ``probe``, mean imagined reward)."""
    k1, k2 = jax.random.split(key)
    s = world.observe(views_.astype(jnp.int32), actions, k1, levels)
    h, z = s.h[:, -1], s.z[:, -1]
    lv0 = world.intero_head(jnp.concatenate([h, z], -1))
    # the first imagined step follows the probe action taken after the last observed strip
    acts = jnp.full((h.shape[0], K), probe, jnp.int32)
    im = world.imagine(h, z, acts, k2)
    feat = jnp.concatenate([im.h, im.z], -1)
    if world.ensemble is not None:
        prev = jnp.concatenate([jnp.concatenate([h, z], -1)[:, None], feat[:, :-1]], 1)
        dis = world.disagreement(prev, jax.nn.one_hot(acts, wm.N_ACTIONS)).mean(1)
    else:
        dis = jnp.zeros(h.shape[:1])
    return lv0, world.intero_head(feat[:, -1]), world.reward(feat).mean(1), dis


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wm", nargs="+", default=[os.path.expanduser("~/flock2_data/m1_wm/run4_negdrive/model.pkl")])
    p.add_argument("--meadows", type=int, default=8)
    p.add_argument("--sheep", type=int, default=40)
    p.add_argument("--warm", type=int, default=300)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--ensemble", type=int, default=0, help="ensemble heads in the checkpoints")
    p.add_argument("--n_intero", type=int, default=D.N_FELT, help="what the checkpoints feel: 6 (levels + taste, birth3 on) or 5 (levels only)")
    p.add_argument("--ensemble_target", default="body", help="what the checkpoints' ensemble predicts: body or z (birth1)")
    p.add_argument("--out", default=None, help="write the results as JSON here")
    a = p.parse_args()

    states, v, act, lv = warm_and_record(jax.random.PRNGKey(a.seed), a.meadows, a.sheep, a.warm)
    # (B, T, S, ...) -> (B*S, T, ...)
    seq = lambda x: np.asarray(x).swapaxes(1, 2).reshape((-1, CONTEXT) + x.shape[3:])
    v, act, lv = seq(v), seq(act).astype(np.int32), seq(lv).astype(np.float32)
    near = np.asarray(jax.vmap(lambda st: sim.near_water(st.meadow, st.flock.pos))(states)).reshape(-1)

    def grass_under(st):
        r, c = sim._tile(st.meadow, st.flock.pos)
        return st.meadow.grass[r, c] >= 1.0

    on_grass = np.asarray(jax.vmap(grass_under)(states)).reshape(-1)
    print(f"{near.size} sheep, {near.sum()} by water, {on_grass.sum()} on grass >= 1; mean hydration {lv[:, -1, D.HYDRATION].mean():.2f}", flush=True)

    true = {}
    for name in PROBED:
        l0, l1, r = (np.asarray(x).reshape((-1,) + x.shape[2:]) for x in truth(states, sim.ACTION_NAMES.index(name)))
        true[name] = (l1 - l0, r)

    results = {}
    for path in a.wm:
        world = wm.WorldModel(nnx.Rngs(0), n_intero=a.n_intero, reward=True, ensemble=a.ensemble, ensemble_target=a.ensemble_target)
        wm.load_into(world, pickle.load(open(os.path.expanduser(path), "rb")))
        res = {}
        for name in PROBED:
            outs = [believe(world, v[i : i + 32], act[i : i + 32], lv[i : i + 32, :, : a.n_intero], sim.ACTION_NAMES.index(name), jax.random.PRNGKey(i)) for i in range(0, len(v), 32)]
            p0, p1, pr, dis = (np.concatenate([np.asarray(o[j]) for o in outs]) for j in range(4))
            dl_true, r_true = true[name]
            row = {}
            groups = (("by_water", near), ("away", ~near)) if name != "graze" else (("on_grass", on_grass & ~near), ("bare", ~on_grass & ~near))
            for where, m in groups:
                if not m.any():
                    continue
                row[where] = {
                    "n": int(m.sum()),
                    "d_hydration_pred": round(float((p1 - p0)[m, D.HYDRATION].mean()), 4),
                    "d_hydration_true": round(float(dl_true[m, D.HYDRATION].mean()), 4),
                    "d_energy_pred": round(float((p1 - p0)[m, D.ENERGY].mean()), 4),
                    "d_energy_true": round(float(dl_true[m, D.ENERGY].mean()), 4),
                    "reward_pred": round(float(pr[m].mean()), 4),
                    "reward_true": round(float(r_true[m].mean()), 4),
                    "d_social_pred": round(float((p1 - p0)[m, D.SOCIAL].mean()), 4),
                    "d_social_true": round(float(dl_true[m, D.SOCIAL].mean()), 4),
                    "disagreement": float(dis[m].mean()),
                }
            res[name] = row
        results[path] = res
        print(json.dumps({"wm": path}), flush=True)
        for name in PROBED:
            for where, row in res[name].items():
                print(f"  {name:6s} {where:9s} n={row['n']:4d}  d_hydration pred {row['d_hydration_pred']:+.3f} true {row['d_hydration_true']:+.3f}   d_energy pred {row['d_energy_pred']:+.3f} true {row['d_energy_true']:+.3f}   reward pred {row['reward_pred']:+.3f} true {row['reward_true']:+.3f}   d_social pred {row['d_social_pred']:+.3f} true {row['d_social_true']:+.3f}   disagreement {row['disagreement']:.2e}", flush=True)
    if a.out:
        json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
