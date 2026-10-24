#!/usr/bin/env python
"""A fixed evaluation of a flock's policy (Milestone 1): the same meadows, the same starts, the
same statistics for every policy, so "beats a random flock" is a number.

    python scripts/m1_eval.py --policy random
    python scripts/m1_eval.py --policy scripted
    python scripts/m1_eval.py --policy rest
    python scripts/m1_eval.py --policy agent --agent ~/flock2_data/m1_agent/birth3

Every policy runs on the same ``--meadows`` generated meadows (seeds 10,000 + i, so not the
training meadows), ``--sheep`` per meadow, from dawn (t = 0) with ``generate``'s starting levels
(energy, hydration, warmth drawn from [0.6, 0.9]; social and interest 0.8), for ``--days`` days.
A learned agent acts stochastically from its own world-model state, as in training.

Reported (and written to ``--out`` as JSON):
- at each dusk and dawn: mean and 10th percentile of every level across sheep;
- ``depleted``: the share of sheep with energy, hydration or warmth at or below 0.05 at the end;
- reward per step (``drives.reward``: minus the drive, minus the taste cost) and the drive;
- feeding: share of steps grazing where there is grass, drinking at water, and ingesting nothing
  (bad taste); and of graze and drink actions, the share that worked;
- the action mix by day and by night;
- night: share of night steps resting within ``drives.COVER_DIST`` of a tree or hut, and the
  mean distance to cover.
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
from flax import nnx

from flock import ac, wm
from flock.meadow import drives as D, scripted, sim
from flock.meadow.world import HUT, TREE, generate

STORES = (D.ENERGY, D.HYDRATION, D.WARMTH)


def _agent_module():
    spec = importlib.util.spec_from_file_location("m1_agent", os.path.join(os.path.dirname(__file__), "m1_agent.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def step_stats(state, new, actions):
    """Per-sheep statistics of one step, from the state before it and after it."""
    m, f = state.meadow, state.flock
    r, c = sim._tile(m, f.pos)
    graze_ok = (actions == sim.GRAZE) & (m.grass[r, c] >= 1.0)
    drink_ok = (actions == sim.DRINK) & sim.near_water(m, f.pos)
    cover = m.obj_present & ((m.obj_kind == TREE) | (m.obj_kind == HUT))
    d_cover = jnp.min(jnp.where(cover[None, :], jnp.linalg.norm(f.pos[:, None, :] - m.obj_pos[None, :, :], axis=-1), jnp.inf), -1)
    return dict(
        levels=new.flock.levels,
        reward=new.flock.reward,
        drive=D.drive(new.flock.levels),
        action=actions,
        graze_ok=graze_ok,
        drink_ok=drink_ok,
        bad_taste=new.flock.taste < 0,
        d_cover=d_cover,
        night=jnp.broadcast_to(sim.is_night(state.t), actions.shape),
    )


def run(policy, meadows, sheep, steps, seed, world=None, agent=None):
    states = jax.vmap(lambda k: generate(k, sheep))(jnp.stack([jax.random.PRNGKey(10_000 + i) for i in range(meadows)]))
    mems = (jnp.zeros((meadows, sheep, wm.DETER)), jnp.zeros((meadows, sheep, wm.STOCH * wm.CLASSES)), jnp.zeros((meadows, sheep), jnp.int32))
    am = _agent_module() if policy == "agent" else None

    def body(w, g, carry, k):
        def one_meadow(st, mem, kk):
            if policy == "agent":
                new, mem, _ = am.act_and_step(w, g, st, mem, kk, jnp.arange(1))
                actions = mem[2]
            else:
                if policy == "random":
                    actions = jax.random.randint(kk, (sheep,), 0, sim.N_ACTIONS)
                elif policy == "rest":
                    actions = jnp.full((sheep,), sim.REST, jnp.int32)
                else:
                    actions = scripted.policy(st, kk)
                new = sim.step(st, actions)
            return new, mem, step_stats(st, new, actions)

        st, mem = carry
        st, mem, rec = jax.vmap(one_meadow)(st, mem, jax.random.split(k, meadows))
        return (st, mem), rec

    @nnx.jit
    def go(w, g, s, m, k):
        return jax.lax.scan(lambda c, kk: body(w, g, c, kk), (s, m), jax.random.split(k, steps))[1]

    return go(world, agent, states, mems, jax.random.PRNGKey(seed))


def summarise(rec, days):
    rec = jax.tree.map(np.asarray, rec)  # (T, B, S, ...)
    T = rec["action"].shape[0]
    out = {}
    marks = []
    for d in range(days):
        marks += [(f"dusk{d + 1}", d * sim.DAY + int(sim.DAY * sim.DAY_FRACTION) - 1), (f"dawn{d + 2}", (d + 1) * sim.DAY - 1)]
    for name, t in marks:
        if t < T:
            lv = rec["levels"][t].reshape(-1, D.N_DRIVES)
            out[name] = {"mean": lv.mean(0).round(3).tolist(), "p10": np.percentile(lv, 10, axis=0).round(3).tolist()}
    end = rec["levels"][-1].reshape(-1, D.N_DRIVES)
    out["depleted"] = float((end[:, list(STORES)] <= 0.05).any(-1).mean())
    out["reward"] = float(rec["reward"].mean())
    out["drive"] = float(rec["drive"].mean())
    a = rec["action"]
    out["graze_ok_steps"] = float(rec["graze_ok"].mean())
    out["drink_ok_steps"] = float(rec["drink_ok"].mean())
    out["bad_taste_steps"] = float(rec["bad_taste"].mean())
    out["graze_success"] = float(rec["graze_ok"].sum() / max(1, (a == sim.GRAZE).sum()))
    out["drink_success"] = float(rec["drink_ok"].sum() / max(1, (a == sim.DRINK).sum()))
    night = rec["night"]
    for label, mask in (("day", ~night), ("night", night)):
        out[f"actions_{label}"] = {sim.ACTION_NAMES[i]: round(float((a[mask] == i).mean()), 3) for i in range(sim.N_ACTIONS)}
    near = rec["d_cover"] <= D.COVER_DIST
    out["night_rest_near_cover"] = float(((a == sim.REST) & near)[night].mean())
    out["night_mean_dist_to_cover"] = float(rec["d_cover"][night].mean())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=("random", "scripted", "rest", "agent"), required=True)
    p.add_argument("--agent", default=None, help="directory with wm.pkl and agent.pkl (policy agent)")
    p.add_argument("--meadows", type=int, default=8)
    p.add_argument("--sheep", type=int, default=40)
    p.add_argument("--days", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    world = agent = None
    if a.policy == "agent":
        d = os.path.expanduser(a.agent)
        world = wm.WorldModel(nnx.Rngs(0), n_intero=D.N_FELT, reward=True)
        wm.load_into(world, pickle.load(open(os.path.join(d, "wm.pkl"), "rb")))
        agent = ac.Agent(nnx.Rngs(1))
        wm.load_into(agent, pickle.load(open(os.path.join(d, "agent.pkl"), "rb")))
    rec = run(a.policy, a.meadows, a.sheep, a.days * sim.DAY, a.seed, world, agent)
    res = {"policy": a.policy if a.policy != "agent" else f"agent:{a.agent}", "meadows": a.meadows, "sheep": a.sheep, "days": a.days, **summarise(rec, a.days)}
    print(json.dumps(res))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
