#!/usr/bin/env python
"""A fixed evaluation for the grid world: every policy on the same ``--worlds`` worlds (seeds
50,000 + i, not training worlds), from food and water 0.7, for ``--steps`` steps.

    python scripts/grid_eval.py --policy random
    python scripts/grid_eval.py --policy scripted
    python scripts/grid_eval.py --policy stay
    python scripts/grid_eval.py --policy agent --agent ~/flock2_data/grid/run1 [--greedy]

Reported (and written to ``--out`` as JSON), with standard errors across worlds where marked:
reward per step (± SE); mean food and water over the last quarter (± SE); ``depleted``, the share
of worlds whose sheep ended with a level at or below 0.05; eat and drink success (share of those
actions that worked); bad taste per step; the action mix. A learned agent acts from its own
world-model state, sampled (default) or greedily (``--greedy``).
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
from flock.grid import scripted, world as Wd
from flock.grid.brain import GRID


def run(policy, n_worlds, steps, world=None, agent=None, greedy=False, seed=0):
    states = jax.vmap(lambda k: Wd.generate(k, jnp.array([0.7, 0.7])))(jnp.stack([jax.random.PRNGKey(50_000 + i) for i in range(n_worlds)]))
    mems = (jnp.zeros((n_worlds, GRID.deter)), jnp.zeros((n_worlds, GRID.stoch * GRID.classes)), jnp.zeros(n_worlds, jnp.int32))
    modes = jnp.zeros(n_worlds, jnp.int32)

    def one(w, g, s, mem, mode, k):
        k1, k2 = jax.random.split(k)
        if policy == "agent":
            h, z, a_prev = mem
            embed = w.encoder(jax.nn.one_hot(Wd.view(s), Wd.N_CLASSES)) + w.intero_in(Wd.felt(s))
            h = w.transition(h, z, jax.nn.one_hot(a_prev, Wd.N_ACTIONS))
            z = wm._sample(w.posterior(h, embed), k1)
            logits = ac.policy_logp(g.actor(ac.feat_of(h, z)))
            a = jnp.argmax(logits) if greedy else jax.random.categorical(k2, logits)
            mem = (h, z, a)
        elif policy == "scripted":
            a, mode = scripted.policy(s, k2, mode)
        elif policy == "stay":
            a = jnp.int32(Wd.STAY)
        else:
            a = jax.random.randint(k2, (), 0, Wd.N_ACTIONS)
        r, c = s.pos
        new = Wd.step(s, a)
        rec = dict(reward=new.reward, levels=new.levels, action=a, ate=(a == Wd.EAT) & (s.grass[r, c] >= 1.0), drank=(a == Wd.DRINK) & Wd.near_water(s), bad=new.taste < 0)
        return new, mem, mode, rec

    @nnx.jit
    def go(w, g, states, mems, modes, key):
        def body(c, k):
            st, mm, md = c
            st, mm, md, rec = jax.vmap(lambda s, m, d, kk: one(w, g, s, m, d, kk))(st, mm, md, jax.random.split(k, n_worlds))
            return (st, mm, md), rec

        return jax.lax.scan(body, (states, mems, modes), jax.random.split(key, steps))[1]

    return jax.tree.map(np.asarray, go(world, agent, states, mems, modes, jax.random.PRNGKey(seed)))


def summarise(rec):
    se = lambda x: float(x.std(ddof=1) / np.sqrt(len(x)))
    per_world_reward = rec["reward"].mean(0)
    q = rec["levels"].shape[0] // 4
    lv = rec["levels"][-q:].mean(0)  # (worlds, 2)
    a = rec["action"]
    out = {
        "reward": float(per_world_reward.mean()),
        "reward_se": se(per_world_reward),
        "food": float(lv[:, 0].mean()),
        "food_se": se(lv[:, 0]),
        "water": float(lv[:, 1].mean()),
        "water_se": se(lv[:, 1]),
        "depleted": float((rec["levels"][-1] <= 0.05).any(-1).mean()),
        "eat_success": float(rec["ate"].sum() / max(1, (a == Wd.EAT).sum())),
        "drink_success": float(rec["drank"].sum() / max(1, (a == Wd.DRINK).sum())),
        "bad_taste": float(rec["bad"].mean()),
        "actions": {Wd.ACTION_NAMES[i]: round(float((a == i).mean()), 3) for i in range(Wd.N_ACTIONS)},
    }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=("random", "stay", "scripted", "agent"), required=True)
    p.add_argument("--agent", default=None)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--worlds", type=int, default=64)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    world = agent = None
    if a.policy == "agent":
        d = os.path.expanduser(a.agent)
        cfg = json.load(open(os.path.join(d, "config.json")))
        world = wm.WorldModel(nnx.Rngs(0), n_intero=Wd.N_FELT, reward=True, ensemble=cfg.get("ensemble", 0), cfg=GRID)
        wm.load_into(world, pickle.load(open(os.path.join(d, "wm.pkl"), "rb")))
        agent = ac.Agent(nnx.Rngs(1), twohot=cfg.get("twohot", False), cfg=GRID)
        wm.load_into(agent, pickle.load(open(os.path.join(d, "agent.pkl"), "rb")))
    res = {"policy": a.policy + (f":{a.agent}" if a.agent else "") + (" greedy" if a.greedy else ""), **summarise(run(a.policy, a.worlds, a.steps, world, agent, a.greedy))}
    print(json.dumps(res))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
