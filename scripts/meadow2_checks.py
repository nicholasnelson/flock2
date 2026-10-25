#!/usr/bin/env python
"""The pre-registered checks of meadow v2 before any learning (spec § 7).

    python scripts/meadow2_checks.py --out runs/meadow2/checks.json

The scripted flock (``flock.meadow2.scripted``) on ``--meadows`` meadows from random starts
(``world.random_start``: levels in [0.2, 1.0], fat in [0.2, 0.8], a random time of day), for
``--steps`` steps (default 5 days). Pass/fail:

1. survival: at most 2 % of sheep die, and no meadow loses more than 1;
2. needs met: after day 1, the living flock's mean gut energy and hydration both stay above 0.4;
3. grass lasts: mean biomass of dry tiles on the last step is at least 70 % of the first;
4. water is near: from the flock's start, the nearest water is at most 12 tiles in the median
   meadow and 20 at the 90th percentile.

Reported, not pass/fail: the share of sheep that, walking straight on and turning a random 22.5°
with probability 1/30 a step, get their mouth onto water within 500 steps (flagged below 30 %).
"""

from __future__ import annotations

import argparse
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from flock.meadow2 import drives as D, scripted, sim
from flock.meadow2.world import random_start, start_distance


def run_scripted(states, key, steps):
    def body(st, k):
        a = jax.vmap(scripted.policy)(st, jax.random.split(k, st.t.shape[0]))
        st2 = jax.vmap(sim.step)(st, a)
        f = st2.flock
        al = f.alive.astype(jnp.float32)
        n_al = jnp.maximum(al.sum(-1), 1.0)
        rec = dict(
            alive=f.alive.sum(-1),
            energy=(f.levels[..., D.ENERGY] * al).sum(-1) / n_al,
            hydration=(f.levels[..., D.HYDRATION] * al).sum(-1) / n_al,
            warmth=(f.levels[..., D.WARMTH] * al).sum(-1) / n_al,
            fat=(f.fat * al).sum(-1) / n_al,
            reward=(f.reward * al).sum(-1) / n_al,
            bad=(-f.taste * al).sum(-1) / n_al,
            actions=jnp.stack([((a == i) & f.alive).sum(-1) for i in range(sim.N_ACTIONS)], -1),
        )
        return st2, rec

    return jax.lax.scan(body, states, jax.random.split(key, steps))


def run_walker(states, key, steps):
    def body(c, k):
        st, reached = c
        n = st.flock.pos.shape[:2]
        u = jax.random.uniform(k, n)
        a = jnp.where(u < 1 / 60, sim.LEFT, jnp.where(u < 1 / 30, sim.RIGHT, sim.WALK)).astype(jnp.int32)
        st = jax.vmap(sim.step)(st, a)
        on = jax.vmap(lambda m, p, h: sim.mouth_tile(m, p, h)[0])(st.meadow, st.flock.pos, st.flock.heading)
        return (st, reached | on), None

    (_, reached), _ = jax.lax.scan(body, (states, jnp.zeros(states.flock.alive.shape, bool)), jax.random.split(key, steps))
    return reached


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--meadows", type=int, default=64)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--walk_steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/meadow2/checks.json")
    a = p.parse_args()
    keys = jnp.stack([jax.random.PRNGKey(30_000 + i) for i in range(a.meadows)])
    states = jax.jit(jax.vmap(random_start))(keys)
    grass0 = np.asarray(states.meadow.grass)
    water = np.asarray(states.meadow.water)
    d_start = np.asarray(jax.vmap(lambda w, p: start_distance(w, p.mean(0)[None])[0])(states.meadow.water, states.flock.pos))

    end, rec = jax.jit(run_scripted, static_argnums=2)(states, jax.random.PRNGKey(a.seed), a.steps)
    rec = jax.tree.map(np.asarray, rec)
    n_sheep = states.flock.pos.shape[1]
    deaths = n_sheep - rec["alive"][-1]
    dead_end = ~np.asarray(end.flock.alive)
    lv_end = np.asarray(end.flock.levels)
    thirst_deaths = int((dead_end & (lv_end[..., D.HYDRATION] <= 0)).sum())
    after = slice(1200, None)
    grass_end = np.asarray(end.meadow.grass)
    dry = ~water
    g0, g1 = (grass0 * dry).sum() / dry.sum(), (grass_end * dry).sum() / dry.sum()

    reached = np.asarray(jax.jit(run_walker, static_argnums=2)(states, jax.random.PRNGKey(a.seed + 1), a.walk_steps))

    res = {
        "meadows": a.meadows,
        "sheep_per_meadow": int(n_sheep),
        "steps": a.steps,
        "1_survival": {
            "death_share": float(deaths.sum() / (a.meadows * n_sheep)),
            "max_deaths_in_a_meadow": int(deaths.max()),
            "deaths_of_thirst": thirst_deaths,
            "deaths_of_hunger": int(dead_end.sum()) - thirst_deaths,
            "pass": bool(deaths.sum() / (a.meadows * n_sheep) <= 0.02 and deaths.max() <= 1),
        },
        "2_needs_met": {
            "energy_after_day1_mean": float(rec["energy"][after].mean()),
            "energy_after_day1_min_meadow_step": float(rec["energy"][after].min()),
            "hydration_after_day1_mean": float(rec["hydration"][after].mean()),
            "hydration_after_day1_min_meadow_step": float(rec["hydration"][after].min()),
            "pass": bool(rec["energy"][after].mean() > 0.4 and rec["hydration"][after].mean() > 0.4),
        },
        "3_grass_lasts": {"dry_biomass_start": float(g0), "dry_biomass_end": float(g1), "ratio": float(g1 / g0), "pass": bool(g1 / g0 >= 0.7)},
        "4_water_near": {
            "start_to_water_median": float(np.median(d_start)),
            "start_to_water_p90": float(np.percentile(d_start, 90)),
            "pass": bool(np.median(d_start) <= 12 and np.percentile(d_start, 90) <= 20),
        },
        "reported": {
            "walker_reaches_water_within_%d_steps" % a.walk_steps: float(reached.mean()),
            "walker_flag_below_30pct": bool(reached.mean() < 0.3),
            "warmth_after_day1_mean": float(rec["warmth"][after].mean()),
            "fat_start_mean": float(rec["fat"][0].mean()),
            "fat_end_mean": float(rec["fat"][-1].mean()),
            "reward_per_step_mean": float(rec["reward"].mean()),
            "bad_taste_per_step": float(rec["bad"].mean()),
            "action_mix": {sim.ACTION_NAMES[i]: round(float(rec["actions"][..., i].sum() / rec["actions"].sum()), 3) for i in range(sim.N_ACTIONS)},
            "energy_by_day": [round(float(rec["energy"][d * 1200 : (d + 1) * 1200].mean()), 3) for d in range(a.steps // 1200)],
            "hydration_by_day": [round(float(rec["hydration"][d * 1200 : (d + 1) * 1200].mean()), 3) for d in range(a.steps // 1200)],
            "alive_by_day": [int(rec["alive"][min((d + 1) * 1200, a.steps) - 1].sum()) for d in range(a.steps // 1200)],
        },
    }
    print(json.dumps(res, indent=1))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
