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

v2.1 (spec § 11): the naive-explorer balance rule. The same walker's first time to water is
compared with each sheep's hydration budget (its starting hydration / the hydration loss per step)
at several ``NEED_SCALE`` settings. The rule: it finds water before it would die in at least half
of starts, at the ``NEED_SCALE`` in force.
"""

from __future__ import annotations

import argparse
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from flock.meadow2 import drives as D, scripted, sim
from flock.meadow2.world import DAY, random_start, start_distance


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
            seen=(f.seen * al).sum(-1) / n_al,
            huddle=jax.vmap(lambda p, r, al_: ((jnp.linalg.norm(p[:, None] - p[None], axis=-1) < D.HUDDLE_DIST) & r[None, :] & r[:, None] & ~jnp.eye(p.shape[0], dtype=bool)).any(-1).sum() / jnp.maximum(al_.sum(), 1))(f.pos, f.resting, f.alive),
            night=jax.vmap(sim.is_night)(st2.t),
            actions=jnp.stack([((a == i) & f.alive).sum(-1) for i in range(sim.N_ACTIONS)], -1),
        )
        return st2, rec

    return jax.lax.scan(body, states, jax.random.split(key, steps))


def run_walker(states, key, steps, bounce=False):
    """(S,) each walker's first step with its mouth on water (``steps`` if never). Walkers are
    immortal here (hydration is held) so first times can be compared with any budget."""
    def body(c, k):
        st, first, i = c
        n = st.flock.pos.shape[:2]
        u = jax.random.uniform(k, n)
        a = jnp.where(u < 1 / 60, sim.LEFT, jnp.where(u < 1 / 30, sim.RIGHT, sim.WALK)).astype(jnp.int32)
        if bounce:  # turn (a random way, kept per sheep) when the step ahead is blocked
            ahead = st.flock.pos + sim.WALK_SPEED * sim.direction(st.flock.heading)
            ok = jax.vmap(sim._allowed)(st.meadow, ahead)
            side = jnp.where(jnp.arange(n[1]) % 2 == 0, sim.LEFT, sim.RIGHT)[None, :]
            a = jnp.where(ok, a, side).astype(jnp.int32)
        st = jax.vmap(sim.step)(st, a)
        st = st._replace(flock=st.flock._replace(levels=jnp.ones_like(st.flock.levels), fat=jnp.ones_like(st.flock.fat), alive=jnp.ones_like(st.flock.alive)))
        on = jax.vmap(lambda m, p, h: sim.mouth_tile(m, p, h)[0])(st.meadow, st.flock.pos, st.flock.heading)
        first = jnp.where(on & (first == steps), i, first)
        return (st, first, i + 1), None

    init = (states, jnp.full(states.flock.alive.shape, steps), jnp.int32(0))
    (_, first, _), _ = jax.lax.scan(body, init, jax.random.split(key, steps))
    return first


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--meadows", type=int, default=64)
    p.add_argument("--steps", type=int, default=5 * DAY)
    p.add_argument("--walk_steps", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/meadow2/checks.json")
    p.add_argument("--walker_only", action="store_true", help="only the explorer measures (both walkers)")
    a = p.parse_args()
    keys = jnp.stack([jax.random.PRNGKey(30_000 + i) for i in range(a.meadows)])
    states = jax.jit(jax.vmap(random_start))(keys)
    grass0 = np.asarray(states.meadow.grass)
    water = np.asarray(states.meadow.water)
    d_start = np.asarray(jax.vmap(lambda w, p: start_distance(w, p.mean(0)[None])[0])(states.meadow.water, states.flock.pos))

    if a.walker_only:
        h0 = np.asarray(states.flock.levels[..., D.HYDRATION])
        loss = D.HYDRATION_DECAY * D.NEED_SCALE
        out = {}
        for bounce in (False, True):
            first = np.asarray(jax.jit(run_walker, static_argnums=(2, 3))(states, jax.random.PRNGKey(a.seed + 1), a.walk_steps, bounce))
            found = first < a.walk_steps
            out["turns_when_blocked" if bounce else "turns_at_random_only"] = {
                "never_within_%d" % a.walk_steps: float((~found).mean()),
                "within_500": float((first < 500).mean()),
                "before_death": {f"need_scale_{ns}": float((found & (first < h0 / (loss / ns))).mean()) for ns in (1, 2, 3, 4, 6)},
            }
        print(json.dumps(out, indent=1))
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)
        return
    end, rec = jax.jit(run_scripted, static_argnums=2)(states, jax.random.PRNGKey(a.seed), a.steps)
    rec = jax.tree.map(np.asarray, rec)
    n_sheep = states.flock.pos.shape[1]
    deaths = n_sheep - rec["alive"][-1]
    dead_end = ~np.asarray(end.flock.alive)
    lv_end = np.asarray(end.flock.levels)
    thirst_deaths = int((dead_end & (lv_end[..., D.HYDRATION] <= 0)).sum())
    after = slice(DAY, None)
    grass_end = np.asarray(end.meadow.grass)
    dry = ~water
    g0, g1 = (grass0 * dry).sum() / dry.sum(), (grass_end * dry).sum() / dry.sum()

    first = np.asarray(jax.jit(run_walker, static_argnums=2)(states, jax.random.PRNGKey(a.seed + 1), a.walk_steps))
    h0 = np.asarray(states.flock.levels[..., D.HYDRATION])
    loss = D.HYDRATION_DECAY * D.NEED_SCALE  # the loss per step at NEED_SCALE 1
    found = first < a.walk_steps  # a walker that never found water within walk_steps did not find it in time
    budget = {f"need_scale_{ns}": float((found & (first < h0 / (loss / ns))).mean()) for ns in (1, 2, 3, 4, 6)}

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
            "walker_reaches_water_within_500_steps": float((first < 500).mean()),
            "walker_flag_below_30pct": bool((first < 500).mean() < 0.3),
            "walker_first_water_median_steps": float(np.median(first)),
            "walker_never_within_%d" % a.walk_steps: float((first >= a.walk_steps).mean()),
            "naive_explorer_finds_water_before_death": budget,
            "naive_explorer_rule_pass_at_current_need_scale": bool(budget.get(f"need_scale_{D.NEED_SCALE:g}", 0.0) >= 0.5) if f"need_scale_{D.NEED_SCALE:g}" in budget else None,
            "warmth_after_day1_mean": float(rec["warmth"][after].mean()),
            "fat_start_mean": float(rec["fat"][0].mean()),
            "fat_end_mean": float(rec["fat"][-1].mean()),
            "reward_per_step_mean": float(rec["reward"].mean()),
            "bad_taste_per_step": float(rec["bad"].mean()),
            "action_mix": {sim.ACTION_NAMES[i]: round(float(rec["actions"][..., i].sum() / rec["actions"].sum()), 3) for i in range(sim.N_ACTIONS)},
            "energy_by_day": [round(float(rec["energy"][d * DAY : (d + 1) * DAY].mean()), 3) for d in range(a.steps // DAY)],
            "hydration_by_day": [round(float(rec["hydration"][d * DAY : (d + 1) * DAY].mean()), 3) for d in range(a.steps // DAY)],
            "alive_by_day": [int(rec["alive"][min((d + 1) * DAY, a.steps) - 1].sum()) for d in range(a.steps // DAY)],
            "seen_mean": float(rec["seen"].mean()),
            "huddling_share_at_night": float(rec["huddle"][rec["night"]].mean()),
        },
    }
    print(json.dumps(res, indent=1))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
