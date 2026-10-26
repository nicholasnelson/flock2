#!/usr/bin/env python
"""The fixed evaluation for meadow v2: every policy on the same ``--meadows`` meadows (seeds
50,000 + i, never used in training), 12 sheep each, from ``world.random_start``'s varied starts,
for ``--days`` days.

    python scripts/m2_eval.py --policy random
    python scripts/m2_eval.py --policy scripted
    python scripts/m2_eval.py --policy rest
    python scripts/m2_eval.py --policy ppo --agent ~/flock2_data/m2_ppo/s0
    python scripts/m2_eval.py --policy hybrid --agent ~/flock2_data/m2_hybrid/s0   # world model + PPO, from its own view

Reported (and written to ``--out``), with standard errors across meadows where marked:
- reward per sheep per step, counting 0 after death (so dying costs), ± SE;
- the mean drive D over living sheep-steps, ± SE (the alive bonus shifts reward; D does not);
- the share of sheep that die, and mean days survived;
- each level and fat over the last day, among the living;
- drinking and grazing success, bad taste per step, the share of resting sheep huddled at night;
- the action mix.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from flock import wm
from flock.meadow2 import drives as D, privileged as P, scripted, sim
from flock.meadow2.world import DAY, random_start


def policy_fn(name, agent):
    """(init_mem(states), act(states, mem, key) -> (actions (B, S), mem)) over a batch of meadows."""
    stateless = lambda per: (lambda st: jnp.zeros(()), lambda st, mem, k: (jax.vmap(per)(st, jax.random.split(k, st.t.shape[0])), mem))
    if name == "random":
        return stateless(lambda st, k: jax.random.randint(k, st.flock.alive.shape, 0, sim.N_ACTIONS))
    if name == "rest":
        return stateless(lambda st, k: jnp.full(st.flock.alive.shape, sim.REST))
    if name == "scripted":
        return stateless(scripted.policy)
    if name == "ppo":
        return stateless(lambda st, k: jax.random.categorical(k, agent(P.observe(st))[0], -1))
    if name == "hybrid":
        world, policy, hy = agent

        def init(st):
            B, S = st.flock.alive.shape
            return hy.zero_mems(world, B, S)

        def act(st, mem, k):
            new, mem, rec = hy.sheep_step(world, policy, st, mem, k)
            return rec["act"], mem

        return init, act
    raise ValueError(name)


def run(policy, n_meadows, steps, seed=0):
    init, act = policy
    states = jax.vmap(random_start)(jnp.stack([jax.random.PRNGKey(50_000 + i) for i in range(n_meadows)]))

    def body(carry, k):
        st, mem = carry
        a, mem = act(st, mem, k)
        a = a.astype(jnp.int32)
        on_water, density, _ = jax.vmap(sim.mouth_tile)(st.meadow, st.flock.pos, st.flock.heading)
        new = jax.vmap(sim.step)(st, a)
        f = new.flock
        d_sh = jnp.linalg.norm(f.pos[:, :, None] - f.pos[:, None], axis=-1)
        huddled = ((d_sh < D.HUDDLE_DIST) & f.resting[:, None, :] & ~jnp.eye(f.pos.shape[1], dtype=bool)).any(-1) & f.resting
        rec = dict(
            reward=f.reward, alive_before=st.flock.alive, alive=f.alive, drive=D.drive(f.levels), levels=f.levels, fat=f.fat,
            drank=(a == sim.DRINK) & on_water, grazed=(a == sim.GRAZE) & (density >= sim.MIN_BITE), drink=a == sim.DRINK, graze=a == sim.GRAZE,
            bad=f.taste < 0, act=a, huddled=huddled, resting=f.resting, night=jax.vmap(sim.is_night)(st.t),
        )
        return (new, mem), rec

    _, rec = jax.jit(lambda s, k: jax.lax.scan(body, (s, init(s)), jax.random.split(k, steps)))(states, jax.random.PRNGKey(seed))
    return jax.tree.map(np.asarray, rec)


def summarise(rec):
    se = lambda x: float(x.std(ddof=1) / np.sqrt(len(x)))
    T, B, S = rec["reward"].shape
    alive_b = rec["alive_before"]
    per_meadow_reward = rec["reward"].mean((0, 2))
    drive_m = np.array([(rec["drive"][:, b][alive_b[:, b]]).mean() if alive_b[:, b].any() else np.nan for b in range(B)])
    last = slice(T - DAY, T)
    al_last = rec["alive"][last]
    lv = (rec["levels"][last] * al_last[..., None]).sum((0, 1, 2)) / max(1, al_last.sum())
    fat = float((rec["fat"][last] * al_last).sum() / max(1, al_last.sum()))
    died = ~rec["alive"][-1]
    survived_steps = alive_b.sum(0)
    night_rest = rec["resting"] & rec["night"][..., None]
    a = rec["act"][alive_b]
    return {
        "reward": float(per_meadow_reward.mean()),
        "reward_se": se(per_meadow_reward),
        "drive": float(np.nanmean(drive_m)),
        "drive_se": se(drive_m[np.isfinite(drive_m)]),
        "death_share": float(died.mean()),
        "deaths_per_meadow_max": int(died.sum(-1).max()),
        "days_survived_mean": float(survived_steps.mean() / DAY),
        "last_day_levels": {n: round(float(v), 3) for n, v in zip(D.NAMES, lv)},
        "last_day_fat": round(fat, 3),
        "drink_success": float(rec["drank"].sum() / max(1, rec["drink"].sum())),
        "graze_success": float(rec["grazed"].sum() / max(1, rec["graze"].sum())),
        "bad_taste_per_step": float(rec["bad"][alive_b].mean()),
        "huddled_share_of_night_rest": float(rec["huddled"][night_rest].mean()) if night_rest.any() else None,
        "actions": {sim.ACTION_NAMES[i]: round(float((a == i).mean()), 3) for i in range(sim.N_ACTIONS)},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=("random", "rest", "scripted", "ppo", "hybrid"), required=True)
    p.add_argument("--agent", default=None)
    p.add_argument("--meadows", type=int, default=16)
    p.add_argument("--days", type=float, default=3.0)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    agent = None
    if a.policy == "ppo":
        import importlib.util

        spec = importlib.util.spec_from_file_location("m2_ppo", os.path.join(os.path.dirname(__file__), "m2_ppo.py"))
        mp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mp)
        d = os.path.expanduser(a.agent)
        cfg = json.load(open(os.path.join(d, "config.json")))
        agent = mp.make_policy(0, cfg.get("hidden", 256))
        wm.load_into(agent, pickle.load(open(os.path.join(d, "ppo.pkl"), "rb")))
    if a.policy == "hybrid":
        import importlib.util

        spec = importlib.util.spec_from_file_location("m2_hybrid", os.path.join(os.path.dirname(__file__), "m2_hybrid.py"))
        hy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hy)
        d = os.path.expanduser(a.agent)
        cfg = json.load(open(os.path.join(d, "config.json")))
        world = hy.make_world(0, cfg.get("model", "vector"))
        policy = hy.make_policy(0, cfg.get("hidden", 256), world.cfg.feat)
        wm.load_into(world, pickle.load(open(os.path.join(d, "wm.pkl"), "rb")))
        wm.load_into(policy, pickle.load(open(os.path.join(d, "policy.pkl"), "rb")))
        agent = (world, policy, hy)
    rec = run(policy_fn(a.policy, agent), a.meadows, int(a.days * DAY))
    res = {"policy": a.policy + (f":{a.agent}" if a.agent else ""), "meadows": a.meadows, "days": a.days, **summarise(rec)}
    print(json.dumps(res, indent=1))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
