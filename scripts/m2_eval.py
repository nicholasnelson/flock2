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
- the action mix;
- dispersion (spec § 15): mean nearest-neighbour distance and other sheep in view, among the living;
- the **information gap** (spec § 15): the share of thirsty sheep-steps (hydration < 0.5) in which the
  sheep cannot see water but a living flockmate within ``EARSHOT`` tiles saw water in the last
  ``KNOWS_FOR`` steps. "Sees water" is a geometric proxy: a water tile within ``SEE_RANGE`` tiles
  inside the 300-degree field of view, ignoring occlusion. Also reported: the share of thirsty steps
  in which the sheep itself sees water.
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
        world, policy, hy, use_embed = agent

        def init(st):
            B, S = st.flock.alive.shape
            return hy.zero_mems(world, B, S)

        def act(st, mem, k):
            new, mem, rec = hy.sheep_step(world, policy, st, mem, k, use_embed)
            return rec["act"], mem

        return init, act
    raise ValueError(name)


EARSHOT, KNOWS_FOR, SEE_RANGE, THIRSTY = 10.0, 300, 16.0, 0.5
_TILES = jnp.stack(jnp.meshgrid(jnp.arange(48) + 0.5, jnp.arange(48) + 0.5, indexing="ij"), -1).reshape(-1, 2)


def sees_water(st):
    """(S,) whether a water tile lies within SEE_RANGE inside each sheep's field of view (occlusion ignored)."""
    from flock.meadow2.world import water_at

    wet = water_at(st.meadow.water, _TILES)
    rel = _TILES[None] - st.flock.pos[:, None]
    d = jnp.linalg.norm(rel, axis=-1)
    alpha = (jnp.arctan2(rel[..., 1], -rel[..., 0]) - st.flock.heading[:, None] + jnp.pi) % (2 * jnp.pi) - jnp.pi
    return jnp.any(wet[None] & (d <= SEE_RANGE) & (jnp.abs(alpha) < sim.FOV / 2), -1)


def run(policy, n_meadows, steps, seed=0):
    init, act = policy
    states = jax.vmap(random_start)(jnp.stack([jax.random.PRNGKey(50_000 + i) for i in range(n_meadows)]))

    def body(carry, k):
        (st, mem), (last_seen, t) = carry[:2], carry[2:]
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
        # dispersion and the information gap, among the living, before the step
        al = st.flock.alive
        sees = jax.vmap(sees_water)(st) & al
        last_seen = jnp.where(sees, t, last_seen)
        p = st.flock.pos
        d = jnp.linalg.norm(p[:, :, None] - p[:, None], axis=-1)
        others = al[:, None, :] & al[:, :, None] & ~jnp.eye(p.shape[1], dtype=bool)
        nn = jnp.min(jnp.where(others, d, jnp.inf), -1)
        knows = (t - last_seen <= KNOWS_FOR) & al
        told = jnp.any(others & (d <= EARSHOT) & knows[:, None, :], -1)
        thirsty = al & (st.flock.levels[..., D.HYDRATION] < THIRSTY)
        rec.update(nn=nn, seen=st.flock.seen, thirsty=thirsty, thirsty_sees=thirsty & sees, gap=thirsty & ~sees & told, alive_now=al)
        return (new, mem, last_seen, t + 1), rec

    def scan(s, k):
        B, S = s.flock.alive.shape
        return jax.lax.scan(body, (s, init(s), jnp.full((B, S), -100_000), jnp.int32(0)), jax.random.split(k, steps))

    _, rec = jax.jit(scan)(states, jax.random.PRNGKey(seed))
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
        "nearest_neighbour_tiles": float(np.nanmean(np.where(np.isfinite(rec["nn"]) & rec["alive_now"], rec["nn"], np.nan)[rec["alive_now"]])) if rec["alive_now"].any() else None,
        "others_in_view": float(rec["seen"][rec["alive_now"]].mean()),
        "thirsty_share_of_living_steps": float(rec["thirsty"].sum() / max(1, rec["alive_now"].sum())),
        "thirsty_sees_water": float(rec["thirsty_sees"].sum() / max(1, rec["thirsty"].sum())),
        "information_gap": float(rec["gap"].sum() / max(1, rec["thirsty"].sum())),
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
        use_embed = cfg.get("policy_input", "state") == "state_embed"
        policy = hy.make_policy(0, cfg.get("hidden", 256), hy.n_policy_in(world, use_embed))
        wm.load_into(world, pickle.load(open(os.path.join(d, "wm.pkl"), "rb")))
        wm.load_into(policy, pickle.load(open(os.path.join(d, "policy.pkl"), "rb")))
        agent = (world, policy, hy, use_embed)
    rec = run(policy_fn(a.policy, agent), a.meadows, int(a.days * DAY))
    res = {"policy": a.policy + (f":{a.agent}" if a.agent else ""), "meadows": a.meadows, "days": a.days, **summarise(rec)}
    print(json.dumps(res, indent=1))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
