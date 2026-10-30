#!/usr/bin/env python
"""The fixed evaluation for meadow v2: every policy on the same ``--meadows`` meadows (seeds
50,000 + i, never used in training), 12 sheep each, from ``world.random_start``'s varied starts,
for ``--days`` days.

    python scripts/m2_eval.py --policy random
    python scripts/m2_eval.py --policy scripted
    python scripts/m2_eval.py --policy rest
    python scripts/m2_eval.py --policy ppo --agent ~/flock2_data/m2_ppo/s0
    python scripts/m2_eval.py --policy hybrid --agent ~/flock2_data/m2_hybrid/s0   # world model + PPO, from its own view
    python scripts/m2_eval.py --policy reactive --agent ~/flock2_data/m2_reactive/a1_s0   # stage 1: eyes + PPO, no world model

The meadows carry the willows of the agent's ``config.json`` (``--willows`` for the baselines, or to
override), so a policy is always scored on the world it learned.

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
- **Stage 0 (spec § 19.1):** the same two with an 8-tile cone (``NEAR_RANGE``, the eyes' range); the
  share of living steps within 8 tiles of water; deaths per day; and for every death, the day, the
  levels, the distance to water and the steps since water was last in the 8-tile cone and last within
  8 tiles (summarised in the result; the full list written to ``--deaths``).
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from flock import tiles as _tiles

_tiles.bootstrap()  # spec § 21.8: the meadow's side (--tiles, else the brain's config) before the world is imported

from flock import wm
from flock.meadow2 import drives as D, privileged as P, scripted, sim
from flock.meadow2.world import DAY, N_TILES, foggy, random_start, with_terrain


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
        policy, cfg = agent

        def init(st):
            B, S = st.flock.alive.shape
            return jax.tree.map(lambda x: jnp.broadcast_to(x, (B,) + x.shape), P.init_memory(S))

        def act(st, mem, k):
            obs, mem = jax.vmap(lambda s, m: P.observe_mem(s, m, cfg))(st, mem)
            return jax.random.categorical(k, policy(obs)[0], -1), mem

        return init, act
    if name == "hybrid":
        world, policy, hy, use_embed, column, col_size = agent

        def init(st):
            B, S = st.flock.alive.shape
            return hy.zero_mems(world, B, S, col_size)

        def act(st, mem, k):
            new, mem, rec = hy.sheep_step(world, policy, st, mem, k, use_embed, column)
            return rec["act"], mem

        return init, act
    if name == "reactive":
        policy, ra = agent
        return stateless_batch(lambda st, k: ra.act(policy, st, k)[0])
    raise ValueError(name)


def stateless_batch(per_batch):
    """A policy over the whole batch of meadows at once, without memory."""
    return lambda st: jnp.zeros(()), lambda st, mem, k: (per_batch(st, k), mem)


EARSHOT, KNOWS_FOR, SEE_RANGE, NEAR_RANGE, THIRSTY = 10.0, 300, 16.0, 8.0, 0.5
NEVER = -100_000
_TILES = jnp.stack(jnp.meshgrid(jnp.arange(N_TILES) + 0.5, jnp.arange(N_TILES) + 0.5, indexing="ij"), -1).reshape(-1, 2)


def sees_water(st, rng=SEE_RANGE):
    """(S,) whether a water tile lies within ``rng`` tiles inside each sheep's field of view (occlusion ignored)."""
    from flock.meadow2.world import water_at

    wet = water_at(st.meadow.water, _TILES)
    rel = _TILES[None] - st.flock.pos[:, None]
    d = jnp.linalg.norm(rel, axis=-1)
    alpha = (jnp.arctan2(rel[..., 1], -rel[..., 0]) - st.flock.heading[:, None] + jnp.pi) % (2 * jnp.pi) - jnp.pi
    return jnp.any(wet[None] & (d <= rng) & (jnp.abs(alpha) < sim.FOV / 2), -1)


def run(policy, n_meadows, steps, seed=0, willows=0, n_sheep=None, inner=0.0, edge=2.0, trample=0.0, fog=0.0, skyline=False, river=False):
    """``n_sheep`` (default the world's 12) for lone-sheep evaluations; ``inner`` > 0 puts stage 2a's mud
    on the banks (no grass or regrowth within ``inner`` tiles of water, regrowth ramping to full over
    ``edge`` more), ``trample`` the biomass lost under a moving sheep (spec § 21); ``fog`` the
    probability of each day being foggy (spec § 21.4); ``skyline`` the fixed peaks (spec § 21.7);
    ``river`` the river meadow (spec § 21.8)."""
    init, act = policy
    kw = {"willows": willows, "fog": fog, "skyline": skyline, "river": river} | ({} if n_sheep is None else {"n_sheep": n_sheep})
    states = jax.vmap(functools.partial(random_start, **kw))(jnp.stack([jax.random.PRNGKey(50_000 + i) for i in range(n_meadows)]))
    if inner > 0 or trample > 0:
        states = with_terrain(states, trample=trample, bank=(inner + edge) if inner > 0 else None, mud=inner > 0, inner=inner)

    def body(carry, k):
        (st, mem), (last_seen, last_seen8, last_near8, t) = carry[:2], carry[2:]
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
        sees8 = jax.vmap(lambda s: sees_water(s, NEAR_RANGE))(st) & al
        last_seen8 = jnp.where(sees8, t, last_seen8)
        wdist = jax.vmap(lambda s: P.visible_water(s, None)[1])(st)
        last_near8 = jnp.where((wdist <= NEAR_RANGE) & al, t, last_near8)
        p = st.flock.pos
        d = jnp.linalg.norm(p[:, :, None] - p[:, None], axis=-1)
        others = al[:, None, :] & al[:, :, None] & ~jnp.eye(p.shape[1], dtype=bool)
        nn = jnp.min(jnp.where(others, d, jnp.inf), -1)
        knows = (t - last_seen <= KNOWS_FOR) & al
        told = jnp.any(others & (d <= EARSHOT) & knows[:, None, :], -1)
        knows8 = (t - last_seen8 <= KNOWS_FOR) & al
        told8 = jnp.any(others & (d <= EARSHOT) & knows8[:, None, :], -1)
        thirsty = al & (st.flock.levels[..., D.HYDRATION] < THIRSTY)
        rec.update(nn=nn, seen=st.flock.seen, thirsty=thirsty, thirsty_sees=thirsty & sees, gap=thirsty & ~sees & told, alive_now=al,
                   thirsty_sees8=thirsty & sees8, gap8=thirsty & ~sees8 & told8, wdist=wdist, last_seen8=last_seen8, last_near8=last_near8,
                   levels_before=st.flock.levels, sees8=sees8, fog=jnp.broadcast_to(jax.vmap(foggy)(st.meadow, st.t)[:, None], al.shape))
        return (new, mem, last_seen, last_seen8, last_near8, t + 1), rec

    def scan(s, k):
        B, S = s.flock.alive.shape
        never = jnp.full((B, S), NEVER)
        return jax.lax.scan(body, (s, init(s), never, never, never, jnp.int32(0)), jax.random.split(k, steps))

    _, rec = jax.jit(scan)(states, jax.random.PRNGKey(seed))
    return jax.tree.map(np.asarray, rec)


def deaths(rec):
    """One record per sheep that died: when, its levels, how far water was and how long since it was
    last in the 8-tile cone or within 8 tiles (spec § 19.1)."""
    died_step = rec["alive_before"] & ~rec["alive"]  # (T, B, S), true once
    out = []
    for t, b, s in zip(*np.nonzero(died_step)):
        since = lambda last: int(t - last[t, b, s]) if last[t, b, s] != NEVER else None
        out.append({"meadow": int(b), "sheep": int(s), "step": int(t), "day": int(t // DAY),
                    "levels": {n: round(float(v), 3) for n, v in zip(D.NAMES, rec["levels_before"][t, b, s])},
                    "water_dist": round(float(rec["wdist"][t, b, s]), 2),
                    "since_seen8": since(rec["last_seen8"]), "since_near8": since(rec["last_near8"])})
    return out


def forensics(dl, steps):
    """Summary of ``deaths``: deaths per day, and the distribution of time since water."""
    days = int(np.ceil(steps / DAY))
    by_day = [sum(1 for d in dl if d["day"] == k) for k in range(days)]
    if not dl:
        return {"deaths_by_day": by_day}
    q = lambda xs: [round(float(v), 1) for v in np.percentile(xs, (25, 50, 75))] if len(xs) else None
    seen = [d["since_seen8"] for d in dl if d["since_seen8"] is not None]
    near = [d["since_near8"] for d in dl if d["since_near8"] is not None]
    return {
        "deaths_by_day": by_day,
        "deaths": len(dl),
        "water_dist_at_death_q": q([d["water_dist"] for d in dl]),
        "since_seen8_q": q(seen),
        "since_seen8_over_300": round(float(np.mean([v > 300 for v in seen])), 3) if seen else None,
        "never_seen8": round(float(np.mean([d["since_seen8"] is None for d in dl])), 3),
        "since_near8_q": q(near),
        "hydration_at_death_median": round(float(np.median([d["levels"]["hydration"] for d in dl])), 3),
        "energy_at_death_median": round(float(np.median([d["levels"]["energy"] for d in dl])), 3),
    }


# The behavioural memory probe M1 (spec § 21.4). An onset is a living sheep whose hydration has fallen
# below the scripted flock's urgency line (scripted.URGENT, where even its night rest gives way to
# water, so waiting is no longer an option), with no water in its 8-tile cone, more than 8 tiles from
# water, and within 8 tiles of water in the last RECENT steps (it has something to remember). The
# episode ends when water is back in the 8-tile cone (resolved), or after M1_CAP steps, or at death or
# the end (censored at M1_CAP); the next onset needs a new approach to within 8 tiles first (one
# episode per excursion). RECENT is 4,000 steps, more than a full hydration reserve (3,750): a sheep
# that drinks its fill and leaves is urgent about 2,800 steps later, so a half-day window (the first
# draft, 1,200) found almost no onsets (spec section 21.5); the age of the knowledge at onset is
# reported in two halves, fresh (<= 1,200 steps) and stale. The time to resight is compared, per bin of the distance at onset, with the
# best memoryless search rule of § 20.1 (the wall-aware correlated trot, turn probability 0.1, scattered
# starts, 2026-10-29): its median steps to the 8-tile cone. The rule reads the "unsighted" split,
# onsets in fog or at night, where the willows cannot carry the water's bearing; the other splits are
# reported. The scripted flock, which knows where the water is, is the positive control.
RECENT, M1_CAP, FRESH = 4000, 600, 1200
M1_THIRST = scripted.URGENT
M1_BINS = ((8, 12), (12, 16), (16, 24), (24, 1e9))
M1_BASELINE_MEDIAN = {"8-12": 90.0, "12-16": 254.0, "16-24": 244.0, "24-inf": 321.5}  # walk_p0.1_wall, meadow2-results/search/search.json
M1_FACTOR, M1_MIN_ONSETS = 0.5, 30


def memory_return(rec, thirst=M1_THIRST):
    """Spec § 21.4's M1. ``thirst`` other than the registered urgency line is a diagnostic reading (§ 21.6)."""
    T, B, S = rec["alive_now"].shape
    al, sees8, wdist, near8, fog, night = (rec[k] for k in ("alive_now", "sees8", "wdist", "last_near8", "fog", "night"))
    urgent = rec["levels_before"][..., D.HYDRATION] < thirst
    night = np.broadcast_to(night[..., None], al.shape)
    t_idx = np.arange(T)[:, None, None]
    knows = (t_idx - near8 <= RECENT) & (near8 != NEVER)
    can_start = al & urgent & ~sees8 & (wdist > NEAR_RANGE) & knows
    onsets = []  # (dist, duration, resolved, fog, night, age of the knowledge at onset)
    active = np.zeros((B, S), bool)
    t0 = np.zeros((B, S), int)
    t_close = np.full((B, S), NEVER)  # when the last episode closed; a new one needs a later approach
    d0 = np.zeros((B, S))
    f0 = np.zeros((B, S), bool)
    n0 = np.zeros((B, S), bool)
    a0 = np.zeros((B, S), int)

    def close(mask, t, resolved):
        for b, s in zip(*np.nonzero(mask)):
            onsets.append((d0[b, s], min(t - t0[b, s], M1_CAP), bool(resolved), bool(f0[b, s]), bool(n0[b, s]), a0[b, s]))
        active[mask] = False
        t_close[mask] = t

    for t in range(T):
        close(active & sees8[t] & al[t], t, True)
        close(active & (~al[t] | (t - t0 >= M1_CAP)), t, False)
        start = can_start[t] & ~active & (near8[t] > t_close)
        active |= start
        t0[start], d0[start], f0[start], n0[start], a0[start] = t, wdist[t][start], fog[t][start], night[t][start], (t - near8[t])[start]
    close(active, T, False)
    arr = np.array(onsets, dtype=float).reshape(-1, 6)
    out = {"onsets": int(len(arr)), "recent": RECENT, "cap": M1_CAP, "thirst": float(thirst), "baseline": "walk_p0.1_wall median steps to the 8-tile cone (spec § 20.1)"}
    unsighted = (arr[:, 3] > 0) | (arr[:, 4] > 0)
    splits = {"all": np.ones(len(arr), bool), "unsighted": unsighted, "clear_day": (arr[:, 3] == 0) & (arr[:, 4] == 0),
              "fog_day": (arr[:, 3] > 0) & (arr[:, 4] == 0), "night": arr[:, 4] > 0,
              "clear_night": (arr[:, 3] == 0) & (arr[:, 4] > 0), "fog_night": (arr[:, 3] > 0) & (arr[:, 4] > 0),
              "unsighted_fresh": unsighted & (arr[:, 5] <= FRESH), "unsighted_stale": unsighted & (arr[:, 5] > FRESH)}
    passes = []
    for name, sm in splits.items():
        bins = {}
        for lo, hi in M1_BINS:
            key = f"{lo}-{hi if hi < 1e8 else 'inf'}"
            m = sm & (arr[:, 0] >= lo) & (arr[:, 0] < hi)
            n = int(m.sum())
            base = M1_BASELINE_MEDIAN[key]
            row = {"onsets": n, "baseline_median": base}
            if n:
                dur = arr[m, 1]
                row.update(median_censored=float(np.median(dur)), resolved_share=float(arr[m, 2].mean()),
                           resolved_median=float(np.median(dur[arr[m, 2] > 0])) if (arr[m, 2] > 0).any() else None)
                if name == "unsighted" and n >= M1_MIN_ONSETS:
                    passes.append(row["median_censored"] <= M1_FACTOR * base)
            bins[key] = row
        out[name] = bins
    out["pass"] = bool(passes) and all(passes)
    out["bins_read"] = len(passes)
    return out


R1_CAP = 600
R1_BINS = ((0, 8), (8, 12), (12, 16), (16, 24), (24, 1e9))


def _bin_key(lo, hi):
    return f"{lo}-{hi if hi < 1e8 else 'inf'}"


def _conditions(rec):
    al, fog = rec["alive_now"], rec["fog"]
    night = np.broadcast_to(rec["night"][..., None], al.shape)
    return {"all": np.ones_like(al), "clear_day": ~fog & ~night, "fog_day": fog & ~night, "night": night, "unsighted": fog | night}


def wdist_share(rec):
    """Spec § 21.6's R1: the share of living steps per distance-from-water bin, by clear day / fog day / night."""
    al, wdist = rec["alive_now"], rec["wdist"]
    out = {}
    for name, c in _conditions(rec).items():
        m = al & c
        n = int(m.sum())
        out[name] = {"steps": n} | {_bin_key(lo, hi): round(float(((wdist >= lo) & (wdist < hi) & m).sum() / max(1, n)), 3) for lo, hi in R1_BINS}
    return out


def excursions(rec):
    """Spec § 21.6's R1: every spell more than 8 tiles from water by a sheep that had been within 8 before: its
    farthest point and the conditions there, and the time from the farthest point back to within 8 tiles
    (censored at R1_CAP, by death or the end of the run), per bin of the farthest distance."""
    T, B, S = rec["alive_now"].shape
    al, wdist, near8, fog = (rec[k] for k in ("alive_now", "wdist", "last_near8", "fog"))
    hyd = rec["levels_before"][..., D.HYDRATION]
    night = np.broadcast_to(rec["night"][..., None], al.shape)
    out_now = al & (wdist > NEAR_RANGE) & (near8 != NEVER)
    active = np.zeros((B, S), bool)
    dmax, hmax = np.zeros((B, S)), np.zeros((B, S))
    tmax = np.zeros((B, S), int)
    fmax, nmax = np.zeros((B, S), bool), np.zeros((B, S), bool)
    rows = []  # (d_max, return steps, resolved, fog, night, hydration at the farthest point, hydration at re-entry)

    def close(mask, t, resolved):
        for b, s in zip(*np.nonzero(mask)):
            dur = t - tmax[b, s]
            rows.append((dmax[b, s], min(dur, R1_CAP), bool(resolved and dur <= R1_CAP), bool(fmax[b, s]), bool(nmax[b, s]),
                         hmax[b, s], hyd[t, b, s] if resolved else np.nan))
        active[mask] = False

    for t in range(T):
        close(active & al[t] & (wdist[t] <= NEAR_RANGE), t, True)
        close(active & ~al[t], t, False)
        start = out_now[t] & ~active
        active |= start
        dmax[start] = 0.0
        further = active & (wdist[t] > dmax)
        dmax[further], tmax[further], fmax[further], nmax[further], hmax[further] = wdist[t][further], t, fog[t][further], night[t][further], hyd[t][further]
    close(active, T - 1, False)
    arr = np.array(rows, dtype=float).reshape(-1, 7)
    out = {"excursions": int(len(arr)), "cap": R1_CAP, "straight_steps_per_tile": {"walk": round(1 / sim.WALK_SPEED, 1), "trot": round(1 / sim.TROT_SPEED, 1)}}
    unsighted = (arr[:, 3] > 0) | (arr[:, 4] > 0)
    splits = {"all": np.ones(len(arr), bool), "clear_day": ~unsighted, "fog_day": (arr[:, 3] > 0) & (arr[:, 4] == 0), "night": arr[:, 4] > 0,
              "clear_night": (arr[:, 3] == 0) & (arr[:, 4] > 0), "fog_night": (arr[:, 3] > 0) & (arr[:, 4] > 0), "unsighted": unsighted}
    med = lambda x: float(np.median(x)) if len(x) else None
    for name, sm in splits.items():
        bins = {}
        for lo, hi in R1_BINS[1:]:
            m = sm & (arr[:, 0] >= lo) & (arr[:, 0] < hi)
            n = int(m.sum())
            row = {"n": n}
            if n:
                dur, res = arr[m, 1], arr[m, 2] > 0
                row.update(median_return=med(dur), resolved_share=float(res.mean()), resolved_median=med(dur[res]),
                           steps_per_tile_median=med(dur / np.maximum(arr[m, 0] - NEAR_RANGE, 0.5)),
                           hydration_at_farthest_median=med(arr[m, 5]), hydration_at_reentry_median=med(arr[m, 6][res]))
            bins[_bin_key(lo, hi)] = row
        out[name] = bins
    return out


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
        "thirsty_sees_water_8": float(rec["thirsty_sees8"].sum() / max(1, rec["thirsty"].sum())),
        "information_gap_8": float(rec["gap8"].sum() / max(1, rec["thirsty"].sum())),
        "within_8_share_of_living_steps": float(((rec["wdist"] <= NEAR_RANGE) & rec["alive_now"]).sum() / max(1, rec["alive_now"].sum())),
        "forensics": forensics(deaths(rec), T),
        "M1": memory_return(rec),
        # spec § 21.6's diagnostic R1 (readings, not pass rules)
        "wdist_share": wdist_share(rec),
        "excursions": excursions(rec),
        "M1_at_thirsty": memory_return(rec, THIRSTY),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=("random", "rest", "scripted", "ppo", "hybrid", "reactive"), required=True)
    p.add_argument("--willows", type=int, default=None, help="willows per pond (default: the agent's config, else 0)")
    p.add_argument("--agent", default=None)
    p.add_argument("--meadows", type=int, default=16)
    p.add_argument("--days", type=float, default=3.0)
    p.add_argument("--out", default=None)
    p.add_argument("--deaths", default=None, help="write every death's record here (spec § 19.1)")
    p.add_argument("--sheep", type=int, default=None, help="sheep per meadow (default 12); 1 for a lone-sheep evaluation")
    p.add_argument("--inner", type=float, default=0.0, help="stage 2a's mud radius in tiles (0: none)")
    p.add_argument("--edge", type=float, default=2.0, help="tiles over which regrowth ramps back to full outside the mud")
    p.add_argument("--trample", type=float, default=0.0, help="biomass lost per step under a moving sheep")
    p.add_argument("--zero_column", action="store_true", help="ablation (spec § 21.4): the policy sees zeros where the column's state would be")
    p.add_argument("--fog", type=float, default=0.0, help="probability of each day being foggy (spec § 21.4)")
    p.add_argument("--thirst", type=float, default=1.0, help="scale on the hydration decay (spec § 21.6: 2 halves the reserve)")
    p.add_argument("--skyline", type=int, default=None, help="1: the fixed peaks on the horizon (spec § 21.7); default the agent's config, else 0; 0 on a skyline-trained brain is the ablation")
    p.add_argument("--river", type=int, default=None, help="1: the river meadow (spec § 21.8); default the agent's config, else 0")
    p.add_argument("--tiles", type=int, default=None, help="the meadow's side in tiles (spec § 21.8); default the agent's config, else 48")
    a = p.parse_args()
    if a.thirst != 1.0:
        # the constant is read at trace time, so scaling it here before anything is jitted changes the world
        D.HYDRATION_DECAY = D.HYDRATION_DECAY * a.thirst
    agent = None
    willows = a.willows or 0
    skyline = bool(a.skyline or 0)
    river = bool(a.river or 0)
    if a.agent:
        cfg0 = json.load(open(os.path.join(os.path.expanduser(a.agent), "config.json")))
        if a.willows is None:
            willows = cfg0.get("willows", 0)
        if a.skyline is None:
            skyline = bool(cfg0.get("skyline", False))
        if a.river is None:
            river = bool(cfg0.get("river", False))
    if a.policy == "reactive":
        import importlib.util

        spec = importlib.util.spec_from_file_location("m2_reactive", os.path.join(os.path.dirname(__file__), "m2_reactive.py"))
        ra = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ra)
        d = os.path.expanduser(a.agent)
        cfg = json.load(open(os.path.join(d, "config.json")))
        policy = ra.make_policy(0, cfg.get("hidden", 256))
        wm.load_into(policy, pickle.load(open(os.path.join(d, "policy.pkl"), "rb")))
        agent = (policy, ra)
    if a.policy == "ppo":
        import importlib.util

        spec = importlib.util.spec_from_file_location("m2_ppo", os.path.join(os.path.dirname(__file__), "m2_ppo.py"))
        mp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mp)
        d = os.path.expanduser(a.agent)
        cfg = json.load(open(os.path.join(d, "config.json")))
        perc = mp.perception(cfg)
        policy = mp.make_policy(0, cfg.get("hidden", 256), P.n_obs(perc))
        wm.load_into(policy, pickle.load(open(os.path.join(d, "ppo.pkl"), "rb")))
        agent = (policy, perc)
    if a.policy == "hybrid":
        import importlib.util

        spec = importlib.util.spec_from_file_location("m2_hybrid", os.path.join(os.path.dirname(__file__), "m2_hybrid.py"))
        hy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hy)
        d = os.path.expanduser(a.agent)
        cfg = json.load(open(os.path.join(d, "config.json")))
        world = hy.make_world(0, cfg.get("model", "vector"))
        use_embed = cfg.get("policy_input", "state") == "state_embed"
        col = int(cfg.get("column", 0))
        policy = hy.make_policy(0, cfg.get("hidden", 256), hy.n_policy_in(world, use_embed, col))
        wm.load_into(world, pickle.load(open(os.path.join(d, "wm.pkl"), "rb")))
        wm.load_into(policy, pickle.load(open(os.path.join(d, "policy.pkl"), "rb")))
        column = hy.make_column(0, world, col)
        if column is not None:
            wm.load_into(column, pickle.load(open(os.path.join(d, "column.pkl"), "rb")))
        # --zero_column: the column is not run and its state stays at the zeros it starts from, so the policy
        # reads nothing there; if survival does not move, the column is unused whatever it holds
        agent = (world, policy, hy, use_embed, None if a.zero_column else column, col)
    rec = run(policy_fn(a.policy, agent), a.meadows, int(a.days * DAY), willows=willows, n_sheep=a.sheep, inner=a.inner, edge=a.edge, trample=a.trample, fog=a.fog, skyline=skyline, river=river)
    res = {"policy": a.policy + (f":{a.agent}" if a.agent else ""), "meadows": a.meadows, "days": a.days, "willows": willows,
           "sheep": a.sheep or 12, "inner": a.inner, "edge": a.edge, "trample": a.trample, "fog": a.fog, "thirst": a.thirst, "skyline": skyline, "river": river, "tiles": N_TILES, "zero_column": a.zero_column, **summarise(rec)}
    print(json.dumps(res, indent=1))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
    if a.deaths:
        os.makedirs(os.path.dirname(a.deaths) or ".", exist_ok=True)
        json.dump({"policy": res["policy"], "deaths": deaths(rec)}, open(a.deaths, "w"), indent=1)


if __name__ == "__main__":
    main()
