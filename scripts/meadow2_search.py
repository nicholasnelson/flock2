#!/usr/bin/env python
"""How long a memoryless search takes to find water (spec § 20.1).

Sheep are placed at uniform random dry positions and headings in the willow world, with full levels so
nobody dies inside the window, and run for one day under memoryless search rules. Recorded per sheep:
the first step at which water lies within 8 tiles inside the 300° field of view (§ 19's criterion,
occlusion ignored), binned by the start distance to water.

    JAX_PLATFORMS=cpu python scripts/meadow2_search.py --out runs/meadow2/search.json
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from flock.meadow2 import drives as D
from flock.meadow2 import sim
from flock.meadow2.world import DAY, N_TILES, random_start, water_at

NEAR = 8.0
BINS = ((4, 8), (8, 12), (12, 16), (16, 24), (24, 1e9))
_TILES = jnp.stack(jnp.meshgrid(jnp.arange(N_TILES) + 0.5, jnp.arange(N_TILES) + 0.5, indexing="ij"), -1).reshape(-1, 2)
# A1 s0's evaluation action marginals (stop, walk, trot, left, right, graze, drink, rest)
A1_MIX = np.array([0.059, 0.124, 0.157, 0.09, 0.11, 0.228, 0.04, 0.192])
A1_MIX = A1_MIX / A1_MIX.sum()


def sees_water(st, rng=NEAR):
    """(S,) whether a water tile lies within ``rng`` tiles inside each sheep's field of view."""
    wet = water_at(st.meadow.water, _TILES)
    rel = _TILES[None] - st.flock.pos[:, None]
    d = jnp.linalg.norm(rel, axis=-1)
    alpha = (jnp.arctan2(rel[..., 1], -rel[..., 0]) - st.flock.heading[:, None] + jnp.pi) % (2 * jnp.pi) - jnp.pi
    return jnp.any(wet[None] & (d <= rng) & (jnp.abs(alpha) < sim.FOV / 2), -1)


def water_distance(st):
    wet = water_at(st.meadow.water, _TILES)
    d = jnp.linalg.norm(_TILES[None] - st.flock.pos[:, None], axis=-1)
    return jnp.min(jnp.where(wet[None], d, jnp.inf), -1)


def scatter(key, willows):
    """A willow world with the sheep at uniform random allowed positions and headings, full levels."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    st = random_start(k1, willows=willows)
    n = st.flock.pos.shape[0]
    # rejection sampling with a fixed budget: 64 candidates per sheep, the first allowed one is used
    cand = jax.random.uniform(k2, (64, n, 2), minval=0.5, maxval=N_TILES - 0.5)
    ok = jax.vmap(lambda p: sim._allowed(st.meadow, p))(cand)  # (64, n)
    first = jnp.argmax(ok, 0)
    pos = cand[first, jnp.arange(n)]
    heading = jax.random.uniform(k3, (n,), minval=0.0, maxval=2 * jnp.pi)
    f = st.flock._replace(pos=pos, heading=heading, levels=jnp.full_like(st.flock.levels, D.MAX_LEVEL), fat=jnp.full((n,), 0.8))
    return st._replace(flock=f, t=jax.random.randint(k4, (), 0, DAY))


# --- search rules: (key, state, prev_action, blocked) -> actions -------------------------------------

def rule_mix(probs):
    probs = jnp.asarray(probs)

    def rule(key, st, prev, blocked):
        n = st.flock.pos.shape[0]
        return jax.random.choice(key, sim.N_ACTIONS, (n,), p=probs)

    return rule


def rule_walk(p_turn, wall=False, move=None):
    """Trot (or, with ``move`` = (p_walk, p_trot), move that often), turning one notch with probability
    ``p_turn`` per step; ``wall`` turns whenever the last trot was blocked."""

    def rule(key, st, prev, blocked):
        n = st.flock.pos.shape[0]
        k1, k2, k3 = jax.random.split(key, 3)
        turn = jax.random.bernoulli(k1, p_turn, (n,))
        if wall:
            turn = turn | blocked
        side = jnp.where(jax.random.bernoulli(k2, 0.5, (n,)), sim.LEFT, sim.RIGHT)
        if move is None:
            go = jnp.full((n,), sim.TROT)
        else:
            u = jax.random.uniform(k3, (n,))
            go = jnp.where(u < move[0], sim.WALK, jnp.where(u < move[0] + move[1], sim.TROT, sim.STOP))
        return jnp.where(turn, side, go)

    return rule


def rules():
    r = {"uniform": rule_mix(np.full(8, 1 / 8)), "a1_mix": rule_mix(A1_MIX)}
    for p in (0.02, 0.05, 0.1, 0.25, 0.5):
        r[f"walk_p{p}"] = rule_walk(p)
    for p in (0.02, 0.05, 0.1):
        r[f"walk_p{p}_wall"] = rule_walk(p, wall=True)
    r["lazy_p0.05"] = rule_walk(0.05, move=(0.124, 0.157))
    r["lazy_p0.05_wall"] = rule_walk(0.05, wall=True, move=(0.124, 0.157))
    return r


def run_rule(rule, st, key, steps):
    """(first step water is within 8 tiles in view (steps if never), start distance) per sheep."""
    n = st.flock.pos.shape[0]

    def body(carry, k):
        st, prev, blocked, first, i = carry
        a = rule(k, st, prev, blocked)
        nxt = sim.step(st, a)
        moved_wanted = (a == sim.WALK) | (a == sim.TROT)
        blocked = moved_wanted & (jnp.linalg.norm(nxt.flock.pos - st.flock.pos, axis=-1) < 1e-5)
        seen = sees_water(nxt)
        first = jnp.where((first == steps) & seen, i + 1, first)
        return (nxt, a, blocked, first, i + 1), None

    d0 = water_distance(st)
    first0 = jnp.where(sees_water(st), 0, steps)
    carry = (st, jnp.zeros((n,), jnp.int32), jnp.zeros((n,), bool), first0, jnp.int32(0))
    (_, _, _, first, _), _ = jax.lax.scan(body, carry, jax.random.split(key, steps))
    return first, d0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--meadows", type=int, default=32)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--steps", type=int, default=DAY)
    p.add_argument("--willows", type=int, default=3)
    p.add_argument("--rules", default="all")
    p.add_argument("--out", default="runs/meadow2/search.json")
    a = p.parse_args()
    all_rules = rules()
    names = list(all_rules) if a.rules == "all" else a.rules.split(",")
    out = {"meadows": a.meadows, "seeds": a.seeds, "steps": a.steps, "willows": a.willows, "near": NEAR, "rules": {}}
    for name in names:
        t0 = time.time()
        fn = jax.jit(jax.vmap(lambda st, k: run_rule(all_rules[name], st, k, a.steps)))
        firsts, d0s = [], []
        for seed in range(a.seeds):
            keys = jax.random.split(jax.random.PRNGKey(1000 + seed), a.meadows)
            sts = jax.vmap(lambda k: scatter(k, a.willows))(keys)
            first, d0 = fn(sts, jax.random.split(jax.random.PRNGKey(seed), a.meadows))
            firsts.append(np.asarray(first).ravel())
            d0s.append(np.asarray(d0).ravel())
        first, d0 = np.concatenate(firsts), np.concatenate(d0s)
        rows = {}
        for lo, hi in BINS:
            m = (d0 >= lo) & (d0 < hi) & (first > 0)  # exclude sheep that already see water at the start
            f = first[m]
            if len(f) == 0:
                continue
            found = f < a.steps
            rows[f"{lo}-{hi if hi < 1e8 else 'inf'}"] = {
                "n": int(len(f)), "found": round(float(found.mean()), 3),
                "median": float(np.median(f)), "p90": float(np.percentile(f, 90)),
                "within_500": round(float((found & (f <= 500)).mean()), 3), "within_1100": round(float((found & (f <= 1100)).mean()), 3),
            }
        out["rules"][name] = rows
        print(name, f"{time.time() - t0:.0f}s")
        for b, r in rows.items():
            print(f"  {b:>7} n={r['n']:4d} found={r['found']:.2f} median={r['median']:6.0f} p90={r['p90']:6.0f} <=500 {r['within_500']:.2f} <=1100 {r['within_1100']:.2f}")
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
