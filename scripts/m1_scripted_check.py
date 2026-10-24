#!/usr/bin/env python
"""Can the scripted flock keep its drives up? (Milestone 1: the existence proof that the drive
rates are survivable.)

    python scripts/m1_scripted_check.py --days 3
    python scripts/m1_scripted_check.py --flat_energy 0.0006   # the old action-independent energy use

Runs the scripted flock on fresh meadows from dawn and reports, per quarter-day: the mean of each
level, the 10th percentile across sheep (the struggling ones), and the scripted flock's action mix.
"""

from __future__ import annotations

import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np

from flock.meadow import drives as D, scripted, sim
from flock.meadow.world import generate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--meadows", type=int, default=8)
    p.add_argument("--sheep", type=int, default=40)
    p.add_argument("--days", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--flat_energy", type=float, default=None, help="use this energy use for every action instead of drives.ENERGY_USE")
    a = p.parse_args()
    if a.flat_energy is not None:
        D.ENERGY_USE = jnp.full(sim.N_ACTIONS, a.flat_energy)
    steps = a.days * sim.DAY
    quarter = sim.DAY // 4

    def run(key):
        k1, k2 = jax.random.split(key)
        st = generate(k1, a.sheep)._replace(t=jnp.int32(0))

        def one(s, k):
            act = scripted.policy(s, k)
            s = sim.step(s, act)
            return s, (s.flock.levels, act)

        return jax.lax.scan(one, st, jax.random.split(k2, steps))[1]

    lv, act = jax.jit(jax.vmap(run))(jax.random.split(jax.random.PRNGKey(a.seed), a.meadows))
    lv, act = np.asarray(lv), np.asarray(act)  # (B, T, S, 5), (B, T, S)
    out = []
    print("quarter  phase  " + " ".join(f"{n[:6]:>13}" for n in D.NAMES) + "   (mean / 10th pct)   actions")
    for q in range(steps // quarter):
        sl = slice(q * quarter, (q + 1) * quarter)
        x = lv[:, sl].reshape(-1, D.N_DRIVES)
        mean, p10 = x.mean(0), np.percentile(x, 10, axis=0)
        mix = np.bincount(act[:, sl].ravel(), minlength=sim.N_ACTIONS) / act[:, sl].size
        phase = "night" if (q % 4) >= 2 and (q % 4) * quarter >= sim.DAY * sim.DAY_FRACTION - quarter else "day"
        print(f"{q:7d}  {q % 4 / 4:.2f}   " + " ".join(f"  {m:.2f} / {l:.2f}" for m, l in zip(mean, p10)) + "   " + " ".join(f"{sim.ACTION_NAMES[i]} {mix[i]:.2f}" for i in np.argsort(-mix)[:4]))
        out.append({"quarter": q, "mean": mean.round(3).tolist(), "p10": p10.round(3).tolist(), "actions": mix.round(3).tolist()})
    print(json.dumps({"energy_use": np.asarray(D.ENERGY_USE).tolist(), "final_day_mean": lv[:, -sim.DAY:].reshape(-1, D.N_DRIVES).mean(0).round(3).tolist()}))


if __name__ == "__main__":
    main()
