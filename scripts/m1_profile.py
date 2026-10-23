#!/usr/bin/env python
"""Profile the meadow (Milestone 1, step 1): agent-steps per second for B meadows of S sheep,
scripted flock, with and without every sheep's first-person view rendered each step.

    python scripts/m1_profile.py
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp

from flock.meadow import scripted, sim, views
from flock.meadow.world import generate


def make_run(n_steps: int, with_views: bool):
    def one(st, k):
        st = sim.step(st, scripted.policy(st, k))
        # reduce the views to a number so they are computed but not stored
        out = jnp.sum(views.render_flock(st).astype(jnp.float32)) if with_views else jnp.float32(0)
        return st, out

    def run(st, key):
        return jax.lax.scan(one, st, jax.random.split(key, n_steps))

    return jax.jit(jax.vmap(run))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sheep", type=int, default=40)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--reps", type=int, default=5)
    a = p.parse_args()
    print(jax.devices())
    print(f"{'meadows':>8s} {'sheep':>6s} {'views':>6s} {'agent-steps/s':>14s} {'ms per step':>12s}")
    for B in (1, 8, 32, 64):
        states = jax.vmap(lambda k: generate(k, a.sheep))(jax.random.split(jax.random.PRNGKey(0), B))
        keys = jax.random.split(jax.random.PRNGKey(1), B)
        for wv in (False, True):
            run = make_run(a.steps, wv)
            jax.block_until_ready(run(states, keys))
            t = time.time()
            reps = a.reps
            for _ in range(reps):
                out = run(states, keys)
            jax.block_until_ready(out)
            dt = (time.time() - t) / reps
            print(f"{B:>8d} {a.sheep:>6d} {str(wv):>6s} {B * a.sheep * a.steps / dt:14.0f} {1000 * dt / a.steps:12.2f}", flush=True)


if __name__ == "__main__":
    main()
