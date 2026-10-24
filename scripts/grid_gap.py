#!/usr/bin/env python
"""The imagination gap in the grid world (second review of 2026-10-24, § 3): from the same states,
does the world model's imagined future match the real one?

    python scripts/grid_gap.py --agent ~/flock2_data/grid/plain_s0

On 64 fixed worlds (seeds 60,000 + i, levels drawn as in training), the trained sheep acts for
``--warm`` real steps with its own policy so its model state is filtered from real views; then,
from that state:
- **per action, one step:** the world model's predicted reward for each of the 7 actions against
  the real reward of that action (the simulator stepped from the same true state);
- **a 32-step rollout:** the imagined return under the sheep's actor (rewards from the reward
  head) against the real return under the same actor acting in the simulator from the same
  state (rewards undiscounted, means over worlds, with the correlation across worlds).
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
from flock.grid import world as Wd
from flock.grid.brain import GRID

H = 32


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", required=True)
    p.add_argument("--worlds", type=int, default=64)
    p.add_argument("--warm", type=int, default=40)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    d = os.path.expanduser(a.agent)
    cfg = json.load(open(os.path.join(d, "config.json")))
    world = wm.WorldModel(nnx.Rngs(0), n_intero=Wd.N_FELT, reward=True, ensemble=cfg.get("ensemble", 0), cfg=GRID)
    wm.load_into(world, pickle.load(open(os.path.join(d, "wm.pkl"), "rb")))
    agent = ac.Agent(nnx.Rngs(1), twohot=cfg.get("twohot", False), cfg=GRID)
    wm.load_into(agent, pickle.load(open(os.path.join(d, "agent.pkl"), "rb")))

    def filt(w, g, s, mem, k):
        h, z, a_prev = mem
        k1, k2 = jax.random.split(k)
        embed = w.encoder(jax.nn.one_hot(Wd.view(s), Wd.N_CLASSES)) + w.intero_in(Wd.felt(s))
        h = w.transition(h, z, jax.nn.one_hot(a_prev, Wd.N_ACTIONS))
        z = wm._sample(w.posterior(h, embed), k1)
        act = g.act(ac.feat_of(h, z), k2)
        return Wd.step(s, act), (h, z, act)

    @nnx.jit
    def go(w, g, key):
        states = jax.vmap(lambda k: Wd.generate(k))(jnp.stack([jax.random.PRNGKey(60_000 + i) for i in range(a.worlds)]))
        mems = (jnp.zeros((a.worlds, GRID.deter)), jnp.zeros((a.worlds, GRID.stoch * GRID.classes)), jnp.zeros(a.worlds, jnp.int32))
        k_warm, k_im, k_real, k_one = jax.random.split(key, 4)

        def warm(c, k):
            s, m = c
            s, m = jax.vmap(lambda ss, mm, kk: filt(w, g, ss, mm, kk))(s, m, jax.random.split(k, a.worlds))
            return (s, m), None

        (states, mems), _ = jax.lax.scan(warm, (states, mems), jax.random.split(k_warm, a.warm))
        h, z, a_prev = mems
        # the model state that would act next: transition with the last action, then the posterior on the current view
        embed = w.encoder(jax.nn.one_hot(jax.vmap(Wd.view)(states), Wd.N_CLASSES)) + w.intero_in(jax.vmap(Wd.felt)(states))
        h1 = w.transition(h, z, jax.nn.one_hot(a_prev, Wd.N_ACTIONS))
        z1 = wm._sample(w.posterior(h1, embed), k_one)
        # one step, every action
        def one_action(act):
            hn = w.transition(h1, z1, jax.nn.one_hot(jnp.full(a.worlds, act), Wd.N_ACTIONS))
            zn = wm._sample(w.prior(hn), k_one)
            pred = w.reward(ac.feat_of(hn, zn))
            real = jax.vmap(lambda s: Wd.step(s, act).reward)(states)
            return pred, real

        pred1, real1 = jax.vmap(one_action)(jnp.arange(Wd.N_ACTIONS))  # (A, worlds)
        # 32 steps: imagined with the actor
        feats, acts, ext, _ = ac.imagine(w, g, h1, z1, k_im)
        imag = ext.sum(0)
        # 32 steps: real, the same actor acting from the same filtered state
        def real_step(c, k):
            s, m = c
            s2, m2 = jax.vmap(lambda ss, mm, kk: filt(w, g, ss, mm, kk))(s, m, jax.random.split(k, a.worlds))
            return (s2, m2), s2.reward

        _, rr = jax.lax.scan(real_step, (states, mems), jax.random.split(k_real, ac.HORIZON))
        return pred1, real1, imag, rr.sum(0)

    pred1, real1, imag, real = (np.asarray(x) for x in go(world, agent, jax.random.PRNGKey(0)))
    res = {
        "agent": a.agent,
        "one_step": {Wd.ACTION_NAMES[i]: {"pred": round(float(pred1[i].mean()), 4), "real": round(float(real1[i].mean()), 4)} for i in range(Wd.N_ACTIONS)},
        "one_step_corr": float(np.corrcoef(pred1.ravel(), real1.ravel())[0, 1]),
        "return_imagined": float(imag.mean()),
        "return_real": float(real.mean()),
        "return_corr": float(np.corrcoef(imag, real)[0, 1]) if imag.std() > 0 and real.std() > 0 else None,
        "horizon": int(ac.HORIZON),
    }
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
