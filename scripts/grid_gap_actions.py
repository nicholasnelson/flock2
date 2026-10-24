#!/usr/bin/env python
"""The third review's S3: does the world model rank *actions* correctly within a state? (The
earlier ``grid_gap.py`` pooled over actions and worlds, so it mostly measured how well the model
tracks the drive levels of different worlds.)

    python scripts/grid_gap_actions.py --agent ~/flock2_data/grid/plain_s0

States: ``--worlds`` fixed worlds (seeds 70,000 + i, levels drawn as in training), each walked by
a *random* policy (so the states are not the agent's possibly collapsed ones), with snapshots
every ``--every`` steps after ``--warm``; the world model filters each sheep's state along the
way, as a learning sheep's. From each snapshot and for each of the 7 actions: take the action,
then follow one shared random action sequence for 31 more steps, both in the simulator (real
rewards) and in imagination (the reward head) — "do a, then act randomly". Reported:
- per horizon n in {1, 8, 32}: the correlation *within each state*, across the 7 actions, of the
  predicted and real n-step reward sums, averaged over states;
- in the states that decide the task — beside water with water < 0.5, and on grass >= 1 with
  food < 0.5 — the share in which the model's best action over 8 steps is drink (eat), and the
  same share for the real outcome as a reference.
Pre-registered (README): *worked if* drink and eat are ranked first in more than 80 % of those
states.
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
from flock.grid import brain, world as Wd

H = 32


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", required=True)
    p.add_argument("--worlds", type=int, default=256)
    p.add_argument("--warm", type=int, default=30)
    p.add_argument("--snapshots", type=int, default=6)
    p.add_argument("--every", type=int, default=25)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    d = os.path.expanduser(a.agent)
    cfgj = json.load(open(os.path.join(d, "config.json")))
    view = cfgj.get("view", "local")
    cfg = brain.config(view)
    world = wm.WorldModel(nnx.Rngs(0), n_intero=Wd.N_FELT, reward=True, ensemble=cfgj.get("ensemble", 0), cfg=cfg)
    wm.load_into(world, pickle.load(open(os.path.join(d, "wm.pkl"), "rb")))

    def filt(w, s, mem, k):
        h, z, a_prev = mem
        k1, k2 = jax.random.split(k)
        embed = w.encoder(jax.nn.one_hot(Wd.observe(s, view), cfg.n_classes)) + w.intero_in(Wd.felt(s))
        h = w.transition(h, z, jax.nn.one_hot(a_prev, Wd.N_ACTIONS))
        z = wm._sample(w.posterior(h, embed), k1)
        act = jax.random.randint(k2, (), 0, Wd.N_ACTIONS)
        return Wd.step(s, act), (h, z, act)

    def probe(w, s, mem, key):
        """From one true state and its filtered model state: (pred, real) reward sums (7, H)."""
        h, z, a_prev = mem
        embed = w.encoder(jax.nn.one_hot(Wd.observe(s, view), cfg.n_classes)) + w.intero_in(Wd.felt(s))
        k0, k1, k2 = jax.random.split(key, 3)
        h1 = w.transition(h, z, jax.nn.one_hot(a_prev, Wd.N_ACTIONS))
        z1 = wm._sample(w.posterior(h1, embed), k0)
        rand = jax.random.randint(k1, (H - 1,), 0, Wd.N_ACTIONS)

        def per_action(first):
            seq = jnp.concatenate([first[None], rand])

            def real_body(st, act):
                st2 = Wd.step(st, act)
                return st2, st2.reward

            _, rr = jax.lax.scan(real_body, s, seq)
            im = w.imagine(h1[None], z1[None], seq[None], k2)
            pr = w.reward(jnp.concatenate([im.h, im.z], -1))[0]
            return pr, rr

        return jax.vmap(per_action)(jnp.arange(Wd.N_ACTIONS))

    @nnx.jit
    def go(w, key):
        states = jax.vmap(lambda k: Wd.generate(k))(jnp.stack([jax.random.PRNGKey(70_000 + i) for i in range(a.worlds)]))
        mems = (jnp.zeros((a.worlds, cfg.deter)), jnp.zeros((a.worlds, cfg.stoch * cfg.classes)), jnp.zeros(a.worlds, jnp.int32))

        def walk(c, k):
            st, mm = c
            return jax.vmap(lambda s, m, kk: filt(w, s, m, kk))(st, mm, jax.random.split(k, a.worlds)), None

        out = []
        k = key
        for snap in range(a.snapshots):
            k, kw, kp = jax.random.split(k, 3)
            n = a.warm if snap == 0 else a.every
            (states, mems), _ = jax.lax.scan(walk, (states, mems), jax.random.split(kw, n))
            pr, rr = jax.vmap(lambda s, m, kk: probe(w, s, m, kk))(states, mems, jax.random.split(kp, a.worlds))
            r, c = states.pos[:, 0], states.pos[:, 1]
            beside = jax.vmap(Wd.near_water)(states)
            grass_ok = jax.vmap(lambda s: s.grass[s.pos[0], s.pos[1]] >= 1.0)(states)
            out.append((pr, rr, beside & (states.levels[:, 1] < 0.5), grass_ok & (states.levels[:, 0] < 0.5)))
        return [jnp.concatenate(x) for x in zip(*out)]

    pr, rr, thirsty_at_water, hungry_on_grass = (np.asarray(x) for x in go(world, jax.random.PRNGKey(1)))
    res = {"agent": a.agent, "states": int(pr.shape[0])}
    for n in (1, 8, 32):
        p_n, r_n = pr[:, :, :n].sum(-1), rr[:, :, :n].sum(-1)  # (states, 7)
        ok = (r_n.std(1) > 1e-6) & (p_n.std(1) > 1e-6)
        pc, rc = p_n[ok] - p_n[ok].mean(1, keepdims=True), r_n[ok] - r_n[ok].mean(1, keepdims=True)
        corr = (pc * rc).sum(1) / np.sqrt((pc ** 2).sum(1) * (rc ** 2).sum(1))
        res[f"within_state_corr_n{n}"] = round(float(corr.mean()), 3)
        res[f"states_with_action_differences_n{n}"] = int(ok.sum())
    p8, r8 = pr[:, :, :8].sum(-1), rr[:, :, :8].sum(-1)
    for name, mask, act in (("drink_first_beside_water_thirsty", thirsty_at_water, Wd.DRINK), ("eat_first_on_grass_hungry", hungry_on_grass, Wd.EAT)):
        res[name] = {"states": int(mask.sum()), "model": round(float((p8[mask].argmax(1) == act).mean()), 3) if mask.any() else None,
                     "real": round(float((r8[mask].argmax(1) == act).mean()), 3) if mask.any() else None}
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
