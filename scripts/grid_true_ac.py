#!/usr/bin/env python
"""The third review's S1: the actor-critic (``flock.ac``) learning on the *true* grid simulator,
with no world model at all — does the update rule itself learn this task, and is gamma the
culprit?

    python scripts/grid_true_ac.py --out ~/flock2_data/grid/true_g098_s0 --gamma 0.98 --seed 0

The actor and critic see the whole map, one-hot (cell classes, the sheep's tile marked), and the
sheep's felt numbers (``N_IN`` = 16 x 16 x 6 + 3). Each update: every one of ``--worlds`` worlds
rolls ``ac.HORIZON`` real steps from where it is, the actor choosing; ``ac._update`` is then
applied to that real rollout exactly as it is to an imagined one (rewards real and centred; no
reward head, no curiosity; the critic bootstraps the last state). Worlds continue from where the
rollout ended and restart every ``--reset_every`` steps, staggered. Checkpoint ``agent.pkl``,
``config.json``; scored by ``grid_eval.py --policy true``.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from flock import ac
from flock.grid import brain, world as Wd

N_IN = Wd.G * Wd.G * Wd.N_CLASSES_FULL + Wd.N_FELT


def features(s: Wd.State) -> jax.Array:
    return jnp.concatenate([jax.nn.one_hot(Wd.view_full(s), Wd.N_CLASSES_FULL).reshape(-1), Wd.felt(s)])


def make_agent(seed, **kw):
    return ac.Agent(nnx.Rngs(seed), cfg=brain.GRID, n_in=N_IN, **kw)


@nnx.jit
def rollout_and_update(agent, opt, states, key):
    def step(st, k):
        f = jax.vmap(features)(st)
        a = agent.act(f, k)
        st2 = jax.vmap(Wd.step)(st, a)
        return st2, (f, a, st2.reward, st2.levels, (a == Wd.EAT), (a == Wd.DRINK), st2.taste < 0)

    end, (feats, acts, rew, lv, eat, drink, bad) = jax.lax.scan(step, states, jax.random.split(key, ac.HORIZON))
    feats = jnp.concatenate([feats, jax.vmap(features)(end)[None]], 0)
    aux = ac._update(agent, opt, feats, acts, rew, jnp.zeros_like(rew))
    aux.update(reward=rew.mean(), food=lv[..., 0].mean(), water=lv[..., 1].mean(), bad_taste=bad.mean())
    aux["actions"] = jnp.stack([(acts == i).mean() for i in range(Wd.N_ACTIONS)])
    return end, aux


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.expanduser("~/flock2_data/grid/true_s0"))
    p.add_argument("--updates", type=int, default=1500)
    p.add_argument("--worlds", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.98)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--critic_lr", type=float, default=None)
    p.add_argument("--entropy", type=float, default=ac.ENTROPY)
    p.add_argument("--norm", choices=("adv", "ret"), default="ret")
    p.add_argument("--reset_every", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    json.dump({**vars(a), "kind": "true"}, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    agent = make_agent(a.seed + 1, gamma=a.gamma, entropy=a.entropy, norm=a.norm)
    opt = ac.make_optimizer(agent, a.lr, a.critic_lr)
    fresh = jax.jit(jax.vmap(lambda k: Wd.generate(k)))
    key = jax.random.PRNGKey(a.seed)
    key, k = jax.random.split(key)
    states = fresh(jax.random.split(k, a.worlds))
    age = np.arange(a.worlds) * (a.reset_every // a.worlds) if a.reset_every >= a.worlds else np.random.default_rng(a.seed).integers(0, a.reset_every, a.worlds)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    t0 = time.time()
    for u in range(1, a.updates + 1):
        key, k = jax.random.split(key)
        states, aux = rollout_and_update(agent, opt, states, k)
        age += ac.HORIZON
        reset = age >= a.reset_every
        if reset.any():
            key, k = jax.random.split(key)
            nw = fresh(jax.random.split(k, a.worlds))
            m = jnp.asarray(reset)
            states = jax.tree.map(lambda o, n: jnp.where(m.reshape((-1,) + (1,) * (o.ndim - 1)), n, o), states, nw)
            age[reset] = 0
        if u % 10 == 0 or u == 1:
            rec = {"update": u, "t": time.time() - t0, **{k2: (float(v) if np.ndim(v) == 0 else np.round(np.asarray(v), 3).tolist()) for k2, v in aux.items()}}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if u % 100 == 0 or u == 1:
                print(json.dumps({k2: rec[k2] for k2 in ("update", "t", "reward", "food", "water", "bad_taste", "entropy", "return", "value_start", "abs_adv")}), flush=True)
        if u % 250 == 0 or u == a.updates:
            pickle.dump(jax.tree.map(np.asarray, nnx.to_pure_dict(nnx.state(agent))), open(os.path.join(a.out, "agent.pkl"), "wb"))


if __name__ == "__main__":
    main()
