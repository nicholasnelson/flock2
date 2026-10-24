#!/usr/bin/env python
"""PPO on the same true-state features as ``grid_true_ac.py`` (the third review's S1 fallback):
if PPO learns the grid from these inputs and ``ac.py``'s update does not, the fault is in the
update rule, not the task or the features.

    python scripts/grid_ppo_true.py --out ~/flock2_data/grid/ppo_true_s0 --seed 0

PPO (Schulman et al. 2017, arXiv:1707.06347) with GAE (Schulman et al. 2015, arXiv:1506.02438),
using ``m1_ppo.py``'s policy and update: each update rolls ``--steps`` real steps in each of
``--worlds`` worlds, then ``--epochs`` passes over ``--minibatches`` minibatches; advantages
normalised per batch. Rewards centred by a running mean (as ``ac.py``) so value targets are
small. Checkpoint ``ppo.pkl``; scored by ``grid_eval.py --policy ppo_true``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from flock.grid import world as Wd


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(__file__), f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_ppo = _load("m1_ppo")
_true = _load("grid_true_ac")


def make_policy(seed, hidden=256):
    return _ppo.Policy(nnx.Rngs(seed), hidden=hidden, n_in=_true.N_IN)


@nnx.jit(static_argnums=(3,))
def rollout(policy, states, key, steps):
    def one(st, k):
        f = jax.vmap(_true.features)(st)
        logits, value = policy(f)
        a = jax.random.categorical(k, logits, -1)
        logp = jax.nn.log_softmax(logits)[jnp.arange(a.shape[0]), a]
        st2 = jax.vmap(Wd.step)(st, a)
        return st2, dict(obs=f, act=a, logp=logp, value=value, reward=st2.reward, levels=st2.levels, bad=st2.taste < 0)

    end, tr = jax.lax.scan(one, states, jax.random.split(key, steps))
    _, last_v = policy(jax.vmap(_true.features)(end))
    return end, tr, last_v


@jax.jit
def gae(reward, value, last_v, gamma, lam):
    def body(carry, x):
        adv_next, v_next = carry
        r, v = x
        delta = r + gamma * v_next - v
        adv = delta + gamma * lam * adv_next
        return (adv, v), adv

    _, adv = jax.lax.scan(body, (jnp.zeros_like(last_v), last_v), (reward, value), reverse=True)
    return adv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.expanduser("~/flock2_data/grid/ppo_true_s0"))
    p.add_argument("--updates", type=int, default=600)
    p.add_argument("--worlds", type=int, default=256)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.98)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent", type=float, default=0.01)
    p.add_argument("--reset_every", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    json.dump({**vars(a), "kind": "ppo_true"}, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    policy = make_policy(a.seed)
    opt = nnx.Optimizer(policy, optax.chain(optax.clip_by_global_norm(0.5), optax.adam(a.lr)), wrt=nnx.Param)
    fresh = jax.jit(jax.vmap(lambda k: Wd.generate(k)))
    key = jax.random.PRNGKey(a.seed)
    key, k = jax.random.split(key)
    states = fresh(jax.random.split(k, a.worlds))
    age = np.arange(a.worlds) * (a.reset_every // a.worlds) if a.reset_every >= a.worlds else np.random.default_rng(a.seed).integers(0, a.reset_every, a.worlds)
    rng = np.random.default_rng(a.seed)
    r_bar = None
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    t0 = time.time()
    for u in range(1, a.updates + 1):
        key, k = jax.random.split(key)
        states, tr, last_v = rollout(policy, states, k, a.steps)
        r_mean = float(tr["reward"].mean())
        r_bar = r_mean if r_bar is None else 0.95 * r_bar + 0.05 * r_mean
        adv = gae(tr["reward"] - r_bar, tr["value"], last_v, a.gamma, a.lam)
        ret = adv + tr["value"]
        flat = lambda x: np.asarray(x).reshape((-1,) + x.shape[2:])
        obs, act, lp, ad, rt = flat(tr["obs"]), flat(tr["act"]), flat(tr["logp"]), flat(adv), flat(ret)
        auxes = []
        for _ in range(a.epochs):
            for mb in np.array_split(rng.permutation(len(act)), a.minibatches):
                auxes.append(_ppo.update(policy, opt, obs[mb], act[mb], lp[mb], ad[mb], rt[mb], a.clip, a.ent))
        age += a.steps
        reset = age >= a.reset_every
        if reset.any():
            key, k = jax.random.split(key)
            nw = fresh(jax.random.split(k, a.worlds))
            m = jnp.asarray(reset)
            states = jax.tree.map(lambda o, n: jnp.where(m.reshape((-1,) + (1,) * (o.ndim - 1)), n, o), states, nw)
            age[reset] = 0
        if u % 10 == 0 or u == 1:
            lv = np.asarray(tr["levels"]).mean((0, 1))
            rec = {"update": u, "t": time.time() - t0, "reward": r_mean, "food": float(lv[0]), "water": float(lv[1]), "bad_taste": float(np.asarray(tr["bad"]).mean()),
                   **{k2: float(np.mean([float(x[k2]) for x in auxes])) for k2 in auxes[0]}}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if u % 50 == 0 or u == 1:
                print(json.dumps(rec), flush=True)
        if u % 100 == 0 or u == a.updates:
            pickle.dump(jax.tree.map(np.asarray, nnx.to_pure_dict(nnx.state(policy))), open(os.path.join(a.out, "ppo.pkl"), "wb"))


if __name__ == "__main__":
    main()
