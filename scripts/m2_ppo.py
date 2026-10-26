#!/usr/bin/env python
"""The upper bound for meadow v2: a small policy on privileged features
(``flock.meadow2.privileged``) trained model-free with PPO in the simulator, on the sheep's own
reward (``drives.reward``: the alive bonus minus the drive and bad taste, 0 once dead). No world
model, no vision.

    python scripts/m2_ppo.py --out ~/flock2_data/m2_ppo/s0 --seed 0

PPO (Schulman et al. 2017, arXiv:1707.06347) with GAE (Schulman et al. 2015, arXiv:1506.02438), as
``m1_ppo.py``, with two changes for death:
- **death ends a sheep's future**: GAE does not bootstrap past the step a sheep dies, and steps
  taken while already dead are left out of the loss (weight 0);
- **no reward centring**: subtracting the mean reward would cancel the alive bonus and make
  dying free. Rewards are instead scaled by (1 - gamma), so values stay near the reward per step.

Each update collects ``--steps`` steps from ``--meadows`` meadows of 12 sheep (one shared policy).
Meadows start as ``world.random_start`` and restart every ``--reset_every`` steps (staggered), or
as soon as every sheep in one has died. Checkpoint ``ppo.pkl`` and ``config.json``; scored by
``m2_eval.py --policy ppo``.
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

from flock.meadow2 import drives as D, privileged as P, sim
from flock.meadow2.world import DAY, random_start


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(__file__), f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_ppo = _load("m1_ppo")


def make_policy(seed, hidden=256):
    return _ppo.Policy(nnx.Rngs(seed), hidden=hidden, n_in=P.N_OBS, n_out=sim.N_ACTIONS)


@nnx.jit(static_argnums=(4,))
def rollout(policy, states, key, gamma_lam, steps):
    gamma, lam = gamma_lam

    def one(st, k):
        obs = P.observe(st)
        logits, value = policy(obs)
        a = jax.random.categorical(k, logits, -1)
        logp = jax.nn.log_softmax(logits)[jnp.arange(a.shape[0]), a]
        new = sim.step(st, a)
        return new, dict(obs=obs, act=a, logp=logp, value=value, reward=new.flock.reward * (1 - gamma), raw_reward=new.flock.reward,
                         alive_before=st.flock.alive, alive_after=new.flock.alive, levels=new.flock.levels, fat=new.flock.fat, bad=new.flock.taste < 0)

    def run(st, k):
        return jax.lax.scan(one, st, jax.random.split(k, steps))

    states, tr = jax.vmap(run)(states, jax.random.split(key, states.t.shape[0]))  # (B, T, S, ...)
    _, last_v = policy(jax.vmap(P.observe)(states))

    def gae(carry, x):
        adv_next, v_next = carry
        r, v, cont = x
        delta = r + gamma * v_next * cont - v
        adv = delta + gamma * lam * cont * adv_next
        return (adv, v), adv

    def per_meadow(r, v, cont, lv):
        _, adv = jax.lax.scan(gae, (jnp.zeros_like(lv), lv), (r, v, cont), reverse=True)
        return adv

    cont = tr["alive_after"].astype(jnp.float32)  # 0 at the step a sheep dies (and after)
    adv = jax.vmap(per_meadow)(tr["reward"], tr["value"], cont, last_v)
    tr["adv"] = adv
    tr["ret"] = adv + tr["value"]
    return states, tr


@nnx.jit
def update(policy, opt, obs, act_, logp_old, adv, ret, wgt, clip, ent_coef):
    """The clipped PPO update, with each sample weighted (0 for steps taken while dead)."""

    def loss_fn(p):
        logits, v = p(obs)
        logp_all = jax.nn.log_softmax(logits)
        logp = jnp.take_along_axis(logp_all, act_[:, None], -1)[:, 0]
        ratio = jnp.exp(logp - logp_old)
        n = jnp.maximum(wgt.sum(), 1.0)
        mean = lambda x: jnp.sum(x * wgt) / n
        mu = mean(adv)
        a = (adv - mu) / (jnp.sqrt(mean((adv - mu) ** 2)) + 1e-8)
        pg = -mean(jnp.minimum(ratio * a, jnp.clip(ratio, 1 - clip, 1 + clip) * a))
        vf = 0.5 * mean((v - ret) ** 2)
        ent = mean(-jnp.sum(jnp.exp(logp_all) * logp_all, -1))
        return pg + 0.5 * vf - ent_coef * ent, {"pg": pg, "vf": vf, "entropy": ent, "approx_kl": mean(logp_old - logp)}

    (loss, aux), g = nnx.value_and_grad(loss_fn, has_aux=True)(policy)
    opt.update(policy, g)
    return aux


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.expanduser("~/flock2_data/m2_ppo/s0"))
    p.add_argument("--updates", type=int, default=1500)
    p.add_argument("--meadows", type=int, default=32)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.998)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent", type=float, default=0.01)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--reset_every", type=int, default=2 * DAY)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    assert a.reset_every % a.steps == 0
    os.makedirs(a.out, exist_ok=True)
    json.dump({**vars(a), "kind": "m2_ppo"}, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    policy = make_policy(a.seed, a.hidden)
    opt = nnx.Optimizer(policy, optax.chain(optax.clip_by_global_norm(0.5), optax.adam(a.lr)), wrt=nnx.Param)
    fresh = jax.jit(jax.vmap(random_start))
    key = jax.random.PRNGKey(a.seed)
    key, k = jax.random.split(key)
    states = fresh(jax.random.split(k, a.meadows))
    age = (np.arange(a.meadows) % (a.reset_every // a.steps)) * a.steps
    rng = np.random.default_rng(a.seed)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    t0 = time.time()
    for u in range(1, a.updates + 1):
        key, k = jax.random.split(key)
        states, tr = rollout(policy, states, k, (a.gamma, a.lam), a.steps)
        tr = jax.tree.map(np.asarray, tr)
        flat = lambda x: x.reshape((-1,) + x.shape[3:])
        obs, act, lp, ad, rt, w = (flat(tr[k2]) for k2 in ("obs", "act", "logp", "adv", "ret", "alive_before"))
        w = w.astype(np.float32)
        auxes = []
        for _ in range(a.epochs):
            for mb in np.array_split(rng.permutation(len(act)), a.minibatches):
                auxes.append(update(policy, opt, obs[mb], act[mb], lp[mb], ad[mb], rt[mb], w[mb], a.clip, a.ent))
        age += a.steps
        all_dead = ~np.asarray(states.flock.alive).any(-1)
        reset = (age >= a.reset_every) | all_dead
        if reset.any():
            key, k = jax.random.split(key)
            nw = fresh(jax.random.split(k, a.meadows))
            m = jnp.asarray(reset)
            states = jax.tree.map(lambda o, n: jnp.where(m.reshape((-1,) + (1,) * (o.ndim - 1)), n, o), states, nw)
            age[reset] = 0
        alive = tr["alive_before"]
        rec = {
            "update": u,
            "t": time.time() - t0,
            "reward": float((tr["raw_reward"] * alive).sum() / max(1, alive.sum())),
            "alive_share": float(alive.mean()),
            "deaths": int((tr["alive_before"] & ~tr["alive_after"]).sum()),
            "levels": np.round((tr["levels"] * alive[..., None]).sum((0, 1, 2)) / max(1, alive.sum()), 3).tolist(),
            "fat": float((tr["fat"] * alive).sum() / max(1, alive.sum())),
            "bad_taste": float((tr["bad"] & alive).sum() / max(1, alive.sum())),
            **{k2: float(np.mean([float(x[k2]) for x in auxes])) for k2 in auxes[0]},
        }
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if u % 25 == 0 or u == 1:
            print(json.dumps({k2: rec[k2] for k2 in ("update", "t", "reward", "alive_share", "deaths", "levels", "fat", "bad_taste", "entropy")}), flush=True)
        if u % 100 == 0 or u == a.updates:
            pickle.dump(jax.tree.map(np.asarray, nnx.to_pure_dict(nnx.state(policy))), open(os.path.join(a.out, "ppo.pkl"), "wb"))


if __name__ == "__main__":
    main()
