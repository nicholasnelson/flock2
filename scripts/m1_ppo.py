#!/usr/bin/env python
"""The sanity ladder's first rung (Milestone 1; review of 2026-10-24, § 4.1): is the meadow
learnable at all, and what does a competent sheep score? A small policy on privileged features
(``flock.meadow.privileged``: levels, time of day, egocentric vectors to the nearest water, grass
and cover) trained model-free with PPO directly in the simulator, on the sheep's own reward
(``drives.reward``). No world model, no imagination, no curiosity.

    python scripts/m1_ppo.py --out ~/flock2_data/m1_ppo/run1 --updates 300

PPO (Schulman et al. 2017, arXiv:1707.06347) with GAE (Schulman et al. 2015, arXiv:1506.02438):
each update collects ``--steps`` steps from ``--meadows`` meadows of 40 sheep (one shared policy;
every sheep is a sample), then ``--epochs`` passes of clipped-surrogate updates over
``--minibatches`` minibatches; advantages normalised per batch. Meadows start as the agent's
varied starts (``world.random_start``) and are restarted every ``--reset_every`` steps,
staggered. The checkpoint (``ppo.pkl``) is scored by ``m1_eval.py --policy ppo``.
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
import optax
from flax import nnx

from flock.meadow import drives as D, privileged as P, sim
from flock.meadow.world import random_start


class Policy(nnx.Module):
    """Separate actor and critic MLPs (tanh) on an observation of ``n_in`` numbers: the privileged
    features here, the world model's state (h, z) in ``m1_hybrid.py``."""

    def __init__(self, rngs: nnx.Rngs, hidden: int = 128, n_in: int = P.N_OBS):
        self.pi = nnx.Sequential(nnx.Linear(n_in, hidden, rngs=rngs), jnp.tanh, nnx.Linear(hidden, hidden, rngs=rngs), jnp.tanh, nnx.Linear(hidden, sim.N_ACTIONS, rngs=rngs, kernel_init=nnx.initializers.normal(0.01)))
        self.v = nnx.Sequential(nnx.Linear(n_in, hidden, rngs=rngs), jnp.tanh, nnx.Linear(hidden, hidden, rngs=rngs), jnp.tanh, nnx.Linear(hidden, 1, rngs=rngs))

    def __call__(self, obs):
        return self.pi(obs), self.v(obs)[..., 0]


def act(policy, state, key):
    """Every sheep's action from the privileged policy (sampled)."""
    logits, _ = policy(P.observe(state))
    return jax.random.categorical(key, logits, -1)


@nnx.jit(static_argnums=(4,))
def rollout(policy, states, key, gamma_lam, steps):
    gamma, lam = gamma_lam

    def one(st, k):
        obs = P.observe(st)
        logits, value = policy(obs)
        a = jax.random.categorical(k, logits, -1)
        logp = jax.nn.log_softmax(logits)[jnp.arange(a.shape[0]), a]
        new = sim.step(st, a)
        return new, dict(obs=obs, act=a, logp=logp, value=value, reward=new.flock.reward, levels=new.flock.levels, night=jnp.broadcast_to(sim.is_night(st.t), a.shape))

    def run(st, k):
        return jax.lax.scan(one, st, jax.random.split(k, steps))

    states, tr = jax.vmap(run)(states, jax.random.split(key, states.t.shape[0]))  # tr: (B, T, S, ...)
    _, last_v = policy(jax.vmap(P.observe)(states))  # (B, S)

    def gae(carry, x):
        adv_next, v_next = carry
        r, v = x
        delta = r + gamma * v_next - v
        adv = delta + gamma * lam * adv_next
        return (adv, v), adv

    def per_meadow(r, v, lv):
        _, adv = jax.lax.scan(gae, (jnp.zeros_like(lv), lv), (r, v), reverse=True)
        return adv

    adv = jax.vmap(per_meadow)(tr["reward"], tr["value"], last_v)
    tr["adv"] = adv
    tr["ret"] = adv + tr["value"]
    return states, tr


@nnx.jit
def update(policy, opt, obs, act_, logp_old, adv, ret, clip, ent_coef):
    def loss_fn(p):
        logits, v = p(obs)
        logp_all = jax.nn.log_softmax(logits)
        logp = jnp.take_along_axis(logp_all, act_[:, None], -1)[:, 0]
        ratio = jnp.exp(logp - logp_old)
        a = (adv - adv.mean()) / (adv.std() + 1e-8)
        pg = -jnp.mean(jnp.minimum(ratio * a, jnp.clip(ratio, 1 - clip, 1 + clip) * a))
        vf = 0.5 * jnp.mean((v - ret) ** 2)
        ent = -jnp.mean(jnp.sum(jnp.exp(logp_all) * logp_all, -1))
        return pg + 0.5 * vf - ent_coef * ent, {"pg": pg, "vf": vf, "entropy": ent, "approx_kl": jnp.mean(logp_old - logp)}

    (loss, aux), g = nnx.value_and_grad(loss_fn, has_aux=True)(policy)
    opt.update(policy, g)
    return aux


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.expanduser("~/flock2_data/m1_ppo/run1"))
    p.add_argument("--updates", type=int, default=300)
    p.add_argument("--meadows", type=int, default=16)
    p.add_argument("--sheep", type=int, default=40)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent", type=float, default=0.01)
    p.add_argument("--reset_every", type=int, default=2400)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    policy = Policy(nnx.Rngs(a.seed))
    opt = nnx.Optimizer(policy, optax.chain(optax.clip_by_global_norm(0.5), optax.adam(a.lr)), wrt=nnx.Param)
    fresh = jax.jit(jax.vmap(random_start, in_axes=(0, None)), static_argnums=1)
    key = jax.random.PRNGKey(a.seed)
    key, k = jax.random.split(key)
    states = fresh(jax.random.split(k, a.meadows), a.sheep)
    age = np.arange(a.meadows) * (a.reset_every // a.meadows)
    rng = np.random.default_rng(a.seed)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    t0 = time.time()
    for u in range(1, a.updates + 1):
        key, k = jax.random.split(key)
        states, tr = rollout(policy, states, k, (a.gamma, a.lam), a.steps)
        flat = {n: np.asarray(tr[n]).reshape((-1,) + tr[n].shape[3:]) for n in ("obs", "act", "logp", "adv", "ret")}
        N = flat["act"].shape[0]
        auxes = []
        for _ in range(a.epochs):
            perm = rng.permutation(N)
            for mb in np.array_split(perm, a.minibatches):
                auxes.append(update(policy, opt, flat["obs"][mb], flat["act"][mb], flat["logp"][mb], flat["adv"][mb], flat["ret"][mb], a.clip, a.ent))
        age += a.steps
        reset = age >= a.reset_every
        if reset.any():
            key, k = jax.random.split(key)
            new = fresh(jax.random.split(k, a.meadows), a.sheep)
            m = jnp.asarray(reset)
            states = jax.tree.map(lambda o, n: jnp.where(m.reshape((-1,) + (1,) * (o.ndim - 1)), n, o), states, new)
            age[reset] = 0
        acts = np.asarray(tr["act"])
        night = np.asarray(tr["night"])
        rec = {
            "update": u,
            "t": time.time() - t0,
            "reward": float(np.asarray(tr["reward"]).mean()),
            "levels": np.asarray(tr["levels"]).mean((0, 1, 2)).round(3).tolist(),
            "actions_day": {sim.ACTION_NAMES[i]: round(float((acts[~night] == i).mean()), 3) for i in range(sim.N_ACTIONS)},
            "actions_night": {sim.ACTION_NAMES[i]: round(float((acts[night] == i).mean()), 3) for i in range(sim.N_ACTIONS)} if night.any() else None,
            **{k2: float(np.mean([float(x[k2]) for x in auxes])) for k2 in auxes[0]},
        }
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if u % 10 == 0 or u == 1:
            print(json.dumps({k2: rec[k2] for k2 in ("update", "t", "reward", "levels", "entropy", "approx_kl")}), flush=True)
        if u % 25 == 0 or u == a.updates:
            pickle.dump(jax.tree.map(np.asarray, nnx.to_pure_dict(nnx.state(policy))), open(os.path.join(a.out, "ppo.pkl"), "wb"))


if __name__ == "__main__":
    main()
