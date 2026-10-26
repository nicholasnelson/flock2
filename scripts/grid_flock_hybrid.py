#!/usr/bin/env python
"""Track B (``docs/flock-2/trackB-grid-communication.md``): the grid hybrid with 4 sheep per world
and a signal channel.

    python scripts/grid_flock_hybrid.py --out ~/flock2_data/grid_flock/learned_s0 --signal learned --seed 0

Each sheep: a world model (``flock.wm``, grid size) learns from its own 8 x 8 view (other sheep
visible) and its felt numbers, which include what it hears (``flock.grid.flock.felt``). PPO on the
model's state picks a move (7 actions) and a signal: 2 numbers, sampled from a Gaussian with learned
spread and squashed to [-1, 1] by tanh. One policy and one world model are shared by all sheep.

``--signal``:
- ``learned``: the medium only, no extra losses (condition 1).
- ``off``: nobody hears anything (condition 2, following alone).
- ``beacon``: each sheep's signal is (1, 0) while it drinks and (0, 0) otherwise; listeners learn
  (condition 3).

PPO as ``grid_hybrid.py`` (clipped, GAE, rewards centred by a running mean: the grid has no death).
For ``learned``, the PPO ratio covers the move and the signal together (the tanh correction cancels
in the ratio, so the Gaussian log-probability of the unsquashed sample is used). Checkpoints
``wm.pkl``, ``policy.pkl``, ``config.json``.
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

from flock import ac, wm
from flock.grid import brain, flock as F, world as W1


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(__file__), f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_grid = _load("grid_hybrid")  # wm_burst
_agent = _load("grid_agent")  # Replay
_ppo_true = _load("grid_ppo_true")

CFG = brain.GRID._replace(n_classes=F.N_CLASSES)


class SigPolicy(nnx.Module):
    """Separate actor and critic MLPs (tanh). The actor gives move logits and the signal's mean; the
    signal's log spread is a learned parameter."""

    def __init__(self, rngs: nnx.Rngs, n_in: int, hidden: int = 256):
        self.pi = nnx.Sequential(nnx.Linear(n_in, hidden, rngs=rngs), jnp.tanh, nnx.Linear(hidden, hidden, rngs=rngs), jnp.tanh)
        self.move = nnx.Linear(hidden, F.N_ACTIONS, rngs=rngs, kernel_init=nnx.initializers.normal(0.01))
        self.mu = nnx.Linear(hidden, F.N_SIGNAL, rngs=rngs, kernel_init=nnx.initializers.normal(0.01))
        self.log_std = nnx.Param(jnp.full((F.N_SIGNAL,), -0.5))
        self.v = nnx.Sequential(nnx.Linear(n_in, hidden, rngs=rngs), jnp.tanh, nnx.Linear(hidden, hidden, rngs=rngs), jnp.tanh, nnx.Linear(hidden, 1, rngs=rngs))

    def __call__(self, x):
        h = self.pi(x)
        return self.move(h), self.mu(h), self.log_std[...], self.v(x)[..., 0]


def gauss_logp(u, mu, log_std):
    return jnp.sum(-0.5 * ((u - mu) / jnp.exp(log_std)) ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi), -1)


def make_world(seed):
    return wm.WorldModel(nnx.Rngs(seed), n_intero=F.N_FELT, reward=True, cfg=CFG)


def emit(mode, u, drank):
    if mode == "learned":
        return jnp.tanh(u)
    if mode == "beacon":
        return jnp.where(drank[..., None], jnp.array([1.0, 0.0]), 0.0)
    return jnp.zeros_like(u)


def with_heard(s, mode):
    return s if mode != "off" else s._replace(heard=jnp.zeros_like(s.heard))


def sheep_step(world, policy, states, mems, key, mode):
    """Every sheep in a batch of worlds sees, updates its model state, moves and signals."""
    h, z, a_prev = mems  # (B, K, ...)
    k1, k2, k3 = jax.random.split(key, 3)
    states = jax.vmap(lambda s: with_heard(s, mode))(states)
    views = jax.vmap(F.views)(states)  # (B, K, V, V)
    feel = jax.vmap(F.felt)(states)
    embed = world.encoder(jax.nn.one_hot(views, CFG.n_classes)) + world.intero_in(feel)
    h = world.transition(h, z, jax.nn.one_hot(a_prev, CFG.n_actions))
    z = wm._sample(world.posterior(h, embed), k1)
    feat = jax.lax.stop_gradient(ac.feat_of(h, z))
    logits, mu, log_std, value = policy(feat)
    a = jax.random.categorical(k2, logits, -1)
    u = mu + jnp.exp(log_std) * jax.random.normal(k3, mu.shape)
    logp = jnp.take_along_axis(jax.nn.log_softmax(logits), a[..., None], -1)[..., 0]
    if mode == "learned":
        logp = logp + gauss_logp(u, mu, log_std)
    near = jax.vmap(lambda s: F.near_water(s.water, s.pos))(states)
    drank = (a == W1.DRINK) & near
    sig = emit(mode, u, drank)
    new = jax.vmap(F.step)(states, a, sig)
    sees = jax.vmap(F.sees_water)(states)
    rec = dict(views=views.astype(jnp.uint8), feel=feel, feat=feat, act=a, u=u, logp=logp, value=value, reward=new.reward, levels=new.levels,
               drank=drank, sees=sees, thirsty=states.levels[..., 1] < 0.5, signal=sig, bad=new.taste < 0)
    return new, (h, z, a), rec


@nnx.jit(static_argnums=(5, 6))
def collect(world, policy, states, mems, key, steps, mode):
    def one(carry, k):
        st, mm = carry
        st, mm, rec = sheep_step(world, policy, st, mm, k, mode)
        return (st, mm), rec

    (states, mems), tr = jax.lax.scan(one, (states, mems), jax.random.split(key, steps))
    _, _, _, last_v = policy(jax.lax.stop_gradient(ac.feat_of(*sheep_step(world, policy, states, mems, jax.random.fold_in(key, 9), mode)[1][:2])))
    return states, mems, tr, last_v


@nnx.jit
def update(policy, opt, obs, act, u, logp_old, adv, ret, clip, ent_coef, learned):
    def loss_fn(p):
        logits, mu, log_std, v = p(obs)
        lp_all = jax.nn.log_softmax(logits)
        logp = jnp.take_along_axis(lp_all, act[:, None], -1)[:, 0] + learned * gauss_logp(u, mu, log_std)
        ratio = jnp.exp(logp - logp_old)
        a = (adv - adv.mean()) / (adv.std() + 1e-8)
        pg = -jnp.mean(jnp.minimum(ratio * a, jnp.clip(ratio, 1 - clip, 1 + clip) * a))
        vf = 0.5 * jnp.mean((v - ret) ** 2)
        ent = -jnp.mean(jnp.sum(jnp.exp(lp_all) * lp_all, -1)) + learned * jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e))
        return pg + 0.5 * vf - ent_coef * ent, {"pg": pg, "vf": vf, "entropy": ent, "approx_kl": jnp.mean(logp_old - logp), "sig_std": jnp.mean(jnp.exp(log_std))}

    (_, aux), g = nnx.value_and_grad(loss_fn, has_aux=True)(policy)
    opt.update(policy, g)
    return aux


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.expanduser("~/flock2_data/grid_flock/learned_s0"))
    p.add_argument("--signal", choices=("learned", "off", "beacon"), default="learned")
    p.add_argument("--iters", type=int, default=800)
    p.add_argument("--worlds", type=int, default=64)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--length", type=int, default=32)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--wm_steps", type=int, default=32)
    p.add_argument("--replay", type=int, default=16384)
    p.add_argument("--ppo_start", type=int, default=50)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.98)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent", type=float, default=0.01)
    p.add_argument("--reset_every", type=int, default=384)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    assert a.steps % a.length == 0 and a.reset_every % a.steps == 0
    os.makedirs(a.out, exist_ok=True)
    json.dump({**vars(a), "kind": "grid_flock"}, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    world = make_world(a.seed)
    wm_opt = nnx.Optimizer(world, optax.chain(optax.clip_by_global_norm(1000.0), optax.adam(1e-4, eps=1e-8)), wrt=nnx.Param)
    policy = SigPolicy(nnx.Rngs(a.seed + 1), CFG.feat)
    opt = nnx.Optimizer(policy, optax.chain(optax.clip_by_global_norm(0.5), optax.adam(a.lr)), wrt=nnx.Param)
    replay = _agent.Replay(a.replay, a.length, F.V)
    replay.f = np.zeros((a.replay, a.length, F.N_FELT), np.float32)
    rng = np.random.default_rng(a.seed)
    key = jax.random.PRNGKey(a.seed)
    fresh = jax.jit(jax.vmap(F.generate))
    key, k = jax.random.split(key)
    states = fresh(jax.random.split(k, a.worlds))
    zero = lambda: (jnp.zeros((a.worlds, F.K, CFG.deter)), jnp.zeros((a.worlds, F.K, CFG.stoch * CFG.classes)), jnp.zeros((a.worlds, F.K), jnp.int32))
    mems = zero()
    age = (np.arange(a.worlds) % (a.reset_every // a.steps)) * a.steps
    r_bar = None
    learned = jnp.float32(a.signal == "learned")
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    t0 = time.time()
    for it in range(1, a.iters + 1):
        key, k = jax.random.split(key)
        states, mems, tr, last_v = collect(world, policy, states, mems, k, a.steps, a.signal)
        tr = jax.tree.map(np.asarray, tr)
        last_v = np.asarray(last_v)
        T = a.steps
        cut = lambda x: x.transpose((1, 2, 0) + tuple(range(3, x.ndim))).reshape((-1, T // a.length, a.length) + x.shape[3:]).reshape((-1, a.length) + x.shape[3:])
        replay.add(cut(tr["views"]), cut(tr["act"]), cut(tr["feel"]), cut(tr["reward"]))
        V_, A_, F_, R_ = replay.sample(rng, a.batch, a.wm_steps)
        key, k = jax.random.split(key)
        wm_aux = {k2: float(np.mean(np.asarray(v))) for k2, v in _grid.wm_burst(world, wm_opt, V_, A_, F_, R_, jax.random.split(k, a.wm_steps)).items()}
        ppo_aux = {}
        if it > a.ppo_start:
            r_mean = float(tr["reward"].mean())
            r_bar = r_mean if r_bar is None else 0.95 * r_bar + 0.05 * r_mean
            adv = np.asarray(_ppo_true.gae(tr["reward"] - r_bar, tr["value"], last_v, a.gamma, a.lam))
            ret = adv + tr["value"]
            flat = lambda x: np.asarray(x).reshape((-1,) + x.shape[3:])
            obs, act, u, lp, ad, rt = flat(tr["feat"]), flat(tr["act"]), flat(tr["u"]), flat(tr["logp"]), flat(adv), flat(ret)
            auxes = []
            for _ in range(a.epochs):
                for mb in np.array_split(rng.permutation(len(act)), a.minibatches):
                    auxes.append(update(policy, opt, obs[mb], act[mb], u[mb], lp[mb], ad[mb], rt[mb], a.clip, a.ent, learned))
            ppo_aux = {k2: float(np.mean([float(x[k2]) for x in auxes])) for k2 in auxes[0]}
        age += a.steps
        reset = age >= a.reset_every
        if reset.any():
            key, k = jax.random.split(key)
            nw = fresh(jax.random.split(k, a.worlds))
            m = jnp.asarray(reset)
            pick = lambda o, n: jnp.where(m.reshape((-1,) + (1,) * (o.ndim - 1)), n, o)
            states = jax.tree.map(pick, states, nw)
            mems = jax.tree.map(pick, mems, zero())
            age[reset] = 0
        sig = tr["signal"]
        drk = tr["drank"]
        rec = {
            "iter": it,
            "t": time.time() - t0,
            "reward": float(tr["reward"].mean()),
            "levels": tr["levels"].mean((0, 1, 2)).round(3).tolist(),
            "drink_success": float(drk.sum() / max(1, (tr["act"] == W1.DRINK).sum())),
            "bad_taste": float(tr["bad"].mean()),
            "thirsty_blind_share": float((tr["thirsty"] & ~tr["sees"]).mean()),
            "signal_abs": float(np.abs(sig).mean()),
            "signal_when_drinking": np.round(sig[drk].mean(0), 3).tolist() if drk.any() else None,
            "signal_otherwise": np.round(sig[~drk].mean(0), 3).tolist(),
            **{f"wm_{k2}": v for k2, v in wm_aux.items()},
            **{f"ppo_{k2}": v for k2, v in ppo_aux.items()},
        }
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if it % 25 == 0 or it == 1:
            print(json.dumps({k2: rec.get(k2) for k2 in ("iter", "t", "reward", "levels", "drink_success", "thirsty_blind_share", "signal_abs", "signal_when_drinking", "signal_otherwise", "ppo_entropy", "ppo_sig_std")}), flush=True)
        if it % 50 == 0 or it == a.iters:
            pickle.dump(jax.tree.map(np.asarray, nnx.to_pure_dict(nnx.state(world))), open(os.path.join(a.out, "wm.pkl"), "wb"))
            pickle.dump(jax.tree.map(np.asarray, nnx.to_pure_dict(nnx.state(policy))), open(os.path.join(a.out, "policy.pkl"), "wb"))


if __name__ == "__main__":
    main()
