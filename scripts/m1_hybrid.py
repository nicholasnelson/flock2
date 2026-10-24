#!/usr/bin/env python
"""The sanity ladder's next rung (Milestone 1): the world model learns from birth exactly as in
``m1_agent.py``, but the policy learns from *real* experience — PPO on the world model's state
(h, z) — instead of from imagined rollouts. Real rewards cannot be exploited the way the world
model's errors were (sheep drinking on dry ground because the model believed it worked).

    python scripts/m1_hybrid.py --out ~/flock2_data/m1_hybrid/s0 --seed 0

Per iteration: every sheep in ``--meadows`` meadows acts for ``--steps`` steps from its own
world-model state (the per-sheep posterior filter, as in ``m1_agent.py``), sampling from the PPO
policy; the logged sheep's last 64 steps go to the world model's replay; ``--wm_steps``
world-model updates; then (after ``--ppo_start`` iterations of world-model-only infancy, in which
the near-uniform initial policy acts but is not trained) PPO epochs on the collected
(state, action, reward) samples, the state taken as fixed input (no gradient into the world
model). Rewards are the simulator's (``drives.reward``); no curiosity. GAE bootstraps the last
step with the value of the last state seen (the state after it is not computed). Meadows start as
``world.random_start`` and restart every ``--reset_every`` steps, staggered, between rollouts.
Checkpoints: ``wm.pkl`` and ``policy.pkl``; scored by ``m1_eval.py --policy hybrid``.
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
from flock.meadow import drives as D, sim
from flock.meadow.rollouts import pack
from flock.meadow.views import view_one
from flock.meadow.world import ground_state, random_start


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(__file__), f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_wmt = _load("m1_wm_train")
_ppo = _load("m1_ppo")
_agent = _load("m1_agent")


def hybrid_step(world, policy, state, mem, key, rec_idx):
    """One step for one meadow: every sheep sees, updates its model state, acts from the policy."""
    h, z, a_prev = mem
    k1, k2 = jax.random.split(key)
    n = state.flock.pos.shape[0]
    tiles = ground_state(state.meadow)
    views = jax.vmap(lambda i: view_one(state, i, tiles))(jnp.arange(n))
    feel = D.felt(state.flock.levels, state.flock.taste)
    embed = world.encoder(jax.nn.one_hot(views, 16)) + world.intero_in(feel)
    h = world.transition(h, z, jax.nn.one_hot(a_prev, wm.N_ACTIONS))
    z = wm._sample(world.posterior(h, embed), k1)
    feat = jax.lax.stop_gradient(ac.feat_of(h, z))
    logits, value = policy(feat)
    actions = jax.random.categorical(k2, logits, -1)
    logp = jax.nn.log_softmax(logits)[jnp.arange(n), actions]
    new = sim.step(state, actions)
    r, c = sim._tile(state.meadow, state.flock.pos)
    rec = dict(
        view=views[rec_idx].astype(jnp.uint8), action=actions[rec_idx], levels=feel[rec_idx], reward=new.flock.reward[rec_idx],
        feat=feat, all_actions=actions, logp=logp, value=value, all_reward=new.flock.reward, all_levels=new.flock.levels,
        night=jnp.broadcast_to(sim.is_night(state.t), actions.shape),
        dry_drink=(actions == sim.DRINK) & ~sim.near_water(state.meadow, state.flock.pos),
        bare_graze=(actions == sim.GRAZE) & (state.meadow.grass[r, c] < 1.0),
    )
    return new, (h, z, actions), rec


@nnx.jit(static_argnums=(5,))
def collect(world, policy, states, mems, key, steps, rec_idx):
    def one(carry, k):
        st, mm = carry
        st, mm, rec = jax.vmap(lambda s, m, kk: hybrid_step(world, policy, s, m, kk, rec_idx))(st, mm, jax.random.split(k, st.t.shape[0]))
        return (st, mm), rec

    (states, mems), recs = jax.lax.scan(one, (states, mems), jax.random.split(key, steps))
    return states, mems, recs


@jax.jit
def gae(reward, value, gamma, lam):
    """reward, value (T, B, S) -> advantages (T, B, S); the last step bootstraps with its own value."""

    def body(carry, x):
        adv_next, v_next = carry
        r, v = x
        delta = r + gamma * v_next - v
        adv = delta + gamma * lam * adv_next
        return (adv, v), adv

    _, adv = jax.lax.scan(body, (jnp.zeros_like(value[-1]), value[-1]), (reward, value), reverse=True)
    return adv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.expanduser("~/flock2_data/m1_hybrid/s0"))
    p.add_argument("--iters", type=int, default=600)
    p.add_argument("--meadows", type=int, default=16)
    p.add_argument("--sheep", type=int, default=40)
    p.add_argument("--record", type=int, default=8)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--length", type=int, default=64)
    p.add_argument("--wm_steps", type=int, default=32)
    p.add_argument("--replay", type=int, default=8192)
    p.add_argument("--ppo_start", type=int, default=100)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent", type=float, default=0.01)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--reset_every", type=int, default=1200)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    world = wm.WorldModel(nnx.Rngs(a.seed), n_intero=D.N_FELT, reward=True)
    wm_opt = nnx.Optimizer(world, optax.chain(optax.clip_by_global_norm(1000.0), optax.adam(1e-4, eps=1e-8)), wrt=nnx.Param)
    policy = _ppo.Policy(nnx.Rngs(a.seed + 1), hidden=a.hidden, n_in=ac.FEAT)
    opt = nnx.Optimizer(policy, optax.chain(optax.clip_by_global_norm(0.5), optax.adam(a.lr)), wrt=nnx.Param)
    replay = _agent.Replay(a.replay, a.length)
    n_rec = a.meadows * a.record
    window = _agent.Replay(n_rec, a.length)
    fill = np.zeros(n_rec, int)
    rng = np.random.default_rng(a.seed)
    key = jax.random.PRNGKey(a.seed)
    key, k = jax.random.split(key)
    states = _agent.fresh_many(jax.random.split(k, a.meadows), a.sheep)
    mems = (jnp.zeros((a.meadows, a.sheep, wm.DETER)), jnp.zeros((a.meadows, a.sheep, wm.STOCH * wm.CLASSES)), jnp.zeros((a.meadows, a.sheep), jnp.int32))
    age = np.arange(a.meadows) * (a.reset_every // a.meadows)
    rec_idx = jnp.arange(a.record)
    # compile (and autotune) both update steps before acting, while the GPU is empty, on throwaway
    # copies (XLA's autotuning ran out of memory otherwise)
    w_tmp, o_tmp, p_tmp, q_tmp = nnx.clone(world), nnx.clone(wm_opt), nnx.clone(policy), nnx.clone(opt)
    _wmt.train_step(w_tmp, o_tmp, np.zeros((4, a.length, wm.H, wm.W // 2), np.uint8), np.zeros((4, a.length), np.int32), np.ones((4, a.length, D.N_FELT), np.float32), np.zeros((4, a.length), np.float32), key)
    mb = a.steps * a.meadows * a.sheep // a.minibatches
    _ppo.update(p_tmp, q_tmp, np.zeros((mb, ac.FEAT), np.float32), np.zeros(mb, np.int32), np.zeros(mb, np.float32), np.zeros(mb, np.float32), np.zeros(mb, np.float32), a.clip, a.ent)
    del w_tmp, o_tmp, p_tmp, q_tmp
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    t0 = time.time()
    for it in range(1, a.iters + 1):
        key, k = jax.random.split(key)
        states, mems, recs = collect(world, policy, states, mems, k, a.steps, rec_idx)
        # the world model's replay: (T, B, R, ...) -> (B*R, T, ...)
        T = a.steps
        seq = lambda x: np.asarray(x).swapaxes(0, 1).swapaxes(1, 2).reshape((-1, T) + x.shape[3:])
        new = (pack(seq(recs["view"])), seq(recs["action"]), seq(recs["levels"]), seq(recs["reward"]))
        for buf, x in zip((window.v, window.a, window.l, window.r), new):
            buf[:] = np.concatenate([buf[:, T:], x], 1)[:, -a.length :] if T < a.length else x[:, -a.length :]
        fill += T
        full = fill >= a.length
        if full.any():
            replay.add(window.v[full], window.a[full], window.l[full], window.r[full])
        wm_aux = []
        for _ in range(a.wm_steps if replay.n >= 4 else 0):
            pk, ak, lk, rk = replay.sample(rng, 4)
            key, k = jax.random.split(key)
            wm_aux.append(_wmt.train_step(world, wm_opt, pk, ak, lk, rk, k))
        # PPO on the real experience
        ppo_aux = []
        if it > a.ppo_start:
            adv = gae(recs["all_reward"], recs["value"], a.gamma, a.lam)
            ret = adv + recs["value"]
            flat = lambda x: np.asarray(x).reshape((-1,) + x.shape[3:])
            obs, act_, lp, ad, rt = flat(recs["feat"]), flat(recs["all_actions"]), flat(recs["logp"]), flat(adv), flat(ret)
            N = act_.shape[0]
            for _ in range(a.epochs):
                for mb in np.array_split(rng.permutation(N), a.minibatches):
                    ppo_aux.append(_ppo.update(policy, opt, obs[mb], act_[mb], lp[mb], ad[mb], rt[mb], a.clip, a.ent))
        # restarts, between rollouts
        age += T
        reset = age >= a.reset_every
        if reset.any():
            key, k = jax.random.split(key)
            fresh = _agent.fresh_many(jax.random.split(k, a.meadows), a.sheep)
            m = jnp.asarray(reset)
            pick = lambda old, nw: jnp.where(m.reshape((-1,) + (1,) * (old.ndim - 1)), nw, old)
            states = jax.tree.map(pick, states, fresh)
            mems = jax.tree.map(lambda old: pick(old, jnp.zeros_like(old)), mems)
            age[reset] = 0
            fill[np.repeat(reset, a.record)] = 0
        acts = np.asarray(recs["all_actions"])
        night = np.asarray(recs["night"])
        frac = lambda msk: {sim.ACTION_NAMES[i]: float((acts[msk] == i).mean()) for i in range(sim.N_ACTIONS)} if msk.any() else None
        dd = np.asarray(recs["dry_drink"])
        rec = {
            "iter": it,
            "t": time.time() - t0,
            "reward": float(np.asarray(recs["all_reward"]).mean()),
            "levels": np.round(np.asarray(recs["all_levels"]).mean((0, 1, 2)), 3).tolist(),
            "night_frac": float(night.mean()),
            "actions_day": frac(~night),
            "actions_night": frac(night),
            "dry_drink": float(dd.mean()),
            "bare_graze": float(np.asarray(recs["bare_graze"]).mean()),
            "drink_dry_share": float(dd.sum() / max(1, (acts == sim.DRINK).sum())),
            **({f"wm_{k2}": float(np.mean([float(x[k2]) for x in wm_aux])) for k2 in wm_aux[0]} if wm_aux else {}),
            **({f"ppo_{k2}": float(np.mean([float(x[k2]) for x in ppo_aux])) for k2 in ppo_aux[0]} if ppo_aux else {}),
        }
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if it % 10 == 0 or it == 1:
            print(json.dumps({k2: rec.get(k2) for k2 in ("iter", "t", "reward", "levels", "drink_dry_share", "bare_graze", "ppo_entropy", "ppo_approx_kl", "wm_pixel_acc", "wm_reward_ce")}), flush=True)
        if it % 25 == 0 or it == a.iters:
            pickle.dump(jax.tree.map(np.asarray, nnx.to_pure_dict(nnx.state(world))), open(os.path.join(a.out, "wm.pkl"), "wb"))
            pickle.dump(jax.tree.map(np.asarray, nnx.to_pure_dict(nnx.state(policy))), open(os.path.join(a.out, "policy.pkl"), "wb"))


if __name__ == "__main__":
    main()
