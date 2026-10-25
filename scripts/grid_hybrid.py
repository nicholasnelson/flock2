#!/usr/bin/env python
"""The hybrid in the grid world (owner's choice of 2026-10-25): the world model learns from the
sheep's own view exactly as in ``grid_agent.py``, but the policy learns with PPO from *real*
experience, on the world model's state (h, z) — the meadow's ``m1_hybrid.py`` at grid size.

    python scripts/grid_hybrid.py --out ~/flock2_data/grid/hybrid_s0 --seed 0

Per iteration: ``--worlds`` worlds (one sheep each) act for ``--steps`` steps from each sheep's own
model state (the posterior filter, carried across iterations), sampling from the PPO policy; the
rollout, cut into sequences of ``--length``, goes to the world model's replay; ``--wm_steps``
world-model updates on batches of ``--batch`` sequences (one compiled call); then (after
``--ppo_start`` iterations of world-model-only infancy, in which the untrained, near-uniform
policy acts) ``--epochs`` PPO passes over ``--minibatches`` minibatches of the rollout's
(state, action, reward) samples. The state is taken as fixed input: no gradient flows from the
policy into the world model. PPO and GAE as ``grid_ppo_true.py`` (Schulman et al. 2017,
arXiv:1707.06347; 2015, arXiv:1506.02438), rewards centred by a running mean (Naik et al. 2024,
arXiv:2405.09999). GAE bootstraps the last step with the value of the state after it. Worlds
restart every ``--reset_every`` steps (a multiple of ``--steps``), staggered, between rollouts,
with the model state zeroed. Checkpoints ``wm.pkl``, ``policy.pkl`` and ``config.json``; scored
by ``grid_eval.py --policy hybrid``.
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
from flock.grid import brain, world as Wd


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(__file__), f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_ppo = _load("m1_ppo")
_grid = _load("grid_agent")
_ppo_true = _load("grid_ppo_true")


def make_policy(seed, cfg, hidden=256):
    return _ppo.Policy(nnx.Rngs(seed), hidden=hidden, n_in=cfg.feat, n_out=Wd.N_ACTIONS)


def model_step(world, s, mem, key, view):
    """The sheep sees and updates its model state; returns the state's features and the new memory
    without its action (set once the policy has chosen)."""
    h, z, a_prev = mem
    v = Wd.observe(s, view)
    feel = Wd.felt(s)
    embed = world.encoder(jax.nn.one_hot(v, world.cfg.n_classes)) + world.intero_in(feel)
    h = world.transition(h, z, jax.nn.one_hot(a_prev, Wd.N_ACTIONS))
    z = wm._sample(world.posterior(h, embed), key)
    return v, feel, h, z, jax.lax.stop_gradient(ac.feat_of(h, z))


@nnx.jit(static_argnums=(5, 6))
def collect(world, policy, states, mems, key, steps, view):
    def one(carry, k):
        st, (h, z, a_prev) = carry
        k1, k2 = jax.random.split(k)
        v, feel, h, z, feat = jax.vmap(lambda s, m, kk: model_step(world, s, m, kk, view))(st, (h, z, a_prev), jax.random.split(k1, a_prev.shape[0]))
        logits, value = policy(feat)
        a = jax.random.categorical(k2, logits, -1)
        logp = jax.nn.log_softmax(logits)[jnp.arange(a.shape[0]), a]
        st2 = jax.vmap(Wd.step)(st, a)
        r, c = st.pos[:, 0], st.pos[:, 1]
        ate = (a == Wd.EAT) & (st.grass[jnp.arange(a.shape[0]), r, c] >= 1.0)
        drank = (a == Wd.DRINK) & jax.vmap(Wd.near_water)(st)
        rec = dict(view=v.astype(jnp.uint8), felt=feel, feat=feat, action=a, logp=logp, value=value, reward=st2.reward, levels=st2.levels, ate=ate, drank=drank, bad=st2.taste < 0)
        return (st2, (h, z, a)), rec

    (states, mems), recs = jax.lax.scan(one, (states, mems), jax.random.split(key, steps))
    # the value of the state after the last step, for GAE's bootstrap (the model state advanced
    # with the last action and the new view)
    k1 = jax.random.fold_in(key, 1)
    _, _, h, z, feat = jax.vmap(lambda s, m, kk: model_step(world, s, m, kk, view))(states, mems, jax.random.split(k1, states.t.shape[0]))
    _, last_v = policy(feat)
    return states, mems, recs, last_v


@nnx.jit
def wm_burst(world, opt, V, A, F, R, keys):
    """K world-model steps (minibatches stacked on a leading axis) in one compiled call."""
    auxes = []
    for i in range(V.shape[0]):
        (_, aux), g = nnx.value_and_grad(wm.loss_fn, has_aux=True)(world, V[i], A[i], keys[i], F[i], R[i])
        opt.update(world, g)
        auxes.append(aux)
    return jax.tree.map(lambda *v: jnp.stack(v), *auxes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.expanduser("~/flock2_data/grid/hybrid_s0"))
    p.add_argument("--iters", type=int, default=600)
    p.add_argument("--worlds", type=int, default=256)
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
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--reset_every", type=int, default=384)
    p.add_argument("--view", choices=("local", "full"), default="local")
    p.add_argument("--recon_weight", type=float, default=1.0)
    p.add_argument("--reward_weight", type=float, default=1.0)
    p.add_argument("--recon_balance", type=float, default=0.0, help="class-balanced reconstruction cap (0: off)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    assert a.steps % a.length == 0 and a.reset_every % a.steps == 0
    cfg = brain.config(a.view)._replace(recon_weight=a.recon_weight, reward_weight=a.reward_weight, recon_balance=a.recon_balance)
    os.makedirs(a.out, exist_ok=True)
    json.dump({**vars(a), "kind": "hybrid"}, open(os.path.join(a.out, "config.json"), "w"), indent=1)

    world = wm.WorldModel(nnx.Rngs(a.seed), n_intero=Wd.N_FELT, reward=True, cfg=cfg)
    wm_opt = nnx.Optimizer(world, optax.chain(optax.clip_by_global_norm(1000.0), optax.adam(1e-4, eps=1e-8)), wrt=nnx.Param)
    policy = make_policy(a.seed + 1, cfg, a.hidden)
    opt = nnx.Optimizer(policy, optax.chain(optax.clip_by_global_norm(0.5), optax.adam(a.lr)), wrt=nnx.Param)
    replay = _grid.Replay(a.replay, a.length, cfg.h)
    rng = np.random.default_rng(a.seed)
    key = jax.random.PRNGKey(a.seed)
    key, k = jax.random.split(key)
    states = _grid.fresh(jax.random.split(k, a.worlds))
    zero_mems = lambda: (jnp.zeros((a.worlds, cfg.deter)), jnp.zeros((a.worlds, cfg.stoch * cfg.classes)), jnp.zeros(a.worlds, jnp.int32))
    mems = zero_mems()
    # staggered restarts, on rollout boundaries
    age = (np.arange(a.worlds) % (a.reset_every // a.steps)) * a.steps
    r_bar = None
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    t0 = time.time()
    for it in range(1, a.iters + 1):
        t_a = time.time()
        key, k = jax.random.split(key)
        states, mems, recs, last_v = collect(world, policy, states, mems, k, a.steps, a.view)
        recs = jax.tree.map(np.asarray, recs)
        t_collect = time.time() - t_a
        # the world model's replay: (T, B, ...) -> (B * T / L, L, ...); a rollout never spans a
        # restart, but a sequence may start mid-episode (as in grid_agent.py's windows)
        cut = lambda x: x.swapaxes(0, 1).reshape((-1, a.length) + x.shape[2:])
        replay.add(cut(recs["view"]), cut(recs["action"]), cut(recs["felt"]), cut(recs["reward"]))
        t_b = time.time()
        V, A, F, R = replay.sample(rng, a.batch, a.wm_steps)
        key, k = jax.random.split(key)
        wm_aux = {k2: float(np.mean(np.asarray(v))) for k2, v in wm_burst(world, wm_opt, V, A, F, R, jax.random.split(k, a.wm_steps)).items()}
        ppo_aux = {}
        if it > a.ppo_start:
            r_mean = float(recs["reward"].mean())
            r_bar = r_mean if r_bar is None else 0.95 * r_bar + 0.05 * r_mean
            adv = _ppo_true.gae(recs["reward"] - r_bar, recs["value"], last_v, a.gamma, a.lam)
            ret = adv + recs["value"]
            flat = lambda x: np.asarray(x).reshape((-1,) + x.shape[2:])
            obs, act, lp, ad, rt = flat(recs["feat"]), flat(recs["action"]), flat(recs["logp"]), flat(adv), flat(ret)
            auxes = []
            for _ in range(a.epochs):
                for mb in np.array_split(rng.permutation(len(act)), a.minibatches):
                    auxes.append(_ppo.update(policy, opt, obs[mb], act[mb], lp[mb], ad[mb], rt[mb], a.clip, a.ent))
            ppo_aux = {k2: float(np.mean([float(x[k2]) for x in auxes])) for k2 in auxes[0]}
        t_train = time.time() - t_b
        age += a.steps
        reset = age >= a.reset_every
        if reset.any():
            key, k = jax.random.split(key)
            nw = _grid.fresh(jax.random.split(k, a.worlds))
            m = jnp.asarray(reset)
            pick = lambda o, n: jnp.where(m.reshape((-1,) + (1,) * (o.ndim - 1)), n, o)
            states = jax.tree.map(pick, states, nw)
            mems = jax.tree.map(pick, mems, zero_mems())
            age[reset] = 0
        acts = recs["action"]
        nA = max(1, int((acts == Wd.EAT).sum())), max(1, int((acts == Wd.DRINK).sum()))
        rec = {
            "iter": it,
            "t": time.time() - t0,
            "reward": float(recs["reward"].mean()),
            "levels": recs["levels"].mean((0, 1)).round(3).tolist(),
            "actions": {Wd.ACTION_NAMES[i]: round(float((acts == i).mean()), 3) for i in range(Wd.N_ACTIONS)},
            "eat_success": float(recs["ate"].sum() / nA[0]),
            "drink_success": float(recs["drank"].sum() / nA[1]),
            "bad_taste": float(recs["bad"].mean()),
            "t_collect": t_collect,
            "t_train": t_train,
            **{f"wm_{k2}": v for k2, v in wm_aux.items()},
            **{f"ppo_{k2}": v for k2, v in ppo_aux.items()},
        }
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if it % 25 == 0 or it == 1:
            print(json.dumps({k2: rec.get(k2) for k2 in ("iter", "t", "reward", "levels", "eat_success", "drink_success", "bad_taste", "ppo_entropy", "ppo_approx_kl", "wm_pixel_acc", "t_collect", "t_train")}), flush=True)
        if it % 50 == 0 or it == a.iters:
            pickle.dump(jax.tree.map(np.asarray, nnx.to_pure_dict(nnx.state(world))), open(os.path.join(a.out, "wm.pkl"), "wb"))
            pickle.dump(jax.tree.map(np.asarray, nnx.to_pure_dict(nnx.state(policy))), open(os.path.join(a.out, "policy.pkl"), "wb"))


if __name__ == "__main__":
    main()
