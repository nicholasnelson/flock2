#!/usr/bin/env python
"""Train a sheep in the grid world from birth (``flock.grid``) with the meadow's own learning
machinery: the world model (``flock.wm``, ``GRID`` sizes) learns from the sheep's experience,
and the actor-critic (``flock.ac``) learns in imagination, optionally with body curiosity.

    python scripts/grid_agent.py --out ~/flock2_data/grid/run1 --iters 400

Per iteration: ``--worlds`` worlds (one sheep each) act for ``--collect`` steps from each sheep's
own model state; each sheep's last ``--length`` steps go to the replay; ``--wm_steps`` world-model
updates on batches of ``--batch`` sequences; then (after ``--ac_start`` iterations of infancy, in
which the untrained, near-uniform actor acts) ``--ac_steps`` actor-critic updates from model
states inferred on replay sequences. Worlds restart (new layout, levels from [0.2, 1.0]) every
``--reset_every`` steps, staggered; no sequence spans a restart. Checkpoints ``wm.pkl``,
``agent.pkl`` and ``config.json``; scored by ``scripts/grid_eval.py``.

Since the third review of 2026-10-24: ``--gamma``, ``--slowtar``, ``--norm`` (default "ret"),
``--critic_lr`` and ``--repval`` (the replay value loss on real replayed rewards, on by default)
set the actor-critic as in ``flock.ac``; ``--view full`` shows the whole map instead of the 8x8
window; each iteration's updates run as one compiled call (``burst``), and the log records the
time spent collecting and training.
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

from flock import ac, wm
from flock.grid import world as Wd
from flock.grid import brain


def step_one(world, agent, s, mem, key):
    """One step for one world: the sheep sees, updates its model state, acts."""
    h, z, a_prev = mem
    k1, k2 = jax.random.split(key)
    v = Wd.observe(s, VIEW)
    feel = Wd.felt(s)
    embed = world.encoder(jax.nn.one_hot(v, world.cfg.n_classes)) + world.intero_in(feel)
    h = world.transition(h, z, jax.nn.one_hot(a_prev, Wd.N_ACTIONS))
    z = wm._sample(world.posterior(h, embed), k1)
    a = agent.act(ac.feat_of(h, z), k2)
    new = Wd.step(s, a)
    r, c = s.pos
    rec = dict(view=v.astype(jnp.uint8), action=a, felt=feel, reward=new.reward, levels=new.levels,
               ate=(a == Wd.EAT) & (s.grass[r, c] >= 1.0), drank=(a == Wd.DRINK) & Wd.near_water(s), bad=new.taste < 0)
    return new, (h, z, a), rec


@nnx.jit(static_argnums=(5,))
def collect(world, agent, states, mems, key, steps):
    def one(carry, k):
        st, mm = carry
        st, mm, rec = jax.vmap(lambda s, m, kk: step_one(world, agent, s, m, kk))(st, mm, jax.random.split(k, st.t.shape[0]))
        return (st, mm), rec

    (states, mems), recs = jax.lax.scan(one, (states, mems), jax.random.split(key, steps))
    return states, mems, recs


VIEW = "local"


@nnx.jit(static_argnums=(10,))
def burst(world, wm_opt, agent, ac_opt, V, A, F, R, wm_keys, ac_keys, do_ac):
    """One iteration's updates in one compiled call: for each of the K minibatches (V, A, F, R
    stacked on a leading axis), a world-model step, then (``do_ac``) an actor-critic step from
    the model states inferred on the next minibatch, with its real rewards for the replay value
    loss. Returns the stacked aux dicts."""
    wm_aux, ac_aux = [], []
    K = V.shape[0]
    for i in range(K):
        (loss, aux), g = nnx.value_and_grad(wm.loss_fn, has_aux=True)(world, V[i], A[i], wm_keys[i], F[i], R[i])
        wm_opt.update(world, g)
        wm_aux.append(aux)
    if do_ac:
        for i in range(K):
            j = (i + 1) % K
            k1, k2 = jax.random.split(ac_keys[i])
            s = world.observe(V[j], A[j], k1, F[j])
            h0, z0 = s.h.reshape(-1, s.h.shape[-1]), s.z.reshape(-1, s.z.shape[-1])
            real = jax.lax.stop_gradient(ac.feat_of(s.h, s.z)).swapaxes(0, 1)  # (L, B, F)
            ac_aux.append(_ac_step(world, agent, ac_opt, h0, z0, k2, real, R[j].swapaxes(0, 1)))
    stack = lambda xs: jax.tree.map(lambda *v: jnp.stack(v), *xs) if xs else {}
    return stack(wm_aux), stack(ac_aux)


def _ac_step(world, agent, opt, h0, z0, key, real_feats, real_rewards):
    feats, acts, extrinsic, dis = ac.imagine(world, agent, h0, z0, key)
    return ac._update(agent, opt, feats, acts, extrinsic, dis, real_feats, real_rewards)


fresh = jax.jit(jax.vmap(lambda k: Wd.generate(k)))


class Replay:
    def __init__(self, cap, length, size=Wd.V):
        self.v = np.zeros((cap, length, size, size), np.uint8)
        self.a = np.zeros((cap, length), np.int32)
        self.f = np.zeros((cap, length, Wd.N_FELT), np.float32)
        self.r = np.zeros((cap, length), np.float32)
        self.n, self.i, self.cap = 0, 0, cap

    def add(self, v, a, f, r):
        idx = (self.i + np.arange(len(v))) % self.cap
        self.v[idx], self.a[idx], self.f[idx], self.r[idx] = v, a, f, r
        self.i = (self.i + len(v)) % self.cap
        self.n = min(self.n + len(v), self.cap)

    def sample(self, rng, B, K=None):
        idx = rng.integers(self.n, size=B if K is None else (K, B))
        return self.v[idx].astype(np.int32), self.a[idx], self.f[idx], self.r[idx]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.expanduser("~/flock2_data/grid/run1"))
    p.add_argument("--iters", type=int, default=400)
    p.add_argument("--worlds", type=int, default=256)
    p.add_argument("--collect", type=int, default=16)
    p.add_argument("--length", type=int, default=32)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--wm_steps", type=int, default=16)
    p.add_argument("--ac_steps", type=int, default=16)
    p.add_argument("--ac_start", type=int, default=50)
    p.add_argument("--replay", type=int, default=16384)
    p.add_argument("--reset_every", type=int, default=400)
    p.add_argument("--ensemble", type=int, default=0)
    p.add_argument("--curiosity", type=float, default=0.0)
    p.add_argument("--no_center", action="store_true")
    p.add_argument("--twohot", action="store_true")
    p.add_argument("--ac_lr", type=float, default=3e-5)
    p.add_argument("--entropy", type=float, default=ac.ENTROPY)
    p.add_argument("--norm", choices=("adv", "ret"), default="ret")
    p.add_argument("--gamma", type=float, default=ac.GAMMA)
    p.add_argument("--slowtar", action="store_true", help="bootstrap from the slow critic (the old ac.py)")
    p.add_argument("--critic_lr", type=float, default=None, help="default: --ac_lr")
    p.add_argument("--repval", type=float, default=ac.REPVAL, help="weight of the replay value loss (0: off)")
    p.add_argument("--view", choices=("local", "full"), default="local")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    global VIEW
    VIEW = a.view
    cfg = brain.config(a.view)
    size = cfg.h
    assert a.ac_steps == a.wm_steps, "updates run in bursts of K world-model + K actor-critic steps"
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "config.json"), "w"), indent=1)

    world = wm.WorldModel(nnx.Rngs(a.seed), n_intero=Wd.N_FELT, reward=True, ensemble=a.ensemble, cfg=cfg)
    wm_opt = nnx.Optimizer(world, optax.chain(optax.clip_by_global_norm(1000.0), optax.adam(1e-4, eps=1e-8)), wrt=nnx.Param)
    agent = ac.Agent(nnx.Rngs(a.seed + 1), curiosity=a.curiosity, center=not a.no_center, twohot=a.twohot, cfg=cfg, entropy=a.entropy, norm=a.norm, gamma=a.gamma, slowtar=a.slowtar, repval=a.repval)
    ac_opt = ac.make_optimizer(agent, a.ac_lr, a.critic_lr)
    replay = Replay(a.replay, a.length, size)
    window = Replay(a.worlds, a.length, size)
    fill = np.zeros(a.worlds, int)
    rng = np.random.default_rng(a.seed)
    key = jax.random.PRNGKey(a.seed)
    key, k = jax.random.split(key)
    states = fresh(jax.random.split(k, a.worlds))
    mems = (jnp.zeros((a.worlds, cfg.deter)), jnp.zeros((a.worlds, cfg.stoch * cfg.classes)), jnp.zeros(a.worlds, jnp.int32))
    age = np.arange(a.worlds) * (a.reset_every // a.worlds) if a.reset_every >= a.worlds else rng.integers(0, a.reset_every, a.worlds)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    t0 = time.time()
    for it in range(1, a.iters + 1):
        t_a = time.time()
        key, k = jax.random.split(key)
        states, mems, recs = collect(world, agent, states, mems, k, a.collect)
        jax.block_until_ready(states)
        t_collect = time.time() - t_a
        T = a.collect
        seq = lambda x: np.asarray(x).swapaxes(0, 1)  # (T, B, ...) -> (B, T, ...)
        new = (seq(recs["view"]), seq(recs["action"]), seq(recs["felt"]), seq(recs["reward"]))
        for buf, x in zip((window.v, window.a, window.f, window.r), new):
            buf[:] = np.concatenate([buf[:, T:], x], 1)[:, -a.length :]
        fill += T
        full = fill >= a.length
        if full.any():
            replay.add(window.v[full], window.a[full], window.f[full], window.r[full])
        age += T
        reset = age >= a.reset_every
        if reset.any():
            key, k = jax.random.split(key)
            nw = fresh(jax.random.split(k, a.worlds))
            m = jnp.asarray(reset)
            pick = lambda o, n: jnp.where(m.reshape((-1,) + (1,) * (o.ndim - 1)), n, o)
            states = jax.tree.map(pick, states, nw)
            mems = jax.tree.map(lambda o: pick(o, jnp.zeros_like(o)), mems)
            age[reset] = 0
            fill[reset] = 0
        if replay.n < a.batch:
            continue
        t_b = time.time()
        V, A, F, R = replay.sample(rng, a.batch, a.wm_steps)
        key, k1, k2 = jax.random.split(key, 3)
        wm_st, ac_st = burst(world, wm_opt, agent, ac_opt, V, A, F, R, jax.random.split(k1, a.wm_steps), jax.random.split(k2, a.wm_steps), it > a.ac_start)
        wm_aux = {k2: float(np.mean(np.asarray(v))) for k2, v in wm_st.items()}
        ac_aux = {k2: float(np.mean(np.asarray(v))) for k2, v in ac_st.items()}
        t_train = time.time() - t_b
        acts = np.asarray(recs["action"])
        nA = max(1, int((acts == Wd.EAT).sum())), max(1, int((acts == Wd.DRINK).sum()))
        rec = {
            "iter": it,
            "t": time.time() - t0,
            "reward": float(np.asarray(recs["reward"]).mean()),
            "levels": np.asarray(recs["levels"]).mean((0, 1)).round(3).tolist(),
            "actions": {Wd.ACTION_NAMES[i]: round(float((acts == i).mean()), 3) for i in range(Wd.N_ACTIONS)},
            "eat_success": float(np.asarray(recs["ate"]).sum() / nA[0]),
            "drink_success": float(np.asarray(recs["drank"]).sum() / nA[1]),
            "bad_taste": float(np.asarray(recs["bad"]).mean()),
            "t_collect": t_collect,
            "t_train": t_train,
            **{f"wm_{k2}": v for k2, v in wm_aux.items()},
            **{f"ac_{k2}": v for k2, v in ac_aux.items()},
        }
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if it % 25 == 0:
            print(json.dumps({k2: rec.get(k2) for k2 in ("iter", "t", "reward", "levels", "eat_success", "drink_success", "bad_taste", "ac_entropy", "ac_return", "ac_value_start", "ac_value_corr", "ac_abs_adv", "wm_pixel_acc", "t_collect", "t_train")}), flush=True)
        if it % 50 == 0 or it == a.iters:
            pickle.dump(jax.tree.map(np.asarray, nnx.to_pure_dict(nnx.state(world))), open(os.path.join(a.out, "wm.pkl"), "wb"))
            pickle.dump(jax.tree.map(np.asarray, nnx.to_pure_dict(nnx.state(agent))), open(os.path.join(a.out, "agent.pkl"), "wb"))


if __name__ == "__main__":
    main()
