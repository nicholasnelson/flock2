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
from flock.grid.brain import GRID


def step_one(world, agent, s, mem, key):
    """One step for one world: the sheep sees, updates its model state, acts."""
    h, z, a_prev = mem
    k1, k2 = jax.random.split(key)
    v = Wd.view(s)
    feel = Wd.felt(s)
    embed = world.encoder(jax.nn.one_hot(v, Wd.N_CLASSES)) + world.intero_in(feel)
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


@nnx.jit
def wm_step(world, opt, views, actions, felt, reward, key):
    (loss, aux), g = nnx.value_and_grad(wm.loss_fn, has_aux=True)(world, views, actions, key, felt, reward)
    opt.update(world, g)
    return aux


@nnx.jit
def start_states(world, views, actions, felt, key):
    s = world.observe(views, actions, key, felt)
    return s.h.reshape(-1, s.h.shape[-1]), s.z.reshape(-1, s.z.shape[-1])


fresh = jax.jit(jax.vmap(lambda k: Wd.generate(k)))


class Replay:
    def __init__(self, cap, length):
        self.v = np.zeros((cap, length, Wd.V, Wd.V), np.uint8)
        self.a = np.zeros((cap, length), np.int32)
        self.f = np.zeros((cap, length, Wd.N_FELT), np.float32)
        self.r = np.zeros((cap, length), np.float32)
        self.n, self.i, self.cap = 0, 0, cap

    def add(self, v, a, f, r):
        idx = (self.i + np.arange(len(v))) % self.cap
        self.v[idx], self.a[idx], self.f[idx], self.r[idx] = v, a, f, r
        self.i = (self.i + len(v)) % self.cap
        self.n = min(self.n + len(v), self.cap)

    def sample(self, rng, B):
        idx = rng.integers(self.n, size=B)
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
    p.add_argument("--norm", choices=("adv", "ret"), default="adv")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "config.json"), "w"), indent=1)

    world = wm.WorldModel(nnx.Rngs(a.seed), n_intero=Wd.N_FELT, reward=True, ensemble=a.ensemble, cfg=GRID)
    wm_opt = nnx.Optimizer(world, optax.chain(optax.clip_by_global_norm(1000.0), optax.adam(1e-4, eps=1e-8)), wrt=nnx.Param)
    agent = ac.Agent(nnx.Rngs(a.seed + 1), curiosity=a.curiosity, center=not a.no_center, twohot=a.twohot, cfg=GRID, entropy=a.entropy, norm=a.norm)
    ac_opt = ac.make_optimizer(agent, a.ac_lr)
    replay = Replay(a.replay, a.length)
    window = Replay(a.worlds, a.length)
    fill = np.zeros(a.worlds, int)
    rng = np.random.default_rng(a.seed)
    key = jax.random.PRNGKey(a.seed)
    key, k = jax.random.split(key)
    states = fresh(jax.random.split(k, a.worlds))
    mems = (jnp.zeros((a.worlds, GRID.deter)), jnp.zeros((a.worlds, GRID.stoch * GRID.classes)), jnp.zeros(a.worlds, jnp.int32))
    age = np.arange(a.worlds) * (a.reset_every // a.worlds) if a.reset_every >= a.worlds else rng.integers(0, a.reset_every, a.worlds)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    t0 = time.time()
    for it in range(1, a.iters + 1):
        key, k = jax.random.split(key)
        states, mems, recs = collect(world, agent, states, mems, k, a.collect)
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
        wm_aux, ac_aux = [], []
        for _ in range(a.wm_steps):
            v, ac_, f, r = replay.sample(rng, a.batch)
            key, k = jax.random.split(key)
            wm_aux.append(wm_step(world, wm_opt, v, ac_, f, r, k))
        for _ in range(a.ac_steps if it > a.ac_start else 0):
            v, ac_, f, _ = replay.sample(rng, a.batch)
            key, k1, k2 = jax.random.split(key, 3)
            h0, z0 = start_states(world, v, ac_, f, k1)
            ac_aux.append(ac.train_step(world, agent, ac_opt, h0, z0, k2))
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
            **{f"wm_{k2}": float(np.mean([float(x[k2]) for x in wm_aux])) for k2 in wm_aux[0]},
            **({f"ac_{k2}": float(np.mean([float(x[k2]) for x in ac_aux])) for k2 in ac_aux[0]} if ac_aux else {}),
        }
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if it % 25 == 0:
            print(json.dumps({k2: rec.get(k2) for k2 in ("iter", "t", "reward", "levels", "eat_success", "drink_success", "bad_taste", "ac_entropy", "ac_return", "ac_abs_adv", "wm_pixel_acc")}), flush=True)
        if it % 50 == 0 or it == a.iters:
            pickle.dump(jax.tree.map(np.asarray, nnx.to_pure_dict(nnx.state(world))), open(os.path.join(a.out, "wm.pkl"), "wb"))
            pickle.dump(jax.tree.map(np.asarray, nnx.to_pure_dict(nnx.state(agent))), open(os.path.join(a.out, "agent.pkl"), "wb"))


if __name__ == "__main__":
    main()
