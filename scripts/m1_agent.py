#!/usr/bin/env python
"""Train the flock's shared policy online (Milestone 1, step 3): act in the meadows, log to a
replay, fine-tune the world model, train the actor-critic in imagination; repeat.

    python scripts/m1_agent.py --wm ~/flock2_data/m1_wm/run2_intero/model.pkl --out ~/flock2_data/m1_agent/run1 --iters 500

Every sheep in every meadow acts from the actor, from its own world-model state (h, z) updated
each step with its strip and its drive levels. Eight sheep per meadow are logged; each logged
sheep's last ``--length`` steps form a sliding window that goes to the replay every iteration
once full (so sequences overlap by ``--length - --collect`` steps). Per iteration:
``--collect`` steps of acting, then ``--wm_steps`` world-model updates (``--wm_online`` of
each batch of 4 sequences from the online replay, the rest from the scripted shards, so the
model keeps what it learned there) and
``--ac_steps`` actor-critic updates from model states inferred on the online replay.
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
from flock.meadow import drives as D, sim
from flock.meadow.rollouts import pack, unpack
from flock.meadow.views import view_one
from flock.meadow.world import generate, ground_state

import importlib.util

_spec = importlib.util.spec_from_file_location("m1_wm_train", os.path.join(os.path.dirname(__file__), "m1_wm_train.py"))
_wmt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_wmt)


def act_and_step(world, agent, state, mem, key, rec_idx):
    """One step for one meadow: every sheep sees, updates its model state, acts."""
    h, z, a_prev = mem
    k1, k2, k3 = jax.random.split(key, 3)
    n = state.flock.pos.shape[0]
    tiles = ground_state(state.meadow)
    views = jax.vmap(lambda i: view_one(state, i, tiles))(jnp.arange(n))
    embed = world.encoder(jax.nn.one_hot(views, 16)) + world.intero_in(state.flock.levels)
    h = world.transition(h, z, jax.nn.one_hot(a_prev, wm.N_ACTIONS))
    z = wm._sample(world.posterior(h, embed), k1)
    actions = agent.act(ac.feat_of(h, z), k2)
    new = sim.step(state, actions)
    rec = dict(view=views[rec_idx].astype(jnp.uint8), action=actions[rec_idx], levels=state.flock.levels[rec_idx], reward=new.flock.reward[rec_idx], night=sim.is_night(state.t), all_actions=actions, all_levels=new.flock.levels, all_reward=new.flock.reward)
    return new, (h, z, actions), rec


@nnx.jit(static_argnums=(5,))
def collect(world, agent, states, mems, key, steps, rec_idx):
    def one(carry, k):
        st, mm = carry
        st, mm, rec = jax.vmap(lambda s, m, kk: act_and_step(world, agent, s, m, kk, rec_idx))(st, mm, jax.random.split(k, st.t.shape[0]))
        return (st, mm), rec

    (states, mems), recs = jax.lax.scan(one, (states, mems), jax.random.split(key, steps))
    return states, mems, recs


@nnx.jit
def start_states(world, packed, actions, levels, key):
    s = world.observe(unpack(packed).astype(jnp.int32), actions, key, levels)
    return s.h.reshape(-1, wm.DETER), s.z.reshape(-1, wm.STOCH * wm.CLASSES)


class Replay:
    """A ring of fixed-length sequences (packed strips, actions, levels, rewards)."""

    def __init__(self, capacity: int, length: int):
        self.v = np.zeros((capacity, length, wm.H, wm.W // 2), np.uint8)
        self.a = np.zeros((capacity, length), np.int32)
        self.l = np.zeros((capacity, length, D.N_DRIVES), np.float32)
        self.r = np.zeros((capacity, length), np.float32)
        self.n, self.i, self.cap = 0, 0, capacity

    def add(self, v, a, l, r):
        for k in range(v.shape[0]):
            self.v[self.i], self.a[self.i], self.l[self.i], self.r[self.i] = v[k], a[k], l[k], r[k]
            self.i = (self.i + 1) % self.cap
            self.n = min(self.n + 1, self.cap)

    def sample(self, rng, B):
        idx = rng.integers(self.n, size=B)
        return self.v[idx], self.a[idx], self.l[idx], self.r[idx]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wm", default=os.path.expanduser("~/flock2_data/m1_wm/run4_negdrive/model.pkl"))
    p.add_argument("--scripted", default=os.path.expanduser("~/flock2_data/m1_rollouts_drives"))
    p.add_argument("--out", default=os.path.expanduser("~/flock2_data/m1_agent/run1"))
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--meadows", type=int, default=4)
    p.add_argument("--sheep", type=int, default=40)
    p.add_argument("--record", type=int, default=8)
    p.add_argument("--collect", type=int, default=16)
    p.add_argument("--length", type=int, default=64)
    p.add_argument("--wm_steps", type=int, default=64)
    p.add_argument("--wm_online", type=int, default=3)
    p.add_argument("--ac_steps", type=int, default=64)
    p.add_argument("--replay", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    world = wm.WorldModel(nnx.Rngs(a.seed), n_intero=D.N_DRIVES, reward=True)
    wm.load_into(world, pickle.load(open(a.wm, "rb")))
    wm_opt = nnx.Optimizer(world, optax.chain(optax.clip_by_global_norm(1000.0), optax.adam(1e-4, eps=1e-8)), wrt=nnx.Param)
    agent = ac.Agent(nnx.Rngs(a.seed + 1))
    ac_opt = ac.make_optimizer(agent)
    scripted = _wmt.load_shards(a.scripted, intero=True, reward=True, reward_from_levels=True)[:-1]
    replay = Replay(a.replay, a.length)
    n_rec = a.meadows * a.record
    window = Replay(n_rec, a.length)  # the logged sheep's last --length steps, oldest first
    rng = np.random.default_rng(a.seed)
    key = jax.random.PRNGKey(a.seed)

    key, k = jax.random.split(key)
    states = jax.vmap(lambda kk: generate(kk, a.sheep))(jax.random.split(k, a.meadows))
    key, k = jax.random.split(key)
    states = states._replace(t=jax.random.randint(k, (a.meadows,), 0, sim.DAY))
    mems = (jnp.zeros((a.meadows, a.sheep, wm.DETER)), jnp.zeros((a.meadows, a.sheep, wm.STOCH * wm.CLASSES)), jnp.zeros((a.meadows, a.sheep), jnp.int32))
    rec_idx = jnp.arange(a.record)
    # compile (and autotune) the two update steps before acting, while the GPU is empty
    while replay.n < 4:
        replay.add(*(np.concatenate([x, x]) for x in _wmt.sample(scripted, rng, 2, a.length)))
    pk, ak, lk, rk = replay.sample(rng, 4)
    key, k1, k2 = jax.random.split(key, 3)
    _wmt.train_step(world, wm_opt, pk, ak, lk, rk, k1)
    ac.train_step(world, agent, ac_opt, *start_states(world, pk, ak, lk, k1), k2)
    replay.n = replay.i = 0
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    t0 = time.time()
    for it in range(1, a.iters + 1):
        key, k = jax.random.split(key)
        states, mems, recs = collect(world, agent, states, mems, k, a.collect, rec_idx)
        # to the replay: (T, B, R, ...) -> (B*R, T, ...)
        v = np.asarray(recs["view"]).transpose(1, 2, 0, 3, 4).reshape(-1, a.collect, wm.H, wm.W)
        packed = pack(v)
        del v
        seq = lambda x: np.asarray(x).swapaxes(0, 1).swapaxes(1, 2).reshape((-1, a.collect) + x.shape[3:])
        new = (packed, seq(recs["action"]), seq(recs["levels"]), seq(recs["reward"]))
        del recs["view"]
        for buf, x in zip((window.v, window.a, window.l, window.r), new):
            buf[:] = np.concatenate([buf[:, a.collect :], x], 1)
        if it * a.collect >= a.length:
            replay.add(window.v, window.a, window.l, window.r)
        if replay.n < 4:
            continue
        # world model
        wm_aux = []
        for _ in range(a.wm_steps):
            on = replay.sample(rng, a.wm_online)
            off = _wmt.sample(scripted, rng, 4 - a.wm_online, a.length)
            pk, ak, lk, rk = (np.concatenate([x, y]) for x, y in zip(on, off))
            key, k = jax.random.split(key)
            wm_aux.append(_wmt.train_step(world, wm_opt, pk, ak, lk, rk, k))
        # actor-critic
        ac_aux = []
        for _ in range(a.ac_steps):
            pk, ak, lk, _ = replay.sample(rng, 4)
            key, k1, k2 = jax.random.split(key, 3)
            h0, z0 = start_states(world, pk, ak, lk, k1)
            ac_aux.append(ac.train_step(world, agent, ac_opt, h0, z0, k2))
        # behaviour this iteration (all sheep, all meadows)
        acts = np.asarray(recs["all_actions"])  # (T, B, S)
        night = np.asarray(recs["night"])[:, :, None] & np.ones_like(acts, bool)
        frac = lambda m: {sim.ACTION_NAMES[i]: float((acts[m] == i).mean()) for i in range(sim.N_ACTIONS)} if m.any() else None
        rec = {
            "iter": it,
            "t": time.time() - t0,
            "reward": float(np.asarray(recs["all_reward"]).mean()),
            "levels": np.round(np.asarray(recs["all_levels"]).mean((0, 1, 2)), 3).tolist(),
            "night_frac": float(night.mean()),
            "actions_day": frac(~night),
            "actions_night": frac(night),
            **{f"wm_{k2}": float(np.mean([float(x[k2]) for x in wm_aux])) for k2 in wm_aux[0]},
            **{f"ac_{k2}": float(np.mean([float(x[k2]) for x in ac_aux])) for k2 in ac_aux[0]},
        }
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if it % 10 == 0 or it == 1:
            print(json.dumps({k2: rec[k2] for k2 in ("iter", "t", "reward", "levels", "night_frac", "ac_return", "ac_entropy", "wm_pixel_acc")}), flush=True)
        if it % 50 == 0 or it == a.iters:
            pickle.dump(jax.tree.map(np.asarray, nnx.to_pure_dict(nnx.state(world))), open(os.path.join(a.out, "wm.pkl"), "wb"))
            pickle.dump(jax.tree.map(np.asarray, nnx.to_pure_dict(nnx.state(agent))), open(os.path.join(a.out, "agent.pkl"), "wb"))


if __name__ == "__main__":
    main()
