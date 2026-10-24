#!/usr/bin/env python
"""Train the flock's shared policy online (Milestone 1, step 3): act in the meadows, log to a
replay, fine-tune the world model, train the actor-critic in imagination; repeat.

    python scripts/m1_agent.py --wm ~/flock2_data/m1_wm/run4_negdrive/model.pkl --out ~/flock2_data/m1_agent/run3
    python scripts/m1_agent.py --wm none --scripted none --ensemble 5 --curiosity 0.1 --out ~/flock2_data/m1_agent/birth1

``--wm none`` starts the world model from random weights and ``--scripted none`` uses no scripted
data: the sheep learn from birth on their own behaviour. ``--ensemble K`` gives the world model
K next-state heads whose disagreement, weighted ``--curiosity``, is added to the imagined reward.
The actor-critic starts training after ``--ac_start`` iterations (before that the policy is
uniform: an infancy of random behaviour while the world model learns). ``--varied`` starts each
meadow fresh — a new layout, the flock at a random place, a random time of day, every sheep's
levels drawn from [0.2, 1.0] — and ``--reset_every`` steps (staggered across meadows) starts it
fresh again, so the model sees healthy, hungry and thirsty sheep in every situation rather than
one flock running down; no replay sequence spans a reset.

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
    # ingesting nothing: drinking away from water, grazing where the grass is too short to bite
    r, c = sim._tile(state.meadow, state.flock.pos)
    dry_drink = (actions == sim.DRINK) & ~sim.near_water(state.meadow, state.flock.pos)
    bare_graze = (actions == sim.GRAZE) & (state.meadow.grass[r, c] < 1.0)
    rec = dict(view=views[rec_idx].astype(jnp.uint8), action=actions[rec_idx], levels=state.flock.levels[rec_idx], reward=new.flock.reward[rec_idx], night=sim.is_night(state.t), all_actions=actions, all_levels=new.flock.levels, all_reward=new.flock.reward, dry_drink=dry_drink, bare_graze=bare_graze)
    return new, (h, z, actions), rec


@nnx.jit(static_argnums=(5,))
def collect(world, agent, states, mems, key, steps, rec_idx):
    def one(carry, k):
        st, mm = carry
        st, mm, rec = jax.vmap(lambda s, m, kk: act_and_step(world, agent, s, m, kk, rec_idx))(st, mm, jax.random.split(k, st.t.shape[0]))
        return (st, mm), rec

    (states, mems), recs = jax.lax.scan(one, (states, mems), jax.random.split(key, steps))
    return states, mems, recs


def fresh(key, sheep):
    """A new meadow with the flock at a random place, a random time of day and random levels."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    st = generate(k1, sheep)
    centre = jax.random.uniform(k2, (2,), minval=24.0, maxval=sim.N_TILES - 24.0)
    pos = jnp.clip(st.flock.pos - sim.N_TILES / 2 + centre, 1.0, sim.N_TILES - 1.0)
    levels = jax.random.uniform(k3, (sheep, D.N_DRIVES), minval=0.2, maxval=1.0)
    return st._replace(flock=st.flock._replace(pos=pos, levels=levels), t=jax.random.randint(k4, (), 0, sim.DAY))


fresh_many = jax.jit(jax.vmap(fresh, in_axes=(0, None)), static_argnums=1)


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
    p.add_argument("--ensemble", type=int, default=0, help="curiosity ensemble heads in the world model (0: none)")
    p.add_argument("--ensemble_target", default="body", help="body: curiosity about level changes; z: about the next view state (birth1)")
    p.add_argument("--curiosity", type=float, default=0.0, help="weight of the ensemble-disagreement reward")
    p.add_argument("--ac_start", type=int, default=0, help="iterations of world-model-only learning before the actor-critic trains")
    p.add_argument("--varied", action="store_true", help="fresh meadows: random flock place, time and levels")
    p.add_argument("--reset_every", type=int, default=0, help="with --varied: start each meadow fresh every this many steps (0: never)")
    a = p.parse_args()
    if a.scripted == "none":
        a.wm_online = 4
    os.makedirs(a.out, exist_ok=True)

    world = wm.WorldModel(nnx.Rngs(a.seed), n_intero=D.N_DRIVES, reward=True, ensemble=a.ensemble, ensemble_target=a.ensemble_target)
    if a.wm != "none":
        wm.load_into(world, pickle.load(open(os.path.expanduser(a.wm), "rb")))
    wm_opt = nnx.Optimizer(world, optax.chain(optax.clip_by_global_norm(1000.0), optax.adam(1e-4, eps=1e-8)), wrt=nnx.Param)
    agent = ac.Agent(nnx.Rngs(a.seed + 1), curiosity=a.curiosity)
    ac_opt = ac.make_optimizer(agent)
    scripted = _wmt.load_shards(a.scripted, intero=True, reward=True, reward_from_levels=True)[:-1] if a.scripted != "none" else None
    replay = Replay(a.replay, a.length)
    n_rec = a.meadows * a.record
    window = Replay(n_rec, a.length)  # the logged sheep's last --length steps, oldest first
    rng = np.random.default_rng(a.seed)
    key = jax.random.PRNGKey(a.seed)

    key, k = jax.random.split(key)
    if a.varied:
        states = fresh_many(jax.random.split(k, a.meadows), a.sheep)
    else:
        states = jax.vmap(lambda kk: generate(kk, a.sheep))(jax.random.split(k, a.meadows))
        key, k = jax.random.split(key)
        states = states._replace(t=jax.random.randint(k, (a.meadows,), 0, sim.DAY))
    mems = (jnp.zeros((a.meadows, a.sheep, wm.DETER)), jnp.zeros((a.meadows, a.sheep, wm.STOCH * wm.CLASSES)), jnp.zeros((a.meadows, a.sheep), jnp.int32))
    # steps since each meadow started, staggered so resets are spread out; steps in each logged
    # sheep's window since its meadow started (a window is pushed only when full)
    age = (np.arange(a.meadows) * (a.reset_every // max(1, a.meadows))) if a.reset_every else np.zeros(a.meadows, int)
    fill = np.zeros(n_rec, int)
    rec_idx = jnp.arange(a.record)
    # compile (and autotune) the two update steps before acting, while the GPU is empty; on
    # throwaway copies, so the warm-up changes nothing
    pk = np.zeros((4, a.length, wm.H, wm.W // 2), np.uint8)
    ak = np.zeros((4, a.length), np.int32)
    lk = np.ones((4, a.length, D.N_DRIVES), np.float32)
    rk = np.zeros((4, a.length), np.float32)
    w_tmp, g_tmp = nnx.clone(world), nnx.clone(agent)
    o_tmp, a_tmp = nnx.clone(wm_opt), nnx.clone(ac_opt)
    key, k1, k2 = jax.random.split(key, 3)
    _wmt.train_step(w_tmp, o_tmp, pk, ak, lk, rk, k1)
    ac.train_step(w_tmp, g_tmp, a_tmp, *start_states(w_tmp, pk, ak, lk, k1), k2)
    del w_tmp, g_tmp, o_tmp, a_tmp
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
        fill += a.collect
        full = fill >= a.length
        if full.any():
            replay.add(window.v[full], window.a[full], window.l[full], window.r[full])
        age += a.collect
        if a.varied and a.reset_every:
            reset = age >= a.reset_every
            if reset.any():
                key, k = jax.random.split(key)
                new_states = fresh_many(jax.random.split(k, a.meadows), a.sheep)
                m = jnp.asarray(reset)
                pick = lambda old, nw: jnp.where(m.reshape((-1,) + (1,) * (old.ndim - 1)), nw, old)
                states = jax.tree.map(pick, states, new_states)
                mems = jax.tree.map(lambda old: pick(old, jnp.zeros_like(old)), mems)
                age[reset] = 0
                fill[np.repeat(reset, a.record)] = 0
        if replay.n < 4:
            continue
        # world model
        wm_aux = []
        for _ in range(a.wm_steps):
            on = replay.sample(rng, a.wm_online)
            if a.wm_online < 4:
                off = _wmt.sample(scripted, rng, 4 - a.wm_online, a.length)
                pk, ak, lk, rk = (np.concatenate([x, y]) for x, y in zip(on, off))
            else:
                pk, ak, lk, rk = on
            key, k = jax.random.split(key)
            wm_aux.append(_wmt.train_step(world, wm_opt, pk, ak, lk, rk, k))
        # actor-critic
        ac_aux = []
        for _ in range(a.ac_steps if it > a.ac_start else 0):
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
            "dry_drink": float(np.asarray(recs["dry_drink"]).mean()),
            "bare_graze": float(np.asarray(recs["bare_graze"]).mean()),
            "drink_dry_share": float(np.asarray(recs["dry_drink"]).sum() / max(1, (acts == sim.DRINK).sum())),
            **{f"wm_{k2}": float(np.mean([float(x[k2]) for x in wm_aux])) for k2 in wm_aux[0]},
            **({f"ac_{k2}": float(np.mean([float(x[k2]) for x in ac_aux])) for k2 in ac_aux[0]} if ac_aux else {}),
        }
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if it % 10 == 0 or it == 1:
            print(json.dumps({k2: rec.get(k2) for k2 in ("iter", "t", "reward", "levels", "drink_dry_share", "bare_graze", "ac_return", "ac_entropy", "ac_intrinsic", "wm_pixel_acc", "wm_reward_ce", "wm_ensemble_mse")}), flush=True)
        if it % 50 == 0 or it == a.iters:
            pickle.dump(jax.tree.map(np.asarray, nnx.to_pure_dict(nnx.state(world))), open(os.path.join(a.out, "wm.pkl"), "wb"))
            pickle.dump(jax.tree.map(np.asarray, nnx.to_pure_dict(nnx.state(agent))), open(os.path.join(a.out, "agent.pkl"), "wb"))


if __name__ == "__main__":
    main()
