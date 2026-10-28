#!/usr/bin/env python
"""The hybrid on meadow v2: the world model (``flock.wm``, RGB mode) learns from each sheep's own
view (``flock.meadow2.render``) and felt numbers, from birth; the policy learns with PPO from real
experience on the world model's state (h, z). The meadow's ``m1_hybrid.py``, on meadow v2.

    python scripts/m2_hybrid.py --out ~/flock2_data/m2_hybrid/s0 --seed 0

Per iteration:
- every sheep in ``--meadows`` meadows acts for ``--steps`` steps from its own model state (the
  posterior filter), sampling from the policy;
- the first ``--record`` sheep of each meadow give their views, actions, felt numbers and rewards
  to the world model's replay, in sequences of ``--length``, kept only if the sheep was alive
  throughout;
- ``--wm_steps`` world-model updates on batches of ``--batch`` sequences;
- then, after ``--ppo_start`` iterations of world-model-only infancy, PPO epochs on every living
  sheep's (state, action, reward) samples. The state is fixed input: no gradient reaches the world
  model.

PPO handles death as ``m2_ppo.py`` does: no bootstrap past a death, no reward centring (rewards
scaled by 1 - gamma), and steps taken while dead are left out. Meadows restart every
``--reset_every`` steps (staggered), or when every sheep in one has died, with model states zeroed.
Checkpoints ``wm.pkl``, ``policy.pkl`` and ``config.json``; scored by ``m2_eval.py --policy hybrid``.

``--policy_input state_embed`` (the plan review's arm, 2026-10-26): PPO sees the current view's
embedding as well as (h, z), with the gradient stopped, so PPO does not train the encoder. The
embedding is the sheep's own eyes, not privileged information.

``--kl_dyn`` and ``--ms_weight`` set the world model's loss (``flock.wm.Config``); ``--add_frac`` adds
only a random share of each iteration's sequences to the replay, so a replay of a given size covers
a longer history.

``--model spatial`` uses rung 3's world model (``flock.wm_spatial``: a grid-shaped state, all of it
learned) instead of the RSSM's single vector (``--model vector``, the default). ``--model rotmem`` is
the RSSM with rung 4's turn-rotated memory (``flock.wm``, ``rot_bins``; spec § 18): a ring of
``ROT_BINS`` x 16 numbers in h, rolled ``ROT_SHIFT`` bins by each 22.5-degree turn, so one bin is
11.25 degrees.
"""

from __future__ import annotations

import argparse
import functools
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

from flock import ac, wm, wm_spatial
from flock.meadow2 import drives as D, render as R, sim
from flock.meadow2.world import DAY, random_start


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(__file__), f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_ppo1 = _load("m1_ppo")
_ppo2 = _load("m2_ppo")

CFG = wm.Config(n_classes=3, rgb=True, n_actions=sim.N_ACTIONS, body_dims=())
ROT_BINS, ROT_SHIFT = 32, 2  # rung 4's ring: 360 degrees in 32 bins, a turn (22.5 degrees) is 2 of them


def make_world(seed, kind="vector", kl_dyn=wm.KL_DYN, ms_weight=0.0, ms_horizons=(2, 4, 8), ms_stride=8):
    """The world model. ``kl_dyn`` and ``ms_weight`` change only the loss, not the parameters, so a
    checkpoint loads into a model made with the defaults."""
    if kind == "spatial":
        return wm_spatial.SpatialWorldModel(nnx.Rngs(seed), n_intero=D.N_FELT, cfg=wm_spatial.SpatialConfig(n_actions=sim.N_ACTIONS, kl_dyn=kl_dyn, ms_weight=ms_weight, ms_horizons=ms_horizons, ms_stride=ms_stride))
    cfg = CFG._replace(kl_dyn=kl_dyn, ms_weight=ms_weight, ms_horizons=ms_horizons, ms_stride=ms_stride)
    if kind == "rotmem":
        cfg = cfg._replace(rot_bins=ROT_BINS, rot_shift=ROT_SHIFT, rot_left=sim.LEFT, rot_right=sim.RIGHT)
    return wm.WorldModel(nnx.Rngs(seed), n_intero=D.N_FELT, reward=True, cfg=cfg)


def make_policy(seed, hidden=256, n_in=CFG.feat):
    return _ppo1.Policy(nnx.Rngs(seed), hidden=hidden, n_in=n_in, n_out=sim.N_ACTIONS)


def zero_mems(world, B, S):
    c = world.cfg
    return jnp.zeros((B, S, getattr(c, "state", c.deter))), jnp.zeros((B, S, c.stoch * c.classes)), jnp.zeros((B, S), jnp.int32)


def to_bytes(rgb):
    return jnp.round(rgb * 255.0).astype(jnp.uint8)


def policy_feat(h, z, embed, use_embed):
    f = ac.feat_of(h, z)
    return jax.lax.stop_gradient(jnp.concatenate([f, embed], -1) if use_embed else f)


def n_policy_in(world, use_embed):
    return world.cfg.feat + (world.cfg.embed if use_embed else 0)


def sheep_step(world, policy, states, mems, key, use_embed=False):
    """Every sheep in a batch of meadows sees, updates its model state, and acts."""
    h, z, a_prev = mems  # (B, S, ...)
    k1, k2 = jax.random.split(key)
    views = to_bytes(R.render_meadows(states))  # (B, S, H, W, 3) uint8
    f = states.flock
    feel = D.felt(f.levels, f.fat, f.taste)
    embed = world.embed_obs(views, feel)
    h = world.transition(h, z, jax.nn.one_hot(a_prev, world.cfg.n_actions))
    z = wm._sample(world.posterior(h, embed), k1)
    feat = policy_feat(h, z, embed, use_embed)
    logits, value = policy(feat)
    a = jax.random.categorical(k2, logits, -1)
    a = jnp.where(f.alive, a, sim.STOP)
    logp = jnp.take_along_axis(jax.nn.log_softmax(logits), a[..., None], -1)[..., 0]
    new = jax.vmap(sim.step)(states, a)
    return new, (h, z, a), dict(views=views, feel=feel, feat=feat, act=a, logp=logp, value=value, reward=new.flock.reward,
                                alive_before=f.alive, alive_after=new.flock.alive, levels=new.flock.levels, fat=new.flock.fat, bad=new.flock.taste < 0)


@nnx.jit(static_argnums=(5, 6, 7, 8))
def collect(world, policy, states, mems, key, steps, record, gamma_lam, use_embed=False):
    gamma, lam = gamma_lam

    def one(carry, k):
        st, mm = carry
        st, mm, rec = sheep_step(world, policy, st, mm, k, use_embed)
        rec["views"] = rec["views"][:, :record]  # only the recorded sheep's views are kept
        return (st, mm), rec

    (states, mems), tr = jax.lax.scan(one, (states, mems), jax.random.split(key, steps))  # (T, B, S, ...)
    # the value of the state after the last step, for GAE's bootstrap
    h, z, a_prev = mems
    k = jax.random.fold_in(key, 7)
    views = to_bytes(R.render_meadows(states))
    f = states.flock
    embed = world.embed_obs(views, D.felt(f.levels, f.fat, f.taste))
    h2 = world.transition(h, z, jax.nn.one_hot(a_prev, world.cfg.n_actions))
    z2 = wm._sample(world.posterior(h2, embed), k)
    _, last_v = policy(policy_feat(h2, z2, embed, use_embed))
    r = tr["reward"] * (1 - gamma)
    cont = tr["alive_after"].astype(jnp.float32)

    def gae(carry, x):
        adv_next, v_next = carry
        rr, v, c = x
        delta = rr + gamma * v_next * c - v
        adv = delta + gamma * lam * c * adv_next
        return (adv, v), adv

    _, adv = jax.lax.scan(gae, (jnp.zeros_like(last_v), last_v), (r, tr["value"], cont), reverse=True)
    tr["adv"] = adv
    tr["ret"] = adv + tr["value"]
    return states, mems, tr


@nnx.jit
def wm_update(world, opt, v, a, f, r, key):
    (_, aux), g = nnx.value_and_grad(wm.loss_fn, has_aux=True)(world, v, a, key, f, r)
    opt.update(world, g)
    return aux


def wm_burst(world, opt, V, A, F, Rw, keys):
    """K world-model updates, one compiled step called K times. (Unrolling all K into one compiled
    call took 8 minutes to compile with multi-step prediction, and ran the spatial model out of
    memory: 15 GB of host RAM while compiling, 2026-10-26.)"""
    auxes = [wm_update(world, opt, V[i], A[i], F[i], Rw[i], keys[i]) for i in range(V.shape[0])]
    return jax.tree.map(lambda *v: jnp.stack(v), *auxes)


class Replay:
    def __init__(self, cap, length):
        self.v = np.zeros((cap, length, R.H, R.W, 3), np.uint8)
        self.a = np.zeros((cap, length), np.int32)
        self.f = np.zeros((cap, length, D.N_FELT), np.float32)
        self.r = np.zeros((cap, length), np.float32)
        self.n, self.i, self.cap = 0, 0, cap

    def add(self, v, a, f, r):
        idx = (self.i + np.arange(len(v))) % self.cap
        self.v[idx], self.a[idx], self.f[idx], self.r[idx] = v, a, f, r
        self.i = (self.i + len(v)) % self.cap
        self.n = min(self.n + len(v), self.cap)

    def sample(self, rng, B, K):
        idx = rng.integers(self.n, size=(K, B))
        return self.v[idx], self.a[idx], self.f[idx], self.r[idx]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.expanduser("~/flock2_data/m2_hybrid/s0"))
    p.add_argument("--iters", type=int, default=600)
    p.add_argument("--meadows", type=int, default=8)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--record", type=int, default=4)
    p.add_argument("--length", type=int, default=32)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--wm_steps", type=int, default=16)
    p.add_argument("--replay", type=int, default=1024)
    p.add_argument("--ppo_start", type=int, default=50)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.998)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent", type=float, default=0.01)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--reset_every", type=int, default=2 * DAY)
    p.add_argument("--model", choices=("vector", "spatial", "rotmem"), default="vector")
    p.add_argument("--policy_input", choices=("state", "state_embed"), default="state")
    p.add_argument("--kl_dyn", type=float, default=wm.KL_DYN, help="dynamics KL weight (DreamerV3: 1)")
    p.add_argument("--ms_weight", type=float, default=0.0, help="multi-step view prediction weight (0: off)")
    p.add_argument("--ms_horizons", default="2,4,8", help="steps ahead at which the view is predicted")
    p.add_argument("--ms_stride", type=int, default=8, help="steps between prediction starts")
    p.add_argument("--add_frac", type=float, default=1.0, help="share of each iteration's sequences added to the replay")
    p.add_argument("--willows", type=int, default=0, help="willows per pond bank (stage 1's world change; 0 = none)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    assert a.steps % a.length == 0 and a.reset_every % a.steps == 0
    os.makedirs(a.out, exist_ok=True)
    json.dump({**vars(a), "kind": "m2_hybrid"}, open(os.path.join(a.out, "config.json"), "w"), indent=1)

    horizons = tuple(int(x) for x in a.ms_horizons.split(","))
    assert a.ms_weight == 0 or max(horizons) < a.length
    world = make_world(a.seed, a.model, a.kl_dyn, a.ms_weight, horizons, a.ms_stride)
    wm_opt = nnx.Optimizer(world, optax.chain(optax.clip_by_global_norm(1000.0), optax.adam(1e-4, eps=1e-8)), wrt=nnx.Param)
    use_embed = a.policy_input == "state_embed"
    policy = make_policy(a.seed + 1, a.hidden, n_policy_in(world, use_embed))
    opt = nnx.Optimizer(policy, optax.chain(optax.clip_by_global_norm(0.5), optax.adam(a.lr)), wrt=nnx.Param)
    replay = Replay(a.replay, a.length)
    rng = np.random.default_rng(a.seed)
    key = jax.random.PRNGKey(a.seed)
    fresh = jax.jit(jax.vmap(functools.partial(random_start, willows=a.willows)))
    key, k = jax.random.split(key)
    states = fresh(jax.random.split(k, a.meadows))
    S = states.flock.pos.shape[1]
    zero = lambda: zero_mems(world, a.meadows, S)
    mems = zero()
    age = (np.arange(a.meadows) % (a.reset_every // a.steps)) * a.steps
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    t0 = time.time()
    for it in range(1, a.iters + 1):
        t_a = time.time()
        key, k = jax.random.split(key)
        states, mems, tr = collect(world, policy, states, mems, k, a.steps, a.record, (a.gamma, a.lam), use_embed)
        tr = jax.tree.map(np.asarray, tr)
        t_collect = time.time() - t_a
        # world-model replay: recorded sheep, cut into sequences, kept if alive throughout
        T, L = a.steps, a.length
        cut = lambda x: x[:, :, : a.record].transpose((1, 2, 0) + tuple(range(3, x.ndim))).reshape((-1, T // L, L) + x.shape[3:]).reshape((-1, L) + x.shape[3:])
        alive_seq = cut(tr["alive_before"]).all(-1) & (rng.random(cut(tr["alive_before"]).shape[0]) < a.add_frac)
        if alive_seq.any():
            replay.add(cut(tr["views"])[alive_seq], cut(tr["act"])[alive_seq], cut(tr["feel"])[alive_seq], cut(tr["reward"])[alive_seq])
        t_b = time.time()
        wm_aux = {}
        if replay.n >= a.batch:
            V, A, F, Rw = replay.sample(rng, a.batch, a.wm_steps)
            key, k = jax.random.split(key)
            wm_aux = {k2: float(np.mean(np.asarray(v))) for k2, v in wm_burst(world, wm_opt, V, A, F, Rw, jax.random.split(k, a.wm_steps)).items()}
        ppo_aux = {}
        if it > a.ppo_start:
            flat = lambda x: x.reshape((-1,) + x.shape[3:])
            obs, act, lp, ad, rt, w = (flat(tr[k2]) for k2 in ("feat", "act", "logp", "adv", "ret", "alive_before"))
            w = w.astype(np.float32)
            auxes = []
            for _ in range(a.epochs):
                for mb in np.array_split(rng.permutation(len(act)), a.minibatches):
                    auxes.append(_ppo2.update(policy, opt, obs[mb], act[mb], lp[mb], ad[mb], rt[mb], w[mb], a.clip, a.ent))
            ppo_aux = {k2: float(np.mean([float(x[k2]) for x in auxes])) for k2 in auxes[0]}
        t_train = time.time() - t_b
        age += a.steps
        reset = (age >= a.reset_every) | ~np.asarray(states.flock.alive).any(-1)
        if reset.any():
            key, k = jax.random.split(key)
            nw = fresh(jax.random.split(k, a.meadows))
            m = jnp.asarray(reset)
            pick = lambda o, n: jnp.where(m.reshape((-1,) + (1,) * (o.ndim - 1)), n, o)
            states = jax.tree.map(pick, states, nw)
            mems = jax.tree.map(pick, mems, zero())
            age[reset] = 0
        alive = tr["alive_before"]
        n_al = max(1, alive.sum())
        acts = tr["act"][alive]
        rec = {
            "iter": it,
            "t": time.time() - t0,
            "reward": float((tr["reward"] * alive).sum() / n_al),
            "alive_share": float(alive.mean()),
            "deaths": int((tr["alive_before"] & ~tr["alive_after"]).sum()),
            "levels": np.round((tr["levels"] * alive[..., None]).sum((0, 1, 2)) / n_al, 3).tolist(),
            "fat": float((tr["fat"] * alive).sum() / n_al),
            "bad_taste": float((tr["bad"] & alive).sum() / n_al),
            "actions": {sim.ACTION_NAMES[i]: round(float((acts == i).mean()), 3) for i in range(sim.N_ACTIONS)},
            "replay": int(replay.n),
            "t_collect": t_collect,
            "t_train": t_train,
            **{f"wm_{k2}": v for k2, v in wm_aux.items()},
            **{f"ppo_{k2}": v for k2, v in ppo_aux.items()},
        }
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if it % 10 == 0 or it == 1:
            print(json.dumps({k2: rec.get(k2) for k2 in ("iter", "t", "reward", "alive_share", "deaths", "levels", "fat", "bad_taste", "ppo_entropy", "wm_recon", "wm_multistep", "wm_pixel_acc", "wm_reward_ce", "t_collect", "t_train")}), flush=True)
        if it % 25 == 0 or it == a.iters:
            pickle.dump(jax.tree.map(np.asarray, nnx.to_pure_dict(nnx.state(world))), open(os.path.join(a.out, "wm.pkl"), "wb"))
            pickle.dump(jax.tree.map(np.asarray, nnx.to_pure_dict(nnx.state(policy))), open(os.path.join(a.out, "policy.pkl"), "wb"))


if __name__ == "__main__":
    main()
