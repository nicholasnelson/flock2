#!/usr/bin/env python
"""Stage 1 (spec § 20): the reactive sheep on meadow v2. A conv encoder on the sheep's own view
(``flock.meadow2.render``, 64 x 256 RGB) and its felt numbers, an MLP actor and critic, trained
end-to-end by PPO on the sheep's own reward. **No world model, no recurrence, no replay.** The
encoder is the world model's (``flock.wm.Encoder`` in RGB mode), trained here by the reward instead
of by reconstruction (the review's § 5.4: which of the sheep's own signals trains the eyes is a
designer's choice; recorded here).

    python scripts/m2_reactive.py --out ~/flock2_data/m2_reactive/a1_s0 --willows 3 --seed 0

The loop is ``m2_hybrid.py``'s without the world model: every sheep in ``--meadows`` meadows acts for
``--steps`` steps (sampling from the policy), then ``--epochs`` PPO epochs over every sample, with the
views re-encoded in each minibatch so the gradient reaches the encoder. Death as ``m2_ppo.py``: no
bootstrap past a death, no reward centring (rewards scaled by 1 - gamma), dead steps weighted 0.
Meadows restart every ``--reset_every`` steps (staggered) or when every sheep has died.
``--willows k`` stands k willows on each pond's bank (``world.generate``; 0 = meadow v2.1 as it was).
Checkpoint ``policy.pkl`` and ``config.json``; scored by ``m2_eval.py --policy reactive``.
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

from flock import wm
from flock.meadow2 import drives as D, render as R, sim
from flock.meadow2.world import DAY, random_start


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(__file__), f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_ppo1 = _load("m1_ppo")

CFG = wm.Config(n_classes=3, rgb=True, n_actions=sim.N_ACTIONS, body_dims=())


class ReactivePolicy(nnx.Module):
    """Eyes (the RGB conv encoder) plus the felt numbers -> an embedding -> actor and critic MLPs."""

    def __init__(self, rngs: nnx.Rngs, hidden: int = 256, cfg: wm.Config = CFG):
        self.cfg = cfg
        self.encoder = wm.Encoder(rngs, cfg)
        self.feel_in = nnx.Linear(D.N_FELT, cfg.embed, rngs=rngs)
        self.norm = nnx.LayerNorm(cfg.embed, rngs=rngs)
        self.head = _ppo1.Policy(rngs, hidden=hidden, n_in=cfg.embed + D.N_FELT, n_out=sim.N_ACTIONS)

    def embed(self, views: jax.Array, feel: jax.Array) -> jax.Array:
        """views (..., H, W, 3) uint8, feel (..., N_FELT) -> (..., embed)."""
        e = self.encoder(views.astype(jnp.float32) / 255.0 - 0.5) + self.feel_in(feel)
        return jax.nn.silu(self.norm(e))

    def __call__(self, views: jax.Array, feel: jax.Array):
        return self.head(jnp.concatenate([self.embed(views, feel), feel], -1))


def make_policy(seed, hidden=256):
    return ReactivePolicy(nnx.Rngs(seed), hidden=hidden)


def to_bytes(rgb):
    return jnp.round(rgb * 255.0).astype(jnp.uint8)


def see(states):
    """(views (B, S, H, W, 3) uint8, feel (B, S, N_FELT)) for a batch of meadows."""
    f = states.flock
    return to_bytes(R.render_meadows(states)), D.felt(f.levels, f.fat, f.taste)


def act(policy, states, key):
    """Every sheep's action (sampled; dead sheep stop) with the policy's log-prob and value."""
    views, feel = see(states)
    logits, value = policy(views, feel)
    a = jax.random.categorical(key, logits, -1)
    a = jnp.where(states.flock.alive, a, sim.STOP)
    logp = jnp.take_along_axis(jax.nn.log_softmax(logits), a[..., None], -1)[..., 0]
    return a, dict(views=views, feel=feel, logp=logp, value=value)


@nnx.jit(static_argnums=(3, 4))
def collect(policy, states, key, steps, gamma_lam):
    gamma, lam = gamma_lam

    def one(st, k):
        a, rec = act(policy, st, k)
        new = jax.vmap(sim.step)(st, a)
        rec.update(act=a, reward=new.flock.reward, alive_before=st.flock.alive, alive_after=new.flock.alive,
                   levels=new.flock.levels, fat=new.flock.fat, bad=new.flock.taste < 0)
        return new, rec

    states, tr = jax.lax.scan(one, states, jax.random.split(key, steps))  # (T, B, S, ...)
    _, last_v = policy(*see(states))
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
    return states, tr


@nnx.jit
def update(policy, opt, views, feel, act_, logp_old, adv, ret, wgt, clip, ent_coef):
    """The clipped PPO update through the encoder, each sample weighted (0 for steps taken while dead)."""

    def loss_fn(p):
        logits, v = p(views, feel)
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
    p.add_argument("--out", default=os.path.expanduser("~/flock2_data/m2_reactive/s0"))
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--meadows", type=int, default=8)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.998)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent", type=float, default=0.01)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--reset_every", type=int, default=2 * DAY)
    p.add_argument("--willows", type=int, default=0, help="willows per pond bank (stage 1's world change; 0 = none)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    assert a.reset_every % a.steps == 0
    os.makedirs(a.out, exist_ok=True)
    json.dump({**vars(a), "kind": "m2_reactive"}, open(os.path.join(a.out, "config.json"), "w"), indent=1)

    policy = make_policy(a.seed, a.hidden)
    opt = nnx.Optimizer(policy, optax.chain(optax.clip_by_global_norm(0.5), optax.adam(a.lr)), wrt=nnx.Param)
    rng = np.random.default_rng(a.seed)
    key = jax.random.PRNGKey(a.seed)
    fresh = jax.jit(jax.vmap(functools.partial(random_start, willows=a.willows)))
    key, k = jax.random.split(key)
    states = fresh(jax.random.split(k, a.meadows))
    age = (np.arange(a.meadows) % (a.reset_every // a.steps)) * a.steps
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    t0 = time.time()
    for it in range(1, a.iters + 1):
        t_a = time.time()
        key, k = jax.random.split(key)
        states, tr = collect(policy, states, k, a.steps, (a.gamma, a.lam))
        tr = jax.tree.map(np.asarray, tr)
        t_collect = time.time() - t_a
        t_b = time.time()
        flat = lambda x: x.reshape((-1,) + x.shape[3:])
        views, feel, act_, lp, ad, rt, w = (flat(tr[k2]) for k2 in ("views", "feel", "act", "logp", "adv", "ret", "alive_before"))
        w = w.astype(np.float32)
        auxes = []
        for _ in range(a.epochs):
            for mb in np.array_split(rng.permutation(len(act_)), a.minibatches):
                auxes.append(update(policy, opt, views[mb], feel[mb], act_[mb], lp[mb], ad[mb], rt[mb], w[mb], a.clip, a.ent))
        ppo_aux = {k2: float(np.mean([float(x[k2]) for x in auxes])) for k2 in auxes[0]}
        t_train = time.time() - t_b
        age += a.steps
        reset = (age >= a.reset_every) | ~np.asarray(states.flock.alive).any(-1)
        if reset.any():
            key, k = jax.random.split(key)
            nw = fresh(jax.random.split(k, a.meadows))
            m = jnp.asarray(reset)
            states = jax.tree.map(lambda o, n: jnp.where(m.reshape((-1,) + (1,) * (o.ndim - 1)), n, o), states, nw)
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
            "t_collect": t_collect,
            "t_train": t_train,
            **{f"ppo_{k2}": v for k2, v in ppo_aux.items()},
        }
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if it % 10 == 0 or it == 1:
            print(json.dumps({k2: rec.get(k2) for k2 in ("iter", "t", "reward", "alive_share", "deaths", "levels", "fat", "bad_taste", "ppo_entropy", "t_collect", "t_train")}), flush=True)
        if it % 25 == 0 or it == a.iters:
            pickle.dump(jax.tree.map(np.asarray, nnx.to_pure_dict(nnx.state(policy))), open(os.path.join(a.out, "policy.pkl"), "wb"))


if __name__ == "__main__":
    main()
