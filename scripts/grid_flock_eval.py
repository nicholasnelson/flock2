#!/usr/bin/env python
"""Track B's evaluation (``docs/flock-2/trackB-grid-communication.md``, Measurements).

    python scripts/grid_flock_eval.py --agent ~/flock2_data/grid_flock/learned_s0 --out runs/grid_flock/eval/learned_s0.json

The trained sheep on ``--worlds`` fixed worlds (seeds 60,000 + i), 4 sheep each, for ``--steps`` steps,
in the condition it was trained in. Reported:
- reward per step, mean levels, drinking success;
- **time to water**: for each spell in which a sheep is thirsty (water < 0.5) and cannot see the pond,
  the steps until it drinks (spells still open at the end are counted as censored); the mean, and the
  share found within 100 steps;
- **causal influence, instant**: at every step, the mean KL between the move distribution given what the
  sheep heard and given what the same-numbered sheep in the next world heard (same model state h);
- **causal influence, outcome**: the whole run repeated with every sheep hearing the next world's
  hearing (``--swap``); its time to water and reward are reported beside the normal run's;
- **context**: mutual information (bits) between the emitted signal (each number binned into 4) and the
  sender's situation (drinking; thirsty and seeing the pond; thirsty and not; neither).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from flock import ac, wm
from flock.grid import flock as F, world as W1


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(__file__), f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def rollout(gf, world, policy, n_worlds, steps, mode, swap, seed=0):
    states = jax.vmap(lambda k: F.generate(k))(jnp.stack([jax.random.PRNGKey(60_000 + i) for i in range(n_worlds)]))
    mems = (jnp.zeros((n_worlds, F.K, gf.CFG.deter)), jnp.zeros((n_worlds, F.K, gf.CFG.stoch * gf.CFG.classes)), jnp.zeros((n_worlds, F.K), jnp.int32))

    def body(carry, k):
        st, (h, z, a_prev) = carry
        k1, k2, k3 = jax.random.split(k, 3)
        st = jax.vmap(lambda s: gf.with_heard(s, mode))(st)
        heard = st.heard
        heard_other = jnp.roll(heard, 1, axis=0)  # the same-numbered sheep in the next world
        if swap:
            st = st._replace(heard=heard_other)
        views = jax.vmap(F.views)(st)
        h = world.transition(h, z, jax.nn.one_hot(a_prev, gf.CFG.n_actions))
        enc = world.encoder(jax.nn.one_hot(views, gf.CFG.n_classes))
        feel = jax.vmap(F.felt)(st)
        z = wm._sample(world.posterior(h, enc + world.intero_in(feel)), k1)
        logits, mu, log_std, _ = policy(ac.feat_of(h, z))
        # instant influence: the same h, the other world's hearing
        feel_o = feel.at[..., W1.N_FELT :].set(st.heard if swap else heard_other)
        z_o = wm._sample(world.posterior(h, enc + world.intero_in(feel_o)), k1)
        logits_o = policy(ac.feat_of(h, z_o))[0]
        lp, lpo = jax.nn.log_softmax(logits), jax.nn.log_softmax(logits_o)
        kl = jnp.sum(jnp.exp(lp) * (lp - lpo), -1)
        a = jax.random.categorical(k2, logits, -1)
        u = mu + jnp.exp(log_std) * jax.random.normal(k3, mu.shape)
        near = jax.vmap(lambda s: F.near_water(s.water, s.pos))(st)
        drank = (a == W1.DRINK) & near
        sig = gf.emit(mode, u, drank)
        sees = jax.vmap(F.sees_water)(st)
        new = jax.vmap(F.step)(st, a, sig)
        return (new, (h, z, a)), dict(reward=new.reward, levels=new.levels, act=a, drank=drank, sees=sees, thirsty=st.levels[..., 1] < 0.5, sig=sig, kl=kl)

    run = jax.jit(lambda s, m, kk: jax.lax.scan(body, (s, m), jax.random.split(kk, steps))[1])
    return jax.tree.map(np.asarray, run(states, mems, jax.random.PRNGKey(seed)))


def time_to_water(rec, cap=100):
    """Spells in which a sheep is thirsty and blind to the pond, and the steps until it drinks."""
    blind = rec["thirsty"] & ~rec["sees"]
    T, B, K = blind.shape
    times, found = [], []
    for b in range(B):
        for k in range(K):
            t = 0
            while t < T:
                if blind[t, b, k]:
                    d = np.nonzero(rec["drank"][t:, b, k])[0]
                    if len(d):
                        times.append(d[0])
                        found.append(d[0] <= cap)
                        t += d[0] + 1
                    else:
                        found.append(False)
                        times.append(T - t)  # censored
                        break
                else:
                    t += 1
    times, found = np.array(times), np.array(found)
    return {"spells": int(len(times)), "mean_steps": float(times.mean()) if len(times) else None, "found_within_100": float(found.mean()) if len(found) else None}


def context_mi(rec):
    sig = rec["sig"].reshape(-1, F.N_SIGNAL)
    ctx = np.where(rec["drank"], 0, np.where(rec["thirsty"] & rec["sees"], 1, np.where(rec["thirsty"], 2, 3))).reshape(-1)
    bins = np.clip(((sig + 1) / 2 * 4).astype(int), 0, 3)
    code = bins[:, 0] * 4 + bins[:, 1]
    joint = np.zeros((16, 4))
    np.add.at(joint, (code, ctx), 1)
    p = joint / joint.sum()
    ps, pc = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log2(p[nz] / (ps @ pc)[nz])).sum())


def summarise(rec):
    drink = rec["act"] == W1.DRINK
    return {
        "reward": float(rec["reward"].mean()),
        "levels": rec["levels"].mean((0, 1, 2)).round(3).tolist(),
        "drink_success": float(rec["drank"].sum() / max(1, drink.sum())),
        "time_to_water": time_to_water(rec),
        "influence_kl_mean": float(rec["kl"].mean()),
        "context_mi_bits": context_mi(rec),
        "signal_var": float(rec["sig"].reshape(-1, F.N_SIGNAL).var(0).sum()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", required=True)
    p.add_argument("--worlds", type=int, default=64)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    gf = _load("grid_flock_hybrid")
    d = os.path.expanduser(a.agent)
    cfg = json.load(open(os.path.join(d, "config.json")))
    world = gf.make_world(0)
    wm.load_into(world, pickle.load(open(os.path.join(d, "wm.pkl"), "rb")))
    from flax import nnx

    policy = gf.SigPolicy(nnx.Rngs(0), gf.CFG.feat)
    wm.load_into(policy, pickle.load(open(os.path.join(d, "policy.pkl"), "rb")))
    mode = cfg["signal"]
    normal = summarise(rollout(gf, world, policy, a.worlds, a.steps, mode, False))
    swapped = summarise(rollout(gf, world, policy, a.worlds, a.steps, mode, True))
    res = {"agent": a.agent, "signal": mode, "normal": normal, "heard_swapped": swapped}
    print(json.dumps(res, indent=1))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
