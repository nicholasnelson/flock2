"""Logged rollouts of the scripted flock, for world-model pretraining (Milestone 1, step 2).

A rollout runs B meadows (fresh each rollout) from a random time of day, warms the flock up
without recording, then records R sheep per meadow for T steps: each recorded sheep's
first-person strip, the action it took, and some state for analysis (night flag, position,
heading). Strips are packed two pixels per byte (palette indices are 4 bits).

On disk (a directory of shards, one per rollout):
    shard_XXXX.views.npy    uint8 (N, T, fpv.H, fpv.W // 2)   packed strips
    shard_XXXX.actions.npy  uint8 (N, T)                      action after seeing the strip
    shard_XXXX.night.npy    bool  (N, T)
    shard_XXXX.pos.npy      float16 (N, T, 2)
    shard_XXXX.heading.npy  float16 (N, T)
    shard_XXXX.levels.npy   float16 (N, T, N_DRIVES)  drive levels when the strip was seen
    shard_XXXX.reward.npy   float32 (N, T)            reward for the step taken (drives.reward at logging time)
with N = B * R sequences. The strip at t is what the sheep saw before acting at t.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np

from flock import fpv
from flock.meadow import drives as D, scripted, sim, views
from flock.meadow.world import ground_state, generate


def pack(v: np.ndarray) -> np.ndarray:
    """(..., W) palette indices (< 16) -> (..., W // 2) bytes."""
    return (v[..., 0::2] | (v[..., 1::2] << 4)).astype(np.uint8)


def unpack(p: jax.Array) -> jax.Array:
    """(..., W // 2) bytes -> (..., W) palette indices (works on numpy and jax arrays)."""
    lo, hi = p & 0xF, p >> 4
    return jnp.stack([lo, hi], -1).reshape(p.shape[:-1] + (p.shape[-1] * 2,))


def _record_chunk(states, keys, rec_idx, n_steps):
    """Advance B meadows ``n_steps`` with the scripted flock; record the chosen sheep."""

    def one(st, k):
        tiles = ground_state(st.meadow)
        v = jax.vmap(lambda i: views.view_one(st, i, tiles))(rec_idx)
        a = scripted.policy(st, k)
        out = dict(view=v, action=a[rec_idx], night=sim.is_night(st.t), pos=st.flock.pos[rec_idx], heading=st.flock.heading[rec_idx], levels=st.flock.levels[rec_idx])
        st2 = sim.step(st, a)
        out["reward"] = st2.flock.reward[rec_idx]
        return st2, out

    def run(st, key):
        return jax.lax.scan(one, st, jax.random.split(key, n_steps))

    return jax.vmap(run)(states, keys)


_record_chunk_jit = jax.jit(_record_chunk, static_argnums=(3,))


def _warm(states, keys, n_steps):
    def run(st, key):
        return jax.lax.scan(lambda s, k: (sim.step(s, scripted.policy(s, k)), None), st, jax.random.split(key, n_steps))[0]

    return jax.vmap(run)(states, keys)


_warm_jit = jax.jit(_warm, static_argnums=(2,))


def write_rollout(out_dir: str, shard: int, seed: int, meadows: int = 32, sheep: int = 40, record: int = 8, steps: int = 1024, warm: int = 200, chunk: int = 128) -> int:
    """One shard; returns the number of frames written."""
    os.makedirs(out_dir, exist_ok=True)
    key = jax.random.PRNGKey(seed)
    k_gen, k_t, k_rec, k_warm, k_run = jax.random.split(key, 5)
    states = jax.vmap(lambda k: generate(k, sheep))(jax.random.split(k_gen, meadows))
    t0 = jax.random.randint(k_t, (meadows,), 0, sim.DAY)
    states = states._replace(t=t0)
    states = _warm_jit(states, jax.random.split(k_warm, meadows), warm)
    rec_idx = jax.random.permutation(k_rec, sheep)[:record]
    N = meadows * record
    base = os.path.join(out_dir, f"shard_{shard:04d}")
    mm = {
        "views": np.lib.format.open_memmap(base + ".views.npy", "w+", np.uint8, (N, steps, fpv.H, fpv.W // 2)),
        "actions": np.lib.format.open_memmap(base + ".actions.npy", "w+", np.uint8, (N, steps)),
        "night": np.lib.format.open_memmap(base + ".night.npy", "w+", bool, (N, steps)),
        "pos": np.lib.format.open_memmap(base + ".pos.npy", "w+", np.float16, (N, steps, 2)),
        "heading": np.lib.format.open_memmap(base + ".heading.npy", "w+", np.float16, (N, steps)),
        "levels": np.lib.format.open_memmap(base + ".levels.npy", "w+", np.float16, (N, steps, D.N_DRIVES)),
        "reward": np.lib.format.open_memmap(base + ".reward.npy", "w+", np.float32, (N, steps)),
    }
    keys = jax.random.split(k_run, steps // chunk)
    for c in range(steps // chunk):
        states, out = _record_chunk_jit(states, jax.random.split(keys[c], meadows), rec_idx, chunk)
        sl = slice(c * chunk, (c + 1) * chunk)
        v = np.asarray(out["view"])  # (B, chunk, R, H, W)
        mm["views"][:, sl] = pack(v.transpose(0, 2, 1, 3, 4).reshape(N, chunk, fpv.H, fpv.W))
        mm["actions"][:, sl] = np.asarray(out["action"]).transpose(0, 2, 1).reshape(N, chunk)
        mm["night"][:, sl] = np.repeat(np.asarray(out["night"])[:, None, :], record, 1).reshape(N, chunk)
        mm["pos"][:, sl] = np.asarray(out["pos"]).transpose(0, 2, 1, 3).reshape(N, chunk, 2).astype(np.float16)
        mm["heading"][:, sl] = np.asarray(out["heading"]).transpose(0, 2, 1).reshape(N, chunk).astype(np.float16)
        mm["levels"][:, sl] = np.asarray(out["levels"]).transpose(0, 2, 1, 3).reshape(N, chunk, D.N_DRIVES).astype(np.float16)
        mm["reward"][:, sl] = np.asarray(out["reward"]).transpose(0, 2, 1).reshape(N, chunk)
    for a in mm.values():
        a.flush()
    return N * steps
