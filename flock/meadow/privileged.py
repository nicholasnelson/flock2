"""Privileged observations for the sanity-ladder upper bound (Milestone 1; review of 2026-10-24,
§ 4.1): what a sheep would need to look after itself, computed from the true state rather than
seen. Used only by ``scripts/m1_ppo.py`` and its evaluation, never by the sheep's own learning.

Per sheep (``N_OBS`` numbers): the five levels and the last mouthful's taste; time of day as
sin/cos and a night flag; whether it is resting; and, for the nearest water tile, the nearest
tile with grass to bite (>= 1) and the nearest tree or hut, the direction in the sheep's own frame
(forward, left) and the distance (scaled by 1/32, capped at 1) — plus whether it is already close
enough to drink, graze or shelter.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from flock.meadow import drives as D, sim
from flock.meadow.world import HUT, N_TILES, TREE, State

N_OBS = 6 + 3 + 1 + 3 * 3 + 3
_CENTRES = jnp.stack(jnp.meshgrid(jnp.arange(N_TILES) + 0.5, jnp.arange(N_TILES) + 0.5, indexing="ij"), -1).reshape(-1, 2)


def _nearest(pos, targets, ok):
    """pos (S, 2), targets (M, 2), ok (M,) -> (vector to the nearest ok target (S, 2), distance (S,))."""
    d2 = jnp.sum((pos[:, None, :] - targets[None, :, :]) ** 2, -1)
    d2 = jnp.where(ok[None, :], d2, jnp.inf)
    i = jnp.argmin(d2, -1)
    vec = targets[i] - pos
    dist = jnp.sqrt(jnp.min(d2, -1))
    vec = jnp.where(jnp.isfinite(dist)[:, None], vec, 0.0)
    return vec, jnp.where(jnp.isfinite(dist), dist, 99.0)


def _egocentric(vec, dist, heading):
    fwd = sim.direction(heading)  # (S, 2)
    left = jnp.stack([-fwd[:, 1], fwd[:, 0]], -1)
    unit = vec / jnp.maximum(dist, 1e-3)[:, None]
    return jnp.stack([jnp.sum(unit * fwd, -1), jnp.sum(unit * left, -1), jnp.minimum(dist / 32.0, 1.0)], -1)


def observe(state: State) -> jax.Array:
    """(S, N_OBS) privileged observations for every sheep."""
    m, f = state.meadow, state.flock
    phase = 2 * jnp.pi * (state.t % sim.DAY) / sim.DAY
    S = f.pos.shape[0]
    water_v, water_d = _nearest(f.pos, _CENTRES, m.water.reshape(-1))
    grass_v, grass_d = _nearest(f.pos, _CENTRES, (m.grass >= 1.0).reshape(-1) & ~m.water.reshape(-1))
    cover = m.obj_present & ((m.obj_kind == TREE) | (m.obj_kind == HUT))
    cover_v, cover_d = _nearest(f.pos, m.obj_pos, cover)
    r, c = sim._tile(m, f.pos)
    feats = [
        D.felt(f.levels, f.taste),
        jnp.broadcast_to(jnp.stack([jnp.sin(phase), jnp.cos(phase), sim.is_night(state.t).astype(jnp.float32)]), (S, 3)),
        f.resting.astype(jnp.float32)[:, None],
        _egocentric(water_v, water_d, f.heading),
        _egocentric(grass_v, grass_d, f.heading),
        _egocentric(cover_v, cover_d, f.heading),
        jnp.stack([sim.near_water(m, f.pos), m.grass[r, c] >= 1.0, cover_d <= D.COVER_DIST], -1).astype(jnp.float32),
    ]
    return jnp.concatenate(feats, -1)
