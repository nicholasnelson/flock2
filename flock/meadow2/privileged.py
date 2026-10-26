"""Privileged observations for meadow v2's upper bound (as ``flock.meadow.privileged``): what a
sheep would need to look after itself, computed from the true state rather than seen. Used only by
``scripts/m2_ppo.py`` and its evaluation, never by a learning sheep.

Per sheep (``N_OBS`` numbers):
- what it feels (``drives.felt``: four levels, fat, taste);
- time of day as sin/cos, and the light level;
- whether it is resting;
- for the nearest water, the nearest good grass (a grass cell of density at least
  ``GOOD_GRASS``) and the nearest tree or hut: the direction in the sheep's own frame (forward,
  left) and the distance (scaled by 1/32, capped at 1);
- what is at its mouth: water, biteable grass, the patch's mean density / 3;
- whether it is near cover;
- how many resting sheep are close enough to huddle with (/ 4).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from flock.meadow2 import drives as D, sim
from flock.meadow2.world import DAY, HUT, N_TILES, TREE, State, cell_centres, water_at

GOOD_GRASS = 1.5
N_OBS = D.N_FELT + 3 + 1 + 3 * 3 + 3 + 1 + 1
_TILES = jnp.stack(jnp.meshgrid(jnp.arange(N_TILES) + 0.5, jnp.arange(N_TILES) + 0.5, indexing="ij"), -1).reshape(-1, 2)
_CELLS = cell_centres().reshape(-1, 2)


def _nearest(pos, targets, ok):
    d2 = jnp.sum((pos[:, None, :] - targets[None, :, :]) ** 2, -1)
    d2 = jnp.where(ok[None, :], d2, jnp.inf)
    i = jnp.argmin(d2, -1)
    dist = jnp.sqrt(jnp.min(d2, -1))
    vec = jnp.where(jnp.isfinite(dist)[:, None], targets[i] - pos, 0.0)
    return vec, jnp.where(jnp.isfinite(dist), dist, 99.0)


def _egocentric(vec, dist, heading):
    fwd = sim.direction(heading)
    left = jnp.stack([-fwd[:, 1], fwd[:, 0]], -1)
    unit = vec / jnp.maximum(dist, 1e-3)[:, None]
    return jnp.stack([jnp.sum(unit * fwd, -1), jnp.sum(unit * left, -1), jnp.minimum(dist / 32.0, 1.0)], -1)


def observe(state: State) -> jax.Array:
    """(S, N_OBS) privileged observations for every sheep."""
    m, f = state.meadow, state.flock
    S = f.pos.shape[0]
    phase = 2 * jnp.pi * (state.t % DAY) / DAY
    water_v, water_d = _nearest(f.pos, _TILES, water_at(m.water, _TILES))
    grass_v, grass_d = _nearest(f.pos, _CELLS, m.grass.reshape(-1) >= GOOD_GRASS)
    cover = m.obj_present & ((m.obj_kind == TREE) | (m.obj_kind == HUT))
    cover_v, cover_d = _nearest(f.pos, m.obj_pos, cover)
    on_water, density, _ = sim.mouth_tile(m, f.pos, f.heading)
    d_sheep = jnp.linalg.norm(f.pos[:, None] - f.pos[None], axis=-1)
    huddle = jnp.sum((d_sheep < D.HUDDLE_DIST) & f.resting[None, :] & f.alive[None, :] & ~jnp.eye(S, dtype=bool), -1)
    feats = [
        D.felt(f.levels, f.fat, f.taste),
        jnp.broadcast_to(jnp.stack([jnp.sin(phase), jnp.cos(phase), sim.light(state.t)]), (S, 3)),
        f.resting.astype(jnp.float32)[:, None],
        _egocentric(water_v, water_d, f.heading),
        _egocentric(grass_v, grass_d, f.heading),
        _egocentric(cover_v, cover_d, f.heading),
        jnp.stack([on_water, density >= sim.MIN_BITE, cover_d <= D.COVER_DIST], -1).astype(jnp.float32),
        (density / 3.0)[:, None],
        (jnp.minimum(huddle, D.HUDDLE_MAX) / D.HUDDLE_MAX)[:, None],
    ]
    return jnp.concatenate(feats, -1)
