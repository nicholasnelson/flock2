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

**Stage 0 (spec § 19), the myopic oracle.** ``Perception`` restricts the water features to the nearest
water tile within ``water_range`` tiles and inside the 300-degree field of view (masked to (0, 0, 1)
when there is none), and can add an ideal memory of the last water seen under that mask: forward,
left, distance / 32 and age / ``window``, masked to (0, 0, 1, 1) once older than ``window`` steps or
if water was never seen. Memory kinds: ``rot`` keeps the bearing when last seen and turns it by the
sheep's heading change since (the capacity the D3 probe measures; the distance is held and the sheep's
own travel is ignored); ``place`` keeps the water tile's position and re-expresses it in the sheep's
frame each step (perfect path integration). The memory is carried by the caller (``init_memory``,
``observe_mem``).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from flock.meadow2 import drives as D, sim
from flock.meadow2.world import DAY, HUT, N_TILES, TREE, State, cell_centres, water_at

GOOD_GRASS = 1.5
N_OBS = D.N_FELT + 3 + 1 + 3 * 3 + 3 + 1 + 1
N_MEMORY = 4
_TILES = jnp.stack(jnp.meshgrid(jnp.arange(N_TILES) + 0.5, jnp.arange(N_TILES) + 0.5, indexing="ij"), -1).reshape(-1, 2)
_CELLS = cell_centres().reshape(-1, 2)
NEVER = -1_000_000


class Perception(NamedTuple):
    """How the oracle sees water. ``water_range`` None: any water, anywhere (the original upper
    bound). ``memory`` None, "rot" or "place"; ``window`` in steps."""

    water_range: float | None = None
    memory: str | None = None
    window: int = 0


FULL = Perception()


class WaterMemory(NamedTuple):
    pos: jax.Array  # (S, 2) the last-seen water tile, world frame
    bearing: jax.Array  # (S,) its bearing when last seen, left-positive radians
    heading: jax.Array  # (S,) the sheep's heading then
    dist: jax.Array  # (S,) its distance then
    t_seen: jax.Array  # (S,) int step; NEVER if none


def n_obs(cfg: Perception) -> int:
    return N_OBS + (N_MEMORY if cfg.memory else 0)


def init_memory(n_sheep: int) -> WaterMemory:
    z = jnp.zeros((n_sheep,))
    return WaterMemory(jnp.zeros((n_sheep, 2)), z, z, z, jnp.full((n_sheep,), NEVER, jnp.int32))


def _nearest(pos, targets, ok):
    d2 = jnp.sum((pos[:, None, :] - targets[None, :, :]) ** 2, -1)
    d2 = jnp.where(ok, d2, jnp.inf)
    i = jnp.argmin(d2, -1)
    dist = jnp.sqrt(jnp.min(d2, -1))
    vec = jnp.where(jnp.isfinite(dist)[:, None], targets[i] - pos, 0.0)
    return vec, jnp.where(jnp.isfinite(dist), dist, 99.0), jnp.isfinite(dist)


def _egocentric(vec, dist, heading):
    fwd = sim.direction(heading)
    left = jnp.stack([-fwd[:, 1], fwd[:, 0]], -1)
    unit = vec / jnp.maximum(dist, 1e-3)[:, None]
    return jnp.stack([jnp.sum(unit * fwd, -1), jnp.sum(unit * left, -1), jnp.minimum(dist / 32.0, 1.0)], -1)


def bearing(vec, heading):
    """Left-positive bearing (radians) of a world-frame vector from a sheep with ``heading``."""
    fwd = sim.direction(heading)
    left = jnp.stack([-fwd[:, 1], fwd[:, 0]], -1)
    return jnp.arctan2(jnp.sum(vec * left, -1), jnp.sum(vec * fwd, -1))


def visible_water(state: State, water_range: float | None):
    """The nearest water tile each sheep can use: (vec (S, 2), dist (S,), found (S,)). With a range,
    only tiles within it and inside the field of view count; occlusion and night are ignored."""
    m, f = state.meadow, state.flock
    wet = water_at(m.water, _TILES)[None, :]
    if water_range is None:
        return _nearest(f.pos, _TILES, wet)
    rel = _TILES[None] - f.pos[:, None]
    d = jnp.linalg.norm(rel, axis=-1)
    alpha = bearing(rel.reshape(-1, 2), jnp.repeat(f.heading, _TILES.shape[0])).reshape(d.shape)
    ok = wet & (d <= water_range) & (jnp.abs(alpha) < sim.FOV / 2)
    return _nearest(f.pos, _TILES, ok)


def _observe(state: State, water_v, water_d, water_found) -> jax.Array:
    m, f = state.meadow, state.flock
    S = f.pos.shape[0]
    phase = 2 * jnp.pi * (state.t % DAY) / DAY
    grass_v, grass_d, _ = _nearest(f.pos, _CELLS, (m.grass.reshape(-1) >= GOOD_GRASS)[None, :])
    cover = m.obj_present & ((m.obj_kind == TREE) | (m.obj_kind == HUT))
    cover_v, cover_d, _ = _nearest(f.pos, m.obj_pos, cover[None, :])
    on_water, density, _ = sim.mouth_tile(m, f.pos, f.heading)
    d_sheep = jnp.linalg.norm(f.pos[:, None] - f.pos[None], axis=-1)
    huddle = jnp.sum((d_sheep < D.HUDDLE_DIST) & f.resting[None, :] & f.alive[None, :] & ~jnp.eye(S, dtype=bool), -1)
    water = jnp.where(water_found[:, None], _egocentric(water_v, water_d, f.heading), jnp.array([0.0, 0.0, 1.0]))
    feats = [
        D.felt(f.levels, f.fat, f.taste),
        jnp.broadcast_to(jnp.stack([jnp.sin(phase), jnp.cos(phase), sim.light(state.t)]), (S, 3)),
        f.resting.astype(jnp.float32)[:, None],
        water,
        _egocentric(grass_v, grass_d, f.heading),
        _egocentric(cover_v, cover_d, f.heading),
        jnp.stack([on_water, density >= sim.MIN_BITE, cover_d <= D.COVER_DIST], -1).astype(jnp.float32),
        (density / 3.0)[:, None],
        (jnp.minimum(huddle, D.HUDDLE_MAX) / D.HUDDLE_MAX)[:, None],
    ]
    return jnp.concatenate(feats, -1)


def observe(state: State, cfg: Perception = FULL) -> jax.Array:
    """(S, N_OBS) privileged observations for every sheep (no memory)."""
    return _observe(state, *visible_water(state, cfg.water_range))


def update_memory(mem: WaterMemory, state: State, water_v, water_d, found) -> WaterMemory:
    f = state.flock
    return WaterMemory(
        jnp.where(found[:, None], f.pos + water_v, mem.pos),
        jnp.where(found, bearing(water_v, f.heading), mem.bearing),
        jnp.where(found, f.heading, mem.heading),
        jnp.where(found, water_d, mem.dist),
        jnp.where(found, state.t, mem.t_seen),
    )


def memory_features(mem: WaterMemory, state: State, cfg: Perception) -> jax.Array:
    """(S, N_MEMORY): forward, left, distance / 32, age / window; masked when expired or never seen."""
    f = state.flock
    age = state.t - mem.t_seen
    fresh = (mem.t_seen != NEVER) & (age <= cfg.window)
    if cfg.memory == "rot":
        dpsi = (f.heading - mem.heading + jnp.pi) % (2 * jnp.pi) - jnp.pi
        est = mem.bearing + dpsi
        ego = jnp.stack([jnp.cos(est), jnp.sin(est), jnp.minimum(mem.dist / 32.0, 1.0)], -1)
    elif cfg.memory == "place":
        vec = mem.pos - f.pos
        dist = jnp.linalg.norm(vec, axis=-1)
        ego = _egocentric(vec, dist, f.heading)
    else:
        raise ValueError(cfg.memory)
    feats = jnp.concatenate([ego, (jnp.minimum(age, cfg.window) / max(cfg.window, 1))[:, None]], -1)
    return jnp.where(fresh[:, None], feats, jnp.array([0.0, 0.0, 1.0, 1.0]))


def observe_mem(state: State, mem: WaterMemory, cfg: Perception) -> tuple[jax.Array, WaterMemory]:
    """(obs (S, n_obs(cfg)), updated memory). The memory is refreshed from this step's visible water
    before the features are read, so visible water has age 0."""
    water_v, water_d, found = visible_water(state, cfg.water_range)
    obs = _observe(state, water_v, water_d, found)
    if not cfg.memory:
        return obs, mem
    mem = update_memory(mem, state, water_v, water_d, found)
    return jnp.concatenate([obs, memory_features(mem, state, cfg)], -1), mem
