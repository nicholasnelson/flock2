"""Meadow dynamics: continuous movement, collisions, grazing, drinking, regrowth, day and night.

Actions (owner's set, 2026-10-23): turn left, turn right, walk, trot, stop, graze, drink, rest,
head up. One action per sheep per step. Movement is continuous: a turn is TURN radians, walking
and trotting move WALK / TROT tiles along the heading. A move is refused if it would enter
water, leave the meadow or overlap a blocking object (trees, huts, the salt lick). Sheep that
overlap afterwards are pushed apart (one relaxation step of physical separation) unless the
push would land somewhere a move could not. Grazing eats the tile's biomass; grass regrows
everywhere off the water. Drinking works within one tile of water.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from flock.meadow import drives as D
from flock.meadow.world import HUT, N_TILES, RADIUS, SHEEP_RADIUS, TREE, Flock, Meadow, State

STOP, WALK, TROT, LEFT, RIGHT, GRAZE, DRINK, REST, HEAD_UP = range(9)
N_ACTIONS = 9
ACTION_NAMES = ("stop", "walk", "trot", "left", "right", "graze", "drink", "rest", "head_up")

TURN = jnp.pi / 8  # 22.5 degrees per turn step
WALK_SPEED, TROT_SPEED = 0.12, 0.35  # tiles per step
BITE = 0.05  # biomass eaten per graze step
REGROW = 0.0004  # fraction of the gap to full (3) regrown per step
DAY = 1200  # steps per day; the first 60 % is light
DAY_FRACTION = 0.6
# how far a sheep sees (tiles): by day / day with head up / night / night with head up
RANGE_DAY, RANGE_DAY_UP, RANGE_NIGHT, RANGE_NIGHT_UP = 12.0, 16.0, 5.0, 8.0
FOV = jnp.deg2rad(300.0)  # outside it, the rear blind wedge


def is_night(t: jax.Array) -> jax.Array:
    return ((t % DAY) / DAY) >= DAY_FRACTION


def direction(heading: jax.Array) -> jax.Array:
    """Unit (row, col) step for a heading (0 = north = -row, clockwise)."""
    return jnp.stack([-jnp.cos(heading), jnp.sin(heading)], -1)


def _tile(m: Meadow, p: jax.Array) -> tuple[jax.Array, jax.Array]:
    i = jnp.clip(jnp.floor(p).astype(jnp.int32), 0, N_TILES - 1)
    return i[..., 0], i[..., 1]


def near_water(m: Meadow, pos: jax.Array) -> jax.Array:
    """(S,) bool: any water tile in the 3 x 3 tiles around each sheep."""
    r, c = _tile(m, pos)
    off = jnp.array([-1, 0, 1])
    rr = jnp.clip(r[:, None, None] + off[None, :, None], 0, N_TILES - 1)
    cc = jnp.clip(c[:, None, None] + off[None, None, :], 0, N_TILES - 1)
    return m.water[rr, cc].any((1, 2))


def _allowed(m: Meadow, p: jax.Array) -> jax.Array:
    """(S,) bool: positions inside the meadow, on dry ground, clear of blocking objects."""
    inside = jnp.all((p >= 0.5) & (p <= N_TILES - 0.5), -1)
    r, c = _tile(m, p)
    dry = ~m.water[r, c]
    d_obj = jnp.linalg.norm(p[:, None, :] - m.obj_pos[None, :, :], axis=-1)  # (S, M)
    rad = RADIUS[m.obj_kind][None, :]
    hits = jnp.any((d_obj < rad + SHEEP_RADIUS) & m.obj_present[None, :] & (rad > 0), -1)
    return inside & dry & ~hits


def view_range(night: jax.Array, head_up: jax.Array) -> jax.Array:
    return jnp.where(night, jnp.where(head_up, RANGE_NIGHT_UP, RANGE_NIGHT), jnp.where(head_up, RANGE_DAY_UP, RANGE_DAY))


def n_visible(pos: jax.Array, heading: jax.Array, rng: jax.Array) -> jax.Array:
    """(S,) how many other sheep each sheep can see: in range and outside the blind wedge."""
    rel = pos[None, :, :] - pos[:, None, :]
    d = jnp.linalg.norm(rel, axis=-1)
    bearing = jnp.arctan2(rel[..., 1], -rel[..., 0])
    alpha = (bearing - heading[:, None] + jnp.pi) % (2 * jnp.pi) - jnp.pi
    seen = (d > 1e-6) & (d <= rng[:, None]) & (jnp.abs(alpha) < FOV / 2)
    return jnp.sum(seen, -1).astype(jnp.float32)


def step(state: State, actions: jax.Array) -> State:
    m, f = state.meadow, state.flock
    a = actions
    heading = f.heading + jnp.where(a == LEFT, -TURN, 0.0) + jnp.where(a == RIGHT, TURN, 0.0)
    heading = heading % (2 * jnp.pi)
    speed = jnp.where(a == WALK, WALK_SPEED, 0.0) + jnp.where(a == TROT, TROT_SPEED, 0.0)
    new = f.pos + speed[:, None] * direction(heading)

    pos = jnp.where(_allowed(m, new)[:, None], new, f.pos)
    # separation: overlapping sheep are pushed apart, half the overlap each
    diff = pos[:, None, :] - pos[None, :, :]
    d = jnp.linalg.norm(diff, axis=-1)
    eye = jnp.eye(pos.shape[0], dtype=bool)
    overlap = jnp.where(eye, 0.0, jax.nn.relu(2 * SHEEP_RADIUS - d))
    push = jnp.sum((overlap / 2)[..., None] * diff / jnp.maximum(d, 1e-3)[..., None], axis=1)
    pushed = pos + push
    pos = jnp.where(_allowed(m, pushed)[:, None], pushed, pos)

    # grazing (several sheep on one tile share it) and regrowth
    tr, tc = _tile(m, pos)
    grazing = (a == GRAZE) & (m.grass[tr, tc] >= 1.0)
    eaten = jnp.zeros_like(m.grass).at[tr, tc].add(jnp.where(grazing, BITE, 0.0))
    grass = jnp.clip(m.grass - eaten, 0.0, 3.0)
    grass = jnp.where(m.water, grass, grass + REGROW * (3.0 - grass))

    drinking = (a == DRINK) & near_water(m, pos)
    resting = a == REST
    head_up = a == HEAD_UP
    night = is_night(state.t + 1)
    cover = m.obj_present & ((m.obj_kind == TREE) | (m.obj_kind == HUT))
    d_cover = jnp.min(jnp.where(cover[None, :], jnp.linalg.norm(pos[:, None, :] - m.obj_pos[None, :, :], axis=-1), jnp.inf), -1)
    # interest: entering a cell, novelty 1/sqrt(1 + previous visits); visits counted on entry
    cell_old = jnp.clip(jnp.floor(f.pos / D.CELL).astype(jnp.int32), 0, N_TILES // D.CELL - 1)
    cell = jnp.clip(jnp.floor(pos / D.CELL).astype(jnp.int32), 0, N_TILES // D.CELL - 1)
    entered = jnp.any(cell != cell_old, -1)
    s_idx = jnp.arange(pos.shape[0])
    prev = f.visits[s_idx, cell[:, 0], cell[:, 1]]
    novelty = jnp.where(entered, 1.0 / jnp.sqrt(1.0 + prev), 0.0)
    visits = f.visits.at[s_idx, cell[:, 0], cell[:, 1]].add(entered.astype(jnp.int32))
    levels = D.update(
        f.levels,
        grazed=grazing,
        drank=drinking,
        resting=resting,
        night=night,
        near_cover=d_cover <= D.COVER_DIST,
        n_visible=n_visible(pos, heading, view_range(night, head_up)),
        novelty=novelty,
    )
    flock = Flock(pos=pos, heading=heading, resting=resting, head_up=head_up, levels=levels, visits=visits, reward=D.reward(f.levels, levels))
    return State(meadow=m._replace(grass=grass), flock=flock, t=state.t + 1)
