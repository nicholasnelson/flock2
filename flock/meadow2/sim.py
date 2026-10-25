"""Meadow v2 dynamics (spec § 3, § 5).

As ``flock.meadow.sim`` (continuous movement, refused moves into water, out of the meadow or into
blocking objects, separation pushes, grazing, regrowth, day and night), with these changes:

- 8 actions: head up is gone.
- Grazing and drinking act on the tile under the **mouth point**, ``MOUTH`` tiles straight ahead of
  the sheep's centre. Grazing needs grass of at least 1 there, and drinking needs water there.
- **Light** follows a smooth curve (``light``): 1 at midday, ``NIGHT_LIGHT`` at midnight, with
  dawn and dusk ramps of ``RAMP`` steps. "Night" for warmth and cover is still the last 40 % of
  the day.
- Another sheep counts as **visible** (for the social level) if it is alive, outside the rear
  blind wedge, and within 4 + 20 x light tiles.
- **Death** (``drives.update``): a dead sheep stays where it fell, is frozen, takes no part in
  grazing, pushing or being seen, and earns 0.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from flock.meadow2 import drives as D
from flock.meadow2.world import CELL, DAY, HUT, N_TILES, RADIUS, SHEEP_RADIUS, TREE, Flock, Meadow, State

STOP, WALK, TROT, LEFT, RIGHT, GRAZE, DRINK, REST = range(8)
N_ACTIONS = 8
ACTION_NAMES = ("stop", "walk", "trot", "left", "right", "graze", "drink", "rest")
TURN = jnp.pi / 8
WALK_SPEED, TROT_SPEED = 0.12, 0.35
MOUTH = 0.6  # tiles ahead of the sheep's centre
BITE = 0.05
REGROW = 0.0004
DAY_FRACTION = 0.6
NIGHT_LIGHT = 0.12
RAMP = 120  # steps for dawn and for dusk
FOV = jnp.deg2rad(300.0)


def is_night(t: jax.Array) -> jax.Array:
    return ((t % DAY) / DAY) >= DAY_FRACTION


def light(t: jax.Array) -> jax.Array:
    """Light level in [NIGHT_LIGHT, 1]: smooth ramps centred on dawn (step 0) and dusk (0.6 DAY)."""
    s = (t % DAY).astype(jnp.float32)
    dusk = DAY_FRACTION * DAY
    # signed distance to the nearest dawn or dusk: positive in daylight, negative at night
    into = jnp.where(s < dusk, jnp.minimum(s, dusk - s), -jnp.minimum(s - dusk, DAY - s))
    u = jnp.clip(into / RAMP + 0.5, 0.0, 1.0)
    return NIGHT_LIGHT + (1.0 - NIGHT_LIGHT) * u * u * (3.0 - 2.0 * u)


def direction(heading: jax.Array) -> jax.Array:
    return jnp.stack([-jnp.cos(heading), jnp.sin(heading)], -1)


def tile(p: jax.Array) -> tuple[jax.Array, jax.Array]:
    i = jnp.clip(jnp.floor(p).astype(jnp.int32), 0, N_TILES - 1)
    return i[..., 0], i[..., 1]


def mouth(pos: jax.Array, heading: jax.Array) -> jax.Array:
    """(S, 2) the mouth points."""
    return pos + MOUTH * direction(heading)


def mouth_tile(m: Meadow, pos: jax.Array, heading: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """(on water, grass biomass, inside the meadow) at each sheep's mouth point."""
    p = mouth(pos, heading)
    inside = jnp.all((p >= 0.0) & (p < N_TILES), -1)
    r, c = tile(p)
    return m.water[r, c] & inside, jnp.where(inside, m.grass[r, c], 0.0), inside


def _allowed(m: Meadow, p: jax.Array) -> jax.Array:
    inside = jnp.all((p >= 0.5) & (p <= N_TILES - 0.5), -1)
    r, c = tile(p)
    d_obj = jnp.linalg.norm(p[:, None, :] - m.obj_pos[None, :, :], axis=-1)
    rad = RADIUS[m.obj_kind][None, :]
    hits = jnp.any((d_obj < rad + SHEEP_RADIUS) & m.obj_present[None, :] & (rad > 0), -1)
    return inside & ~m.water[r, c] & ~hits


def n_visible(pos: jax.Array, heading: jax.Array, alive: jax.Array, rng: jax.Array) -> jax.Array:
    rel = pos[None, :, :] - pos[:, None, :]
    d = jnp.linalg.norm(rel, axis=-1)
    bearing = jnp.arctan2(rel[..., 1], -rel[..., 0])
    alpha = (bearing - heading[:, None] + jnp.pi) % (2 * jnp.pi) - jnp.pi
    seen = (d > 1e-6) & (d <= rng) & (jnp.abs(alpha) < FOV / 2) & alive[None, :]
    return jnp.sum(seen, -1).astype(jnp.float32)


def step(state: State, actions: jax.Array) -> State:
    m, f = state.meadow, state.flock
    alive = f.alive
    a = jnp.where(alive, actions, STOP)
    heading = f.heading + jnp.where(a == LEFT, -TURN, 0.0) + jnp.where(a == RIGHT, TURN, 0.0)
    heading = jnp.where(alive, heading % (2 * jnp.pi), f.heading)
    speed = jnp.where(a == WALK, WALK_SPEED, 0.0) + jnp.where(a == TROT, TROT_SPEED, 0.0)
    new = f.pos + speed[:, None] * direction(heading)
    pos = jnp.where(_allowed(m, new)[:, None], new, f.pos)
    # separation among the living
    diff = pos[:, None, :] - pos[None, :, :]
    d = jnp.linalg.norm(diff, axis=-1)
    both = alive[:, None] & alive[None, :] & ~jnp.eye(pos.shape[0], dtype=bool)
    overlap = jnp.where(both, jax.nn.relu(2 * SHEEP_RADIUS - d), 0.0)
    pushed = pos + jnp.sum((overlap / 2)[..., None] * diff / jnp.maximum(d, 1e-3)[..., None], axis=1)
    pos = jnp.where((_allowed(m, pushed) & alive)[:, None], pushed, pos)
    # mouth: grazing and drinking
    m_water, m_grass, m_inside = mouth_tile(m, pos, heading)
    grazing = (a == GRAZE) & (m_grass >= 1.0) & ~m_water & alive
    drinking = (a == DRINK) & m_water & alive
    bad_taste = ((a == DRINK) & ~drinking) | ((a == GRAZE) & ~grazing)
    bad_taste = bad_taste & alive
    mp = mouth(pos, heading)
    mr, mc = tile(mp)
    eaten = jnp.zeros_like(m.grass).at[mr, mc].add(jnp.where(grazing, BITE, 0.0))
    grass = jnp.clip(m.grass - eaten, 0.0, 3.0)
    grass = jnp.where(m.water, grass, grass + REGROW * (3.0 - grass))
    # the rest of the body
    night = is_night(state.t + 1)
    cover = m.obj_present & ((m.obj_kind == TREE) | (m.obj_kind == HUT))
    d_cover = jnp.min(jnp.where(cover[None, :], jnp.linalg.norm(pos[:, None, :] - m.obj_pos[None, :, :], axis=-1), jnp.inf), -1)
    n_cells = N_TILES // CELL
    cell_old = jnp.clip(jnp.floor(f.pos / CELL).astype(jnp.int32), 0, n_cells - 1)
    cell = jnp.clip(jnp.floor(pos / CELL).astype(jnp.int32), 0, n_cells - 1)
    entered = jnp.any(cell != cell_old, -1) & alive
    s_idx = jnp.arange(pos.shape[0])
    prev = f.visits[s_idx, cell[:, 0], cell[:, 1]]
    novelty = jnp.where(entered, 1.0 / jnp.sqrt(1.0 + prev), 0.0)
    visits = f.visits.at[s_idx, cell[:, 0], cell[:, 1]].add(entered.astype(jnp.int32))
    levels, fat, died = D.update(
        f.levels,
        f.fat,
        action=a,
        grazed=grazing,
        drank=drinking,
        resting=a == REST,
        night=night,
        near_cover=d_cover <= D.COVER_DIST,
        n_visible=n_visible(pos, heading, alive, 4.0 + 20.0 * light(state.t + 1)),
        novelty=novelty,
    )
    levels = jnp.where(alive[:, None], levels, f.levels)
    fat = jnp.where(alive, fat, f.fat)
    alive_next = alive & ~died
    flock = Flock(
        pos=pos,
        heading=heading,
        resting=(a == REST) & alive,
        alive=alive_next,
        levels=levels,
        fat=fat,
        visits=visits,
        reward=D.reward(levels, bad_taste, alive_next),
        taste=-bad_taste.astype(jnp.float32),
    )
    return State(meadow=m._replace(grass=grass), flock=flock, t=state.t + 1)
