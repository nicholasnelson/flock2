"""Meadow v2 dynamics (spec §§ 3, 5; v2.1 changes in § 11).

As ``flock.meadow.sim`` (continuous movement, refused moves into water, out of the meadow or into
blocking objects, grazing, regrowth, day and night), with these changes:

- 8 actions: head up is gone.
- Grazing and drinking act at the **mouth point**, ``MOUTH`` tiles straight ahead. Drinking needs a
  water tile there. Grazing reads the smooth grass field there (``world.grass_at``): it needs a
  density of at least ``MIN_BITE``, gives energy in proportion to the density, and takes ``BITE``
  of biomass from the four nearest tiles in proportion to their weights (v2.1).
- **Grass regrows on a scale of days** (v2.1): ``REGROW`` of the gap to full per step, a time
  constant of about 3 days.
- **Light** follows a smooth curve (``light``), with dawn and dusk ramps of ``RAMP`` steps.
- **Sheep are solid** (v2.1): a move that would bring a sheep closer to another living sheep while
  overlapping it is refused, as walking into a tree is. Nobody is pushed. Previously overlapping
  sheep were pushed apart, half the overlap each, which moved sheep that had not acted.
- **Huddling** (v2.1): resting sheep within ``drives.HUDDLE_DIST`` of each other keep warmer. The
  number of other sheep in view is still counted (``Flock.seen``) but is no longer a drive.
- **Death** (``drives.update``): a dead sheep is frozen, takes no part in anything, and earns 0.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from flock.meadow2 import drives as D
from flock.meadow2.world import CELL, DAY, DAY_SCALE, HUT, N_TILES, RADIUS, SHEEP_RADIUS, TREE, Flock, Meadow, State, bite_weights, grass_at

STOP, WALK, TROT, LEFT, RIGHT, GRAZE, DRINK, REST = range(8)
N_ACTIONS = 8
ACTION_NAMES = ("stop", "walk", "trot", "left", "right", "graze", "drink", "rest")
TURN = jnp.pi / 8
WALK_SPEED, TROT_SPEED = 0.12, 0.35
MOUTH = 0.6
BITE = 0.05
MIN_BITE = 0.5  # grass density needed for a mouthful
REGROW = 0.00013  # fraction of the gap to full (3) regrown per step: a time constant of ~7,700 steps
DAY_FRACTION = 0.6
NIGHT_LIGHT = 0.12
RAMP = int(120 * DAY_SCALE)  # steps for dawn and for dusk
FOV = jnp.deg2rad(300.0)
SEEN_RANGE = (4.0, 20.0)  # another sheep is "seen" within 4 + 20 x light tiles (analysis only)


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
    return pos + MOUTH * direction(heading)


def mouth_tile(m: Meadow, pos: jax.Array, heading: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """(on water, grass density, inside the meadow) at each sheep's mouth point."""
    p = mouth(pos, heading)
    inside = jnp.all((p >= 0.0) & (p < N_TILES), -1)
    r, c = tile(p)
    water = m.water[r, c] & inside
    return water, jnp.where(inside & ~water, grass_at(m.grass, p), 0.0), inside


def _allowed(m: Meadow, p: jax.Array) -> jax.Array:
    inside = jnp.all((p >= 0.5) & (p <= N_TILES - 0.5), -1)
    r, c = tile(p)
    d_obj = jnp.linalg.norm(p[:, None, :] - m.obj_pos[None, :, :], axis=-1)
    rad = RADIUS[m.obj_kind][None, :]
    hits = jnp.any((d_obj < rad + SHEEP_RADIUS) & m.obj_present[None, :] & (rad > 0), -1)
    return inside & ~m.water[r, c] & ~hits


def solid(old: jax.Array, new: jax.Array, alive: jax.Array, rounds: int = 3) -> jax.Array:
    """Refuse moves that bring a sheep closer to another living sheep it would overlap; a refused
    sheep stays put, which can refuse another's move in the next round."""
    both = alive[:, None] & alive[None, :] & ~jnp.eye(old.shape[0], dtype=bool)
    d_old = jnp.linalg.norm(old[:, None, :] - old[None, :, :], axis=-1)
    moved = jnp.any(new != old, -1)
    for _ in range(rounds):
        d_new = jnp.linalg.norm(new[:, None, :] - new[None, :, :], axis=-1)
        bad = both & (d_new < 2 * SHEEP_RADIUS) & (d_new < d_old - 1e-6)
        revert = moved & jnp.any(bad, -1)
        new = jnp.where(revert[:, None], old, new)
        moved = moved & ~revert
    return new


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
    pos = solid(f.pos, jnp.where(_allowed(m, new)[:, None], new, f.pos), alive)
    # the mouth: drinking, and grazing on the smooth field
    m_water, density, _ = mouth_tile(m, pos, heading)
    grazing = (a == GRAZE) & (density >= MIN_BITE) & alive
    drinking = (a == DRINK) & m_water & alive
    bad_taste = (((a == DRINK) & ~drinking) | ((a == GRAZE) & ~grazing)) & alive
    idx, w = bite_weights(mouth(pos, heading))
    take = jnp.where(grazing[:, None], BITE * w, 0.0)
    grass = jnp.clip(m.grass - jnp.zeros_like(m.grass).at[idx[..., 0], idx[..., 1]].add(take), 0.0, 3.0)
    grass = jnp.where(m.water, 0.0, grass + REGROW * (3.0 - grass))
    # the rest of the body
    night = is_night(state.t + 1)
    cover = m.obj_present & ((m.obj_kind == TREE) | (m.obj_kind == HUT))
    d_cover = jnp.min(jnp.where(cover[None, :], jnp.linalg.norm(pos[:, None, :] - m.obj_pos[None, :, :], axis=-1), jnp.inf), -1)
    resting = (a == REST) & alive
    d_sheep = jnp.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    huddle = jnp.sum((d_sheep < D.HUDDLE_DIST) & resting[None, :] & ~jnp.eye(pos.shape[0], dtype=bool), -1)
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
        bite_density=jnp.where(grazing, density, 0.0),
        drank=drinking,
        resting=resting,
        night=night,
        near_cover=d_cover <= D.COVER_DIST,
        huddle=huddle,
        novelty=novelty,
    )
    levels = jnp.where(alive[:, None], levels, f.levels)
    fat = jnp.where(alive, fat, f.fat)
    alive_next = alive & ~died
    flock = Flock(
        pos=pos,
        heading=heading,
        resting=resting,
        alive=alive_next,
        levels=levels,
        fat=fat,
        visits=visits,
        reward=D.reward(levels, bad_taste, alive_next),
        taste=-bad_taste.astype(jnp.float32),
        seen=n_visible(pos, heading, alive, SEEN_RANGE[0] + SEEN_RANGE[1] * light(state.t + 1)),
    )
    return State(meadow=m._replace(grass=grass), flock=flock, t=state.t + 1)
