"""Meadow v2 dynamics (spec §§ 3, 5; v2.1 changes in § 11).

As ``flock.meadow.sim`` (continuous movement, refused moves into water, out of the meadow or into
blocking objects, grazing, regrowth, day and night), with these changes:

- 8 actions: head up is gone.
- Grazing and drinking act at the **mouth point**, ``MOUTH`` tiles straight ahead. Drinking needs
  water there. A bite crops a **patch** (v2.1): every half-tile grass cell within
  ``world.PATCH_RADIUS`` of the mouth point loses ``BITE``. It needs a mean density over the patch
  of at least ``MIN_BITE``, and gives energy in proportion to that mean.
- **Grass regrows on a scale of days** (v2.1): ``REGROW`` of the gap to full per step, a time
  constant of about 3 days.
- **Light** follows a smooth curve (``light``), with dawn and dusk ramps of ``RAMP`` steps.
- **Jostling** (v2.1, ``jostle``): overlapping sheep are pushed apart, but unevenly. A sheep that
  walked into a still one takes 70 % of the overlap (so it slides around it) and the still one 30 %.
  The first meadow pushed both by half, which shoved still sheep about; a brief v2.1 rule that
  refused such moves outright made sheep stop dead and could jam a crowd.
- **Huddling** (v2.1): resting sheep within ``drives.HUDDLE_DIST`` of each other keep warmer. The
  number of other sheep in view is still counted (``Flock.seen``) but is no longer a drive.
- **Water is a smooth field** (v2.1, ``world.water_at``): drinking, blocking and drawing all use the
  same curved shoreline.
- **Death** (``drives.update``): a dead sheep is frozen, takes no part in anything, and earns 0.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from flock.meadow2 import drives as D
from flock.meadow2.world import CELL, DAY, DAY_SCALE, HUT, N_TILES, RADIUS, SHEEP_RADIUS, TREE, Flock, Meadow, State, dry_cells, patch, patch_density, water_at

STOP, WALK, TROT, LEFT, RIGHT, GRAZE, DRINK, REST = range(8)
N_ACTIONS = 8
ACTION_NAMES = ("stop", "walk", "trot", "left", "right", "graze", "drink", "rest")
TURN = jnp.pi / 8
WALK_SPEED, TROT_SPEED = 0.12, 0.35
MOUTH = 0.6
BITE = 0.05
MIN_BITE = 0.5  # mean grass density over the patch needed for a mouthful
JOSTLE_MOVER = 0.7  # share of an overlap taken by the sheep that walked into another
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
    """(on water, mean grass density of the patch a bite would crop, inside the meadow) at each
    sheep's mouth point."""
    p = mouth(pos, heading)
    inside = jnp.all((p >= 0.0) & (p < N_TILES), -1)
    water = water_at(m.water, p) & inside
    return water, jnp.where(inside & ~water, patch_density(m.grass, p), 0.0), inside


def _allowed(m: Meadow, p: jax.Array) -> jax.Array:
    inside = jnp.all((p >= 0.5) & (p <= N_TILES - 0.5), -1)
    d_obj = jnp.linalg.norm(p[:, None, :] - m.obj_pos[None, :, :], axis=-1)
    rad = RADIUS[m.obj_kind][None, :]
    hits = jnp.any((d_obj < rad + SHEEP_RADIUS) & m.obj_present[None, :] & (rad > 0), -1)
    return inside & ~water_at(m.water, p) & ~hits


def jostle(old: jax.Array, new: jax.Array, moved: jax.Array, alive: jax.Array, allowed) -> jax.Array:
    """One relaxation step pushing overlapping living sheep apart (v2.1, after the owner's
    play-test: stopping dead at another sheep felt wrong, and could jam a crowd). A sheep that
    moved into a still one takes ``JOSTLE_MOVER`` of the overlap and the still one the rest; two
    that both moved, or neither, share it evenly. A push into water, a tree or out of the meadow is
    not taken."""
    diff = new[:, None, :] - new[None, :, :]
    d = jnp.linalg.norm(diff, axis=-1)
    both = alive[:, None] & alive[None, :] & ~jnp.eye(new.shape[0], dtype=bool)
    overlap = jnp.where(both, jax.nn.relu(2 * SHEEP_RADIUS - d), 0.0)
    mi, mj = moved[:, None], moved[None, :]
    share = jnp.where(mi & ~mj, JOSTLE_MOVER, jnp.where(~mi & mj, 1.0 - JOSTLE_MOVER, 0.5))
    push = jnp.sum((share * overlap)[..., None] * diff / jnp.maximum(d, 1e-3)[..., None], axis=1)
    pushed = new + push
    return jnp.where((allowed(pushed) & alive)[:, None], pushed, new)


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
    ok = _allowed(m, new)
    pos = jnp.where(ok[:, None], new, f.pos)
    pos = jostle(f.pos, pos, (speed > 0) & ok & alive, alive, lambda p: _allowed(m, p))
    # the mouth: drinking, and grazing on the smooth field
    m_water, density, _ = mouth_tile(m, pos, heading)
    grazing = (a == GRAZE) & (density >= MIN_BITE) & alive
    drinking = (a == DRINK) & m_water & alive
    bad_taste = (((a == DRINK) & ~drinking) | ((a == GRAZE) & ~grazing)) & alive
    idx, in_patch = patch(mouth(pos, heading))
    take = jnp.where(grazing[:, None] & in_patch, BITE, 0.0)
    grass = jnp.clip(m.grass - jnp.zeros_like(m.grass).at[idx[..., 0], idx[..., 1]].add(take), 0.0, 3.0)
    grass = jnp.where(dry_cells(m.water), grass + REGROW * (3.0 - grass), 0.0)
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
