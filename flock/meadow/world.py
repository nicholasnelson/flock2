"""Meadow state and generation.

The ground is a tile grid: grass biomass per tile (0-3, shown as the bare/short/medium/long
grass states, regrowing over time) and water tiles (ponds). Everything else is continuous:
objects (trees, huts, the salt lick, flowers, mushrooms) and sheep have float positions in tile
units (row, col); sheep have a float heading (radians, 0 = north = -row, clockwise).

All functions are pure and jittable; generate several meadows with vmap.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from flock import sprites_iso as S

N_TILES = 128
WATER = S.WATER  # ground state index of water
N_GRASS_STATES = 4

# referent kinds (indices into sprites_iso.OBJECT_NAMES)
SALT, TREE, FLOWER, FOX, HUT, POND, MUSHROOM = (S.OBJECT_NAMES.index(n) for n in ("salt", "tree", "flower", "fox", "hut", "pond", "mushroom"))

# fixed object budget per meadow (static shapes): kind, count
LAYOUT = ((TREE, 90), (HUT, 3), (SALT, 1), (FLOWER, 24), (MUSHROOM, 10))
N_OBJECTS = sum(n for _, n in LAYOUT)
# collision radius (tiles) by kind; 0 = walk-through scenery
RADIUS = jnp.zeros(S.N_OBJECT_TYPES).at[TREE].set(0.35).at[HUT].set(0.7).at[SALT].set(0.35)
SHEEP_RADIUS = 0.3


class Meadow(NamedTuple):
    """The static-ish world: ground and objects."""

    water: jax.Array  # (N, N) bool
    grass: jax.Array  # (N, N) float biomass in [0, 3]
    obj_kind: jax.Array  # (M,) int
    obj_variant: jax.Array  # (M,) int
    obj_pos: jax.Array  # (M, 2) float, tile units
    obj_present: jax.Array  # (M,) bool


class Flock(NamedTuple):
    pos: jax.Array  # (S, 2) float
    heading: jax.Array  # (S,) float radians
    resting: jax.Array  # (S,) bool
    head_up: jax.Array  # (S,) bool
    levels: jax.Array  # (S, N_DRIVES) homeostatic levels, set-point 1 (``flock.meadow.drives``)
    visits: jax.Array  # (S, 16, 16) int: each sheep's visits to the interest cells (hashed counts)
    reward: jax.Array  # (S,) the reward of the last step (drives.reward)
    taste: jax.Array  # (S,) -1 if the last step ingested nothing (drank away from water, grazed bare ground), else 0


class State(NamedTuple):
    meadow: Meadow
    flock: Flock
    t: jax.Array  # () int, steps since dawn of day 0


def ground_state(m: Meadow) -> jax.Array:
    """(N, N) int ground state for rendering: water, or the grass state from biomass."""
    g = jnp.clip(jnp.floor(m.grass), 0, N_GRASS_STATES - 1).astype(jnp.int32)
    return jnp.where(m.water, WATER, g)


def _blobs(key, n_blobs: int, r_min: float, r_max: float, n: int = N_TILES) -> jax.Array:
    """(n, n) field: sum of random round blobs (for grass patches and ponds)."""
    k1, k2 = jax.random.split(key)
    c = jax.random.uniform(k1, (n_blobs, 2), minval=0.0, maxval=n)
    r = jax.random.uniform(k2, (n_blobs,), minval=r_min, maxval=r_max)
    yy, xx = jnp.mgrid[0:n, 0:n] + 0.5
    d2 = (yy[None] - c[:, 0, None, None]) ** 2 + (xx[None] - c[:, 1, None, None]) ** 2
    return jnp.max(jnp.exp(-d2 / (2 * r[:, None, None] ** 2)), axis=0)


def generate(key: jax.Array, n_sheep: int = 40) -> State:
    """A meadow: grass with richer and barer patches, two or three ponds, a treeline along the
    north edge plus copses, a few huts, the salt lick, scattered flowers and mushrooms, and a
    flock placed together on dry ground near the middle."""
    k_g, k_b, k_w, k_o, k_s, k_v = jax.random.split(key, 6)
    n = N_TILES
    rich = _blobs(k_g, 14, 4.0, 10.0)
    bare = _blobs(k_b, 8, 2.0, 5.0)
    grass = jnp.clip(1.2 + 1.8 * rich - 1.4 * bare, 0.0, 3.0)
    ponds = _blobs(k_w, 3, 2.5, 4.5)
    water = ponds > 0.6
    # keep the flock's starting area dry
    yy, xx = jnp.mgrid[0:n, 0:n] + 0.5
    water = water & (((yy - n / 2) ** 2 + (xx - n / 2) ** 2) > 10.0**2)

    # objects
    kinds, poss = [], []
    keys = jax.random.split(k_o, len(LAYOUT))
    for (kind, count), kk in zip(LAYOUT, keys):
        ka, kb, kc = jax.random.split(kk, 3)
        if kind == TREE:
            # two thirds along the north edge (a treeline), the rest in a few copses
            n_line = 2 * count // 3
            line = jnp.stack([jax.random.uniform(ka, (n_line,), minval=2.0, maxval=10.0), jax.random.uniform(kb, (n_line,), minval=2.0, maxval=n - 2.0)], -1)
            centres = jax.random.uniform(kc, (4, 2), minval=20.0, maxval=n - 20.0)
            pick = jax.random.randint(ka, (count - n_line,), 0, 4)
            copse = centres[pick] + 3.0 * jax.random.normal(kb, (count - n_line, 2))
            p = jnp.concatenate([line, copse])
        else:
            p = jax.random.uniform(ka, (count, 2), minval=6.0, maxval=n - 6.0)
        kinds.append(jnp.full((count,), kind))
        poss.append(p)
    obj_kind = jnp.concatenate(kinds)
    obj_pos = jnp.clip(jnp.concatenate(poss), 1.0, n - 1.0)
    on_water = water[obj_pos[:, 0].astype(jnp.int32), obj_pos[:, 1].astype(jnp.int32)]
    obj_present = ~on_water
    obj_variant = jax.random.randint(k_v, (N_OBJECTS,), 0, 4)  # training variants only

    # flock: a loose group around the middle
    sp = n / 2 + 4.0 * jax.random.normal(k_s, (n_sheep, 2))
    heading = jax.random.uniform(jax.random.fold_in(k_s, 1), (n_sheep,), maxval=2 * jnp.pi)
    flock = Flock(
        pos=sp,
        heading=heading,
        resting=jnp.zeros((n_sheep,), bool),
        head_up=jnp.zeros((n_sheep,), bool),
        levels=jnp.concatenate([jax.random.uniform(jax.random.fold_in(k_s, 2), (n_sheep, 3), minval=0.6, maxval=0.9), jnp.full((n_sheep, 2), 0.8)], -1),
        visits=jnp.zeros((n_sheep, N_TILES // 8, N_TILES // 8), jnp.int32),
        reward=jnp.zeros((n_sheep,)),
        taste=jnp.zeros((n_sheep,)),
    )
    meadow = Meadow(water=water, grass=grass, obj_kind=obj_kind, obj_variant=obj_variant, obj_pos=obj_pos, obj_present=obj_present)
    return State(meadow=meadow, flock=flock, t=jnp.int32(0))


def random_start(key: jax.Array, n_sheep: int = 40) -> State:
    """A new meadow with the flock at a random place, a random time of day and each sheep's levels
    drawn from [0.2, 1.0] (the agent's ``--varied`` starts)."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    st = generate(k1, n_sheep)
    centre = jax.random.uniform(k2, (2,), minval=24.0, maxval=N_TILES - 24.0)
    pos = jnp.clip(st.flock.pos - N_TILES / 2 + centre, 1.0, N_TILES - 1.0)
    levels = jax.random.uniform(k3, (n_sheep, st.flock.levels.shape[1]), minval=0.2, maxval=1.0)
    return st._replace(flock=st.flock._replace(pos=pos, levels=levels), t=jax.random.randint(k4, (), 0, 1200))  # 1200 = sim.DAY (sim imports this module)
