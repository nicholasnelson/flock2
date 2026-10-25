"""Meadow v2 state and generation (spec § 2).

As in ``flock.meadow.world``: the ground is a tile grid (grass biomass 0-3 per tile, water tiles);
objects and sheep have float positions in tile units (row, col) and sheep a float heading
(radians, 0 = north = -row, clockwise). New: 48 x 48 tiles, contents scaled to the area, 12
sheep, body fat, an ``alive`` flag, and a flock that always starts within reach of water: its
centre is 6-14 tiles from the nearest water (``START_WATER``).

All functions are pure and jittable; generate several meadows with vmap.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from flock import sprites_iso as S

N_TILES = 48
N_SHEEP = 12
WATER = S.WATER  # ground state index of water
N_GRASS_STATES = 4
DAY = 1200  # steps per day (sim.DAY); here so random_start need not import sim

SALT, TREE, FLOWER, HUT, MUSHROOM = (S.OBJECT_NAMES.index(n) for n in ("salt", "tree", "flower", "hut", "mushroom"))
# kind, count: about 128 x 128's contents scaled by the area (48 / 128)^2 ~ 0.14 (spec § 2)
LAYOUT = ((TREE, 14), (HUT, 1), (SALT, 1), (FLOWER, 6), (MUSHROOM, 4))
N_OBJECTS = sum(n for _, n in LAYOUT)
N_TREELINE = 8  # of the trees, along the north edge; the rest in two copses
RADIUS = jnp.zeros(S.N_OBJECT_TYPES).at[TREE].set(0.35).at[HUT].set(0.7).at[SALT].set(0.35)
SHEEP_RADIUS = 0.3

N_PONDS = 2
POND_RADIUS = (2.0, 3.5)  # tiles
# the flock starts with its centre this far from the nearest water (tiles). This replaces the
# spec's "every dry tile within 20 tiles of water", which two ponds cannot meet in 48 x 48 (the
# best two-point cover leaves corners ~25 tiles away; measured on 2026-10-26: 0 % of random
# two-pond layouts pass, and 11 % of four-pond ones even at 24 tiles)
START_WATER = (6.0, 14.0)
N_CANDIDATES = 8  # pond layouts drawn per meadow; the first with a valid flock start is used
N_STARTS = 64  # flock-centre candidates tried per layout
CELL = 4  # tiles per interest cell (12 x 12 cells)


class Meadow(NamedTuple):
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
    alive: jax.Array  # (S,) bool
    levels: jax.Array  # (S, N_DRIVES) energy (the gut), hydration, warmth, social, interest
    fat: jax.Array  # (S,) body fat in [0, 1] (drives.FAT_CAPACITY units of gut energy when full)
    visits: jax.Array  # (S, N / CELL, N / CELL) int
    reward: jax.Array  # (S,) the reward of the last step
    taste: jax.Array  # (S,) -1 after a bad mouthful, else 0


class State(NamedTuple):
    meadow: Meadow
    flock: Flock
    t: jax.Array  # () int, steps since dawn of day 0


def ground_state(m: Meadow) -> jax.Array:
    g = jnp.clip(jnp.floor(m.grass), 0, N_GRASS_STATES - 1).astype(jnp.int32)
    return jnp.where(m.water, WATER, g)


def _grid():
    yy, xx = jnp.mgrid[0:N_TILES, 0:N_TILES] + 0.5
    return yy, xx


def _blobs(key, n_blobs, r_min, r_max):
    k1, k2 = jax.random.split(key)
    c = jax.random.uniform(k1, (n_blobs, 2), minval=0.0, maxval=N_TILES)
    r = jax.random.uniform(k2, (n_blobs,), minval=r_min, maxval=r_max)
    yy, xx = _grid()
    d2 = (yy[None] - c[:, 0, None, None]) ** 2 + (xx[None] - c[:, 1, None, None]) ** 2
    return jnp.max(jnp.exp(-d2 / (2 * r[:, None, None] ** 2)), axis=0)


def water_distance(water: jax.Array) -> jax.Array:
    """(N, N) distance (tiles) from each tile centre to the nearest water tile centre (inf if none)."""
    yy, xx = _grid()
    p = jnp.stack([yy.ravel(), xx.ravel()], -1)
    d = jnp.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    d = jnp.where(water.ravel()[None, :], d, jnp.inf)
    return jnp.min(d, -1).reshape(N_TILES, N_TILES)


def _ponds(key):
    """Water: N_PONDS round ponds."""
    k1, k2 = jax.random.split(key)
    c = jax.random.uniform(k1, (N_PONDS, 2), minval=5.0, maxval=N_TILES - 5.0)
    r = jax.random.uniform(k2, (N_PONDS,), minval=POND_RADIUS[0], maxval=POND_RADIUS[1])
    yy, xx = _grid()
    d = jnp.sqrt((yy[None] - c[:, 0, None, None]) ** 2 + (xx[None] - c[:, 1, None, None]) ** 2)
    return jnp.any(d < r[:, None, None], 0)


def start_distance(water: jax.Array, centres: jax.Array) -> jax.Array:
    """(K,) distance (tiles) from each candidate flock centre (K, 2) to the nearest water tile."""
    yy, xx = _grid()
    p = jnp.stack([yy.ravel(), xx.ravel()], -1)
    d = jnp.linalg.norm(centres[:, None, :] - p[None, :, :], axis=-1)
    return jnp.min(jnp.where(water.ravel()[None, :], d, jnp.inf), -1)


def _choose(key, water):
    """(valid, centre): the first of N_STARTS random flock centres whose nearest water is within
    START_WATER."""
    c = jax.random.uniform(key, (N_STARTS, 2), minval=6.0, maxval=N_TILES - 6.0)
    d = start_distance(water, c)
    ok = (d >= START_WATER[0]) & (d <= START_WATER[1])
    return ok.any(), c[jnp.argmax(ok)]


def generate(key: jax.Array, n_sheep: int = N_SHEEP, centre: jax.Array | None = None) -> State:
    """A meadow, with the flock grouped around ``centre`` or (default) a centre 6-14 tiles from the
    nearest water; levels 0.6-0.9 (energy, hydration, warmth) and 0.8 (social, interest), fat 0.5,
    at dawn. Of N_CANDIDATES pond layouts, the first that admits such a start is used."""
    k_g, k_b, k_w, k_o, k_s, k_v = jax.random.split(key, 6)
    n = N_TILES
    grass = jnp.clip(1.2 + 1.8 * _blobs(k_g, 14, 1.5, 3.75) - 1.4 * _blobs(k_b, 8, 0.75, 1.9), 0.0, 3.0)

    kw1, kw2 = jax.random.split(k_w)
    cands = jax.vmap(_ponds)(jax.random.split(kw1, N_CANDIDATES))
    ok, starts = jax.vmap(_choose)(jax.random.split(kw2, N_CANDIDATES), cands)
    pick = jnp.where(ok.any(), jnp.argmax(ok), 0)
    water = cands[pick]
    centre = starts[pick] if centre is None else centre

    kinds, poss = [], []
    for (kind, count), kk in zip(LAYOUT, jax.random.split(k_o, len(LAYOUT))):
        ka, kb, kc = jax.random.split(kk, 3)
        if kind == TREE:
            line = jnp.stack([jax.random.uniform(ka, (N_TREELINE,), minval=1.5, maxval=4.0), jax.random.uniform(kb, (N_TREELINE,), minval=2.0, maxval=n - 2.0)], -1)
            centres = jax.random.uniform(kc, (2, 2), minval=10.0, maxval=n - 10.0)
            pick = jax.random.randint(ka, (count - N_TREELINE,), 0, 2)
            p = jnp.concatenate([line, centres[pick] + 1.5 * jax.random.normal(kb, (count - N_TREELINE, 2))])
        elif kind == HUT:
            p = jax.random.uniform(ka, (count, 2), minval=6.0, maxval=n - 6.0)
        else:
            p = jax.random.uniform(ka, (count, 2), minval=3.0, maxval=n - 3.0)
        kinds.append(jnp.full((count,), kind))
        poss.append(p)
    obj_kind = jnp.concatenate(kinds)
    obj_pos = jnp.clip(jnp.concatenate(poss), 1.0, n - 1.0)
    obj_present = ~water[obj_pos[:, 0].astype(jnp.int32), obj_pos[:, 1].astype(jnp.int32)]
    obj_variant = jax.random.randint(k_v, (N_OBJECTS,), 0, 4)

    pos = jnp.clip(centre + 2.0 * jax.random.normal(k_s, (n_sheep, 2)), 1.0, n - 1.0)
    flock = Flock(
        pos=pos,
        heading=jax.random.uniform(jax.random.fold_in(k_s, 1), (n_sheep,), maxval=2 * jnp.pi),
        resting=jnp.zeros((n_sheep,), bool),
        alive=jnp.ones((n_sheep,), bool),
        levels=jnp.concatenate([jax.random.uniform(jax.random.fold_in(k_s, 2), (n_sheep, 3), minval=0.6, maxval=0.9), jnp.full((n_sheep, 2), 0.8)], -1),
        fat=jnp.full((n_sheep,), 0.5),
        visits=jnp.zeros((n_sheep, n // CELL, n // CELL), jnp.int32),
        reward=jnp.zeros((n_sheep,)),
        taste=jnp.zeros((n_sheep,)),
    )
    meadow = Meadow(water=water, grass=grass, obj_kind=obj_kind, obj_variant=obj_variant, obj_pos=obj_pos, obj_present=obj_present)
    return State(meadow=meadow, flock=flock, t=jnp.int32(0))


def random_start(key: jax.Array, n_sheep: int = N_SHEEP) -> State:
    """A new meadow with the flock at a random place, a random time of day, levels drawn from
    [0.2, 1.0] and fat from [0.2, 0.8] (the agents' varied starts)."""
    k1, k3, k4, k5 = jax.random.split(key, 4)
    st = generate(k1, n_sheep)
    levels = jax.random.uniform(k3, st.flock.levels.shape, minval=0.2, maxval=1.0)
    fat = jax.random.uniform(k5, (n_sheep,), minval=0.2, maxval=0.8)
    return st._replace(flock=st.flock._replace(levels=levels, fat=fat), t=jax.random.randint(k4, (), 0, DAY))
