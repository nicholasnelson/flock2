"""Meadow v2 state and generation (spec § 2).

As in ``flock.meadow.world``: the ground is a tile grid (grass biomass 0-3 per tile, water tiles);
objects and sheep have float positions in tile units (row, col) and sheep a float heading
(radians, 0 = north = -row, clockwise). New: 48 x 48 tiles, contents scaled to the area, 12
sheep, body fat, an ``alive`` flag, and a flock that always starts within reach of water: its
centre is 6-14 tiles from the nearest water (``START_WATER``).

v2.1 (spec § 11): grass is a smooth field on half-tile cells (``GRASS_RES``), and water is a
smooth field too (``water_at``). ``grass_at`` blends the four nearest cell centres bilinearly. A
bite crops a patch: every cell within ``PATCH_RADIUS`` of the mouth point (``patch``), and its
nutrition is the patch's mean density (``patch_density``). So what the sheep sees under its
mouth, what it eats and what it depletes agree, and a grazed patch shows as a bare spot. Days are ``DAY_SCALE`` times longer (``DAY``). Sheep are
placed so that none overlaps another, a tree or the hut.

All functions are pure and jittable; generate several meadows with vmap.

**Stage 1 (spec § 20): willows.** ``generate(..., willows=k)`` stands k willows (a new object kind,
``WILLOW``: tall, pale, drooping, drawn by ``render``) on each pond's bank, so that water can be seen
from far. A world fact, told to nobody: a sheep must learn that willows mean water. Willows block
movement like trees and are **not** cover for warmth (so resting by the pond gains nothing; the
landmark is only a landmark). ``willows=0`` is meadow v2.1 unchanged.

**Stage 2a (spec § 21): terrain.** Every meadow carries two numbers the sim reads each step:
``regrow`` (the share of the gap to full that grass regrows per step; ``REGROW`` by default, as it
was) and ``trample`` (biomass every grass cell under a moving sheep loses per step; 0 by default).
``with_terrain`` sets them on a meadow or a batch, so a curriculum can change the world live.
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
GRASS_RES = 2  # grass cells per tile (v2.1: half-tile cells, so a grazed patch can show)
N_CELLS = N_TILES * GRASS_RES
PATCH_RADIUS = 0.4  # tiles: a bite crops every grass cell within this of the mouth point
DAY_SCALE = 2.0  # v2.1: days are this many times the first meadow's 1,200 steps
DAY = int(1200 * DAY_SCALE)  # steps per day (sim.DAY); here so random_start need not import sim

SALT, TREE, FLOWER, HUT, MUSHROOM = (S.OBJECT_NAMES.index(n) for n in ("salt", "tree", "flower", "hut", "mushroom"))
# kind, count: about 128 x 128's contents scaled by the area (48 / 128)^2 ~ 0.14 (spec § 2)
LAYOUT = ((TREE, 14), (HUT, 1), (SALT, 1), (FLOWER, 6), (MUSHROOM, 4))
N_OBJECTS = sum(n for _, n in LAYOUT)
N_TREELINE = 8  # of the trees, along the north edge; the rest in two copses
WILLOW = S.N_OBJECT_TYPES  # stage 1's landmark kind, beyond the sprite set's kinds
RADIUS = jnp.zeros(S.N_OBJECT_TYPES + 1).at[TREE].set(0.35).at[HUT].set(0.7).at[SALT].set(0.35).at[WILLOW].set(0.35)
WILLOW_BANK = 0.9  # tiles outside the pond's radius
SHEEP_RADIUS = 0.3

N_PONDS = 2
REGROW = 0.00013  # fraction of the gap to full (3) regrown per step: a time constant of ~7,700 steps
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
    grass: jax.Array  # (N_CELLS, N_CELLS) float biomass in [0, 3], half-tile cells
    obj_kind: jax.Array  # (M,) int
    obj_variant: jax.Array  # (M,) int
    obj_pos: jax.Array  # (M, 2) float, tile units
    obj_present: jax.Array  # (M,) bool
    regrow: jax.Array = jnp.float32(REGROW)  # () grass regrowth per step (stage 2a's first dial)
    trample: jax.Array = jnp.float32(0.0)  # () biomass lost per step under a moving sheep (the second)


class Flock(NamedTuple):
    pos: jax.Array  # (S, 2) float
    heading: jax.Array  # (S,) float radians
    resting: jax.Array  # (S,) bool
    alive: jax.Array  # (S,) bool
    levels: jax.Array  # (S, N_DRIVES) energy (the gut), hydration, warmth, interest
    fat: jax.Array  # (S,) body fat in [0, 1] (drives.FAT_CAPACITY units of gut energy when full)
    visits: jax.Array  # (S, N / CELL, N / CELL) int
    reward: jax.Array  # (S,) the reward of the last step
    taste: jax.Array  # (S,) -1 after a bad mouthful, else 0
    seen: jax.Array  # (S,) other sheep in view (for analysis; no longer a drive)


class State(NamedTuple):
    meadow: Meadow
    flock: Flock
    t: jax.Array  # () int, steps since dawn of day 0


def ground_state(m: Meadow) -> jax.Array:
    """(N, N) int ground state per tile (water, or the grass state of the tile's mean density), for
    displays that want one value per tile."""
    g = m.grass.reshape(N_TILES, GRASS_RES, N_TILES, GRASS_RES).mean((1, 3))
    return jnp.where(m.water, WATER, jnp.clip(jnp.floor(g), 0, N_GRASS_STATES - 1).astype(jnp.int32))


def _bilinear(p, res, n):
    """Indices (..., 4, 2) and weights (..., 4) of the four grid points around p (tiles), on a grid of
    ``res`` points per tile (point i at (i + 0.5) / res)."""
    q = p * res - 0.5
    f = jnp.floor(q)
    u = q - f
    corners = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    idx = jnp.clip(f.astype(jnp.int32)[..., None, :] + corners, 0, n - 1)
    wr = jnp.stack([1 - u[..., 0], 1 - u[..., 0], u[..., 0], u[..., 0]], -1)
    wc = jnp.stack([1 - u[..., 1], u[..., 1], 1 - u[..., 1], u[..., 1]], -1)
    return idx, wr * wc


def grass_at(grass: jax.Array, p: jax.Array) -> jax.Array:
    """(...,) grass density at points p (..., 2): bilinear between grass-cell centres."""
    i, w = _bilinear(p, GRASS_RES, N_CELLS)
    return jnp.sum(grass[i[..., 0], i[..., 1]] * w, -1)


def water_at(water: jax.Array, p: jax.Array) -> jax.Array:
    """(...,) bool: whether points p (..., 2) are in water. The water mask is blended bilinearly
    between tile centres and cut at one half, so shorelines are smooth curves rather than tile
    edges (v2.1); a straight shore still runs along the tile boundary."""
    i, w = _bilinear(p, 1, N_TILES)
    inside = jnp.all((p >= 0.0) & (p < N_TILES), -1)
    return inside & (jnp.sum(water[i[..., 0], i[..., 1]].astype(jnp.float32) * w, -1) >= 0.5)


def cell_centres() -> jax.Array:
    """(N_CELLS, N_CELLS, 2) grass-cell centres in tiles."""
    yy, xx = jnp.mgrid[0:N_CELLS, 0:N_CELLS] + 0.5
    return jnp.stack([yy, xx], -1) / GRASS_RES


def dry_cells(water: jax.Array) -> jax.Array:
    """(N_CELLS, N_CELLS) bool: grass cells on dry ground."""
    return ~water_at(water, cell_centres())


_PATCH = jnp.stack(jnp.meshgrid(jnp.arange(-2, 3), jnp.arange(-2, 3), indexing="ij"), -1).reshape(-1, 2)  # 5 x 5 cells


def patch(p: jax.Array) -> tuple[jax.Array, jax.Array]:
    """(indices (..., 25, 2), mask (..., 25)): the grass cells whose centres lie within
    ``PATCH_RADIUS`` tiles of p, the patch a bite crops (v2.1). At least one cell always does."""
    c = jnp.floor(p * GRASS_RES).astype(jnp.int32)
    idx = c[..., None, :] + _PATCH
    centre = (idx + 0.5) / GRASS_RES
    ok = (jnp.linalg.norm(centre - p[..., None, :], axis=-1) <= PATCH_RADIUS) & jnp.all((idx >= 0) & (idx < N_CELLS), -1)
    return jnp.clip(idx, 0, N_CELLS - 1), ok


def patch_density(grass: jax.Array, p: jax.Array) -> jax.Array:
    """(...,) mean grass density over the patch at p."""
    idx, ok = patch(p)
    g = grass[idx[..., 0], idx[..., 1]]
    return jnp.sum(jnp.where(ok, g, 0.0), -1) / jnp.maximum(ok.sum(-1), 1)


def _grid():
    yy, xx = jnp.mgrid[0:N_TILES, 0:N_TILES] + 0.5
    return yy, xx


def _blobs(key, n_blobs, r_min, r_max):
    """(N_CELLS, N_CELLS) field on the grass cells: the maximum of random round blobs."""
    k1, k2 = jax.random.split(key)
    c = jax.random.uniform(k1, (n_blobs, 2), minval=0.0, maxval=N_TILES)
    r = jax.random.uniform(k2, (n_blobs,), minval=r_min, maxval=r_max)
    q = cell_centres()
    d2 = (q[None, ..., 0] - c[:, 0, None, None]) ** 2 + (q[None, ..., 1] - c[:, 1, None, None]) ** 2
    return jnp.max(jnp.exp(-d2 / (2 * r[:, None, None] ** 2)), axis=0)


def water_distance(water: jax.Array) -> jax.Array:
    """(N, N) distance (tiles) from each tile centre to the nearest water tile centre (inf if none)."""
    yy, xx = _grid()
    p = jnp.stack([yy.ravel(), xx.ravel()], -1)
    d = jnp.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    d = jnp.where(water.ravel()[None, :], d, jnp.inf)
    return jnp.min(d, -1).reshape(N_TILES, N_TILES)


def _ponds(key):
    """Water: N_PONDS round ponds. Returns (water (N, N) bool, centres (N_PONDS, 2), radii (N_PONDS,))."""
    k1, k2 = jax.random.split(key)
    c = jax.random.uniform(k1, (N_PONDS, 2), minval=5.0, maxval=N_TILES - 5.0)
    r = jax.random.uniform(k2, (N_PONDS,), minval=POND_RADIUS[0], maxval=POND_RADIUS[1])
    yy, xx = _grid()
    d = jnp.sqrt((yy[None] - c[:, 0, None, None]) ** 2 + (xx[None] - c[:, 1, None, None]) ** 2)
    return jnp.any(d < r[:, None, None], 0), c, r


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


def generate(key: jax.Array, n_sheep: int = N_SHEEP, centre: jax.Array | None = None, willows: int = 0) -> State:
    """A meadow, with the flock grouped around ``centre`` or (default) a centre 6-14 tiles from the
    nearest water; levels 0.6-0.9 (energy, hydration, warmth) and 0.8 (interest), fat 0.5,
    at dawn. Of N_CANDIDATES pond layouts, the first that admits such a start is used.
    ``willows`` > 0 stands that many willows on each pond's bank (stage 1)."""
    k_g, k_b, k_w, k_o, k_s, k_v, k_l = jax.random.split(key, 7)
    n = N_TILES
    grass = jnp.clip(1.2 + 1.8 * _blobs(k_g, 14, 1.5, 3.75) - 1.4 * _blobs(k_b, 8, 0.75, 1.9), 0.0, 3.0)

    kw1, kw2 = jax.random.split(k_w)
    cands, pond_c, pond_r = jax.vmap(_ponds)(jax.random.split(kw1, N_CANDIDATES))
    ok, starts = jax.vmap(_choose)(jax.random.split(kw2, N_CANDIDATES), cands)
    pick = jnp.where(ok.any(), jnp.argmax(ok), 0)
    water, pond_c, pond_r = cands[pick], pond_c[pick], pond_r[pick]
    centre = starts[pick] if centre is None else centre
    grass = jnp.where(dry_cells(water), grass, 0.0)  # no grass under water (the field fades at the shore)

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
    if willows > 0:
        theta = jax.random.uniform(k_l, (N_PONDS, willows), maxval=2 * jnp.pi)
        ring = jnp.stack([jnp.cos(theta), jnp.sin(theta)], -1) * (pond_r[:, None, None] + WILLOW_BANK)
        kinds.append(jnp.full((N_PONDS * willows,), WILLOW))
        poss.append((pond_c[:, None, :] + ring).reshape(-1, 2))
    obj_kind = jnp.concatenate(kinds)
    obj_pos = jnp.clip(jnp.concatenate(poss), 1.0, n - 1.0)
    obj_present = ~water[obj_pos[:, 0].astype(jnp.int32), obj_pos[:, 1].astype(jnp.int32)]
    obj_variant = jax.random.randint(k_v, (obj_kind.shape[0],), 0, 4)

    # each sheep at the first of 32 random spots around the centre that is dry, clear of blocking
    # objects (a sheep placed inside a tree's footprint could never move) and, since sheep are solid
    # (v2.1), clear of the sheep placed before it
    spots = jnp.clip(centre + 2.0 * jax.random.normal(k_s, (n_sheep, 32, 2)), 1.0, n - 1.0)
    d_obj = jnp.linalg.norm(spots[:, :, None, :] - obj_pos[None, None], axis=-1)
    rad = RADIUS[obj_kind]
    free = ~water_at(water, spots) & ~jnp.any((d_obj < rad + SHEEP_RADIUS) & obj_present & (rad > 0), -1)

    def place(placed, k):
        d = jnp.linalg.norm(spots[k][:, None, :] - placed[None, :, :], axis=-1)  # (32, S)
        ok = free[k] & jnp.all(d >= 2 * SHEEP_RADIUS, -1)
        p = spots[k, jnp.argmax(ok)]
        return placed.at[k].set(p), None

    pos, _ = jax.lax.scan(place, jnp.full((n_sheep, 2), -100.0), jnp.arange(n_sheep))
    flock = Flock(
        pos=pos,
        heading=jax.random.uniform(jax.random.fold_in(k_s, 1), (n_sheep,), maxval=2 * jnp.pi),
        resting=jnp.zeros((n_sheep,), bool),
        alive=jnp.ones((n_sheep,), bool),
        levels=jnp.concatenate([jax.random.uniform(jax.random.fold_in(k_s, 2), (n_sheep, 3), minval=0.6, maxval=0.9), jnp.full((n_sheep, 1), 0.8)], -1),
        fat=jnp.full((n_sheep,), 0.5),
        visits=jnp.zeros((n_sheep, n // CELL, n // CELL), jnp.int32),
        reward=jnp.zeros((n_sheep,)),
        taste=jnp.zeros((n_sheep,)),
        seen=jnp.zeros((n_sheep,)),
    )
    meadow = Meadow(water=water, grass=grass, obj_kind=obj_kind, obj_variant=obj_variant, obj_pos=obj_pos, obj_present=obj_present)
    return State(meadow=meadow, flock=flock, t=jnp.int32(0))


def with_terrain(state: State, regrow: float | jax.Array | None = None, trample: float | jax.Array | None = None) -> State:
    """The same state (or batch of states) with new terrain numbers."""
    m = state.meadow
    like = lambda v, old: jnp.broadcast_to(jnp.asarray(v, jnp.float32), old.shape)
    if regrow is not None:
        m = m._replace(regrow=like(regrow, m.regrow))
    if trample is not None:
        m = m._replace(trample=like(trample, m.trample))
    return state._replace(meadow=m)


def random_start(key: jax.Array, n_sheep: int = N_SHEEP, willows: int = 0) -> State:
    """A new meadow with the flock at a random place, a random time of day, levels drawn from
    [0.2, 1.0] and fat from [0.2, 0.8] (the agents' varied starts)."""
    k1, k3, k4, k5 = jax.random.split(key, 4)
    st = generate(k1, n_sheep, willows=willows)
    levels = jax.random.uniform(k3, st.flock.levels.shape, minval=0.2, maxval=1.0)
    fat = jax.random.uniform(k5, (n_sheep,), minval=0.2, maxval=0.8)
    return st._replace(flock=st.flock._replace(levels=levels, fat=fat), t=jax.random.randint(k4, (), 0, DAY))
