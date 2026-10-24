"""A deliberately simple world for testing the sheep's learning machinery (world model,
imagination, actor-critic, curiosity) without the meadow's perception and scale problems.

The owner's proposal (2026-10-24): two needs, top-down vision. One sheep per world; thousands of
worlds run in parallel.

- A ``G`` x ``G`` grid walled at the border. One pond (water tiles, impassable). Fertile tiles
  carry grass (0-3) that is eaten down and regrows; other ground is bare.
- **Actions (7):** move north / south / east / west (refused into walls and water), stay, eat,
  drink. Eating works on a tile with grass >= 1 (it removes ``BITE``); drinking works beside
  water (any of the four neighbours). Both are allowed anywhere and do nothing elsewhere, so the
  "drinking on dry ground" mistake is possible — and tastes bad.
- **Needs (2):** food and water, set-point 1, clipped to [0, 1.15], falling by ``FOOD_DECAY`` and
  ``WATER_DECAY`` per step. Drive D = (|1 - food|^4 + |1 - water|^4)^(2/4) — Keramati & Gutkin's
  form with n = 4, m = 2, as the meadow's. Reward of a step: -D after it, minus ``TASTE_COST`` for
  ingesting nothing. Felt: food, water and the taste (-1 after a bad mouthful).
- **Vision:** an ``V`` x ``V`` top-down window around the sheep (it sits at row and column V/2),
  not rotated; cell classes: 0 wall or outside, 1 bare, 2 some grass (1-2), 3 lush grass (>= 2),
  4 water.

No day, night, warmth, other sheep or rendering.

``view_full`` is the whole map instead (the sheep's own tile marked ``SHEEP``), for separating
memory and search from learning (third review of 2026-10-24, S4).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

G = 16
V = 8
N_CLASSES = 5
WALL, BARE, GRASS, LUSH, WATER = range(N_CLASSES)
SHEEP = N_CLASSES  # only in view_full
N_CLASSES_FULL = N_CLASSES + 1
N_ACTIONS = 7
NORTH, SOUTH, EAST, WEST, STAY, EAT, DRINK = range(N_ACTIONS)
ACTION_NAMES = ("north", "south", "east", "west", "stay", "eat", "drink")
MOVES = jnp.array([[-1, 0], [1, 0], [0, 1], [0, -1], [0, 0], [0, 0], [0, 0]])

N_FELT = 3  # food, water, taste
FOOD_DECAY, WATER_DECAY = 0.004, 0.006
FOOD_PER_BITE, WATER_PER_SIP = 0.08, 0.1
BITE, REGROW = 1.0, 0.02  # grass removed per bite; regrown per step on fertile tiles (toward 3)
MAX_LEVEL = 1.15
NORM_N, DRIVE_M = 4.0, 2.0
TASTE_COST = 0.05
FERTILE_SHARE = 0.25


class State(NamedTuple):
    water: jax.Array  # (G, G) bool
    fertile: jax.Array  # (G, G) bool
    grass: jax.Array  # (G, G) float, 0-3
    pos: jax.Array  # (2,) int (row, col)
    levels: jax.Array  # (2,) food, water
    taste: jax.Array  # () -1 after a bad mouthful, else 0
    reward: jax.Array  # () reward of the last step
    t: jax.Array  # () int


def drive(levels: jax.Array) -> jax.Array:
    return jnp.sum(jnp.abs(1.0 - levels) ** NORM_N, -1) ** (DRIVE_M / NORM_N)


def felt(s: State) -> jax.Array:
    return jnp.concatenate([s.levels, s.taste[None]])


def _walls() -> jax.Array:
    r = jnp.arange(G)
    edge = (r == 0) | (r == G - 1)
    return edge[:, None] | edge[None, :]


def generate(key: jax.Array, levels: jax.Array | None = None) -> State:
    """A new world: a pond of 2-3 x 2-3 tiles, a quarter of the dry ground fertile (in clumps),
    the sheep on a random dry tile; levels drawn from [0.2, 1.0] unless given."""
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    ph, pw = jax.random.randint(k1, (2,), 2, 4)
    pr = jax.random.randint(k2, (), 1, G - 1 - 3)
    pc = jax.random.randint(k3, (), 1, G - 1 - 3)
    rr, cc = jnp.arange(G)[:, None], jnp.arange(G)[None, :]
    water = (rr >= pr) & (rr < pr + ph) & (cc >= pc) & (cc < pc + pw)
    walls = _walls()
    # fertile clumps: smooth random noise, keep the top share of the dry interior
    noise = jax.random.uniform(k4, (G, G))
    smooth = (noise + jnp.roll(noise, 1, 0) + jnp.roll(noise, -1, 0) + jnp.roll(noise, 1, 1) + jnp.roll(noise, -1, 1)) / 5
    dry = ~water & ~walls
    thresh = jnp.quantile(jnp.where(dry, smooth, -1.0), 1 - FERTILE_SHARE * dry.mean())
    fertile = dry & (smooth >= thresh)
    grass = jnp.where(fertile, 3.0, 0.0)
    # the sheep on a random dry tile
    score = jnp.where(dry, jax.random.uniform(k5, (G, G)), -1.0)
    i = jnp.argmax(score)
    pos = jnp.array([i // G, i % G])
    lv = jax.random.uniform(k6, (2,), minval=0.2, maxval=1.0) if levels is None else levels
    return State(water, fertile, grass, pos, lv, jnp.float32(0.0), jnp.float32(0.0), jnp.int32(0))


def near_water(s: State) -> jax.Array:
    r, c = s.pos
    nb = jnp.stack([s.water[r - 1, c], s.water[r + 1, c], s.water[r, c - 1], s.water[r, c + 1]])
    return nb.any()


def step(s: State, a: jax.Array) -> State:
    target = jnp.clip(s.pos + MOVES[a], 0, G - 1)
    blocked = _walls()[target[0], target[1]] | s.water[target[0], target[1]]
    pos = jnp.where(blocked, s.pos, target)
    r, c = s.pos
    ate = (a == EAT) & (s.grass[r, c] >= 1.0)
    drank = (a == DRINK) & near_water(s)
    bad = ((a == EAT) & ~ate) | ((a == DRINK) & ~drank)
    grass = s.grass.at[r, c].add(jnp.where(ate, -BITE, 0.0))
    grass = jnp.where(s.fertile, jnp.minimum(grass + REGROW, 3.0), grass)
    food = s.levels[0] - FOOD_DECAY + jnp.where(ate, FOOD_PER_BITE, 0.0)
    water = s.levels[1] - WATER_DECAY + jnp.where(drank, WATER_PER_SIP, 0.0)
    levels = jnp.clip(jnp.stack([food, water]), 0.0, MAX_LEVEL)
    reward = -drive(levels) - TASTE_COST * bad
    return State(s.water, s.fertile, grass, pos, levels, -bad.astype(jnp.float32), reward, s.t + 1)


def classes(s: State) -> jax.Array:
    """(G, G) cell classes of the whole world."""
    c = jnp.where(s.grass >= 2.0, LUSH, jnp.where(s.grass >= 1.0, GRASS, BARE))
    c = jnp.where(s.water, WATER, c)
    return jnp.where(_walls(), WALL, c).astype(jnp.int32)


def view(s: State) -> jax.Array:
    """(V, V) cell classes around the sheep (it sits at row and column V // 2); outside is wall."""
    padded = jnp.pad(classes(s), V, constant_values=WALL)
    return jax.lax.dynamic_slice(padded, (s.pos[0] + V - V // 2, s.pos[1] + V - V // 2), (V, V))


def view_full(s: State) -> jax.Array:
    """(G, G) the whole map, the sheep's tile marked SHEEP."""
    return classes(s).at[s.pos[0], s.pos[1]].set(SHEEP)


def observe(s: State, mode: str = "local") -> jax.Array:
    return view(s) if mode == "local" else view_full(s)
