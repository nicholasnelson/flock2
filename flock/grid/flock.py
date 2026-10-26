"""Track B's world (``docs/flock-2/trackB-grid-communication.md``): the grid with several sheep and a
signal channel, as a bench for communication machinery. The one-sheep grid (``flock.grid.world``)
is unchanged; its rules for needs, eating, drinking and taste are reused.

- A ``G`` x ``G`` grid walled at the border, one pond, fertile clumps; ``K`` sheep, each on its own
  tile (sheep may share a tile; there is no collision).
- Each sheep sees an ``V`` x ``V`` top-down window around itself; other sheep appear in it as
  ``SHEEP`` cells, so a sheep can see and follow another.
- Each step every sheep moves (or eats, drinks, stays) and **emits a signal** of ``N_SIGNAL``
  numbers in [-1, 1]. A sheep **hears** the mean signal of the other sheep within ``EARSHOT``
  tiles (Chebyshev distance) and how many it hears (as a fraction of K - 1). Sound has no direction:
  where a caller is can be learned only by seeing it.
- What a sheep feels: food, water, taste (as the one-sheep grid) and what it hears (``N_HEARD``).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from flock.grid import world as W1

G = 24
V = 8
K = 4
N_SIGNAL = 2
N_HEARD = N_SIGNAL + 1
EARSHOT = 6
SHEEP = W1.N_CLASSES  # cell class of another sheep in a view
N_CLASSES = W1.N_CLASSES + 1
N_ACTIONS = W1.N_ACTIONS
N_FELT = W1.N_FELT + N_HEARD
FOOD_DECAY, WATER_DECAY = W1.FOOD_DECAY, W1.WATER_DECAY


class State(NamedTuple):
    water: jax.Array  # (G, G) bool
    fertile: jax.Array  # (G, G) bool
    grass: jax.Array  # (G, G) float
    pos: jax.Array  # (K, 2) int
    levels: jax.Array  # (K, 2) food, water
    taste: jax.Array  # (K,)
    reward: jax.Array  # (K,)
    signal: jax.Array  # (K, N_SIGNAL) last emitted
    heard: jax.Array  # (K, N_HEARD) what each sheep heard after the last step
    t: jax.Array  # ()


def _walls():
    r = jnp.arange(G)
    edge = (r == 0) | (r == G - 1)
    return edge[:, None] | edge[None, :]


def generate(key: jax.Array, levels: jax.Array | None = None) -> State:
    """A pond of 2-3 x 2-3 tiles, fertile clumps on a quarter of the dry ground, K sheep on random dry
    tiles, levels drawn from [0.2, 1.0] unless given."""
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    ph, pw = jax.random.randint(k1, (2,), 2, 4)
    pr = jax.random.randint(k2, (), 1, G - 1 - 3)
    pc = jax.random.randint(k3, (), 1, G - 1 - 3)
    rr, cc = jnp.arange(G)[:, None], jnp.arange(G)[None, :]
    water = (rr >= pr) & (rr < pr + ph) & (cc >= pc) & (cc < pc + pw)
    walls = _walls()
    noise = jax.random.uniform(k4, (G, G))
    smooth = (noise + jnp.roll(noise, 1, 0) + jnp.roll(noise, -1, 0) + jnp.roll(noise, 1, 1) + jnp.roll(noise, -1, 1)) / 5
    dry = ~water & ~walls
    thresh = jnp.quantile(jnp.where(dry, smooth, -1.0), 1 - W1.FERTILE_SHARE * dry.mean())
    fertile = dry & (smooth >= thresh)
    score = jnp.where(dry, jax.random.uniform(k5, (G, G)), -1.0)
    _, idx = jax.lax.top_k(score.reshape(-1), K)
    pos = jnp.stack([idx // G, idx % G], -1)
    lv = jax.random.uniform(k6, (K, 2), minval=0.2, maxval=1.0) if levels is None else levels
    z = jnp.zeros((K,))
    return State(water, fertile, jnp.where(fertile, 3.0, 0.0), pos, lv, z, z, jnp.zeros((K, N_SIGNAL)), jnp.zeros((K, N_HEARD)), jnp.int32(0))


def near_water(water, pos):
    r, c = pos[..., 0], pos[..., 1]
    return water[r - 1, c] | water[r + 1, c] | water[r, c - 1] | water[r, c + 1]


def hear(pos: jax.Array, signal: jax.Array) -> jax.Array:
    """(K, N_HEARD): the mean signal of the others within EARSHOT, and the fraction of others heard."""
    d = jnp.max(jnp.abs(pos[:, None, :] - pos[None, :, :]), -1)
    near = (d <= EARSHOT) & ~jnp.eye(pos.shape[0], dtype=bool)
    n = near.sum(-1)
    mean = jnp.sum(near[..., None] * signal[None, :, :], 1) / jnp.maximum(n, 1)[:, None]
    return jnp.concatenate([mean, (n / (pos.shape[0] - 1))[:, None]], -1)


def step(s: State, actions: jax.Array, signals: jax.Array) -> State:
    """Every sheep acts and emits at once. Several sheep eating one tile each take a bite."""
    a = actions
    target = jnp.clip(s.pos + W1.MOVES[a], 0, G - 1)
    blocked = _walls()[target[:, 0], target[:, 1]] | s.water[target[:, 0], target[:, 1]]
    pos = jnp.where(blocked[:, None], s.pos, target)
    r, c = s.pos[:, 0], s.pos[:, 1]
    ate = (a == W1.EAT) & (s.grass[r, c] >= 1.0)
    drank = (a == W1.DRINK) & near_water(s.water, s.pos)
    bad = ((a == W1.EAT) & ~ate) | ((a == W1.DRINK) & ~drank)
    grass = s.grass.at[r, c].add(jnp.where(ate, -W1.BITE, 0.0))
    grass = jnp.clip(grass, 0.0, 3.0)
    grass = jnp.where(s.fertile, jnp.minimum(grass + W1.REGROW, 3.0), grass)
    food = s.levels[:, 0] - FOOD_DECAY + jnp.where(ate, W1.FOOD_PER_BITE, 0.0)
    water = s.levels[:, 1] - WATER_DECAY + jnp.where(drank, W1.WATER_PER_SIP, 0.0)
    levels = jnp.clip(jnp.stack([food, water], -1), 0.0, W1.MAX_LEVEL)
    signals = jnp.clip(signals, -1.0, 1.0)
    return State(s.water, s.fertile, grass, pos, levels, -bad.astype(jnp.float32), -W1.drive(levels) - W1.TASTE_COST * bad, signals, hear(pos, signals), s.t + 1)


def classes(s: State) -> jax.Array:
    c = jnp.where(s.grass >= 2.0, W1.LUSH, jnp.where(s.grass >= 1.0, W1.GRASS, W1.BARE))
    c = jnp.where(s.water, W1.WATER, c)
    return jnp.where(_walls(), W1.WALL, c).astype(jnp.int32)


def views(s: State) -> jax.Array:
    """(K, V, V) each sheep's window (it sits at row and column V // 2), other sheep as SHEEP."""
    base = classes(s)

    def one(k):
        others = jnp.zeros((G, G), bool).at[s.pos[:, 0], s.pos[:, 1]].set(True).at[s.pos[k, 0], s.pos[k, 1]].set(
            jnp.any(jnp.all(s.pos == s.pos[k], -1) & (jnp.arange(K) != k)))
        grid = jnp.where(others, SHEEP, base)
        padded = jnp.pad(grid, V, constant_values=W1.WALL)
        return jax.lax.dynamic_slice(padded, (s.pos[k, 0] + V - V // 2, s.pos[k, 1] + V - V // 2), (V, V))

    return jax.vmap(one)(jnp.arange(K))


def felt(s: State) -> jax.Array:
    """(K, N_FELT): food, water, taste, and what was heard."""
    return jnp.concatenate([s.levels, s.taste[:, None], s.heard], -1)


def sees_water(s: State) -> jax.Array:
    """(K,) whether the pond is in each sheep's window."""
    return jnp.any(views(s) == W1.WATER, (-2, -1))
