"""A scripted sheep for the grid world (privileged: it knows the whole map) — the baseline a
learned sheep should approach. It tends the more urgent need once that falls below ``HUNGRY``:
it walks to the nearest tile where that need can be met (beside water; on grass >= 1), greedily
by Manhattan distance among the moves not blocked, and drinks or eats there until the level
reaches ``FULL``; otherwise it stays put."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from flock.grid import world as Wd

HUNGRY, FULL = 0.6, 0.95


def _targets(s: Wd.State):
    w = s.water
    beside_water = (jnp.roll(w, 1, 0) | jnp.roll(w, -1, 0) | jnp.roll(w, 1, 1) | jnp.roll(w, -1, 1)) & ~w & ~Wd._walls()
    grass = (s.grass >= 1.0) & ~w & ~Wd._walls()
    return beside_water, grass


def _toward(s: Wd.State, target: jax.Array, key: jax.Array) -> jax.Array:
    """The unblocked compass move that most reduces the Manhattan distance to the nearest target."""
    rr, cc = jnp.arange(Wd.G)[:, None], jnp.arange(Wd.G)[None, :]
    d = jnp.where(target, jnp.abs(rr - s.pos[0]) + jnp.abs(cc - s.pos[1]), 10_000)
    i = jnp.argmin(d)
    goal = jnp.array([i // Wd.G, i % Wd.G])
    cand = jnp.clip(s.pos + Wd.MOVES[:4], 0, Wd.G - 1)
    blocked = Wd._walls()[cand[:, 0], cand[:, 1]] | s.water[cand[:, 0], cand[:, 1]]
    dist = jnp.sum(jnp.abs(cand - goal), -1) + 100 * blocked + 0.01 * jax.random.uniform(key, (4,))
    return jnp.argmin(dist)


def policy(s: Wd.State, key: jax.Array, mode: jax.Array) -> tuple[jax.Array, jax.Array]:
    """-> (action, new mode); mode 0 idle, 1 drinking, 2 eating (kept until FULL)."""
    food, water = s.levels
    beside_water, grass = _targets(s)
    r, c = s.pos
    mode = jnp.where(mode == 1, jnp.where(water >= FULL, 0, 1), mode)
    mode = jnp.where(mode == 2, jnp.where(food >= FULL, 0, 2), mode)
    start = jnp.where(water < food, jnp.where(water < HUNGRY, 1, jnp.where(food < HUNGRY, 2, 0)), jnp.where(food < HUNGRY, 2, jnp.where(water < HUNGRY, 1, 0)))
    mode = jnp.where(mode == 0, start, mode)
    at_water, at_grass = beside_water[r, c], grass[r, c]
    a_water = jnp.where(at_water, Wd.DRINK, _toward(s, beside_water, key))
    a_food = jnp.where(at_grass, Wd.EAT, _toward(s, grass, key))
    action = jnp.where(mode == 1, a_water, jnp.where(mode == 2, a_food, Wd.STAY))
    return action, mode
