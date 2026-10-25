"""Meadow v2 drives, body fat, death and the reward (spec § 3).

The homeostatic drive is unchanged from ``flock.meadow.drives`` (Keramati & Gutkin 2014,
https://elifesciences.org/articles/04811): D(H) = (sum_i |1 - h_i|^4)^(2/4) over the five driven
levels (energy, hydration, warmth, social, interest). Changes:

- **Energy is the gut**, and empties in about a day of standing (``ENERGY_USE``).
- **Body fat** buffers it. While the gut is above ``STORE_ABOVE``, up to ``STORE_RATE`` a step of
  gut energy is stored as fat at ``STORE_EFFICIENCY``. While the gut is below ``RELEASE_BELOW``, fat
  releases up to ``RELEASE_RATE`` a step back into the gut. Full fat (1.0) holds ``FAT_CAPACITY``
  units of gut energy, about 4 days of standing. Fat is felt but has no drive term.
- **Death**: hydration at 0, or gut energy and fat both at 0. Dead sheep are frozen and earn
  nothing.
- **Reward**: ``ALIVE_BONUS`` - D - TASTE_COST x bad taste while alive, 0 when dead. D is at most
  sqrt(5) ~ 2.24, so every step alive is worth at least +0.26 and dying never pays.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

NAMES = ("energy", "hydration", "warmth", "social", "interest")
N_DRIVES = len(NAMES)
ENERGY, HYDRATION, WARMTH, SOCIAL, INTEREST = range(N_DRIVES)
MAX_LEVEL = 1.15
NORM_N, DRIVE_M = 4.0, 2.0
TASTE_COST = 0.05
ALIVE_BONUS = 2.5
N_FELT = N_DRIVES + 2  # the five levels, fat, the last mouthful's taste

ENERGY_PER_BITE = 0.012
# gut energy used per step, in sim's action order: stop, walk, trot, left, right, graze, drink, rest
ENERGY_USE = jnp.array([0.0008, 0.0013, 0.0026, 0.0009, 0.0009, 0.0008, 0.0008, 0.0005])
FAT_CAPACITY = 4.0  # gut-energy units in full fat
STORE_ABOVE, STORE_RATE, STORE_EFFICIENCY = 0.95, 0.002, 0.7
RELEASE_BELOW, RELEASE_RATE = 0.5, 0.0008
HYDRATION_DECAY, HYDRATION_PER_SIP = 0.0008, 0.04
WARMTH_DECAY_DAY, WARMTH_DECAY_NIGHT_OPEN, WARMTH_DECAY_NIGHT_COVER = 0.0002, 0.0016, 0.0007
WARMTH_REST, WARMTH_REST_COVER = 0.003, 0.006
COVER_DIST = 3.0
SOCIAL_SATURATION = 8.0
INTEREST_DECAY, INTEREST_NOVELTY = 0.0004, 0.08


def drive(h: jax.Array) -> jax.Array:
    return jnp.sum(jnp.abs(1.0 - h) ** NORM_N, -1) ** (DRIVE_M / NORM_N)


def reward(h_after: jax.Array, bad_taste: jax.Array, alive: jax.Array) -> jax.Array:
    return jnp.where(alive, ALIVE_BONUS - drive(h_after) - TASTE_COST * bad_taste, 0.0)


def felt(levels: jax.Array, fat: jax.Array, taste: jax.Array) -> jax.Array:
    """What a sheep feels, (..., N_FELT): its five levels, its fat, the last mouthful's taste."""
    return jnp.concatenate([levels, fat[..., None], taste[..., None]], -1)


def fat_exchange(energy: jax.Array, fat: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Store surplus gut energy as fat, or release fat into a low gut."""
    store = jnp.where(fat < 1.0, jnp.clip(energy - STORE_ABOVE, 0.0, STORE_RATE), 0.0)
    release = jnp.minimum(jnp.clip(RELEASE_BELOW - energy, 0.0, RELEASE_RATE), fat * FAT_CAPACITY)
    energy = energy - store + release
    fat = jnp.clip(fat + (STORE_EFFICIENCY * store - release) / FAT_CAPACITY, 0.0, 1.0)
    return energy, fat


def update(h, fat, *, action, grazed, drank, resting, night, near_cover, n_visible, novelty):
    """New (levels (S, N_DRIVES), fat (S,), died (S,)) from the step's action and events."""
    energy = h[:, ENERGY] - ENERGY_USE[action] + jnp.where(grazed, ENERGY_PER_BITE, 0.0)
    energy, fat = fat_exchange(jnp.clip(energy, 0.0, MAX_LEVEL), fat)
    hydration = h[:, HYDRATION] - HYDRATION_DECAY + jnp.where(drank, HYDRATION_PER_SIP, 0.0)
    w_decay = jnp.where(night, jnp.where(near_cover, WARMTH_DECAY_NIGHT_COVER, WARMTH_DECAY_NIGHT_OPEN), WARMTH_DECAY_DAY)
    w_gain = jnp.where(resting, jnp.where(near_cover, WARMTH_REST_COVER, WARMTH_REST), 0.0)
    warmth = h[:, WARMTH] - w_decay + w_gain
    social = jnp.minimum(n_visible / SOCIAL_SATURATION, 1.0)
    interest = h[:, INTEREST] - INTEREST_DECAY + INTEREST_NOVELTY * novelty
    levels = jnp.clip(jnp.stack([energy, hydration, warmth, social, interest], -1), 0.0, MAX_LEVEL)
    died = (levels[:, HYDRATION] <= 0.0) | ((levels[:, ENERGY] <= 0.0) & (fat <= 0.0))
    return levels, fat, died
