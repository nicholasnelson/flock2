"""Meadow v2 drives, body fat, death and the reward (spec § 3; v2.1 changes in § 11).

The homeostatic drive (Keramati & Gutkin 2014, https://elifesciences.org/articles/04811) is
D(H) = (sum_i |1 - h_i|^4)^(2/4) over the driven levels: since v2.1, energy (the gut), hydration,
warmth and interest. **Social is no longer a drive** (owner, 2026-10-26: flocking should be learned
because it helps, not rewarded for itself). Instead warmth gains from **huddling**: at rest, other
resting sheep within ``HUDDLE_DIST`` slow the loss of warmth and speed its recovery.

Two clocks *(v2.1)*:
- **body** rates (gut energy use, hydration loss, fat storage and release) are divided by
  ``NEED_SCALE``, set so that a naive explorer usually finds water before it dies (spec § 11);
- **environment** rates (warmth by day and night, boredom) are divided by ``world.DAY_SCALE``, so a
  day or a night costs the same whatever the day length.

Body fat buffers the gut: while the gut is above ``STORE_ABOVE``, surplus is stored at
``STORE_EFFICIENCY``; while it is below ``RELEASE_BELOW``, fat is released. Full fat holds
``FAT_CAPACITY`` units of gut energy. A bite gives ``ENERGY_PER_BIOMASS`` x the grass density at the
mouth (v2.1; was a fixed 0.012). Death: hydration at 0, or gut energy and fat both at 0. Reward:
``ALIVE_BONUS`` - D - TASTE_COST x bad taste while alive, 0 when dead; D is at most sqrt(4) = 2,
so every step alive is worth at least +0.5.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from flock.meadow2.world import DAY_SCALE

NAMES = ("energy", "hydration", "warmth", "interest")
N_DRIVES = len(NAMES)
ENERGY, HYDRATION, WARMTH, INTEREST = range(N_DRIVES)
MAX_LEVEL = 1.15
NORM_N, DRIVE_M = 4.0, 2.0
TASTE_COST = 0.05
ALIVE_BONUS = 2.5
N_FELT = N_DRIVES + 2  # the four levels, fat, the last mouthful's taste

NEED_SCALE = 3.0  # body rates are divided by this; set by the naive-explorer rule (spec § 11: 53 % at 3)
ENERGY_PER_BIOMASS = 0.008  # gut energy per bite per unit of grass density (0.012 at density 1.5)
# gut energy used per step, in sim's action order: stop, walk, trot, left, right, graze, drink, rest
ENERGY_USE = jnp.array([0.0008, 0.0013, 0.0026, 0.0009, 0.0009, 0.0008, 0.0008, 0.0005]) / NEED_SCALE
FAT_CAPACITY = 4.0
STORE_ABOVE, STORE_RATE, STORE_EFFICIENCY = 0.95, 0.002 / NEED_SCALE, 0.7
RELEASE_BELOW, RELEASE_RATE = 0.5, 0.0008 / NEED_SCALE
HYDRATION_DECAY, HYDRATION_PER_SIP = 0.0008 / NEED_SCALE, 0.04
WARMTH_DECAY_DAY, WARMTH_DECAY_NIGHT_OPEN, WARMTH_DECAY_NIGHT_COVER = (r / DAY_SCALE for r in (0.0002, 0.0016, 0.0007))
WARMTH_REST, WARMTH_REST_COVER = 0.003 / DAY_SCALE, 0.006 / DAY_SCALE
COVER_DIST = 3.0
HUDDLE_DIST = 1.0  # tiles between centres (a sheep is 0.6 across)
HUDDLE_MAX = 4  # neighbours that count
HUDDLE_LOSS, HUDDLE_GAIN = 0.5, 0.25  # per neighbour: loss / (1 + 0.5 n), rest gain x (1 + 0.25 n)
INTEREST_DECAY, INTEREST_NOVELTY = 0.0004 / DAY_SCALE, 0.08


def drive(h: jax.Array) -> jax.Array:
    return jnp.sum(jnp.abs(1.0 - h) ** NORM_N, -1) ** (DRIVE_M / NORM_N)


def reward(h_after: jax.Array, bad_taste: jax.Array, alive: jax.Array) -> jax.Array:
    return jnp.where(alive, ALIVE_BONUS - drive(h_after) - TASTE_COST * bad_taste, 0.0)


def felt(levels: jax.Array, fat: jax.Array, taste: jax.Array) -> jax.Array:
    """What a sheep feels, (..., N_FELT): its four levels, its fat, the last mouthful's taste."""
    return jnp.concatenate([levels, fat[..., None], taste[..., None]], -1)


def fat_exchange(energy: jax.Array, fat: jax.Array) -> tuple[jax.Array, jax.Array]:
    store = jnp.where(fat < 1.0, jnp.clip(energy - STORE_ABOVE, 0.0, STORE_RATE), 0.0)
    release = jnp.minimum(jnp.clip(RELEASE_BELOW - energy, 0.0, RELEASE_RATE), fat * FAT_CAPACITY)
    energy = energy - store + release
    fat = jnp.clip(fat + (STORE_EFFICIENCY * store - release) / FAT_CAPACITY, 0.0, 1.0)
    return energy, fat


def update(h, fat, *, action, bite_density, drank, resting, night, near_cover, huddle, novelty):
    """New (levels (S, N_DRIVES), fat (S,), died (S,)). ``bite_density`` is the grass density
    eaten this step (0 if no bite); ``huddle`` the number of resting neighbours within HUDDLE_DIST."""
    energy = h[:, ENERGY] - ENERGY_USE[action] + ENERGY_PER_BIOMASS * bite_density
    energy, fat = fat_exchange(jnp.clip(energy, 0.0, MAX_LEVEL), fat)
    hydration = h[:, HYDRATION] - HYDRATION_DECAY + jnp.where(drank, HYDRATION_PER_SIP, 0.0)
    n = jnp.minimum(huddle, HUDDLE_MAX).astype(jnp.float32)
    w_decay = jnp.where(night, jnp.where(near_cover, WARMTH_DECAY_NIGHT_COVER, WARMTH_DECAY_NIGHT_OPEN), WARMTH_DECAY_DAY)
    w_decay = jnp.where(resting, w_decay / (1.0 + HUDDLE_LOSS * n), w_decay)
    w_gain = jnp.where(resting, jnp.where(near_cover, WARMTH_REST_COVER, WARMTH_REST) * (1.0 + HUDDLE_GAIN * n), 0.0)
    warmth = h[:, WARMTH] - w_decay + w_gain
    interest = h[:, INTEREST] - INTEREST_DECAY + INTEREST_NOVELTY * novelty
    levels = jnp.clip(jnp.stack([energy, hydration, warmth, interest], -1), 0.0, MAX_LEVEL)
    died = (levels[:, HYDRATION] <= 0.0) | ((levels[:, ENERGY] <= 0.0) & (fat <= 0.0))
    return levels, fat, died
