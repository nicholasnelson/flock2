"""Homeostatic drives and the reward (Milestone 1, step 3).

Homeostatic RL (Keramati & Gutkin 2014, https://elifesciences.org/articles/04811): the internal
state H is a vector of physiological levels with set-points; the drive is a norm of the
deviations, D(H) = (sum_i |h*_i - h_i|^n)^(1/n). A large exponent n makes the norm close to the
worst deficit, so topping up an already small deficit pays less than fixing the largest one (the
research notes' structural guard against gaming a single drive). Levels may overshoot the
set-point a little, and overshoot counts as deviation (over-satiation is penalised).

The reward of a step is -D(H_{t+1}), the drive after it. Keramati & Gutkin define the reward as
the drive reduction D(H_t) - D(H_{t+1}) and prove that maximising its discounted sum is the same
as minimising the discounted drive; directly, sum_t g^t (D_t - D_{t+1}) =
D_0 - ((1 - g) / g) sum_{t>=1} g^t D_t, so the two rewards rank policies identically, but the
drive-reduction form carries the signal at a scale of (1 - g) ~ 0.003 and is flat once levels
bottom out. The first agent run on drive reduction (``drive_reduction``, scaled by 100 as in the
deep-HRRL work the notes cite) collapsed with drives at zero, so the reward is now the drive
itself (M1 step 3, owner's decision).

Levels (set-point 1 for all):
- energy: falls slowly; grazing restores it in proportion to the grass under the sheep;
- hydration: falls a little faster; drinking at the water's edge restores it;
- warmth: falls slowly by day, fast at night in the open, less fast at night near cover; only
  resting restores it (faster beside cover);
- social: an instantaneous level, not a store — the number of other sheep in view (in range and
  outside the rear blind wedge), saturating at ``SOCIAL_SATURATION``;
- interest: falls slowly (boredom); entering a little-visited cell of the meadow (hashed visit
  counts over static 8 x 8-tile cells, blind to other sheep) restores it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

NAMES = ("energy", "hydration", "warmth", "social", "interest")
N_DRIVES = len(NAMES)
ENERGY, HYDRATION, WARMTH, SOCIAL, INTEREST = range(N_DRIVES)
SETPOINT = jnp.ones(N_DRIVES)
MAX_LEVEL = 1.15  # overshoot allowed (and penalised as deviation)
NORM_N = 4.0
REWARD_SCALE = 100.0  # for drive_reduction

ENERGY_DECAY, ENERGY_PER_BITE = 0.0006, 0.012  # a bite is BITE of biomass at grass level >= 1
HYDRATION_DECAY, HYDRATION_PER_SIP = 0.0008, 0.04
WARMTH_DECAY_DAY, WARMTH_DECAY_NIGHT_OPEN, WARMTH_DECAY_NIGHT_COVER = 0.0002, 0.0016, 0.0007
WARMTH_REST, WARMTH_REST_COVER = 0.003, 0.006
COVER_DIST = 3.0  # tiles to the nearest tree or hut that count as "near cover"
SOCIAL_SATURATION = 8.0
INTEREST_DECAY, INTEREST_NOVELTY = 0.0004, 0.08
CELL = 8  # tiles per interest cell (128 / 8 = 16 cells a side)


def drive(h: jax.Array) -> jax.Array:
    """D(H) for levels (..., N_DRIVES)."""
    return jnp.sum(jnp.abs(SETPOINT - h) ** NORM_N, -1) ** (1.0 / NORM_N)


def drive_reduction(h_before: jax.Array, h_after: jax.Array) -> jax.Array:
    """Keramati & Gutkin's reward form (used in the first agent run)."""
    return REWARD_SCALE * (drive(h_before) - drive(h_after))


def reward(h_before: jax.Array, h_after: jax.Array) -> jax.Array:
    """The reward of a step: minus the drive after it (``h_before`` is unused)."""
    return -drive(h_after)


def update(h, *, grazed, drank, resting, night, near_cover, n_visible, novelty):
    """New levels (S, N_DRIVES) from the step's events (each (S,) except ``night`` ())."""
    energy = h[:, ENERGY] - ENERGY_DECAY + jnp.where(grazed, ENERGY_PER_BITE, 0.0)
    hydration = h[:, HYDRATION] - HYDRATION_DECAY + jnp.where(drank, HYDRATION_PER_SIP, 0.0)
    w_decay = jnp.where(night, jnp.where(near_cover, WARMTH_DECAY_NIGHT_COVER, WARMTH_DECAY_NIGHT_OPEN), WARMTH_DECAY_DAY)
    w_gain = jnp.where(resting, jnp.where(near_cover, WARMTH_REST_COVER, WARMTH_REST), 0.0)
    warmth = h[:, WARMTH] - w_decay + w_gain
    social = jnp.minimum(n_visible / SOCIAL_SATURATION, 1.0)
    interest = h[:, INTEREST] - INTEREST_DECAY + INTEREST_NOVELTY * novelty
    out = jnp.stack([energy, hydration, warmth, social, interest], -1)
    return jnp.clip(out, 0.0, MAX_LEVEL)
