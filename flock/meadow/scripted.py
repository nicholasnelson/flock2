"""A scripted flock: privileged rules, not a learned policy. It makes the meadow watchable before
anything is learned, and its rollouts are the logged data the world model is pre-trained on
(plan, Milestone 1, step 2).

Rules, in priority order, per sheep:
1. Night: walk to nearby cover (a tree or hut; each sheep has a stable preference among the
   near ones, so the flock spreads over a copse rather than queueing at one tree) and rest
   within a few tiles of it.
2. Thirsty: walk to the nearest water and drink.
3. Hungry: graze where there is grass, unless a clearly richer patch is a few tiles away
   (then head for the richest of a ring of nearby points).
4. Otherwise wander with some persistence, drifting back towards the flock when more than six
   tiles from its centre.
Steering turns toward the target direction, then walks (trots if the target is far). A little
randomness (occasional random actions, occasional head-up) keeps the flock from moving in lock-step.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from flock.meadow import drives as D, sim
from flock.meadow.world import HUT, N_TILES, TREE, State

THIRSTY, HUNGRY = 0.6, 0.3
FULL = 0.95  # keep drinking or grazing, once started, until the level reaches this
COVER_REST_DIST = 3.0
COHESION_DIST = 6.0
EPS = 0.05


def _steer(heading: jax.Array, vec: jax.Array, far: jax.Array) -> jax.Array:
    """Action turning toward ``vec`` (row, col), else walking (trotting if ``far``)."""
    want = jnp.arctan2(vec[..., 1], -vec[..., 0])
    diff = (want - heading + jnp.pi) % (2 * jnp.pi) - jnp.pi
    turn = jnp.where(diff < 0, sim.LEFT, sim.RIGHT)
    go = jnp.where(far, sim.TROT, sim.WALK)
    return jnp.where(jnp.abs(diff) > sim.TURN / 2, turn, go)


def policy(state: State, key: jax.Array) -> jax.Array:
    """(S,) actions for the scripted flock."""
    m, f = state.meadow, state.flock
    n = f.pos.shape[0]
    k1, k2, k3, k4 = jax.random.split(key, 4)

    # nearest cover
    cover = m.obj_present & ((m.obj_kind == TREE) | (m.obj_kind == HUT))
    to_obj = m.obj_pos[None, :, :] - f.pos[:, None, :]
    d_obj = jnp.where(cover[None, :], jnp.linalg.norm(to_obj, axis=-1), jnp.inf)
    taste = 4.0 * jax.random.uniform(jax.random.PRNGKey(7), d_obj.shape)  # stable per sheep and cover
    j = jnp.argmin(d_obj + taste, -1)
    to_cover, d_cover = to_obj[jnp.arange(n), j], d_obj[jnp.arange(n), j]
    night_act = jnp.where(d_cover > COVER_REST_DIST, _steer(f.heading, to_cover, d_cover > 10.0), sim.REST)

    # nearest water (whole grid; cheap on the GPU)
    yy, xx = jnp.mgrid[0:N_TILES, 0:N_TILES] + 0.5
    wpos = jnp.stack([yy.ravel(), xx.ravel()], -1)
    d_w = jnp.where(m.water.ravel()[None, :], jnp.linalg.norm(wpos[None, :, :] - f.pos[:, None, :], axis=-1), jnp.inf)
    to_water = wpos[jnp.argmin(d_w, -1)] - f.pos
    thirsty_act = jnp.where(sim.near_water(m, f.pos), sim.DRINK, _steer(f.heading, to_water, jnp.min(d_w, -1) > 10.0))

    # grass: here, or the richest of eight points three tiles away
    r = jnp.clip(f.pos.astype(jnp.int32), 0, N_TILES - 1)
    here = m.grass[r[:, 0], r[:, 1]]
    ang = jnp.arange(8) * (jnp.pi / 4)
    ring = f.pos[:, None, :] + 3.0 * jnp.stack([-jnp.cos(ang), jnp.sin(ang)], -1)[None]
    ri = jnp.clip(ring.astype(jnp.int32), 0, N_TILES - 1)
    g_ring = jnp.where(m.water[ri[..., 0], ri[..., 1]], -1.0, m.grass[ri[..., 0], ri[..., 1]])
    best = ring[jnp.arange(n), jnp.argmax(g_ring, -1)] - f.pos
    richer = jnp.max(g_ring, -1) > here + 1.0
    hungry_act = jnp.where((here >= 1.0) & ~richer, sim.GRAZE, _steer(f.heading, best, jnp.zeros(n, bool)))

    # wander, with cohesion
    centre = jnp.mean(f.pos, 0)
    to_c = centre[None, :] - f.pos
    far_from_flock = jnp.linalg.norm(to_c, axis=-1) > COHESION_DIST
    u = jax.random.uniform(k1, (n,))
    wander = jnp.where(u < 0.12, sim.LEFT, jnp.where(u < 0.24, sim.RIGHT, jnp.where(u < 0.3, sim.STOP, sim.WALK)))
    wander = jnp.where(here >= 2.5, jnp.where(u < 0.5, sim.GRAZE, wander), wander)
    idle_act = jnp.where(far_from_flock, _steer(f.heading, to_c, jnp.zeros(n, bool)), wander)

    night = sim.is_night(state.t)
    thirst, hunger = 1.0 - f.levels[:, D.HYDRATION], 1.0 - f.levels[:, D.ENERGY]
    # hysteresis without memory: once at the water (on good grass), keep drinking (grazing) until nearly full
    topping_up_water = sim.near_water(m, f.pos) & (thirst > 1.0 - FULL)
    topping_up_grass = (here >= 1.5) & (hunger > 1.0 - FULL)
    want_water = (thirst > THIRSTY) | topping_up_water
    want_grass = (hunger > HUNGRY) | topping_up_grass
    act = jnp.where(night, night_act, jnp.where(want_water, thirsty_act, jnp.where(want_grass, hungry_act, idle_act)))
    act = jnp.where(jax.random.uniform(k2, (n,)) < 0.02, sim.HEAD_UP, act)
    rand = jax.random.randint(k3, (n,), 0, sim.N_ACTIONS)
    act = jnp.where(jax.random.uniform(k4, (n,)) < EPS, rand, act)
    return act.astype(jnp.int32)
