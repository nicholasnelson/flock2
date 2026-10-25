"""The scripted flock for meadow v2: privileged rules, not a learned policy (spec § 7).

Rules, in priority order, per living sheep:
1. Urgent thirst or hunger (hydration, or gut energy, below ``URGENT``): go for water or grass,
   even at night, trotting if the target is more than ``TROT_FAR`` tiles away.
2. Night: walk to nearby cover (a stable preference among the near trees and huts) and rest
   within ``COVER_REST_DIST`` of it.
3. Thirsty (hydration below 1 - ``THIRSTY``): walk to the nearest water. Drink once the mouth point
   is on water, and keep drinking until hydration reaches ``FULL``.
4. Hungry (gut energy below 1 - ``HUNGRY``): graze where the mouth point has grass, unless a clearly
   richer patch is three tiles away. Keep grazing good grass until the gut reaches ``FULL``.
5. Otherwise wander with some persistence, drifting back toward the flock beyond
   ``COHESION_DIST`` tiles.

Steering picks, of 16 headings, the one nearest the target whose next 1.2 tiles are clear of
objects, the fence and (unless it is going to drink) water; it turns toward that, then walks.
Changes from ``flock.meadow.scripted``:
- obstacle avoidance (steering straight at a target stranded sheep behind trees and the hut, and
  sent starving sheep into ponds: 38 % died in the first check, 2026-10-26);
- urgent needs override night (the old flock went to shelter at dusk even when dying of thirst);
- walking is the rule, and trotting only for urgent, far targets (the old flock trotted to any
  target more than 10 tiles away);
- the mouth point replaces "within a tile of water";
- head up is gone.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from flock.meadow2 import drives as D, sim
from flock.meadow2.world import HUT, N_TILES, RADIUS, SHEEP_RADIUS, TREE, State

THIRSTY, HUNGRY = 0.6, 0.3
URGENT = 0.25
FULL = 0.95
TROT_FAR = 6.0
COVER_REST_DIST = 3.0
COHESION_DIST = 5.0
EPS = 0.05


# tiles checked ahead when choosing a heading; the first two are a walk and a trot step (without
# them, a sheep pressed against the hut heading along its edge saw the far points clear and
# stayed stuck)
LOOKAHEAD = jnp.array([0.12, 0.35, 0.7, 1.2])
N_DIRS = 16  # candidate headings, every 22.5 degrees


def _clear(m, pos, dirs, water_ok):
    """(S, N_DIRS) whether the LOOKAHEAD points along each candidate direction are inside the
    meadow and clear of blocking objects, and (unless ``water_ok``, for a sheep heading to drink)
    dry. Where no direction is clear that far (a tight gap, as between treeline trees), only the
    first two points (one walk or trot step) are required."""
    p = pos[:, None, None, :] + LOOKAHEAD[None, None, :, None] * dirs[:, :, None, :]  # (S, D, L, 2)
    inside = jnp.all((p >= 0.5) & (p <= N_TILES - 0.5), -1)
    d_obj = jnp.linalg.norm(p[..., None, :] - m.obj_pos[None, None, None, :, :], axis=-1)
    rad = RADIUS[m.obj_kind]
    hit = jnp.any((d_obj < rad + SHEEP_RADIUS) & m.obj_present & (rad > 0), -1)
    r, c = sim.tile(p)
    wet = m.water[r, c] & ~water_ok[:, None, None]
    ok = inside & ~hit & ~wet
    far, near = jnp.all(ok, -1), jnp.all(ok[..., :2], -1)
    return jnp.where(far.any(-1, keepdims=True), far, near)


def _steer(m, pos, heading: jax.Array, vec: jax.Array, trot: jax.Array, water_ok: jax.Array | None = None) -> jax.Array:
    """Turn toward, then walk (trot) along, the clear heading nearest the direction of ``vec``."""
    water_ok = jnp.zeros(heading.shape, bool) if water_ok is None else water_ok
    cand = heading[:, None] + sim.TURN * (jnp.arange(N_DIRS) - N_DIRS // 2)[None, :]
    ok = _clear(m, pos, sim.direction(cand), water_ok)
    want = jnp.arctan2(vec[..., 1], -vec[..., 0])
    off = jnp.abs((cand - want[:, None] + jnp.pi) % (2 * jnp.pi) - jnp.pi)
    pick = jnp.argmin(jnp.where(ok, off, jnp.inf), -1)
    goal = jnp.where(ok.any(-1), cand[jnp.arange(heading.shape[0]), pick], heading + sim.TURN)
    diff = (goal - heading + jnp.pi) % (2 * jnp.pi) - jnp.pi
    turn = jnp.where(diff < 0, sim.LEFT, sim.RIGHT)
    return jnp.where(jnp.abs(diff) > sim.TURN / 2, turn, jnp.where(trot, sim.TROT, sim.WALK))


def policy(state: State, key: jax.Array) -> jax.Array:
    """(S,) actions for the scripted flock (dead sheep's actions are ignored by the simulator)."""
    m, f = state.meadow, state.flock
    n = f.pos.shape[0]
    k1, k3, k4 = jax.random.split(key, 3)
    hydration, energy = f.levels[:, D.HYDRATION], f.levels[:, D.ENERGY]
    urgent_water, urgent_food = hydration < URGENT, energy < URGENT

    # cover
    cover = m.obj_present & ((m.obj_kind == TREE) | (m.obj_kind == HUT))
    to_obj = m.obj_pos[None, :, :] - f.pos[:, None, :]
    d_obj = jnp.where(cover[None, :], jnp.linalg.norm(to_obj, axis=-1), jnp.inf)
    pref = 4.0 * jax.random.uniform(jax.random.PRNGKey(7), d_obj.shape)  # stable per sheep and cover
    j = jnp.argmin(d_obj + pref, -1)
    to_cover, d_cover = to_obj[jnp.arange(n), j], d_obj[jnp.arange(n), j]
    night_act = jnp.where(d_cover > COVER_REST_DIST, _steer(m, f.pos, f.heading, to_cover, jnp.zeros(n, bool)), sim.REST)

    # water
    m_water, m_grass, _ = sim.mouth_tile(m, f.pos, f.heading)
    yy, xx = jnp.mgrid[0:N_TILES, 0:N_TILES] + 0.5
    wpos = jnp.stack([yy.ravel(), xx.ravel()], -1)
    d_w = jnp.where(m.water.ravel()[None, :], jnp.linalg.norm(wpos[None, :, :] - f.pos[:, None, :], axis=-1), jnp.inf)
    to_water, dist_w = wpos[jnp.argmin(d_w, -1)] - f.pos, jnp.min(d_w, -1)
    water_act = jnp.where(m_water, sim.DRINK, _steer(m, f.pos, f.heading, to_water, urgent_water & (dist_w > TROT_FAR), jnp.ones(n, bool)))

    # grass: at the mouth, or the richest of eight points three tiles away
    ang = jnp.arange(8) * (jnp.pi / 4)
    ring = f.pos[:, None, :] + 3.0 * jnp.stack([-jnp.cos(ang), jnp.sin(ang)], -1)[None]
    ri = jnp.clip(ring.astype(jnp.int32), 0, N_TILES - 1)
    g_ring = jnp.where(m.water[ri[..., 0], ri[..., 1]], -1.0, m.grass[ri[..., 0], ri[..., 1]])
    best = ring[jnp.arange(n), jnp.argmax(g_ring, -1)] - f.pos
    here = jnp.where(m_water, -1.0, m_grass)
    richer = jnp.max(g_ring, -1) > here + 1.0
    food_act = jnp.where((here >= 1.0) & ~richer, sim.GRAZE, _steer(m, f.pos, f.heading, best, urgent_food & richer))

    # wander, with cohesion toward the living flock
    alive_w = f.alive.astype(jnp.float32)[:, None]
    centre = jnp.sum(f.pos * alive_w, 0) / jnp.maximum(alive_w.sum(), 1.0)
    to_c = centre[None, :] - f.pos
    u = jax.random.uniform(k1, (n,))
    wander = jnp.where(u < 0.12, sim.LEFT, jnp.where(u < 0.24, sim.RIGHT, jnp.where(u < 0.3, sim.STOP, sim.WALK)))
    wander = jnp.where(here >= 2.5, jnp.where(u < 0.5, sim.GRAZE, wander), wander)
    idle_act = jnp.where(jnp.linalg.norm(to_c, axis=-1) > COHESION_DIST, _steer(m, f.pos, f.heading, to_c, jnp.zeros(n, bool)), wander)

    night = sim.is_night(state.t)
    want_water = (1.0 - hydration > THIRSTY) | (m_water & (hydration < FULL))
    want_food = (1.0 - energy > HUNGRY) | ((here >= 1.5) & (energy < FULL))
    act = jnp.where(want_food, food_act, idle_act)
    act = jnp.where(want_water, water_act, act)
    act = jnp.where(night, night_act, act)
    act = jnp.where(urgent_food, food_act, act)
    act = jnp.where(urgent_water, water_act, act)
    rand = jax.random.randint(k3, (n,), 0, sim.N_ACTIONS)
    act = jnp.where(jax.random.uniform(k4, (n,)) < EPS, rand, act)
    # out of a turning loop: a sheep that is turning walks instead, a fifth of the time, when the
    # step ahead is clear
    ahead_ok = _clear(m, f.pos, sim.direction(f.heading)[:, None, :], jnp.zeros(n, bool))[:, 0]
    turning = (act == sim.LEFT) | (act == sim.RIGHT)
    act = jnp.where(turning & ahead_ok & (jax.random.uniform(jax.random.fold_in(k1, 1), (n,)) > 0.8), sim.WALK, act)
    return act.astype(jnp.int32)
