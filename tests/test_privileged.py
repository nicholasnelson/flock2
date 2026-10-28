"""The myopic oracle (spec § 19): the water-range mask and the two ideal memories."""

import jax
import jax.numpy as jnp
import numpy as np

from flock.meadow2 import privileged as P, sim
from flock.meadow2.world import random_start

W_OFF = 6 + 3 + 1  # where the water features start in the observation


def _state():
    return random_start(jax.random.PRNGKey(3))


def _face(st, i, target):
    """Turn sheep i to face a world point."""
    v = target - st.flock.pos[i]
    h = jnp.arctan2(v[1], -v[0])  # heading 0 = north (-y), clockwise
    return st._replace(flock=st.flock._replace(heading=st.flock.heading.at[i].set(h)))


def test_full_perception_is_unchanged_and_sizes():
    st = _state()
    obs = P.observe(st)
    assert obs.shape == (st.flock.pos.shape[0], P.N_OBS)
    assert P.n_obs(P.Perception()) == P.N_OBS
    assert P.n_obs(P.Perception(8.0, "rot", 30)) == P.N_OBS + P.N_MEMORY
    full = P.observe(st, P.Perception(water_range=1000.0))
    assert np.allclose(np.asarray(full), np.asarray(obs), atol=1e-5)


def test_range_mask_hides_far_and_behind():
    st = _state()
    vec, dist, found = P.visible_water(st, None)
    assert bool(found.all())
    # face the nearest water: within range it is seen; with a range below its distance it is not
    i = int(jnp.argmin(dist))
    st = _face(st, i, st.flock.pos[i] + vec[i])
    near = P.observe(st, P.Perception(water_range=float(dist[i]) + 0.5))
    far = P.observe(st, P.Perception(water_range=float(dist[i]) - 0.5))
    assert near[i, W_OFF] > 0.9  # forward component: straight ahead
    assert np.allclose(np.asarray(far[i, W_OFF : W_OFF + 3]), [0.0, 0.0, 1.0])
    # turn the sheep's back on it (heading + pi): the blind wedge hides it even within range
    back = st._replace(flock=st.flock._replace(heading=st.flock.heading.at[i].add(jnp.pi)))
    hidden = P.observe(back, P.Perception(water_range=float(dist[i]) + 0.5))
    assert np.allclose(np.asarray(hidden[i, W_OFF : W_OFF + 3]), [0.0, 0.0, 1.0])


def test_rot_memory_turns_with_the_sheep_and_expires():
    st = _state()
    cfg = P.Perception(water_range=12.0, memory="rot", window=30)
    vec, dist, _ = P.visible_water(st, None)
    i = int(jnp.argmin(dist))
    st = _face(st, i, st.flock.pos[i] + vec[i])
    mem = P.init_memory(st.flock.pos.shape[0])
    obs, mem = P.observe_mem(st, mem, cfg)
    assert obs[i, P.N_OBS] > 0.9 and abs(float(obs[i, P.N_OBS + 1])) < 0.1 and float(obs[i, P.N_OBS + 3]) == 0.0
    # eight left turns (180 degrees) with the water out of range: the memory says "behind"
    hidden = P.Perception(water_range=0.0, memory="rot", window=30)
    turned = st._replace(flock=st.flock._replace(heading=st.flock.heading.at[i].add(-8 * sim.TURN)), t=st.t + 8)
    obs2, mem2 = P.observe_mem(turned, mem, hidden)
    assert obs2[i, P.N_OBS] < -0.9
    assert abs(float(obs2[i, P.N_OBS + 3]) - 8 / 30) < 1e-5
    # four left turns (90 degrees): the water is now on the right (left-negative)
    quarter = st._replace(flock=st.flock._replace(heading=st.flock.heading.at[i].add(-4 * sim.TURN)), t=st.t + 4)
    obs3, _ = P.observe_mem(quarter, mem, hidden)
    assert obs3[i, P.N_OBS + 1] < -0.9
    # past the window: masked
    old = turned._replace(t=st.t + 31)
    obs4, _ = P.observe_mem(old, mem, hidden)
    assert np.allclose(np.asarray(obs4[i, P.N_OBS :]), [0.0, 0.0, 1.0, 1.0])
    # never seen: masked
    obs5, _ = P.observe_mem(st, P.init_memory(st.flock.pos.shape[0]), hidden)
    assert np.allclose(np.asarray(obs5[:, P.N_OBS :]), [0.0, 0.0, 1.0, 1.0])


def test_place_memory_follows_the_sheeps_travel():
    st = _state()
    cfg = P.Perception(water_range=12.0, memory="place", window=1000)
    vec, dist, _ = P.visible_water(st, None)
    i = int(jnp.argmin(dist))
    st = _face(st, i, st.flock.pos[i] + vec[i])
    mem = P.init_memory(st.flock.pos.shape[0])
    _, mem = P.observe_mem(st, mem, cfg)
    # walk the sheep past the water tile along its heading: the memory now points behind, nearer
    hidden = P.Perception(water_range=0.0, memory="place", window=1000)
    moved = st._replace(flock=st.flock._replace(pos=st.flock.pos.at[i].add(sim.direction(st.flock.heading)[i] * (dist[i] + 2.0))), t=st.t + 100)
    obs, _ = P.observe_mem(moved, mem, hidden)
    assert obs[i, P.N_OBS] < -0.9
    assert abs(float(obs[i, P.N_OBS + 2]) - 2.0 / 32) < 1e-3
    assert abs(float(obs[i, P.N_OBS + 3]) - 0.1) < 1e-5
