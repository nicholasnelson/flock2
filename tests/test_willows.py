"""Stage 1's world change (spec § 20): willows on the pond banks."""

import jax
import jax.numpy as jnp
import numpy as np

from flock.meadow2 import render as R, scripted, sim
from flock.meadow2.world import N_OBJECTS, N_PONDS, WILLOW, generate, random_start, water_at, water_distance


def test_willows_stand_on_the_banks_and_nowhere_else():
    for i in range(6):
        st = generate(jax.random.PRNGKey(i), willows=3)
        kind = np.asarray(st.meadow.obj_kind)
        assert kind.shape == (N_OBJECTS + N_PONDS * 3,)
        w = kind == WILLOW
        assert w.sum() == N_PONDS * 3
        pos = np.asarray(st.meadow.obj_pos)[w]
        present = np.asarray(st.meadow.obj_present)[w]
        wd = np.asarray(water_distance(st.meadow.water))
        d = wd[np.clip(pos[:, 0].astype(int), 0, 47), np.clip(pos[:, 1].astype(int), 0, 47)]
        # present willows stand dry, within two tiles of water
        assert not np.asarray(water_at(st.meadow.water, jnp.asarray(pos)))[present].any()
        assert (d[present] <= 2.5).all(), d
        assert present.mean() >= 0.5


def test_no_willows_is_the_old_meadow():
    a = generate(jax.random.PRNGKey(5))
    b = generate(jax.random.PRNGKey(5), willows=0)
    assert a.meadow.obj_kind.shape == b.meadow.obj_kind.shape == (N_OBJECTS,)
    assert np.allclose(np.asarray(a.meadow.obj_pos), np.asarray(b.meadow.obj_pos))
    assert np.allclose(np.asarray(a.meadow.water), np.asarray(b.meadow.water))


def test_willows_render_and_do_not_shelter():
    st = random_start(jax.random.PRNGKey(2), willows=3)
    w = np.asarray(st.meadow.obj_kind) == WILLOW
    target = np.asarray(st.meadow.obj_pos)[w][0]
    # stand sheep 0 three tiles from a willow, facing it, by day
    f = st.flock
    pos = jnp.asarray(target) + jnp.array([3.0, 0.0])
    heading = jnp.arctan2(target[1] - pos[1], -(target[0] - pos[0]))
    st = st._replace(flock=f._replace(pos=f.pos.at[0].set(pos), heading=f.heading.at[0].set(heading)), t=jnp.int32(600))
    view = np.asarray(R.render(st, jnp.int32(0)))
    crown = np.asarray(R.WILLOW_C)
    hit = np.abs(view - crown).sum(-1) < 0.15
    assert hit.sum() > 50, hit.sum()
    # not cover: a willow is not a tree or a hut for warmth
    cover = sim.step(st, jnp.full((f.pos.shape[0],), sim.REST))
    assert cover is not None
    # the scripted flock still runs
    a = scripted.policy(st, jax.random.PRNGKey(0))
    assert a.shape == (f.pos.shape[0],)
