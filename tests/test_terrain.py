"""Stage 2a's dials (spec § 21): regrowth as a meadow number, trampling under moving sheep."""

import jax
import jax.numpy as jnp
import numpy as np

from flock.meadow2 import sim
from flock.meadow2.world import REGROW, generate, patch, with_terrain


def test_defaults_are_the_old_world():
    st = generate(jax.random.PRNGKey(0))
    assert np.isclose(float(st.meadow.regrow), REGROW) and float(st.meadow.trample) == 0.0
    a = sim.step(st, jnp.full((12,), sim.WALK))
    b = sim.step(with_terrain(st, trample=0.0), jnp.full((12,), sim.WALK))
    assert np.allclose(np.asarray(a.meadow.grass), np.asarray(b.meadow.grass))


def test_regrow_dial():
    st = generate(jax.random.PRNGKey(1))
    half = with_terrain(st, regrow=REGROW / 2)
    a = sim.step(st, jnp.full((12,), sim.STOP))
    b = sim.step(half, jnp.full((12,), sim.STOP))
    ga, gb, g0 = (np.asarray(x.meadow.grass) for x in (a, b, st))
    dry = g0 > 0
    assert np.allclose((ga - g0)[dry], 2 * (gb - g0)[dry], atol=1e-6)


def test_trampling_wears_under_moving_sheep_only():
    st = with_terrain(generate(jax.random.PRNGKey(2)), trample=0.05)
    f = st.flock
    acts = jnp.full((12,), sim.REST).at[0].set(sim.WALK)
    new = sim.step(st, acts)
    g0, g1 = np.asarray(st.meadow.grass), np.asarray(new.meadow.grass)
    # under sheep 0 (after its step) the grass fell by about the trample loss (minus a step's regrowth)
    idx, under = patch(new.flock.pos[0])
    idx, under = np.asarray(idx)[np.asarray(under)], None
    drop = (g0 - g1)[idx[:, 0], idx[:, 1]]
    assert (drop > 0.03).all(), drop
    # under a resting sheep the grass only regrew
    idx1, u1 = patch(f.pos[1])
    idx1 = np.asarray(idx1)[np.asarray(u1)]
    assert ((g1 - g0)[idx1[:, 0], idx1[:, 1]] >= -1e-6).all()


def test_terrain_on_a_batch():
    sts = jax.vmap(generate)(jax.random.split(jax.random.PRNGKey(3), 3))
    out = with_terrain(sts, regrow=1e-5, trample=0.01)
    assert out.meadow.regrow.shape == (3,) and np.isclose(float(out.meadow.trample[2]), 0.01)
    stepped = jax.vmap(sim.step)(out, jnp.zeros((3, 12), jnp.int32))
    assert stepped.meadow.grass.shape == sts.meadow.grass.shape
