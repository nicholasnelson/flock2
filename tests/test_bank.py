"""Muddy banks (spec § 21, grid 3): regrowth scaled by distance from water."""

import jax
import jax.numpy as jnp
import numpy as np

from flock.meadow2 import sim
from flock.meadow2.world import bank_map, cell_water_distance, generate, with_terrain


def test_bank_map_shape_and_limits():
    st = generate(jax.random.PRNGKey(0))
    d = np.asarray(cell_water_distance(st.meadow.water))
    m = np.asarray(bank_map(st.meadow.water, 8.0))
    assert m.shape == d.shape
    assert (m[d >= 8] == 1.0).all() and (m[d < 1] < 0.13).all()
    assert np.allclose(np.asarray(bank_map(st.meadow.water, 0.0)), 1.0)


def test_bank_slows_regrowth_near_water_only():
    st = generate(jax.random.PRNGKey(1))
    muddy = with_terrain(st, bank=8.0)
    a = sim.step(st, jnp.full((12,), sim.STOP))
    b = sim.step(muddy, jnp.full((12,), sim.STOP))
    g0, ga, gb = (np.asarray(x.meadow.grass) for x in (st, a, b))
    d = np.asarray(cell_water_distance(st.meadow.water))
    far, near = (d >= 8) & (g0 > 0), (d < 2) & (g0 > 0)
    assert np.allclose(ga[far], gb[far])
    assert ((ga - g0)[near] > (gb - g0)[near]).all()


def test_bank_on_a_batch():
    sts = jax.vmap(generate)(jax.random.split(jax.random.PRNGKey(2), 2))
    out = with_terrain(sts, bank=jnp.array([4.0, 12.0]))
    assert out.meadow.regrow_map.shape == (2,) + sts.meadow.grass.shape[1:]
    assert float(out.meadow.regrow_map[0].mean()) > float(out.meadow.regrow_map[1].mean())
