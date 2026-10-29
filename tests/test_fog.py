"""Fog days (spec section 21.4): drawn per day on the meadow, shortening the haze and flattening the sky."""

import jax
import jax.numpy as jnp
import numpy as np

from flock.meadow2 import render as R
from flock.meadow2.world import DAY, MAX_DAYS, foggy, random_start


def test_fog_days_are_drawn_per_day_with_the_given_probability():
    none = random_start(jax.random.PRNGKey(0))
    assert none.meadow.fog_days.shape == (MAX_DAYS,) and not bool(none.meadow.fog_days.any())
    draws = jax.vmap(lambda k: random_start(k, fog=0.5).meadow.fog_days)(jax.random.split(jax.random.PRNGKey(1), 64))
    share = float(np.asarray(draws).mean())
    assert 0.35 < share < 0.65, share
    st = random_start(jax.random.PRNGKey(2), fog=0.5)._replace(meadow=none.meadow._replace(fog_days=jnp.arange(MAX_DAYS) % 2 == 1))
    assert not bool(foggy(st.meadow, jnp.int32(10))) and bool(foggy(st.meadow, jnp.int32(DAY + 10)))
    assert bool(foggy(st.meadow, jnp.int32(DAY * 100 + 10)))  # past MAX_DAYS: the last day's fog holds


def test_fog_hides_the_far_and_keeps_the_near():
    st = random_start(jax.random.PRNGKey(3), willows=3)._replace(t=jnp.int32(DAY // 4))
    clear = st._replace(meadow=st.meadow._replace(fog_days=jnp.zeros((MAX_DAYS,), bool)))
    fog = st._replace(meadow=st.meadow._replace(fog_days=jnp.ones((MAX_DAYS,), bool)))
    rgb_c, _, depth = R.render_aux(clear, 0)
    rgb_f, _, _ = R.render_aux(fog, 0)
    rgb_c, rgb_f, depth = np.asarray(rgb_c), np.asarray(rgb_f), np.asarray(depth)
    horizon = np.asarray(R.sky(jnp.zeros(()), R.sim.light(st.t)))
    far = np.isfinite(depth) & (depth > 12)
    near = np.isfinite(depth) & (depth < 1.5)
    assert far.any() and near.any()
    # far pixels sit on the horizon colour in fog, and the sky is that colour too
    assert np.abs(rgb_f[far] - horizon).max() < 0.03
    assert np.abs(rgb_f[~np.isfinite(depth)] - horizon).max() < 1e-4
    # near pixels keep most of their colour
    assert np.abs(rgb_f[near] - rgb_c[near]).mean() < np.abs(rgb_f[far] - rgb_c[far]).mean()
