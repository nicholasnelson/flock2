"""Track B's multi-sheep grid with a signal channel."""

import jax
import jax.numpy as jnp
import numpy as np

from flock.grid import flock as F, world as W1


def test_generate_and_views():
    s = F.generate(jax.random.PRNGKey(0))
    assert s.pos.shape == (F.K, 2)
    assert len({tuple(p) for p in np.asarray(s.pos)}) == F.K  # distinct start tiles
    v = F.views(s)
    assert v.shape == (F.K, F.V, F.V)
    assert (np.asarray(v)[:, F.V // 2, F.V // 2] != F.SHEEP).all()  # a sheep does not see itself


def test_others_visible():
    s = F.generate(jax.random.PRNGKey(1))
    s = s._replace(pos=jnp.array([[10, 10], [10, 12], [20, 20], [2, 2]]))
    v = np.asarray(F.views(s))
    assert v[0, F.V // 2, F.V // 2 + 2] == F.SHEEP  # sheep 1 is two to the east of sheep 0
    assert (v[0] == F.SHEEP).sum() == 1  # sheep 2 and 3 are out of sheep 0's window


def test_hearing():
    pos = jnp.array([[10, 10], [10, 14], [10, 20], [3, 3]])
    sig = jnp.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0], [0.5, 0.5]])
    h = np.asarray(F.hear(pos, sig))
    np.testing.assert_allclose(h[0], [0.0, 1.0, 1 / 3])  # hears only sheep 1 (4 away); sheep 2 is 10 away
    np.testing.assert_allclose(h[1], [0.0, -0.5, 2 / 3])  # hears sheep 0 and 2 (both 4 and 6 away)
    np.testing.assert_allclose(h[3], [0.0, 0.0, 0.0])  # hears nobody


def test_step_drinks_and_signals():
    s = F.generate(jax.random.PRNGKey(2))
    wr, wc = np.argwhere(np.asarray(s.water))[0]
    p0 = (wr - 1, wc) if wr - 1 >= 1 and not s.water[wr - 1, wc] else (wr, wc - 1)
    s = s._replace(pos=s.pos.at[0].set(jnp.array(p0)), levels=jnp.full((F.K, 2), 0.5))
    a = jnp.full((F.K,), W1.STAY).at[0].set(W1.DRINK)
    s2 = F.step(s, a, jnp.ones((F.K, F.N_SIGNAL)) * 2.0)
    assert float(s2.levels[0, 1]) > 0.5 and float(s2.taste[0]) == 0.0
    assert float(s2.signal.max()) == 1.0  # clipped to [-1, 1]
    assert F.felt(s2).shape == (F.K, F.N_FELT)
