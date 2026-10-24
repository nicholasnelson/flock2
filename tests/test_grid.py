import jax
import jax.numpy as jnp
import numpy as np

from flock.grid import scripted, world as Wd


def _plain(levels=(0.5, 0.5)):
    """A world with a 2x2 pond at rows 3-4, cols 3-4 and grass 3 at (8, 8)."""
    s = Wd.generate(jax.random.PRNGKey(0), jnp.array(levels))
    rr, cc = jnp.arange(Wd.G)[:, None], jnp.arange(Wd.G)[None, :]
    water = (rr >= 3) & (rr < 5) & (cc >= 3) & (cc < 5)
    fertile = (rr == 8) & (cc == 8)
    return s._replace(water=water, fertile=fertile, grass=jnp.where(fertile, 3.0, 0.0))


def test_moves_walls_and_water():
    s = _plain()._replace(pos=jnp.array([1, 1]))
    assert np.array_equal(Wd.step(s, Wd.NORTH).pos, [1, 1])  # wall
    assert np.array_equal(Wd.step(s, Wd.SOUTH).pos, [2, 1])
    s = s._replace(pos=jnp.array([3, 2]))
    assert np.array_equal(Wd.step(s, Wd.EAST).pos, [3, 2])  # water


def test_drinking_eating_and_bad_taste():
    s = _plain()._replace(pos=jnp.array([3, 2]))  # beside the pond
    d = Wd.step(s, Wd.DRINK)
    assert float(d.levels[1]) > 0.5 and float(d.taste) == 0.0
    s = s._replace(pos=jnp.array([10, 10]))  # dry, bare
    dry = Wd.step(s, Wd.DRINK)
    stay = Wd.step(s, Wd.STAY)
    assert float(dry.taste) == -1.0 and float(dry.reward) < float(stay.reward)
    s = s._replace(pos=jnp.array([8, 8]))
    e = Wd.step(s, Wd.EAT)
    assert float(e.levels[0]) > 0.5 and float(e.grass[8, 8]) < 3.0


def test_drive_is_convex_and_view_is_centred():
    assert abs(float(Wd.drive(jnp.array([0.8, 1.0]))) - 0.04) < 1e-6
    s = _plain()._replace(pos=jnp.array([4, 2]))
    v = Wd.view(s)
    assert v.shape == (Wd.V, Wd.V)
    assert int(v[Wd.V // 2, Wd.V // 2 + 1]) == Wd.WATER  # the tile east of the sheep is pond
    assert int(v[0, 0]) == Wd.WALL  # beyond the border


def test_generated_worlds_have_water_grass_and_a_dry_start():
    for i in range(20):
        s = Wd.generate(jax.random.PRNGKey(i))
        assert bool(s.water.any()) and bool(s.fertile.any())
        r, c = (int(x) for x in s.pos)
        assert not bool(s.water[r, c]) and 0 < r < Wd.G - 1 and 0 < c < Wd.G - 1


def test_the_scripted_sheep_keeps_both_needs_up():
    def run(key):
        s = Wd.generate(key, jnp.array([0.7, 0.7]))

        def body(c, k):
            s, mode = c
            a, mode = scripted.policy(s, k, mode)
            s = Wd.step(s, a)
            return (s, mode), s.levels

        return jax.lax.scan(body, (s, jnp.int32(0)), jax.random.split(key, 1000))[1]

    lv = np.asarray(jax.jit(jax.vmap(run))(jax.random.split(jax.random.PRNGKey(1), 32)))
    assert lv[:, -200:].min() > 0.2, lv[:, -200:].min()
