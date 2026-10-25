"""Meadow v2: generator checks, mouth point, fat, death, alive bonus, light (spec §§ 2-5)."""

import jax
import jax.numpy as jnp
import numpy as np

from flock.meadow2 import drives as D, scripted, sim
from flock.meadow2.world import N_SHEEP, N_TILES, START_WATER, generate, random_start, start_distance


def _states(n=16):
    return [random_start(jax.random.PRNGKey(i)) for i in range(n)]


def test_layout_checks():
    for st in _states():
        w = np.asarray(st.meadow.water)
        assert w.any()
        # the flock's centre is within reach of water (sheep scatter ~2 tiles around it)
        d = float(start_distance(st.meadow.water, st.flock.pos.mean(0)[None])[0])
        assert START_WATER[0] - 3.0 <= d <= START_WATER[1] + 3.0
        # objects are never on water
        p = np.asarray(st.meadow.obj_pos).astype(int)
        assert not w[p[:, 0], p[:, 1]][np.asarray(st.meadow.obj_present)].any()
        assert st.flock.pos.shape == (N_SHEEP, 2)


def test_flock_starts_dry_and_clear_of_water():
    for st in _states():
        r, c = sim.tile(st.flock.pos)
        assert not np.asarray(st.meadow.water[r, c]).any()
        assert np.asarray(sim._allowed(st.meadow, st.flock.pos)).all()  # not inside a tree or the hut


def _one_sheep_facing_water():
    """A sheep beside a straight shoreline, facing it: water from column 20 on."""
    st = generate(jax.random.PRNGKey(0), n_sheep=1)
    water = jnp.zeros((N_TILES, N_TILES), bool).at[:, 20:].set(True)
    f = st.flock._replace(pos=jnp.array([[24.0, 19.6]]), heading=jnp.array([jnp.pi / 2]), levels=jnp.full((1, 5), 0.5))
    return st._replace(meadow=st.meadow._replace(water=water, grass=jnp.where(water, 0.0, 2.0)), flock=f)


def test_drink_only_with_mouth_on_water():
    st = _one_sheep_facing_water()
    s2 = sim.step(st, jnp.array([sim.DRINK]))
    assert float(s2.flock.levels[0, D.HYDRATION]) > 0.5 and float(s2.flock.taste[0]) == 0.0
    away = st._replace(flock=st.flock._replace(heading=jnp.array([3 * jnp.pi / 2])))  # facing west, away
    s3 = sim.step(away, jnp.array([sim.DRINK]))
    assert float(s3.flock.levels[0, D.HYDRATION]) < 0.5 and float(s3.flock.taste[0]) == -1.0


def test_graze_eats_the_mouth_tile():
    st = _one_sheep_facing_water()
    st = st._replace(flock=st.flock._replace(heading=jnp.array([3 * jnp.pi / 2])))  # facing west: mouth at col 19.0
    s2 = sim.step(st, jnp.array([sim.GRAZE]))
    assert float(s2.meadow.grass[24, 19]) < float(st.meadow.grass[24, 19])
    assert float(s2.flock.levels[0, D.ENERGY]) > 0.5


def test_fat_buffers_the_gut():
    e, f = jnp.array([0.3]), jnp.array([0.5])
    e2, f2 = D.fat_exchange(e, f)
    assert float(e2[0]) > 0.3 and float(f2[0]) < 0.5
    e3, f3 = D.fat_exchange(jnp.array([1.1]), jnp.array([0.5]))
    assert float(e3[0]) < 1.1 and float(f3[0]) > 0.5
    assert float(f3[0]) - 0.5 < (1.1 - float(e3[0])) / D.FAT_CAPACITY  # stored at < 100 % efficiency


def test_death_and_alive_bonus():
    st = _one_sheep_facing_water()
    thirsty = st._replace(flock=st.flock._replace(levels=st.flock.levels.at[0, D.HYDRATION].set(0.0005)))
    s2 = sim.step(thirsty, jnp.array([sim.STOP]))
    assert not bool(s2.flock.alive[0]) and float(s2.flock.reward[0]) == 0.0
    s3 = sim.step(s2, jnp.array([sim.WALK]))  # the dead don't move
    np.testing.assert_allclose(s3.flock.pos, s2.flock.pos)
    ok = sim.step(st, jnp.array([sim.STOP]))
    assert bool(ok.flock.alive[0]) and float(ok.flock.reward[0]) > 0.0
    worst = D.reward(jnp.zeros((1, D.N_DRIVES)), jnp.array([1.0]), jnp.array([True]))
    assert float(worst[0]) > 0.0  # living always beats dying


def test_fat_keeps_a_starving_sheep_alive():
    st = _one_sheep_facing_water()
    st = st._replace(flock=st.flock._replace(levels=st.flock.levels.at[0, D.ENERGY].set(0.0), fat=jnp.array([0.2])))
    s2 = sim.step(st, jnp.array([sim.STOP]))
    assert bool(s2.flock.alive[0])
    s3 = sim.step(st._replace(flock=st.flock._replace(fat=jnp.array([0.0]))), jnp.array([sim.STOP]))
    assert not bool(s3.flock.alive[0])


def test_light_is_smooth():
    L = np.asarray(jax.vmap(sim.light)(jnp.arange(2 * 1200)))
    assert abs(L.max() - 1.0) < 1e-6 and abs(L.min() - sim.NIGHT_LIGHT) < 1e-6
    assert np.abs(np.diff(L)).max() < 0.02


def test_scripted_runs():
    st = random_start(jax.random.PRNGKey(3))
    for i in range(20):
        a = scripted.policy(st, jax.random.PRNGKey(i))
        assert a.shape == (N_SHEEP,) and int(a.max()) < sim.N_ACTIONS
        st = sim.step(st, a)
