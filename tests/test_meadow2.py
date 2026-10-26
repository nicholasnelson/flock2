"""Meadow v2 (and v2.1): generator, mouth point, smooth grass, fat, death, alive bonus, solid sheep,
huddling, light (spec §§ 2-5, 11)."""

import jax
import jax.numpy as jnp
import numpy as np

from flock.meadow2 import drives as D, scripted, sim
from flock.meadow2.world import DAY, N_SHEEP, N_TILES, SHEEP_RADIUS, START_WATER, generate, grass_at, random_start, start_distance


def _states(n=16):
    return [random_start(jax.random.PRNGKey(i)) for i in range(n)]


def test_layout_checks():
    for st in _states():
        w = np.asarray(st.meadow.water)
        assert w.any()
        d = float(start_distance(st.meadow.water, st.flock.pos.mean(0)[None])[0])
        assert START_WATER[0] - 3.0 <= d <= START_WATER[1] + 3.0
        p = np.asarray(st.meadow.obj_pos).astype(int)
        assert not w[p[:, 0], p[:, 1]][np.asarray(st.meadow.obj_present)].any()
        assert st.flock.pos.shape == (N_SHEEP, 2)
        assert (np.asarray(st.meadow.grass)[w] == 0).all()


def test_flock_starts_dry_clear_and_apart():
    for st in _states():
        r, c = sim.tile(st.flock.pos)
        assert not np.asarray(st.meadow.water[r, c]).any()
        assert np.asarray(sim._allowed(st.meadow, st.flock.pos)).all()
        p = np.asarray(st.flock.pos)
        d = np.linalg.norm(p[:, None] - p[None], axis=-1) + np.eye(len(p)) * 9
        assert d.min() >= 2 * SHEEP_RADIUS - 1e-5


def test_grass_field_is_smooth():
    g = jnp.zeros((N_TILES, N_TILES)).at[10, 10].set(2.0)
    assert abs(float(grass_at(g, jnp.array([10.5, 10.5]))) - 2.0) < 1e-6  # a tile centre
    assert abs(float(grass_at(g, jnp.array([10.5, 11.0]))) - 1.0) < 1e-6  # halfway to the next centre
    assert float(grass_at(g, jnp.array([12.5, 12.5]))) == 0.0


def _one_sheep_facing_water(n=1):
    """Sheep beside a straight shoreline facing it (east): water from column 20 on."""
    st = generate(jax.random.PRNGKey(0), n_sheep=n)
    water = jnp.zeros((N_TILES, N_TILES), bool).at[:, 20:].set(True)
    pos = jnp.array([[24.0 + 2 * i, 19.6] for i in range(n)])
    f = st.flock._replace(pos=pos, heading=jnp.full((n,), jnp.pi / 2), levels=jnp.full((n, D.N_DRIVES), 0.5))
    return st._replace(meadow=st.meadow._replace(water=water, grass=jnp.where(water, 0.0, 2.0)), flock=f)


def test_drink_only_with_mouth_on_water():
    st = _one_sheep_facing_water()
    s2 = sim.step(st, jnp.array([sim.DRINK]))
    assert float(s2.flock.levels[0, D.HYDRATION]) > 0.5 and float(s2.flock.taste[0]) == 0.0
    away = st._replace(flock=st.flock._replace(heading=jnp.array([3 * jnp.pi / 2])))
    s3 = sim.step(away, jnp.array([sim.DRINK]))
    assert float(s3.flock.levels[0, D.HYDRATION]) < 0.5 and float(s3.flock.taste[0]) == -1.0


def test_graze_eats_near_the_mouth_and_pays_by_density():
    st = _one_sheep_facing_water()
    st = st._replace(flock=st.flock._replace(heading=jnp.array([3 * jnp.pi / 2])))  # facing west: mouth at (24.0, 19.0)
    s2 = sim.step(st, jnp.array([sim.GRAZE]))
    before, after = np.asarray(st.meadow.grass[23:25, 18:20]).sum(), np.asarray(s2.meadow.grass[23:25, 18:20]).sum()
    assert before - after > 0.04  # a bite of 0.05, less a little regrowth
    gain_rich = float(s2.flock.levels[0, D.ENERGY]) - 0.5
    poor = st._replace(meadow=st.meadow._replace(grass=jnp.where(st.meadow.water, 0.0, 1.0)))
    gain_poor = float(sim.step(poor, jnp.array([sim.GRAZE])).flock.levels[0, D.ENERGY]) - 0.5
    assert gain_rich > gain_poor > 0
    bare = st._replace(meadow=st.meadow._replace(grass=jnp.zeros((N_TILES, N_TILES))))
    assert float(sim.step(bare, jnp.array([sim.GRAZE])).flock.taste[0]) == -1.0


def test_sheep_are_solid():
    st = generate(jax.random.PRNGKey(0), n_sheep=2)
    f = st.flock._replace(pos=jnp.array([[24.0, 24.0], [24.0, 24.65]]), heading=jnp.array([jnp.pi / 2, jnp.pi / 2]))
    st = st._replace(meadow=st.meadow._replace(water=jnp.zeros_like(st.meadow.water), obj_present=jnp.zeros_like(st.meadow.obj_present)), flock=f)
    s2 = sim.step(st, jnp.array([sim.WALK, sim.STOP]))  # sheep 0 walks east into sheep 1
    np.testing.assert_allclose(s2.flock.pos, st.flock.pos)  # refused, and nobody pushed
    s3 = sim.step(st, jnp.array([sim.WALK, sim.WALK]))  # both walk east: no overlap, both move
    assert float(s3.flock.pos[1, 1]) > 24.65 and float(s3.flock.pos[0, 1]) > 24.0


def test_huddling_keeps_warm():
    lv = jnp.full((3, D.N_DRIVES), 0.5)
    common = dict(action=jnp.full((3,), sim.REST), bite_density=jnp.zeros(3), drank=jnp.zeros(3, bool), resting=jnp.ones(3, bool), night=True, near_cover=jnp.zeros(3, bool), novelty=jnp.zeros(3))
    alone, _, _ = D.update(lv, jnp.full((3,), 0.5), huddle=jnp.zeros(3), **common)
    together, _, _ = D.update(lv, jnp.full((3,), 0.5), huddle=jnp.full((3,), 2), **common)
    assert float(together[0, D.WARMTH]) > float(alone[0, D.WARMTH])


def test_fat_buffers_the_gut():
    e2, f2 = D.fat_exchange(jnp.array([0.3]), jnp.array([0.5]))
    assert float(e2[0]) > 0.3 and float(f2[0]) < 0.5
    e3, f3 = D.fat_exchange(jnp.array([1.1]), jnp.array([0.5]))
    assert float(e3[0]) < 1.1 and float(f3[0]) > 0.5
    assert float(f3[0]) - 0.5 < (1.1 - float(e3[0])) / D.FAT_CAPACITY


def test_death_and_alive_bonus():
    st = _one_sheep_facing_water()
    thirsty = st._replace(flock=st.flock._replace(levels=st.flock.levels.at[0, D.HYDRATION].set(D.HYDRATION_DECAY / 2)))
    s2 = sim.step(thirsty, jnp.array([sim.STOP]))
    assert not bool(s2.flock.alive[0]) and float(s2.flock.reward[0]) == 0.0
    s3 = sim.step(s2, jnp.array([sim.WALK]))
    np.testing.assert_allclose(s3.flock.pos, s2.flock.pos)
    ok = sim.step(st, jnp.array([sim.STOP]))
    assert bool(ok.flock.alive[0]) and float(ok.flock.reward[0]) > 0.0
    worst = D.reward(jnp.zeros((1, D.N_DRIVES)), jnp.array([1.0]), jnp.array([True]))
    assert float(worst[0]) > 0.0


def test_fat_keeps_a_starving_sheep_alive():
    st = _one_sheep_facing_water()
    st = st._replace(flock=st.flock._replace(levels=st.flock.levels.at[0, D.ENERGY].set(0.0), fat=jnp.array([0.2])))
    assert bool(sim.step(st, jnp.array([sim.STOP])).flock.alive[0])
    assert not bool(sim.step(st._replace(flock=st.flock._replace(fat=jnp.array([0.0]))), jnp.array([sim.STOP])).flock.alive[0])


def test_light_is_smooth():
    L = np.asarray(jax.vmap(sim.light)(jnp.arange(2 * DAY)))
    assert abs(L.max() - 1.0) < 1e-6 and abs(L.min() - sim.NIGHT_LIGHT) < 1e-6
    assert np.abs(np.diff(L)).max() < 0.02


def test_scripted_runs():
    st = random_start(jax.random.PRNGKey(3))
    for i in range(20):
        a = scripted.policy(st, jax.random.PRNGKey(i))
        assert a.shape == (N_SHEEP,) and int(a.max()) < sim.N_ACTIONS
        st = sim.step(st, a)
