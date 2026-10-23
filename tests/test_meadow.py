import jax
import jax.numpy as jnp
import numpy as np

from flock import fpv
from flock.meadow import scripted, sim, views
from flock.meadow.world import N_TILES, TREE, Flock, Meadow, State, generate, ground_state


def _empty(n_sheep=2, pos=((64.0, 64.0), (64.0, 70.0)), heading=(0.0, 0.0)):
    """A dry meadow with medium grass and one tree, for exact tests."""
    m = Meadow(
        water=jnp.zeros((N_TILES, N_TILES), bool).at[60:63, 50:53].set(True),
        grass=jnp.full((N_TILES, N_TILES), 2.0),
        obj_kind=jnp.array([TREE]),
        obj_variant=jnp.array([0]),
        obj_pos=jnp.array([[62.0, 64.0]]),
        obj_present=jnp.array([True]),
    )
    f = Flock(
        pos=jnp.array(pos[:n_sheep]),
        heading=jnp.array(heading[:n_sheep]),
        resting=jnp.zeros(n_sheep, bool),
        head_up=jnp.zeros(n_sheep, bool),
        thirst=jnp.full(n_sheep, 0.5),
        hunger=jnp.full(n_sheep, 0.5),
    )
    return State(meadow=m, flock=f, t=jnp.int32(0))


def test_walking_turning_and_a_tree_blocking():
    s = _empty(heading=(np.pi / 2, 0.0))  # sheep 0 faces east; the tree stands two tiles north of it
    s2 = sim.step(s, jnp.array([sim.WALK, sim.RIGHT]))
    assert np.allclose(np.asarray(s2.flock.pos[0]), [64.0, 64.0 + sim.WALK_SPEED], atol=1e-5)  # east = +col
    assert np.isclose(float(s2.flock.heading[1]), float(sim.TURN))
    # sheep 0 turned north walks into the tree at (62, 64): stops short of it
    s3 = s._replace(flock=s.flock._replace(heading=jnp.array([0.0, 0.0])))
    for _ in range(40):
        s3 = sim.step(s3, jnp.array([sim.TROT, sim.STOP]))
    d = np.linalg.norm(np.asarray(s3.flock.pos[0]) - np.array([62.0, 64.0]))
    assert d >= 0.35 + 0.3 - 1e-4 and d < 1.2


def test_water_blocks_and_drinking_works_at_the_edge():
    s = _empty(n_sheep=1, pos=((61.5, 56.0),), heading=(3 * np.pi / 2,))  # west, toward the pond at cols 50-52
    for _ in range(60):
        s = sim.step(s, jnp.array([sim.TROT]))
    c = float(s.flock.pos[0, 1])
    assert c >= 53.0  # never entered the water
    before = float(s.flock.thirst[0])
    s = sim.step(s, jnp.array([sim.DRINK]))
    assert float(s.flock.thirst[0]) < before


def test_grazing_eats_and_separation_pushes_apart():
    s = _empty(pos=((64.0, 64.0), (64.0, 64.2)))  # overlapping
    s2 = sim.step(s, jnp.array([sim.GRAZE, sim.GRAZE]))
    assert float(s2.meadow.grass[64, 64]) < 2.0  # eaten (both on the same tile)
    d0 = float(jnp.linalg.norm(s.flock.pos[0] - s.flock.pos[1]))
    d1 = float(jnp.linalg.norm(s2.flock.pos[0] - s2.flock.pos[1]))
    assert d1 > d0


def test_views_night_and_facing():
    s = _empty(pos=((64.0, 64.0), (60.0, 64.0)), heading=(0.0, 0.0))  # sheep 1 four tiles north of 0, facing north (away)
    v = np.asarray(views.render_flock(s))
    assert v.shape == (2, fpv.H, fpv.W)
    assert (v[0, :8] == views.SKY_DAY).mean() > 0.9
    night = s._replace(t=jnp.int32(int(sim.DAY * 0.8)))
    vn = np.asarray(views.render_flock(night))
    assert (vn[0, :8] == views.SKY_NIGHT).mean() > 0.9
    # sheep 1 walks away from sheep 0: seen from behind
    assert int(views.sheep_facing(s.flock.pos[0], s.flock.pos[1], s.flock.heading[1])) in (0, 3)
    # and toward: from the front
    assert int(views.sheep_facing(s.flock.pos[0], s.flock.pos[1], jnp.float32(np.pi))) in (1, 2)


def test_generated_meadow_and_scripted_flock_run():
    st = generate(jax.random.PRNGKey(0), 12)
    g = np.asarray(ground_state(st.meadow))
    assert g.shape == (N_TILES, N_TILES) and (g == 4).any() and (g < 4).mean() > 0.9
    run = jax.jit(lambda s, k: jax.lax.scan(lambda c, kk: (sim.step(c, scripted.policy(c, kk)), None), s, jax.random.split(k, 50)))
    st2, _ = run(st, jax.random.PRNGKey(1))
    p = np.asarray(st2.flock.pos)
    assert np.isfinite(p).all() and (p > 0).all() and (p < N_TILES).all()
    assert int(st2.t) == 50
