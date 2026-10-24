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
        levels=jnp.full((n_sheep, 5), 0.5),
        visits=jnp.zeros((n_sheep, 16, 16), jnp.int32),
        reward=jnp.zeros(n_sheep),
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
    before = float(s.flock.levels[0, 1])
    s = sim.step(s, jnp.array([sim.DRINK]))
    assert float(s.flock.levels[0, 1]) > before  # hydration restored


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


def test_pack_roundtrip_and_a_tiny_rollout(tmp_path):
    from flock.meadow import rollouts

    v = np.random.randint(0, 16, (3, 64, 256)).astype(np.uint8)
    assert (np.asarray(rollouts.unpack(rollouts.pack(v))) == v).all()
    n = rollouts.write_rollout(str(tmp_path), 0, seed=0, meadows=2, sheep=6, record=2, steps=16, warm=4, chunk=8)
    assert n == 2 * 2 * 16
    views = np.load(tmp_path / "shard_0000.views.npy")
    acts = np.load(tmp_path / "shard_0000.actions.npy")
    assert views.shape == (4, 16, 64, 128) and acts.shape == (4, 16)
    full = np.asarray(rollouts.unpack(views))
    assert full.max() < 16  # palette indices
    assert acts.max() < 9


def test_drive_reward_signs():
    from flock.meadow import drives as D

    hungry = jnp.array([[0.3, 0.9, 0.9, 1.0, 0.9]])
    after = D.update(hungry, action=jnp.array([5]), grazed=jnp.array([True]), drank=jnp.array([False]), resting=jnp.array([False]), night=jnp.bool_(False), near_cover=jnp.array([False]), n_visible=jnp.array([8.0]), novelty=jnp.array([0.0]))
    assert float(D.drive_reduction(hungry, after)[0]) > 0  # eating when hungry pays
    assert float(D.reward(hungry, after)[0]) > float(D.reward(hungry, hungry)[0])
    full = jnp.array([[1.1, 0.9, 0.9, 1.0, 0.9]])
    over = D.update(full, action=jnp.array([5]), grazed=jnp.array([True]), drank=jnp.array([False]), resting=jnp.array([False]), night=jnp.bool_(False), near_cover=jnp.array([False]), n_visible=jnp.array([8.0]), novelty=jnp.array([0.0]))
    assert float(D.drive_reduction(full, over)[0]) < 0  # over-eating past the set-point costs
    assert float(D.reward(full, over)[0]) < float(D.reward(full, full)[0])
    assert float(D.reward(full, jnp.ones((1, 5)))[0]) == 0.0  # at the set-points nothing is owed
    # warmth at night: the open chills faster than cover
    h = jnp.full((2, 5), 0.8)
    n = D.update(h, action=jnp.zeros(2, jnp.int32), grazed=jnp.zeros(2, bool), drank=jnp.zeros(2, bool), resting=jnp.zeros(2, bool), night=jnp.bool_(True), near_cover=jnp.array([False, True]), n_visible=jnp.zeros(2), novelty=jnp.zeros(2))
    assert float(n[0, D.WARMTH]) < float(n[1, D.WARMTH]) < 0.8


def test_social_counts_only_sheep_in_view_and_interest_rewards_new_cells():
    pos = jnp.array([[64.0, 64.0], [60.0, 64.0], [68.0, 64.0]])  # one ahead (north), one behind
    heading = jnp.zeros(3)
    nv = np.asarray(sim.n_visible(pos, heading, jnp.full(3, 12.0)))
    assert nv[0] == 1  # the sheep behind is in the blind wedge
    s = _empty(n_sheep=1, pos=((63.9, 63.9),), heading=(np.pi / 2,))  # east, about to cross into a new 8-tile cell
    s = s._replace(flock=s.flock._replace(pos=jnp.array([[60.0, 63.95]])))
    before = float(s.flock.levels[0, 4])
    s2 = sim.step(s, jnp.array([sim.WALK]))
    assert float(s2.flock.levels[0, 4]) > before  # entering a fresh cell raised interest
    assert int(s2.flock.visits.sum()) == 1


def test_energy_use_depends_on_the_action():
    from flock.meadow import drives as D, sim

    assert len(D.ENERGY_USE) == sim.N_ACTIONS
    h = jnp.full((3, 5), 0.8)
    acts = jnp.array([sim.REST, sim.WALK, sim.TROT])
    n = D.update(h, action=acts, grazed=jnp.zeros(3, bool), drank=jnp.zeros(3, bool), resting=acts == sim.REST, night=jnp.bool_(False), near_cover=jnp.zeros(3, bool), n_visible=jnp.zeros(3), novelty=jnp.zeros(3))
    e = np.asarray(n[:, D.ENERGY])
    assert e[0] > e[1] > e[2]  # resting saves energy, trotting burns it
