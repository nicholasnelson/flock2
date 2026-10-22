import jax
import jax.numpy as jnp
import numpy as np

from flock import iso, sprites_iso as S
from flock.palette import TRANSPARENT


def _world(key, n=24):
    tm = jax.random.randint(key, (n, n), 0, 4).astype(jnp.int32)  # grass states only
    objects = iso.Objects(kind=jnp.array([0, 1, 2]), pos=jnp.array([[8, 12], [12, 16], [20, 20]]), present=jnp.array([True, True, True]))
    agents = iso.Agents(kind=jnp.array([S.SHEEP]), pos=jnp.array([[12, 12]]), heading=jnp.array([0]), present=jnp.array([True]))
    return tm, objects, agents


def test_inverse_projection_matches_diamond_sprites():
    # a world with a single distinct tile state should render that tile's diamond exactly at
    # the predicted screen position
    n = 24
    tm = jnp.full((n, n), 1, jnp.int32).at[10, 14].set(S.WATER)
    objects = iso.Objects(kind=jnp.zeros((1,), jnp.int32), pos=jnp.zeros((1, 2), jnp.int32), present=jnp.array([False]))
    agents = iso.Agents(kind=jnp.array([S.SHEEP]), pos=jnp.array([[12, 12]]), heading=jnp.array([0]), present=jnp.array([False]))
    for h in range(4):
        v = np.asarray(iso.render_view(tm, objects, agents, jnp.array([12, 12]), jnp.int32(h)))
        assert v.shape == (64, 64)
        centre = np.asarray(iso.tile_screen_centre(jnp.array([-2, 2]), jnp.int32(h)))
        x, y = int(centre[0]), int(centre[1])
        box = v[y - 2 : y + 2, x - 4 : x + 4]
        water = S.GROUND_VARIANTS["water"][S.tile_variant(10, 14)]
        m = S.DIAMOND
        assert (box[m] == water[m]).all(), h
        # and that is the only water in the view
        assert ((v == 4) | (v == 5)).sum() == m.sum(), h
        # no transparent markers leak into the ground
        assert not (v == TRANSPARENT).any()


def test_sprites_billboarded_and_depth_sorted():
    tm, objects, agents = _world(jax.random.PRNGKey(0))
    # tree (kind 1) two tiles south-east of the sheep; sheep at centre
    objects = iso.Objects(kind=jnp.array([1]), pos=jnp.array([[14, 14]]), present=jnp.array([True]))
    for h in range(4):
        v = np.asarray(iso.render_view(tm, objects, agents, jnp.array([12, 12]), jnp.int32(h)))
        # the sheep's wool is at the view centre regardless of heading
        assert (v[26:36, 26:38] == 8).sum() > 10, h
        # the tree is upright: its trunk (bark, colour 6; nothing else in this scene uses it)
        # sits wholly below the top of its canopy outline (colour 15)
        ys, _ = np.where(v == 15)
        bark_ys, _ = np.where(v == 6)
        assert len(ys) > 0 and len(bark_ys) > 0, h
        assert bark_ys.min() > ys.min() + 4, h


def test_agent_facing_flips_with_view_rotation():
    tm, objects, agents = _world(jax.random.PRNGKey(1))
    other = iso.Agents(kind=jnp.array([S.SHEEP, S.SHEEP]), pos=jnp.array([[12, 12], [12, 16]]), heading=jnp.array([0, 1]), present=jnp.array([True, True]))
    f0 = int(iso.facing(jnp.int32(1), jnp.int32(0)))  # E seen from a N-up view: toward, right
    f2 = int(iso.facing(jnp.int32(1), jnp.int32(2)))  # E seen from an S-up view: away, left
    assert f0 == 1 and f2 == 3
    # left-facing sprites are mirrors of right-facing ones; front and back differ
    st = S.ANIMAL_STACK[S.SHEEP]
    assert (st[2] == st[1][:, ::-1]).all() and (st[3] == st[0][:, ::-1]).all()
    assert (st[0] != st[1]).any()


def test_out_of_world_is_void_and_night_radius():
    tm, objects, agents = _world(jax.random.PRNGKey(2))
    v = np.asarray(iso.render_view(tm, objects, agents, jnp.array([0, 0]), jnp.int32(0)))
    assert (v == 0).mean() > 0.3
    vn = np.asarray(iso.render_view(tm, objects, agents, jnp.array([12, 12]), jnp.int32(0), radius_px=12))
    assert (vn[0, :] == 0).all() and (vn[:, 0] == 0).all()


def test_render_world_runs():
    tm, objects, agents = _world(jax.random.PRNGKey(3))
    img = iso.render_world(tm, objects, agents)
    assert img.ndim == 2 and img.shape[0] > 40
