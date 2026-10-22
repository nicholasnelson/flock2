import jax
import jax.numpy as jnp
import numpy as np

from flock import raster, sprites, world
from flock.palette import N_COLOURS, TRANSPARENT


def test_sprites_are_well_formed():
    assert sprites.OBJECT_STACK.shape == (sprites.N_OBJECT_TYPES, 8, 8)
    assert sprites.GROUND_STACK.shape[1:] == (4, 4)
    assert not (sprites.GROUND_STACK == TRANSPARENT).any()
    # every object has some opaque pixels and more than one colour
    for i, name in enumerate(sprites.OBJECT_NAMES):
        s = sprites.OBJECT_STACK[i]
        opaque = s[s != TRANSPARENT]
        assert opaque.size >= 12, name
        assert len(np.unique(opaque)) >= 2, name


def _small_world(key, h=24, w=24):
    tile_map = world.random_ground(key, h, w)
    objects = world.Objects(
        kind=jnp.array([0, 1, 2]),
        pos=jnp.array([[2, 2], [10, 10], [20, 5]]),
        present=jnp.array([True, True, False]),
    )
    agents = world.Agents(kind=jnp.array([world.SHEEP]), pos=jnp.array([[10, 14]]), heading=jnp.array([0]), present=jnp.array([True]))
    return tile_map, objects, agents


def test_render_world_pastes_objects_and_respects_presence():
    tile_map, objects, agents = _small_world(jax.random.PRNGKey(0))
    img = world.render_world(tile_map, objects, agents)
    assert img.shape == (96, 96)
    # object 0 (grass tuft) at tile (2,2) -> px (8,8); its light-green pixels should appear
    patch = np.asarray(img[8:16, 8:16])
    spr = sprites.OBJECT_STACK[0]
    mask = spr != TRANSPARENT
    assert (patch[mask] == spr[mask]).all()
    # absent object 2 at tile (20,5) leaves ground untouched
    patch2 = np.asarray(img[80:88, 20:28])
    assert not (patch2 == 8).any()  # salt is white; none pasted
    # sheep at tile (10,14) -> px (40,56)
    patch3 = np.asarray(img[40:48, 56:64])
    assert (patch3 == 8).sum() > 10  # white wool


def test_egocentric_view_is_centred_and_rotation_consistent():
    tile_map, objects, agents = _small_world(jax.random.PRNGKey(1))
    img = world.render_world(tile_map, objects, agents)
    pos = jnp.array([10, 14])
    v0 = world.egocentric_view(img, pos, jnp.int32(0))
    assert v0.shape == (64, 64)
    # the agent's own sprite is at the centre of its view
    centre = np.asarray(v0[28:36, 28:36])
    assert (centre == 8).sum() > 10
    # heading rotations map the same world tile to the predicted view tile
    tile = jnp.array([10, 10])  # pond at (10,10), 4 tiles west of the sheep
    for k in range(4):
        vk = world.egocentric_view(img, pos, jnp.int32(k))
        vt = world.world_tile_in_view(pos, jnp.int32(k), tile)
        r, c = int(vt[0]), int(vt[1])
        assert 0 <= r < 16 and 0 <= c < 16
        patch = np.asarray(vk[r * 4 : r * 4 + 8, c * 4 : c * 4 + 8])
        assert (patch == 4).sum() > 5, (k, r, c)  # water pixels found where predicted


def test_view_out_of_bounds_is_void():
    tile_map, objects, agents = _small_world(jax.random.PRNGKey(2))
    img = world.render_world(tile_map, objects, agents)
    v = world.egocentric_view(img, jnp.array([0, 0]), jnp.int32(0))
    assert (np.asarray(v[:20, :20]) == 0).all()


def test_rasteriser_shapes_and_gradients():
    k = 6
    key = jax.random.PRNGKey(0)
    strokes = raster.Strokes(
        endpoints=jax.random.normal(key, (k, 4)),
        width=jnp.zeros((k,)),
        colour=jnp.zeros((k, N_COLOURS)),
        gate=jnp.full((k,), 3.0),
    )
    canvas = raster.rasterise(strokes)
    assert canvas.colour.shape == (16, 16, N_COLOURS)
    assert canvas.alpha.shape == (16, 16)
    assert float(canvas.alpha.max()) > 0.5
    np.testing.assert_allclose(np.asarray(canvas.colour.sum(-1)), 1.0, atol=1e-4)

    def loss(s):
        c = raster.rasterise(s)
        return jnp.sum(c.colour[..., 12] * c.alpha)  # "draw red"

    g = jax.grad(loss)(strokes)
    assert float(jnp.abs(g.colour).sum()) > 0
    assert float(jnp.abs(g.endpoints).sum()) > 0


def test_gate_off_draws_nothing():
    k = 3
    strokes = raster.Strokes(endpoints=jnp.zeros((k, 4)), width=jnp.zeros((k,)), colour=jnp.zeros((k, N_COLOURS)), gate=jnp.full((k,), -20.0))
    canvas = raster.rasterise(strokes)
    assert float(canvas.alpha.max()) < 1e-6
    assert float(canvas.ink) < 1e-6


def test_composite_canvas_bubble_and_stamp():
    view = world.to_onehot(jnp.full((64, 64), 1))  # all dark grass
    colour = jax.nn.one_hot(jnp.full((16, 16), 12), N_COLOURS)  # red
    alpha = jnp.zeros((16, 16)).at[4:8, 4:8].set(1.0)
    bubble = world.composite_canvas(view, colour, alpha, jnp.array([0, 0]), opaque_paper=True)
    stamp = world.composite_canvas(view, colour, alpha, jnp.array([0, 0]), opaque_paper=False)
    assert float(bubble[0, 0, 8]) == 1.0  # paper where no ink
    assert float(stamp[0, 0, 1]) == 1.0  # ground where no ink
    assert float(bubble[5, 5, 12]) == 1.0 and float(stamp[5, 5, 12]) == 1.0


def test_fixed_width_lines_are_thin_and_ink_is_length_plus_stroke_cost():
    from flock.raster import Strokes, rasterise

    # one horizontal stroke across the middle, gate fully on; the raw width output is huge
    e = jnp.array([[-2.0, 0.0, 2.0, 0.0]])  # sigmoid -> x from ~0.12 to ~0.88 of the canvas, y = 0.5
    st = Strokes(endpoints=e, width=jnp.array([10.0]), colour=jnp.zeros((1, 16)), gate=jnp.array([20.0]))
    c = rasterise(st, size=32, fixed_width=2.0)
    a = np.asarray(c.alpha)
    col = a[:, 16]
    assert 1 <= (col > 0.5).sum() <= 3  # about 2 px thick, whatever the width output says
    length = 32 * (float(jax.nn.sigmoid(2.0)) - float(jax.nn.sigmoid(-2.0)))
    assert abs(float(c.ink) - length) < 0.1
    c4 = rasterise(st, size=32, fixed_width=2.0, stroke_cost=4.0)
    assert abs(float(c4.ink) - (length + 4.0)) < 0.1


def test_hard_gate_is_all_or_nothing_with_a_gradient():
    from flock.raster import Strokes, rasterise

    e = jnp.array([[-2.0, 0.0, 2.0, 0.0]])
    f = lambda g: rasterise(Strokes(e, jnp.zeros(1), jnp.zeros((1, 16)), g), size=32, fixed_width=2.0, hard_gate=True)
    half_on, half_off = f(jnp.array([0.3])), f(jnp.array([-0.3]))  # sigmoid 0.57 / 0.43
    # a 2 px line centred between pixel rows peaks at ~0.81 alpha (soft edges), gate or no gate
    assert float(half_on.alpha.max()) > 0.75 and float(half_off.alpha.max()) < 1e-6
    g = jax.grad(lambda g: jnp.sum(f(g).alpha))(jnp.array([0.3]))
    assert float(jnp.abs(g).sum()) > 0


def test_min_strokes_keeps_the_strongest_stroke_on():
    from flock.raster import Strokes, rasterise

    e = jnp.array([[-2.0, 0.0, 2.0, 0.0], [0.0, -2.0, 0.0, 2.0]])
    st = Strokes(endpoints=e, width=jnp.zeros(2), colour=jnp.zeros((2, 16)), gate=jnp.array([-3.0, -1.0]))  # both "off"
    off = rasterise(st, size=32, fixed_width=2.0, hard_gate=True)
    one = rasterise(st, size=32, fixed_width=2.0, hard_gate=True, min_strokes=1)
    assert float(off.alpha.max()) < 1e-6
    a = np.asarray(one.alpha)
    assert a.max() > 0.75 and (a[:, 16] > 0.5).sum() > 10 and (a[16, :] > 0.5).sum() <= 3  # the vertical one (gate -1)
