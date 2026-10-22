import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from flock import world
from flock.nets import Agents
from flock.toy import env, game


def _model(action_space="strokes", shared=True):
    return Agents(nnx.Rngs(0), action_space=action_space, k_strokes=4, shared_encoder=shared, width=8)


def test_forward_shapes_and_losses_finite():
    ep = env.make_batch(jax.random.PRNGKey(0), 4, 3, "bubble")
    for action_space in ("strokes", "latent", "pixels"):
        for shared in (True, False):
            m = _model(action_space, shared)
            fwd = game.forward(m, ep, jax.random.PRNGKey(1), opaque=True)
            assert fwd.logits.shape == (4, 16, 16)
            assert fwd.canvas.colour.shape == (4, 16, 16, 16)
            ce, acc = game.success_loss(fwd.logits, ep.target_mask)
            assert np.isfinite(np.asarray(ce)).all()
            assert acc.shape == (4,)


def test_canvas_is_visible_in_shared_receiver_view():
    ep = env.make_batch(jax.random.PRNGKey(0), 2, 3, "stamp")
    m = _model()
    # force a fully red opaque canvas and check it lands at canvas_px in the view
    from flock.raster import Canvas

    red = Canvas(colour=jax.nn.one_hot(jnp.full((2, 16, 16), 12), 16), alpha=jnp.ones((2, 16, 16)), ink=jnp.zeros((2,)))
    _, rv = game.receive(m, ep.receiver_base, red, ep.canvas_px, opaque=False)
    for b in range(2):
        r, c = np.asarray(ep.canvas_px[b])
        patch = np.asarray(rv[b, r : r + 16, c : c + 16, 12])
        assert (patch == 1.0).all()


def test_signalling_and_listening_losses():
    ep = env.make_batch(jax.random.PRNGKey(0), 6, 3, "bubble")
    m = _model()
    fwd = game.forward(m, ep, None, opaque=True)
    s = game.signalling_loss(fwd.canvas, ep.target_type)
    assert np.isfinite(float(s))
    blank = game.blank_canvas(6)
    l0, _ = game.receive(m, ep.receiver_base, blank, ep.canvas_px, True)
    l = game.listening_loss(fwd.logits, l0)
    assert float(l) <= 0.0


def test_coarse_canvas_is_blocky_and_trainable():
    m = Agents(nnx.Rngs(0), action_space="strokes", k_strokes=4, width=8, canvas_res=8)
    ep = env.make_batch(jax.random.PRNGKey(0), 2, 3, "bubble")
    fwd = game.forward(m, ep, jax.random.PRNGKey(1), opaque=True)
    a = np.asarray(fwd.canvas.alpha)
    assert a.shape == (2, 16, 16)
    # every 2x2 block is one coarse pixel
    assert np.allclose(a[:, ::2, ::2], a[:, 1::2, 1::2]) and np.allclose(a[:, ::2, ::2], a[:, ::2, 1::2])
    assert a.max() > 0.1

    def loss(mm):
        f = game.forward(mm, ep, jax.random.PRNGKey(1), opaque=True)
        return jnp.mean(game.success_loss(f.logits, ep.target_mask)[0])

    g = nnx.grad(loss)(m)
    assert any(float(jnp.abs(x).sum()) > 0 for x in jax.tree_util.tree_leaves(g.sender))


def test_mono_sender_draws_only_the_mono_colour():
    m = Agents(nnx.Rngs(0), action_space="strokes", k_strokes=4, width=8, mono_colour=15)
    ep = env.make_batch(jax.random.PRNGKey(0), 2, 3, "bubble")
    fwd = game.forward(m, ep, jax.random.PRNGKey(1), opaque=True)
    a = np.asarray(fwd.canvas.alpha)
    col = np.asarray(fwd.canvas.colour)
    inked = a > 0.05
    assert inked.any()
    assert np.allclose(col[inked].argmax(-1), 15) and col[inked][:, 15].min() > 0.99


def test_iso_forward_at_128_with_32px_mono_strokes():
    from flock.toy import env_iso

    ep = env_iso.make_batch(jax.random.PRNGKey(0), 3, 4, "bubble")
    m = Agents(nnx.Rngs(0), action_space="strokes", k_strokes=4, width=8, mono_colour=15, view=env_iso.VIEW, canvas=env_iso.CANVAS)
    fwd = game.forward(m, ep, jax.random.PRNGKey(1), opaque=True)
    assert fwd.logits.shape == (3, 16, 16)
    assert fwd.canvas.alpha.shape == (3, 32, 32)
    assert fwd.receiver_view.shape == (3, 128, 128, 16)
    ground = world.to_onehot(jax.vmap(env_iso.plain_ground_view)(jax.random.split(jax.random.PRNGKey(2), 3)))

    def loss(mm):
        f = game.forward(mm, ep, jax.random.PRNGKey(1), opaque=True)
        pl = game.perceptual_loss(mm.encoder, f.canvas, world.to_onehot(ep.sender_view), ground, env_iso.PROBE_CANVAS_PX)
        return jnp.mean(game.success_loss(f.logits, ep.target_mask)[0]) + pl

    g = nnx.grad(loss)(m)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(g))
    assert any(float(jnp.abs(x).sum()) > 0 for x in jax.tree_util.tree_leaves(g.sender))


def test_channel_jitter_moves_the_drawing_and_keeps_ink():
    from flock.raster import Canvas

    a = jnp.zeros((2, 32, 32)).at[:, 10, 10].set(1.0)
    c = Canvas(colour=jnp.full((2, 32, 32, 16), 1.0 / 16), alpha=a, ink=jnp.array([3.0, 4.0]))
    moved = game.transmit(c, jax.random.PRNGKey(0), jitter=4)
    m = np.asarray(moved.alpha)
    assert np.allclose(m.sum((1, 2)), 1.0)  # the mark is still there, once
    for b in range(2):
        y, x = np.argwhere(m[b] > 0.5)[0]
        assert abs(y - 10) <= 4 and abs(x - 10) <= 4
    assert np.allclose(np.asarray(moved.ink), [3.0, 4.0])
    assert game.transmit(c, jax.random.PRNGKey(0)) is c


def test_transmit_centres_on_ink_and_rotates():
    from flock.raster import Canvas

    # a short horizontal dash near the top-left of a 32 px frame, in two drawings
    a = jnp.zeros((2, 32, 32)).at[:, 3, 2:8].set(1.0)
    c = Canvas(colour=jnp.full((2, 32, 32, 16), 1.0 / 16), alpha=a, ink=jnp.zeros((2,)))
    out = np.asarray(game.transmit(c, jax.random.PRNGKey(1), centre=True).alpha)
    for b in range(2):
        ys, xs = np.nonzero(out[b] > 0.5)
        assert abs(ys.mean() - 15.5) <= 1 and abs(xs.mean() - 15.5) <= 1  # centred
        assert len(ys) == 6  # nothing lost
    rot = np.asarray(game.transmit(c, jax.random.PRNGKey(2), centre=True, rotate=True).alpha)
    shapes = {tuple(np.ptp(np.argwhere(r > 0.5), 0)) for r in rot}
    assert shapes <= {(0, 5), (5, 0)}  # a dash, either way round
