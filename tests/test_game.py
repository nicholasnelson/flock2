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
