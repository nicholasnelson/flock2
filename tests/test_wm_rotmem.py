"""Rung 4's turn-rotated memory (``flock.wm``, ``rot_bins``): the roll, and the model with it."""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from flock import wm

CFG = wm.Config(h=32, w=64, n_classes=3, rgb=True, n_actions=8, enc_ch=(8, 16, 16, 32), bottom=(2, 4), deter=64, hidden=64, stoch=4,
                classes=4, embed=64, body_dims=(), rot_bins=8, rot_ch=3, rot_shift=2, rot_left=3, rot_right=4)


def test_rotation():
    m = jax.random.normal(jax.random.PRNGKey(0), (2, CFG.rot_bins, CFG.rot_ch))
    a = lambda i: jax.nn.one_hot(jnp.full((2,), i), CFG.n_actions)
    left = wm.rotate_memory(m, a(3), CFG)
    np.testing.assert_allclose(left, jnp.roll(m, 2, 1))
    np.testing.assert_allclose(wm.rotate_memory(left, a(4), CFG), m, atol=1e-6)  # right undoes left
    for i in (0, 1, 2, 5, 6, 7):  # no other action moves it
        np.testing.assert_allclose(wm.rotate_memory(m, a(i), CFG), m)
    four = m
    for _ in range(CFG.rot_bins // CFG.rot_shift):  # a full circle of turns returns it
        four = wm.rotate_memory(four, a(3), CFG)
    np.testing.assert_allclose(four, m, atol=1e-6)


def test_model_with_ring():
    model = wm.WorldModel(nnx.Rngs(0), n_intero=5, reward=True, cfg=CFG._replace(ms_weight=1.0, ms_horizons=(2, 4), ms_stride=2))
    assert CFG.state == 64 + 8 * 3 and CFG.feat == CFG.state + 16
    B, T = 2, 8
    k = jax.random.PRNGKey(1)
    views = jax.random.randint(k, (B, T, CFG.h, CFG.w, 3), 0, 256).astype(jnp.uint8)
    acts = jax.random.randint(k, (B, T), 0, CFG.n_actions)
    felt = jax.random.uniform(k, (B, T, 5))
    s = model.observe(views, acts, k, felt)
    assert s.h.shape == (B, T, CFG.state)
    (loss, aux), g = nnx.value_and_grad(wm.loss_fn, has_aux=True)(model, views, acts, k, felt, jax.random.normal(k, (B, T)))
    assert np.isfinite(float(loss)) and "multistep" in aux
    assert float(jnp.abs(g.mem_write.kernel[...]).sum()) > 0  # the write is trained through the losses
    h, z = model.initial(3)
    h2 = model.transition(h, z, jax.nn.one_hot(jnp.array([3, 4, 0]), CFG.n_actions))
    assert h2.shape == (3, CFG.state)


def test_no_ring_unchanged():
    c = CFG._replace(rot_bins=0)
    assert c.state == c.deter
    model = wm.WorldModel(nnx.Rngs(0), cfg=c)
    assert model.mem_write is None
