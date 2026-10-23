import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from flock import wm


def test_shapes_loss_and_gradients():
    m = wm.WorldModel(nnx.Rngs(0))
    n = wm.count_params(m)
    assert 2e6 < n < 8e6, n
    views = jax.random.randint(jax.random.PRNGKey(0), (2, 5, wm.H, wm.W), 0, 16)
    acts = jax.random.randint(jax.random.PRNGKey(1), (2, 5), 0, wm.N_ACTIONS)
    (loss, aux), grads = nnx.value_and_grad(wm.loss_fn, has_aux=True)(m, views, acts, jax.random.PRNGKey(2))
    assert np.isfinite(float(loss))
    leaves = jax.tree.leaves(nnx.state(grads, nnx.Param))
    assert all(np.isfinite(np.asarray(g)).all() for g in leaves)
    assert sum(float(jnp.abs(g).sum()) > 0 for g in leaves) > 0.9 * len(leaves)
    s = m.observe(views, acts, jax.random.PRNGKey(3))
    assert s.h.shape == (2, 5, wm.DETER) and s.z.shape == (2, 5, wm.STOCH * wm.CLASSES)
    im = m.imagine(s.h[:, -1], s.z[:, -1], acts[:, :3], jax.random.PRNGKey(4))
    assert m.decode(im).shape == (2, 3, wm.H, wm.W, 16)


def test_learns_a_constant_strip_quickly():
    import optax

    m = wm.WorldModel(nnx.Rngs(1))
    opt = nnx.Optimizer(m, optax.adam(1e-3), wrt=nnx.Param)
    views = jnp.broadcast_to(jnp.where(jnp.arange(wm.H)[:, None] < 24, 5, 1), (2, 4, wm.H, wm.W)).astype(jnp.int32)
    acts = jnp.zeros((2, 4), jnp.int32)

    @nnx.jit
    def step(m, opt, k):
        (l, aux), g = nnx.value_and_grad(wm.loss_fn, has_aux=True)(m, views, acts, k)
        opt.update(m, g)
        return aux

    for i in range(60):
        aux = step(m, opt, jax.random.PRNGKey(i))
    assert float(aux["pixel_acc"]) > 0.99
