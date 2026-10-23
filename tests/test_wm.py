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


def test_interoception_and_loading_a_vision_only_checkpoint():
    import pickle

    old = wm.WorldModel(nnx.Rngs(5))
    pure = jax.tree.map(np.asarray, nnx.to_pure_dict(nnx.state(old)))
    new = wm.WorldModel(nnx.Rngs(6), n_intero=5)
    wm.load_into(new, pickle.loads(pickle.dumps(pure)))
    a = np.asarray(old.encoder.out.kernel[...])
    b = np.asarray(new.encoder.out.kernel[...])
    assert np.allclose(a, b)  # shared parts loaded
    views = jax.random.randint(jax.random.PRNGKey(0), (2, 4, wm.H, wm.W), 0, 16)
    acts = jax.random.randint(jax.random.PRNGKey(1), (2, 4), 0, wm.N_ACTIONS)
    intero = jax.random.uniform(jax.random.PRNGKey(2), (2, 4, 5))
    loss, aux = wm.loss_fn(new, views, acts, jax.random.PRNGKey(3), intero)
    assert np.isfinite(float(loss)) and "intero_mse" in aux
    s = new.observe(views, acts, jax.random.PRNGKey(4), intero)
    assert new.decode_intero(s).shape == (2, 4, 5)


def test_twohot_round_trips_through_the_bins():
    y = jnp.array([-73.0, -0.08, 0.0, 0.24, 3.2, 71.8])
    t = wm.twohot(wm.symlog(y))
    assert np.allclose(np.asarray(t.sum(-1)), 1.0, atol=1e-5)
    back = wm.symexp(jnp.sum(t * wm.BINS, -1))
    assert np.allclose(np.asarray(back), np.asarray(y), rtol=1e-4, atol=1e-5)


def test_reward_head_starts_at_zero_and_trains():
    m = wm.WorldModel(nnx.Rngs(7), n_intero=5, reward=True)
    views = jax.random.randint(jax.random.PRNGKey(0), (2, 4, wm.H, wm.W), 0, 16)
    acts = jax.random.randint(jax.random.PRNGKey(1), (2, 4), 0, wm.N_ACTIONS)
    intero = jax.random.uniform(jax.random.PRNGKey(2), (2, 4, 5))
    reward = jnp.full((2, 4), 5.0)
    s = m.observe(views, acts, jax.random.PRNGKey(3), intero)
    assert np.allclose(np.asarray(m.reward(jnp.concatenate([s.h, s.z], -1))), 0.0, atol=1e-5)
    (loss, aux), g = nnx.value_and_grad(wm.loss_fn, has_aux=True)(m, views, acts, jax.random.PRNGKey(3), intero, reward)
    assert np.isfinite(float(aux["reward_ce"]))
    assert float(jnp.abs(g.reward_head.b.kernel[...]).sum()) > 0
