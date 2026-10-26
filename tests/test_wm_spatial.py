"""The spatial world model (rung 3): shapes, and a finite loss and gradients through wm.loss_fn."""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from flock import wm
from flock.wm_spatial import SPATIAL, SpatialWorldModel


def test_shapes_and_loss():
    c = SPATIAL
    model = SpatialWorldModel(nnx.Rngs(0), n_intero=6)
    B, T = 2, 3
    k = jax.random.PRNGKey(0)
    views = jax.random.randint(k, (B, T, c.h, c.w, 3), 0, 256).astype(jnp.uint8)
    acts = jax.random.randint(k, (B, T), 0, c.n_actions)
    felt = jax.random.uniform(k, (B, T, 6))
    rew = jax.random.normal(k, (B, T))
    s = model.observe(views, acts, k, felt)
    assert s.h.shape == (B, T, c.deter) and s.z.shape == (B, T, c.stoch * c.classes)
    assert s.post.shape == (B, T, c.stoch, c.classes)
    assert model.decode(s).shape == (B, T, c.h, c.w, 3)
    (loss, aux), g = nnx.value_and_grad(wm.loss_fn, has_aux=True)(model, views, acts, k, felt, rew)
    assert np.isfinite(float(loss))
    leaves = jax.tree.leaves(nnx.state(g))
    assert all(np.isfinite(np.asarray(x)).all() for x in leaves)
    n_params = sum(int(np.prod(x.shape)) for x in jax.tree.leaves(nnx.state(model, nnx.Param)))
    print("params", n_params, "feat", c.feat)


def test_one_step_interface():
    """The hybrid's per-step use: embed, transition, posterior sample."""
    c = SPATIAL
    model = SpatialWorldModel(nnx.Rngs(1), n_intero=6)
    S = 5
    h, z = model.initial(S)
    views = jnp.zeros((S, c.h, c.w, 3), jnp.uint8)
    e = model.embed_obs(views, jnp.zeros((S, 6)))
    h = model.transition(h, z, jax.nn.one_hot(jnp.zeros(S, jnp.int32), c.n_actions))
    z = wm._sample(model.posterior(h, e), jax.random.PRNGKey(0))
    assert h.shape == (S, c.deter) and z.shape == (S, c.stoch * c.classes)
