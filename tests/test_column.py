"""Stage 2a's recurrent column (spec § 21.1): the widened policy starts as stage 1 exactly, the mud
ramp, and the PPO update through the column."""

import importlib.util
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

_spec = importlib.util.spec_from_file_location("m2_hybrid", os.path.join(os.path.dirname(__file__), "..", "scripts", "m2_hybrid.py"))
hy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hy)


def test_widened_policy_is_stage_1_at_start():
    old = hy.make_policy(0, 32, n_in=10)
    pure = jax.tree.map(np.asarray, nnx.to_pure_dict(nnx.state(old)))
    new = hy.make_policy(1, 32, n_in=10 + 4)
    hy.load_widened(new, pure)
    x = jax.random.normal(jax.random.PRNGKey(2), (5, 10))
    g = jax.random.normal(jax.random.PRNGKey(3), (5, 4)) * 10.0
    lo, vo = old(x)
    ln, vn = new(jnp.concatenate([x, g], -1))
    assert np.allclose(np.asarray(lo), np.asarray(ln), atol=1e-6)
    assert np.allclose(np.asarray(vo), np.asarray(vn), atol=1e-6)


def test_mud_radius_ramp():
    ramp = (200.0, 1400.0, 8.0)
    assert hy.mud_radius(1, ramp) == 0.0 and hy.mud_radius(200, ramp) == 0.0
    assert abs(hy.mud_radius(800, ramp) - 4.0) < 1e-9
    assert hy.mud_radius(1400, ramp) == 8.0 and hy.mud_radius(2000, ramp) == 8.0
    assert hy.mud_radius(5000, None) == 0.0


def test_update_recurrent_trains_the_column():
    column = hy.Column(nnx.Rngs(0), n_in=7, size=8)
    policy = hy.make_policy(0, 16, n_in=5 + 8)
    brain = hy.Brain(policy, column)
    opt = nnx.Optimizer(brain, optax.adam(1e-2), wrt=nnx.Param)
    before = jax.tree.map(np.array, nnx.to_pure_dict(nnx.state(column)))
    T, N = 6, 4
    k = jax.random.split(jax.random.PRNGKey(0), 5)
    feat = jax.random.normal(k[0], (T, N, 5))
    x = jax.random.normal(k[1], (T, N, 7))
    act = jax.random.randint(k[2], (T, N), 0, 8)
    adv = jax.random.normal(k[3], (T, N))
    ret = jax.random.normal(k[4], (T, N))
    aux = hy.update_recurrent(brain, opt, feat, x, jnp.zeros((N, 8)), act, jnp.zeros((T, N)), adv, ret, jnp.ones((T, N)), 0.2, 0.01)
    assert set(aux) == {"pg", "vf", "entropy", "approx_kl"}
    after = jax.tree.map(np.array, nnx.to_pure_dict(nnx.state(column)))
    changed = jax.tree.leaves(jax.tree.map(lambda a, b: bool(np.any(a != b)), before, after))
    assert any(changed)


def test_zero_mems_carry_the_column_state():
    world = hy.make_world(0)
    h, z, a, g = hy.zero_mems(world, 2, 3, 16)
    assert g.shape == (2, 3, 16) and h.shape[:2] == (2, 3)
    assert hy.zero_mems(world, 2, 3)[3].shape == (2, 3, 0)
    assert hy.n_policy_in(world, True, 16) == world.cfg.feat + world.cfg.embed + 16
