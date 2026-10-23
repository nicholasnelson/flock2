import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from flock import ac, wm


def test_lambda_returns_match_a_hand_computation():
    r = jnp.array([[1.0], [0.0], [2.0]])
    v = jnp.array([[0.0], [0.5], [1.0], [3.0]])
    rets = np.asarray(ac.lambda_returns(r, v))[:, 0]
    g, l = ac.GAMMA, ac.LAMBDA
    R2 = 2.0 + g * 3.0
    R1 = 0.0 + g * ((1 - l) * 1.0 + l * R2)
    R0 = 1.0 + g * ((1 - l) * 0.5 + l * R1)
    assert np.allclose(rets, [R0, R1, R2], atol=1e-5)


def test_an_actor_critic_step_runs_and_changes_only_the_agent():
    world = wm.WorldModel(nnx.Rngs(0), n_intero=5, reward=True)
    agent = ac.Agent(nnx.Rngs(1))
    opt = ac.make_optimizer(agent)
    h0 = jax.random.normal(jax.random.PRNGKey(0), (6, wm.DETER))
    z0 = jax.nn.one_hot(jax.random.randint(jax.random.PRNGKey(1), (6, wm.STOCH), 0, wm.CLASSES), wm.CLASSES).reshape(6, -1)
    w_before = np.asarray(world.encoder.out.kernel[...]).copy()
    a_before = np.asarray(agent.actor.c.kernel[...]).copy()  # the zero-initialised output layer moves first
    aux = ac.train_step(world, agent, opt, h0, z0, jax.random.PRNGKey(2))
    assert all(np.isfinite(float(v)) for v in aux.values())
    assert np.allclose(np.asarray(world.encoder.out.kernel[...]), w_before)
    assert not np.allclose(np.asarray(agent.actor.c.kernel[...]), a_before)
