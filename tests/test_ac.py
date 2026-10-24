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


def test_curiosity_adds_a_reward_of_about_its_weight():
    world = wm.WorldModel(nnx.Rngs(0), n_intero=5, reward=True, ensemble=3)
    agent = ac.Agent(nnx.Rngs(1), curiosity=0.5)
    opt = ac.make_optimizer(agent)
    h0 = jax.random.normal(jax.random.PRNGKey(0), (6, wm.DETER))
    z0 = jax.nn.one_hot(jax.random.randint(jax.random.PRNGKey(1), (6, wm.STOCH), 0, wm.CLASSES), wm.CLASSES).reshape(6, -1)
    aux = ac.train_step(world, agent, opt, h0, z0, jax.random.PRNGKey(2))
    assert float(aux["disagreement"]) > 0
    assert abs(float(aux["intrinsic"]) - 0.5) < 0.05  # normalised by its own running mean on the first step


class _BanditWorld(nnx.Module):
    """A stand-in world: the state remembers the last action (h[:9] one-hot) and only action 3
    is rewarded. A working actor-critic must learn to prefer it."""

    ensemble = None

    def transition(self, h, z, a_onehot):
        return jnp.zeros_like(h).at[..., : wm.N_ACTIONS].set(a_onehot)

    def prior(self, h):
        return jnp.zeros(h.shape[:-1] + (wm.STOCH, wm.CLASSES))

    def reward(self, feat):
        return feat[..., 3]


def test_the_actor_critic_learns_a_bandit():
    world = _BanditWorld()
    agent = ac.Agent(nnx.Rngs(3))
    opt = ac.make_optimizer(agent)
    h0 = jnp.zeros((64, wm.DETER))
    z0 = jnp.zeros((64, wm.STOCH * wm.CLASSES))
    p = lambda: float(jnp.exp(ac.policy_logp(agent.actor(ac.feat_of(h0, z0))))[:, 3].mean())
    before = p()
    for i in range(300):
        ac.train_step(world, agent, opt, h0, z0, jax.random.PRNGKey(i))
    assert before < 0.15 and p() > 0.3, (before, p())  # 0.11 -> ~0.42 in 300 updates at Adam 3e-5


def test_the_curiosity_scale_is_frozen_after_the_first_update():
    world = wm.WorldModel(nnx.Rngs(0), n_intero=5, reward=True, ensemble=3)
    agent = ac.Agent(nnx.Rngs(1), curiosity=0.5)
    opt = ac.make_optimizer(agent)
    h0 = jax.random.normal(jax.random.PRNGKey(0), (6, wm.DETER))
    z0 = jax.nn.one_hot(jax.random.randint(jax.random.PRNGKey(1), (6, wm.STOCH), 0, wm.CLASSES), wm.CLASSES).reshape(6, -1)
    ac.train_step(world, agent, opt, h0, z0, jax.random.PRNGKey(2))
    first = float(agent.intr_scale[...])
    ac.train_step(world, agent, opt, 3 * h0, z0, jax.random.PRNGKey(3))
    assert float(agent.intr_scale[...]) == first
