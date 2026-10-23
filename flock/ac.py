"""Actor-critic trained in imagination, in the DreamerV3 style (Hafner et al. 2023,
arXiv:2301.04104), on the drive-reduction reward (Milestone 1, step 3).

- Actor: an MLP over the model state (h, z) giving a categorical over the nine actions (1 %
  uniform mix). Critic: an MLP giving the value; a slow copy (EMA) supplies bootstrap targets and
  a regulariser.
- Imagination: from model states inferred on real data, roll the world model's prior forward
  ``HORIZON`` steps with actions sampled from the actor. The reward of each imagined step is the
  world model's reward head read at the state it leads to (the drive-reduction reward it was
  trained on, ``flock.meadow.drives.reward``).
- Curiosity (optional, the world model having an ensemble): each imagined step also earns
  ``curiosity`` x the ensemble's disagreement about (state, action) divided by a running mean of
  the disagreement, so a typical step earns about ``curiosity``.
- Returns: lambda-returns (gamma 0.997, lambda 0.95) bootstrapped by the slow critic. The actor
  is trained by REINFORCE on advantages divided by max(1, S), S an EMA of the 5th-95th
  percentile range of the returns, with a small entropy bonus; the critic regresses on the
  returns and toward the slow critic. The world model is frozen inside these updates.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from flock import wm

HORIZON = 32  # DreamerV3 uses 15; too short here for a thirsty sheep to reach water (M1 run 2)
GAMMA, LAMBDA = 0.997, 0.95
ENTROPY = 3e-4  # as DreamerV3 (3e-3 kept M1 run 2's policy uniform for 530 iterations)
SLOW_DECAY = 0.98
RETURN_DECAY = 0.99
UNIMIX = 0.01
FEAT = wm.DETER + wm.STOCH * wm.CLASSES


class Head(nnx.Module):
    def __init__(self, out: int, rngs: nnx.Rngs, hidden: int = 256):
        self.a = nnx.Linear(FEAT, hidden, rngs=rngs)
        self.na = nnx.LayerNorm(hidden, rngs=rngs)
        self.b = nnx.Linear(hidden, hidden, rngs=rngs)
        self.nb = nnx.LayerNorm(hidden, rngs=rngs)
        self.c = nnx.Linear(hidden, out, rngs=rngs, kernel_init=nnx.initializers.zeros)

    def __call__(self, x):
        x = jax.nn.silu(self.na(self.a(x)))
        x = jax.nn.silu(self.nb(self.b(x)))
        return self.c(x)


def policy_logp(logits: jax.Array) -> jax.Array:
    p = (1 - UNIMIX) * jax.nn.softmax(logits, -1) + UNIMIX / logits.shape[-1]
    return jnp.log(p)


class Agent(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, curiosity: float = 0.0):
        self.actor = Head(wm.N_ACTIONS, rngs)
        self.critic = Head(1, rngs)
        self.slow_critic = Head(1, rngs)
        nnx.update(self.slow_critic, nnx.state(self.critic))
        self.ret_scale = nnx.Variable(jnp.float32(1.0))
        self.curiosity = nnx.Variable(jnp.float32(curiosity))
        self.intr_scale = nnx.Variable(jnp.float32(0.0))  # running mean of the disagreement (0: unset)

    def act(self, feat: jax.Array, key: jax.Array) -> jax.Array:
        return jax.random.categorical(key, policy_logp(self.actor(feat)), -1)


def feat_of(h, z):
    return jnp.concatenate([h, z], -1)


def imagine(world: wm.WorldModel, agent: Agent, h, z, key):
    """Roll the prior ``HORIZON`` steps from (h, z) (N, ...) with the actor's actions.
    Returns feats (H+1, N, FEAT), actions (H, N), extrinsic rewards (H, N) and the ensemble's
    disagreement about each (state, action) (H, N; zeros without an ensemble)."""

    def body(carry, k):
        h, z = carry
        k1, k2 = jax.random.split(k)
        f = feat_of(h, z)
        a = agent.act(jax.lax.stop_gradient(f), k1)
        h2 = world.transition(h, z, jax.nn.one_hot(a, wm.N_ACTIONS))
        z2 = wm._sample(world.prior(h2), k2)
        return (h2, z2), (f, a)

    (hT, zT), (feats, acts) = jax.lax.scan(body, (h, z), jax.random.split(key, HORIZON))
    feats = jnp.concatenate([feats, feat_of(hT, zT)[None]], 0)
    rewards = world.reward(feats[1:])
    if world.ensemble is not None:
        dis = world.disagreement(feats[:-1], jax.nn.one_hot(acts, wm.N_ACTIONS))
    else:
        dis = jnp.zeros_like(rewards)
    sg = jax.lax.stop_gradient
    return sg(feats), acts, sg(rewards), sg(dis)


def lambda_returns(rewards, values):
    """rewards (H, N), values (H+1, N) -> returns (H, N)."""

    def body(nxt, x):
        r, v_next = x
        ret = r + GAMMA * ((1 - LAMBDA) * v_next + LAMBDA * nxt)
        return ret, ret

    _, rets = jax.lax.scan(body, values[-1], (rewards, values[1:]), reverse=True)
    return rets


def ac_loss(agent: Agent, feats, acts, rewards, scale):
    values = agent.critic(feats)[..., 0]
    slow = jax.lax.stop_gradient(agent.slow_critic(feats)[..., 0])
    rets = jax.lax.stop_gradient(lambda_returns(rewards, slow))
    logp_all = policy_logp(agent.actor(feats[:-1]))
    logp = jnp.take_along_axis(logp_all, acts[..., None], -1)[..., 0]
    entropy = -jnp.sum(jnp.exp(logp_all) * logp_all, -1)
    adv = jax.lax.stop_gradient((rets - values[:-1]) / jnp.maximum(1.0, scale))
    actor_loss = -jnp.mean(adv * logp) - ENTROPY * jnp.mean(entropy)
    critic_loss = jnp.mean(0.5 * (values[:-1] - rets) ** 2 + 0.5 * (values[:-1] - slow[:-1]) ** 2)
    aux = {"actor_loss": actor_loss, "critic_loss": critic_loss, "entropy": jnp.mean(entropy), "imag_reward": jnp.mean(rewards), "return": jnp.mean(rets)}
    return actor_loss + critic_loss, (aux, rets)


@nnx.jit
def train_step(world: wm.WorldModel, agent: Agent, opt: nnx.Optimizer, h0, z0, key):
    """One actor-critic update from start states (h0, z0) (N, ...)."""
    feats, acts, extrinsic, dis = imagine(world, agent, h0, z0, key)
    old = agent.intr_scale[...]
    intr_scale = jnp.where(old > 0, RETURN_DECAY * old + (1 - RETURN_DECAY) * dis.mean(), dis.mean())
    agent.intr_scale[...] = intr_scale
    intrinsic = agent.curiosity[...] * dis / jnp.maximum(intr_scale, 1e-8)
    rewards = extrinsic + intrinsic
    scale = agent.ret_scale[...]
    (loss, (aux, rets)), grads = nnx.value_and_grad(ac_loss, has_aux=True)(agent, feats, acts, rewards, scale)
    opt.update(agent, grads)
    # return-scale EMA and the slow critic
    spread = jnp.percentile(rets, 95) - jnp.percentile(rets, 5)
    agent.ret_scale[...] = RETURN_DECAY * scale + (1 - RETURN_DECAY) * spread
    slow = nnx.state(agent.slow_critic, nnx.Param)
    fast = nnx.state(agent.critic, nnx.Param)
    nnx.update(agent.slow_critic, jax.tree.map(lambda s, f: SLOW_DECAY * s + (1 - SLOW_DECAY) * f, slow, fast))
    aux["ret_scale"] = agent.ret_scale[...]
    aux["extrinsic"] = jnp.mean(extrinsic)
    aux["intrinsic"] = jnp.mean(intrinsic)
    aux["disagreement"] = jnp.mean(dis)
    return aux


def make_optimizer(agent: Agent) -> nnx.Optimizer:
    wrt = nnx.All(nnx.Param, nnx.Not(nnx.PathContains("slow_critic")))
    return nnx.Optimizer(agent, optax.chain(optax.clip_by_global_norm(100.0), optax.adam(3e-5, eps=1e-5)), wrt=wrt)
