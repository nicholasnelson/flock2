"""Actor-critic trained in imagination, in the DreamerV3 style (Hafner et al. 2023,
arXiv:2301.04104), on the homeostatic reward (Milestone 1, step 3).

- Actor: an MLP over the model state (h, z) giving a categorical over the nine actions (1 %
  uniform mix). Critic: an MLP giving the value; a slow copy (EMA) supplies bootstrap targets and
  a regulariser.
- Imagination: from model states inferred on real data, roll the world model's prior forward
  ``HORIZON`` steps with actions sampled from the actor. The reward of each imagined step is the
  world model's reward head read at the state it leads to (the drive-reduction reward it was
  trained on, ``flock.meadow.drives.reward``).
- Curiosity (optional, the world model having an ensemble): each imagined step also earns
  ``curiosity`` x the ensemble's disagreement about (state, action) divided by the mean
  disagreement at the first actor-critic update (the end of infancy), then frozen — so curiosity
  pays about ``curiosity`` per step at first and fades as the model learns. (Until the review of
  2026-10-24 the divisor was a running mean, so curiosity paid ~``curiosity`` per step forever.)
- Reward centring (``center``, on by default since the second review of 2026-10-24): the
  extrinsic reward has a running mean of itself subtracted before returns are computed (Naik, Wan,
  Tomar & Sutton 2024, "Reward Centering", arXiv:2405.09999). In a task that never ends this
  removes the state-independent part of every value, r_bar / (1 - gamma) (~ -333 here), and
  leaves the small part that differs between states; it changes no policy's ranking. Without
  it the critic, a scalar output starting at zero and trained at Adam 3e-5, was still drifting
  toward -333 after 38,000 updates, so the actor's advantages were mostly the critic's lag.
  (Valid while sheep cannot die; revisit when they can.)
- Critic output (``twohot``, off by default until tested): a scalar trained by squared error, or
  as in DreamerV3 a categorical over the 255 symlog bins of ``wm.BINS`` trained by cross-entropy
  against the two-hot encoding of symlog(return), the value being symexp of the expected bin — so
  a value of -300 is an ordinary target rather than one the output weights must grow to reach.
  The port used the scalar head until the second review of 2026-10-24 found the discrepancy.
- Returns: lambda-returns (gamma 0.997, lambda 0.95) bootstrapped by the slow critic. The actor
  is trained by REINFORCE on advantages divided by max(1, S), S an EMA of the 5th-95th
  percentile range of the *advantages* (returns minus the critic's value), with a small entropy
  bonus; the critic regresses on the returns and toward the slow critic. The world model is
  frozen inside these updates. DreamerV3 takes S from the returns; with varied starting states
  that range measured how differently good and bad states are valued (S grew to ~140 in birth2),
  not what an action changes, and divided the action signal away (review of 2026-10-24, § 1).
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
CENTER_DECAY = 0.99  # EMA of the imagined extrinsic reward, for centring
UNIMIX = 0.01
FEAT = wm.DETER + wm.STOCH * wm.CLASSES


class Head(nnx.Module):
    def __init__(self, out: int, rngs: nnx.Rngs, hidden: int = 256, n_in: int = FEAT):
        self.a = nnx.Linear(n_in, hidden, rngs=rngs)
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
    def __init__(self, rngs: nnx.Rngs, curiosity: float = 0.0, center: bool = True, twohot: bool = False, cfg: wm.Config = wm.MEADOW, hidden: int = 256, entropy: float = ENTROPY, norm: str = "adv"):
        """``norm``: what sets the actor's scale S — the spread of the advantages ("adv", since the
        first review) or of the returns ("ret", DreamerV3's)."""
        assert norm in ("adv", "ret")
        self.n_actions = cfg.n_actions
        self.entropy = entropy
        self.norm = norm
        self.actor = Head(cfg.n_actions, rngs, hidden, cfg.feat)
        self.twohot = twohot
        self.critic = Head(wm.REWARD_BINS if twohot else 1, rngs, hidden, cfg.feat)
        self.slow_critic = Head(wm.REWARD_BINS if twohot else 1, rngs, hidden, cfg.feat)
        nnx.update(self.slow_critic, nnx.state(self.critic))
        self.ret_scale = nnx.Variable(jnp.float32(1.0))
        self.curiosity = nnx.Variable(jnp.float32(curiosity))
        self.center = center
        self.r_bar = nnx.Variable(jnp.float32(jnp.nan))  # running mean of the extrinsic reward (nan: unset)
        self.intr_scale = nnx.Variable(jnp.float32(0.0))  # mean disagreement at the first update, then frozen (0: unset)

    def act(self, feat: jax.Array, key: jax.Array) -> jax.Array:
        return jax.random.categorical(key, policy_logp(self.actor(feat)), -1)


def value_of(agent: Agent, head: Head, feats: jax.Array) -> jax.Array:
    """The value a critic head gives (..., FEAT) -> (...)."""
    out = head(feats)
    if agent.twohot:
        return wm.symexp(jnp.sum(jax.nn.softmax(out, -1) * wm.BINS, -1))
    return out[..., 0]


def _twohot_ce(logits, target):
    return -jnp.sum(wm.twohot(wm.symlog(target)) * jax.nn.log_softmax(logits, -1), -1)


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
        h2 = world.transition(h, z, jax.nn.one_hot(a, agent.n_actions))
        z2 = wm._sample(world.prior(h2), k2)
        return (h2, z2), (f, a)

    (hT, zT), (feats, acts) = jax.lax.scan(body, (h, z), jax.random.split(key, HORIZON))
    feats = jnp.concatenate([feats, feat_of(hT, zT)[None]], 0)
    rewards = world.reward(feats[1:])
    if world.ensemble is not None:
        dis = world.disagreement(feats[:-1], jax.nn.one_hot(acts, agent.n_actions))
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
    values = value_of(agent, agent.critic, feats)
    slow = jax.lax.stop_gradient(value_of(agent, agent.slow_critic, feats))
    rets = jax.lax.stop_gradient(lambda_returns(rewards, slow))
    logp_all = policy_logp(agent.actor(feats[:-1]))
    logp = jnp.take_along_axis(logp_all, acts[..., None], -1)[..., 0]
    entropy = -jnp.sum(jnp.exp(logp_all) * logp_all, -1)
    raw_adv = jax.lax.stop_gradient(rets - values[:-1])
    adv = raw_adv / jnp.maximum(1.0, scale)
    actor_loss = -jnp.mean(adv * logp) - agent.entropy * jnp.mean(entropy)
    if agent.twohot:
        logits = agent.critic(feats[:-1])
        critic_loss = jnp.mean(_twohot_ce(logits, rets) + _twohot_ce(logits, slow[:-1]))
    else:
        critic_loss = jnp.mean(0.5 * (values[:-1] - rets) ** 2 + 0.5 * (values[:-1] - slow[:-1]) ** 2)
    aux = {"actor_loss": actor_loss, "critic_loss": critic_loss, "entropy": jnp.mean(entropy), "imag_reward": jnp.mean(rewards), "return": jnp.mean(rets), "abs_adv": jnp.mean(jnp.abs(raw_adv))}
    return actor_loss + critic_loss, (aux, raw_adv if agent.norm == "adv" else rets)


@nnx.jit
def train_step(world: wm.WorldModel, agent: Agent, opt: nnx.Optimizer, h0, z0, key):
    """One actor-critic update from start states (h0, z0) (N, ...)."""
    feats, acts, extrinsic, dis = imagine(world, agent, h0, z0, key)
    old = agent.intr_scale[...]
    intr_scale = jnp.where(old > 0, old, dis.mean())
    agent.intr_scale[...] = intr_scale
    intrinsic = agent.curiosity[...] * dis / jnp.maximum(intr_scale, 1e-8)
    old_bar = agent.r_bar[...]
    r_bar = jnp.where(jnp.isnan(old_bar), extrinsic.mean(), CENTER_DECAY * old_bar + (1 - CENTER_DECAY) * extrinsic.mean())
    agent.r_bar[...] = r_bar
    rewards = (extrinsic - r_bar if agent.center else extrinsic) + intrinsic
    scale = agent.ret_scale[...]
    (loss, (aux, spread_of)), grads = nnx.value_and_grad(ac_loss, has_aux=True)(agent, feats, acts, rewards, scale)
    opt.update(agent, grads)
    # the actor's scale (EMA of the 5-95 % spread of the advantages or the returns) and the slow critic
    spread = jnp.percentile(spread_of, 95) - jnp.percentile(spread_of, 5)
    agent.ret_scale[...] = RETURN_DECAY * scale + (1 - RETURN_DECAY) * spread
    slow = nnx.state(agent.slow_critic, nnx.Param)
    fast = nnx.state(agent.critic, nnx.Param)
    nnx.update(agent.slow_critic, jax.tree.map(lambda s, f: SLOW_DECAY * s + (1 - SLOW_DECAY) * f, slow, fast))
    aux["ret_scale"] = agent.ret_scale[...]
    aux["extrinsic"] = jnp.mean(extrinsic)
    aux["r_bar"] = r_bar
    aux["intrinsic"] = jnp.mean(intrinsic)
    aux["disagreement"] = jnp.mean(dis)
    return aux


def make_optimizer(agent: Agent, lr: float = 3e-5) -> nnx.Optimizer:
    wrt = nnx.All(nnx.Param, nnx.Not(nnx.PathContains("slow_critic")))
    return nnx.Optimizer(agent, optax.chain(optax.clip_by_global_norm(100.0), optax.adam(lr, eps=1e-5)), wrt=wrt)
