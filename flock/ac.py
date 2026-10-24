"""Actor-critic trained in imagination, in the DreamerV3 style (Hafner et al. 2023,
arXiv:2301.04104; Nature 2025, doi:10.1038/s41586-025-08744-2), on the homeostatic reward
(Milestone 1, step 3).

- Actor: an MLP over the model state (h, z) giving a categorical over the actions (1 % uniform
  mix). Critic: an MLP giving the value; a slow copy (EMA) is a regulariser.
- Imagination: from model states inferred on real data, roll the world model's prior forward
  ``HORIZON`` steps with actions sampled from the actor. The reward of each imagined step is the
  world model's reward head read at the state it leads to. ``update`` takes any rollout
  (features, actions, rewards), so the same update can run on the true simulator
  (``scripts/grid_true_ac.py``) — the sanity rung that separates the actor-critic from the model.
- Curiosity (optional, the world model having an ensemble): each imagined step also earns
  ``curiosity`` x the ensemble's disagreement about (state, action) divided by the mean
  disagreement at the first update, then frozen, so it fades as the model learns.
- Reward centring (``center``, on by default): a running mean of the extrinsic reward is
  subtracted before returns are computed (Naik, Wan, Tomar & Sutton 2024, arXiv:2405.09999). In a
  task that never ends this removes the state-independent r_bar / (1 - gamma) from every value
  and changes no policy's ranking. Any error in r_bar is multiplied by 1 / (1 - gamma) in the
  values (third review of 2026-10-24), so ``gamma`` is a setting: 0.997 (DreamerV3's) or lower
  where the task's cycles are short.
- Returns: lambda-returns (``gamma``, lambda 0.95). As DreamerV3's current defaults
  (``configs.yaml``, checked 2026-10-24): returns are bootstrapped from the *same* critic that
  is the baseline (``slowtar: False``; until the third review ac.py bootstrapped from the slow
  critic, which put the fast/slow gap into every advantage); the slow critic only regularises;
  the actor's advantages are divided by max(1, S), S an EMA of the 5th-95th percentile range of
  the *returns* (``retnorm``; the first review's switch to the advantages' range, ``norm="adv"``,
  is kept as an option); and the critic is also trained on lambda-returns computed from *real*
  replayed rewards (``repval``, weight 0.3), which anchors it to real reward.
- Critic output (``twohot``): a scalar trained by squared error, or as in DreamerV3 a
  categorical over the 255 symlog bins of ``wm.BINS`` trained by cross-entropy against the
  two-hot of symlog(return). Alone it does not fix a lagging critic (offset bandit test).
- Health readings in ``aux``: ``value_start`` (mean value at the start states; with centring it
  should stay near 0) and, with replay sequences, ``value_corr`` (correlation of the critic's
  value with the real lambda-return on them).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from flock import wm

HORIZON = 32  # DreamerV3 uses 15; too short here for a thirsty sheep to reach water (M1 run 2)
GAMMA, LAMBDA = 0.997, 0.95
ENTROPY = 3e-4  # as DreamerV3
SLOW_DECAY = 0.98
RETURN_DECAY = 0.99
CENTER_DECAY = 0.99  # EMA of the imagined extrinsic reward, for centring
REPVAL = 0.3  # weight of the replay value loss, as DreamerV3
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
    def __init__(
        self,
        rngs: nnx.Rngs,
        curiosity: float = 0.0,
        center: bool = True,
        twohot: bool = False,
        cfg: wm.Config = wm.MEADOW,
        hidden: int = 256,
        entropy: float = ENTROPY,
        norm: str = "ret",
        gamma: float = GAMMA,
        slowtar: bool = False,
        repval: float = REPVAL,
        n_in: int | None = None,
    ):
        """``norm``: what sets the actor's scale S — the spread of the returns ("ret", DreamerV3's)
        or of the advantages ("adv"). ``n_in``: the feature size (default the world model's state;
        set it for an agent on other features, e.g. the true grid state)."""
        assert norm in ("adv", "ret")
        n_in = cfg.feat if n_in is None else n_in
        self.n_actions = cfg.n_actions
        self.entropy, self.norm, self.gamma, self.slowtar, self.repval = entropy, norm, gamma, slowtar, repval
        self.actor = Head(cfg.n_actions, rngs, hidden, n_in)
        self.twohot = twohot
        self.critic = Head(wm.REWARD_BINS if twohot else 1, rngs, hidden, n_in)
        self.slow_critic = Head(wm.REWARD_BINS if twohot else 1, rngs, hidden, n_in)
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


def _critic_fit(agent: Agent, feats, target):
    """The critic's loss toward ``target`` (stop-gradient) at ``feats``."""
    if agent.twohot:
        return _twohot_ce(agent.critic(feats), target)
    return 0.5 * (value_of(agent, agent.critic, feats) - target) ** 2


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


def lambda_returns(rewards, values, gamma=GAMMA):
    """rewards (H, N), values (H+1, N) -> returns (H, N)."""

    def body(nxt, x):
        r, v_next = x
        ret = r + gamma * ((1 - LAMBDA) * v_next + LAMBDA * nxt)
        return ret, ret

    _, rets = jax.lax.scan(body, values[-1], (rewards, values[1:]), reverse=True)
    return rets


def ac_loss(agent: Agent, feats, acts, rewards, scale, real_feats=None, real_rewards=None):
    sg = jax.lax.stop_gradient
    values = value_of(agent, agent.critic, feats)
    slow = sg(value_of(agent, agent.slow_critic, feats))
    rets = sg(lambda_returns(rewards, slow if agent.slowtar else sg(values), agent.gamma))
    logp_all = policy_logp(agent.actor(feats[:-1]))
    logp = jnp.take_along_axis(logp_all, acts[..., None], -1)[..., 0]
    entropy = -jnp.sum(jnp.exp(logp_all) * logp_all, -1)
    raw_adv = sg(rets - values[:-1])
    adv = raw_adv / jnp.maximum(1.0, scale)
    actor_loss = -jnp.mean(adv * logp) - agent.entropy * jnp.mean(entropy)
    critic_loss = jnp.mean(_critic_fit(agent, feats[:-1], rets) + _critic_fit(agent, feats[:-1], slow[:-1]))
    aux = {"actor_loss": actor_loss, "critic_loss": critic_loss, "entropy": jnp.mean(entropy), "imag_reward": jnp.mean(rewards),
           "return": jnp.mean(rets), "abs_adv": jnp.mean(jnp.abs(raw_adv)), "value_start": jnp.mean(values[0])}
    if real_feats is not None:
        # the replay value loss: lambda-returns on real replayed rewards, bootstrapped by the critic
        rv = sg(value_of(agent, agent.critic, real_feats))
        real_rets = sg(lambda_returns(real_rewards[:-1], rv, agent.gamma))
        rep = jnp.mean(_critic_fit(agent, real_feats[:-1], real_rets))
        critic_loss = critic_loss + agent.repval * rep
        v, r = rv[:-1].ravel(), real_rets.ravel()
        aux["repval_loss"] = rep
        aux["value_corr"] = jnp.sum((v - v.mean()) * (r - r.mean())) / jnp.sqrt(jnp.sum((v - v.mean()) ** 2) * jnp.sum((r - r.mean()) ** 2) + 1e-12)
        aux["real_return"] = jnp.mean(real_rets)
    return actor_loss + critic_loss, (aux, raw_adv if agent.norm == "adv" else rets)


def _centre(agent: Agent, extrinsic):
    old_bar = agent.r_bar[...]
    r_bar = jnp.where(jnp.isnan(old_bar), extrinsic.mean(), CENTER_DECAY * old_bar + (1 - CENTER_DECAY) * extrinsic.mean())
    agent.r_bar[...] = r_bar
    return r_bar


def _update(agent: Agent, opt, feats, acts, extrinsic, dis, real_feats=None, real_rewards=None):
    """One actor-critic update from a rollout: feats (H+1, N, F), acts (H, N), extrinsic rewards
    and curiosity disagreement (H, N); optionally real sequences (T, N, F) and their rewards
    (T, N) for the replay value loss."""
    old = agent.intr_scale[...]
    intr_scale = jnp.where(old > 0, old, dis.mean())
    agent.intr_scale[...] = intr_scale
    intrinsic = agent.curiosity[...] * dis / jnp.maximum(intr_scale, 1e-8)
    r_bar = _centre(agent, extrinsic)
    shift = r_bar if agent.center else 0.0
    rewards = extrinsic - shift + intrinsic
    real_c = None if real_rewards is None else real_rewards - shift
    scale = agent.ret_scale[...]
    (loss, (aux, spread_of)), grads = nnx.value_and_grad(ac_loss, has_aux=True)(agent, feats, acts, rewards, scale, real_feats, real_c)
    opt.update(agent, grads)
    # the actor's scale (EMA of the 5-95 % spread of the returns or the advantages) and the slow critic
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


update = nnx.jit(_update)


@nnx.jit
def train_step(world: wm.WorldModel, agent: Agent, opt, h0, z0, key, real_feats=None, real_rewards=None):
    """One actor-critic update from start states (h0, z0) (N, ...), imagining with the world model;
    optionally with real replay sequences for the replay value loss."""
    feats, acts, extrinsic, dis = imagine(world, agent, h0, z0, key)
    return _update(agent, opt, feats, acts, extrinsic, dis, real_feats, real_rewards)


class Optim(nnx.Module):
    """Adam for the actor and the critic, with their own learning rates (the slow critic is not
    trained, only averaged)."""

    def __init__(self, agent: Agent, lr: float, critic_lr: float):
        tx = lambda r: optax.chain(optax.clip_by_global_norm(100.0), optax.adam(r, eps=1e-5))
        self.actor = nnx.Optimizer(agent, tx(lr), wrt=nnx.All(nnx.Param, nnx.PathContains("actor")))
        self.critic = nnx.Optimizer(agent, tx(critic_lr), wrt=nnx.All(nnx.Param, nnx.PathContains("critic"), nnx.Not(nnx.PathContains("slow_critic"))))

    def update(self, agent: Agent, grads):
        self.actor.update(agent, grads)
        self.critic.update(agent, grads)


def make_optimizer(agent: Agent, lr: float = 3e-5, critic_lr: float | None = None) -> Optim:
    return Optim(agent, lr, lr if critic_lr is None else critic_lr)
