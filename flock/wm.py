"""A world model in the DreamerV3 style (Hafner et al. 2023, arXiv:2301.04104), for the sheep's
first-person strip (``flock.fpv``: 64 x 256, 16 palette colours) and their actions.

- Encoder: five stride-2 convolutions over the one-hot strip (64 x 256 -> 2 x 8), then a dense
  embedding.
- RSSM: a layer-normalised GRU deterministic state ``h`` and a stochastic state ``z`` of
  STOCH categorical variables with CLASSES classes each (1 % uniform mix, straight-through
  samples). The transition takes (z_{t-1}, a_{t-1}); the prior predicts z_t from h_t; the
  posterior also sees the embedding of the strip at t.
- Decoder: the mirror of the encoder, logits over the 16 colours per pixel.
- Loss: per-pixel cross-entropy summed over the strip, plus KL balancing — a dynamics term
  KL(sg(post) || prior) weighted 0.5 and a representation term KL(post || sg(prior)) weighted
  0.1, each with 1 nat of free bits (all as in DreamerV3).

Sizes are chosen to land in the plan's 2-8M parameters; ``count_params`` reports the real number.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx

from flock.palette import N_COLOURS

H, W = 64, 256
N_ACTIONS = 9
DETER, HIDDEN = 384, 384
STOCH, CLASSES = 32, 16
EMBED = 512
UNIMIX = 0.01
FREE_NATS, KL_DYN, KL_REP = 1.0, 0.5, 0.1
ENC_CH = (32, 64, 96, 128, 192)  # 64x256 -> 32x128 -> 16x64 -> 8x32 -> 4x16 -> 2x8
BOTTOM = (2, 8)


def _act(x):
    return jax.nn.silu(x)


class Encoder(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        ch = (N_COLOURS,) + ENC_CH
        self.convs = nnx.List([nnx.Conv(ch[i], ch[i + 1], (4, 4), strides=2, padding="SAME", rngs=rngs) for i in range(len(ENC_CH))])
        self.norms = nnx.List([nnx.LayerNorm(c, rngs=rngs) for c in ENC_CH])
        self.out = nnx.Linear(BOTTOM[0] * BOTTOM[1] * ENC_CH[-1], EMBED, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """x: (..., 64, 256, 16) one-hot -> (..., EMBED)."""
        lead = x.shape[:-3]
        h = x.reshape((-1,) + x.shape[-3:])
        for conv, norm in zip(self.convs, self.norms):
            h = _act(norm(conv(h)))
        return self.out(h.reshape(h.shape[0], -1)).reshape(lead + (EMBED,))


class Decoder(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.inp = nnx.Linear(DETER + STOCH * CLASSES, BOTTOM[0] * BOTTOM[1] * ENC_CH[-1], rngs=rngs)
        ch = ENC_CH[::-1] + (N_COLOURS,)
        self.deconvs = nnx.List([nnx.ConvTranspose(ch[i], ch[i + 1], (4, 4), strides=(2, 2), padding="SAME", rngs=rngs) for i in range(len(ENC_CH))])
        self.norms = nnx.List([nnx.LayerNorm(c, rngs=rngs) for c in ch[1:-1]])

    def __call__(self, feat: jax.Array) -> jax.Array:
        """feat: (..., DETER + STOCH*CLASSES) -> logits (..., 64, 256, 16)."""
        lead = feat.shape[:-1]
        h = self.inp(feat.reshape(-1, feat.shape[-1])).reshape((-1,) + BOTTOM + (ENC_CH[-1],))
        for i, dc in enumerate(self.deconvs):
            h = dc(h)
            if i < len(self.norms):
                h = _act(self.norms[i](h))
        return h.reshape(lead + (H, W, N_COLOURS))


class GRU(nnx.Module):
    """DreamerV3's layer-normalised GRU cell (update gate biased toward keeping the state)."""

    def __init__(self, inp: int, size: int, rngs: nnx.Rngs):
        self.lin = nnx.Linear(inp + size, 3 * size, use_bias=False, rngs=rngs)
        self.norm = nnx.LayerNorm(3 * size, rngs=rngs)

    def __call__(self, x: jax.Array, h: jax.Array) -> jax.Array:
        parts = self.norm(self.lin(jnp.concatenate([x, h], -1)))
        reset, cand, update = jnp.split(parts, 3, -1)
        reset = jax.nn.sigmoid(reset)
        cand = jnp.tanh(reset * cand)
        update = jax.nn.sigmoid(update - 1.0)
        return update * cand + (1 - update) * h


class MLP(nnx.Module):
    def __init__(self, inp: int, out: int, rngs: nnx.Rngs):
        self.a = nnx.Linear(inp, HIDDEN, rngs=rngs)
        self.n = nnx.LayerNorm(HIDDEN, rngs=rngs)
        self.b = nnx.Linear(HIDDEN, out, rngs=rngs)

    def __call__(self, x):
        return self.b(_act(self.n(self.a(x))))


class Step(NamedTuple):
    h: jax.Array  # (..., DETER)
    z: jax.Array  # (..., STOCH*CLASSES) one-hot (straight-through)
    prior: jax.Array  # (..., STOCH, CLASSES) logits
    post: jax.Array  # (..., STOCH, CLASSES) logits (= prior when imagining)


def _probs(logits):
    p = jax.nn.softmax(logits, -1)
    return (1 - UNIMIX) * p + UNIMIX / CLASSES


def _sample(logits, key):
    p = _probs(logits)
    idx = jax.random.categorical(key, jnp.log(p), -1)
    onehot = jax.nn.one_hot(idx, CLASSES)
    st = onehot + p - jax.lax.stop_gradient(p)
    return st.reshape(st.shape[:-2] + (STOCH * CLASSES,))


def kl(post_logits, prior_logits):
    """Sum over the STOCH variables of KL(post || prior), categorical."""
    p, q = _probs(post_logits), _probs(prior_logits)
    return jnp.sum(p * (jnp.log(p) - jnp.log(q)), axis=(-2, -1))


class WorldModel(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.encoder = Encoder(rngs)
        self.decoder = Decoder(rngs)
        self.img_in = MLP(STOCH * CLASSES + N_ACTIONS, HIDDEN, rngs)
        self.gru = GRU(HIDDEN, DETER, rngs)
        self.prior_net = MLP(DETER, STOCH * CLASSES, rngs)
        self.post_net = MLP(DETER + EMBED, STOCH * CLASSES, rngs)

    # --- one step -----------------------------------------------------------------------------
    def transition(self, h, z, a_onehot):
        return self.gru(_act(self.img_in(jnp.concatenate([z, a_onehot], -1))), h)

    def prior(self, h):
        return self.prior_net(h).reshape(h.shape[:-1] + (STOCH, CLASSES))

    def posterior(self, h, embed):
        return self.post_net(jnp.concatenate([h, embed], -1)).reshape(h.shape[:-1] + (STOCH, CLASSES))

    def initial(self, batch: int):
        return jnp.zeros((batch, DETER)), jnp.zeros((batch, STOCH * CLASSES))

    # --- sequences ----------------------------------------------------------------------------
    def observe(self, views: jax.Array, actions: jax.Array, key: jax.Array) -> Step:
        """views (B, T, 64, 256) palette indices, actions (B, T) (taken after each view).
        Returns the posterior rollout (time-major fields are (B, T, ...))."""
        B, T = views.shape[:2]
        embed = self.encoder(jax.nn.one_hot(views, N_COLOURS))
        a_prev = jnp.concatenate([jnp.zeros((B, 1, N_ACTIONS)), jax.nn.one_hot(actions[:, :-1], N_ACTIONS)], 1)
        h0, z0 = self.initial(B)

        def body(carry, x):
            h, z = carry
            e, a, k = x
            h = self.transition(h, z, a)
            prior = self.prior(h)
            post = self.posterior(h, e)
            z = _sample(post, k)
            return (h, z), Step(h, z, prior, post)

        keys = jax.random.split(key, T)
        _, out = jax.lax.scan(body, (h0, z0), (embed.swapaxes(0, 1), a_prev.swapaxes(0, 1), keys))
        return jax.tree.map(lambda x: x.swapaxes(0, 1), out)

    def imagine(self, h, z, actions: jax.Array, key: jax.Array) -> Step:
        """From (h, z) after the last observed step, roll the prior forward through ``actions``
        (B, K) (the first is the action taken after the last observed view)."""
        K = actions.shape[1]

        def body(carry, x):
            h, z = carry
            a, k = x
            h = self.transition(h, z, a)
            prior = self.prior(h)
            z = _sample(prior, k)
            return (h, z), Step(h, z, prior, prior)

        keys = jax.random.split(key, K)
        _, out = jax.lax.scan(body, (h, z), (jax.nn.one_hot(actions, N_ACTIONS).swapaxes(0, 1), keys))
        return jax.tree.map(lambda x: x.swapaxes(0, 1), out)

    def decode(self, s: Step) -> jax.Array:
        return self.decoder(jnp.concatenate([s.h, s.z], -1))


def loss_fn(model: WorldModel, views: jax.Array, actions: jax.Array, key: jax.Array):
    s = model.observe(views, actions, key)
    logits = model.decode(s)
    ce = -jnp.take_along_axis(jax.nn.log_softmax(logits, -1), views[..., None], -1)[..., 0]
    recon = jnp.mean(jnp.sum(ce, axis=(-2, -1)))  # summed over the strip, mean over (B, T)
    sg = jax.lax.stop_gradient
    dyn = jnp.mean(jnp.maximum(kl(sg(s.post), s.prior), FREE_NATS))
    rep = jnp.mean(jnp.maximum(kl(s.post, sg(s.prior)), FREE_NATS))
    loss = recon + KL_DYN * dyn + KL_REP * rep
    acc = jnp.mean(jnp.argmax(logits, -1) == views)
    return loss, {"loss": loss, "recon": recon, "kl_dyn": dyn, "kl_rep": rep, "pixel_acc": acc}


def count_params(model: nnx.Module) -> int:
    return sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param)))
