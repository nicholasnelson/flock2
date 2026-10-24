"""A world model in the DreamerV3 style (Hafner et al. 2023, arXiv:2301.04104), for the sheep's
first-person strip (``flock.fpv``: 64 x 256, 16 palette colours) and their actions.

- Encoder: five stride-2 convolutions over the one-hot strip (64 x 256 -> 2 x 8), then a dense
  embedding.
- RSSM: a layer-normalised GRU deterministic state ``h`` and a stochastic state ``z`` of
  STOCH categorical variables with CLASSES classes each (1 % uniform mix, straight-through
  samples). The transition takes (z_{t-1}, a_{t-1}); the prior predicts z_t from h_t; the
  posterior also sees the embedding of the strip at t.
- Decoder: the mirror of the encoder, logits over the 16 colours per pixel.
- Interoception (optional, ``n_intero`` > 0): the sheep's drive levels enter as a second input
  added to the strip's embedding, and a head reconstructs them from (h, z) (Milestone 1, step 3).
- Reward head (optional, ``reward=True``): as in DreamerV3, the reward is predicted from (h, z)
  as a categorical over 255 bins spaced evenly in symlog space on [-20, 20], trained by
  cross-entropy against the two-hot encoding of symlog(reward); the prediction is the symexp of
  the expected bin. The state at t predicts the reward of the step that led to it. (Computing
  the reward from the predicted levels through the drive formula was tried first: over a
  15-step horizon it correlated only 0.28 with the true reward and was strongly biased, since
  small level errors swamp the per-step drive changes.)
- Curiosity ensemble (optional, ``ensemble`` > 0): heads that each predict, from the state (h, z)
  and the action taken, the distribution of the next stochastic state z' (logits over its 32 x 16
  classes). Each head has a shared hidden layer and one output branch per action (rather than the
  action concatenated to the input as in Plan2Explore: with 9 action inputs beside 896 state
  inputs an action never tried got no more disagreement than a familiar one — tests/test_wm.py);
  an untried branch keeps each head's own random weights, so the heads disagree until it is
  trained. Heads are trained by cross-entropy against the posterior the model actually reached, on
  stop-gradient inputs (they measure the model, they do not shape it). Their disagreement — the
  variance across heads of the predicted class probabilities — is high where the model lacks
  experience and low where it has it, including for genuinely random outcomes, which every head
  learns to predict as the same spread. It is the curiosity reward of Plan2Explore (Sekar et al.
  2020, arXiv:2005.05960), here over the categorical latent instead of the embedding.
  ``ensemble_target="body"`` (the default since birth1) instead predicts the change in the bodily
  stores among the levels (``BODY_DIMS``: energy, hydration, warmth, interest — not social, which
  is an instantaneous count of the sheep in view and so changes with every turn of the head),
  (levels(t+1) - levels(t)) x 100, by squared error, and disagreement is the variance of
  those predictions: curiosity about what an action does to the sheep's body, not about what it
  will see. With the visual target (birth1) turning — which reveals new parts of the panorama —
  drew the most disagreement and the policy learned to spin in place.
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
REWARD_BINS = 255
REWARD_WEIGHT = 1.0
ENSEMBLE_WEIGHT = 1.0
BODY_SCALE = 100.0  # the body ensemble predicts level changes x this
BODY_DIMS = (0, 1, 2, 4)  # in flock.meadow.drives order: energy, hydration, warmth, (social,) interest
INTERO_WEIGHT = 1000.0  # on the summed squared error of the levels (the strip's CE sums 16,384 pixels)
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


class EnsembleHead(nnx.Module):
    """A prediction from the state, one output branch per action (see the module docstring)."""

    def __init__(self, rngs: nnx.Rngs, out: int, hidden: int = 256):
        self.out = out
        self.a = nnx.Linear(DETER + STOCH * CLASSES, hidden, rngs=rngs)
        self.n = nnx.LayerNorm(hidden, rngs=rngs)
        self.b = nnx.Linear(hidden, N_ACTIONS * out, rngs=rngs)

    def __call__(self, feat: jax.Array, a_onehot: jax.Array) -> jax.Array:
        y = self.b(_act(self.n(self.a(feat)))).reshape(feat.shape[:-1] + (N_ACTIONS, self.out))
        return jnp.einsum("...a,...ao->...o", a_onehot, y)


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


def symlog(x):
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x):
    return jnp.sign(x) * jnp.expm1(jnp.abs(x))


BINS = jnp.linspace(-20.0, 20.0, REWARD_BINS)


def twohot(y: jax.Array) -> jax.Array:
    """(...) values already in symlog space -> (..., REWARD_BINS) two-hot weights."""
    y = jnp.clip(y, BINS[0], BINS[-1])
    hi = jnp.clip(jnp.searchsorted(BINS, y, side="right"), 1, REWARD_BINS - 1)
    lo = hi - 1
    w_hi = (y - BINS[lo]) / (BINS[hi] - BINS[lo])
    return jax.nn.one_hot(lo, REWARD_BINS) * (1 - w_hi)[..., None] + jax.nn.one_hot(hi, REWARD_BINS) * w_hi[..., None]


def kl(post_logits, prior_logits):
    """Sum over the STOCH variables of KL(post || prior), categorical."""
    p, q = _probs(post_logits), _probs(prior_logits)
    return jnp.sum(p * (jnp.log(p) - jnp.log(q)), axis=(-2, -1))


class WorldModel(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, n_intero: int = 0, reward: bool = False, ensemble: int = 0, ensemble_target: str = "body"):
        self.n_intero = n_intero
        self.encoder = Encoder(rngs)
        self.decoder = Decoder(rngs)
        self.intero_in = nnx.Linear(n_intero, EMBED, rngs=rngs) if n_intero else None
        self.intero_head = MLP(DETER + STOCH * CLASSES, n_intero, rngs) if n_intero else None
        self.reward_head = MLP(DETER + STOCH * CLASSES, REWARD_BINS, rngs) if reward else None
        if reward:  # zero-initialised output, as in DreamerV3
            self.reward_head.b.kernel[...] = jnp.zeros_like(self.reward_head.b.kernel[...])
        assert ensemble_target in ("body", "z") and (ensemble_target == "z" or not ensemble or n_intero)
        self.ensemble_target = ensemble_target
        out = len(BODY_DIMS) if ensemble_target == "body" else STOCH * CLASSES
        self.ensemble = nnx.List([EnsembleHead(rngs, out) for _ in range(ensemble)]) if ensemble else None
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
    def observe(self, views: jax.Array, actions: jax.Array, key: jax.Array, intero: jax.Array | None = None) -> Step:
        """views (B, T, 64, 256) palette indices, actions (B, T) (taken after each view),
        intero (B, T, n_intero) the drive levels felt with each view (if the model has them).
        Returns the posterior rollout (time-major fields are (B, T, ...))."""
        B, T = views.shape[:2]
        embed = self.encoder(jax.nn.one_hot(views, N_COLOURS))
        if self.intero_in is not None:
            embed = embed + self.intero_in(intero)
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

    def decode_intero(self, s: Step) -> jax.Array:
        return self.intero_head(jnp.concatenate([s.h, s.z], -1))

    def ensemble_predict(self, feat: jax.Array, a_onehot: jax.Array) -> jax.Array:
        """(..., FEAT), (..., N_ACTIONS) -> (K, ..., len(BODY_DIMS)) predicted store changes x BODY_SCALE,
        or (K, ..., STOCH, CLASSES) next-z logits."""
        y = jnp.stack([e(feat, a_onehot) for e in self.ensemble])
        return y if self.ensemble_target == "body" else y.reshape(y.shape[:-1] + (STOCH, CLASSES))

    def ensemble_logits(self, feat: jax.Array, a_onehot: jax.Array) -> jax.Array:
        return self.ensemble_predict(feat, a_onehot)

    def disagreement(self, feat: jax.Array, a_onehot: jax.Array) -> jax.Array:
        """(...) the ensemble's variance across heads: of the predicted level changes, averaged over
        the drives (body); or of the predicted class probabilities, summed over classes and
        averaged over the STOCH variables (z)."""
        y = self.ensemble_predict(feat, a_onehot)
        if self.ensemble_target == "body":
            return jnp.mean(jnp.var(y, axis=0), -1)
        return jnp.mean(jnp.sum(jnp.var(_probs(y), axis=0), -1), -1)

    def reward(self, feat: jax.Array) -> jax.Array:
        """feat (..., DETER + STOCH*CLASSES) -> predicted reward of the step that led there."""
        return symexp(jnp.sum(jax.nn.softmax(self.reward_head(feat), -1) * BINS, -1))


def loss_fn(model: WorldModel, views: jax.Array, actions: jax.Array, key: jax.Array, intero: jax.Array | None = None, reward: jax.Array | None = None):
    """``reward`` (B, T): the reward of the step taken after each view (as logged)."""
    s = model.observe(views, actions, key, intero)
    logits = model.decode(s)
    ce = -jnp.take_along_axis(jax.nn.log_softmax(logits, -1), views[..., None], -1)[..., 0]
    recon = jnp.mean(jnp.sum(ce, axis=(-2, -1)))  # summed over the strip, mean over (B, T)
    sg = jax.lax.stop_gradient
    dyn = jnp.mean(jnp.maximum(kl(sg(s.post), s.prior), FREE_NATS))
    rep = jnp.mean(jnp.maximum(kl(s.post, sg(s.prior)), FREE_NATS))
    loss = recon + KL_DYN * dyn + KL_REP * rep
    acc = jnp.mean(jnp.argmax(logits, -1) == views)
    aux = {"recon": recon, "kl_dyn": dyn, "kl_rep": rep, "pixel_acc": acc}
    if model.intero_head is not None:
        err = jnp.mean(jnp.sum((model.decode_intero(s) - intero) ** 2, -1))
        loss = loss + INTERO_WEIGHT * err
        aux["intero_mse"] = err
    if model.reward_head is not None:
        r_logits = model.reward_head(jnp.concatenate([s.h, s.z], -1))[:, 1:]  # state t+1 <- reward of step t
        target = twohot(symlog(reward[:, :-1]))
        r_ce = -jnp.mean(jnp.sum(target * jax.nn.log_softmax(r_logits, -1), -1))
        loss = loss + REWARD_WEIGHT * r_ce
        aux["reward_ce"] = r_ce
    if model.ensemble is not None:
        sg = jax.lax.stop_gradient
        feat = sg(jnp.concatenate([s.h, s.z], -1))[:, :-1]
        a = jax.nn.one_hot(actions[:, :-1], N_ACTIONS)
        if model.ensemble_target == "body":
            body = intero[..., jnp.array(BODY_DIMS)]
            target = BODY_SCALE * (body[:, 1:] - body[:, :-1])
            e_loss = jnp.mean(jnp.sum((model.ensemble_predict(feat, a) - target[None]) ** 2, -1))
            aux["ensemble_mse"] = e_loss
        else:
            target = sg(_probs(s.post[:, 1:]))
            logp = jax.nn.log_softmax(model.ensemble_predict(feat, a), -1)
            e_loss = -jnp.mean(jnp.sum(target[None] * logp, -1).sum(-1))  # per head and step, summed over the 32 variables
            aux["ensemble_ce"] = e_loss
        loss = loss + ENSEMBLE_WEIGHT * e_loss
    aux["loss"] = loss
    return loss, aux


def load_into(model: WorldModel, pure: dict) -> None:
    """Load a checkpoint's parameters into ``model``; parts the checkpoint lacks (e.g. the
    interoception layers when fine-tuning a vision-only model) keep their fresh initialisation."""
    state = nnx.state(model)
    flat = dict(nnx.to_flat_state(state))
    src = {k: v for k, v in _flatten(pure)}
    for path in flat:
        key = tuple(str(p) for p in path)
        if key in src:
            flat[path] = flat[path].replace(jnp.asarray(src[key]))
    nnx.update(model, nnx.from_flat_state(flat))


def _flatten(d, prefix=()):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _flatten(v, prefix + (str(k),))
        else:
            yield prefix + (str(k),), v


def count_params(model: nnx.Module) -> int:
    return sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param)))
