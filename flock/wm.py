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
- Turn-rotated memory (optional, ``rot_bins`` > 0; rung 4's minimal piece, meadow v2 spec § 18): a
  ring of ``rot_bins`` x ``rot_ch`` numbers carried beside the GRU state in h. Each transition first
  writes to it from (z, GRU state) through a learned gate, then **rolls it by ``rot_shift`` bins
  when the action taken was a turn** (one way for ``rot_left``, the other for ``rot_right``). The
  roll is the only built-in geometry: what is written where, and how it is read (by the GRU's
  input, and by everything that reads h), is learned. With ``rot_bins`` = 0 the model is unchanged
  and h is the GRU state alone.

Sizes are chosen to land in the plan's 2-8M parameters; ``count_params`` reports the real number.
They live in a ``Config``: ``MEADOW`` (the default, the meadow's first-person strip) and any other,
e.g. the grid world's (``flock.grid``), so one implementation serves both.
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


class Config(NamedTuple):
    """Sizes of a world model. Observations are (h, w) grids of class indices (n_classes), seen
    one-hot, or with ``rgb`` (h, w, 3) uint8 colour; the encoder halves (h, w) at each stage down
    to ``bottom``."""

    h: int = H
    w: int = W
    n_classes: int = N_COLOURS
    n_actions: int = N_ACTIONS
    enc_ch: tuple = ENC_CH
    bottom: tuple = BOTTOM
    deter: int = DETER
    hidden: int = HIDDEN
    stoch: int = STOCH
    classes: int = CLASSES
    embed: int = EMBED
    body_dims: tuple = BODY_DIMS  # which felt numbers the body-curiosity ensemble predicts
    intero_weight: float = INTERO_WEIGHT
    recon_weight: float = 1.0  # loss rebalancing (third review, S3): <1 lets reward and felt levels count more
    reward_weight: float = REWARD_WEIGHT
    recon_balance: float = 0.0  # >0: weight each pixel by its class's inverse frequency in the batch, capped at this (third review, S5)
    rgb: bool = False  # meadow v2: views are (h, w, 3) uint8 colour, not class indices; n_classes must be 3, and
    # the strip is reconstructed by summed squared error on colour in [0, 1] (as DreamerV3 does for images)
    kl_dyn: float = KL_DYN  # weight of the dynamics KL; 0.5 here historically; DreamerV3 uses 1 (plan review, 2026-10-26)
    ms_weight: float = 0.0  # >0 (RGB only): multi-step prediction; from starts every ms_stride steps, the prior is rolled
    ms_horizons: tuple = (2, 4, 8)  # forward with the actions taken and the view is decoded at these horizons and
    ms_stride: int = 8  # compared with the real one (summed squared error, averaged), weighted by ms_weight
    rot_bins: int = 0  # >0: the turn-rotated memory (module docstring), a ring of rot_bins x rot_ch numbers in h
    rot_ch: int = 16
    rot_shift: int = 1  # bins the ring rolls per turn action
    rot_left: int = -1  # the turn actions' indices
    rot_right: int = -1

    @property
    def state(self) -> int:
        """Width of h: the GRU state, plus the ring when there is one."""
        return self.deter + self.rot_bins * self.rot_ch

    @property
    def feat(self) -> int:
        return self.state + self.stoch * self.classes


MEADOW = Config()


def _act(x):
    return jax.nn.silu(x)


class Encoder(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, cfg: Config = MEADOW):
        ch = (cfg.n_classes,) + cfg.enc_ch
        self.embed = cfg.embed
        self.convs = nnx.List([nnx.Conv(ch[i], ch[i + 1], (4, 4), strides=2, padding="SAME", rngs=rngs) for i in range(len(cfg.enc_ch))])
        self.norms = nnx.List([nnx.LayerNorm(c, rngs=rngs) for c in cfg.enc_ch])
        self.out = nnx.Linear(cfg.bottom[0] * cfg.bottom[1] * cfg.enc_ch[-1], cfg.embed, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """x: (..., h, w, n_classes) one-hot -> (..., embed)."""
        lead = x.shape[:-3]
        h = x.reshape((-1,) + x.shape[-3:])
        for conv, norm in zip(self.convs, self.norms):
            h = _act(norm(conv(h)))
        return self.out(h.reshape(h.shape[0], -1)).reshape(lead + (self.embed,))


class Decoder(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, cfg: Config = MEADOW):
        self.cfg = cfg
        self.inp = nnx.Linear(cfg.feat, cfg.bottom[0] * cfg.bottom[1] * cfg.enc_ch[-1], rngs=rngs)
        ch = cfg.enc_ch[::-1] + (cfg.n_classes,)
        self.deconvs = nnx.List([nnx.ConvTranspose(ch[i], ch[i + 1], (4, 4), strides=(2, 2), padding="SAME", rngs=rngs) for i in range(len(cfg.enc_ch))])
        self.norms = nnx.List([nnx.LayerNorm(c, rngs=rngs) for c in ch[1:-1]])

    def __call__(self, feat: jax.Array) -> jax.Array:
        """feat: (..., feat) -> logits (..., h, w, n_classes)."""
        c = self.cfg
        lead = feat.shape[:-1]
        h = self.inp(feat.reshape(-1, feat.shape[-1])).reshape((-1,) + tuple(c.bottom) + (c.enc_ch[-1],))
        for i, dc in enumerate(self.deconvs):
            h = dc(h)
            if i < len(self.norms):
                h = _act(self.norms[i](h))
        return h.reshape(lead + (c.h, c.w, c.n_classes))


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
    def __init__(self, inp: int, out: int, rngs: nnx.Rngs, hidden: int = HIDDEN):
        self.a = nnx.Linear(inp, hidden, rngs=rngs)
        self.n = nnx.LayerNorm(hidden, rngs=rngs)
        self.b = nnx.Linear(hidden, out, rngs=rngs)

    def __call__(self, x):
        return self.b(_act(self.n(self.a(x))))


class EnsembleHead(nnx.Module):
    """A prediction from the state, one output branch per action (see the module docstring)."""

    def __init__(self, rngs: nnx.Rngs, out: int, hidden: int = 256, cfg: Config = MEADOW):
        self.out = out
        self.n_actions = cfg.n_actions
        self.a = nnx.Linear(cfg.feat, hidden, rngs=rngs)
        self.n = nnx.LayerNorm(hidden, rngs=rngs)
        self.b = nnx.Linear(hidden, cfg.n_actions * out, rngs=rngs)

    def __call__(self, feat: jax.Array, a_onehot: jax.Array) -> jax.Array:
        y = self.b(_act(self.n(self.a(feat)))).reshape(feat.shape[:-1] + (self.n_actions, self.out))
        return jnp.einsum("...a,...ao->...o", a_onehot, y)


class Step(NamedTuple):
    h: jax.Array  # (..., cfg.state): DETER, plus the ring if there is one
    z: jax.Array  # (..., STOCH*CLASSES) one-hot (straight-through)
    prior: jax.Array  # (..., STOCH, CLASSES) logits
    post: jax.Array  # (..., STOCH, CLASSES) logits (= prior when imagining)


def rotate_memory(m: jax.Array, a_onehot: jax.Array, cfg: Config) -> jax.Array:
    """(..., rot_bins, rot_ch) -> the ring rolled by rot_shift bins for a left turn, back for a right."""
    left, right = a_onehot[..., cfg.rot_left, None, None], a_onehot[..., cfg.rot_right, None, None]
    return left * jnp.roll(m, cfg.rot_shift, -2) + right * jnp.roll(m, -cfg.rot_shift, -2) + (1 - left - right) * m


def _probs(logits):
    p = jax.nn.softmax(logits, -1)
    return (1 - UNIMIX) * p + UNIMIX / logits.shape[-1]


def _sample(logits, key):
    """(..., stoch, classes) logits -> (..., stoch * classes) straight-through one-hot sample."""
    p = _probs(logits)
    idx = jax.random.categorical(key, jnp.log(p), -1)
    onehot = jax.nn.one_hot(idx, logits.shape[-1])
    st = onehot + p - jax.lax.stop_gradient(p)
    return st.reshape(st.shape[:-2] + (st.shape[-2] * st.shape[-1],))


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
    def __init__(self, rngs: nnx.Rngs, n_intero: int = 0, reward: bool = False, ensemble: int = 0, ensemble_target: str = "body", cfg: Config = MEADOW):
        self.cfg = cfg
        c = cfg
        self.n_intero = n_intero
        self.encoder = Encoder(rngs, c)
        self.decoder = Decoder(rngs, c)
        self.intero_in = nnx.Linear(n_intero, c.embed, rngs=rngs) if n_intero else None
        self.intero_head = MLP(c.feat, n_intero, rngs, c.hidden) if n_intero else None
        self.reward_head = MLP(c.feat, REWARD_BINS, rngs, c.hidden) if reward else None
        if reward:  # zero-initialised output, as in DreamerV3
            self.reward_head.b.kernel[...] = jnp.zeros_like(self.reward_head.b.kernel[...])
        assert ensemble_target in ("body", "z") and (ensemble_target == "z" or not ensemble or n_intero)
        self.ensemble_target = ensemble_target
        out = len(c.body_dims) if ensemble_target == "body" else c.stoch * c.classes
        self.ensemble = nnx.List([EnsembleHead(rngs, out, cfg=c) for _ in range(ensemble)]) if ensemble else None
        n_mem = c.rot_bins * c.rot_ch
        self.mem_write = nnx.Linear(c.stoch * c.classes + c.deter, 2 * n_mem, rngs=rngs) if n_mem else None
        self.img_in = MLP(c.stoch * c.classes + c.n_actions + n_mem, c.hidden, rngs, c.hidden)
        self.gru = GRU(c.hidden, c.deter, rngs)
        self.prior_net = MLP(c.state, c.stoch * c.classes, rngs, c.hidden)
        self.post_net = MLP(c.state + c.embed, c.stoch * c.classes, rngs, c.hidden)

    # --- one step -----------------------------------------------------------------------------
    def transition(self, h, z, a_onehot):
        c = self.cfg
        if not c.rot_bins:
            return self.gru(_act(self.img_in(jnp.concatenate([z, a_onehot], -1))), h)
        g, m = h[..., : c.deter], h[..., c.deter :].reshape(h.shape[:-1] + (c.rot_bins, c.rot_ch))
        cand, gate = jnp.split(self.mem_write(jnp.concatenate([z, g], -1)).reshape(m.shape[:-1] + (2 * c.rot_ch,)), 2, -1)
        gate = jax.nn.sigmoid(gate - 1.0)  # biased toward keeping, as the GRU's update gate
        m = rotate_memory(gate * jnp.tanh(cand) + (1 - gate) * m, a_onehot, c).reshape(h.shape[:-1] + (-1,))
        g = self.gru(_act(self.img_in(jnp.concatenate([z, a_onehot, m], -1))), g)
        return jnp.concatenate([g, m], -1)

    def prior(self, h):
        return self.prior_net(h).reshape(h.shape[:-1] + (self.cfg.stoch, self.cfg.classes))

    def posterior(self, h, embed):
        return self.post_net(jnp.concatenate([h, embed], -1)).reshape(h.shape[:-1] + (self.cfg.stoch, self.cfg.classes))

    def obs_input(self, views: jax.Array) -> jax.Array:
        """The encoder's input for views: one-hot classes, or colour centred on 0 (``cfg.rgb``)."""
        if self.cfg.rgb:
            return views.astype(jnp.float32) / 255.0 - 0.5
        return jax.nn.one_hot(views, self.cfg.n_classes)

    def embed_obs(self, views: jax.Array, intero: jax.Array | None = None) -> jax.Array:
        e = self.encoder(self.obs_input(views))
        return e if self.intero_in is None else e + self.intero_in(intero)

    def initial(self, batch: int):
        return jnp.zeros((batch, self.cfg.state)), jnp.zeros((batch, self.cfg.stoch * self.cfg.classes))

    # --- sequences ----------------------------------------------------------------------------
    def observe(self, views: jax.Array, actions: jax.Array, key: jax.Array, intero: jax.Array | None = None, start=None) -> Step:
        """views (B, T, h, w) class indices, actions (B, T) (taken after each view),
        intero (B, T, n_intero) the drive levels felt with each view (if the model has them).
        ``start`` (h, z, a_prev): the state carried into the first step and the action taken before
        it, as recorded online (meadow v2 spec § 21.12, arm W); default a zero state and no action.
        Returns the posterior rollout (time-major fields are (B, T, ...))."""
        B, T = views.shape[:2]
        nA = self.cfg.n_actions
        embed = self.embed_obs(views, intero)
        if start is None:
            h0, z0 = self.initial(B)
            first = jnp.zeros((B, 1, nA))
        else:
            h0, z0, a0 = start
            first = jax.nn.one_hot(a0, nA)[:, None]
        a_prev = jnp.concatenate([first, jax.nn.one_hot(actions[:, :-1], nA)], 1)

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
        _, out = jax.lax.scan(body, (h, z), (jax.nn.one_hot(actions, self.cfg.n_actions).swapaxes(0, 1), keys))
        return jax.tree.map(lambda x: x.swapaxes(0, 1), out)

    def decode(self, s: Step) -> jax.Array:
        return self.decoder(jnp.concatenate([s.h, s.z], -1))

    def decode_intero(self, s: Step) -> jax.Array:
        return self.intero_head(jnp.concatenate([s.h, s.z], -1))

    def ensemble_predict(self, feat: jax.Array, a_onehot: jax.Array) -> jax.Array:
        """(..., FEAT), (..., N_ACTIONS) -> (K, ..., len(BODY_DIMS)) predicted store changes x BODY_SCALE,
        or (K, ..., STOCH, CLASSES) next-z logits."""
        y = jnp.stack([e(feat, a_onehot) for e in self.ensemble])
        return y if self.ensemble_target == "body" else y.reshape(y.shape[:-1] + (self.cfg.stoch, self.cfg.classes))

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


def loss_fn(model: WorldModel, views: jax.Array, actions: jax.Array, key: jax.Array, intero: jax.Array | None = None, reward: jax.Array | None = None, start=None):
    """``reward`` (B, T): the reward of the step taken after each view (as logged); ``start`` the
    recorded state before the first view (see ``observe``)."""
    s = model.observe(views, actions, key, intero, start)
    logits = model.decode(s)
    if model.cfg.rgb:
        target = views.astype(jnp.float32) / 255.0
        err = jnp.sum((logits + 0.5 - target) ** 2, -1)  # the decoder predicts colour centred on 0
        recon = jnp.mean(jnp.sum(err, axis=(-2, -1)))
        acc = jnp.mean(jnp.abs(logits + 0.5 - target))  # reported as "pixel_acc": here the mean absolute colour error
        loss, aux = _rest(model, s, recon, acc, actions, intero, reward)
        if getattr(model.cfg, "ms_weight", 0.0) > 0:
            ms = _multistep(model, s, views, actions, jax.random.fold_in(key, 1))
            loss = loss + model.cfg.ms_weight * ms
            aux = {**aux, "multistep": ms, "loss": loss}
        return loss, aux
    ce = -jnp.take_along_axis(jax.nn.log_softmax(logits, -1), views[..., None], -1)[..., 0]
    if model.cfg.recon_balance > 0:
        # class-balanced reconstruction: rare classes (water) count as much as common ones (sky),
        # weights capped and normalised so the mean pixel weight is 1 (the loss keeps its scale)
        counts = jnp.bincount(views.ravel(), length=model.cfg.n_classes).astype(jnp.float32)
        present = counts > 0
        w = jnp.where(present, jnp.minimum(model.cfg.recon_balance, counts.sum() / jnp.maximum(counts, 1.0) / present.sum()), 0.0)
        pix = jax.lax.stop_gradient(w[views])
        ce = ce * pix / jnp.mean(pix)
    recon = jnp.mean(jnp.sum(ce, axis=(-2, -1)))  # summed over the strip, mean over (B, T)
    acc = jnp.mean(jnp.argmax(logits, -1) == views)
    return _rest(model, s, recon, acc, actions, intero, reward)


def _multistep(model, s, views, actions, key):
    """Open-loop prediction of the view: from the posterior at each start, roll the prior forward
    with the actions taken, decode at each horizon, and compare with the real view (summed squared
    colour error per frame, averaged over starts and horizons)."""
    c = model.cfg
    T = views.shape[1]
    H = max(c.ms_horizons)
    starts = list(range(0, T - H, c.ms_stride))
    target = views.astype(jnp.float32) / 255.0
    errs = []
    for i, t0 in enumerate(starts):
        h, z = s.h[:, t0], s.z[:, t0]
        k = jax.random.fold_in(key, i)
        for step in range(1, H + 1):
            h = model.transition(h, z, jax.nn.one_hot(actions[:, t0 + step - 1], c.n_actions))
            z = _sample(model.prior(h), jax.random.fold_in(k, step))
            if step in c.ms_horizons:
                pred = model.decode(Step(h, z, None, None))
                errs.append(jnp.mean(jnp.sum((pred + 0.5 - target[:, t0 + step]) ** 2, axis=(-3, -2, -1))))
    return jnp.mean(jnp.stack(errs))


def _rest(model, s, recon, acc, actions, intero, reward):
    """The losses after reconstruction: KL, felt numbers, reward, ensemble."""
    sg = jax.lax.stop_gradient
    dyn = jnp.mean(jnp.maximum(kl(sg(s.post), s.prior), FREE_NATS))
    rep = jnp.mean(jnp.maximum(kl(s.post, sg(s.prior)), FREE_NATS))
    loss = model.cfg.recon_weight * recon + getattr(model.cfg, "kl_dyn", KL_DYN) * dyn + KL_REP * rep
    aux = {"recon": recon, "kl_dyn": dyn, "kl_rep": rep, "pixel_acc": acc}
    if model.intero_head is not None:
        err = jnp.mean(jnp.sum((model.decode_intero(s) - intero) ** 2, -1))
        loss = loss + model.cfg.intero_weight * err
        aux["intero_mse"] = err
    if model.reward_head is not None:
        r_logits = model.reward_head(jnp.concatenate([s.h, s.z], -1))[:, 1:]  # state t+1 <- reward of step t
        target = twohot(symlog(reward[:, :-1]))
        r_ce = -jnp.mean(jnp.sum(target * jax.nn.log_softmax(r_logits, -1), -1))
        loss = loss + model.cfg.reward_weight * r_ce
        aux["reward_ce"] = r_ce
    if model.ensemble is not None:
        sg = jax.lax.stop_gradient
        feat = sg(jnp.concatenate([s.h, s.z], -1))[:, :-1]
        a = jax.nn.one_hot(actions[:, :-1], model.cfg.n_actions)
        if model.ensemble_target == "body":
            body = intero[..., jnp.array(model.cfg.body_dims)]
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
