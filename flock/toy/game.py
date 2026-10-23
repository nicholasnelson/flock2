"""The forward pass of the drawing game and its losses (shared by training and evaluation)."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx

from flock import world
from flock.nets import Agents
from flock.raster import CANVAS, Canvas
from flock.toy import env
from flock.world import VIEW_TILES


class Forward(NamedTuple):
    logits: jax.Array  # (B, 16, 16) receiver tile logits
    canvas: Canvas  # the sender's drawing
    sender_emb: jax.Array  # (B, 256)
    receiver_view: jax.Array  # (B, 64, 64, 16) the receiver's view with the canvas in it


def blank_canvas(batch: int, size: int = CANVAS) -> Canvas:
    return Canvas(
        colour=jnp.full((batch, size, size, 16), 1.0 / 16),
        alpha=jnp.zeros((batch, size, size)),
        ink=jnp.zeros((batch,)),
    )


def draw(model: Agents, sender_view_onehot: jax.Array, key: jax.Array | None, hard: bool) -> tuple[Canvas, jax.Array]:
    emb_s, _ = model.encoder(sender_view_onehot)
    if model.action_space == "pixels":
        canvas = model.sender(emb_s, hard_colour=hard, key=key)
    else:
        canvas = model.sender(emb_s, hard_colour=hard)
    return canvas, emb_s


def receive(model: Agents, receiver_base: jax.Array, canvas: Canvas, canvas_px: jax.Array, opaque: bool) -> tuple[jax.Array, jax.Array]:
    """Receiver tile logits given a canvas. Returns (logits, receiver view)."""
    rb = world.to_onehot(receiver_base)
    comp = jax.vmap(lambda v, c, a, p: world.composite_canvas(v, c, a, p, opaque_paper=opaque))
    if model.shared_encoder:
        rv = comp(rb, canvas.colour, canvas.alpha, canvas_px)
        emb_r, fmap = model.encoder(rv)
        context = emb_r
    else:
        # The perception encoder sees the view with the paper (or ground) but no ink; a
        # separate CNN reads the canvas. Same information, different pathway.
        empty = blank_canvas(rb.shape[0], canvas.alpha.shape[1])
        rv = comp(rb, empty.colour, empty.alpha, canvas_px)
        emb_r, fmap = model.encoder(rv)
        context = emb_r + model.message_encoder(canvas)
    cell = rb.shape[1] // VIEW_TILES  # px per receiver cell: 4 at a 64 view, 8 at 128
    logits = model.receiver(fmap, context, canvas_px // cell)
    return logits, rv


CLUTTER_COLOUR = 15  # the outline/ink colour: dirt looks like a mark
CLUTTER_MAX_LEN = 4.0  # px
LOWRES = 4  # block size of the reduced-resolution reading (32 -> 8 px)


def _clutter(key: jax.Array, b: int, s: int, n: int) -> jax.Array:
    """(b, s, s) alpha of ``n`` random specks and short dashes (0-4 px long, 2 px wide) per frame."""
    from flock.raster import Strokes, rasterise

    k1, k2, k3 = jax.random.split(key, 3)
    p0 = jax.random.uniform(k1, (b, n, 2), minval=2.0, maxval=s - 2.0)
    ang = jax.random.uniform(k2, (b, n), maxval=2 * jnp.pi)
    ln = jax.random.uniform(k3, (b, n), maxval=CLUTTER_MAX_LEN)
    p1 = jnp.clip(p0 + ln[..., None] * jnp.stack([jnp.cos(ang), jnp.sin(ang)], -1), 0.5, s - 0.5)
    logit = lambda x: jnp.log(x / s) - jnp.log1p(-x / s)
    ends = jnp.concatenate([logit(p0), logit(p1)], -1)  # (b, n, 4) x0, y0, x1, y1 in logit space
    one = lambda e: rasterise(Strokes(e, jnp.zeros(n), jnp.zeros((n, 16)), jnp.full((n,), 20.0)), size=s, fixed_width=2.0).alpha
    return jax.vmap(one)(ends)


def transmit(canvas: Canvas, key: jax.Array, jitter: int = 0, centre: bool = False, rotate: bool = False, clutter: int = 0, lowres_p: float = 0.0) -> Canvas:
    """What the drawing looks like on arrival. ``centre``: moved so its ink centroid is at the
    middle of the frame (where a mark sits carries no information); then moved by up to
    ``jitter`` px; ``rotate``: turned by a random multiple of 90 degrees (a ground mark can
    be read from any side); ``lowres_p``: with this probability read at 1/LOWRES resolution
    (block-averaged, as if from further away), so the smallest marks fade; ``clutter``: this
    many random specks and short dashes in the ink colour are added, like dirt on the ground,
    so a dot cannot be told from noise. Ink is unchanged."""
    if not (centre or rotate or clutter > 0 or lowres_p > 0) and jitter <= 0:
        return canvas
    b, s = canvas.alpha.shape[0], canvas.alpha.shape[1]
    k1, k2, k3, k4 = jax.random.split(key, 4)
    d = jax.random.randint(k1, (b, 2), -jitter, jitter + 1) if jitter > 0 else jnp.zeros((b, 2), jnp.int32)
    if centre:
        a = jax.lax.stop_gradient(canvas.alpha)
        idx = jnp.arange(s) + 0.5
        m = jnp.sum(a, axis=(1, 2))
        cy = jnp.sum(a * idx[None, :, None], axis=(1, 2)) / jnp.maximum(m, 1e-6)
        cx = jnp.sum(a * idx[None, None, :], axis=(1, 2)) / jnp.maximum(m, 1e-6)
        c = jnp.stack([cy, cx], -1)
        c = jnp.where(m[:, None] > 1e-3, c, s / 2)  # a blank drawing stays put
        d = d + jnp.round(s / 2 - c).astype(jnp.int32)
    P = s  # enough padding for any shift that keeps some ink in frame
    d = jnp.clip(d, -P, P)
    n = canvas.colour.shape[-1]

    def move(col, al, dd):
        colp = jnp.pad(col, ((P, P), (P, P), (0, 0)), constant_values=1.0 / n)
        ap = jnp.pad(al, P)
        r, cc = P - dd[0], P - dd[1]
        return jax.lax.dynamic_slice(colp, (r, cc, 0), (s, s, n)), jax.lax.dynamic_slice(ap, (r, cc), (s, s))

    col, al = jax.vmap(move)(canvas.colour, canvas.alpha, d)
    if rotate:
        k = jax.random.randint(k2, (b,), 0, 4)
        rot = lambda x, kk: jax.lax.switch(kk, [lambda v: v, lambda v: jnp.rot90(v, 1), lambda v: jnp.rot90(v, 2), lambda v: jnp.rot90(v, 3)], x)
        col, al = jax.vmap(rot)(col, k), jax.vmap(rot)(al, k)
    if lowres_p > 0:
        f = LOWRES
        blk = al.reshape(b, s // f, f, s // f, f).mean((2, 4))
        lo = jnp.repeat(jnp.repeat(blk, f, axis=1), f, axis=2)
        use = jax.random.bernoulli(k3, lowres_p, (b,))[:, None, None]
        al = jnp.where(use, lo, al)
    if clutter > 0:
        ca = jax.lax.stop_gradient(_clutter(k4, b, s, clutter))
        cc = jax.nn.one_hot(CLUTTER_COLOUR, n)
        new = al + ca * (1.0 - al)
        col = (col * al[..., None] + cc * (ca * (1.0 - al))[..., None]) / jnp.maximum(new[..., None], 1e-6)
        col = jnp.where(new[..., None] > 1e-6, col, 1.0 / n)
        al = new
    return Canvas(colour=col, alpha=al, ink=canvas.ink)


def channel_args(cfg) -> dict:
    """How a drawing is transformed on its way to the reader (and the judge), from a config."""
    return dict(
        jitter=getattr(cfg, "jitter_px", 0),
        centre=getattr(cfg, "centre_glyph", False),
        rotate=getattr(cfg, "rotate_stamps", False) and cfg.mode == "stamp",
        clutter=getattr(cfg, "clutter", 0),
        lowres_p=getattr(cfg, "lowres_p", 0.0),
    )


def forward(model: Agents, ep: env.Episode, key: jax.Array | None, opaque: bool, hard: bool = False, jitter: int = 0, centre: bool = False, rotate: bool = False, clutter: int = 0, lowres_p: float = 0.0) -> Forward:
    """Sender draws, receiver reads the drawing as it arrives through ``transmit`` (centring,
    jitter, rotation); ``Forward.canvas`` is the drawing as drawn."""
    sv = world.to_onehot(ep.sender_view)
    canvas, emb_s = draw(model, sv, key, hard)
    shown = transmit(canvas, jax.random.fold_in(key, 7), jitter, centre, rotate, clutter, lowres_p) if (jitter > 0 or centre or rotate or clutter > 0 or lowres_p > 0) else canvas
    logits, rv = receive(model, ep.receiver_base, shown, ep.canvas_px, opaque)
    return Forward(logits=logits, canvas=canvas, sender_emb=emb_s, receiver_view=rv)


def tile_log_probs(logits: jax.Array) -> jax.Array:
    b = logits.shape[0]
    return jax.nn.log_softmax(logits.reshape(b, -1), axis=-1).reshape(b, VIEW_TILES, VIEW_TILES)


def success_loss(logits: jax.Array, target_mask: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Cross-entropy of landing on the target footprint, and per-episode accuracy."""
    logp = tile_log_probs(logits)
    masked = jnp.where(target_mask, logp, -1e9)
    ce = -jax.nn.logsumexp(masked.reshape(logp.shape[0], -1), axis=-1)
    b = logits.shape[0]
    best = jnp.argmax(logits.reshape(b, -1), axis=-1)
    acc = target_mask.reshape(b, -1)[jnp.arange(b), best]
    return ce, acc


def perceptual_loss(judge, canvas: Canvas, sender_view_onehot: jax.Array, ground_views: jax.Array, probe_px: jax.Array) -> jax.Array:
    """Distance, in a *frozen* perception encoder's embedding space, between the drawing
    stamped on plain ground where the referent was and the sender's actual view of the referent.

    The judge must be frozen (it is the pre-trained perception encoder, before any game
    training). Measuring resemblance through a trainable encoder lets the encoder minimise the
    loss by learning to ignore the canvas, and the drawing collapses to nothing — observed in
    the first fine-tuned-encoder run.
    """
    probe_px = jnp.broadcast_to(probe_px, (canvas.alpha.shape[0], 2))  # one position, or one per episode
    probe = jax.vmap(lambda v, c, a, p: world.composite_canvas(v, c, a, p, opaque_paper=False))(ground_views, canvas.colour, canvas.alpha, probe_px)
    emb_p, _ = judge(probe)
    target, _ = judge(sender_view_onehot)
    target = jax.lax.stop_gradient(target)
    n = lambda x: x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)
    return jnp.mean(jnp.sum((n(emb_p) - n(target)) ** 2, axis=-1))


def listening_loss(logits_with: jax.Array, logits_without: jax.Array) -> jax.Array:
    """Negative L1 distance between the receiver's tile distributions with and without the
    message (Eccles et al.'s positive-listening bias)."""
    b = logits_with.shape[0]
    p1 = jax.nn.softmax(logits_with.reshape(b, -1), axis=-1)
    p0 = jax.nn.softmax(jax.lax.stop_gradient(logits_without).reshape(b, -1), axis=-1)
    return -jnp.mean(jnp.sum(jnp.abs(p1 - p0), axis=-1))


def signalling_loss(canvas: Canvas, target_type: jax.Array, margin: float = 0.15) -> jax.Array:
    """Repulsion between drawings of different referents (Villanger & Alexandersen's continuous
    positive-signalling bias): penalise pairs whose canvases are closer than ``margin``."""
    b = canvas.alpha.shape[0]
    x = (canvas.colour * canvas.alpha[..., None]).reshape(b, -1)
    d = jnp.mean(jnp.abs(x[:, None, :] - x[None, :, :]), axis=-1)  # (B, B)
    diff = target_type[:, None] != target_type[None, :]
    pen = jnp.maximum(margin - d, 0.0) * diff
    return jnp.sum(pen) / jnp.maximum(jnp.sum(diff), 1.0)
