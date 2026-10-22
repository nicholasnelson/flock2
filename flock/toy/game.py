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


def blank_canvas(batch: int) -> Canvas:
    return Canvas(
        colour=jnp.full((batch, CANVAS, CANVAS, 16), 1.0 / 16),
        alpha=jnp.zeros((batch, CANVAS, CANVAS)),
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
        empty = blank_canvas(rb.shape[0])
        rv = comp(rb, empty.colour, empty.alpha, canvas_px)
        emb_r, fmap = model.encoder(rv)
        context = emb_r + model.message_encoder(canvas)
    logits = model.receiver(fmap, context, canvas_px // world.TILE)
    return logits, rv


def forward(model: Agents, ep: env.Episode, key: jax.Array | None, opaque: bool, hard: bool = False) -> Forward:
    sv = world.to_onehot(ep.sender_view)
    canvas, emb_s = draw(model, sv, key, hard)
    logits, rv = receive(model, ep.receiver_base, canvas, ep.canvas_px, opaque)
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


def perceptual_loss(model: Agents, canvas: Canvas, sender_emb: jax.Array, ground_views: jax.Array) -> jax.Array:
    """Distance, in the perception encoder's embedding space, between the drawing stamped on
    plain ground where the referent was and the sender's actual view of the referent."""
    probe = jax.vmap(lambda v, c, a: world.composite_canvas(v, c, a, env.PROBE_CANVAS_PX, opaque_paper=False))(ground_views, canvas.colour, canvas.alpha)
    emb_p, _ = model.encoder(probe)
    target = jax.lax.stop_gradient(sender_emb)
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
