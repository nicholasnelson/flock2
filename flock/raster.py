"""Differentiable stroke rasteriser for the 16x16, 16-colour drawing canvas.

A drawing is K line primitives. Each has two endpoints in [0, 1]^2, a width, a softmax over the
palette, and an on/off gate. Pixels are covered by a stroke in proportion to a sigmoid of the
distance to the segment (Mihai & Hare's distance-field relaxation). Strokes are composited in
order ("over"), so the output is a per-pixel colour distribution plus an alpha. Alpha lets the
same drawing be a bubble (opaque paper behind it) or a stamp (ground shows through).

Everything is plain JAX so gradients flow from a receiver's loss back to the stroke parameters.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from flock.palette import N_COLOURS

CANVAS = 16


class Strokes(NamedTuple):
    """Raw stroke parameters as produced by a network (all unconstrained reals)."""

    endpoints: jax.Array  # (K, 4) -> sigmoid -> (x0, y0, x1, y1) in [0, 1]
    width: jax.Array  # (K,)   -> softplus -> width in pixels
    colour: jax.Array  # (K, 16) logits over palette
    gate: jax.Array  # (K,)   -> sigmoid -> on/off


class Canvas(NamedTuple):
    colour: jax.Array  # (16, 16, 16) per-pixel distribution over the palette
    alpha: jax.Array  # (16, 16) coverage in [0, 1]
    ink: jax.Array  # scalar ink cost


def _pixel_centres(size: int) -> jax.Array:
    c = (jnp.arange(size) + 0.5) / size
    xs, ys = jnp.meshgrid(c, c, indexing="xy")
    return jnp.stack([xs, ys], axis=-1)  # (size, size, 2) in canvas units


def _segment_distance(p: jax.Array, a: jax.Array, b: jax.Array) -> jax.Array:
    """Distance from points p (..., 2) to segment a-b (2,)."""
    ab = b - a
    denom = jnp.maximum(jnp.sum(ab * ab), 1e-8)
    t = jnp.clip(jnp.sum((p - a) * ab, axis=-1) / denom, 0.0, 1.0)
    proj = a + t[..., None] * ab
    return jnp.linalg.norm(p - proj + 1e-9, axis=-1)


def rasterise(
    strokes: Strokes,
    size: int = CANVAS,
    edge_softness: float = 0.35,
    min_width: float = 0.6,
    temperature: float = 1.0,
    hard_colour: bool = False,
    fixed_width: float | None = None,
    stroke_cost: float = 0.0,
    hard_gate: bool = False,
    min_strokes: int = 0,
) -> Canvas:
    """Render strokes to a soft canvas.

    Widths and distances are in pixels. ``edge_softness`` is the sigmoid scale of the stroke
    edge; smaller is crisper (and harder to optimise). With ``fixed_width`` every stroke is a
    line of that width (the network's width output is ignored) and ink is length plus
    ``stroke_cost`` per stroke (a pen-lift price, so a zero-length dot is not free); otherwise
    width is free and ink is length + width. With ``hard_gate`` each stroke is fully on or off
    in the forward pass (straight-through gradient), so nothing can be said in faint,
    half-drawn strokes a viewer cannot see. ``min_strokes`` (with ``hard_gate``) forces the
    strokes the network most wants on to be on, so a drawing is never blank.
    """
    pts = _pixel_centres(size) * size  # pixel coordinates
    ends = jax.nn.sigmoid(strokes.endpoints) * size  # (K, 4)
    if fixed_width is None:
        width = jax.nn.softplus(strokes.width) + min_width  # (K,)
    else:
        width = jnp.full(strokes.width.shape, fixed_width)
    gate = jax.nn.sigmoid(strokes.gate)  # (K,)
    if hard_gate:
        on = (gate > 0.5).astype(gate.dtype)
        if min_strokes > 0:
            kth = jnp.sort(strokes.gate)[-min(min_strokes, gate.shape[0])]
            on = jnp.maximum(on, (strokes.gate >= kth).astype(gate.dtype))
        gate = on + gate - jax.lax.stop_gradient(gate)
    colour = jax.nn.softmax(strokes.colour / temperature, axis=-1)  # (K, 16)
    if hard_colour:
        hard = jax.nn.one_hot(jnp.argmax(colour, axis=-1), N_COLOURS)
        colour = hard + colour - jax.lax.stop_gradient(colour)  # straight-through

    def cover(end, w):
        d = _segment_distance(pts, end[:2], end[2:])
        return jax.nn.sigmoid((w / 2.0 - d) / edge_softness)

    coverage = jax.vmap(cover)(ends, width) * gate[:, None, None]  # (K, S, S)

    def composite(carry, inputs):
        col, alpha = carry
        cov, c = inputs
        cov = cov[..., None]
        col = col * (1.0 - cov) + c[None, None, :] * cov
        alpha = alpha + cov[..., 0] * (1.0 - alpha)
        return (col, alpha), None

    init_col = jnp.zeros((size, size, N_COLOURS))
    init_alpha = jnp.zeros((size, size))
    (col, alpha), _ = jax.lax.scan(composite, (init_col, init_alpha), (coverage, colour))
    # Where nothing was drawn, ``col`` is zero; normalise to a distribution anyway so downstream
    # code can treat it as colour probabilities (the alpha says how much to trust it).
    col = col / jnp.maximum(alpha[..., None], 1e-6)
    col = jnp.where(alpha[..., None] > 1e-6, col, 1.0 / N_COLOURS)

    length = jnp.linalg.norm(ends[:, 2:] - ends[:, :2] + 1e-9, axis=-1)
    ink = jnp.sum(gate * (length + stroke_cost if fixed_width is not None else length + width))
    return Canvas(colour=col, alpha=alpha, ink=ink)


def canvas_to_indices(canvas: Canvas, background: int = 8) -> jax.Array:
    """Hard palette-index image of a canvas for display (background colour where alpha < 0.5)."""
    idx = jnp.argmax(canvas.colour, axis=-1)
    return jnp.where(canvas.alpha > 0.5, idx, background)
