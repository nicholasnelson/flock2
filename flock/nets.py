"""Networks for Milestone 0, in flax.nnx.

- ``Encoder``: the perception encoder. One encoder is shared by every agent (and, in the shared
  condition, is also how a receiver reads a drawing). Returns a global embedding and the 16x16
  feature map (one cell per view tile) that the receiver's spatial head uses.
- ``Decoder``: reconstructs the 64x64 palette view from the embedding; used only to pre-train the
  encoder as "world perception" before any communication.
- ``StrokeSender`` / ``LatentSender`` / ``PixelSender``: three drawing action spaces. All take the
  sender's embedding and return a soft 16x16 canvas (colour distribution + alpha).
- ``MessageEncoder``: a separate small CNN over the canvas, for the separate-encoder ablation.
- ``Receiver``: logits over the 16x16 view tiles ("go there") from the feature map and embedding.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from flock.palette import N_COLOURS
from flock.raster import CANVAS, Canvas, Strokes, rasterise

EMBED = 256


def _act(x):
    return jax.nn.silu(x)


def _enlarge(x: jax.Array, f: int) -> jax.Array:
    """Nearest-neighbour enlargement of (B, H, W, C) by an integer factor."""
    return x if f == 1 else jnp.repeat(jnp.repeat(x, f, axis=1), f, axis=2)


class Encoder(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, width: int = 32, view: int = 64):
        w = width
        assert view in (64, 128), view
        # at 128 px one extra stride-2 layer up front, so everything after sees 64 x 64 as before
        self.c0 = nnx.Conv(N_COLOURS, N_COLOURS, (3, 3), strides=2, padding="SAME", rngs=rngs) if view == 128 else None
        self.c1 = nnx.Conv(N_COLOURS, w, (3, 3), strides=2, padding="SAME", rngs=rngs)  # 64 -> 32
        self.c2 = nnx.Conv(w, 2 * w, (3, 3), strides=2, padding="SAME", rngs=rngs)  # 32 -> 16
        self.c3 = nnx.Conv(2 * w, 3 * w, (3, 3), strides=2, padding="SAME", rngs=rngs)  # 16 -> 8
        self.c4 = nnx.Conv(3 * w, 4 * w, (3, 3), strides=2, padding="SAME", rngs=rngs)  # 8 -> 4
        self.n2 = nnx.LayerNorm(2 * w, rngs=rngs)
        self.fc = nnx.Linear(4 * 4 * 4 * w, EMBED, rngs=rngs)
        self.nf = nnx.LayerNorm(EMBED, rngs=rngs)
        self.map_channels = 2 * w

    def __call__(self, view: jax.Array) -> tuple[jax.Array, jax.Array]:
        """view: (B, V, V, 16) float, V = 64 or 128 -> (embedding (B, 256), tile map (B, 16, 16, C))."""
        if self.c0 is not None:
            view = _act(self.c0(view))
        h = _act(self.c1(view))
        h = _act(self.n2(self.c2(h)))
        fmap = h
        h = _act(self.c3(h))
        h = _act(self.c4(h))
        h = h.reshape(h.shape[0], -1)
        emb = self.nf(self.fc(h))
        return emb, fmap


class Decoder(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, width: int = 32, view: int = 64):
        w = width
        big = view == 128
        self.fc = nnx.Linear(EMBED, 4 * 4 * 4 * w, rngs=rngs)
        self.d1 = nnx.ConvTranspose(4 * w, 3 * w, (4, 4), strides=(2, 2), padding="SAME", rngs=rngs)  # 4 -> 8
        self.d2 = nnx.ConvTranspose(3 * w, 2 * w, (4, 4), strides=(2, 2), padding="SAME", rngs=rngs)  # 8 -> 16
        self.d3 = nnx.ConvTranspose(2 * w, w, (4, 4), strides=(2, 2), padding="SAME", rngs=rngs)  # 16 -> 32
        self.d4 = nnx.ConvTranspose(w, w // 2 if big else N_COLOURS, (4, 4), strides=(2, 2), padding="SAME", rngs=rngs)  # 32 -> 64
        self.d5 = nnx.ConvTranspose(w // 2, N_COLOURS, (4, 4), strides=(2, 2), padding="SAME", rngs=rngs) if big else None  # 64 -> 128
        self.w = w

    def __call__(self, emb: jax.Array) -> jax.Array:
        h = _act(self.fc(emb)).reshape(emb.shape[0], 4, 4, 4 * self.w)
        h = _act(self.d1(h))
        h = _act(self.d2(h))
        h = _act(self.d3(h))
        if self.d5 is not None:
            return self.d5(_act(self.d4(h)))  # (B, 128, 128, 16) logits
        return self.d4(h)  # (B, 64, 64, 16) logits


class StrokeSender(nnx.Module):
    """Embedding -> K stroke primitives -> rasterised canvas."""

    def __init__(self, rngs: nnx.Rngs, k_strokes: int = 8, hidden: int = 256, size: int = CANVAS, res: int | None = None, mono_colour: int | None = None):
        """Strokes are rasterised directly at ``size`` px, the drawing's size in the view.
        ``res`` < ``size`` draws on a coarser grid that is enlarged (nearest-neighbour) to
        ``size``: same footprint, chunkier pixels. ``mono_colour`` fixes every stroke to one
        palette colour (the sender still chooses position, width and gate)."""
        res = size if res is None else res
        assert size % res == 0, (size, res)
        self.k = k_strokes
        self.size = size
        self.res = res
        self.mono_colour = mono_colour
        self.h1 = nnx.Linear(EMBED, hidden, rngs=rngs)
        self.h2 = nnx.Linear(hidden, hidden, rngs=rngs)
        per = 4 + 1 + N_COLOURS + 1
        self.out = nnx.Linear(hidden, k_strokes * per, rngs=rngs)
        self.per = per

    def strokes(self, emb: jax.Array) -> Strokes:
        h = _act(self.h1(emb))
        h = _act(self.h2(h))
        p = self.out(h).reshape(emb.shape[0], self.k, self.per)
        colour = p[..., 5 : 5 + N_COLOURS]
        if self.mono_colour is not None:
            # logits so peaked the softmax is one-hot to float precision
            colour = jnp.broadcast_to(50.0 * jax.nn.one_hot(self.mono_colour, N_COLOURS), colour.shape)
        return Strokes(
            endpoints=p[..., 0:4],
            width=p[..., 4] - 1.0,
            colour=colour,
            gate=p[..., 5 + N_COLOURS] + 2.0,  # start mostly "on"
        )

    def __call__(self, emb: jax.Array, hard_colour: bool = False) -> Canvas:
        s = self.strokes(emb)
        canvas = jax.vmap(lambda st: rasterise(st, size=self.res, hard_colour=hard_colour))(s)
        f = self.size // self.res
        if f == 1:
            return canvas
        up = lambda x: jnp.repeat(jnp.repeat(x, f, axis=1), f, axis=2)
        # ink is measured in the coarse canvas's pixels; scale it to displayed pixels so the
        # ink cost means the same thing at every resolution
        return Canvas(colour=up(canvas.colour), alpha=up(canvas.alpha), ink=canvas.ink * f)


class LatentSender(nnx.Module):
    """Embedding -> small transposed-conv decoder -> per-pixel colour logits + alpha.

    A continuous "latent decoder" message: easiest to optimise, and the case the literature
    predicts will not be iconic without a resemblance term.
    """

    def __init__(self, rngs: nnx.Rngs, hidden: int = 256, w: int = 32, size: int = CANVAS):
        self.up = size // CANVAS  # decodes 16 x 16, enlarged to ``size``
        self.fc = nnx.Linear(EMBED, 4 * 4 * 2 * w, rngs=rngs)
        self.d1 = nnx.ConvTranspose(2 * w, w, (4, 4), strides=(2, 2), padding="SAME", rngs=rngs)  # 4 -> 8
        self.d2 = nnx.ConvTranspose(w, N_COLOURS + 1, (4, 4), strides=(2, 2), padding="SAME", rngs=rngs)  # 8 -> 16
        self.w = w

    def __call__(self, emb: jax.Array, hard_colour: bool = False) -> Canvas:
        h = _act(self.fc(emb)).reshape(emb.shape[0], 4, 4, 2 * self.w)
        h = _act(self.d1(h))
        o = _enlarge(self.d2(h), self.up)
        colour = jax.nn.softmax(o[..., :N_COLOURS], axis=-1)
        if hard_colour:
            hard = jax.nn.one_hot(jnp.argmax(colour, -1), N_COLOURS)
            colour = hard + colour - jax.lax.stop_gradient(colour)
        alpha = jax.nn.sigmoid(o[..., N_COLOURS])
        ink = jnp.sum(alpha, axis=(1, 2))
        return Canvas(colour=colour, alpha=alpha, ink=ink)


class PixelSender(nnx.Module):
    """Embedding -> per-pixel categorical over (16 colours + transparent) with straight-through
    Gumbel-softmax samples. The discrete-pixel action space, in its only trainable form."""

    def __init__(self, rngs: nnx.Rngs, hidden: int = 512, size: int = CANVAS):
        self.up = size // CANVAS  # 16 x 16 pixels, enlarged to ``size``
        self.h1 = nnx.Linear(EMBED, hidden, rngs=rngs)
        self.out = nnx.Linear(hidden, CANVAS * CANVAS * (N_COLOURS + 1), rngs=rngs)

    def __call__(self, emb: jax.Array, hard_colour: bool = False, key: jax.Array | None = None, tau: float = 1.0) -> Canvas:
        h = _act(self.h1(emb))
        logits = _enlarge(self.out(h).reshape(emb.shape[0], CANVAS, CANVAS, N_COLOURS + 1), self.up)
        if key is not None:
            g = jax.random.gumbel(key, logits.shape)
            logits = logits + g
        soft = jax.nn.softmax(logits / tau, axis=-1)
        hard = jax.nn.one_hot(jnp.argmax(soft, -1), N_COLOURS + 1)
        st = hard + soft - jax.lax.stop_gradient(soft)
        alpha = 1.0 - st[..., N_COLOURS]
        colour = st[..., :N_COLOURS] / jnp.maximum(alpha[..., None], 1e-6)
        colour = jnp.where(alpha[..., None] > 1e-6, colour, 1.0 / N_COLOURS)
        ink = jnp.sum(alpha, axis=(1, 2))
        return Canvas(colour=colour, alpha=alpha, ink=ink)


class MessageEncoder(nnx.Module):
    """Separate small CNN that reads the canvas directly (separate-encoder ablation)."""

    def __init__(self, rngs: nnx.Rngs, w: int = 32, size: int = CANVAS):
        # a 32 px canvas gets one extra stride-2 layer up front
        self.c0 = nnx.Conv(N_COLOURS + 1, N_COLOURS + 1, (3, 3), strides=2, padding="SAME", rngs=rngs) if size == 32 else None
        self.c1 = nnx.Conv(N_COLOURS + 1, w, (3, 3), strides=2, padding="SAME", rngs=rngs)  # 16 -> 8
        self.c2 = nnx.Conv(w, 2 * w, (3, 3), strides=2, padding="SAME", rngs=rngs)  # 8 -> 4
        self.fc = nnx.Linear(4 * 4 * 2 * w, EMBED, rngs=rngs)
        self.n = nnx.LayerNorm(EMBED, rngs=rngs)

    def __call__(self, canvas: Canvas) -> jax.Array:
        x = jnp.concatenate([canvas.colour * canvas.alpha[..., None], canvas.alpha[..., None]], axis=-1)
        if self.c0 is not None:
            x = _act(self.c0(x))
        h = _act(self.c1(x))
        h = _act(self.c2(h))
        return self.n(self.fc(h.reshape(h.shape[0], -1)))


class Receiver(nnx.Module):
    """Tile logits (B, 16, 16) from the feature map, a global context vector, and the feature
    cells under the canvas.

    The canvas pathway is the receiver "looking at the bubble": the 4x4 feature-map cells at
    the canvas position (features from the *shared* perception encoder) are pooled into the
    context. Without it the only route from canvas pixels to the decision is the global
    bottleneck embedding, which is too weak a channel for communication to get started.
    """

    def __init__(self, rngs: nnx.Rngs, map_channels: int, hidden: int = 64):
        self.ctx = nnx.Linear(EMBED, hidden, rngs=rngs)
        self.canvas_read = nnx.Linear(4 * 4 * map_channels, hidden, rngs=rngs)
        self.loc = nnx.Conv(map_channels, hidden, (3, 3), padding="SAME", rngs=rngs)
        self.mix = nnx.Conv(hidden, hidden, (3, 3), padding="SAME", rngs=rngs)
        self.out = nnx.Conv(hidden, 1, (1, 1), rngs=rngs)

    def __call__(self, fmap: jax.Array, context: jax.Array, canvas_tile: jax.Array) -> jax.Array:
        """canvas_tile: (B, 2) top-left tile (row, col) of the canvas in the view."""
        c = fmap.shape[-1]

        def crop(m, t):
            return jax.lax.dynamic_slice(m, (t[0], t[1], 0), (4, 4, c)).reshape(-1)

        cells = jax.vmap(crop)(fmap, canvas_tile)
        ctx = self.ctx(context) + self.canvas_read(cells)
        h = _act(self.loc(fmap) + ctx[:, None, None, :])
        h = _act(self.mix(h))
        return self.out(h)[..., 0]


class Agents(nnx.Module):
    """All M0 parameters in one module so optax can treat them uniformly."""

    def __init__(
        self,
        rngs: nnx.Rngs,
        action_space: str = "strokes",
        k_strokes: int = 8,
        shared_encoder: bool = True,
        width: int = 32,
        canvas_res: int | None = None,
        mono_colour: int | None = None,
        view: int = 64,
        canvas: int = CANVAS,
    ):
        """``view``: px side of the perception input (64 top-down, 128 isometric). ``canvas``: px
        side of the drawing as it appears in the view (16 or 32). ``canvas_res``: resolution the
        strokes are rasterised at (default: ``canvas``)."""
        self.encoder = Encoder(rngs, width, view)
        self.decoder = Decoder(rngs, width, view)
        self.canvas = canvas
        if (canvas_res is not None or mono_colour is not None) and action_space != "strokes":
            raise ValueError("canvas_res and mono_colour are only implemented for the stroke sender")
        if action_space == "strokes":
            self.sender = StrokeSender(rngs, k_strokes, size=canvas, res=canvas_res, mono_colour=mono_colour)
        elif action_space == "latent":
            self.sender = LatentSender(rngs, size=canvas)
        elif action_space == "pixels":
            self.sender = PixelSender(rngs, size=canvas)
        else:
            raise ValueError(action_space)
        self.action_space = action_space
        self.shared_encoder = shared_encoder
        self.message_encoder = None if shared_encoder else MessageEncoder(rngs, size=canvas)
        self.receiver = Receiver(rngs, self.encoder.map_channels)
