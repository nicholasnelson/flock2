"""Tile world rendering and egocentric views, vectorised in JAX.

Conventions
- The world is an (H, W) grid of tiles at 4 px per tile. World images are palette-index arrays
  of shape (H*4, W*4).
- Objects occupy 2x2 tiles (8x8 px) and are placed by their top-left tile.
- An agent has a tile position and a heading in {0: N, 1: E, 2: S, 3: W}.
- An egocentric view is a 64x64 px window (16x16 tiles) centred on the agent, rotated so that
  the agent's heading points up. Pixels outside the world are palette index 0 (void).
- Views are converted to one-hot (64, 64, 16) floats for the networks, which lets a soft
  drawing canvas be composited in differentiably.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from flock import sprites
from flock.palette import N_COLOURS, TRANSPARENT

TILE = sprites.TILE
SPRITE = sprites.SPRITE
VIEW = 64
VIEW_TILES = VIEW // TILE

GROUND_STACK = jnp.asarray(sprites.GROUND_STACK)  # (n_ground, 4, 4)
OBJECT_STACK = jnp.asarray(sprites.OBJECT_STACK)  # (n_obj, 8, 8)
ANIMAL_STACK = jnp.asarray(sprites.rotated_stack(sprites.ANIMAL_STACK))  # (n_animal, 4, 8, 8)
SHEEP = sprites.ANIMAL_NAMES.index("sheep")
FOX = sprites.ANIMAL_NAMES.index("fox")


class Objects(NamedTuple):
    kind: jax.Array  # (N,) int object type index
    pos: jax.Array  # (N, 2) int top-left tile (row, col)
    present: jax.Array  # (N,) bool


class Agents(NamedTuple):
    kind: jax.Array  # (M,) int animal type index
    pos: jax.Array  # (M, 2) int tile (row, col) of the sprite's top-left
    heading: jax.Array  # (M,) int in 0..3
    present: jax.Array  # (M,) bool


def render_ground(tile_map: jax.Array) -> jax.Array:
    """(H, W) ground-type map -> (H*4, W*4) palette image."""
    h, w = tile_map.shape
    tiles = GROUND_STACK[tile_map]  # (H, W, 4, 4)
    return tiles.transpose(0, 2, 1, 3).reshape(h * TILE, w * TILE)


def _paste(img: jax.Array, sprite: jax.Array, top_left_px: jax.Array, present: jax.Array) -> jax.Array:
    """Paste one 8x8 sprite with transparency at a pixel position (clipped to the image)."""
    hh, ww = img.shape
    r = jnp.clip(top_left_px[0], 0, hh - SPRITE)
    c = jnp.clip(top_left_px[1], 0, ww - SPRITE)
    patch = jax.lax.dynamic_slice(img, (r, c), (SPRITE, SPRITE))
    mask = (sprite != TRANSPARENT) & present
    new = jnp.where(mask, sprite, patch).astype(img.dtype)
    return jax.lax.dynamic_update_slice(img, new, (r, c))


def render_world(tile_map: jax.Array, objects: Objects, agents: Agents | None = None) -> jax.Array:
    """Full world image (H*4, W*4) with ground, objects, then agents on top."""
    img = render_ground(tile_map)

    def paste_obj(im, i):
        return _paste(im, OBJECT_STACK[objects.kind[i]], objects.pos[i] * TILE, objects.present[i]), None

    img, _ = jax.lax.scan(paste_obj, img, jnp.arange(objects.kind.shape[0]))
    if agents is not None:

        def paste_agent(im, i):
            spr = ANIMAL_STACK[agents.kind[i], agents.heading[i]]
            return _paste(im, spr, agents.pos[i] * TILE, agents.present[i]), None

        img, _ = jax.lax.scan(paste_agent, img, jnp.arange(agents.kind.shape[0]))
    return img


def egocentric_view(world_img: jax.Array, pos: jax.Array, heading: jax.Array, radius_px: int | None = None) -> jax.Array:
    """64x64 window centred on the agent's sprite centre, rotated so heading points up.

    ``pos`` is the agent's top-left tile; the sprite centre is at pos*4 + 4 px. Out-of-world
    pixels are void (0). If ``radius_px`` is given, pixels further than that from the centre are
    also void (night vision).
    """
    hh, ww = world_img.shape
    centre = pos * TILE + SPRITE // 2  # (2,) px, (row, col)
    offs = jnp.arange(VIEW) - VIEW // 2  # -32 .. 31
    rr = centre[0] + offs[:, None] + jnp.zeros((1, VIEW), jnp.int32)
    cc = centre[1] + offs[None, :] + jnp.zeros((VIEW, 1), jnp.int32)
    inside = (rr >= 0) & (rr < hh) & (cc >= 0) & (cc < ww)
    view = world_img[jnp.clip(rr, 0, hh - 1), jnp.clip(cc, 0, ww - 1)]
    view = jnp.where(inside, view, 0)
    if radius_px is not None:
        d2 = offs[:, None] ** 2 + offs[None, :] ** 2
        view = jnp.where(d2 <= radius_px * radius_px, view, 0)
    # Rotate so that heading is up: heading k (clockwise from N) -> rotate the image
    # counter-clockwise by k quarter turns.
    view = jax.lax.switch(heading, [lambda v: v, lambda v: jnp.rot90(v, 1), lambda v: jnp.rot90(v, 2), lambda v: jnp.rot90(v, 3)], view)
    return view


def to_onehot(view: jax.Array) -> jax.Array:
    return jax.nn.one_hot(view, N_COLOURS, dtype=jnp.float32)


def composite_canvas(view_onehot: jax.Array, colour: jax.Array, alpha: jax.Array, top_left_px: jax.Array, opaque_paper: bool, paper_colour: int = 8) -> jax.Array:
    """Blend a soft 16x16 canvas into a one-hot view at a pixel position.

    For a bubble, the canvas sits on opaque paper (white) so the view shows paper wherever the
    stroke alpha is low. For a stamp, the ground shows through.
    """
    s = colour.shape[0]
    vh, vw = view_onehot.shape[:2]
    r = jnp.clip(top_left_px[0], 0, vh - s)
    c = jnp.clip(top_left_px[1], 0, vw - s)
    patch = jax.lax.dynamic_slice(view_onehot, (r, c, 0), (s, s, N_COLOURS))
    if opaque_paper:
        paper = jax.nn.one_hot(jnp.full((s, s), paper_colour), N_COLOURS)
        a = alpha[..., None]
        blended = paper * (1.0 - a) + colour * a
    else:
        a = alpha[..., None]
        blended = patch * (1.0 - a) + colour * a
    return jax.lax.dynamic_update_slice(view_onehot, blended, (r, c, 0))


def world_tile_in_view(agent_pos: jax.Array, heading: jax.Array, tile: jax.Array) -> jax.Array:
    """Map a world tile (row, col) to view-tile coordinates (row, col) in 0..15 for an agent.

    Inverse of the view transform at tile resolution. Returns coordinates that may fall outside
    0..15 if the tile is not in view.
    """
    # Offset of the tile's centre from the agent's sprite centre, in tiles.
    d = (tile.astype(jnp.float32) + 0.5) - (agent_pos.astype(jnp.float32) + 1.0)  # (row, col)
    dr, dc = d[0], d[1]
    # Rotate counter-clockwise by heading quarter turns (same as the image rotation).
    def rot(k, v):
        r_, c_ = v
        return jax.lax.switch(k, [lambda: (r_, c_), lambda: (-c_, r_), lambda: (-r_, -c_), lambda: (c_, -r_)])

    vr, vc = rot(heading, (dr, dc))
    row = jnp.floor(vr + VIEW_TILES / 2).astype(jnp.int32)
    col = jnp.floor(vc + VIEW_TILES / 2).astype(jnp.int32)
    return jnp.stack([row, col])


def random_ground(key: jax.Array, h: int, w: int, p_speck: float = 0.25) -> jax.Array:
    """Mostly plain grass with sparse texture specks."""
    k1, k2 = jax.random.split(key)
    speck = jax.random.bernoulli(k1, p_speck, (h, w))
    which = jax.random.randint(k2, (h, w), 1, 3)  # speck / speck2
    return jnp.where(speck, which, 0).astype(jnp.int32)


def np_rgb(world_img) -> np.ndarray:
    from flock.palette import to_rgb

    return to_rgb(np.asarray(world_img))
