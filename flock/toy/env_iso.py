"""Isometric episode generation for the Milestone 0 drawing game.

Same ``Episode`` interface as ``flock.toy.env`` (the top-down baseline), rendered with
``flock.iso``. Differences:

- Views are heading-aligned isometric; "ahead" is the up-right screen diagonal.
- The receiver chooses one of the 16x16 cells of 8x8 px in its 128x128 view; the target is
  the cells around the target tile's diamond centre.
- The canvas (bubble or stamp) sits at a fixed screen position at the bottom-centre of the
  receiver's view, in front of the sender sprite (bubble mode) or on bare ground (stamp mode).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from flock import iso, sprites_iso as S
from flock.iso import VIEW
from flock.palette import TRANSPARENT
from flock.toy.env import Episode

N_TYPES = S.N_OBJECT_TYPES
OBJECT_NAMES = S.OBJECT_NAMES
WORLD_TILES = 24
CELLS = 16  # the receiver chooses one of CELLS x CELLS cells
CELL = VIEW // CELLS  # px per receiver decision cell (8 at a 128 view)
CANVAS = 32  # px side of the drawing as it appears in the view (covers a sprite's footprint)

# Receiver layout, as (row, col) tile offsets in the *rotated* (heading-aligned) frame, chosen so
# that every slot's sprite is inside the view and clear of the canvas block.
_SLOTS = jnp.array([[-3, -3], [-3, 3], [3, -3], [-5, 0], [0, -5], [0, 5], [5, 0]])
MAX_CANDIDATES = _SLOTS.shape[0]
_BUBBLE_SENDER = jnp.array([3, 3])  # rotated offset: directly below the receiver on screen
CANVAS_PX = jnp.array([VIEW - CANVAS, VIEW // 2 - CANVAS // 2])  # (row, col) top-left: bottom-centre
_AHEAD = jnp.array([-3, 0])  # rotated offset of the sender's target


def _cell_mask_for_tile(d_rot: jax.Array, heading_zero: jax.Array = jnp.int32(0)) -> jax.Array:
    """(CELLS, CELLS) bool mask of the cells covering a 2*CELL px square centred on a tile's
    diamond centre (already in the rotated frame, so heading 0 for the projection)."""
    centre = iso.tile_screen_centre(iso.unrotate_offset(d_rot, heading_zero), heading_zero)
    x, y = centre[0], centre[1]
    rows = jnp.arange(CELLS)
    r0, r1 = (y - CELL) // CELL, (y + CELL - 1) // CELL
    c0, c1 = (x - CELL) // CELL, (x + CELL - 1) // CELL
    rm = (rows >= r0) & (rows <= r1)
    cm = (rows >= c0) & (rows <= c1)
    return rm[:, None] & cm[None, :]


def _random_ground(key: jax.Array, n: int) -> jax.Array:
    """Grass states only; mostly short with patches of other growth."""
    return jax.random.choice(key, 4, (n, n), p=jnp.array([0.05, 0.55, 0.25, 0.15])).astype(jnp.int32)


def make_episode(key: jax.Array, n_candidates: int, mode: str, random_heading: bool = True) -> Episode:
    k_t, k_g1, k_g2, k_c, k_slot, k_h1, k_h2, k_f = jax.random.split(key, 8)
    k_f1, k_f2 = jax.random.split(k_f)
    target = jax.random.randint(k_t, (), 0, N_TYPES)
    centre = jnp.array([WORLD_TILES // 2, WORLD_TILES // 2])

    # --- sender ------------------------------------------------------------------------
    h_s = jax.random.randint(k_h1, (), 0, 4) if random_heading else jnp.int32(0)
    obj_s = iso.Objects(kind=target[None], pos=(centre + iso.unrotate_offset(_AHEAD, h_s))[None, :], present=jnp.array([True]), heading=jax.random.randint(k_f1, (1,), 0, 4))
    ag_s = iso.Agents(kind=jnp.array([S.SHEEP]), pos=centre[None, :], heading=h_s[None], present=jnp.array([True]))
    sender_view = iso.render_view(_random_ground(k_g1, WORLD_TILES), obj_s, ag_s, centre, h_s)

    # --- receiver ----------------------------------------------------------------------
    h_r = jax.random.randint(k_h2, (), 0, 4) if random_heading else jnp.int32(0)
    perm = jax.random.permutation(k_c, N_TYPES)
    perm = perm[jnp.argsort(perm == target)]
    cand_types = jnp.concatenate([target[None], perm[: n_candidates - 1]])
    slot_idx = jax.random.permutation(k_slot, MAX_CANDIDATES)[:n_candidates]
    slots_rot = _SLOTS[slot_idx]
    cand_pos = centre[None, :] + jax.vmap(lambda d: iso.unrotate_offset(d, h_r))(slots_rot)
    obj_r = iso.Objects(kind=cand_types, pos=cand_pos, present=jnp.ones((n_candidates,), bool), heading=jax.random.randint(k_f2, (n_candidates,), 0, 4))
    sender_here = jnp.array(mode == "bubble")
    ag_r = iso.Agents(
        kind=jnp.array([S.SHEEP, S.SHEEP]),
        pos=jnp.stack([centre, centre + iso.unrotate_offset(_BUBBLE_SENDER, h_r)]),
        heading=jnp.stack([h_r, h_r]),
        present=jnp.array([True, sender_here]),
    )
    receiver_base = iso.render_view(_random_ground(k_g2, WORLD_TILES), obj_r, ag_r, centre, h_r)

    masks = jax.vmap(_cell_mask_for_tile)(slots_rot)
    return Episode(
        target_type=target,
        sender_heading=h_s,
        sender_view=sender_view,
        receiver_base=receiver_base,
        canvas_px=CANVAS_PX,
        target_mask=masks[0],
        candidate_types=cand_types,
        candidate_masks=masks,
    )


@functools.partial(jax.jit, static_argnames=("batch", "n_candidates", "mode", "random_heading"))
def make_batch(key: jax.Array, batch: int, n_candidates: int = 4, mode: str = "bubble", random_heading: bool = True) -> Episode:
    keys = jax.random.split(key, batch)
    return jax.vmap(lambda k: make_episode(k, n_candidates, mode, random_heading))(keys)


def _object_view(k: jax.Array, silhouette: jax.Array) -> tuple[jax.Array, jax.Array]:
    k1, k2, k3, k4 = jax.random.split(k, 4)
    t = jax.random.randint(k1, (), 0, N_TYPES)
    h = jax.random.randint(k3, (), 0, 4)
    centre = jnp.array([WORLD_TILES // 2, WORLD_TILES // 2])
    obj = iso.Objects(kind=t[None], pos=(centre + iso.unrotate_offset(_AHEAD, h))[None, :], present=jnp.array([True]), heading=jax.random.randint(k4, (1,), 0, 4))
    ag = iso.Agents(kind=jnp.array([S.SHEEP]), pos=centre[None, :], heading=h[None], present=jnp.array([False]))
    return iso.render_view(_random_ground(k2, WORLD_TILES), obj, ag, centre, h, silhouette=silhouette[None]), t


@functools.partial(jax.jit, static_argnames=("batch",))
def object_views(key: jax.Array, batch: int) -> tuple[jax.Array, jax.Array]:
    """Views with a single object ahead of an (invisible) viewer on grass, random heading."""
    return jax.vmap(lambda k: _object_view(k, jnp.bool_(False)))(jax.random.split(key, batch))


@functools.partial(jax.jit, static_argnames=("batch",))
def object_views_dropout(key: jax.Array, batch: int, p: float) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Colour dropout for perception pre-training: each object is drawn, with probability
    ``p``, as a solid outline-colour silhouette on its normal ground. Returns (augmented views,
    the same scenes in full colour, types); the autoencoder is trained to reconstruct the
    full-colour scene, so it must recognise objects by shape."""
    keys = jax.random.split(key, batch + 1)
    sil = jax.random.bernoulli(keys[0], p, (batch,))
    aug, t = jax.vmap(_object_view)(keys[1:], sil)
    clean, _ = jax.vmap(lambda k: _object_view(k, jnp.bool_(False)))(keys[1:])
    return aug, clean, t


@jax.jit
def plain_ground_view(key: jax.Array) -> jax.Array:
    centre = jnp.array([WORLD_TILES // 2, WORLD_TILES // 2])
    obj = iso.Objects(kind=jnp.zeros((1,), jnp.int32), pos=centre[None, :], present=jnp.array([False]))
    ag = iso.Agents(kind=jnp.array([S.SHEEP]), pos=centre[None, :], heading=jnp.array([0]), present=jnp.array([False]))
    return iso.render_view(_random_ground(key, WORLD_TILES), obj, ag, centre, jnp.int32(0))


# Where the "ahead" object's sprite box sits in a sender/object view (heading-aligned, so
# fixed): its bottom on the bottom vertex of the diamond at rotated offset _AHEAD.
_ahead_centre = iso.tile_screen_centre(_AHEAD, jnp.int32(0))
AHEAD_SPRITE_PX = jnp.array([int(_ahead_centre[1]) + iso.TILE_H // 2 - iso.BOX_H, int(_ahead_centre[0]) - iso.BOX_W // 2])
# A CANVAS x CANVAS canvas centred on that sprite box.
PROBE_CANVAS_PX = jnp.array([int(AHEAD_SPRITE_PX[0]) + iso.BOX_H // 2 - CANVAS // 2, int(AHEAD_SPRITE_PX[1]) + iso.BOX_W // 2 - CANVAS // 2])


def object_thumbnail(t: int, bg: int = 1) -> np.ndarray:
    """CANVAS x CANVAS palette tile showing object type ``t`` on a grass diamond, for galleries
    (the fox facing toward the viewer, down-right)."""
    cell = np.full((CANVAS, CANVAS), bg, np.uint8)
    g = S.GROUND["short"]
    m = g != TRANSPARENT
    gy, gx = CANVAS - iso.TILE_H - 1, CANVAS // 2 - iso.TILE_W // 2
    cell[gy : gy + iso.TILE_H, gx : gx + iso.TILE_W][m] = g[m]
    spr = S.OBJECT_STACK[t, 1]
    h, w = spr.shape
    y0, x0 = CANVAS - 1 - h, CANVAS // 2 - w // 2
    sub = cell[max(y0, 0) : y0 + h, x0 : x0 + w]
    spr = spr[h - sub.shape[0] :]
    sm = spr != TRANSPARENT
    sub[sm] = spr[sm]
    return cell
