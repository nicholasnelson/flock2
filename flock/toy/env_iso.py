"""Isometric episode generation for the Milestone 0 drawing game.

Same ``Episode`` interface as ``flock.toy.env`` (the top-down baseline), rendered with
``flock.iso``. Differences:

- Views are heading-aligned isometric; "ahead" is the up-right screen diagonal.
- The receiver chooses one of the 16x16 cells of 4x4 px in its view; the target footprint is
  the cells covering the target sprite's tile.
- The canvas (bubble or stamp) sits at a fixed screen position at the bottom-centre of the
  receiver's view, in front of the sender sprite (bubble mode) or on bare ground (stamp mode).
"""

from __future__ import annotations

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
CELL = 4  # px per receiver decision cell
CELLS = VIEW // CELL  # 16

# Receiver layout, as (row, col) tile offsets in the *rotated* (heading-aligned) frame, chosen so
# that every slot's sprite is inside the 64x64 view and clear of the canvas block.
_SLOTS = jnp.array([[-3, -3], [-3, 3], [3, -3], [-5, 0], [0, -5], [0, 5], [5, 0]])
MAX_CANDIDATES = _SLOTS.shape[0]
_BUBBLE_SENDER = jnp.array([3, 3])  # rotated offset: directly below the receiver on screen
CANVAS_PX = jnp.array([VIEW - 16, VIEW // 2 - 8])  # (row, col) top-left: bottom-centre
_AHEAD = jnp.array([-3, 0])  # rotated offset of the sender's target


def _cell_mask_for_tile(d_rot: jax.Array, heading_zero: jax.Array = jnp.int32(0)) -> jax.Array:
    """(16,16) bool mask of the cells covering an 8x8 px square centred on a tile's diamond
    centre (already in the rotated frame, so heading 0 for the projection)."""
    centre = iso.tile_screen_centre(iso.unrotate_offset(d_rot, heading_zero), heading_zero)
    x, y = centre[0], centre[1]
    rows = jnp.arange(CELLS)
    r0, r1 = (y - 4) // CELL, (y + 3) // CELL
    c0, c1 = (x - 4) // CELL, (x + 3) // CELL
    rm = (rows >= r0) & (rows <= r1)
    cm = (rows >= c0) & (rows <= c1)
    return rm[:, None] & cm[None, :]


def _random_ground(key: jax.Array, n: int) -> jax.Array:
    """Grass states only; mostly short with patches of other growth."""
    return jax.random.choice(key, 4, (n, n), p=jnp.array([0.05, 0.55, 0.25, 0.15])).astype(jnp.int32)


def make_episode(key: jax.Array, n_candidates: int, mode: str, random_heading: bool = True) -> Episode:
    k_t, k_g1, k_g2, k_c, k_slot, k_h1, k_h2 = jax.random.split(key, 7)
    target = jax.random.randint(k_t, (), 0, N_TYPES)
    centre = jnp.array([WORLD_TILES // 2, WORLD_TILES // 2])

    # --- sender ------------------------------------------------------------------------
    h_s = jax.random.randint(k_h1, (), 0, 4) if random_heading else jnp.int32(0)
    obj_s = iso.Objects(kind=target[None], pos=(centre + iso.unrotate_offset(_AHEAD, h_s))[None, :], present=jnp.array([True]))
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
    obj_r = iso.Objects(kind=cand_types, pos=cand_pos, present=jnp.ones((n_candidates,), bool))
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


def make_batch(key: jax.Array, batch: int, n_candidates: int = 4, mode: str = "bubble", random_heading: bool = True) -> Episode:
    keys = jax.random.split(key, batch)
    return jax.vmap(lambda k: make_episode(k, n_candidates, mode, random_heading))(keys)


def object_views(key: jax.Array, batch: int) -> tuple[jax.Array, jax.Array]:
    """Views with a single object ahead of an (invisible) viewer on grass, random heading."""

    def one(k):
        k1, k2, k3 = jax.random.split(k, 3)
        t = jax.random.randint(k1, (), 0, N_TYPES)
        h = jax.random.randint(k3, (), 0, 4)
        centre = jnp.array([WORLD_TILES // 2, WORLD_TILES // 2])
        obj = iso.Objects(kind=t[None], pos=(centre + iso.unrotate_offset(_AHEAD, h))[None, :], present=jnp.array([True]))
        ag = iso.Agents(kind=jnp.array([S.SHEEP]), pos=centre[None, :], heading=h[None], present=jnp.array([False]))
        return iso.render_view(_random_ground(k2, WORLD_TILES), obj, ag, centre, h), t

    return jax.vmap(one)(jax.random.split(key, batch))


def plain_ground_view(key: jax.Array) -> jax.Array:
    centre = jnp.array([WORLD_TILES // 2, WORLD_TILES // 2])
    obj = iso.Objects(kind=jnp.zeros((1,), jnp.int32), pos=centre[None, :], present=jnp.array([False]))
    ag = iso.Agents(kind=jnp.array([S.SHEEP]), pos=centre[None, :], heading=jnp.array([0]), present=jnp.array([False]))
    return iso.render_view(_random_ground(key, WORLD_TILES), obj, ag, centre, jnp.int32(0))


# Where the "ahead" object's sprite sits in a sender/object view (heading-aligned, so fixed):
# tile centre of rotated offset (-3, 0) -> screen (44, 26); the sprite box (9x10) sits with its
# bottom at the diamond bottom (y = 28), so rows 19..27, cols 39..48.
_ahead_centre = iso.tile_screen_centre(_AHEAD, jnp.int32(0))
AHEAD_SPRITE_PX = jnp.array([int(_ahead_centre[1]) + iso.TILE_H // 2 - iso.BOX_H, int(_ahead_centre[0]) - iso.BOX_W // 2])
# A 16x16 canvas centred on that sprite box.
PROBE_CANVAS_PX = jnp.array([int(AHEAD_SPRITE_PX[0]) + iso.BOX_H // 2 - 8, int(AHEAD_SPRITE_PX[1]) + iso.BOX_W // 2 - 8])


def object_thumbnail(t: int, bg: int = 1) -> np.ndarray:
    """16x16 palette tile showing object type ``t`` on a grass diamond, for galleries."""
    cell = np.full((16, 16), bg, np.uint8)
    g = S.GROUND["short"]
    m = g != TRANSPARENT
    cell[11:15, 4:12][m] = g[m]
    spr = S.OBJECT_STACK[t]
    h, w = spr.shape
    x0 = 8 - w // 2
    y0 = 15 - h
    sub = cell[y0 : y0 + h, x0 : x0 + w]
    sm = spr != TRANSPARENT
    sub[sm] = spr[sm]
    return cell
