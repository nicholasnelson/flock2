"""Episode generation for the Milestone 0 drawing game.

One episode:
- A *sender* sheep stands on a plain meadow with the *target* object (one of the object types)
  two tiles ahead of it, in view, and nothing else. It sees its 64x64 view.
- A *receiver* sheep stands elsewhere on a meadow with ``n_candidates`` objects around it, one
  of the target's type and the rest of other types, all in view. The sender's 16x16 canvas is
  shown to the receiver either as a *bubble* (on white paper beside a sender sprite adjacent to
  the receiver) or as a *stamp* (on the ground, sender absent).
- The receiver chooses a view tile; success if the tile is on the target object's 2x2 footprint.

Everything is generated per episode from a PRNG key and vmapped over a batch. Headings are
random so that views are rotation-aligned; the receiver's target footprint is expressed in its
own view-tile coordinates.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from flock import sprites, world
from flock.world import TILE, VIEW, VIEW_TILES

N_TYPES = sprites.N_OBJECT_TYPES
OBJECT_NAMES = sprites.OBJECT_NAMES
WORLD_TILES = 24  # tile side of each toy meadow


class Episode(NamedTuple):
    target_type: jax.Array  # () int
    sender_heading: jax.Array  # () int
    sender_view: jax.Array  # (64, 64) palette indices
    receiver_base: jax.Array  # (64, 64) palette indices, before the canvas is composited
    canvas_px: jax.Array  # (2,) top-left pixel of the canvas in the receiver's view
    target_mask: jax.Array  # (16, 16) bool: the target's footprint in receiver view tiles
    candidate_types: jax.Array  # (n_candidates,) types shown to the receiver (first = target)
    candidate_masks: jax.Array  # (n_candidates, 16, 16) bool footprints


# Layout around the receiver, as (row, col) tile offsets from the receiver's top-left tile in
# *world* coordinates. The view covers offsets -7..8 in both axes for any heading, so all of
# these are in view. Nothing overlaps: the canvas block (4x4 tiles) sits ahead of the
# receiver, the bubble's sender sprite just beyond it, and candidates around the edges.
_CANVAS_TL = jnp.array([-4, -1])  # rows -4..-1, cols -1..2
_BUBBLE_SENDER = jnp.array([-6, 0])  # rows -6..-5, cols 0..1
_SLOTS = jnp.array([[-5, -5], [-5, 4], [4, -5], [4, 4], [4, 0], [0, -6], [0, 5]])
MAX_CANDIDATES = _SLOTS.shape[0]


def _footprint_mask(agent_pos, heading, tile):
    """(16,16) bool mask of a 2x2-tile object at world tile ``tile`` in the agent's view."""
    corners = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]]) + tile[None, :]
    vt = jax.vmap(lambda t: world.world_tile_in_view(agent_pos, heading, t))(corners)
    rows = jnp.arange(VIEW_TILES)
    masks = (rows[None, :, None] == vt[:, 0, None, None]) & (rows[None, None, :] == vt[:, 1, None, None])
    return jnp.any(masks, axis=0)


def make_episode(key: jax.Array, n_candidates: int, mode: str, random_heading: bool = True) -> Episode:
    k_t, k_s, k_r, k_g1, k_g2, k_c, k_slot, k_h1, k_h2 = jax.random.split(key, 9)
    target = jax.random.randint(k_t, (), 0, N_TYPES)

    # --- sender meadow -----------------------------------------------------------------
    h_s = jax.random.randint(k_h1, (), 0, 4) if random_heading else jnp.int32(0)
    s_pos = jnp.array([WORLD_TILES // 2, WORLD_TILES // 2])
    # target object 2 tiles "ahead" of the sender in world coords given its heading
    ahead = jnp.array([[-3, 0], [0, 3], [3, 0], [0, -3]])[h_s]  # (drow, dcol), 3 tiles so it clears the sprite
    obj_s = world.Objects(kind=target[None], pos=(s_pos + ahead)[None, :], present=jnp.array([True]))
    ag_s = world.Agents(kind=jnp.array([world.SHEEP]), pos=s_pos[None, :], heading=h_s[None], present=jnp.array([True]))
    tm_s = world.random_ground(k_g1, WORLD_TILES, WORLD_TILES)
    img_s = world.render_world(tm_s, obj_s, ag_s)
    sender_view = world.egocentric_view(img_s, s_pos, h_s)

    # --- receiver meadow ---------------------------------------------------------------
    h_r = jax.random.randint(k_h2, (), 0, 4) if random_heading else jnp.int32(0)
    r_pos = jnp.array([WORLD_TILES // 2, WORLD_TILES // 2])
    # candidate types: target first, then distinct distractors. A stable argsort on
    # (perm == target) moves the target to the end of a random permutation with static shape.
    perm = jax.random.permutation(k_c, N_TYPES)
    perm = perm[jnp.argsort(perm == target)]
    cand_types = jnp.concatenate([target[None], perm[: n_candidates - 1]])
    # candidate slots: a random subset of _SLOTS, so the target's slot is random
    slot_idx = jax.random.permutation(k_slot, MAX_CANDIDATES)[:n_candidates]
    cand_pos = r_pos[None, :] + _SLOTS[slot_idx]
    obj_r = world.Objects(kind=cand_types, pos=cand_pos, present=jnp.ones((n_candidates,), bool))
    # the sender sprite is present beyond the canvas block in bubble mode, absent for stamps
    sender_here = jnp.array(mode == "bubble")
    ag_r = world.Agents(
        kind=jnp.array([world.SHEEP, world.SHEEP]),
        pos=jnp.stack([r_pos, r_pos + _BUBBLE_SENDER]),
        heading=jnp.stack([h_r, h_r]),
        present=jnp.array([True, sender_here]),
    )
    tm_r = world.random_ground(k_g2, WORLD_TILES, WORLD_TILES)
    img_r = world.render_world(tm_r, obj_r, ag_r)
    receiver_base = world.egocentric_view(img_r, r_pos, h_r)

    # canvas placement: the 4x4-tile block's corners rotate with heading; its view-space
    # top-left is the min over the corners' view tiles.
    canvas_world_tl = r_pos + _CANVAS_TL
    corners = canvas_world_tl[None, :] + jnp.array([[0, 0], [0, 3], [3, 0], [3, 3]])
    vt = jax.vmap(lambda t: world.world_tile_in_view(r_pos, h_r, t))(corners)
    canvas_tile = jnp.min(vt, axis=0)
    canvas_px = jnp.clip(canvas_tile * TILE, 0, VIEW - 16)

    target_mask = _footprint_mask(r_pos, h_r, cand_pos[0])
    cand_masks = jax.vmap(lambda p: _footprint_mask(r_pos, h_r, p))(cand_pos)
    return Episode(
        target_type=target,
        sender_heading=h_s,
        sender_view=sender_view,
        receiver_base=receiver_base,
        canvas_px=canvas_px,
        target_mask=target_mask,
        candidate_types=cand_types,
        candidate_masks=cand_masks,
    )


def make_batch(key: jax.Array, batch: int, n_candidates: int = 4, mode: str = "bubble", random_heading: bool = True) -> Episode:
    keys = jax.random.split(key, batch)
    return jax.vmap(lambda k: make_episode(k, n_candidates, mode, random_heading))(keys)


def object_views(key: jax.Array, batch: int) -> tuple[jax.Array, jax.Array]:
    """Views with a single centred object of a random type on plain ground (for perception
    pre-training and the iconicity probes). Returns (views (B,64,64), types (B,))."""

    def one(k):
        k1, k2, k3 = jax.random.split(k, 3)
        t = jax.random.randint(k1, (), 0, N_TYPES)
        h = jax.random.randint(k3, (), 0, 4)
        pos = jnp.array([WORLD_TILES // 2, WORLD_TILES // 2])
        ahead = jnp.array([[-3, 0], [0, 3], [3, 0], [0, -3]])[h]
        obj = world.Objects(kind=t[None], pos=(pos + ahead)[None, :], present=jnp.array([True]))
        tm = world.random_ground(k2, WORLD_TILES, WORLD_TILES)
        img = world.render_world(tm, obj, None)
        return world.egocentric_view(img, pos, h), t

    return jax.vmap(one)(jax.random.split(key, batch))


def plain_ground_view(key: jax.Array) -> jax.Array:
    pos = jnp.array([WORLD_TILES // 2, WORLD_TILES // 2])
    tm = world.random_ground(key, WORLD_TILES, WORLD_TILES)
    obj = world.Objects(kind=jnp.zeros((1,), jnp.int32), pos=pos[None, :], present=jnp.array([False]))
    img = world.render_world(tm, obj, None)
    return world.egocentric_view(img, pos, jnp.int32(0))


# Pixel position of the "ahead" object in a sender/object view. The view covers tiles
# pos-7..pos+8; the object is at pos-3 (rows) / pos (cols), i.e. view tiles rows 4..5,
# cols 7..8 -> px rows 16..23, cols 28..35. In the view the sprite appears rotated by the
# agent's heading (objects have no heading of their own).
AHEAD_OBJECT_PX = jnp.array([16, 28])
# A 16x16 canvas placed at px (12, 24) covers rows 12..27, cols 24..39: the object centred in it.
PROBE_CANVAS_PX = jnp.array([12, 24])
