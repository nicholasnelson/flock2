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


# Instance variation (``variation=True``): each object independently gets a sprite variant (from
# the training set, or the held-out set for evaluation), a sub-tile offset and a scale, so a
# drawing cannot lean on one remembered example, its exact place, or its size.
OFFSET_MAX = 0.3  # tiles, each axis, world frame
SCALE_MIN = 0.6


def _pixel_offset(off_world: jax.Array, heading: jax.Array) -> jax.Array:
    """Whole-pixel screen offset (x, y) of a world-frame sub-tile offset, as the renderer does it."""
    o = iso.rotate_offset(off_world, heading)
    return jnp.round(jnp.stack([(o[..., 1] - o[..., 0]) * (iso.TILE_W // 2), (o[..., 1] + o[..., 0]) * (iso.TILE_H // 2)], -1)).astype(jnp.int32)


def _cell_mask_for_tile(d_rot: jax.Array, pix: jax.Array = jnp.zeros(2, jnp.int32)) -> jax.Array:
    """(CELLS, CELLS) bool mask of the cells covering a 2*CELL px square centred on a tile's
    diamond centre (rotated frame), moved by the object's pixel offset ``pix`` (x, y)."""
    centre = iso.tile_screen_centre(d_rot, jnp.int32(0))
    x, y = centre[0] + pix[0], centre[1] + pix[1]
    rows = jnp.arange(CELLS)
    r0, r1 = (y - CELL) // CELL, (y + CELL - 1) // CELL
    c0, c1 = (x - CELL) // CELL, (x + CELL - 1) // CELL
    rm = (rows >= r0) & (rows <= r1)
    cm = (rows >= c0) & (rows <= c1)
    return rm[:, None] & cm[None, :]


def _random_ground(key: jax.Array, n: int) -> jax.Array:
    """Grass states only; mostly short with patches of other growth."""
    return jax.random.choice(key, 4, (n, n), p=jnp.array([0.05, 0.55, 0.25, 0.15])).astype(jnp.int32)


def _instances(key: jax.Array, types: jax.Array, variation: bool, held_out: bool, scale_min: float = SCALE_MIN, variants: bool = True):
    """(variant, offset (n, 2), scale) for objects of ``types``; the plain world if not
    ``variation``. ``variants=False`` keeps offsets and scales but always uses variant 0."""
    n = types.shape[0]
    if not variation:
        return jnp.zeros((n,), jnp.int32), jnp.zeros((n, 2)), jnp.ones((n,))
    kv, ko, ks = jax.random.split(key, 3)
    n_train = jnp.asarray(S.N_TRAIN_VARIANTS)[types]
    n_all = jnp.asarray(S.N_OBJECT_VARIANTS)[types]
    lo, hi = (n_train, n_all) if held_out else (jnp.zeros_like(n_train), n_train)
    variant = lo + jnp.floor(jax.random.uniform(kv, (n,)) * (hi - lo)).astype(jnp.int32)
    if not variants:
        variant = jnp.zeros((n,), jnp.int32)
    offset = jax.random.uniform(ko, (n, 2), minval=-OFFSET_MAX, maxval=OFFSET_MAX)
    scale = jax.random.uniform(ks, (n,), minval=scale_min, maxval=1.0)
    return variant, offset, scale


def make_episode(key: jax.Array, n_candidates: int, mode: str, random_heading: bool = True, variation: bool = False, held_out: bool = False, n_types: int = N_TYPES) -> Episode:
    k_t, k_g1, k_g2, k_c, k_slot, k_h1, k_h2, k_f, k_i = jax.random.split(key, 9)
    k_f1, k_f2 = jax.random.split(k_f)
    k_i1, k_i2 = jax.random.split(k_i)
    target = jax.random.randint(k_t, (), 0, n_types)
    centre = jnp.array([WORLD_TILES // 2, WORLD_TILES // 2])

    # --- sender ------------------------------------------------------------------------
    h_s = jax.random.randint(k_h1, (), 0, 4) if random_heading else jnp.int32(0)
    v_s, o_s, sc_s = _instances(k_i1, target[None], variation, held_out)
    obj_s = iso.Objects(
        kind=target[None],
        pos=(centre + iso.unrotate_offset(_AHEAD, h_s))[None, :],
        present=jnp.array([True]),
        heading=jax.random.randint(k_f1, (1,), 0, 4),
        variant=v_s,
        offset=o_s,
        scale=sc_s,
    )
    ag_s = iso.Agents(kind=jnp.array([S.SHEEP]), pos=centre[None, :], heading=h_s[None], present=jnp.array([True]))
    sender_view = iso.render_view(_random_ground(k_g1, WORLD_TILES), obj_s, ag_s, centre, h_s)
    pix_s = _pixel_offset(o_s[0], h_s)

    # --- receiver ----------------------------------------------------------------------
    h_r = jax.random.randint(k_h2, (), 0, 4) if random_heading else jnp.int32(0)
    perm = jax.random.permutation(k_c, n_types)
    perm = perm[jnp.argsort(perm == target)]
    cand_types = jnp.concatenate([target[None], perm[: n_candidates - 1]])
    slot_idx = jax.random.permutation(k_slot, MAX_CANDIDATES)[:n_candidates]
    slots_rot = _SLOTS[slot_idx]
    cand_pos = centre[None, :] + jax.vmap(lambda d: iso.unrotate_offset(d, h_r))(slots_rot)
    v_r, o_r, sc_r = _instances(k_i2, cand_types, variation, held_out)
    obj_r = iso.Objects(
        kind=cand_types,
        pos=cand_pos,
        present=jnp.ones((n_candidates,), bool),
        heading=jax.random.randint(k_f2, (n_candidates,), 0, 4),
        variant=v_r,
        offset=o_r,
        scale=sc_r,
    )
    sender_here = jnp.array(mode == "bubble")
    ag_r = iso.Agents(
        kind=jnp.array([S.SHEEP, S.SHEEP]),
        pos=jnp.stack([centre, centre + iso.unrotate_offset(_BUBBLE_SENDER, h_r)]),
        heading=jnp.stack([h_r, h_r]),
        present=jnp.array([True, sender_here]),
    )
    receiver_base = iso.render_view(_random_ground(k_g2, WORLD_TILES), obj_r, ag_r, centre, h_r)

    pix_r = jax.vmap(lambda o: _pixel_offset(o, h_r))(o_r)
    masks = jax.vmap(_cell_mask_for_tile)(slots_rot, pix_r)
    # Two slots sit close on screen ([3, -3] and [5, 0]); with sub-tile offsets their squares can
    # touch. A cell claimed by more than one candidate counts for none, so no click can hedge.
    masks = masks & ~(masks.sum(0) > 1)[None]
    return Episode(
        target_type=target,
        sender_heading=h_s,
        sender_view=sender_view,
        receiver_base=receiver_base,
        canvas_px=CANVAS_PX,
        target_mask=masks[0],
        candidate_types=cand_types,
        candidate_masks=masks,
        probe_px=PROBE_CANVAS_PX + pix_s[::-1],
    )


@functools.partial(jax.jit, static_argnames=("batch", "n_candidates", "mode", "random_heading", "variation", "held_out", "n_types"))
def make_batch(key: jax.Array, batch: int, n_candidates: int = 4, mode: str = "bubble", random_heading: bool = True, variation: bool = False, held_out: bool = False, n_types: int = N_TYPES) -> Episode:
    """``n_types``: only the first ``n_types`` referent types are used (the referent set)."""
    keys = jax.random.split(key, batch)
    return jax.vmap(lambda k: make_episode(k, n_candidates, mode, random_heading, variation, held_out, n_types))(keys)


def _object_view(k: jax.Array, silhouette: jax.Array, variation: bool = False, scale_min: float = SCALE_MIN, scale: jax.Array | None = None, variants: bool = True, held_out: bool = False, outline: jax.Array | None = None, n_types: int = N_TYPES):
    """One single-object view; returns (view, type). ``scale`` overrides the sampled scale."""
    k1, k2, k3, k4, k5 = jax.random.split(k, 5)
    t = jax.random.randint(k1, (), 0, n_types)
    h = jax.random.randint(k3, (), 0, 4)
    centre = jnp.array([WORLD_TILES // 2, WORLD_TILES // 2])
    v, o, sc = _instances(k5, t[None], variation, held_out, scale_min, variants)
    if scale is not None:
        sc = scale[None]
    obj = iso.Objects(
        kind=t[None],
        pos=(centre + iso.unrotate_offset(_AHEAD, h))[None, :],
        present=jnp.array([True]),
        heading=jax.random.randint(k4, (1,), 0, 4),
        variant=v,
        offset=o,
        scale=sc,
    )
    ag = iso.Agents(kind=jnp.array([S.SHEEP]), pos=centre[None, :], heading=h[None], present=jnp.array([False]))
    ol = None if outline is None else outline[None]
    return iso.render_view(_random_ground(k2, WORLD_TILES), obj, ag, centre, h, silhouette=silhouette[None], outline=ol), t


@functools.partial(jax.jit, static_argnames=("batch", "variation", "n_types"))
def object_views(key: jax.Array, batch: int, variation: bool = False, n_types: int = N_TYPES) -> tuple[jax.Array, jax.Array]:
    """Views with a single object ahead of an (invisible) viewer on grass, random heading
    (training variants, offsets and scales if ``variation``)."""
    return jax.vmap(lambda k: _object_view(k, jnp.bool_(False), variation, n_types=n_types))(jax.random.split(key, batch))


@functools.partial(jax.jit, static_argnames=("batch", "variation", "scale_min", "variants", "n_types"))
def object_views_pretrain(key: jax.Array, batch: int, p_silhouette: float, variation: bool = False, scale_min: float = SCALE_MIN, variants: bool = True, n_types: int = N_TYPES):
    """Single-object views for perception pre-training, three renders of each scene:

    - ``inp``: the input; the object is a solid outline-colour silhouette with probability
      ``p_silhouette`` (colour dropout), at the sampled scale;
    - ``tgt``: the same scene in full colour (the reconstruction target);
    - ``anchor``: the same scene in full colour at scale 1 (the reference an invariant judge's
      embedding is pulled towards).

    Returns (inp, tgt, anchor, types)."""
    keys = jax.random.split(key, batch + 1)
    sil = jax.random.bernoulli(keys[0], p_silhouette, (batch,))
    inp, t = jax.vmap(lambda k, s: _object_view(k, s, variation, scale_min, None, variants, n_types=n_types))(keys[1:], sil)
    tgt, _ = jax.vmap(lambda k: _object_view(k, jnp.bool_(False), variation, scale_min, None, variants, n_types=n_types))(keys[1:])
    anchor, _ = jax.vmap(lambda k: _object_view(k, jnp.bool_(False), variation, scale_min, jnp.float32(1.0), variants, n_types=n_types))(keys[1:])
    return inp, tgt, anchor, t


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
    spr = S.OBJECT_STACK[t, 0, 1]
    h, w = spr.shape
    y0, x0 = CANVAS - 1 - h, CANVAS // 2 - w // 2
    sub = cell[max(y0, 0) : y0 + h, x0 : x0 + w]
    spr = spr[h - sub.shape[0] :]
    sm = spr != TRANSPARENT
    sub[sm] = spr[sm]
    return cell
