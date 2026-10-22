"""Rotatable isometric rendering of the tile world, vectorised in JAX.

Projection (rotation 0). Tile (r, c) has a 2:1 diamond whose bounding box is W x H px
(TILE_W x TILE_H, currently 16x8) with top-left corner at

    X = (c - r) * W/2 - W/2,   Y = (c + r) * H/2

in world-screen coordinates (origin arbitrary). Its centre is (X + W/2, Y + H/2). Inverse: for
a pixel centre (x, y) measured from a reference diamond centre,  u = x/(W/2) + y/(H/2),
v = y/(H/2) - x/(W/2),  c = floor((u + 1) / 2),  r = floor((v + 1) / 2), which is exactly the
L1-diamond tiling the ground sprites discretise (the diamond is the L-inf unit square in (u, v);
``sprites_iso.DIAMOND`` is the same rule).

Heading-aligned view. The view is centred on an agent and the tile *layout* is rotated about the
agent by its heading before projection; every sprite is drawn upright (billboarded), so the art
never rotates. With headings 0..3 = N, E, S, W and N = (-1, 0) in (row, col), "ahead" projects
to the up-right screen diagonal. Out-of-world pixels are void (0).

Drawing order: ground by gather (one lookup per pixel), then objects and agents pasted back to
front by rotated depth (r' + c').
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from flock import sprites_iso as S
from flock.palette import N_COLOURS, TRANSPARENT

TILE_W, TILE_H = S.TILE_W, S.TILE_H
VIEW = 128  # 16 tiles across at 16x8 px tiles
GROUND_STACK = jnp.asarray(S.GROUND_STACK)  # (n_ground, variants, TILE_H, TILE_W)
OBJECT_STACK = jnp.asarray(S.OBJECT_STACK)  # (n_obj, n_variants, 4, BOX_H, BOX_W)
ANIMAL_STACK = jnp.asarray(S.ANIMAL_STACK)  # (n_animal, 4, BOX_H, BOX_W)
BOX_H, BOX_W = S.BOX_H, S.BOX_W


class Objects(NamedTuple):
    kind: jax.Array  # (N,)
    pos: jax.Array  # (N, 2) tile (row, col)
    present: jax.Array  # (N,) bool
    heading: jax.Array | None = None  # (N,) world heading; only the fox's sprite depends on it
    variant: jax.Array | None = None  # (N,) sprite variant within the type (default 0)
    offset: jax.Array | None = None  # (N, 2) float sub-tile offset (row, col) in tiles, world frame
    scale: jax.Array | None = None  # (N,) sprite scale (nearest, anchored at the base; default 1)


class Agents(NamedTuple):
    kind: jax.Array  # (M,)
    pos: jax.Array  # (M, 2)
    heading: jax.Array  # (M,) 0..3 world heading (N, E, S, W)
    present: jax.Array  # (M,) bool


def rotate_offset(d: jax.Array, heading: jax.Array) -> jax.Array:
    """Rotate a (row, col) tile offset so that world heading ``heading`` becomes north."""
    r, c = d[..., 0], d[..., 1]
    return jax.lax.switch(
        heading,
        [
            lambda: jnp.stack([r, c], -1),
            lambda: jnp.stack([-c, r], -1),  # E -> N
            lambda: jnp.stack([-r, -c], -1),  # S -> N
            lambda: jnp.stack([c, -r], -1),  # W -> N
        ],
    )


def unrotate_offset(d: jax.Array, heading: jax.Array) -> jax.Array:
    r, c = d[..., 0], d[..., 1]
    return jax.lax.switch(
        heading,
        [
            lambda: jnp.stack([r, c], -1),
            lambda: jnp.stack([c, -r], -1),
            lambda: jnp.stack([-r, -c], -1),
            lambda: jnp.stack([-c, r], -1),
        ],
    )


def project(d: jax.Array) -> jax.Array:
    """Rotated tile offset (row, col) -> screen offset (x, y) of the diamond's top-left box
    corner, relative to the reference tile's top-left corner."""
    r, c = d[..., 0], d[..., 1]
    return jnp.stack([(c - r) * (TILE_W // 2), (c + r) * (TILE_H // 2)], -1)


def scale_sprite(sprite: jax.Array, s: jax.Array) -> jax.Array:
    """Nearest-neighbour rescale of a (BOX_H, BOX_W) sprite by ``s`` <= 1, anchored at the
    bottom-centre (where it stands), aspect ratio kept; the freed border is transparent."""
    y = jnp.arange(BOX_H)[:, None]
    x = jnp.arange(BOX_W)[None, :]
    src_yb = jnp.floor(((BOX_H - 1 - y) + 0.5) / s).astype(jnp.int32)  # rows up from the base
    src_x = jnp.floor((x + 0.5 - BOX_W / 2) / s + BOX_W / 2).astype(jnp.int32)
    ok = (src_yb < BOX_H) & (src_x >= 0) & (src_x < BOX_W)
    out = sprite[jnp.clip(BOX_H - 1 - src_yb, 0, BOX_H - 1), jnp.clip(src_x, 0, BOX_W - 1)]
    return jnp.where(ok, out, jnp.uint8(TRANSPARENT)).astype(sprite.dtype)


def facing(heading: jax.Array, view_heading: jax.Array) -> jax.Array:
    """Sprite facing index of an agent with world heading ``heading`` as seen in a view rotated
    by ``view_heading``: the heading in the rotated frame (0 N away/up-right, 1 E toward/
    down-right, 2 S toward/down-left, 3 W away/up-left). See ``sprites_iso``."""
    return (heading - view_heading) % 4


def render_view(
    tile_map: jax.Array,
    objects: Objects,
    agents: Agents,
    pos: jax.Array,
    heading: jax.Array,
    radius_px: int | None = None,
    silhouette: jax.Array | None = None,
    silhouette_colour: int = 15,
) -> jax.Array:
    """VIEW x VIEW palette view centred on tile ``pos``, heading-aligned.

    ``silhouette`` (N,) bool, per object: draw that object's sprite as a solid
    ``silhouette_colour`` shape (used for colour-dropout augmentation in perception
    pre-training). Agents are never silhouetted."""
    hh, ww = tile_map.shape
    # Screen position of the reference (agent) tile's box top-left inside the view: the view
    # centre sits on the diamond centre.
    ox = VIEW // 2 - TILE_W // 2
    oy = VIEW // 2 - TILE_H // 2
    xs = jnp.arange(VIEW)[None, :] + jnp.zeros((VIEW, 1), jnp.int32)
    ys = jnp.arange(VIEW)[:, None] + jnp.zeros((1, VIEW), jnp.int32)
    # pixel centres relative to the reference diamond's *centre*. In (u, v) the L1 diamond of
    # tile (r', c') is the L-inf square max(|u - 2c'|, |v - 2r'|) <= 1.
    px = xs - ox - TILE_W // 2 + 0.5
    py = ys - oy - TILE_H // 2 + 0.5
    u = px / (TILE_W // 2) + py / (TILE_H // 2)
    v = py / (TILE_H // 2) - px / (TILE_W // 2)
    # rotated tile offset of the pixel's diamond (r', c') relative to the reference tile
    c_rot = jnp.floor((u + 1.0) / 2.0).astype(jnp.int32)
    r_rot = jnp.floor((v + 1.0) / 2.0).astype(jnp.int32)
    d_rot = jnp.stack([r_rot, c_rot], -1)
    d_world = unrotate_offset(d_rot, heading)
    tr = pos[0] + d_world[..., 0]
    tc = pos[1] + d_world[..., 1]
    inside = (tr >= 0) & (tr < hh) & (tc >= 0) & (tc < ww)
    state = tile_map[jnp.clip(tr, 0, hh - 1), jnp.clip(tc, 0, ww - 1)]
    # offset of the pixel inside its diamond's box
    box = project(d_rot)
    bx = xs - ox - box[..., 0]
    by = ys - oy - box[..., 1]
    bx = jnp.clip(bx, 0, TILE_W - 1)
    by = jnp.clip(by, 0, TILE_H - 1)
    variant = S.tile_variant(jnp.clip(tr, 0, hh - 1), jnp.clip(tc, 0, ww - 1))
    ground = GROUND_STACK[state, variant, by, bx]
    img = jnp.where(inside, ground, 0).astype(jnp.uint8)

    # --- sprites, back to front -----------------------------------------------------------
    def paste(im, sprite, d_rot_item, pix, present, sil):
        sprite = jnp.where(sil & (sprite != TRANSPARENT), jnp.uint8(silhouette_colour), sprite)
        scr = project(d_rot_item)
        x0 = ox + scr[0] + TILE_W // 2 - BOX_W // 2 + pix[0]
        y0 = oy + scr[1] + TILE_H - BOX_H + pix[1]
        # Clip into the view: sprites partly outside are cropped via a padded buffer.
        pad_im = jnp.pad(im, ((BOX_H, BOX_H), (BOX_W, BOX_W)))
        r = jnp.clip(y0 + BOX_H, 0, VIEW + BOX_H)
        c = jnp.clip(x0 + BOX_W, 0, VIEW + BOX_W)
        patch = jax.lax.dynamic_slice(pad_im, (r, c), (BOX_H, BOX_W))
        mask = (sprite != TRANSPARENT) & present
        new = jnp.where(mask, sprite, patch).astype(im.dtype)
        pad_im = jax.lax.dynamic_update_slice(pad_im, new, (r, c))
        return pad_im[BOX_H : BOX_H + VIEW, BOX_W : BOX_W + VIEW]

    # gather all pasteables: objects then agents, with rotated offsets and depth
    o_d = rotate_offset(objects.pos - pos[None, :], heading)
    a_d = rotate_offset(agents.pos - pos[None, :], heading)
    n_obj = objects.kind.shape[0]
    o_head = jnp.zeros_like(objects.kind) if objects.heading is None else objects.heading
    o_var = jnp.zeros_like(objects.kind) if objects.variant is None else objects.variant
    o_spr = OBJECT_STACK[objects.kind, o_var, facing(o_head, heading)]
    if objects.scale is not None:
        o_spr = jax.vmap(scale_sprite)(o_spr, objects.scale)
    # sub-tile offsets: rotate with the view like tile offsets, project to whole pixels
    o_off = jnp.zeros((n_obj, 2)) if objects.offset is None else rotate_offset(objects.offset, heading)
    o_pix = jnp.round(jnp.stack([(o_off[:, 1] - o_off[:, 0]) * (TILE_W // 2), (o_off[:, 1] + o_off[:, 0]) * (TILE_H // 2)], -1)).astype(jnp.int32)
    a_spr = ANIMAL_STACK[agents.kind, facing(agents.heading, heading)]
    sprites = jnp.concatenate([o_spr, a_spr], 0)
    d_all = jnp.concatenate([o_d, a_d], 0)
    pix_all = jnp.concatenate([o_pix, jnp.zeros((agents.kind.shape[0], 2), jnp.int32)], 0)
    off_all = jnp.concatenate([o_off, jnp.zeros((agents.kind.shape[0], 2))], 0)
    present = jnp.concatenate([objects.present, agents.present], 0)
    sil_obj = jnp.zeros(objects.kind.shape, bool) if silhouette is None else silhouette
    sil = jnp.concatenate([sil_obj, jnp.zeros(agents.kind.shape, bool)], 0)
    # only things near the view matter; mark distant ones absent to keep the paste a no-op
    near = (jnp.abs(d_all[:, 0]) + jnp.abs(d_all[:, 1])) <= (VIEW // TILE_H) + 2
    present = present & near
    depth = d_all[:, 0] + d_all[:, 1] + off_all[:, 0] + off_all[:, 1]
    order = jnp.argsort(depth)

    def body(im, i):
        j = order[i]
        return paste(im, sprites[j], d_all[j], pix_all[j], present[j], sil[j]), None

    img, _ = jax.lax.scan(body, img, jnp.arange(sprites.shape[0]))
    if radius_px is not None:
        d2 = (xs - VIEW // 2) ** 2 + (ys - VIEW // 2) ** 2
        img = jnp.where(d2 <= radius_px * radius_px, img, 0).astype(jnp.uint8)
    return img


def tile_screen_centre(d_world: jax.Array, heading: jax.Array) -> jax.Array:
    """Screen (x, y) pixel of a tile's diamond centre in a view, given its world offset from the
    viewing agent. Useful for placing targets, canvases and tests."""
    d_rot = rotate_offset(d_world, heading)
    box = project(d_rot)
    ox = VIEW // 2 - TILE_W // 2
    oy = VIEW // 2 - TILE_H // 2
    return jnp.stack([ox + box[..., 0] + TILE_W // 2, oy + box[..., 1] + TILE_H // 2], -1)


def render_world(tile_map: jax.Array, objects: Objects, agents: Agents, pad: int = 8) -> jax.Array:
    """Whole world at rotation 0 for display (numpy-friendly, not vectorised per agent)."""
    hh, ww = tile_map.shape
    W = (hh + ww) * (TILE_W // 2) + 2 * pad
    H = (hh + ww) * (TILE_H // 2) + 2 * pad + BOX_H
    ox = pad + hh * (TILE_W // 2) - TILE_W // 2
    oy = pad
    img = np.zeros((H, W), np.uint8)
    g = np.asarray(GROUND_STACK)
    tm = np.asarray(tile_map)
    for r in range(hh):
        for c in range(ww):
            x = ox + (c - r) * (TILE_W // 2)
            y = oy + (c + r) * (TILE_H // 2)
            spr = g[tm[r, c], S.tile_variant(r, c)]
            m = spr != TRANSPARENT
            img[y : y + TILE_H, x : x + TILE_W][m] = spr[m]
    items = []
    o_head = np.zeros_like(np.asarray(objects.kind)) if objects.heading is None else np.asarray(objects.heading)
    for k, p, h, pr in zip(np.asarray(objects.kind), np.asarray(objects.pos), o_head, np.asarray(objects.present)):
        if pr:
            items.append((int(p[0] + p[1]), np.asarray(OBJECT_STACK[k, 0, int(facing(h, 0))]), p))
    for k, p, h, pr in zip(np.asarray(agents.kind), np.asarray(agents.pos), np.asarray(agents.heading), np.asarray(agents.present)):
        if pr:
            items.append((int(p[0] + p[1]), np.asarray(ANIMAL_STACK[k, int(facing(h, 0))]), p))
    for _, spr, p in sorted(items, key=lambda t: t[0]):
        x = ox + (p[1] - p[0]) * (TILE_W // 2) + TILE_W // 2 - BOX_W // 2
        y = oy + (p[1] + p[0]) * (TILE_H // 2) + TILE_H - BOX_H
        m = spr != TRANSPARENT
        img[y : y + BOX_H, x : x + BOX_W][m] = spr[m]
    return img


def to_onehot(view: jax.Array) -> jax.Array:
    return jax.nn.one_hot(view, N_COLOURS, dtype=jnp.float32)
