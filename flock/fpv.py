"""The sheep's first-person, Doom-style view (Milestone 1 on; ``flock.meadow.views`` renders it for
every sheep).

Proposed in the Q9 addendum (``docs/flock-2/02-design-questions.md``, 2026-10-23) and adopted by
the owner after the benchmark (``scripts/bench_fpv.py``) against ``iso.render_view``: agents see
a first-person view with billboard sprites as a ~300° strip; humans keep the isometric display.

Projection: cylindrical (a panorama). Column x is a view angle, linear across the field of view;
row y is an elevation angle, linear, with the same angle per pixel (square pixels in angle). A
planar projection cannot show more than 180°, a cylinder can show 300° without distortion
growing at the edges.

- Ground: each pixel below the horizon casts a ray from eye height down to the ground plane;
  the hit point's tile state and position inside the tile pick a texel. The texture is the
  isometric ground art read through the tile-to-diamond mapping, so the same art is reused.
  Beyond ``max_dist`` the ground is a flat far colour; above the horizon, sky.
- Sprites: the isometric sprites as upright billboards. A sprite at distance r appears at its
  bearing, with its base on the ground at that distance and its size proportional to 1/r
  (nearest-neighbour sampling). Painted far to near, so nearer sprites hide farther ones.
- Coordinates: positions are floats in tiles (row, col); heading is an angle in radians, 0 =
  north (-row), increasing clockwise (toward +col).

Everything is per view and pure; vmap for batches.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from flock import iso, sprites_iso as S
from flock.palette import TRANSPARENT

W, H = 256, 64  # strip size (same pixel count as 128 x 128)
FOV = jnp.deg2rad(300.0)
RAD_PER_PX = FOV / W  # ~1.17 degrees per pixel, both axes
HORIZON = 24  # row of the horizon: 24 rows of sky, 40 of ground (looking slightly down)
EYE_H = 0.5  # eye height, in tiles
MAX_DIST = 16.0  # tiles; beyond this the ground is a flat far colour (the Q9 view range)
SKY, FAR = 5, 1  # palette: water_light as sky, grass_dark as the far ground
PX_PER_TILE = 24  # billboard sprite pixels per tile: a sheep (22 px) is ~0.9 tiles tall beside the 0.5-tile eye (owner, 2026-10-23; the isometric art scale of 16 made neighbours loom)


def _ground_texel(tile_map: jax.Array, r: jax.Array, c: jax.Array) -> jax.Array:
    """Palette index of the ground at world point (r, c) (floats, tiles)."""
    n = tile_map.shape[0]
    ri, ci = jnp.floor(r).astype(jnp.int32), jnp.floor(c).astype(jnp.int32)
    inside = (ri >= 0) & (ri < n) & (ci >= 0) & (ci < n)
    rc, cc = jnp.clip(ri, 0, n - 1), jnp.clip(ci, 0, n - 1)
    fu, fv = r - ri, c - ci  # position inside the tile, [0, 1)
    # tile point -> its isometric diamond's 16 x 8 box: x = (fv - fu) W/2 + W/2, y = (fv + fu) H/2
    bx = jnp.clip(jnp.floor((fv - fu) * (S.TILE_W / 2) + S.TILE_W / 2), 0, S.TILE_W - 1).astype(jnp.int32)
    by = jnp.clip(jnp.floor((fv + fu) * (S.TILE_H / 2)), 0, S.TILE_H - 1).astype(jnp.int32)
    tex = iso.GROUND_STACK[tile_map[rc, cc], S.tile_variant(rc, cc), by, bx]
    tex = jnp.where(tex == TRANSPARENT, iso.GROUND_STACK[tile_map[rc, cc], 0, S.TILE_H // 2, S.TILE_W // 2], tex)
    return jnp.where(inside, tex, 0).astype(jnp.uint8)


def render_fpv(
    tile_map: jax.Array,
    sprites: jax.Array,
    spos: jax.Array,
    present: jax.Array,
    eye: jax.Array,
    heading: jax.Array,
    max_dist: jax.Array | float = MAX_DIST,
    sky: jax.Array | int = SKY,
    far: jax.Array | int = FAR,
) -> jax.Array:
    """(H, W) palette view from ``eye`` (2,) looking along ``heading`` (radians). ``max_dist``,
    ``sky`` and ``far`` may vary per view (night shortens the range and darkens the sky).

    ``sprites`` (N, BOX_H, BOX_W) billboards standing at ``spos`` (N, 2) (tiles, float),
    shown where ``present``."""
    xs = jnp.arange(W)
    ys = jnp.arange(H)
    ang = heading + ((xs + 0.5) / W - 0.5) * FOV  # (W,) view angle per column
    elev = (HORIZON - (ys + 0.5)) * RAD_PER_PX  # (H,) elevation per row (+ up)
    # --- ground and sky ---------------------------------------------------------------------
    below = elev < 0
    dist = jnp.where(below, EYE_H / jnp.tan(jnp.maximum(-elev, 1e-4)), jnp.inf)  # (H,)
    dr, dc = -jnp.cos(ang), jnp.sin(ang)  # (W,) ground direction per column
    gr = eye[0] + dist[:, None] * dr[None, :]
    gc = eye[1] + dist[:, None] * dc[None, :]
    near = (dist <= max_dist)[:, None]
    ground = _ground_texel(tile_map, jnp.where(near, gr, eye[0]), jnp.where(near, gc, eye[1]))
    img = jnp.where(below[:, None], jnp.where(near, ground, far), sky).astype(jnp.uint8)

    # --- billboards, far to near --------------------------------------------------------------
    rel = spos - eye[None, :]
    r = jnp.sqrt(jnp.sum(rel**2, -1))  # (N,)
    bearing = jnp.arctan2(rel[:, 1], -rel[:, 0])  # 0 = north, clockwise
    alpha = (bearing - heading + jnp.pi) % (2 * jnp.pi) - jnp.pi  # relative, [-pi, pi)
    visible = present & (r > 0.3) & (r <= max_dist) & (jnp.abs(alpha) < FOV / 2)
    cx = (alpha / FOV + 0.5) * W  # screen column of the sprite's centre
    y_base = HORIZON + jnp.arctan(EYE_H / jnp.maximum(r, 1e-3)) / RAD_PER_PX  # row of its base
    scale = 1.0 / (PX_PER_TILE * jnp.maximum(r, 1e-3) * RAD_PER_PX)  # screen px per sprite px
    order = jnp.argsort(-r)  # far first

    def paste(im, j):
        s = scale[j]
        x0 = cx[j] - (iso.BOX_W / 2) * s
        y0 = y_base[j] - iso.BOX_H * s
        u = jnp.floor((xs[None, :] + 0.5 - x0) / s).astype(jnp.int32)
        v = jnp.floor((ys[:, None] + 0.5 - y0) / s).astype(jnp.int32)
        inside = (u >= 0) & (u < iso.BOX_W) & (v >= 0) & (v < iso.BOX_H)
        tex = sprites[j][jnp.clip(v, 0, iso.BOX_H - 1), jnp.clip(u, 0, iso.BOX_W - 1)]
        draw = inside & (tex != TRANSPARENT) & visible[j]
        return jnp.where(draw, tex, im).astype(jnp.uint8), None

    img, _ = jax.lax.scan(paste, img, order)
    return img


def object_billboards(kind: jax.Array, variant: jax.Array, facing: jax.Array) -> jax.Array:
    """(N, BOX_H, BOX_W) billboards for referent objects."""
    return iso.OBJECT_STACK[kind, variant, facing]
