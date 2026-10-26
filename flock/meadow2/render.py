"""Renderer v2: the sheep's first-person view of meadow v2 (spec §§ 4-5, 11).

A 64 x 256 RGB strip, 300 degrees wide (1.17 degrees per column), from an eye ``EYE`` tiles above
the ground, +34 to -74 degrees high. Rows are not even in angle: their heights follow one smooth
curve (a sinh), finest (``ROW_PEAK`` degrees) at ``ROW_CENTRE`` degrees, a little below the
horizon, and growing gradually toward the top and bottom (``ROW_EDGES``). It replaced three
hard-edged bands (3 / 0.5 / 4 degrees) on 2026-10-26, at the owner's request, to spread the fine
rows toward the middle of the view: a pond 5-7 tiles away now gets 4.3 rows (was 5.1), ground 2-3
tiles away 6.6 (was 1.7), and the sky 18 rows (was 28).

- **Ground.** One continuous raised surface. Its height and colour follow the smooth grass field
  (``world.grass_at``): bare ground is tan and flat, and dense grass is deep green and up to
  ``GRASS_H`` tiles tall. Each column marches outward over ``K`` distances (geometric, 0.15 to 90
  tiles). A row shows the first sample whose top rises above that row's line of sight, so nearer
  tall grass hides what is behind it. Water is flat and blue. Outside the meadow the ground is flat
  and grey-brown, behind a post-and-rail fence on the boundary.
- **Objects and other sheep** are flat shapes standing upright, facing the viewer (billboards).
  They are built from rectangles, circles, ellipses and triangles at the sizes of spec § 4, and
  hidden by anything nearer, grass included. A sheep is drawn from the side (head toward where it
  is going), from the front or from the back, by how it moves relative to the line of sight.
  Dead sheep are not drawn.
- **Light.** Every colour is scaled by the light level (``sim.light``), with a slight blue cast at
  night. Within ``HUT_LIGHT_R`` tiles of the hut, the light is at least ``HUT_LIGHT``, and its
  window glows at night.
- **Haze.** Each pixel is blended toward the horizon colour by 1 - exp(-distance / ``HAZE``).
  Nothing has a view range: things vanish when they shrink below a pixel or fade into the haze.

No textures. Pure and jittable; vmap over sheep or meadows.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from flock.meadow2 import sim
from flock.meadow2.world import FLOWER, HUT, MUSHROOM, N_TILES, SALT, TREE, State, grass_at, water_at

H, W = 64, 256
FOV = np.deg2rad(300.0)
EYE = 0.8
GRASS_H = 0.25  # tiles at full density (3); spec § 11: low enough never to hide the mouth point
HAZE = 30.0
K = 192
DISTS = jnp.asarray(0.15 * (90.0 / 0.15) ** (np.arange(K) / (K - 1)), jnp.float32)
HUT_LIGHT, HUT_LIGHT_R = 0.6, 3.0
FENCE_H, POST_H, POST_EVERY = 0.4, 0.5, 2.0


ROW_TOP, ROW_BOTTOM = 34.0, -74.0  # degrees
ROW_PEAK, ROW_CENTRE = 0.6, -8.0  # degrees per row at the finest point, and its elevation


def _row_edges() -> np.ndarray:
    """(H + 1,) row edges in degrees, top to bottom: e(r) = c + a sinh(b (r0 - r)), a b = ROW_PEAK,
    with e(0) = ROW_TOP and e(H) = ROW_BOTTOM (b found by bisection)."""
    c = ROW_CENTRE

    def bottom_gap(b):
        a = ROW_PEAK / b
        r0 = np.arcsinh((ROW_TOP - c) / a) / b
        return c - a * np.sinh(b * (H - r0)) - ROW_BOTTOM

    lo, hi = 1e-4, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if bottom_gap(lo) * bottom_gap(mid) <= 0 else (mid, hi)
    b = (lo + hi) / 2
    a = ROW_PEAK / b
    r0 = np.arcsinh((ROW_TOP - c) / a) / b
    return c + a * np.sinh(b * (r0 - np.arange(H + 1)))


ROW_EDGES = _row_edges()  # degrees, numpy (H + 1,)
ROW_DEG = -np.diff(ROW_EDGES)  # each row's height in degrees
ROW_ELEV = jnp.asarray(np.deg2rad((ROW_EDGES[:-1] + ROW_EDGES[1:]) / 2), jnp.float32)
COL_OFF = jnp.asarray(((np.arange(W) + 0.5) / W - 0.5) * FOV, jnp.float32)


def row_of(elev: jax.Array) -> jax.Array:
    """Fractional row (0 = top edge) at which an elevation (radians) appears."""
    return jnp.interp(-jnp.rad2deg(elev), jnp.asarray(-ROW_EDGES), jnp.arange(H + 1, dtype=jnp.float32))


def _c(*rgb):
    return jnp.array(rgb, jnp.float32)


BARE, SHORT, LUSH = _c(0.62, 0.55, 0.38), _c(0.55, 0.69, 0.33), _c(0.18, 0.44, 0.14)
WATER_C, OUTSIDE_C, WOOD = _c(0.26, 0.50, 0.76), _c(0.46, 0.42, 0.36), _c(0.45, 0.32, 0.20)
SKY_DAY_HI, SKY_DAY_LO = _c(0.53, 0.77, 0.93), _c(0.80, 0.89, 0.96)
SKY_NIGHT_HI, SKY_NIGHT_LO = _c(0.04, 0.06, 0.13), _c(0.08, 0.10, 0.18)
NIGHT_CAST = _c(0.01, 0.015, 0.04)
CREAM, FACE, LEGS = _c(0.93, 0.91, 0.84), _c(0.25, 0.24, 0.24), _c(0.20, 0.18, 0.17)
TRUNK, LEAVES, PINE = _c(0.42, 0.29, 0.17), _c(0.16, 0.38, 0.17), _c(0.10, 0.30, 0.16)
WALL, ROOF, DOOR, GLASS, GLOW = _c(0.78, 0.65, 0.45), _c(0.55, 0.22, 0.16), _c(0.35, 0.22, 0.12), _c(0.30, 0.35, 0.40), _c(1.0, 0.85, 0.45)
SALT_C, STEM, PINK, YELLOW, MUSH_STEM, MUSH_CAP = _c(0.86, 0.86, 0.84), _c(0.30, 0.55, 0.22), _c(0.90, 0.45, 0.65), _c(0.95, 0.85, 0.30), _c(0.90, 0.88, 0.80), _c(0.72, 0.55, 0.36)


def grass_colour(g: jax.Array) -> jax.Array:
    """(..., 3) colour of grass density g in [0, 3]: tan, then light green at 1, then deep green."""
    t1 = jnp.clip(g, 0.0, 1.0)[..., None]
    t2 = (jnp.clip(g, 1.0, 3.0)[..., None] - 1.0) / 2.0
    return jnp.where(g[..., None] < 1.0, BARE + (SHORT - BARE) * t1, SHORT + (LUSH - SHORT) * t2)


def sky(elev: jax.Array, L: jax.Array) -> jax.Array:
    t = jnp.clip(elev / np.deg2rad(30.0), 0.0, 1.0)[..., None]
    lo = SKY_NIGHT_LO + (SKY_DAY_LO - SKY_NIGHT_LO) * L
    hi = SKY_NIGHT_HI + (SKY_DAY_HI - SKY_NIGHT_HI) * L
    return lo + (hi - lo) * t


# --- shapes: (mask, colour) layers in object coordinates (x across, tiles; y up from the ground) ---


def _rect(x, y, x0, x1, y0, y1):
    return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)


def _circle(x, y, cx, cy, r):
    return (x - cx) ** 2 + (y - cy) ** 2 <= r * r


def _ellipse(x, y, cx, cy, a, b):
    return ((x - cx) / a) ** 2 + ((y - cy) / b) ** 2 <= 1.0


def _tri(x, y, y0, y1, w):
    return (y >= y0) & (y <= y1) & (jnp.abs(x) <= 0.5 * w * (y1 - y) / (y1 - y0))


def _layers(x, y, kind, variant, view, side, is_sheep, night_glow):
    """Painter's layers, back to front: (mask, colour, emits light). Only the hut's window emits,
    so the night does not dim it."""
    k = lambda kk: (kind == kk) & ~is_sheep
    s = side  # +1: head to the right
    side_v, front_v, back_v = is_sheep & (view == 0), is_sheep & (view == 1), is_sheep & (view == 2)
    pine = (variant % 2) == 1
    flower_c = jnp.where(((variant % 2) == 0)[..., None], PINK, YELLOW)
    window = GLASS + (GLOW - GLASS) * night_glow
    return [
        # sheep from the side
        (side_v & (_rect(x * s, y, 0.16, 0.25, 0.0, 0.32) | _rect(x * s, y, -0.32, -0.23, 0.0, 0.32)), LEGS, False),
        (side_v & _ellipse(x, y, -0.03 * s, 0.5, 0.42, 0.22), CREAM, False),
        (side_v & _circle(x, y, 0.42 * s, 0.68, 0.14), FACE, False),
        # from the front and the back
        ((front_v | back_v) & (_rect(x, y, -0.17, -0.08, 0.0, 0.3) | _rect(x, y, 0.08, 0.17, 0.0, 0.3)), LEGS, False),
        ((front_v | back_v) & _circle(x, y, 0.0, 0.5, 0.3), CREAM, False),
        (front_v & _circle(x, y, 0.0, 0.52, 0.15), FACE, False),
        (back_v & _circle(x, y, 0.0, 0.38, 0.05), CREAM * 0.85, False),
        # trees: broadleaf (even variants) or pine
        (k(TREE) & ~pine & _rect(x, y, -0.075, 0.075, 0.0, 0.85), TRUNK, False),
        (k(TREE) & ~pine & _circle(x, y, 0.0, 1.3, 0.5), LEAVES, False),
        (k(TREE) & pine & _rect(x, y, -0.07, 0.07, 0.0, 0.4), TRUNK, False),
        (k(TREE) & pine & _tri(x, y, 0.3, 2.0, 0.9), PINE, False),
        # the hut
        (k(HUT) & _rect(x, y, -0.7, 0.7, 0.0, 1.0), WALL, False),
        (k(HUT) & _tri(x, y, 1.0, 1.5, 1.6), ROOF, False),
        (k(HUT) & _rect(x, y, -0.15, 0.15, 0.0, 0.6), DOOR, False),
        (k(HUT) & _rect(x, y, 0.3, 0.55, 0.5, 0.75), window, True),
        # small things
        (k(SALT) & _rect(x, y, -0.2, 0.2, 0.0, 0.3), SALT_C, False),
        (k(FLOWER) & _rect(x, y, -0.012, 0.012, 0.0, 0.12), STEM, False),
        (k(FLOWER) & _circle(x, y, 0.0, 0.13, 0.04), flower_c, False),
        (k(MUSHROOM) & _rect(x, y, -0.02, 0.02, 0.0, 0.08), MUSH_STEM, False),
        (k(MUSHROOM) & _ellipse(x, y, 0.0, 0.08, 0.075, 0.06) & (y >= 0.08), MUSH_CAP, False),
    ]


N_NEAREST = 48


def render(state: State, i: jax.Array) -> jax.Array:
    """(H, W, 3) float RGB in [0, 1]: sheep ``i``'s view."""
    m, f = state.meadow, state.flock
    eye, head = f.pos[i], f.heading[i]
    L = sim.light(state.t)
    ang = head + COL_OFF  # (W,)
    dirv = sim.direction(ang)  # (W, 2)

    # --- ground: march outward along each column ---
    pts = eye[None, None, :] + DISTS[:, None, None] * dirv[None, :, :]  # (K, W, 2)
    inside = jnp.all((pts >= 0.0) & (pts < N_TILES), -1)
    wet = water_at(m.water, pts) & inside
    g = jnp.where(inside & ~wet, grass_at(m.grass, pts), 0.0)
    top = jnp.arctan2(GRASS_H * g / 3.0 - EYE, DISTS[:, None])  # (K, W) elevation of each sample's top
    reach = jax.lax.cummax(top, axis=0)
    idx = jnp.sum(reach[None, :, :] < ROW_ELEV[:, None, None], axis=1)  # (H, W) first sample in sight
    hit = idx < K
    kk = jnp.clip(idx, 0, K - 1)
    take = lambda a: jnp.take_along_axis(a, kk, axis=0)
    depth = jnp.where(hit, DISTS[kk], jnp.inf)
    g_px, wet_px, in_px = take(g), take(wet), take(inside)
    colour = jnp.where(wet_px[..., None], WATER_C, jnp.where(in_px[..., None], grass_colour(g_px), OUTSIDE_C))
    gpos = eye + depth[..., None] * dirv[None, :, :]  # (H, W, 2) where each ground pixel is

    # --- the fence on the meadow's edge ---
    dsafe = jnp.where(jnp.abs(dirv) < 1e-6, 1e-6, dirv)
    t_edges = jnp.stack([(0.0 - eye) / dsafe, (N_TILES - eye) / dsafe], 0)  # (2, W, 2)
    t_b = jnp.min(jnp.where(t_edges > 0, t_edges, jnp.inf), axis=(0, 2))  # (W,)
    bpt = eye + t_b[:, None] * dirv
    on_row_edge = (jnp.abs(bpt[:, 0]) < 1e-3) | (jnp.abs(bpt[:, 0] - N_TILES) < 1e-3)
    along = jnp.where(on_row_edge, bpt[:, 1], bpt[:, 0])  # position along the edge it meets
    post = jnp.mod(along, POST_EVERY) < 0.12
    y_at = EYE + t_b[None, :] * jnp.tan(ROW_ELEV[:, None])  # (H, W) height of each row's ray at the fence
    rail = ((y_at >= 0.18) & (y_at <= 0.24)) | ((y_at >= 0.32) & (y_at <= FENCE_H))
    fence = ((rail | (post[None, :] & (y_at >= 0.0) & (y_at <= POST_H))) & (t_b[None, :] < depth))
    colour = jnp.where(fence[..., None], WOOD, colour)
    depth = jnp.where(fence, t_b[None, :], depth)
    gpos = jnp.where(fence[..., None], bpt[None, :, :], gpos)

    # --- objects and other sheep, nearest first ---
    n_sh = f.pos.shape[0]
    pos = jnp.concatenate([m.obj_pos, f.pos])
    kind = jnp.concatenate([m.obj_kind, jnp.zeros((n_sh,), jnp.int32)])
    variant = jnp.concatenate([m.obj_variant, jnp.zeros((n_sh,), jnp.int32)])
    is_sheep = jnp.concatenate([jnp.zeros_like(m.obj_present), jnp.ones((n_sh,), bool)])
    present = jnp.concatenate([m.obj_present, (jnp.arange(n_sh) != i) & f.alive])
    rel = pos - eye
    dist = jnp.linalg.norm(rel, axis=-1)
    dsort = jnp.where(present & (dist > 0.05), dist, jnp.inf)
    _, near = jax.lax.top_k(-dsort, min(N_NEAREST, pos.shape[0]))
    rel, dist, kind, variant, is_sheep, present = rel[near], dist[near], kind[near], variant[near], is_sheep[near], present[near] & jnp.isfinite(dsort[near])
    bearing = jnp.arctan2(rel[:, 1], -rel[:, 0])
    other_heading = jnp.concatenate([jnp.zeros_like(m.obj_kind, jnp.float32), f.heading])[near]
    d = (other_heading - bearing + jnp.pi) % (2 * jnp.pi) - jnp.pi
    view = jnp.where(jnp.cos(d) > 0.7, 2, jnp.where(jnp.cos(d) < -0.7, 1, 0))  # 2 back (going away), 1 front, 0 side
    side = jnp.where(jnp.sin(d) > 0, 1.0, -1.0)
    dth = (ang[None, :] - bearing[:, None] + jnp.pi) % (2 * jnp.pi) - jnp.pi  # (J, W)
    ahead = jnp.abs(dth) < np.pi / 2
    x = jnp.where(ahead, dist[:, None] * jnp.tan(dth), 1e3)[:, None, :]  # (J, 1, W) across the billboard
    rng = jnp.where(ahead, dist[:, None] / jnp.maximum(jnp.cos(dth), 1e-3), jnp.inf)  # (J, W) range to its plane
    y = EYE + rng[:, None, :] * jnp.tan(ROW_ELEV)[None, :, None]  # (J, H, W) height on the billboard
    J = near.shape[0]
    bc = lambda a: a.reshape((J,) + (1,) * 2)
    obj_c = jnp.zeros((J, H, W, 3))
    obj_m = jnp.zeros((J, H, W), bool)
    obj_e = jnp.zeros((J, H, W), bool)
    for mask, col, emits in _layers(x, y, bc(kind), bc(variant), bc(view), bc(side), bc(is_sheep), 1.0 - L):
        mask = mask & bc(present)
        obj_c = jnp.where(mask[..., None], col, obj_c)
        obj_e = jnp.where(mask, emits, obj_e)
        obj_m = obj_m | mask
    odepth = jnp.where(obj_m, rng[:, None, :], jnp.inf)
    j = jnp.argmin(odepth, 0)  # (H, W)
    od = jnp.min(odepth, 0)
    front = od < depth
    o_col = jnp.take_along_axis(obj_c, j[None, :, :, None], 0)[0]
    o_pos = (eye + rel)[j]  # (H, W, 2)
    glow_px = front & jnp.take_along_axis(obj_e, j[None], 0)[0]
    colour = jnp.where(front[..., None], o_col, colour)
    depth = jnp.where(front, od, depth)
    gpos = jnp.where(front[..., None], o_pos, gpos)

    # --- light, the hut's light pool, haze, sky ---
    hut = m.obj_present & (m.obj_kind == HUT)
    d_hut = jnp.min(jnp.where(hut[None, None, :], jnp.linalg.norm(gpos[:, :, None, :] - m.obj_pos[None, None], axis=-1), jnp.inf), -1)
    lit = jnp.maximum(L, HUT_LIGHT * jnp.clip(1.0 - d_hut / HUT_LIGHT_R, 0.0, 1.0))
    colour = jnp.where(glow_px[..., None], colour, colour * lit[..., None] + (1.0 - L) * NIGHT_CAST)
    horizon = sky(jnp.zeros(()), L)
    fade = 1.0 - jnp.exp(-jnp.where(jnp.isfinite(depth), depth, 1e3) / HAZE)
    colour = colour + (horizon - colour) * fade[..., None]
    above = ~jnp.isfinite(depth)
    colour = jnp.where(above[..., None], sky(ROW_ELEV, L)[:, None, :], colour)
    return jnp.clip(colour, 0.0, 1.0)


def reticle_row(state: State, i: jax.Array) -> jax.Array:
    """Fractional row of the grass surface (or water) at sheep ``i``'s mouth point; its column is
    the centre, W / 2."""
    m, f = state.meadow, state.flock
    p = sim.mouth(f.pos[i : i + 1], f.heading[i : i + 1])
    on_water, _, inside = sim.mouth_tile(m, f.pos[i : i + 1], f.heading[i : i + 1])
    h = jnp.where(on_water | ~inside, 0.0, GRASS_H * grass_at(m.grass, p) / 3.0)[0]
    return row_of(jnp.arctan2(h - EYE, sim.MOUTH))
