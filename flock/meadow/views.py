"""Every sheep's first-person view of the meadow (``flock.fpv``).

Each viewer renders only its ``K_NEAREST`` nearest objects and other sheep (a fixed budget
whatever the meadow holds; farther things are beyond the view range anyway in all but the
emptiest meadows). Other sheep are billboards whose facing — front or back, left or right — is
how they move relative to the viewer's line of sight. Range: 12 tiles by day and 16 with the
head up; 5 at night and 8 with the head up; at night the sky and the far ground are dark.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from flock import fpv, iso, sprites_iso as S
from flock.meadow.sim import is_night
from flock.meadow.world import State, ground_state

K_NEAREST = 48
RANGE_DAY, RANGE_DAY_UP, RANGE_NIGHT, RANGE_NIGHT_UP = 12.0, 16.0, 5.0, 8.0
SKY_DAY, SKY_NIGHT = 5, 0  # water_light; void
FAR_DAY, FAR_NIGHT = 1, 0  # grass_dark; void

_N_VAR, _N_FACE = iso.OBJECT_STACK.shape[1], iso.OBJECT_STACK.shape[2]
_OBJ_FLAT = iso.OBJECT_STACK.reshape(-1, iso.BOX_H, iso.BOX_W)
_ANIMAL_OFFSET = _OBJ_FLAT.shape[0]
BILLBOARDS = jnp.concatenate([_OBJ_FLAT, iso.ANIMAL_STACK.reshape(-1, iso.BOX_H, iso.BOX_W)])


def sheep_facing(viewer: jax.Array, other_pos: jax.Array, other_heading: jax.Array) -> jax.Array:
    """Facing index (0 away/right, 1 toward/right, 2 toward/left, 3 away/left; see sprites_iso)
    of a sheep at ``other_pos`` with ``other_heading`` as seen from ``viewer``."""
    rel = other_pos - viewer
    bearing = jnp.arctan2(rel[..., 1], -rel[..., 0])
    d = (other_heading - bearing + jnp.pi) % (2 * jnp.pi) - jnp.pi
    away = jnp.abs(d) < jnp.pi / 2
    right = jnp.sin(d) > 0
    return jnp.where(away, jnp.where(right, 0, 3), jnp.where(right, 1, 2)).astype(jnp.int32)


def view_one(state: State, i: jax.Array, tiles: jax.Array) -> jax.Array:
    """(fpv.H, fpv.W) view of sheep ``i``."""
    m, f = state.meadow, state.flock
    eye = f.pos[i]
    n_sheep = f.pos.shape[0]
    # candidates: objects, then sheep (not the viewer)
    obj_idx = m.obj_kind * (_N_VAR * _N_FACE) + m.obj_variant * _N_FACE
    sh_idx = _ANIMAL_OFFSET + S.SHEEP * _N_FACE + sheep_facing(eye, f.pos, f.heading)
    pos = jnp.concatenate([m.obj_pos, f.pos])
    idx = jnp.concatenate([obj_idx, sh_idx])
    present = jnp.concatenate([m.obj_present, jnp.arange(n_sheep) != i])
    d = jnp.linalg.norm(pos - eye[None, :], axis=-1)
    d = jnp.where(present, d, jnp.inf)
    _, near = jax.lax.top_k(-d, min(K_NEAREST, pos.shape[0]))
    night = is_night(state.t)
    up = f.head_up[i]
    rng = jnp.where(night, jnp.where(up, RANGE_NIGHT_UP, RANGE_NIGHT), jnp.where(up, RANGE_DAY_UP, RANGE_DAY))
    return fpv.render_fpv(
        tiles,
        BILLBOARDS[idx[near]],
        pos[near],
        present[near],
        eye,
        f.heading[i],
        max_dist=rng,
        sky=jnp.where(night, SKY_NIGHT, SKY_DAY),
        far=jnp.where(night, FAR_NIGHT, FAR_DAY),
    )


def render_flock(state: State) -> jax.Array:
    """(S, fpv.H, fpv.W) uint8: every sheep's view."""
    tiles = ground_state(state.meadow)
    return jax.vmap(lambda i: view_one(state, i, tiles))(jnp.arange(state.flock.pos.shape[0]))
