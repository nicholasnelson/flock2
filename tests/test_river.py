"""The river meadow (spec section 21.8): one straight band of water through the centre at a random
angle, two halves a sheep cannot cross, willows along both banks, the flock on one side; the meadow's
side from ``FLOCK2_TILES``; the streamed distance tables equal the all-pairs ones."""

import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flock.meadow2 import world as W


def _components(dry):
    """Number of 4-connected components of a boolean mask (plain flood fill)."""
    dry = np.asarray(dry).copy()
    n = 0
    for i in range(dry.shape[0]):
        for j in range(dry.shape[1]):
            if dry[i, j]:
                n += 1
                stack = [(i, j)]
                dry[i, j] = False
                while stack:
                    a, b = stack.pop()
                    for c, d in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)):
                        if 0 <= c < dry.shape[0] and 0 <= d < dry.shape[1] and dry[c, d]:
                            dry[c, d] = False
                            stack.append((c, d))
    return n


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_river_splits_the_meadow(seed):
    st = W.generate(jax.random.PRNGKey(seed), willows=3, river=True)
    water = np.asarray(st.meadow.water)
    assert _components(~water) == 2, "the river must cut the meadow into exactly two dry halves"
    # about RIVER_WIDTH tiles wide along its length (the length is between N and N * sqrt 2)
    assert W.N_TILES * 2.5 <= water.sum() <= W.N_TILES * 1.5 * 3.6
    c, normal = W.river_geometry(st.meadow.water)
    assert jnp.all(jnp.abs(c - W.N_TILES / 2) <= W.RIVER_JITTER + 1.0)
    # every water tile centre lies within the band of the recovered line
    yy, xx = W._grid()
    dist = (yy - c[0]) * normal[0] + (xx - c[1]) * normal[1]
    assert float(jnp.max(jnp.where(st.meadow.water, jnp.abs(dist), 0.0))) < W.RIVER_WIDTH / 2 + 0.6
    # the flock starts on one side, 6-14 tiles from the water
    pos = st.flock.pos
    side = (pos[:, 0] - c[0]) * normal[0] + (pos[:, 1] - c[1]) * normal[1]
    assert jnp.all(side > 0) or jnp.all(side < 0)
    d = W.start_distance(st.meadow.water, pos.mean(0, keepdims=True))[0]
    assert W.START_WATER[0] - 1.0 <= float(d) <= W.START_WATER[1] + 1.0
    assert not jnp.any(W.water_at(st.meadow.water, pos))


def test_river_willows_line_the_banks():
    st = W.generate(jax.random.PRNGKey(5), willows=3, river=True)
    m = st.meadow
    wil = (m.obj_kind == W.WILLOW)
    assert int(wil.sum()) == 2 * W.N_RIVER_WILLOWS
    present = wil & m.obj_present
    assert int(present.sum()) >= W.N_TILES // W.RIVER_WILLOW_SPACING * 2 - 2  # at least a river's length of them on each bank
    d = W.start_distance(m.water, m.obj_pos[present])
    assert float(jnp.max(d)) < W.WILLOW_BANK + 1.5 and float(jnp.min(d)) > 0.3
    # absent willows (off the meadow) sit on the clipped edge and are not drawn; present ones are inside
    p = m.obj_pos[present]
    assert jnp.all((p >= 1.0) & (p <= W.N_TILES - 1.0))
    # no object stands in the river
    assert not jnp.any(m.water[m.obj_pos[m.obj_present][:, 0].astype(int), m.obj_pos[m.obj_present][:, 1].astype(int)])


def test_ponds_unchanged_without_river():
    a = W.generate(jax.random.PRNGKey(7), willows=3)
    assert _components(~np.asarray(a.meadow.water)) == 1
    assert int((a.meadow.obj_kind == W.WILLOW).sum()) == W.N_PONDS * 3


def test_streamed_distances_match_all_pairs():
    st = W.generate(jax.random.PRNGKey(3), river=True)
    water = st.meadow.water
    yy, xx = W._grid()
    tiles = jnp.stack([yy.ravel(), xx.ravel()], -1)
    full = jnp.linalg.norm(tiles[:, None, :] - tiles[None, :, :], axis=-1)
    ref = jnp.min(jnp.where(water.ravel()[None, :], full, jnp.inf), -1).reshape(W.N_TILES, W.N_TILES)
    assert jnp.allclose(W.water_distance(water), ref)
    cells = W.cell_centres()
    ref_c = jnp.min(jnp.where(water.ravel()[None, :], jnp.linalg.norm(cells.reshape(-1, 1, 2) - tiles[None], axis=-1), jnp.inf), -1).reshape(W.N_CELLS, W.N_CELLS)
    assert jnp.allclose(W.cell_water_distance(water), ref_c)


def test_random_start_river_and_terrain():
    st = W.random_start(jax.random.PRNGKey(0), willows=3, river=True, skyline=True)
    st = W.with_terrain(st, bank=16.0, mud=True, inner=8.0)
    rm = st.meadow.regrow_map
    d = W.cell_water_distance(st.meadow.water)
    assert jnp.all(rm[d < 7.5] == 0.0) and jnp.all(rm[d > 16.5] == 1.0)
    assert bool(st.meadow.skyline)


def test_size_from_environment():
    code = (
        "import jax, numpy as np; from flock.meadow2 import world as W, render as R, sim; "
        "assert W.N_TILES == 72 and W.N_CELLS == 144, (W.N_TILES, W.N_CELLS); "
        "st = W.random_start(jax.random.PRNGKey(0), willows=3, river=True); "
        "assert st.meadow.water.shape == (72, 72) and st.meadow.grass.shape == (144, 144); "
        "st = W.with_terrain(st, bank=16.0, mud=True, inner=8.0); "
        "st2 = sim.step(st, np.zeros((W.N_SHEEP,), dtype=np.int32)); assert st2.flock.pos.shape == (W.N_SHEEP, 2); "
        "print('ok', int(st.meadow.water.sum()))"
    )
    env = dict(os.environ, FLOCK2_TILES="72", JAX_PLATFORMS="cpu")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.startswith("ok")
