"""The skyline (spec section 21.7): fixed peaks on the horizon, a compass the sheep can learn, hidden by fog."""

import jax
import jax.numpy as jnp
import numpy as np

from flock.meadow2 import render as R
from flock.meadow2.world import DAY, MAX_DAYS, random_start


def _frames(skyline, t, heading, fog=False, seed=4):
    st = random_start(jax.random.PRNGKey(seed), willows=3, skyline=skyline)
    f = st.flock
    pos = f.pos.at[0].set(jnp.array([24.0, 24.0]))
    m = st.meadow._replace(fog_days=jnp.full((MAX_DAYS,), fog))
    st = st._replace(meadow=m, flock=f._replace(pos=pos, heading=f.heading.at[0].set(heading)), t=jnp.int32(t))
    rgb, _, depth = R.render_aux(st, 0)
    return np.asarray(rgb), np.asarray(depth), st


def _ridge_cols(rgb, depth, t, band=(0.0, 1.0)):
    """Columns whose sky pixels in the elevation band (degrees) differ from the sky colour of their row."""
    skycol = np.asarray(R.sky(R.ROW_ELEV, R.sim.light(jnp.int32(t))))  # (H, 3)
    rows = (np.asarray(R.ROW_ELEV) >= np.deg2rad(band[0])) & (np.asarray(R.ROW_ELEV) < np.deg2rad(band[1]))
    sky = ~np.isfinite(depth)
    dark = np.linalg.norm(rgb - skycol[:, None, :], axis=-1) > 0.02
    return np.any(sky[rows] & dark[rows], 0)


def test_skyline_profile_peaks_where_stated_and_wraps():
    ang = jnp.linspace(-np.pi, 3 * np.pi, 4001)
    e = np.asarray(R.skyline_elev(ang))
    for b, h, _ in R.PEAKS:
        at = np.asarray(R.skyline_elev(jnp.asarray([b, b + 2 * np.pi, b - 2 * np.pi])))
        assert np.allclose(at, np.deg2rad(h), atol=1e-3), (b, at)
    assert e.min() >= 0.0 and e.max() < np.deg2rad(7.0)
    # between the peaks the horizon is flat
    assert float(R.skyline_elev(jnp.asarray(2.0))) < np.deg2rad(0.1)


def test_skyline_is_off_by_default_and_shows_the_peak_ahead():
    t = int(0.3 * DAY)
    b = R.PEAKS[0][0]
    rgb0, d0, st0 = _frames(False, t, b)
    assert not bool(st0.meadow.skyline)
    assert not _ridge_cols(rgb0, d0, t).any()
    rgb1, d1, st1 = _frames(True, t, b)
    cols = _ridge_cols(rgb1, d1, t)
    assert cols[R.W // 2 - 3 : R.W // 2 + 3].sum() >= 4  # the peak is straight ahead (a sheep or a tree may cover a column)
    assert cols.sum() < R.W // 2  # and the horizon is flat elsewhere
    # the ground, the fence and the water are untouched: only sky pixels change
    changed = np.linalg.norm(rgb1 - rgb0, axis=-1) > 1e-4
    assert not changed[np.isfinite(d1)].any()


def test_turning_moves_the_peak_across_the_image():
    t = int(0.3 * DAY)
    b = R.PEAKS[0][0]
    turn = np.deg2rad(60.0)
    rgb_a, d_a, _ = _frames(True, t, b)
    rgb_b, d_b, _ = _frames(True, t, b + turn)
    # only the tallest peak reaches 5 degrees, so these columns are its alone
    ca, cb = _ridge_cols(rgb_a, d_a, t, (4.5, 5.5)), _ridge_cols(rgb_b, d_b, t, (4.5, 5.5))
    assert ca.any() and cb.any()
    ia, ib = np.flatnonzero(ca).mean(), np.flatnonzero(cb).mean()
    # a column covers FOV / W; turning left by 60 degrees moves the peak right by 60 / 300 of the width
    expect = 60.0 / 300.0 * R.W
    assert abs((ia - ib) - expect) < 3.0 or abs((ib - ia) - expect) < 3.0, (ia, ib, expect)


def test_skyline_reads_at_night_and_vanishes_in_fog():
    b = R.PEAKS[0][0]
    night = int(0.8 * DAY)
    rgb_n, d_n, _ = _frames(True, night, b)
    cols = _ridge_cols(rgb_n, d_n, night)
    assert cols[R.W // 2 - 3 : R.W // 2 + 3].sum() >= 4
    skycol = np.asarray(R.sky(R.ROW_ELEV, R.sim.light(jnp.int32(night))))
    rows = (np.asarray(R.ROW_ELEV) >= 0.0) & (np.asarray(R.ROW_ELEV) < np.deg2rad(1.0))
    contrast = np.abs(rgb_n[rows][:, R.W // 2] - skycol[rows]).max()
    assert contrast >= 0.04, contrast  # a silhouette the 8-bit view keeps
    rgb_f, d_f, _ = _frames(True, int(0.3 * DAY), b, fog=True)
    assert not _ridge_cols(rgb_f, d_f, int(0.3 * DAY)).any()
