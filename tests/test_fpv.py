import jax.numpy as jnp
import numpy as np

from flock import fpv, sprites_iso as S
from flock.palette import TRANSPARENT


def _scene(dist: float, behind: bool = False):
    tm = jnp.full((24, 24), 1, jnp.int32)  # short grass
    eye = jnp.array([12.0, 12.0])
    d = dist if not behind else -dist
    spos = jnp.array([[12.0 - d, 12.0]])  # straight ahead (north) or straight behind
    tree = fpv.object_billboards(jnp.array([1]), jnp.array([0]), jnp.array([0]))
    return fpv.render_fpv(tm, tree, spos, jnp.array([True]), eye, jnp.float32(0.0))


def test_sky_ground_and_a_tree_ahead():
    v = np.asarray(_scene(4.0))
    assert v.shape == (fpv.H, fpv.W)
    assert (v[: fpv.HORIZON - 12] == fpv.SKY).mean() > 0.9  # sky well above the horizon
    tree_cols = np.where((v == 6).any(0))[0]  # bark: only the tree's trunk uses it
    assert len(tree_cols) > 0
    assert abs(tree_cols.mean() - fpv.W / 2) < 3  # centred
    rows = np.where((v == 6).any(1))[0]
    assert rows.max() > fpv.HORIZON  # stands on the ground, below the horizon


def test_nearer_is_bigger_and_behind_is_hidden():
    near = np.asarray(_scene(2.0))
    far = np.asarray(_scene(6.0))
    size = lambda v: ((v != fpv.SKY) & ~np.isin(v, [1, 2, 3, 7])).sum()  # non-ground, non-sky pixels
    assert size(near) > 4 * size(far) > 0
    assert (np.asarray(_scene(4.0, behind=True)) == 6).sum() == 0  # in the blind wedge
