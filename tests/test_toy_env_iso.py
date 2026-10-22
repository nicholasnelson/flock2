import jax
import jax.numpy as jnp
import numpy as np

from flock import iso, sprites_iso as S
from flock.palette import TRANSPARENT
from flock.toy import env_iso as env


def test_batch_shapes_and_invariants():
    ep = env.make_batch(jax.random.PRNGKey(0), batch=12, n_candidates=4, mode="bubble")
    assert ep.sender_view.shape == (12, iso.VIEW, iso.VIEW)
    assert ep.receiver_base.shape == (12, iso.VIEW, iso.VIEW)
    assert ep.target_mask.shape == (12, 16, 16)
    tm = np.asarray(ep.target_mask.sum((1, 2)))
    assert (tm >= 4).all() and (tm <= 9).all()
    ct = np.asarray(ep.candidate_types)
    assert (ct[:, 0] == np.asarray(ep.target_type)).all()
    assert all(len(set(row)) == 4 for row in ct)
    cm = np.asarray(ep.candidate_masks).astype(int)
    assert (cm.sum(1) <= 1).all()
    # canvas block clear of all candidate cells
    px = np.asarray(ep.canvas_px)
    for b in range(12):
        r0, c0 = px[b] // env.CELL
        k = env.CANVAS // env.CELL
        block = np.zeros((16, 16), bool)
        block[r0 : r0 + k, c0 : c0 + k] = True
        assert not (block & cm[b].any(0)).any(), b


def test_sender_view_shows_target_sprite_ahead():
    ep = env.make_batch(jax.random.PRNGKey(1), batch=8, n_candidates=4, mode="stamp")
    r, c = np.asarray(env.AHEAD_SPRITE_PX)
    for b in range(8):
        t = int(ep.target_type[b])
        patch = np.asarray(ep.sender_view[b, r : r + iso.BOX_H, c : c + iso.BOX_W])
        # one of the object's four facings (only the fox's differ) is drawn there exactly
        assert any((patch[spr != TRANSPARENT] == spr[spr != TRANSPARENT]).all() for spr in S.OBJECT_STACK[t]), b


def test_target_mask_covers_target_sprite_pixels():
    ep = env.make_batch(jax.random.PRNGKey(2), batch=8, n_candidates=3, mode="stamp")
    for b in range(8):
        t = int(ep.target_type[b])
        spr = S.OBJECT_STACK[t]
        colours = set(int(v) for v in spr[spr != TRANSPARENT]) - {15}  # every sprite has the outline colour
        v = np.asarray(ep.receiver_base[b])
        mask = np.asarray(ep.target_mask[b])
        # pixels under the target's cells must include sprite colours (the sprite is there)
        rows, cols = np.where(mask)
        k = env.CELL
        region = np.concatenate([v[r * k : r * k + k, c * k : c * k + k].ravel() for r, c in zip(rows, cols)])
        assert any(int(p) in colours for p in region), b


def test_probe_canvas_centres_on_ahead_sprite():
    p = np.asarray(env.PROBE_CANVAS_PX)
    a = np.asarray(env.AHEAD_SPRITE_PX)
    assert p[0] <= a[0] and p[0] + env.CANVAS >= a[0] + iso.BOX_H
    assert p[1] <= a[1] and p[1] + env.CANVAS >= a[1] + iso.BOX_W


def test_colour_dropout_views_differ_only_at_the_object():
    aug, clean, t = env.object_views_dropout(jax.random.PRNGKey(4), 32, 0.5)
    aug, clean = np.asarray(aug), np.asarray(clean)
    diff = aug != clean
    changed = diff.reshape(32, -1).any(-1)
    # roughly half the scenes are silhouetted; the rest are pixel-identical to the clean render
    assert 4 < changed.sum() < 28
    # every changed pixel became the outline colour, and it sits inside the ahead sprite box
    assert (aug[diff] == 15).all()
    r, c = map(int, env.AHEAD_SPRITE_PX)
    outside = diff.copy()
    outside[:, r : r + iso.BOX_H, c : c + iso.BOX_W] = False
    assert not outside.any()
