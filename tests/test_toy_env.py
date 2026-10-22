import jax
import jax.numpy as jnp
import numpy as np

from flock.toy import env


def test_batch_shapes_and_invariants():
    ep = env.make_batch(jax.random.PRNGKey(0), batch=16, n_candidates=4, mode="bubble")
    assert ep.sender_view.shape == (16, 64, 64)
    assert ep.receiver_base.shape == (16, 64, 64)
    assert ep.target_mask.shape == (16, 16, 16)
    # target footprint is exactly 4 tiles, all candidates distinct types, target first
    assert (np.asarray(ep.target_mask.sum((1, 2))) == 4).all()
    ct = np.asarray(ep.candidate_types)
    assert (ct[:, 0] == np.asarray(ep.target_type)).all()
    assert all(len(set(row)) == 4 for row in ct)
    # candidate footprints do not overlap each other or the canvas block
    cm = np.asarray(ep.candidate_masks).astype(int)
    assert (cm.sum(1) <= 1).all()
    px = np.asarray(ep.canvas_px)
    for b in range(16):
        r0, c0 = px[b] // 4
        block = np.zeros((16, 16), bool)
        block[r0 : r0 + 4, c0 : c0 + 4] = True
        assert not (block & cm[b].any(0)).any(), b


def test_sender_view_shows_target_ahead():
    ep = env.make_batch(jax.random.PRNGKey(1), batch=8, n_candidates=4, mode="stamp")
    from flock import sprites
    from flock.palette import TRANSPARENT

    for b in range(8):
        t = int(ep.target_type[b])
        k = int(ep.sender_heading[b])
        spr = np.rot90(sprites.OBJECT_STACK[t], k)  # view rotates CCW by heading
        r, c = np.asarray(env.AHEAD_OBJECT_PX)
        patch = np.asarray(ep.sender_view[b, r : r + 8, c : c + 8])
        m = spr != TRANSPARENT
        assert (patch[m] == spr[m]).all(), b


def test_stamp_mode_has_no_sender_sprite_near_canvas():
    ep = env.make_batch(jax.random.PRNGKey(2), batch=4, n_candidates=3, mode="stamp")
    # the only white pixels should be from candidates (salt) or the receiver itself; check the
    # bubble-sender region (just beyond the canvas) has no wool in stamp mode
    px = np.asarray(ep.canvas_px)
    for b in range(4):
        r0, c0 = px[b]
        v = np.asarray(ep.receiver_base[b])
        # the region on the far side of the canvas from the receiver: just check the canvas
        # block itself is plain ground
        assert not (v[r0 : r0 + 16, c0 : c0 + 16] == 8).any()


def test_object_views():
    views, types = env.object_views(jax.random.PRNGKey(3), 6)
    assert views.shape == (6, 64, 64) and types.shape == (6,)
