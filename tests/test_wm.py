import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from flock import wm


def test_shapes_loss_and_gradients():
    m = wm.WorldModel(nnx.Rngs(0))
    n = wm.count_params(m)
    assert 2e6 < n < 8e6, n
    views = jax.random.randint(jax.random.PRNGKey(0), (2, 5, wm.H, wm.W), 0, 16)
    acts = jax.random.randint(jax.random.PRNGKey(1), (2, 5), 0, wm.N_ACTIONS)
    (loss, aux), grads = nnx.value_and_grad(wm.loss_fn, has_aux=True)(m, views, acts, jax.random.PRNGKey(2))
    assert np.isfinite(float(loss))
    leaves = jax.tree.leaves(nnx.state(grads, nnx.Param))
    assert all(np.isfinite(np.asarray(g)).all() for g in leaves)
    assert sum(float(jnp.abs(g).sum()) > 0 for g in leaves) > 0.9 * len(leaves)
    s = m.observe(views, acts, jax.random.PRNGKey(3))
    assert s.h.shape == (2, 5, wm.DETER) and s.z.shape == (2, 5, wm.STOCH * wm.CLASSES)
    im = m.imagine(s.h[:, -1], s.z[:, -1], acts[:, :3], jax.random.PRNGKey(4))
    assert m.decode(im).shape == (2, 3, wm.H, wm.W, 16)


def test_learns_a_constant_strip_quickly():
    import optax

    m = wm.WorldModel(nnx.Rngs(1))
    opt = nnx.Optimizer(m, optax.adam(1e-3), wrt=nnx.Param)
    views = jnp.broadcast_to(jnp.where(jnp.arange(wm.H)[:, None] < 24, 5, 1), (2, 4, wm.H, wm.W)).astype(jnp.int32)
    acts = jnp.zeros((2, 4), jnp.int32)

    @nnx.jit
    def step(m, opt, k):
        (l, aux), g = nnx.value_and_grad(wm.loss_fn, has_aux=True)(m, views, acts, k)
        opt.update(m, g)
        return aux

    for i in range(60):
        aux = step(m, opt, jax.random.PRNGKey(i))
    assert float(aux["pixel_acc"]) > 0.99


def test_interoception_and_loading_a_vision_only_checkpoint():
    import pickle

    old = wm.WorldModel(nnx.Rngs(5))
    pure = jax.tree.map(np.asarray, nnx.to_pure_dict(nnx.state(old)))
    new = wm.WorldModel(nnx.Rngs(6), n_intero=5)
    wm.load_into(new, pickle.loads(pickle.dumps(pure)))
    a = np.asarray(old.encoder.out.kernel[...])
    b = np.asarray(new.encoder.out.kernel[...])
    assert np.allclose(a, b)  # shared parts loaded
    views = jax.random.randint(jax.random.PRNGKey(0), (2, 4, wm.H, wm.W), 0, 16)
    acts = jax.random.randint(jax.random.PRNGKey(1), (2, 4), 0, wm.N_ACTIONS)
    intero = jax.random.uniform(jax.random.PRNGKey(2), (2, 4, 5))
    loss, aux = wm.loss_fn(new, views, acts, jax.random.PRNGKey(3), intero)
    assert np.isfinite(float(loss)) and "intero_mse" in aux
    s = new.observe(views, acts, jax.random.PRNGKey(4), intero)
    assert new.decode_intero(s).shape == (2, 4, 5)


def test_twohot_round_trips_through_the_bins():
    y = jnp.array([-73.0, -0.08, 0.0, 0.24, 3.2, 71.8])
    t = wm.twohot(wm.symlog(y))
    assert np.allclose(np.asarray(t.sum(-1)), 1.0, atol=1e-5)
    back = wm.symexp(jnp.sum(t * wm.BINS, -1))
    assert np.allclose(np.asarray(back), np.asarray(y), rtol=1e-4, atol=1e-5)


def test_reward_head_starts_at_zero_and_trains():
    m = wm.WorldModel(nnx.Rngs(7), n_intero=5, reward=True)
    views = jax.random.randint(jax.random.PRNGKey(0), (2, 4, wm.H, wm.W), 0, 16)
    acts = jax.random.randint(jax.random.PRNGKey(1), (2, 4), 0, wm.N_ACTIONS)
    intero = jax.random.uniform(jax.random.PRNGKey(2), (2, 4, 5))
    reward = jnp.full((2, 4), 5.0)
    s = m.observe(views, acts, jax.random.PRNGKey(3), intero)
    assert np.allclose(np.asarray(m.reward(jnp.concatenate([s.h, s.z], -1))), 0.0, atol=1e-5)
    (loss, aux), g = nnx.value_and_grad(wm.loss_fn, has_aux=True)(m, views, acts, jax.random.PRNGKey(3), intero, reward)
    assert np.isfinite(float(aux["reward_ce"]))
    assert float(jnp.abs(g.reward_head.b.kernel[...]).sum()) > 0


def test_ensemble_learns_and_disagreement_falls_where_trained():
    import optax

    m = wm.WorldModel(nnx.Rngs(8), n_intero=5, reward=True, ensemble=3, ensemble_target="z")
    feat = jax.random.normal(jax.random.PRNGKey(0), (16, wm.DETER + wm.STOCH * wm.CLASSES))
    a_seen = jax.nn.one_hot(jnp.zeros(16, jnp.int32), wm.N_ACTIONS)
    a_new = jax.nn.one_hot(jnp.full(16, 5), wm.N_ACTIONS)
    target = jax.nn.one_hot(jax.random.randint(jax.random.PRNGKey(1), (16, wm.STOCH), 0, wm.CLASSES), wm.CLASSES)
    opt = nnx.Optimizer(m, optax.adam(1e-3), wrt=nnx.Param)

    @nnx.jit
    def step(m, opt):
        def loss(m):
            return -jnp.mean(jnp.sum(target[None] * jax.nn.log_softmax(m.ensemble_logits(feat, a_seen), -1), -1))

        l, g = nnx.value_and_grad(loss)(m)
        opt.update(m, g)
        return l

    before = float(m.disagreement(feat, a_seen).mean())
    for _ in range(150):
        step(m, opt)
    trained, untrained = float(m.disagreement(feat, a_seen).mean()), float(m.disagreement(feat, a_new).mean())
    assert trained < before
    assert untrained > 2 * trained  # the action it never saw stays uncertain


def test_ensemble_loss_is_in_the_world_model_loss():
    views = jax.random.randint(jax.random.PRNGKey(0), (2, 4, wm.H, wm.W), 0, 16)
    acts = jax.random.randint(jax.random.PRNGKey(1), (2, 4), 0, wm.N_ACTIONS)
    intero = jax.random.uniform(jax.random.PRNGKey(2), (2, 4, 5))
    for target, key in (("z", "ensemble_ce"), ("body", "ensemble_mse")):
        m = wm.WorldModel(nnx.Rngs(9), n_intero=5, reward=True, ensemble=2, ensemble_target=target)
        (loss, aux), g = nnx.value_and_grad(wm.loss_fn, has_aux=True)(m, views, acts, jax.random.PRNGKey(3), intero, jnp.zeros((2, 4)))
        assert np.isfinite(float(aux[key]))
        assert float(jnp.abs(g.ensemble[0].b.kernel[...]).sum()) > 0
    feat = jax.random.normal(jax.random.PRNGKey(4), (3, wm.DETER + wm.STOCH * wm.CLASSES))
    assert m.ensemble_predict(feat, jax.nn.one_hot(jnp.zeros(3, jnp.int32), wm.N_ACTIONS)).shape == (2, 3, len(wm.BODY_DIMS))
    assert m.disagreement(feat, jax.nn.one_hot(jnp.zeros(3, jnp.int32), wm.N_ACTIONS)).shape == (3,)


def test_class_balanced_reconstruction_keeps_scale_and_upweights_rare_classes():
    m = wm.WorldModel(nnx.Rngs(3), cfg=wm.MEADOW._replace(recon_balance=10.0))
    plain = wm.WorldModel(nnx.Rngs(3))
    views = jnp.zeros((1, 2, wm.H, wm.W), jnp.int32).at[..., :2, :2].set(4)  # a little water in a sea of class 0
    acts = jnp.zeros((1, 2), jnp.int32)
    _, a1 = wm.loss_fn(m, views, acts, jax.random.PRNGKey(0))
    _, a0 = wm.loss_fn(plain, views, acts, jax.random.PRNGKey(0))
    assert np.isfinite(float(a1["recon"])) and 0.2 < float(a1["recon"]) / float(a0["recon"]) < 5.0


def test_observe_from_a_stored_start_reproduces_the_online_state():
    """Meadow v2 spec section 21.12, arm W: a sequence observed from the recorded (h, z, previous action)
    continues the online run exactly (the deterministic state and the posterior at its first step equal
    the full run's), where a zero start does not; the loss accepts the start."""
    m = wm.WorldModel(nnx.Rngs(11))
    views = jax.random.randint(jax.random.PRNGKey(0), (2, 8, wm.H, wm.W), 0, 16)
    acts = jax.random.randint(jax.random.PRNGKey(1), (2, 8), 0, wm.N_ACTIONS)
    full = m.observe(views, acts, jax.random.PRNGKey(3))
    L = 4
    start = (full.h[:, L - 1], full.z[:, L - 1], acts[:, L - 1])
    half = m.observe(views[:, L:], acts[:, L:], jax.random.PRNGKey(9), start=start)
    assert np.allclose(np.asarray(half.h[:, 0]), np.asarray(full.h[:, L]), atol=1e-5)
    assert np.allclose(np.asarray(half.post[:, 0]), np.asarray(full.post[:, L]), atol=1e-4)
    zeros = m.observe(views[:, L:], acts[:, L:], jax.random.PRNGKey(9))
    assert not np.allclose(np.asarray(zeros.h[:, 0]), np.asarray(full.h[:, L]), atol=1e-3)
    (loss, _), grads = nnx.value_and_grad(wm.loss_fn, has_aux=True)(m, views[:, L:], acts[:, L:], jax.random.PRNGKey(2), None, None, start)
    assert np.isfinite(float(loss))
    assert all(np.isfinite(np.asarray(g)).all() for g in jax.tree.leaves(nnx.state(grads, nnx.Param)))
