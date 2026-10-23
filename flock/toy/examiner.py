"""An independent examiner for drawings: a classifier that knows the world but never the game.

The judge probe (``metrics.iconicity``) measures resemblance with the same frozen encoder whose
embeddings drive the resemblance loss, and every run pre-trains its own judge: the sender is
trained against the network that grades it, and scores are not comparable across runs. The
examiner fixes both. It is a supervised CNN trained once per world (sprite set + view size) on
labelled views of every referent variant — training and held-out alike, since the examiner is
not an agent — at all sub-tile offsets and scales, a fifth as solid outline-colour silhouettes
and nearly a third as 2 px outline drawings of the sprites (people have a lifetime of outline
drawings; without them the examiner read every line drawing as the one long, low sprite),
then frozen and cached on disk. It never sees a drawing in training. Grading a
run asks: can a model at our scale, knowing only the world's sprites, tell which referent a
drawing stands for?
"""

from __future__ import annotations

import hashlib
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from flock import iso, sprites_iso as S, world
from flock.nets import EMBED, Encoder
from flock.toy import env_iso, game

CACHE_DIR = os.path.join("runs", "examiner")
P_SILHOUETTE = 0.2
P_OUTLINE = 0.3  # outline drawings of the sprites: the line-drawing literacy people have
VERSION = "v2-outline"  # part of the cache key: bump when the training recipe changes
SCALE_MIN = 0.5
SHIFT = 6


class Examiner(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, width: int = 32, view: int = iso.VIEW, n_types: int = S.N_OBJECT_TYPES):
        self.encoder = Encoder(rngs, width, view)
        self.head = nnx.Linear(EMBED, n_types, rngs=rngs)

    def __call__(self, x_onehot: jax.Array) -> jax.Array:
        emb, _ = self.encoder(x_onehot)
        return self.head(emb)


def world_signature() -> str:
    """Changes whenever the sprites, ground or view size change (then the examiner is retrained)."""
    h = hashlib.sha256()
    for a in (S.OBJECT_STACK, S.GROUND_STACK, np.array([iso.VIEW, S.HELD_OUT])):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(VERSION.encode())
    return h.hexdigest()[:16]


@jax.jit
def _labelled_views(key: jax.Array) -> tuple[jax.Array, jax.Array]:
    """128 labelled single-object views: half training variants, half held-out; offsets, scales
    0.5-1; some silhouettes, some outline drawings; then a random shift."""
    n = 128
    k1, k2, k3, k4 = jax.random.split(key, 4)
    u = jax.random.uniform(k3, (n,))
    sil, ol = u < P_SILHOUETTE, (u >= P_SILHOUETTE) & (u < P_SILHOUETTE + P_OUTLINE)
    one = lambda held: (lambda k, s, o: env_iso._object_view(k, s, True, SCALE_MIN, None, True, held, o))
    v1, t1 = jax.vmap(one(False))(jax.random.split(k1, n // 2), sil[: n // 2], ol[: n // 2])
    v2, t2 = jax.vmap(one(True))(jax.random.split(k2, n // 2), sil[n // 2 :], ol[n // 2 :])
    v, t = jnp.concatenate([v1, v2]), jnp.concatenate([t1, t2])
    d = jax.random.randint(k4, (n, 2), -SHIFT, SHIFT + 1)
    v = jax.vmap(lambda im, dd: jnp.roll(im, (dd[0], dd[1]), axis=(0, 1)))(v, d)
    return v, t


@nnx.jit
def _train_step(model: Examiner, opt: nnx.Optimizer, views: jax.Array, types: jax.Array):
    def loss_fn(m):
        logits = m(world.to_onehot(views))
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, types)), jnp.mean(jnp.argmax(logits, -1) == types)

    (loss, acc), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    opt.update(model, grads)
    return loss, acc


def get_examiner(steps: int = 3000, seed: int = 20261023, log=print) -> Examiner:
    """The cached examiner for the current world, training (and caching) it if needed."""
    path = os.path.join(CACHE_DIR, f"examiner_{world_signature()}.pkl")
    model = Examiner(nnx.Rngs(seed))
    if os.path.exists(path):
        with open(path, "rb") as f:
            pure = pickle.load(f)
        state = nnx.state(model)
        nnx.replace_by_pure_dict(state, pure)
        nnx.update(model, state)
        return model
    opt = nnx.Optimizer(model, optax.adamw(3e-4), wrt=nnx.Param)
    key = jax.random.PRNGKey(seed)
    t0 = time.time()
    for step in range(steps):
        key, k = jax.random.split(key)
        loss, acc = _train_step(model, opt, *_labelled_views(k))
        if step % 500 == 0 or step == steps - 1:
            log({"phase": "examiner", "step": step, "loss": float(loss), "acc": float(acc), "t": time.time() - t0})
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(jax.tree.map(np.asarray, nnx.to_pure_dict(nnx.state(model))), f)
    os.replace(tmp, path)
    return model


def _classify(model: Examiner, views: jax.Array, chunk: int = 256, n_types: int | None = None) -> np.ndarray:
    """Predicted type per view; with ``n_types`` the choice is among the first n types only
    (the run's referent set), so chance is 1/n."""
    out = []
    for i in range(0, views.shape[0], chunk):
        logits = model(world.to_onehot(views[i : i + chunk]))
        if n_types is not None:
            logits = logits[:, :n_types]
        out.append(np.asarray(jnp.argmax(logits, -1)))
    return np.concatenate(out)


def examine(model: Examiner, alpha: np.ndarray, types: np.ndarray, key: jax.Array, ink_colour: int = 15, n_types: int | None = None) -> dict:
    """Grade drawings (alpha (N, C, C), black ink) against their referent types. Each drawing is
    pasted on plain ground where the referent would stand, as drawn and re-centred on its ink.
    Returns accuracies, the confusion matrix (rows true, columns guessed) and, as a check on the
    examiner itself, its accuracy on real held-out sprite views."""
    n = alpha.shape[0]
    a = jnp.asarray(alpha, jnp.float32)
    colour = jnp.broadcast_to(jax.nn.one_hot(ink_colour, 16), a.shape + (16,))
    canvas = game.Canvas(colour=colour, alpha=a, ink=jnp.zeros((n,)))
    centred = game.transmit(canvas, key, centre=True)
    ground = world.to_onehot(jax.vmap(env_iso.plain_ground_view)(jax.random.split(jax.random.fold_in(key, 1), n)))
    paste = lambda c: jax.vmap(lambda v, cc, aa: jnp.argmax(world.composite_canvas(v, cc, aa, env_iso.PROBE_CANVAS_PX, opaque_paper=False), -1))(ground, c.colour, c.alpha)
    t = np.asarray(types)
    out = {}
    for name, c in (("as_drawn", canvas), ("centred", centred)):
        pred = _classify(model, paste(c).astype(jnp.uint8), n_types=n_types)
        out[f"examiner_acc_{name}"] = float((pred == t).mean())
        k = n_types or S.N_OBJECT_TYPES
        cm = np.zeros((k, k), int)
        np.add.at(cm, (t, pred), 1)
        out[f"examiner_confusion_{name}"] = cm.tolist()
    real, rt = _labelled_views(jax.random.fold_in(key, 2))
    out["examiner_acc_real_views"] = float((_classify(model, real) == np.asarray(rt)).mean())
    ov, ot = _outline_views(jax.random.fold_in(key, 3))
    out["examiner_acc_outline_views"] = float((_classify(model, ov) == np.asarray(ot)).mean())
    return out


@jax.jit
def _outline_views(key: jax.Array) -> tuple[jax.Array, jax.Array]:
    """128 outline drawings of real sprites (held-out variants): the examiner's own reading test."""
    n = 128
    one = lambda k: env_iso._object_view(k, jnp.bool_(False), True, SCALE_MIN, None, True, True, jnp.bool_(True))
    return jax.vmap(one)(jax.random.split(key, n))
