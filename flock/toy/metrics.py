"""Evaluation for Milestone 0.

Three families of measurement, matching the success criteria in the plan:

- *Used*: accuracy, plus a causal test — accuracy and the receiver's tile distribution when the
  canvas is blanked or swapped with another episode's. "CIC" here is the mean KL between the
  receiver's distribution with the real canvas and with a blank one.
- *Iconic*: a frozen "judge" encoder (the pre-trained perception encoder, never trained on the
  game) with a linear object-type probe fitted on real object views; applied to drawings stamped
  on plain ground where the object would be. If the probe transfers, the drawing looks like the
  object *as perception sees it*. Also feature-similarity ranking against per-type reference
  embeddings.
- *Readable*: left to humans; see ``gallery.py``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from flock import world
from flock.nets import EMBED, Agents, Encoder
from flock.toy import envs, game


class TypeProbe(NamedTuple):
    w: jax.Array  # (256, N_TYPES)
    b: jax.Array  # (N_TYPES,)
    refs: jax.Array  # (N_TYPES, 256) mean embedding per type


def _embed(enc: Encoder, views_onehot: jax.Array, chunk: int = 256) -> jax.Array:
    outs = []
    for i in range(0, views_onehot.shape[0], chunk):
        e, _ = enc(views_onehot[i : i + chunk])
        outs.append(e)
    return jnp.concatenate(outs, axis=0)


def fit_type_probe(judge: Encoder, key: jax.Array, E, n: int = 2048, steps: int = 300, lr: float = 0.1) -> TypeProbe:
    """Logistic-regression probe on judge embeddings of real single-object views.

    ``E`` is the episode module (``flock.toy.envs.get(cfg)``)."""
    n_types = E.N_TYPES
    views, types = E.object_views(key, n)
    x = jax.lax.stop_gradient(_embed(judge, world.to_onehot(views)))
    y = types
    refs = jnp.stack([jnp.mean(x[y == t], axis=0) for t in range(n_types)])
    w = jnp.zeros((EMBED, n_types))
    b = jnp.zeros((n_types,))

    def loss(params):
        w_, b_ = params
        logits = x @ w_ + b_
        return jnp.mean(-jax.nn.log_softmax(logits)[jnp.arange(n), y]) + 1e-4 * jnp.sum(w_ ** 2)

    g = jax.jit(jax.grad(loss))
    for _ in range(steps):
        gw, gb = g((w, b))
        w, b = w - lr * gw, b - lr * gb
    return TypeProbe(w=w, b=b, refs=refs)


def probe_views_of_drawings(canvas, ground_views_onehot: jax.Array, probe_px: jax.Array) -> jax.Array:
    """Drawings stamped on plain ground at the position the object occupies in a sender view."""
    return jax.vmap(lambda v, c, a: world.composite_canvas(v, c, a, probe_px, opaque_paper=False))(ground_views_onehot, canvas.colour, canvas.alpha)


def iconicity(enc: Encoder, probe: TypeProbe, canvas, target_type: jax.Array, ground_onehot: jax.Array, probe_px: jax.Array) -> dict:
    n_types = probe.refs.shape[0]
    pv = probe_views_of_drawings(canvas, ground_onehot, probe_px)
    e = _embed(enc, pv)
    logits = e @ probe.w + probe.b
    probe_acc = jnp.mean(jnp.argmax(logits, -1) == target_type)
    n = lambda x: x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)
    sims = n(e) @ n(probe.refs).T  # (B, N_TYPES)
    rank_acc = jnp.mean(jnp.argmax(sims, -1) == target_type)
    true_sim = sims[jnp.arange(sims.shape[0]), target_type]
    other = jnp.where(jax.nn.one_hot(target_type, n_types) > 0, -jnp.inf, sims).max(-1)
    return {
        "probe_acc": float(probe_acc),
        "rank_acc": float(rank_acc),
        "sim_margin": float(jnp.mean(true_sim - other)),
    }


def evaluate(model: Agents, judge: Encoder, probe: TypeProbe, cfg, key: jax.Array) -> dict:
    E = envs.get(cfg)
    k_ep, k_g, k_s = jax.random.split(key, 3)
    ep = E.make_batch(k_ep, cfg.eval_batch, cfg.n_candidates, cfg.mode, cfg.random_heading)
    ground = world.to_onehot(jax.vmap(E.plain_ground_view)(jax.random.split(k_g, cfg.eval_batch)))
    opaque = cfg.opaque

    fwd = game.forward(model, ep, k_s, opaque, hard=True)
    ce, acc = game.success_loss(fwd.logits, ep.target_mask)
    b = cfg.eval_batch
    p_real = jax.nn.softmax(fwd.logits.reshape(b, -1), -1)

    # causal tests
    blank = game.blank_canvas(b)
    logits_blank, _ = game.receive(model, ep.receiver_base, blank, ep.canvas_px, opaque)
    _, acc_blank = game.success_loss(logits_blank, ep.target_mask)
    p_blank = jax.nn.softmax(logits_blank.reshape(b, -1), -1)
    kl = jnp.sum(p_real * (jnp.log(p_real + 1e-9) - jnp.log(p_blank + 1e-9)), -1)
    swapped = jax.tree_util.tree_map(lambda x: jnp.roll(x, 1, axis=0), fwd.canvas)
    logits_swap, _ = game.receive(model, ep.receiver_base, swapped, ep.canvas_px, opaque)
    _, acc_swap = game.success_loss(logits_swap, ep.target_mask)
    # swapped canvases whose referent happens to match are not a fair test; exclude them
    same = jnp.roll(ep.target_type, 1) == ep.target_type
    acc_swap = jnp.sum(acc_swap * (~same)) / jnp.maximum(jnp.sum(~same), 1)

    out = {
        "acc": float(jnp.mean(acc)),
        "ce": float(jnp.mean(ce)),
        "acc_blank": float(jnp.mean(acc_blank)),
        "acc_swapped": float(acc_swap),
        "cic_kl": float(jnp.mean(kl)),
        "ink": float(jnp.mean(fwd.canvas.ink)),
        "alpha_mean": float(jnp.mean(fwd.canvas.alpha)),
        "chance": 1.0 / cfg.n_candidates,
    }
    icon_judge = iconicity(judge, probe, fwd.canvas, ep.target_type, ground, E.PROBE_CANVAS_PX)
    out.update({f"judge_{k}": v for k, v in icon_judge.items()})
    # the model's own encoder as a (less honest) judge, with a probe fitted on it
    own_probe = fit_type_probe(model.encoder, jax.random.PRNGKey(123), E, n=1024, steps=200)
    icon_own = iconicity(model.encoder, own_probe, fwd.canvas, ep.target_type, ground, E.PROBE_CANVAS_PX)
    out.update({f"own_{k}": v for k, v in icon_own.items()})
    # per-type accuracy for the gallery / analysis
    tt = np.asarray(ep.target_type)
    a = np.asarray(acc)
    out["acc_by_type"] = [float(a[tt == t].mean()) if (tt == t).any() else None for t in range(E.N_TYPES)]
    return out
