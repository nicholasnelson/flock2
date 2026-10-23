#!/usr/bin/env python
"""Pretrain the world model (``flock.wm``) on logged scripted-flock rollouts (Milestone 1, step 2).

    python scripts/m1_wm_train.py --data ~/flock2_data/m1_rollouts --out ~/flock2_data/m1_wm/run1 --steps 20000

The last shard is held out. Every ``--eval_every`` steps an open-loop test: the model observes 16
strips of held-out sequences, then imagines 48 more from the true actions alone; reports pixel
accuracy and recall of object pixels (sheep, trees, huts, flowers, mushrooms, the salt lick) by
horizon, and writes expected-vs-actual strip images.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from PIL import Image

from flock import wm
from flock.meadow.rollouts import unpack
from flock.palette import RGB

OBJECT_COLOURS = np.array([6, 8, 9, 10, 11, 12, 13, 14, 15])  # bark, white, stones, fox, red, yellow, pink, outline
CONTEXT, HORIZON = 16, 48


def load_shards(data: str):
    names = sorted(glob.glob(os.path.join(data, "shard_*.views.npy")))
    return [(np.load(n, mmap_mode="r"), np.load(n.replace(".views.", ".actions."), mmap_mode="r")) for n in names]


def sample(shards, rng, B, L):
    vs, acts = [], []
    for _ in range(B):
        v, a = shards[rng.integers(len(shards))]
        n, o = rng.integers(v.shape[0]), rng.integers(v.shape[1] - L + 1)
        vs.append(v[n, o : o + L])
        acts.append(a[n, o : o + L])
    return np.stack(vs), np.stack(acts).astype(np.int32)


@nnx.jit
def train_step(model, opt, packed, actions, key):
    views = unpack(packed).astype(jnp.int32)
    (loss, aux), grads = nnx.value_and_grad(wm.loss_fn, has_aux=True)(model, views, actions, key)
    opt.update(model, grads)
    return aux


@nnx.jit
def open_loop(model, packed, actions, key):
    """Observe CONTEXT strips, imagine HORIZON more; returns (truth, predicted) palette indices."""
    views = unpack(packed).astype(jnp.int32)
    k1, k2 = jax.random.split(key)
    s = model.observe(views[:, :CONTEXT], actions[:, :CONTEXT], k1)
    im = model.imagine(s.h[:, -1], s.z[:, -1], actions[:, CONTEXT - 1 : CONTEXT - 1 + HORIZON], k2)
    pred = jnp.argmax(model.decode(im), -1)
    return views[:, CONTEXT : CONTEXT + HORIZON], pred


def evaluate(model, shard, step, out, key, n=16):
    v, a = shard
    rng = np.random.default_rng(0)  # the same held-out windows every time
    idx = [(rng.integers(v.shape[0]), rng.integers(v.shape[1] - CONTEXT - HORIZON)) for _ in range(n)]
    packed = np.stack([v[i, o : o + CONTEXT + HORIZON] for i, o in idx])
    acts = np.stack([a[i, o : o + CONTEXT + HORIZON] for i, o in idx]).astype(np.int32)
    truth, pred = (np.asarray(x) for x in open_loop(model, packed, acts, key))
    res = {"step": step}
    for h in (1, 4, 16, 32, 48):
        t, p = truth[:, h - 1], pred[:, h - 1]
        obj = np.isin(t, OBJECT_COLOURS)
        res[f"acc_h{h}"] = float((t == p).mean())
        res[f"obj_recall_h{h}"] = float((p[obj] == t[obj]).mean()) if obj.any() else None
    # image: for 3 sequences, actual (top) vs expected (bottom) at horizons 1, 4, 16, 32
    tiles = []
    for b in range(3):
        cols = [np.concatenate([RGB[truth[b, h - 1]], np.full((2, wm.W, 3), 40, np.uint8), RGB[pred[b, h - 1]]], 0) for h in (1, 4, 16, 32)]
        tiles.append(np.concatenate([np.pad(c, ((0, 0), (0, 4), (0, 0)), constant_values=40) for c in cols], 1))
    img = np.concatenate([np.pad(t, ((0, 6), (0, 0), (0, 0)), constant_values=20) for t in tiles], 0)
    Image.fromarray(img).resize((img.shape[1] * 2, img.shape[0] * 2), Image.NEAREST).save(os.path.join(out, f"open_loop_{step:06d}.png"))
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.expanduser("~/flock2_data/m1_rollouts"))
    p.add_argument("--out", default=os.path.expanduser("~/flock2_data/m1_wm/run1"))
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--length", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    shards = load_shards(a.data)
    train, held = shards[:-1], shards[-1]
    model = wm.WorldModel(nnx.Rngs(a.seed))
    opt = nnx.Optimizer(model, optax.chain(optax.clip_by_global_norm(1000.0), optax.adam(a.lr, eps=1e-8)), wrt=nnx.Param)
    print(f"params {wm.count_params(model):,}; train frames {sum(v.shape[0] * v.shape[1] for v, _ in train):,}", flush=True)
    rng = np.random.default_rng(a.seed)
    key = jax.random.PRNGKey(a.seed)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    t0 = time.time()
    for step in range(1, a.steps + 1):
        packed, acts = sample(train, rng, a.batch, a.length)
        key, k = jax.random.split(key)
        aux = train_step(model, opt, packed, acts, k)
        if step % 100 == 0 or step == 1:
            rec = {"step": step, **{k2: float(v) for k2, v in aux.items()}, "t": time.time() - t0}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            key, k = jax.random.split(key)
            res = evaluate(model, held, step, a.out, k)
            log.write(json.dumps({"eval": res}) + "\n")
            log.flush()
            print(json.dumps({"eval": res}), flush=True)
            with open(os.path.join(a.out, "model.pkl"), "wb") as f:
                pickle.dump(jax.tree.map(np.asarray, nnx.to_pure_dict(nnx.state(model))), f)


if __name__ == "__main__":
    main()
