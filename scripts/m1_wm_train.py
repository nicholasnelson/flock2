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


def load_shards(data: str, intero: bool = False, reward: bool = False):
    """Shards as (views, actions, levels | None, reward | None), memory-mapped."""
    names = sorted(glob.glob(os.path.join(data, "shard_*.views.npy")))
    out = []
    for n in names:
        lv = np.load(n.replace(".views.", ".levels."), mmap_mode="r") if intero else None
        rw = np.load(n.replace(".views.", ".reward."), mmap_mode="r") if reward else None
        out.append((np.load(n, mmap_mode="r"), np.load(n.replace(".views.", ".actions."), mmap_mode="r"), lv, rw))
    return out


def sample(shards, rng, B, L):
    vs, acts, lvs, rws = [], [], [], []
    for _ in range(B):
        v, a, lv, rw = shards[rng.integers(len(shards))]
        n, o = rng.integers(v.shape[0]), rng.integers(v.shape[1] - L + 1)
        vs.append(v[n, o : o + L])
        acts.append(a[n, o : o + L])
        lvs.append(lv[n, o : o + L] if lv is not None else np.zeros((L, 0), np.float16))
        rws.append(rw[n, o : o + L] if rw is not None else np.zeros((L,), np.float32))
    return np.stack(vs), np.stack(acts).astype(np.int32), np.stack(lvs).astype(np.float32), np.stack(rws).astype(np.float32)


@nnx.jit
def train_step(model, opt, packed, actions, levels, reward, key):
    views = unpack(packed).astype(jnp.int32)
    intero = levels if model.intero_head is not None else None
    (loss, aux), grads = nnx.value_and_grad(wm.loss_fn, has_aux=True)(model, views, actions, key, intero, reward)
    opt.update(model, grads)
    return aux


@nnx.jit
def open_loop(model, packed, actions, levels, key):
    """Observe CONTEXT strips, imagine HORIZON more; returns (truth, predicted) palette indices
    and, with interoception, the imagined drive levels; with a reward head, the imagined rewards
    (entry k: the reward of the step taken at CONTEXT - 1 + k)."""
    views = unpack(packed).astype(jnp.int32)
    k1, k2 = jax.random.split(key)
    intero = levels[:, :CONTEXT] if model.intero_head is not None else None
    s = model.observe(views[:, :CONTEXT], actions[:, :CONTEXT], k1, intero)
    im = model.imagine(s.h[:, -1], s.z[:, -1], actions[:, CONTEXT - 1 : CONTEXT - 1 + HORIZON], k2)
    pred = jnp.argmax(model.decode(im), -1)
    lv = model.decode_intero(im) if model.intero_head is not None else jnp.zeros(pred.shape[:2] + (0,))
    rw = model.reward(jnp.concatenate([im.h, im.z], -1)) if model.reward_head is not None else jnp.zeros(pred.shape[:2])
    return views[:, CONTEXT : CONTEXT + HORIZON], pred, lv, rw


def evaluate(model, shard, step, out, key, n=64):
    v, a, lv, rw = shard
    rng = np.random.default_rng(0)  # the same held-out windows every time
    idx = [(rng.integers(v.shape[0]), rng.integers(v.shape[1] - CONTEXT - HORIZON)) for _ in range(n)]
    packed = np.stack([v[i, o : o + CONTEXT + HORIZON] for i, o in idx])
    acts = np.stack([a[i, o : o + CONTEXT + HORIZON] for i, o in idx]).astype(np.int32)
    levels = np.stack([lv[i, o : o + CONTEXT + HORIZON] for i, o in idx]).astype(np.float32) if lv is not None else np.zeros((n, CONTEXT + HORIZON, 0), np.float32)
    rewards = np.stack([rw[i, o : o + CONTEXT + HORIZON] for i, o in idx]).astype(np.float32) if rw is not None else None
    chunks = [[np.asarray(x) for x in open_loop(model, packed[c : c + 16], acts[c : c + 16], levels[c : c + 16], jax.random.fold_in(key, c))] for c in range(0, n, 16)]
    truth, pred, lv_pred, rw_pred = (np.concatenate(xs) for xs in zip(*chunks))
    res = {"step": step}
    if lv_pred.shape[-1]:
        lv_true = levels[:, CONTEXT : CONTEXT + HORIZON]
        res["intero_abs_err_h1"] = float(np.abs(lv_pred[:, 0] - lv_true[:, 0]).mean())
        res["intero_abs_err_h48"] = float(np.abs(lv_pred[:, -1] - lv_true[:, -1]).mean())
    if rewards is not None and model.reward_head is not None:
        corr = lambda x, y: float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else None
        r_true = rewards[:, CONTEXT - 1 : CONTEXT - 1 + HORIZON]
        res["reward_corr_h1"] = corr(r_true[:, 0], rw_pred[:, 0])
        res["reward_corr"] = corr(r_true.ravel(), rw_pred.ravel())
        # what the actor-critic sees: reward summed over its 15-step imagination horizon
        s_true, s_pred = r_true[:, :15].sum(1), rw_pred[:, :15].sum(1)
        res["reward15_corr"] = corr(s_true, s_pred)
        res["reward15_mean_true_pred"] = [float(s_true.mean()), float(s_pred.mean())]
        res["reward15_std_true_pred"] = [float(s_true.std()), float(s_pred.std())]
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
    p.add_argument("--intero", action="store_true", help="train with the drive levels (interoception)")
    p.add_argument("--reward", action="store_true", help="train a reward head on the logged reward")
    p.add_argument("--init", default=None, help="checkpoint to start from (parts it lacks start fresh)")
    p.add_argument("--eval_only", type=int, default=0, help="just evaluate --init on this many held-out windows")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    shards = load_shards(a.data, a.intero, a.reward)
    train, held = shards[:-1], shards[-1]
    model = wm.WorldModel(nnx.Rngs(a.seed), n_intero=5 if a.intero else 0, reward=a.reward)
    if a.init:
        wm.load_into(model, pickle.load(open(os.path.expanduser(a.init), "rb")))
    opt = nnx.Optimizer(model, optax.chain(optax.clip_by_global_norm(1000.0), optax.adam(a.lr, eps=1e-8)), wrt=nnx.Param)
    if a.eval_only:
        print(json.dumps({"eval": evaluate(model, held, 0, a.out, jax.random.PRNGKey(a.seed), n=a.eval_only)}), flush=True)
        return
    print(f"params {wm.count_params(model):,}; train frames {sum(v.shape[0] * v.shape[1] for v, *_ in train):,}", flush=True)
    rng = np.random.default_rng(a.seed)
    key = jax.random.PRNGKey(a.seed)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    t0 = time.time()
    for step in range(1, a.steps + 1):
        packed, acts, lvs, rws = sample(train, rng, a.batch, a.length)
        key, k = jax.random.split(key)
        aux = train_step(model, opt, packed, acts, lvs, rws, k)
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
