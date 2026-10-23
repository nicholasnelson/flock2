#!/usr/bin/env python
"""Surprise in the pretrained world model: the posterior-vs-prior KL along held-out sequences
(Milestone 1 success criterion: KL should spike on genuinely novel events, not on routine
neighbour motion).

Three measurements on the held-out shard:
1. The KL trace along a long sequence, with night and the sheep's actions marked.
2. Mean KL by the action just taken (is turning, walking or standing more surprising?).
3. An injected novelty: half-way through a sequence the strips switch to another sheep in
   another meadow (a "teleport"). The KL at the switch should stand far above the routine level.

    python scripts/m1_wm_surprise.py --model ~/flock2_data/m1_wm/run1/model.pkl --out runs/m1/surprise
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from PIL import Image, ImageDraw

from flock import wm
from flock.meadow.rollouts import unpack
from flock.meadow.sim import ACTION_NAMES


@nnx.jit
def _kl_one(model, packed, actions, key):
    s = model.observe(unpack(packed).astype(jnp.int32), actions, key)
    return wm.kl(s.post, s.prior)  # (1, T)


def kl_trace(model, packed, actions, key):
    """(B, T) KL per step, one sequence at a time (a 512-step sequence is already 512 strips)."""
    return np.concatenate([np.asarray(_kl_one(model, packed[i : i + 1], actions[i : i + 1], jax.random.fold_in(key, i))) for i in range(len(packed))])


def plot(trace, night, actions, path, mark=None):
    T = len(trace)
    W, H = 1000, 220
    im = Image.new("RGB", (W, H), (250, 250, 250))
    d = ImageDraw.Draw(im)
    top = float(np.percentile(trace, 99.5)) * 1.1 + 1e-6
    xs = lambda t: int(t / (T - 1) * (W - 20)) + 10
    for t in range(T):
        if night[t]:
            d.line([(xs(t), 10), (xs(t), H - 30)], fill=(215, 215, 235))
    pts = [(xs(t), H - 30 - int(min(trace[t], top) / top * (H - 50))) for t in range(T)]
    d.line(pts, fill=(30, 30, 30), width=1)
    for t in range(T):  # action ticks
        col = {3: (60, 120, 220), 4: (60, 120, 220), 1: (60, 170, 60), 2: (220, 120, 40), 5: (120, 90, 40)}.get(int(actions[t]))
        if col:
            d.line([(xs(t), H - 26), (xs(t), H - 20)], fill=col)
    if mark is not None:
        d.line([(xs(mark), 10), (xs(mark), H - 30)], fill=(220, 40, 40), width=2)
    d.text((10, H - 16), "KL (posterior || prior) per step; shaded = night; ticks: blue turn, green walk, orange trot, brown graze; red = teleport", fill=(20, 20, 20))
    im.save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.expanduser("~/flock2_data/m1_wm/run1/model.pkl"))
    p.add_argument("--data", default=os.path.expanduser("~/flock2_data/m1_rollouts"))
    p.add_argument("--out", default="runs/m1/surprise")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model = wm.WorldModel(nnx.Rngs(0))
    state = nnx.state(model)
    nnx.replace_by_pure_dict(state, pickle.load(open(a.model, "rb")))
    nnx.update(model, state)

    held = sorted(glob.glob(os.path.join(a.data, "shard_*.views.npy")))[-1]
    v = np.load(held, mmap_mode="r")
    acts = np.load(held.replace(".views.", ".actions."), mmap_mode="r")
    night = np.load(held.replace(".views.", ".night."), mmap_mode="r")
    T = 512
    res = {}

    # 1 and 2: KL along 8 long held-out sequences
    seqs = np.arange(8)
    tr = np.asarray(kl_trace(model, np.stack([v[i, :T] for i in seqs]), np.stack([acts[i, :T] for i in seqs]).astype(np.int32), jax.random.PRNGKey(0)))
    plot(tr[0], night[seqs[0], :T], acts[seqs[0], :T], os.path.join(a.out, "kl_trace.png"))
    a_prev = np.stack([acts[i, : T - 1] for i in seqs])  # KL at t reflects the action at t-1
    by_action = {ACTION_NAMES[k]: float(tr[:, 1:][a_prev == k].mean()) for k in range(len(ACTION_NAMES)) if (a_prev == k).any()}
    res["kl_mean"] = float(tr[:, 16:].mean())
    res["kl_p99"] = float(np.percentile(tr[:, 16:], 99))
    res["kl_by_previous_action"] = by_action
    nt = np.stack([night[i, :T] for i in seqs])
    res["kl_day"], res["kl_night"] = float(tr[:, 16:][~nt[:, 16:]].mean()), float(tr[:, 16:][nt[:, 16:]].mean())

    # 3: teleport half-way: sequence i's first half, then sequence j's second half (another meadow)
    half = T // 2
    pk, ak = [], []
    for i in range(8):
        j = (i * 8 + 37) % v.shape[0]  # a sheep from a different meadow (8 recorded sheep per meadow)
        pk.append(np.concatenate([v[i, :half], v[j, half:T]]))
        ak.append(np.concatenate([acts[i, :half], acts[j, half:T]]))
    tt = np.asarray(kl_trace(model, np.stack(pk), np.stack(ak).astype(np.int32), jax.random.PRNGKey(1)))
    res["kl_at_teleport"] = float(tt[:, half].mean())
    res["kl_after_teleport_5"] = float(tt[:, half + 5].mean())
    res["teleport_over_p99"] = res["kl_at_teleport"] / res["kl_p99"]
    plot(tt[0], night[0, :T], ak[0], os.path.join(a.out, "kl_teleport.png"), mark=half)
    json.dump(res, open(os.path.join(a.out, "surprise.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
