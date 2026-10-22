"""Milestone 0 training: perception pre-training, then the drawing game.

Run with ``python -m flock.toy.train --help`` or via ``scripts/m0_run.py``.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from flock import world
from flock.nets import Agents, Encoder
from flock.toy import env, game, metrics
from flock.toy.gallery import export_run_gallery


@dataclass
class Config:
    seed: int = 0
    out_dir: str = "runs/m0/default"
    # game
    n_candidates: int = 4
    mode: str = "bubble"  # "bubble" | "stamp"
    random_heading: bool = True
    # model
    action_space: str = "strokes"  # "strokes" | "latent" | "pixels"
    k_strokes: int = 8
    shared_encoder: bool = True
    freeze_encoder: bool = True
    width: int = 32
    # losses
    perceptual_weight: float = 0.0
    ink_weight: float = 0.002
    listening_weight: float = 0.0
    signalling_weight: float = 0.0
    # training
    pretrain_steps: int = 1500
    steps: int = 4000
    batch: int = 64
    lr: float = 3e-4
    eval_every: int = 500
    eval_batch: int = 256
    log_every: int = 50

    @property
    def opaque(self) -> bool:
        return self.mode == "bubble"


def _encoder_filter(cfg: Config):
    """Trainable-parameter filter for the game phase."""
    if cfg.freeze_encoder:
        return nnx.All(nnx.Param, nnx.Not(nnx.PathContains("encoder")), nnx.Not(nnx.PathContains("decoder")))
    return nnx.All(nnx.Param, nnx.Not(nnx.PathContains("decoder")))


# ----------------------------------------------------------------------------------------------
# Phase 1: perception pre-training (autoencoding views, no communication)
# ----------------------------------------------------------------------------------------------


def pretrain_views(key: jax.Array, batch: int, cfg: Config) -> jax.Array:
    """A mix of single-object views and game views (sender and receiver sides)."""
    k1, k2 = jax.random.split(key)
    ov, _ = env.object_views(k1, batch // 2)
    ep = env.make_batch(k2, batch // 4, cfg.n_candidates, cfg.mode, cfg.random_heading)
    return jnp.concatenate([ov, ep.sender_view, ep.receiver_base], axis=0)


@nnx.jit
def pretrain_step(model: Agents, optimizer: nnx.Optimizer, views: jax.Array):
    def loss_fn(m: Agents):
        x = world.to_onehot(views)
        emb, _ = m.encoder(x)
        logits = m.decoder(emb)
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, views)
        return jnp.mean(ce)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


def pretrain(model: Agents, cfg: Config, key: jax.Array, log) -> None:
    tx = optax.adamw(cfg.lr)
    opt = nnx.Optimizer(model, tx, wrt=nnx.Param)
    t0 = time.time()
    for step in range(cfg.pretrain_steps):
        key, k = jax.random.split(key)
        views = pretrain_views(k, cfg.batch, cfg)
        loss = pretrain_step(model, opt, views)
        if step % cfg.log_every == 0 or step == cfg.pretrain_steps - 1:
            log({"phase": "pretrain", "step": step, "recon_ce": float(loss), "t": time.time() - t0})


# ----------------------------------------------------------------------------------------------
# Phase 2: the game
# ----------------------------------------------------------------------------------------------


def make_game_step(cfg: Config):
    wrt = _encoder_filter(cfg)

    @nnx.jit
    def step(model: Agents, optimizer: nnx.Optimizer, ep: env.Episode, ground: jax.Array, key: jax.Array):
        def loss_fn(m: Agents):
            fwd = game.forward(m, ep, key, cfg.opaque)
            ce, acc = game.success_loss(fwd.logits, ep.target_mask)
            loss = jnp.mean(ce)
            aux = {"ce": jnp.mean(ce), "acc": jnp.mean(acc), "ink": jnp.mean(fwd.canvas.ink)}
            loss = loss + cfg.ink_weight * jnp.mean(fwd.canvas.ink)
            if cfg.perceptual_weight > 0:
                pl = game.perceptual_loss(m, fwd.canvas, fwd.sender_emb, ground)
                aux["perceptual"] = pl
                loss = loss + cfg.perceptual_weight * pl
            if cfg.listening_weight > 0:
                blank = game.blank_canvas(ep.target_type.shape[0])
                logits0, _ = game.receive(m, ep.receiver_base, blank, ep.canvas_px, cfg.opaque)
                ll = game.listening_loss(fwd.logits, logits0)
                aux["listening"] = ll
                loss = loss + cfg.listening_weight * ll
            if cfg.signalling_weight > 0:
                sl = game.signalling_loss(fwd.canvas, ep.target_type)
                aux["signalling"] = sl
                loss = loss + cfg.signalling_weight * sl
            aux["loss"] = loss
            return loss, aux

        (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True, argnums=nnx.DiffState(0, wrt))(model)
        optimizer.update(model, grads)
        return aux

    return step


def train(cfg: Config) -> dict:
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)
    log_path = os.path.join(cfg.out_dir, "log.jsonl")
    log_file = open(log_path, "w")

    def log(rec: dict):
        log_file.write(json.dumps(rec) + "\n")
        log_file.flush()
        print(json.dumps(rec), flush=True)

    key = jax.random.PRNGKey(cfg.seed)
    key, k_model, k_pre, k_judge = jax.random.split(key, 4)
    model = Agents(nnx.Rngs(k_model), cfg.action_space, cfg.k_strokes, cfg.shared_encoder, cfg.width)

    # Phase 1
    pretrain(model, cfg, k_pre, log)
    # The judge: a frozen copy of the pre-trained perception encoder that never sees the game.
    judge: Encoder = nnx.clone(model.encoder)
    probe = metrics.fit_type_probe(judge, k_judge)

    # Phase 2
    step_fn = make_game_step(cfg)
    opt = nnx.Optimizer(model, optax.adamw(cfg.lr), wrt=_encoder_filter(cfg))
    t0 = time.time()
    last_eval = None
    for step in range(cfg.steps + 1):
        if step % cfg.eval_every == 0 or step == cfg.steps:
            key, k_eval = jax.random.split(key)
            last_eval = metrics.evaluate(model, judge, probe, cfg, k_eval)
            log({"phase": "eval", "step": step, **last_eval, "t": time.time() - t0})
        if step == cfg.steps:
            break
        key, k_ep, k_g, k_s = jax.random.split(key, 4)
        ep = env.make_batch(k_ep, cfg.batch, cfg.n_candidates, cfg.mode, cfg.random_heading)
        ground = world.to_onehot(jax.vmap(env.plain_ground_view)(jax.random.split(k_g, cfg.batch)))
        aux = step_fn(model, opt, ep, ground, k_s)
        if step % cfg.log_every == 0:
            log({"phase": "game", "step": step, **{k: float(v) for k, v in aux.items()}, "t": time.time() - t0})

    key, k_gal = jax.random.split(key)
    export_run_gallery(model, cfg, k_gal)
    summary = {"config": dataclasses.asdict(cfg), "final": last_eval}
    with open(os.path.join(cfg.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log_file.close()
    return summary


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(description="FLOCK 2.0 Milestone 0 toy drawing game")
    for f in dataclasses.fields(Config):
        if f.type == "bool" or f.type is bool:
            p.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes"), default=f.default)
        else:
            p.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    args = p.parse_args(argv)
    cfg = Config(**vars(args))
    summary = train(cfg)
    print(json.dumps(summary["final"], indent=2))


if __name__ == "__main__":
    main()
