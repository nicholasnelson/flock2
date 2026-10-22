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
from flock.nets import Agents, Decoder, Encoder
from flock.toy import env, envs, game, metrics
from flock.toy.gallery import export_run_gallery


@dataclass
class Config:
    seed: int = 0
    out_dir: str = "runs/m0/default"
    # game
    renderer: str = "topdown"  # "topdown" (tested baseline) | "iso" (isometric, heading-aligned)
    n_candidates: int = 4
    mode: str = "bubble"  # "bubble" | "stamp"
    random_heading: bool = True
    # model
    action_space: str = "strokes"  # "strokes" | "latent" | "pixels"
    k_strokes: int = 8
    canvas_res: int = 0  # 0: strokes are rasterised at the drawing's own size in the view (16 top-down, 32 iso); else at this coarser resolution, then enlarged
    mono: bool = False  # every stroke is drawn in the outline colour (15); the sender chooses shape only
    stroke_width: float = 0.0  # >0: every stroke is a line of this width (displayed px); 0: width is free
    stroke_cost: float = 4.0  # with fixed width: ink per stroke on top of its length (px), so dots are not free
    hard_gate: bool = False  # strokes fully on or off in the forward pass (straight-through); no faint strokes
    shared_encoder: bool = True
    freeze_encoder: bool = True
    width: int = 32
    # losses
    perceptual_weight: float = 0.0
    ink_weight: float = 0.002
    ink_ramp_steps: int = 1000  # ink cost ramps linearly from 0 to ink_weight over this many steps
    ink_delay_steps: int = 0  # no ink cost at all for this many game steps, then the ramp (lets reading get started
    #   before any cost: with Adam even a tiny consistent ink gradient switches hard-gated strokes off)
    sender_warmup_steps: int = 0  # "iconic initialisation": steps of perceptual-only training of the sender before the game
    listening_weight: float = 0.0
    signalling_weight: float = 0.0
    # training
    pretrain_colour_dropout: float = 0.0  # P(an object in a pre-training view is a silhouette); iso only
    judge_invariance: float = 0.0  # >0: the judge is a separate encoder, pre-trained with shift/flip/scale/blur and
    #   this weight pulling augmented embeddings to the clean one, so it sees shape, not placement; iso only.
    #   The sheep's own perception is pre-trained without it (strong invariance blinds the receiver).
    variation: bool = False  # instance variation: sprite variants, sub-tile offsets, scales; held-out variants for eval; iso only
    jitter_px: int = 0  # channel noise: the drawing is moved by up to this many px within its frame when read or judged
    centre_glyph: bool = False  # the channel re-centres each drawing on its ink centroid (before jitter)
    rotate_stamps: bool = False  # in stamp mode, the channel turns each drawing by a random multiple of 90 degrees
    min_strokes: int = 0  # with hard_gate: at least this many strokes are always on (no blank drawings)
    blank_weight: float = 0.0  # penalty weight on drawings with less than blank_ink px of ink
    blank_ink: float = 8.0  # px of ink below which the blank penalty applies (one short stroke)
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


JUDGE_SHIFT = 6  # px, each axis
JUDGE_SCALE_MIN = 0.5  # the far view of the planned two-scale vision is half scale
JUDGE_P_FLIP = 0.5
JUDGE_P_BLUR = 0.3


@jax.jit
def _image_aug(key: jax.Array, inp: jax.Array, tgt: jax.Array) -> tuple[jax.Array, jax.Array]:
    """The same random shift, horizontal flip and blur (2x down, then up) applied to each input
    and its target. Palette images; shifts wrap round (the ground texture is uniform)."""
    b = inp.shape[0]
    k1, k2, k3 = jax.random.split(key, 3)
    d = jax.random.randint(k1, (b, 2), -JUDGE_SHIFT, JUDGE_SHIFT + 1)
    flip = jax.random.bernoulli(k2, JUDGE_P_FLIP, (b,))
    blur = jax.random.bernoulli(k3, JUDGE_P_BLUR, (b,))

    def one(v, dd, f, bl):
        v = jnp.roll(v, (dd[0], dd[1]), axis=(0, 1))
        v = jnp.where(f, v[:, ::-1], v)
        lo = jnp.repeat(jnp.repeat(v[::2, ::2], 2, axis=0), 2, axis=1)
        return jnp.where(bl, lo, v)

    f = jax.vmap(one)
    return f(inp, d, flip, blur), f(tgt, d, flip, blur)


def pretrain_views(key: jax.Array, batch: int, cfg: Config, invariant: bool = False) -> tuple[jax.Array, jax.Array, jax.Array]:
    """A mix of single-object views and game views (sender and receiver sides).

    Returns (inputs, reconstruction targets, anchors). Inputs and targets differ only under
    colour dropout (silhouettes in, full colour out). With ``invariant`` (the judge's
    pre-training) the single-object views are also shifted, flipped, blurred and scaled, and
    ``anchors`` are the same scenes clean and at scale 1; otherwise ``anchors`` are the inputs."""
    E = envs.get(cfg)
    k1, k2, k3 = jax.random.split(key, 3)
    n_obj = batch // 2
    if cfg.renderer == "iso":
        inv = invariant
        smin = JUDGE_SCALE_MIN if inv else E.SCALE_MIN
        ov_in, ov_tgt, anchor, _ = E.object_views_pretrain(
            k1, n_obj, cfg.pretrain_colour_dropout, variation=cfg.variation or inv, scale_min=smin, variants=cfg.variation
        )
        if inv:
            ov_in, ov_tgt = _image_aug(k3, ov_in, ov_tgt)
        else:
            anchor = ov_in
    else:
        ov_in, _ = E.object_views(k1, n_obj)
        ov_tgt = anchor = ov_in
    ep = envs.make_batch(cfg, k2, batch // 4)
    rest = jnp.concatenate([ep.sender_view, ep.receiver_base], axis=0)
    return jnp.concatenate([ov_in, rest], axis=0), jnp.concatenate([ov_tgt, rest], axis=0), anchor


@nnx.jit
def pretrain_step(model: Agents, optimizer: nnx.Optimizer, views: jax.Array, targets: jax.Array, anchors: jax.Array, inv_weight: jax.Array):
    def loss_fn(m: Agents):
        x = world.to_onehot(views)
        emb, _ = m.encoder(x)
        logits = m.decoder(emb)
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        # invariance: the augmented single-object views' embeddings pulled towards the clean
        # scene's (stop-gradient); reconstruction keeps the embedding from collapsing
        n = anchors.shape[0]
        e_anchor = jax.lax.stop_gradient(m.encoder(world.to_onehot(anchors))[0])
        nrm = lambda z: z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-6)
        inv = jnp.mean(1.0 - jnp.sum(nrm(emb[:n]) * nrm(e_anchor), axis=-1))
        return jnp.mean(ce) + inv_weight * inv

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


def pretrain(model, cfg: Config, key: jax.Array, log, invariance: float = 0.0, phase: str = "pretrain") -> None:
    """Autoencoder pre-training of ``model.encoder`` / ``model.decoder`` (the sheep's perception,
    or the separate judge when ``invariance`` > 0)."""
    tx = optax.adamw(cfg.lr)
    opt = nnx.Optimizer(model, tx, wrt=nnx.Param)
    t0 = time.time()
    for step in range(cfg.pretrain_steps):
        key, k = jax.random.split(key)
        views, targets, anchors = pretrain_views(k, cfg.batch, cfg, invariant=invariance > 0)
        loss = pretrain_step(model, opt, views, targets, anchors, jnp.float32(invariance))
        if step % cfg.log_every == 0 or step == cfg.pretrain_steps - 1:
            log({"phase": phase, "step": step, "loss": float(loss), "t": time.time() - t0})


class JudgeAE(nnx.Module):
    """A separate perception autoencoder whose encoder becomes the invariant judge."""

    def __init__(self, rngs: nnx.Rngs, width: int, view: int):
        self.encoder = Encoder(rngs, width, view)
        self.decoder = Decoder(rngs, width, view)


# ----------------------------------------------------------------------------------------------
# Phase 2: the game
# ----------------------------------------------------------------------------------------------


def make_game_step(cfg: Config):
    wrt = _encoder_filter(cfg)

    @nnx.jit
    def step(model: Agents, judge: Encoder, optimizer: nnx.Optimizer, ep: env.Episode, ground: jax.Array, key: jax.Array, ink_scale: jax.Array):
        def loss_fn(m: Agents):
            fwd = game.forward(m, ep, key, cfg.opaque, **game.channel_args(cfg))
            ce, acc = game.success_loss(fwd.logits, ep.target_mask)
            loss = jnp.mean(ce)
            aux = {"ce": jnp.mean(ce), "acc": jnp.mean(acc), "ink": jnp.mean(fwd.canvas.ink)}
            loss = loss + ink_scale * cfg.ink_weight * jnp.mean(fwd.canvas.ink)
            if cfg.blank_weight > 0:
                blank = jnp.mean(jax.nn.relu(cfg.blank_ink - fwd.canvas.ink))
                aux["blank"] = blank
                loss = loss + cfg.blank_weight * blank
            if cfg.perceptual_weight > 0:
                judged = game.transmit(fwd.canvas, jax.random.fold_in(key, 11), **game.channel_args(cfg))
                pl = game.perceptual_loss(judge, judged, world.to_onehot(ep.sender_view), ground, envs.probe_px(cfg, ep))
                aux["perceptual"] = pl
                loss = loss + cfg.perceptual_weight * pl
            if cfg.listening_weight > 0:
                blank = game.blank_canvas(ep.target_type.shape[0], m.canvas)
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


def make_warmup_step(cfg: Config):
    """Perceptual-only training of the sender: learn to draw what you see, before the game."""
    wrt = nnx.All(nnx.Param, nnx.PathContains("sender"))

    @nnx.jit
    def step(model: Agents, judge: Encoder, optimizer: nnx.Optimizer, ep: env.Episode, ground: jax.Array, key: jax.Array):
        def loss_fn(m: Agents):
            sv = world.to_onehot(ep.sender_view)
            canvas, _ = game.draw(m, sv, key, False)
            pl = game.perceptual_loss(judge, canvas, sv, ground, envs.probe_px(cfg, ep))
            return pl + cfg.ink_weight * jnp.mean(canvas.ink), {"perceptual": pl, "ink": jnp.mean(canvas.ink)}

        (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True, argnums=nnx.DiffState(0, wrt))(model)
        optimizer.update(model, grads)
        return aux

    return step


def train(cfg: Config) -> dict:
    if (cfg.pretrain_colour_dropout > 0 or cfg.judge_invariance > 0 or cfg.variation) and cfg.renderer != "iso":
        raise ValueError("pretrain_colour_dropout, judge_invariance and variation are implemented for the iso renderer only")
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)
    log_path = os.path.join(cfg.out_dir, "log.jsonl")
    log_file = open(log_path, "w")

    def log(rec: dict):
        log_file.write(json.dumps(rec) + "\n")
        log_file.flush()
        print(json.dumps(rec), flush=True)

    E = envs.get(cfg)
    key = jax.random.PRNGKey(cfg.seed)
    key, k_model, k_pre, k_judge = jax.random.split(key, 4)
    model = Agents(
        nnx.Rngs(k_model),
        cfg.action_space,
        cfg.k_strokes,
        cfg.shared_encoder,
        cfg.width,
        canvas_res=cfg.canvas_res or None,
        mono_colour=15 if cfg.mono else None,
        stroke_width=cfg.stroke_width or None,
        stroke_cost=cfg.stroke_cost,
        hard_gate=cfg.hard_gate,
        min_strokes=cfg.min_strokes,
        view=E.VIEW,
        canvas=E.CANVAS,
    )

    # Phase 1
    pretrain(model, cfg, k_pre, log)
    # The judge: a frozen encoder that never sees the game. By default a copy of the pre-trained
    # perception encoder; with judge_invariance, a separately pre-trained invariant one.
    if cfg.judge_invariance > 0:
        # a separate, invariant observer: same architecture, its own pre-training
        key, k_jae, k_jpre = jax.random.split(key, 3)
        jae = JudgeAE(nnx.Rngs(k_jae), cfg.width, E.VIEW)
        pretrain(jae, cfg, k_jpre, log, invariance=cfg.judge_invariance, phase="pretrain_judge")
        judge: Encoder = jae.encoder
    else:
        judge = nnx.clone(model.encoder)
    probe = metrics.fit_type_probe(judge, k_judge, cfg)

    # Optional sender warm-up (iconic initialisation)
    if cfg.sender_warmup_steps > 0:
        warm = make_warmup_step(cfg)
        w_opt = nnx.Optimizer(model, optax.adamw(cfg.lr), wrt=nnx.All(nnx.Param, nnx.PathContains("sender")))
        t0 = time.time()
        for step in range(cfg.sender_warmup_steps):
            key, k_ep, k_g, k_s = jax.random.split(key, 4)
            ep = envs.make_batch(cfg, k_ep, cfg.batch)
            ground = world.to_onehot(jax.vmap(E.plain_ground_view)(jax.random.split(k_g, cfg.batch)))
            aux = warm(model, judge, w_opt, ep, ground, k_s)
            if step % cfg.log_every == 0 or step == cfg.sender_warmup_steps - 1:
                log({"phase": "warmup", "step": step, **{k: float(v) for k, v in aux.items()}, "t": time.time() - t0})

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
        ep = envs.make_batch(cfg, k_ep, cfg.batch)
        ground = world.to_onehot(jax.vmap(E.plain_ground_view)(jax.random.split(k_g, cfg.batch)))
        ink_scale = jnp.float32(min(1.0, max(0, step - cfg.ink_delay_steps) / max(cfg.ink_ramp_steps, 1)))
        aux = step_fn(model, judge, opt, ep, ground, k_s, ink_scale)
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
