#!/usr/bin/env python
"""Run the Milestone 0 ablation grid and tabulate results.

The grid tests, per the plan:
  1. shared encoder vs separate message encoder
  2. with vs without the perceptual (resemblance) loss
  3. stroke head vs latent decoder vs pixel head
  4. bubble vs stamp
  5. drawing resolution: 16x16 vs 8x8 shown 2x enlarged
  6. black-only drawing, with and without colour-dropout perception pre-training
  7. (128 world) fixed-width lines, sweeping the cost per unit length
  8. (128 world) instance variation x channel jitter x invariant judge, lines at cost 0.002

Not every combination is run; the cells below cover the questions. Each cell runs ``--seeds``
seeds. Results are written to ``<out>/grid.json`` and printed as a table. On the 4080 with the
default sizes a cell takes a few minutes; on CPU, a lot longer — pass ``--quick`` to shrink.

    python scripts/m0_grid.py --out runs/m0/grid --seeds 3 --renderer iso
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os

from flock.toy.train import Config, train

CELLS = {
    # name: overrides
    "shared_noperc": dict(shared_encoder=True, perceptual_weight=0.0),
    "shared_perc": dict(shared_encoder=True, perceptual_weight=1.0),
    "separate_noperc": dict(shared_encoder=False, perceptual_weight=0.0),
    "separate_perc": dict(shared_encoder=False, perceptual_weight=1.0),
    "shared_perc_stamp": dict(shared_encoder=True, perceptual_weight=1.0, mode="stamp"),
    "shared_noperc_stamp": dict(shared_encoder=True, perceptual_weight=0.0, mode="stamp"),
    "latent_noperc": dict(action_space="latent", perceptual_weight=0.0),
    "latent_perc": dict(action_space="latent", perceptual_weight=1.0),
    "pixels_noperc": dict(action_space="pixels", perceptual_weight=0.0),
    "pixels_perc": dict(action_space="pixels", perceptual_weight=1.0),
    "shared_perc_aux": dict(shared_encoder=True, perceptual_weight=1.0, listening_weight=0.1, signalling_weight=0.1),
    "shared_noperc_aux": dict(shared_encoder=True, perceptual_weight=0.0, listening_weight=0.1, signalling_weight=0.1),
    # coarse drawing: strokes drawn on 8x8 and shown 2x enlarged in the same 16x16 slot
    "shared_noperc_c8": dict(shared_encoder=True, perceptual_weight=0.0, canvas_res=8),
    "shared_perc_c8": dict(shared_encoder=True, perceptual_weight=1.0, canvas_res=8),
    "shared_perc_stamp_c8": dict(shared_encoder=True, perceptual_weight=1.0, mode="stamp", canvas_res=8),
    # black-only drawing (outline colour), with the plain judge and with a judge pre-trained
    # with colour dropout (objects sometimes shown as silhouettes); iso only
    "mono_noperc": dict(mono=True, perceptual_weight=0.0),
    "mono_perc": dict(mono=True, perceptual_weight=1.0),
    "mono_noperc_stamp": dict(mono=True, perceptual_weight=0.0, mode="stamp"),
    "mono_perc_stamp": dict(mono=True, perceptual_weight=1.0, mode="stamp"),
    "mono_noperc_cd": dict(mono=True, perceptual_weight=0.0, pretrain_colour_dropout=0.5),
    "mono_perc_cd": dict(mono=True, perceptual_weight=1.0, pretrain_colour_dropout=0.5),
    "mono_noperc_stamp_cd": dict(mono=True, perceptual_weight=0.0, mode="stamp", pretrain_colour_dropout=0.5),
    "mono_perc_stamp_cd": dict(mono=True, perceptual_weight=1.0, mode="stamp", pretrain_colour_dropout=0.5),
}
# 128 x 128 isometric world (16x8 tiles; referents salt lick, pine, flower, fox): black lines of
# fixed 2 px width, strokes fully on or off, ink = length + 4 px per stroke (a pen-lift price, so
# dots are not free), colour-dropout judge; sweep the cost weight.
for _ink, _tag in [(0.002, "002"), (0.01, "01"), (0.03, "03")]:
    for _mode in ("bubble", "stamp"):
        for _pw, _p in [(1.0, "perc"), (0.0, "noperc")]:
            CELLS[f"line_{_p}_{_mode}_ink{_tag}"] = dict(
                mono=True, stroke_width=2.0, stroke_cost=4.0, hard_gate=True, pretrain_colour_dropout=0.5, perceptual_weight=_pw, mode=_mode, ink_weight=_ink
            )

# Instance variation (variants, offsets, scales; eval on held-out variants) x channel jitter x a
# separate invariant judge. Lines at cost 0.002 with the cost held off for 1,500 steps (without
# the delay hard-gated strokes die before the receiver learns to read).
_LINE = dict(mono=True, stroke_width=2.0, stroke_cost=4.0, hard_gate=True, pretrain_colour_dropout=0.5, ink_weight=0.002, ink_delay_steps=1500)
for _v in (False, True):
    for _j in (False, True):
        for _i in (False, True):
            _name = "inv3_" + ("V" if _v else "v") + ("J" if _j else "j") + ("I" if _i else "i")
            CELLS[_name] = dict(_LINE, perceptual_weight=1.0, mode="bubble", variation=_v, jitter_px=4 if _j else 0, judge_invariance=1.0 if _i else 0.0)
CELLS["inv3_VJI_noperc"] = dict(_LINE, perceptual_weight=0.0, mode="bubble", variation=True, jitter_px=4, judge_invariance=1.0)
CELLS["inv3_VJI_stamp"] = dict(_LINE, perceptual_weight=1.0, mode="stamp", variation=True, jitter_px=4, judge_invariance=1.0)


KEYS = ["acc", "acc_seen", "acc_blank", "acc_swapped", "cic_kl", "judge_probe_acc", "judge_rank_acc", "own_probe_acc", "ink"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/m0/grid")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--cells", default="all", help="comma-separated cell names, or 'all'")
    p.add_argument("--quick", action="store_true", help="tiny sizes for a smoke run")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--pretrain_steps", type=int, default=None)
    p.add_argument("--renderer", default="topdown", help="'topdown' or 'iso'; applied to every cell")
    args = p.parse_args()

    names = list(CELLS) if args.cells == "all" else args.cells.split(",")
    results_path = os.path.join(args.out, "grid.json")
    os.makedirs(args.out, exist_ok=True)
    results = json.load(open(results_path)) if os.path.exists(results_path) else {}

    for name in names:
        for seed in range(args.seeds):
            key = f"{name}/s{seed}"
            if key in results:
                continue
            over = dict(CELLS[name], renderer=args.renderer)
            if args.quick:
                over.update(pretrain_steps=50, steps=100, batch=16, eval_batch=64, eval_every=50)
            if args.steps is not None:
                over["steps"] = args.steps
            if args.pretrain_steps is not None:
                over["pretrain_steps"] = args.pretrain_steps
            cfg = Config(seed=seed, out_dir=os.path.join(args.out, name, f"s{seed}"), **over)
            print(f"=== {key}: {over}", flush=True)
            summary = train(cfg)
            results[key] = {k: summary["final"].get(k) for k in KEYS}
            results[key]["config"] = dataclasses.asdict(cfg)
            json.dump(results, open(results_path, "w"), indent=2)

    # table: mean over seeds per cell
    print("\ncell".ljust(22) + "".join(k.rjust(16) for k in KEYS))
    for name in names:
        rows = [v for k, v in results.items() if k.startswith(name + "/")]
        if not rows:
            continue
        line = name.ljust(22)
        for k in KEYS:
            vals = [r[k] for r in rows if r.get(k) is not None]
            line += (f"{sum(vals) / len(vals):.3f}" if vals else "-").rjust(16)
        print(line)


if __name__ == "__main__":
    main()
