#!/usr/bin/env python
"""Run the Milestone 0 ablation grid and tabulate results.

The grid tests, per the plan:
  1. shared encoder vs separate message encoder
  2. with vs without the perceptual (resemblance) loss
  3. stroke head vs latent decoder vs pixel head
  4. bubble vs stamp

Not every combination is run; the cells below cover the questions. Each cell runs ``--seeds``
seeds. Results are written to ``<out>/grid.json`` and printed as a table. On the 4080 with the
default sizes a cell takes a few minutes; on CPU, a lot longer — pass ``--quick`` to shrink.

    python scripts/m0_grid.py --out runs/m0/grid --seeds 3
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
}

KEYS = ["acc", "acc_blank", "acc_swapped", "cic_kl", "judge_probe_acc", "judge_rank_acc", "own_probe_acc", "ink"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/m0/grid")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--cells", default="all", help="comma-separated cell names, or 'all'")
    p.add_argument("--quick", action="store_true", help="tiny sizes for a smoke run")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--pretrain_steps", type=int, default=None)
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
            over = dict(CELLS[name])
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
