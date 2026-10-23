#!/usr/bin/env python
"""Grade every run's saved drawings (``drawings.npz``) with the current independent examiner.

    python scripts/m0_examine.py runs/m0/grid_dot runs/m0/smoke_exam

Writes ``examiner.json`` next to each ``drawings.npz`` and prints the mean over seeds per cell.
Re-grading needs no sheep re-training: the drawings are saved at the end of every iso run.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import jax
import numpy as np

from flock.toy import examiner as X


def main(roots: list[str]) -> None:
    model = X.get_examiner()
    cells = defaultdict(list)
    for root in roots:
        for path in sorted(glob.glob(os.path.join(root, "**", "drawings.npz"), recursive=True)):
            d = np.load(path)
            res = X.examine(model, d["alpha"].astype(np.float32), d["types"], jax.random.PRNGKey(0))
            res["examiner_version"] = X.VERSION
            with open(os.path.join(os.path.dirname(path), "examiner.json"), "w") as f:
                json.dump(res, f, indent=2)
            run = os.path.dirname(path)
            cell = os.path.dirname(run) if os.path.basename(run).startswith("s") and os.path.basename(run)[1:].isdigit() else run
            cells[cell].append(res)
    first = next(iter(cells.values()))[0]
    print(f"examiner {X.VERSION}: real views {first['examiner_acc_real_views']:.2f}, outline drawings of real sprites {first['examiner_acc_outline_views']:.2f}")
    print(f"{'cell':45s} {'as drawn':>10s} {'centred':>10s}  per-type recall as drawn (salt, tree, flower, fox)")
    for cell, rs in cells.items():
        a = np.mean([r["examiner_acc_as_drawn"] for r in rs])
        c = np.mean([r["examiner_acc_centred"] for r in rs])
        cm = np.sum([np.array(r["examiner_confusion_as_drawn"]) for r in rs], 0)
        recall = np.round(np.diag(cm) / np.maximum(cm.sum(1), 1), 2)
        print(f"{cell:45s} {a:10.2f} {c:10.2f}  {recall.tolist()}  (n={len(rs)})")


if __name__ == "__main__":
    main(sys.argv[1:])
