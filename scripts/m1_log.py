#!/usr/bin/env python
"""Log scripted-flock rollouts for world-model pretraining (Milestone 1, step 2).

    python scripts/m1_log.py --out ~/flock2_data/m1_rollouts --shards 4

Each shard: 32 fresh meadows, 8 recorded sheep each, 1,024 steps from a random time of day
(after a 200-step warm-up) — 262,144 frames, ~2 GB packed. Keep the data on the WSL filesystem
(much faster than /mnt/c for training reads).
"""

from __future__ import annotations

import argparse
import os
import time

from flock.meadow.rollouts import write_rollout


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.expanduser("~/flock2_data/m1_rollouts"))
    p.add_argument("--shards", type=int, default=4)
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--steps", type=int, default=1024)
    a = p.parse_args()
    for s in range(a.first, a.first + a.shards):
        t = time.time()
        n = write_rollout(a.out, s, seed=1000 + s, steps=a.steps)
        print(f"shard {s}: {n} frames in {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()
