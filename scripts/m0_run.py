#!/usr/bin/env python
"""Run one Milestone 0 configuration. Thin wrapper over ``flock.toy.train``.

Examples
    python scripts/m0_run.py --out_dir runs/m0/shared_perc --perceptual_weight 1.0
    python scripts/m0_run.py --shared_encoder false --out_dir runs/m0/separate
"""

from flock.toy.train import main

if __name__ == "__main__":
    main()
