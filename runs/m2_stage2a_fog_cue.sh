#!/usr/bin/env bash
# Spec section 21.5: what the embedding reads in fog at 8-16 tiles, the willows or the bare mud ring: the probe on
# a mud-free fog world and, with the 8-12 / 12-16 split, on the mud fog world (one brain). Queued behind the re-check.
cd /mnt/c/Users/nnels/Documents/projects/flock2
while pgrep -f m2_stage2a_fog_recheck.sh > /dev/null; do sleep 30; done
. ~/.venvs/flock2/bin/activate && export XLA_PYTHON_CLIENT_PREALLOCATE=false
G=runs/meadow2/diag
echo CUE_START $(date +%H:%M:%S)
python scripts/m2_diag.py --agent ~/flock2_data/m2_hybrid/s2a_s0 --fog 1.0 --out $G/s2a_s0_plain_fog1.json > $G/s2a_s0_plain_fog1.log 2>&1 || echo CUE_FAILED plain
python scripts/m2_diag.py --agent ~/flock2_data/m2_hybrid/s2a_s0 --inner 8 --fog 1.0 --out $G/s2a_s0_mud8_fog1.json > $G/s2a_s0_mud8_fog1.log 2>&1 || echo CUE_FAILED mud
echo CUE_ALL_DONE $(date +%H:%M:%S)
