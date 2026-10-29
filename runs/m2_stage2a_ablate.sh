#!/usr/bin/env bash
# Spec § 21.4: the stage-2a brains evaluated with the column's state zeroed (mud 8, 5 days), to ask whether
# the column is used at all.
cd /mnt/c/Users/nnels/Documents/projects/flock2 && . ~/.venvs/flock2/bin/activate && export XLA_PYTHON_CLIENT_PREALLOCATE=false
echo ABLATE_START $(date +%H:%M:%S)
for s in 0 1 2; do
  n=s2a_s$s
  python scripts/m2_eval.py --policy hybrid --agent ~/flock2_data/m2_hybrid/$n --inner 8 --days 5 --zero_column --out runs/meadow2/eval/hybrid_${n}_mud8_d5_zerocol.json --deaths runs/meadow2/deaths/hybrid_${n}_mud8_d5_zerocol.json > runs/meadow2/eval/hybrid_${n}_mud8_d5_zerocol.log 2>&1 || { echo ABLATE_FAILED $n; tail -5 runs/meadow2/eval/hybrid_${n}_mud8_d5_zerocol.log; }
  echo ABLATE_DONE $n $(date +%H:%M:%S)
done
echo ABLATE_ALL_DONE $(date +%H:%M:%S)
