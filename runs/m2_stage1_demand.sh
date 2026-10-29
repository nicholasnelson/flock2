#!/usr/bin/env bash
# After the second stage 1 chain: stage 2a's demand check and the lone-sheep evaluation of the
# stage-1 brain (spec § 20.2, points 2 and 3 of the owed list). One GPU job at a time.
#   demand:  each A3 seed on the 8-tile mud world (16 meadows, 3 days); must lose >= 40 % for the mud
#            to count as a pressure. The scripted flock on the same world through the same script, for
#            the comparison in one place.
#   lone:    each A3 seed with one sheep per meadow, 64 meadows, 3 days, the willow world: does the
#            survival hold without a flock to follow?
cd /mnt/c/Users/nnels/Documents/projects/flock2 && . ~/.venvs/flock2/bin/activate && export XLA_PYTHON_CLIENT_PREALLOCATE=false
echo DEMAND_START $(date +%H:%M:%S)
until grep -q "STAGE1B_ALL_DONE" runs/meadow2/stage1b.log; do sleep 60; done
echo STAGE1B_SEEN $(date +%H:%M:%S)
python scripts/m2_eval.py --policy scripted --willows 3 --inner 8 --out runs/meadow2/eval/scripted_mud8.json --deaths runs/meadow2/deaths/scripted_mud8.json > runs/meadow2/eval/scripted_mud8.log 2>&1 || echo FAILED scripted_mud8
for s in 0 1 2; do
  agent=~/flock2_data/m2_hybrid/a3_s$s
  [ -d $agent ] || { echo MISSING $agent; continue; }
  python scripts/m2_eval.py --policy hybrid --agent $agent --inner 8 --out runs/meadow2/eval/hybrid_a3_s${s}_mud8.json --deaths runs/meadow2/deaths/hybrid_a3_s${s}_mud8.json > runs/meadow2/eval/hybrid_a3_s${s}_mud8.log 2>&1 || echo FAILED demand s$s
  echo DEMAND_DONE s$s $(date +%H:%M:%S)
  python scripts/m2_eval.py --policy hybrid --agent $agent --sheep 1 --meadows 64 --out runs/meadow2/eval/hybrid_a3_s${s}_lone.json --deaths runs/meadow2/deaths/hybrid_a3_s${s}_lone.json > runs/meadow2/eval/hybrid_a3_s${s}_lone.log 2>&1 || echo FAILED lone s$s
  echo LONE_DONE s$s $(date +%H:%M:%S)
done
echo DEMAND_ALL_DONE $(date +%H:%M:%S)
