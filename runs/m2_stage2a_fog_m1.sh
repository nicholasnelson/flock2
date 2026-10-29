#!/usr/bin/env bash
# Spec section 21.5: M1 re-read with the 4,000-step knowledge window: the scripted control on 64 meadows and the
# three stage-2a brains on mud 8 + fog 0.5 (5 days, 16 meadows).
cd /mnt/c/Users/nnels/Documents/projects/flock2 && . ~/.venvs/flock2/bin/activate && export XLA_PYTHON_CLIENT_PREALLOCATE=false
E=runs/meadow2/eval; X=runs/meadow2/deaths
echo M1_START $(date +%H:%M:%S)
python scripts/m2_eval.py --policy scripted --willows 3 --inner 8 --fog 0.5 --days 5 --meadows 64 --out $E/scripted_mud8_fog_d5_m64.json --deaths $X/scripted_mud8_fog_d5_m64.json > $E/scripted_mud8_fog_d5_m64.log 2>&1 || echo CONTROL_FAILED
echo CONTROL_DONE $(date +%H:%M:%S)
for s in 0 1 2; do
  n=s2a_s$s; agent=~/flock2_data/m2_hybrid/$n
  python scripts/m2_eval.py --policy hybrid --agent $agent --inner 8 --fog 0.5 --days 5 --out $E/hybrid_${n}_mud8_fog_d5.json --deaths $X/hybrid_${n}_mud8_fog_d5.json > $E/hybrid_${n}_mud8_fog_d5.log 2>&1 || echo EVAL_FAILED $n
  echo M1_DONE $n $(date +%H:%M:%S)
done
echo M1_ALL_DONE $(date +%H:%M:%S)
