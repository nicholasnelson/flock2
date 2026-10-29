#!/usr/bin/env bash
# Spec section 21.5: the demand check once more at fog 0.75 (the rule's step when not every seed fails at 0.5), queued behind the probes.
cd /mnt/c/Users/nnels/Documents/projects/flock2
while pgrep -f m2_stage2a_fog_diag1.sh > /dev/null; do sleep 30; done
. ~/.venvs/flock2/bin/activate && export XLA_PYTHON_CLIENT_PREALLOCATE=false
E=runs/meadow2/eval; X=runs/meadow2/deaths
echo RECHECK_START $(date +%H:%M:%S)
for s in 0 1 2; do
  n=s2a_s$s
  python scripts/m2_eval.py --policy hybrid --agent ~/flock2_data/m2_hybrid/$n --inner 8 --fog 0.75 --days 5 --out $E/hybrid_${n}_mud8_fog075_d5.json --deaths $X/hybrid_${n}_mud8_fog075_d5.json > $E/hybrid_${n}_mud8_fog075_d5.log 2>&1 || echo RECHECK_FAILED $n
  echo RECHECK_DONE $n $(date +%H:%M:%S)
done
python scripts/m2_eval.py --policy scripted --willows 3 --inner 8 --fog 0.75 --days 5 --meadows 64 --out $E/scripted_mud8_fog075_d5_m64.json > $E/scripted_mud8_fog075_d5_m64.log 2>&1 || echo RECHECK_CONTROL_FAILED
echo RECHECK_ALL_DONE $(date +%H:%M:%S)
