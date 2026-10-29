#!/usr/bin/env bash
# Spec section 21.5: the probes on the stage-2a brains with every day foggy (the fog reading), one GPU job at a time.
cd /mnt/c/Users/nnels/Documents/projects/flock2 && . ~/.venvs/flock2/bin/activate && export XLA_PYTHON_CLIENT_PREALLOCATE=false
while pgrep -f m2_stage2a_fog_m1.sh > /dev/null; do sleep 30; done
G=runs/meadow2/diag
echo DIAG1_START $(date +%H:%M:%S)
for s in 0 1 2; do
  n=s2a_s$s
  python scripts/m2_diag.py --agent ~/flock2_data/m2_hybrid/$n --inner 8 --fog 1.0 --out $G/${n}_mud8_fog1.json > $G/${n}_mud8_fog1.log 2>&1 || echo DIAG1_FAILED $n
  echo DIAG1_DONE $n $(date +%H:%M:%S)
done
echo DIAG1_ALL_DONE $(date +%H:%M:%S)
