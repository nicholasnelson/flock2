#!/usr/bin/env bash
# Spec section 21.9: the clear-night check. The skyline brains on the mud world with NO fog days
# (every night clear, the peaks visible), 5 days, skyline on and blanked: does the compass pay on
# the nights it can be seen? Usage:
#   bash runs/m2_sky_clearnight.sh            s2k_s0, s2k_s1, s2k_s2
#   bash runs/m2_sky_clearnight.sh s2l        the fallback brains instead
cd /mnt/c/Users/nnels/Documents/projects/flock2 && . ~/.venvs/flock2/bin/activate && export XLA_PYTHON_CLIENT_PREALLOCATE=false
E=runs/meadow2/eval; X=runs/meadow2/deaths
P=${1:-s2k}
echo CLEARNIGHT_START $P $(date +%H:%M:%S)
for s in 0 1 2; do
  n=${P}_s$s; agent=~/flock2_data/m2_hybrid/$n
  python scripts/m2_eval.py --policy hybrid --agent $agent --inner 8 --fog 0 --days 5 --skyline 1 --out $E/hybrid_${n}_sky_mud8_fog0_d5.json --deaths $X/hybrid_${n}_sky_mud8_fog0_d5.json > $E/hybrid_${n}_sky_mud8_fog0_d5.log 2>&1 || echo CLEARNIGHT_FAILED $n sky
  python scripts/m2_eval.py --policy hybrid --agent $agent --inner 8 --fog 0 --days 5 --skyline 0 --out $E/hybrid_${n}_nosky_mud8_fog0_d5.json --deaths $X/hybrid_${n}_nosky_mud8_fog0_d5.json > $E/hybrid_${n}_nosky_mud8_fog0_d5.log 2>&1 || echo CLEARNIGHT_FAILED $n nosky
  echo CLEARNIGHT_DONE $n $(date +%H:%M:%S)
done
echo CLEARNIGHT_ALL_DONE $P $(date +%H:%M:%S)
