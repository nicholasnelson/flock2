#!/usr/bin/env bash
# Spec section 21.6's recommendation: fog + faster thirst. Usage:
#   bash runs/m2_stage2a_thirst.sh demand     the s2f brains and the scripted control on mud 8 + fog 0.75 + thirst x2, 5 days
#   THIRST=3 bash runs/m2_stage2a_thirst.sh demand
cd /mnt/c/Users/nnels/Documents/projects/flock2 && . ~/.venvs/flock2/bin/activate && export XLA_PYTHON_CLIENT_PREALLOCATE=false
THIRST=${THIRST:-2}; TT=thirst$THIRST
E=runs/meadow2/eval; X=runs/meadow2/deaths
if [ "$1" = demand ]; then
  echo THIRST_DEMAND_START $THIRST $(date +%H:%M:%S)
  python scripts/m2_eval.py --policy scripted --willows 3 --inner 8 --fog 0.75 --thirst $THIRST --days 5 --meadows 64 --out $E/scripted_mud8_fog075_${TT}_d5_m64.json --deaths $X/scripted_mud8_fog075_${TT}_d5_m64.json > $E/scripted_mud8_fog075_${TT}_d5_m64.log 2>&1 || echo THIRST_FAILED scripted
  echo THIRST_DONE scripted $(date +%H:%M:%S)
  for n in s2f_s0 s2f_s1 s2f_s2; do
    python scripts/m2_eval.py --policy hybrid --agent ~/flock2_data/m2_hybrid/$n --inner 8 --fog 0.75 --thirst $THIRST --days 5 --out $E/hybrid_${n}_mud8_fog075_${TT}_d5.json --deaths $X/hybrid_${n}_mud8_fog075_${TT}_d5.json > $E/hybrid_${n}_mud8_fog075_${TT}_d5.log 2>&1 || echo THIRST_FAILED $n
    echo THIRST_DONE $n $(date +%H:%M:%S)
  done
  echo THIRST_DEMAND_ALL_DONE $(date +%H:%M:%S)
  exit
fi
echo "usage: bash runs/m2_stage2a_thirst.sh demand"
