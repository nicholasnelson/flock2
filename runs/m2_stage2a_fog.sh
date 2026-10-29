#!/usr/bin/env bash
# Stage 2a's second pressure, fog days (spec section 21.4). Usage:
#   bash runs/m2_stage2a_fog.sh control        the scripted flock on mud 8 + fog 0.5 (M1's positive control)
#   bash runs/m2_stage2a_fog.sh demand         the three stage-2a brains on mud 8 + fog 0.5: survival, M1, the probes
#   bash runs/m2_stage2a_fog.sh <seed>         the learning stage: continue s2a_s<seed> with the fog ramp, then the evaluations
#   bash runs/m2_stage2a_fog.sh chain          seeds 0, 1, 2 in turn
cd /mnt/c/Users/nnels/Documents/projects/flock2 && . ~/.venvs/flock2/bin/activate && export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p runs/meadow2/hybrid runs/meadow2/eval runs/meadow2/deaths runs/meadow2/diag
FOG=0.5
E=runs/meadow2/eval; X=runs/meadow2/deaths; G=runs/meadow2/diag
if [ "$1" = control ]; then
  echo CONTROL_START $(date +%H:%M:%S)
  python scripts/m2_eval.py --policy scripted --willows 3 --inner 8 --fog $FOG --days 5 --out $E/scripted_mud8_fog_d5.json --deaths $X/scripted_mud8_fog_d5.json > $E/scripted_mud8_fog_d5.log 2>&1 || echo CONTROL_FAILED
  echo CONTROL_DONE $(date +%H:%M:%S)
  exit
fi
if [ "$1" = demand ]; then
  echo DEMAND_START $(date +%H:%M:%S)
  for s in 0 1 2; do
    n=s2a_s$s; agent=~/flock2_data/m2_hybrid/$n
    python scripts/m2_eval.py --policy hybrid --agent $agent --inner 8 --fog $FOG --days 5 --out $E/hybrid_${n}_mud8_fog_d5.json --deaths $X/hybrid_${n}_mud8_fog_d5.json > $E/hybrid_${n}_mud8_fog_d5.log 2>&1 || echo DEMAND_EVAL_FAILED $n
    python scripts/m2_diag.py --agent $agent --inner 8 --fog $FOG --out $G/${n}_mud8_fog.json > $G/${n}_mud8_fog.log 2>&1 || echo DEMAND_DIAG_FAILED $n
    echo DEMAND_DONE $n $(date +%H:%M:%S)
  done
  echo DEMAND_ALL_DONE $(date +%H:%M:%S)
  exit
fi
if [ "$1" = chain ]; then
  echo FOG_START $(date +%H:%M:%S)
  for s in 0 1 2; do bash runs/m2_stage2a_fog.sh $s; echo $(date +%H:%M:%S); done
  echo FOG_ALL_DONE $(date +%H:%M:%S)
  exit
fi
# the learning stage: the stage-2a recipe (column kept), mud held at 8 from the start, fog ramped 0 -> 0.5 over iterations 200-1400
RECIPE="--willows 3 --kl_dyn 1.0 --replay 1024 --add_frac 0.5 --policy_input state_embed --column 256 --steps 128 --reset_every 4864 --ppo_start 0 --mud_ramp 0,1,8 --fog_ramp 200,1400,$FOG"
s=$1; n=s2f_s$s; agent=~/flock2_data/m2_hybrid/$n
MAXGB=12 bash runs/watchdog.sh python scripts/m2_hybrid.py --out $agent $RECIPE --continue_from ~/flock2_data/m2_hybrid/s2a_s$s --iters 2000 --seed $s > runs/meadow2/hybrid/$n.log 2>&1 || { echo TRAIN_FAILED $n; tail -5 runs/meadow2/hybrid/$n.log; exit 1; }
cp $agent/log.jsonl runs/meadow2/hybrid/${n}_log.jsonl
python scripts/m2_eval.py --policy hybrid --agent $agent --inner 8 --fog $FOG --days 5 --out $E/hybrid_${n}_mud8_fog_d5.json --deaths $X/hybrid_${n}_mud8_fog_d5.json > $E/hybrid_${n}_mud8_fog_d5.log 2>&1 || echo EVAL_FAILED fog $n
python scripts/m2_eval.py --policy hybrid --agent $agent --out $E/hybrid_$n.json --deaths $X/hybrid_$n.json > $E/hybrid_$n.log 2>&1 || echo EVAL_FAILED plain $n
python scripts/m2_eval.py --policy hybrid --agent $agent --inner 8 --fog $FOG --days 5 --sheep 1 --meadows 64 --out $E/hybrid_${n}_mud8_fog_lone.json --deaths $X/hybrid_${n}_mud8_fog_lone.json > $E/hybrid_${n}_mud8_fog_lone.log 2>&1 || echo EVAL_FAILED lone $n
python scripts/m2_eval.py --policy hybrid --agent $agent --inner 8 --fog $FOG --days 5 --zero_column --out $E/hybrid_${n}_mud8_fog_d5_zerocol.json > $E/hybrid_${n}_mud8_fog_d5_zerocol.log 2>&1 || echo EVAL_FAILED zerocol $n
python scripts/m2_diag.py --agent $agent --inner 8 --fog $FOG --out $G/${n}_mud8_fog.json > $G/${n}_mud8_fog.log 2>&1 || echo DIAG_FAILED fog $n
grep WATCHDOG runs/meadow2/hybrid/$n.log; echo FOG_DONE $n $(date +%H:%M:%S)
