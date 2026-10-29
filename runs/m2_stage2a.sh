#!/usr/bin/env bash
# Stage 2a's learning stage (spec § 21.1). Usage:
#   bash runs/m2_stage2a.sh smoke          a few GPU iterations at full width, then a tiny evaluation
#   bash runs/m2_stage2a.sh <seed>         continue stage 1's brain a3_s<seed> with the column and the mud ramp,
#                                          then the four evaluations and the probes
#   bash runs/m2_stage2a.sh chain          seeds 0, 1, 2 in turn (one GPU job at a time)
cd /mnt/c/Users/nnels/Documents/projects/flock2 && . ~/.venvs/flock2/bin/activate && export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p runs/meadow2/hybrid runs/meadow2/eval runs/meadow2/deaths runs/meadow2/diag
# the registered recipe: A3's, plus the column, the continuation, the ramp and 128-step rollouts
# (reset_every 4864 = 38 x 128, the nearest multiple to two days of 4800 steps)
RECIPE="--willows 3 --kl_dyn 1.0 --replay 1024 --add_frac 0.5 --policy_input state_embed --column 256 --steps 128 --reset_every 4864 --ppo_start 0 --mud_ramp 200,1400,8"
if [ "$1" = smoke ]; then
  rm -rf ~/flock2_data/m2_hybrid/s2a_smoke
  MAXGB=12 bash runs/watchdog.sh python scripts/m2_hybrid.py --out ~/flock2_data/m2_hybrid/s2a_smoke $RECIPE --continue_from ~/flock2_data/m2_hybrid/a3_s0 --iters 5 --seed 0 > /tmp/s2a_smoke.log 2>&1
  grep -E '"iter"|WATCHDOG|Error|Traceback|continued' /tmp/s2a_smoke.log | cut -c1-220
  python scripts/m2_eval.py --policy hybrid --agent ~/flock2_data/m2_hybrid/s2a_smoke --meadows 2 --days 0.1 --inner 8 2>&1 | grep -E "Error|Traceback|death_share"
  exit
fi
if [ "$1" = chain ]; then
  echo STAGE2A_START $(date +%H:%M:%S)
  for s in 0 1 2; do bash runs/m2_stage2a.sh $s; echo $(date +%H:%M:%S); done
  echo STAGE2A_ALL_DONE $(date +%H:%M:%S)
  exit
fi
s=$1; n=s2a_s$s; agent=~/flock2_data/m2_hybrid/$n
MAXGB=12 bash runs/watchdog.sh python scripts/m2_hybrid.py --out $agent $RECIPE --continue_from ~/flock2_data/m2_hybrid/a3_s$s --iters 2000 --seed $s > runs/meadow2/hybrid/$n.log 2>&1 || { echo TRAIN_FAILED $n; tail -5 runs/meadow2/hybrid/$n.log; exit 1; }
cp $agent/log.jsonl runs/meadow2/hybrid/${n}_log.jsonl
# survival on the mud (5 days), stage 1 still passing (no mud, 3 days), alone on the mud, and the probes on both worlds
python scripts/m2_eval.py --policy hybrid --agent $agent --inner 8 --days 5 --out runs/meadow2/eval/hybrid_${n}_mud8_d5.json --deaths runs/meadow2/deaths/hybrid_${n}_mud8_d5.json > runs/meadow2/eval/hybrid_${n}_mud8_d5.log 2>&1 || echo EVAL_FAILED mud $n
python scripts/m2_eval.py --policy hybrid --agent $agent --out runs/meadow2/eval/hybrid_$n.json --deaths runs/meadow2/deaths/hybrid_$n.json > runs/meadow2/eval/hybrid_$n.log 2>&1 || echo EVAL_FAILED plain $n
python scripts/m2_eval.py --policy hybrid --agent $agent --inner 8 --days 5 --sheep 1 --meadows 64 --out runs/meadow2/eval/hybrid_${n}_mud8_lone.json --deaths runs/meadow2/deaths/hybrid_${n}_mud8_lone.json > runs/meadow2/eval/hybrid_${n}_mud8_lone.log 2>&1 || echo EVAL_FAILED lone $n
python scripts/m2_diag.py --agent $agent --inner 8 --out runs/meadow2/diag/${n}_mud8.json > runs/meadow2/diag/${n}_mud8.log 2>&1 || echo DIAG_FAILED mud $n
python scripts/m2_diag.py --agent $agent --out runs/meadow2/diag/$n.json > runs/meadow2/diag/$n.log 2>&1 || echo DIAG_FAILED plain $n
grep WATCHDOG runs/meadow2/hybrid/$n.log; echo STAGE2A_DONE $n $(date +%H:%M:%S)
