#!/usr/bin/env bash
# Stage 1 (spec § 20): the reactive sheep. Usage:
#   bash runs/m2_stage1.sh smoke
#   bash runs/m2_stage1.sh <arm> <seed>      arms: a1 (reactive, willows) a2 (reactive, no willows) a3 (embed hybrid, willows)
cd /mnt/c/Users/nnels/Documents/projects/flock2 && . ~/.venvs/flock2/bin/activate && export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p runs/meadow2/reactive runs/meadow2/hybrid runs/meadow2/eval runs/meadow2/deaths runs/meadow2/diag
if [ "$1" = smoke ]; then
  rm -rf ~/flock2_data/m2_reactive/smoke
  MAXGB=12 bash runs/watchdog.sh python scripts/m2_reactive.py --out ~/flock2_data/m2_reactive/smoke --willows 3 --iters 3 --meadows 2 --steps 16 --minibatches 2 > /tmp/stage1_smoke.log 2>&1
  grep -E '"iter"|WATCHDOG|Error|Traceback' /tmp/stage1_smoke.log | cut -c1-200
  python scripts/m2_eval.py --policy reactive --agent ~/flock2_data/m2_reactive/smoke --meadows 2 --days 0.2 --out /tmp/stage1_smoke_eval.json 2>&1 | grep -E "Error|Traceback|death_share"
  python scripts/m2_diag.py --agent ~/flock2_data/m2_reactive/smoke --meadows 4 --warm 5 --steps 20 --no_mlp --out /tmp/stage1_smoke_diag.json 2>&1 | grep -E "Error|Traceback|D2_true" | head -3
  exit
fi
arm=$1; seed=$2; n=${arm}_s${seed}
case "$arm" in
  a1) kind=reactive; args="--willows 3" ;;
  a2) kind=reactive; args="--willows 0" ;;
  a3) kind=hybrid; args="--willows 3 --iters 2000 --kl_dyn 1.0 --replay 1024 --add_frac 0.5 --policy_input state_embed" ;;
  *) echo "unknown arm $arm"; exit 2 ;;
esac
if [ $kind = reactive ]; then
  MAXGB=12 bash runs/watchdog.sh python scripts/m2_reactive.py --out ~/flock2_data/m2_reactive/$n $args --seed $seed > runs/meadow2/reactive/$n.log 2>&1 || { echo TRAIN_FAILED $n; tail -5 runs/meadow2/reactive/$n.log; exit 1; }
  cp ~/flock2_data/m2_reactive/$n/log.jsonl runs/meadow2/reactive/${n}_log.jsonl
  agent=~/flock2_data/m2_reactive/$n
else
  MAXGB=12 bash runs/watchdog.sh python scripts/m2_hybrid.py --out ~/flock2_data/m2_hybrid/$n $args --seed $seed > runs/meadow2/hybrid/$n.log 2>&1 || { echo TRAIN_FAILED $n; tail -5 runs/meadow2/hybrid/$n.log; exit 1; }
  cp ~/flock2_data/m2_hybrid/$n/log.jsonl runs/meadow2/hybrid/${n}_log.jsonl
  agent=~/flock2_data/m2_hybrid/$n
fi
python scripts/m2_eval.py --policy $kind --agent $agent --out runs/meadow2/eval/${kind}_$n.json --deaths runs/meadow2/deaths/${kind}_$n.json > runs/meadow2/eval/${kind}_$n.log 2>&1 || echo EVAL_FAILED $n
python scripts/m2_diag.py --agent $agent --out runs/meadow2/diag/$n.json > runs/meadow2/diag/$n.log 2>&1 || echo DIAG_FAILED $n
grep WATCHDOG runs/meadow2/reactive/$n.log runs/meadow2/hybrid/$n.log 2>/dev/null; echo STAGE1_DONE $n $(date +%H:%M:%S)
