cd /mnt/c/Users/nnels/Documents/projects/flock2 && . ~/.venvs/flock2/bin/activate && export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p runs/meadow2/eval runs/meadow2/hybrid runs/meadow2/diag
COMMON="--kl_dyn 1.0 --replay 2048 --add_frac 0.5 --seed 0"
if [ "$1" = smoke ]; then
  for arm in "s1 --model spatial --ms_weight 1.0 --batch 4"; do
    set -- $arm; n=smoke_$1; shift
    rm -rf ~/flock2_data/m2_hybrid/$n
    python scripts/m2_hybrid.py --out ~/flock2_data/m2_hybrid/$n $COMMON "$@" --iters 3 --ppo_start 1 2>&1 | grep -v "^W\|^I0\|^E1\|cuda_\|external/\|Source Location\|^$" | tail -3
  done
  exit
fi
# s1 uses batches of 4 sequences (8 ran out of GPU memory in the smoke test); the same number of updates
for arm in "n0" "n1 --ms_weight 1.0" "s1 --model spatial --ms_weight 1.0 --batch 4"; do
  set -- $arm; n=$1; shift
  MAXGB=12 bash runs/watchdog.sh python scripts/m2_hybrid.py --out ~/flock2_data/m2_hybrid/$n $COMMON "$@" > runs/meadow2/hybrid/$n.log 2>&1 || { echo ${n}_TRAIN_FAILED; tail -5 runs/meadow2/hybrid/$n.log; continue; }
  cp ~/flock2_data/m2_hybrid/$n/log.jsonl runs/meadow2/hybrid/${n}_log.jsonl
  python scripts/m2_eval.py --policy hybrid --agent ~/flock2_data/m2_hybrid/$n --out runs/meadow2/eval/hybrid_$n.json > runs/meadow2/eval/hybrid_$n.log 2>&1 || echo ${n}_EVAL_FAILED
  python scripts/m2_diag.py --agent ~/flock2_data/m2_hybrid/$n --out runs/meadow2/diag/$n.json > runs/meadow2/diag/$n.log 2>&1 || echo ${n}_DIAG_FAILED
  echo ${n}_DONE
done
