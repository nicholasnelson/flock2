cd /mnt/c/Users/nnels/Documents/projects/flock2 && . ~/.venvs/flock2/bin/activate && export XLA_PYTHON_CLIENT_PREALLOCATE=false
for s in 1 2; do
  n=n1_seed$s
  MAXGB=12 bash runs/watchdog.sh python scripts/m2_hybrid.py --out ~/flock2_data/m2_hybrid/$n --kl_dyn 1.0 --replay 2048 --add_frac 0.5 --ms_weight 1.0 --seed $s > runs/meadow2/hybrid/$n.log 2>&1 || { echo ${n}_TRAIN_FAILED; continue; }
  cp ~/flock2_data/m2_hybrid/$n/log.jsonl runs/meadow2/hybrid/${n}_log.jsonl
  python scripts/m2_eval.py --policy hybrid --agent ~/flock2_data/m2_hybrid/$n --out runs/meadow2/eval/hybrid_$n.json > runs/meadow2/eval/hybrid_$n.log 2>&1 || echo ${n}_EVAL_FAILED
  python scripts/m2_diag.py --agent ~/flock2_data/m2_hybrid/$n --out runs/meadow2/diag/$n.json > runs/meadow2/diag/$n.log 2>&1 || echo ${n}_DIAG_FAILED
  echo ${n}_DONE
done
