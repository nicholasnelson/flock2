cd /mnt/c/Users/nnels/Documents/projects/flock2 && . ~/.venvs/flock2/bin/activate && export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p runs/meadow2/eval runs/meadow2/hybrid
for s in 0 1 2; do
  python scripts/m2_hybrid.py --out ~/flock2_data/m2_hybrid/s$s --seed $s > runs/meadow2/hybrid/s$s.log 2>&1 || { echo HYBRID_S${s}_FAILED; exit 1; }
  cp ~/flock2_data/m2_hybrid/s$s/log.jsonl runs/meadow2/hybrid/s${s}_log.jsonl
  python scripts/m2_eval.py --policy hybrid --agent ~/flock2_data/m2_hybrid/s$s --out runs/meadow2/eval/hybrid_s$s.json > runs/meadow2/eval/hybrid_s$s.log 2>&1 && echo HYBRID_S${s}_DONE || echo EVAL_S${s}_FAILED
done
