cd /mnt/c/Users/nnels/Documents/projects/flock2 && . ~/.venvs/flock2/bin/activate && export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p runs/meadow2/eval runs/meadow2/ppo
for s in 0 1 2; do
  python scripts/m2_ppo.py --out ~/flock2_data/m2_ppo/s$s --seed $s > runs/meadow2/ppo/s$s.log 2>&1 || { echo PPO_S${s}_FAILED; exit 1; }
  cp ~/flock2_data/m2_ppo/s$s/log.jsonl runs/meadow2/ppo/s${s}_log.jsonl
  python scripts/m2_eval.py --policy ppo --agent ~/flock2_data/m2_ppo/s$s --out runs/meadow2/eval/ppo_s$s.json > runs/meadow2/eval/ppo_s$s.log 2>&1 && echo PPO_S${s}_DONE || echo EVAL_S${s}_FAILED
done
