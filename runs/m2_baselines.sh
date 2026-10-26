# meadow v2 after the old-meadow runs stop: one job at a time
cd /mnt/c/Users/nnels/Documents/projects/flock2 && . ~/.venvs/flock2/bin/activate
mkdir -p runs/meadow2/eval
export XLA_PYTHON_CLIENT_PREALLOCATE=false
JAX_PLATFORMS=cpu python scripts/meadow2_checks.py --out runs/meadow2/checks_v21_patch.json > runs/meadow2/checks_v21_patch.log 2>&1 && echo CHECKS_DONE || echo CHECKS_FAILED
for pol in random rest scripted; do
  JAX_PLATFORMS=cpu python scripts/m2_eval.py --policy $pol --out runs/meadow2/eval/$pol.json > runs/meadow2/eval/$pol.log 2>&1 && echo EVAL_${pol}_DONE || echo EVAL_${pol}_FAILED
done
# PPO, timing first (5 updates), then three seeds
python scripts/m2_ppo.py --out ~/flock2_data/m2_ppo/timing --updates 5 > runs/meadow2/ppo_timing.log 2>&1 && echo "PPO_TIMING $(tail -1 ~/flock2_data/m2_ppo/timing/log.jsonl | cut -c1-60)" || echo PPO_TIMING_FAILED
