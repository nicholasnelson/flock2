cd /mnt/c/Users/nnels/Documents/projects/flock2 && . ~/.venvs/flock2/bin/activate && export XLA_PYTHON_CLIENT_PREALLOCATE=false
rm -rf ~/flock2_data/m2_hybrid/smoke
python scripts/m2_hybrid.py --out ~/flock2_data/m2_hybrid/smoke --iters 4 --ppo_start 2 2>&1 | grep -v "^W\|^I0\|^E1\|cuda_\|external/" | tail -8
JAX_PLATFORMS= python scripts/m2_eval.py --policy hybrid --agent ~/flock2_data/m2_hybrid/smoke --meadows 2 --days 0.02 2>&1 | grep -E '"reward"|death_share|Error|Traceback' | head
