cd /mnt/c/Users/nnels/Documents/projects/flock2 && . ~/.venvs/flock2/bin/activate && export XLA_PYTHON_CLIENT_PREALLOCATE=false
rm -rf ~/flock2_data/m2_ppo/timing
python scripts/m2_ppo.py --out ~/flock2_data/m2_ppo/timing --updates 5 2>&1 | grep -v "^W\|^I0\|^E1\|cuda_" | tail -4
tail -1 ~/flock2_data/m2_ppo/timing/log.jsonl | cut -c1-120
