cd /mnt/c/Users/nnels/Documents/projects/flock2 && . ~/.venvs/flock2/bin/activate && export JAX_PLATFORMS=cpu
python scripts/grid_flock_eval.py --agent /tmp/gf_learned --worlds 4 --steps 100 2>&1 | grep -v "^W\|^I0" | tail -40
