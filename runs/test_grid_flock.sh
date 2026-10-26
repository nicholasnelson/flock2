cd /mnt/c/Users/nnels/Documents/projects/flock2 && . ~/.venvs/flock2/bin/activate && export JAX_PLATFORMS=cpu
python -m pytest -q tests/test_grid_flock.py 2>&1 | tail -15
