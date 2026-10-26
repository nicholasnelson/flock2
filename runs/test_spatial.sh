cd /mnt/c/Users/nnels/Documents/projects/flock2 && . ~/.venvs/flock2/bin/activate && export JAX_PLATFORMS=cpu
python -m pytest -q -s tests/test_wm_spatial.py 2>&1 | grep -v "^W\|^I0" | tail -15
