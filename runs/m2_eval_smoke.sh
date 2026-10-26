cd /mnt/c/Users/nnels/Documents/projects/flock2 && . ~/.venvs/flock2/bin/activate && export JAX_PLATFORMS=cpu
python scripts/m2_eval.py --policy scripted --meadows 2 --days 0.1 2>&1 | grep -E 'nearest|others_in|thirsty|information_gap|Error|Traceback|line'
