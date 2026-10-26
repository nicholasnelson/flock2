cd /mnt/c/Users/nnels/Documents/projects/flock2 && . ~/.venvs/flock2/bin/activate && export XLA_PYTHON_CLIENT_PREALLOCATE=false
python scripts/m2_diag.py --agent ~/flock2_data/m2_hybrid/mem_s0 --validate_d3 --out runs/meadow2/diag/d3_validation.json 2>&1 | grep -E "^validate|Error|Traceback"
