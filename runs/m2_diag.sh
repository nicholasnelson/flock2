cd /mnt/c/Users/nnels/Documents/projects/flock2 && . ~/.venvs/flock2/bin/activate && export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p runs/meadow2/diag
if [ "$1" = smoke ]; then
  JAX_PLATFORMS=cpu python scripts/m2_diag.py --agent ~/flock2_data/m2_hybrid/s0 --meadows 4 --warm 10 --steps 40 --no_mlp 2>&1 | grep -v "^W\|^I0" | tail -30; exit
fi
for n in "$@"; do
  python scripts/m2_diag.py --agent ~/flock2_data/m2_hybrid/$n --out runs/meadow2/diag/$n.json > runs/meadow2/diag/$n.log 2>&1 && echo DIAG_${n}_DONE || { echo DIAG_${n}_FAILED; tail -5 runs/meadow2/diag/$n.log; }
done
