cd /mnt/c/Users/nnels/Documents/projects/flock2 && . ~/.venvs/flock2/bin/activate && export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p runs/meadow2/probes
if [ "$1" = smoke ]; then
  python scripts/m2_probe.py --agent ~/flock2_data/m2_hybrid/s0 --meadows 8 --warm 20 --steps 60 2>&1 | grep -E "^ridge|^mlp|Error|Traceback" ; exit
fi
for s in 0 1 2; do
  python scripts/m2_probe.py --agent ~/flock2_data/m2_hybrid/s$s --out runs/meadow2/probes/s$s.json > runs/meadow2/probes/s$s.log 2>&1 && echo PROBE_S${s}_DONE || echo PROBE_S${s}_FAILED
done
python scripts/m2_probe.py --untrained --out runs/meadow2/probes/untrained.json > runs/meadow2/probes/untrained.log 2>&1 && echo PROBE_UNTRAINED_DONE || echo PROBE_UNTRAINED_FAILED
