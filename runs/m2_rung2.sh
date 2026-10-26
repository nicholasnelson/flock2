cd /mnt/c/Users/nnels/Documents/projects/flock2 && . ~/.venvs/flock2/bin/activate && export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p runs/meadow2/eval runs/meadow2/hybrid runs/meadow2/probes
n=long_s0
python scripts/m2_hybrid.py --out ~/flock2_data/m2_hybrid/$n --seed 0 --iters 2000 > runs/meadow2/hybrid/$n.log 2>&1 || { echo TRAIN_FAILED; exit 1; }
cp ~/flock2_data/m2_hybrid/$n/log.jsonl runs/meadow2/hybrid/${n}_log.jsonl
python scripts/m2_eval.py --policy hybrid --agent ~/flock2_data/m2_hybrid/$n --out runs/meadow2/eval/hybrid_$n.json > runs/meadow2/eval/hybrid_$n.log 2>&1 && echo EVAL_DONE || echo EVAL_FAILED
python scripts/m2_probe.py --agent ~/flock2_data/m2_hybrid/$n --out runs/meadow2/probes/$n.json > runs/meadow2/probes/$n.log 2>&1 && echo PROBE_DONE || echo PROBE_FAILED
