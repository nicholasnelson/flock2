#!/usr/bin/env bash
# Stage 0, death forensics (spec § 19.1): re-evaluate the existing policies with the death trace.
# Usage: bash runs/m2_forensics.sh            (all)   |   bash runs/m2_forensics.sh scripted ppo:s0 hybrid:embed_s0
cd /mnt/c/Users/nnels/Documents/projects/flock2 && . ~/.venvs/flock2/bin/activate && export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p runs/meadow2/eval runs/meadow2/deaths
targets=("$@")
[ ${#targets[@]} -eq 0 ] && targets=(scripted ppo:s0 hybrid:s0 hybrid:long_s0 hybrid:embed_s0 hybrid:n1 hybrid:n1_seed1 hybrid:n1_seed2 hybrid:mem_s0 hybrid:rot_s0)
for tgt in "${targets[@]}"; do
  kind=${tgt%%:*}; name=${tgt#*:}
  if [ "$kind" = scripted ]; then
    python scripts/m2_eval.py --policy scripted --out runs/meadow2/eval/scripted.json --deaths runs/meadow2/deaths/scripted.json > runs/meadow2/eval/scripted.log 2>&1 || echo EVAL_FAILED $tgt
  else
    python scripts/m2_eval.py --policy $kind --agent ~/flock2_data/m2_$kind/$name --out runs/meadow2/eval/${kind}_$name.json --deaths runs/meadow2/deaths/${kind}_$name.json > runs/meadow2/eval/${kind}_$name.log 2>&1 || echo EVAL_FAILED $tgt
  fi
  echo FORENSICS_DONE $tgt $(date +%H:%M:%S)
done
echo FORENSICS_ALL_DONE
