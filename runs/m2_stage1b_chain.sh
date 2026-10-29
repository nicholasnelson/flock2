#!/usr/bin/env bash
# Stage 1, second chain (spec § 20.2, owner's approval 2026-10-29): once the first chain has finished
# A1 seed 1, stop it before A1 seed 2 and run the new stage arm A3 (embed hybrid, willows) at seeds 1
# and 2; then A1 seed 2 for the record, if the GPU is still free.
cd /mnt/c/Users/nnels/Documents/projects/flock2
echo STAGE1B_START $(date +%H:%M:%S)
until grep -q "STAGE1_DONE a1_s1" runs/meadow2/stage1.log; do sleep 30; done
echo A1_S1_DONE_SEEN $(date +%H:%M:%S)
pkill -f m2_stage1_chain.sh; sleep 2
pkill -f "m2_stage1.sh a1 2"; pkill -f "m2_reactive/a1_s2"; pkill -f watchdog.sh; sleep 10
nvidia-smi --query-gpu=memory.used --format=csv,noheader
for job in "a3 1" "a3 2" "a1 2"; do
  bash runs/m2_stage1.sh $job
  echo $(date +%H:%M:%S)
done
echo STAGE1B_ALL_DONE $(date +%H:%M:%S)
