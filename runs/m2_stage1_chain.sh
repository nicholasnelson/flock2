#!/usr/bin/env bash
# Stage 1 (spec § 20) in one chain, one GPU job at a time: the three screening arms at seed 0, then
# the stage arm A1 at seeds 1 and 2.
cd /mnt/c/Users/nnels/Documents/projects/flock2
echo STAGE1_START $(date +%H:%M:%S)
for job in "a1 0" "a2 0" "a3 0" "a1 1" "a1 2"; do
  bash runs/m2_stage1.sh $job
  echo $(date +%H:%M:%S)
done
echo STAGE1_ALL_DONE $(date +%H:%M:%S)
