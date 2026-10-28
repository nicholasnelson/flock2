#!/usr/bin/env bash
# Stage 0 in one chain (one GPU job at a time): a first read of the forensics, the ladder's deciding
# arms first, then the rest of the ladder and the remaining forensics.
cd /mnt/c/Users/nnels/Documents/projects/flock2
echo STAGE0_START $(date +%H:%M:%S)
bash runs/m2_forensics.sh scripted ppo:s0 hybrid:embed_s0
for arm in r8 r8_rot30 r8_place1000 r8_place30 r4 r16; do
  for seed in 0 1 2; do
    bash runs/m2_oracle.sh $arm $seed
    echo $(date +%H:%M:%S)
  done
done
bash runs/m2_forensics.sh hybrid:s0 hybrid:long_s0 hybrid:n1 hybrid:n1_seed1 hybrid:n1_seed2 hybrid:mem_s0 hybrid:rot_s0
echo STAGE0_ALL_DONE $(date +%H:%M:%S)
