#!/usr/bin/env bash
# 2026-10-31: waits for chain4 (section 21.11) to end, completes s2n_s0 (runs/m2_fix_s2n_s0.sh), then starts the approved
# section 21.12 sequence (runs/m2_arms.sh). On TRAIN_FAILED it stops and leaves a marker instead.
cd /mnt/c/Users/nnels/Documents/projects/flock2
L=runs/meadow2/stage2a_sky3.log
while ! grep -qE "MS_ALL_DONE|TRAIN_FAILED" $L; do sleep 60; done
if grep -q TRAIN_FAILED $L; then echo AFTER_MS_ABORT $(date +%H:%M:%S) >> $L; exit 1; fi
bash runs/m2_fix_s2n_s0.sh >> $L 2>&1
nohup bash runs/m2_arms.sh > runs/meadow2/arms.log 2>&1 &
echo ARMS_LAUNCHED $(date +%H:%M:%S) >> $L
