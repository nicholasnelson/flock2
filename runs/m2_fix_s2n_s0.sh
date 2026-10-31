#!/usr/bin/env bash
# 2026-10-31: runs/m2_stage2a_sky.sh was edited at 13:50 while chain4's "ms 0" shell was still executing it; bash reads a
# script incrementally, so after s2n_s0's training (exit 0, 2,000 iterations, 13:50) that shell ran the new blindleg block's
# evaluations (sky, nosky, sky1: the same commands and seeds as the stage ones) and died on a syntax error. Missing for
# s2n_s0, run here with the stage script's own commands: the training-log copy, the plain-world, lone and zero-column
# evaluations and the two probes. Never edit a script while a bash is running it; this file is new for that reason.
cd /mnt/c/Users/nnels/Documents/projects/flock2 && . ~/.venvs/flock2/bin/activate && export XLA_PYTHON_CLIENT_PREALLOCATE=false
E=runs/meadow2/eval; X=runs/meadow2/deaths; G=runs/meadow2/diag
W="--inner 8 --fog 0.75 --days 5"
n=s2n_s0; agent=~/flock2_data/m2_hybrid/$n
echo FIX_START $n $(date +%H:%M:%S)
cp $agent/log.jsonl runs/meadow2/hybrid/${n}_log.jsonl
python scripts/m2_eval.py --policy hybrid --agent $agent --out $E/hybrid_${n}_sky_plain.json > $E/hybrid_${n}_sky_plain.log 2>&1 || echo EVAL_FAILED $n plain
python scripts/m2_eval.py --policy hybrid --agent $agent --skyline 0 --out $E/hybrid_${n}_plain.json > $E/hybrid_${n}_plain.log 2>&1 || echo EVAL_FAILED $n plain-nosky
python scripts/m2_eval.py --policy hybrid --agent $agent $W --sheep 1 --meadows 64 --out $E/hybrid_${n}_sky_mud8_fog075_d5_lone.json > $E/hybrid_${n}_sky_mud8_fog075_d5_lone.log 2>&1 || echo EVAL_FAILED $n lone
python scripts/m2_eval.py --policy hybrid --agent $agent $W --zero_column --out $E/hybrid_${n}_sky_mud8_fog075_d5_zerocol.json > $E/hybrid_${n}_sky_mud8_fog075_d5_zerocol.log 2>&1 || echo EVAL_FAILED $n zerocol
python scripts/m2_diag.py --agent $agent --inner 8 --out $G/${n}_sky_mud8.json > $G/${n}_sky_mud8.log 2>&1 || echo DIAG_FAILED $n
python scripts/m2_diag.py --agent $agent --inner 8 --fog 1.0 --out $G/${n}_sky_mud8_fog1.json > $G/${n}_sky_mud8_fog1.log 2>&1 || echo DIAG_FAILED $n fog
echo FIX_DONE $n $(date +%H:%M:%S)
