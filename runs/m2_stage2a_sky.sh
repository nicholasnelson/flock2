#!/usr/bin/env bash
# Spec section 21.7: the skyline, fixed peaks on the horizon (a compass the sheep must learn to read). Usage:
#   bash runs/m2_stage2a_sky.sh before       the fog-trained brains (s2f) on the skyline world, untrained on it: survival and the probes
#   bash runs/m2_stage2a_sky.sh smoke        5 training iterations with the skyline (nothing saved that matters)
#   bash runs/m2_stage2a_sky.sh <seed>       the learning stage: continue s2f_s<seed> on mud 8 + fog 0.75 + skyline, then the evaluations
#   bash runs/m2_stage2a_sky.sh chain        before, then seeds 0, 1, 2 in turn
#   bash runs/m2_stage2a_sky.sh more <seed>  fallback (1) of section 21.7: 2,000 more iterations, s2k_s<seed> -> s2l_s<seed>, then the same evaluations
#   bash runs/m2_stage2a_sky.sh chain2       fallback (1) for seeds 0, 1, 2 in turn
#   bash runs/m2_stage2a_sky.sh above <seed> spec section 21.10, peaks above the fog: s2l_s<seed> -> s2m_s<seed> with --skyline 2, then the evaluations (plus the stage world with --skyline 1)
#   bash runs/m2_stage2a_sky.sh chain3       section 21.10 for seeds 0, 1, 2 in turn
#   bash runs/m2_stage2a_sky.sh ms <seed>    spec section 21.11, the multi-step view loss on: s2l_s<seed> -> s2n_s<seed>, then the same evaluations
#   bash runs/m2_stage2a_sky.sh chain4       section 21.11 for seeds 0, 1, 2 in turn
cd /mnt/c/Users/nnels/Documents/projects/flock2 && . ~/.venvs/flock2/bin/activate && export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p runs/meadow2/hybrid runs/meadow2/eval runs/meadow2/deaths runs/meadow2/diag
E=runs/meadow2/eval; X=runs/meadow2/deaths; G=runs/meadow2/diag
W="--inner 8 --fog 0.75 --days 5"   # the stage world, 5 days
if [ "$1" = before ]; then
  echo BEFORE_START $(date +%H:%M:%S)
  for s in 0 1 2; do
    n=s2f_s$s; agent=~/flock2_data/m2_hybrid/$n
    python scripts/m2_eval.py --policy hybrid --agent $agent $W --skyline 1 --out $E/hybrid_${n}_sky_mud8_fog075_d5.json --deaths $X/hybrid_${n}_sky_mud8_fog075_d5.json > $E/hybrid_${n}_sky_mud8_fog075_d5.log 2>&1 || echo BEFORE_EVAL_FAILED $n
    python scripts/m2_diag.py --agent $agent --inner 8 --skyline 1 --out $G/${n}_sky_mud8.json > $G/${n}_sky_mud8.log 2>&1 || echo BEFORE_DIAG_FAILED $n
    echo BEFORE_DONE $n $(date +%H:%M:%S)
  done
  echo BEFORE_ALL_DONE $(date +%H:%M:%S)
  exit
fi
RECIPE="--willows 3 --kl_dyn 1.0 --replay 1024 --add_frac 0.5 --policy_input state_embed --column 256 --steps 128 --reset_every 4864 --ppo_start 0 --mud_ramp 0,1,8 --fog_ramp 0,1,0.75 --skyline"
if [ "$1" = smoke ]; then
  python scripts/m2_hybrid.py --out ~/flock2_data/m2_hybrid/smoke_sky $RECIPE --continue_from ~/flock2_data/m2_hybrid/s2f_s0 --iters 5 --seed 0 2>&1 | tail -3
  exit
fi
if [ "$1" = chain ]; then
  echo SKY_START $(date +%H:%M:%S)
  bash runs/m2_stage2a_sky.sh before
  for s in 0 1 2; do bash runs/m2_stage2a_sky.sh $s; echo $(date +%H:%M:%S); done
  echo SKY_ALL_DONE $(date +%H:%M:%S)
  exit
fi
if [ "$1" = chain2 ]; then
  echo SKY2_START $(date +%H:%M:%S)
  for s in 0 1 2; do bash runs/m2_stage2a_sky.sh more $s; echo $(date +%H:%M:%S); done
  echo SKY2_ALL_DONE $(date +%H:%M:%S)
  exit
fi
if [ "$1" = chain3 ]; then
  echo ABOVE_START $(date +%H:%M:%S)
  for s in 0 1 2; do bash runs/m2_stage2a_sky.sh above $s; echo $(date +%H:%M:%S); done
  echo ABOVE_ALL_DONE $(date +%H:%M:%S)
  exit
fi
if [ "$1" = chain4 ]; then
  echo MS_START $(date +%H:%M:%S)
  for s in 0 1 2; do bash runs/m2_stage2a_sky.sh ms $s; echo $(date +%H:%M:%S); done
  echo MS_ALL_DONE $(date +%H:%M:%S)
  exit
fi
# the learning stage: the fog recipe with mud 8 and fog 0.75 held from the start (the brains know both) and the skyline on every meadow
EXTRA=""
if [ "$1" = more ]; then s=$2; n=s2l_s$s; from=s2k_s$s
elif [ "$1" = above ]; then s=$2; n=s2m_s$s; from=s2l_s$s; EXTRA="--skyline 2"   # spec section 21.10: the peaks drawn on fog days too (the last --skyline wins)
elif [ "$1" = ms ]; then s=$2; n=s2n_s$s; from=s2l_s$s; EXTRA="--ms_weight 0.4 --ms_horizons 8,16,24 --ms_stride 8"   # spec section 21.11: the multi-step view loss, weighted to match recon at the start
else s=$1; n=s2k_s$s; from=s2f_s$s; fi
agent=~/flock2_data/m2_hybrid/$n
MAXGB=12 bash runs/watchdog.sh python scripts/m2_hybrid.py --out $agent $RECIPE $EXTRA --continue_from ~/flock2_data/m2_hybrid/$from --iters 2000 --seed $s > runs/meadow2/hybrid/$n.log 2>&1 || { echo TRAIN_FAILED $n; tail -5 runs/meadow2/hybrid/$n.log; exit 1; }
cp $agent/log.jsonl runs/meadow2/hybrid/${n}_log.jsonl
echo TRAIN_DONE $n $(date +%H:%M:%S)
# the evaluations: the stage world; the ablation (skyline blanked); the plain world (stage 1's regression, with and without the skyline); alone; the column zeroed
python scripts/m2_eval.py --policy hybrid --agent $agent $W --out $E/hybrid_${n}_sky_mud8_fog075_d5.json --deaths $X/hybrid_${n}_sky_mud8_fog075_d5.json > $E/hybrid_${n}_sky_mud8_fog075_d5.log 2>&1 || echo EVAL_FAILED $n
python scripts/m2_eval.py --policy hybrid --agent $agent $W --skyline 0 --out $E/hybrid_${n}_nosky_mud8_fog075_d5.json --deaths $X/hybrid_${n}_nosky_mud8_fog075_d5.json > $E/hybrid_${n}_nosky_mud8_fog075_d5.log 2>&1 || echo EVAL_FAILED $n ablation
if [ "$1" = above ]; then  # section 21.10's second ablation: the peaks hidden by the fog again (the old world)
  python scripts/m2_eval.py --policy hybrid --agent $agent $W --skyline 1 --out $E/hybrid_${n}_sky1_mud8_fog075_d5.json --deaths $X/hybrid_${n}_sky1_mud8_fog075_d5.json > $E/hybrid_${n}_sky1_mud8_fog075_d5.log 2>&1 || echo EVAL_FAILED $n sky1
fi
python scripts/m2_eval.py --policy hybrid --agent $agent --out $E/hybrid_${n}_sky_plain.json > $E/hybrid_${n}_sky_plain.log 2>&1 || echo EVAL_FAILED $n plain
python scripts/m2_eval.py --policy hybrid --agent $agent --skyline 0 --out $E/hybrid_${n}_plain.json > $E/hybrid_${n}_plain.log 2>&1 || echo EVAL_FAILED $n plain-nosky
python scripts/m2_eval.py --policy hybrid --agent $agent $W --sheep 1 --meadows 64 --out $E/hybrid_${n}_sky_mud8_fog075_d5_lone.json > $E/hybrid_${n}_sky_mud8_fog075_d5_lone.log 2>&1 || echo EVAL_FAILED $n lone
python scripts/m2_eval.py --policy hybrid --agent $agent $W --zero_column --out $E/hybrid_${n}_sky_mud8_fog075_d5_zerocol.json > $E/hybrid_${n}_sky_mud8_fog075_d5_zerocol.log 2>&1 || echo EVAL_FAILED $n zerocol
# the probes: heading from one frame and the water's bearing in the world frame, on the clear mud world and the all-fog mud world
python scripts/m2_diag.py --agent $agent --inner 8 --out $G/${n}_sky_mud8.json > $G/${n}_sky_mud8.log 2>&1 || echo DIAG_FAILED $n
python scripts/m2_diag.py --agent $agent --inner 8 --fog 1.0 --out $G/${n}_sky_mud8_fog1.json > $G/${n}_sky_mud8_fog1.log 2>&1 || echo DIAG_FAILED $n fog
echo STAGE_DONE $n $(date +%H:%M:%S)
