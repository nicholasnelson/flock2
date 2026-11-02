#!/usr/bin/env bash
# Spec section 21.14 (the owner's go-ahead of 2026-11-02): the column's reach, --steps 512, from s2r on the 12-tile ring.
# A new file: runs/m2_ring12.sh is left alone (no script is edited while a shell may be executing it).
#   bash runs/m2_steps512.sh smoke         the smoke test: 4 iterations of the recipe at --steps 512 from s2r_s0 into a scratch brain,
#                                          the GPU's memory sampled every second; prints the host and GPU peaks and the seconds an
#                                          iteration takes
#   bash runs/m2_steps512.sh r <seed> [T]  train s2s_s<seed> from s2r_s<seed> at --steps T (default 512; s2s256_s<seed> at 256), then
#                                          its evaluations and probes (section 21.13.5's set, at 32 meadows)
#   bash runs/m2_steps512.sh collapse      after the three seeds: YES if S fails at two seeds with >= 40 % dead (the --steps 256 fallback)
#   bash runs/m2_steps512.sh chain         r 0, r 1, r 2 -> (collapse YES) r 0 256, r 1 256, r 2 256
#   nohup bash runs/m2_steps512.sh chain > runs/meadow2/steps512.log 2>&1 &
cd /mnt/c/Users/nnels/Documents/projects/flock2 && . ~/.venvs/flock2/bin/activate && export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p runs/meadow2/hybrid runs/meadow2/eval runs/meadow2/deaths runs/meadow2/diag
E=runs/meadow2/eval; X=runs/meadow2/deaths; G=runs/meadow2/diag; A=~/flock2_data/m2_hybrid
W12="--inner 12 --fog 0.75 --days 5 --meadows 32"
W8="--inner 8 --fog 0.75 --days 5 --meadows 32"
# the section 21.13 recipe as s2r carries it, less --steps and --reset_every (given per call: the one change, and the reset period it
# forces to a multiple of the rollout) and with the ring at 12 tiles from the first iteration (s2r was trained on it)
RECIPE="--willows 3 --kl_dyn 1.0 --replay 1024 --add_frac 0.5 --policy_input state_embed --column 256 --ppo_start 0 --fog_ramp 0,1,0.75 --skyline 2 --thirst 1.5 --stored_starts --mud_ramp 0,1,12"
reset_for() { if [ "$1" = 128 ]; then echo 4864; else echo 5120; fi; }  # 38 x 128 (section 21.10); 10 x 512 = 20 x 256, 2.13 days
ev() {  # ev <agent dir> <brain name> <tag> <flags...>: one evaluation with its deaths file and log
  local agent=$1 n=$2 tag=$3; shift 3
  python scripts/m2_eval.py --policy hybrid --agent $agent "$@" --out $E/hybrid_${n}_${tag}.json --deaths $X/hybrid_${n}_${tag}.json > $E/hybrid_${n}_${tag}.log 2>&1 || echo EVAL_FAILED $n $tag
}
train() {  # train <name> <from> <T> <iters> <seed> <log>
  local n=$1 from=$2 T=$3 iters=$4 s=$5 log=$6
  MAXGB=13 bash runs/watchdog.sh python scripts/m2_hybrid.py --out $A/$n $RECIPE --steps $T --reset_every $(reset_for $T) --continue_from $A/$from --iters $iters --seed $s > $log 2>&1 &
  local wd=$!
  # the host memory of the training, sampled every minute (seconds, RSS MB, iteration): section 21.14.7's fault was a memory one
  ( while kill -0 $wd 2>/dev/null; do echo "$(date +%s) $(ps -o rss= --ppid $wd 2>/dev/null | awk '{s+=$1} END {print int(s/1024)}') $(tail -1 $A/$n/log.jsonl 2>/dev/null | grep -o '"iter": [0-9]*' | grep -o '[0-9]*')" >> ${log%.log}_rss.txt; sleep 60; done ) &
  wait $wd
}
if [ "$1" = smoke ]; then
  n=smoke_steps512; rm -rf $A/$n; L=runs/meadow2/hybrid/$n.log
  echo SMOKE_START $(date +%H:%M:%S); echo "GPU before: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 1; done ) > ${L%.log}_gpu.txt 2>/dev/null & sampler=$!
  train $n s2r_s0 512 4 0 $L; rc=$?
  kill $sampler 2>/dev/null
  echo "SMOKE exit $rc; $(grep WATCHDOG $L | tail -1); GPU peak $(sort -n ${L%.log}_gpu.txt | tail -1) MB"
  [ $rc != 0 ] && tail -8 $L
  python - <<'PY'
import json
rows = [json.loads(l) for l in open("runs/meadow2/hybrid/smoke_steps512_log.jsonl")] if False else [json.loads(l) for l in open(__import__("os").path.expanduser("~/flock2_data/m2_hybrid/smoke_steps512/log.jsonl"))]
prev = 0.0
for r in rows:
    print(f"iter {r['iter']}: {r['t'] - prev:.1f} s (collect {r.get('t_collect', 0):.1f}, train {r.get('t_train', 0):.1f}); reward {r['reward']:.3f}, alive {r['alive_share']:.2f}, wm_recon {r.get('wm_recon', float('nan')):.1f}")
    prev = r['t']
PY
  echo SMOKE_DONE $(date +%H:%M:%S)
  exit
fi
if [ "$1" = collapse ]; then
  python - <<'PY'
import json, os
n = 0
for s in range(3):
    p = f"runs/meadow2/eval/hybrid_s2s_s{s}_sky_mud12_fog075_d5_m32.json"
    n += os.path.exists(p) and json.load(open(p))["death_share"] >= 0.40
print("YES" if n >= 2 else "NO")
PY
  exit
fi
if [ "$1" = r ]; then
  s=$2; T=${3:-512}; n=s2s_s$s; if [ "$T" != 512 ]; then n=s2s${T}_s$s; fi
  agent=$A/$n; rm -rf $agent  # a fresh start: the trainer appends to an existing log.jsonl
  train $n s2r_s$s $T 2000 $s runs/meadow2/hybrid/$n.log || { echo TRAIN_FAILED $n; tail -5 runs/meadow2/hybrid/$n.log; exit 1; }
  cp $agent/log.jsonl runs/meadow2/hybrid/${n}_log.jsonl
  echo TRAIN_DONE $n $(date +%H:%M:%S)
  # the evaluations (section 21.13.5's set): the ring world; the peaks blanked; the peaks hidden by fog; the 8-tile ring (the transfer);
  # the plain world with and without the peaks (16 meadows, against s2r's plain readings); alone; the column zeroed
  ev $agent $n sky_mud12_fog075_d5_m32 $W12
  ev $agent $n nosky_mud12_fog075_d5_m32 $W12 --skyline 0
  ev $agent $n sky1_mud12_fog075_d5_m32 $W12 --skyline 1
  ev $agent $n sky_mud8_fog075_d5_m32 $W8
  ev $agent $n sky_plain
  ev $agent $n plain --skyline 0
  ev $agent $n sky_mud12_fog075_d5_lone $W12 --sheep 1 --meadows 64
  ev $agent $n sky_mud12_fog075_d5_m32_zerocol $W12 --zero_column
  # the probes: on the clear ring world and the all-fog ring world
  python scripts/m2_diag.py --agent $agent --inner 12 --out $G/${n}_sky_mud12.json > $G/${n}_sky_mud12.log 2>&1 || echo DIAG_FAILED $n
  python scripts/m2_diag.py --agent $agent --inner 12 --fog 1.0 --out $G/${n}_sky_mud12_fog1.json > $G/${n}_sky_mud12_fog1.log 2>&1 || echo DIAG_FAILED $n fog
  echo STAGE_DONE $n $(date +%H:%M:%S)
  exit
fi
if [ "$1" = chain ]; then
  echo STEPS512_START $(date +%H:%M:%S)
  for s in 0 1 2; do bash runs/m2_steps512.sh r $s || { echo CHAIN_STOPPED $(date +%H:%M:%S); exit 1; }; echo $(date +%H:%M:%S); done  # a failed training stops the chain (section 21.14.6)
  echo R_ALL_DONE $(date +%H:%M:%S)
  C=$(bash runs/m2_steps512.sh collapse)
  echo COLLAPSE $C $(date +%H:%M:%S)
  if [ "$C" = YES ]; then
    for s in 0 1 2; do bash runs/m2_steps512.sh r $s 256 || { echo CHAIN_STOPPED $(date +%H:%M:%S); exit 1; }; echo $(date +%H:%M:%S); done
    echo R256_ALL_DONE $(date +%H:%M:%S)
  fi
  echo STEPS512_ALL_DONE $(date +%H:%M:%S)
  exit
fi
echo "usage: bash runs/m2_steps512.sh smoke|collapse|r <seed> [T]|chain"
