#!/usr/bin/env bash
# Spec section 21.13 (the owner's go-ahead of 2026-11-01): the 12-tile mud ring on arm T's recipe, with arm W's fix.
# A new file, since runs/m2_stage2a_sky.sh must never be edited while a shell may be executing it (and nothing else should be either).
#   bash runs/m2_ring12.sh demand        the demand check: s2q_s{0,1,2} evaluated on the 12-tile ring at 32 meadows, and on the
#                                        8-tile ring at 32 meadows (the reference at the new evaluation size); no training
#   bash runs/m2_ring12.sh gate          read the demand check against the pre-registered gate (section 21.13.4); prints GO or STOP
#   bash runs/m2_ring12.sh r <seed> [I]  train s2r_s<seed> from s2q_s<seed> at ring I (default 12; s2r10_s<seed> at 10), then its
#                                        evaluations and probes
#   bash runs/m2_ring12.sh collapse      after the three seeds: YES if S fails at two seeds with >= 40 % dead (the inner-10 fallback)
#   bash runs/m2_ring12.sh chain         demand -> gate -> (GO) r 0, r 1, r 2 -> (collapse YES) r 0 10, r 1 10, r 2 10
#   nohup bash runs/m2_ring12.sh chain > runs/meadow2/ring12.log 2>&1 &
cd /mnt/c/Users/nnels/Documents/projects/flock2 && . ~/.venvs/flock2/bin/activate && export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p runs/meadow2/hybrid runs/meadow2/eval runs/meadow2/deaths runs/meadow2/diag
E=runs/meadow2/eval; X=runs/meadow2/deaths; G=runs/meadow2/diag; A=~/flock2_data/m2_hybrid
W12="--inner 12 --fog 0.75 --days 5 --meadows 32"
W8="--inner 8 --fog 0.75 --days 5 --meadows 32"
# the section 21.10 recipe as s2q carries it (thirst 1.5, the peaks drawn on fog days), plus arm W's stored starts; the mud ramp is given per call
RECIPE="--willows 3 --kl_dyn 1.0 --replay 1024 --add_frac 0.5 --policy_input state_embed --column 256 --steps 128 --reset_every 4864 --ppo_start 0 --fog_ramp 0,1,0.75 --skyline 2 --thirst 1.5 --stored_starts"
ev() {  # ev <agent dir> <brain name> <tag> <flags...>: one evaluation with its deaths file and log
  local agent=$1 n=$2 tag=$3; shift 3
  python scripts/m2_eval.py --policy hybrid --agent $agent "$@" --out $E/hybrid_${n}_${tag}.json --deaths $X/hybrid_${n}_${tag}.json > $E/hybrid_${n}_${tag}.log 2>&1 || echo EVAL_FAILED $n $tag
}
if [ "$1" = demand ]; then
  echo DEMAND_START $(date +%H:%M:%S)
  for s in 0 1 2; do
    n=s2q_s$s
    ev $A/$n $n sky_mud12_fog075_d5_m32 $W12
    ev $A/$n $n sky_mud8_fog075_d5_m32 $W8
    echo DEMAND_DONE $n $(date +%H:%M:%S)
  done
  echo DEMAND_ALL_DONE $(date +%H:%M:%S)
  exit
fi
if [ "$1" = gate ]; then
  python - <<'EOF'
import json, os, sys
# section 21.13.4: a seed is "pressed" on the 12-tile ring if the brain does not live comfortably there (dead >= 15 %, last-day energy
# <= 0.85, or thirsty steps >= 5 %) or spends >= 25 % of its night steps beyond 12 tiles (the night return is forced). Two seeds pressed -> GO.
go = 0
for s in range(3):
    p = f"runs/meadow2/eval/hybrid_s2q_s{s}_sky_mud12_fog075_d5_m32.json"
    if not os.path.exists(p):
        print(f"# seed {s}: {p} missing", file=sys.stderr); print("MISSING"); sys.exit()
    d = json.load(open(p))
    dead = d["death_share"]; en = d["last_day_levels"]["energy"]; th = d["thirsty_share_of_living_steps"]
    night = d["wdist_share"]["night"]; far = sum(v for k, v in night.items() if k != "steps" and int(k.split("-")[0]) >= 12)
    ex = d["excursions"]["all"]; nfar = sum(b["n"] for k, b in ex.items() if int(k.split("-")[0]) >= 12)
    per_day = nfar / max(1.0, d["meadows"] * d["sheep"] * d["days_survived_mean"])
    pressed = dead >= 0.15 or en <= 0.85 or th >= 0.05 or far >= 0.25
    go += pressed
    print(f"# seed {s}: dead {100 * dead:.1f} %, last-day energy {en:.2f}, thirsty {100 * th:.1f} % of steps, night share beyond 12 tiles {far:.2f}, "
          f"excursions beyond 12 per sheep-day {per_day:.2f} -> {'pressed' if pressed else 'not pressed'}", file=sys.stderr)
print("GO" if go >= 2 else "STOP")
EOF
  exit
fi
if [ "$1" = collapse ]; then
  python - <<'EOF'
import json, os
n = 0
for s in range(3):
    p = f"runs/meadow2/eval/hybrid_s2r_s{s}_sky_mud12_fog075_d5_m32.json"
    n += os.path.exists(p) and json.load(open(p))["death_share"] >= 0.40
print("YES" if n >= 2 else "NO")
EOF
  exit
fi
if [ "$1" = r ]; then
  s=$2; I=${3:-12}; n=s2r_s$s; if [ "$I" != 12 ]; then n=s2r${I}_s$s; fi
  agent=$A/$n
  MAXGB=12 bash runs/watchdog.sh python scripts/m2_hybrid.py --out $agent $RECIPE --mud_ramp 0,200,$I --continue_from $A/s2q_s$s --iters 2000 --seed $s > runs/meadow2/hybrid/$n.log 2>&1 || { echo TRAIN_FAILED $n; tail -5 runs/meadow2/hybrid/$n.log; exit 1; }
  cp $agent/log.jsonl runs/meadow2/hybrid/${n}_log.jsonl
  echo TRAIN_DONE $n $(date +%H:%M:%S)
  WI="--inner $I --fog 0.75 --days 5 --meadows 32"
  # the evaluations (section 21.13.5): the ring world; the peaks blanked; the peaks hidden by fog; the 8-tile ring (the transfer); the plain
  # world with and without the peaks (stage 1's regression, 16 meadows as the s2q reference); alone; the column zeroed
  ev $agent $n sky_mud${I}_fog075_d5_m32 $WI
  ev $agent $n nosky_mud${I}_fog075_d5_m32 $WI --skyline 0
  ev $agent $n sky1_mud${I}_fog075_d5_m32 $WI --skyline 1
  ev $agent $n sky_mud8_fog075_d5_m32 $W8
  ev $agent $n sky_plain
  ev $agent $n plain --skyline 0
  ev $agent $n sky_mud${I}_fog075_d5_lone $WI --sheep 1 --meadows 64
  ev $agent $n sky_mud${I}_fog075_d5_m32_zerocol $WI --zero_column
  # the probes: on the clear ring world and the all-fog ring world
  python scripts/m2_diag.py --agent $agent --inner $I --out $G/${n}_sky_mud${I}.json > $G/${n}_sky_mud${I}.log 2>&1 || echo DIAG_FAILED $n
  python scripts/m2_diag.py --agent $agent --inner $I --fog 1.0 --out $G/${n}_sky_mud${I}_fog1.json > $G/${n}_sky_mud${I}_fog1.log 2>&1 || echo DIAG_FAILED $n fog
  echo STAGE_DONE $n $(date +%H:%M:%S)
  exit
fi
if [ "$1" = chain ]; then
  echo RING12_START $(date +%H:%M:%S)
  bash runs/m2_ring12.sh demand
  V=$(bash runs/m2_ring12.sh gate)
  echo GATE $V $(date +%H:%M:%S)
  if [ "$V" != GO ]; then echo RING12_ALL_DONE $(date +%H:%M:%S); exit; fi
  for s in 0 1 2; do bash runs/m2_ring12.sh r $s; echo $(date +%H:%M:%S); done
  echo R_ALL_DONE $(date +%H:%M:%S)
  C=$(bash runs/m2_ring12.sh collapse)
  echo COLLAPSE $C $(date +%H:%M:%S)
  if [ "$C" = YES ]; then
    for s in 0 1 2; do bash runs/m2_ring12.sh r $s 10; echo $(date +%H:%M:%S); done
    echo R10_ALL_DONE $(date +%H:%M:%S)
  fi
  echo RING12_ALL_DONE $(date +%H:%M:%S)
  exit
fi
echo "usage: bash runs/m2_ring12.sh demand|gate|collapse|r <seed> [I]|chain"
