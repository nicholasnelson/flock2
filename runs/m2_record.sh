#!/usr/bin/env bash
# The labelled replays for the top-down viewer (web/replay/), one meadow each, written to
# docs/flock-2/meadow2-results/replays/ with index.json refreshed after each. Usage:
#   bash runs/m2_record.sh            all seven
#   bash runs/m2_record.sh a d g      a selection (letters a-h, below)
# GPU rule: if another job holds more than 9,000 MiB of the GPU, the recording runs on the CPU instead.
cd /mnt/c/Users/nnels/Documents/projects/flock2 && . ~/.venvs/flock2/bin/activate && export XLA_PYTHON_CLIENT_PREALLOCATE=false
B=~/flock2_data/m2_hybrid
OUT=docs/flock-2/meadow2-results/replays

rec() {
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  if [ -n "$used" ] && [ "$used" -gt 9000 ]; then
    echo "RECORD gpu busy (${used} MiB): running on the CPU"; export JAX_PLATFORMS=cpu
  else
    unset JAX_PLATFORMS
  fi
  echo RECORD_START "$2" $(date +%H:%M:%S)
  python scripts/m2_record.py --dir $OUT "$@" || echo RECORD_FAILED "$2"
  echo RECORD_DONE "$2" $(date +%H:%M:%S)
}

run_one() {
  case "$1" in
    a) rec --name a_scripted_plain --policy scripted --willows 3 --days 3 \
         --label "a. Scripted flock, plain world" \
         --description "The hand-written reference policy (it is told where the water is) on the plain world with 3 willows per pond, 3 days. What competent use of the meadow looks like." ;;
    b) rec --name b_scripted_mud8_fog --policy scripted --willows 3 --inner 8 --fog 0.75 --days 5 \
         --label "b. Scripted flock, stage 2a world (mud 8 + fog)" \
         --description "The scripted flock on stage 2a's world: a mud ring of 8 tiles round each pond (no grass there) and 3 days in 4 foggy, 5 days. The positive control for the memory rule: it knows where the water is, so it returns from beyond the ring in fog without searching." ;;
    c) rec --name c_a3_plain --policy hybrid --agent $B/a3_s0 --days 3 \
         --label "c. Stage-1 brain A3, plain world" \
         --description "The stage-1 brain A3 (embed hybrid: world model + PPO, trained with willows) on the plain world, 3 days. It finds and uses the water it can see." ;;
    d) rec --name d_s2a_mud8 --policy hybrid --agent $B/s2a_s0 --inner 8 --days 5 \
         --label "d. Stage-2a brain, mud ring 8" \
         --description "The stage-2a brain s2a_s0 (GRU column added, trained on the mud ramp up to 8 tiles) with the 8-tile mud ring, clear days, 5 days. It grazes beyond the mud and comes back to drink." ;;
    e) rec --name e_s2a_mud8_fog --policy hybrid --agent $B/s2a_s0 --inner 8 --fog 0.75 --days 5 \
         --label "e. Stage-2a brain in fog (before fog training)" \
         --description "The same stage-2a brain s2a_s0 on the fog world (mud 8, 3 days in 4 foggy), 5 days, before any fog training: in the evaluations 39-67 % of the sheep die within 5 days." ;;
    f) rec --name f_s2f_mud8_fog --policy hybrid --agent $B/s2f_s0 --inner 8 --fog 0.75 --days 5 \
         --label "f. Fog-trained brain s2f in fog" \
         --description "The fog-trained brain s2f_s0 (continued from s2a_s0 with the fog probability ramped to 0.75) on the fog world (mud 8), 5 days. It survives by grazing where the mud ring stays visible in fog and keeping its hydration near full." ;;
    g) rec --name g_s2f_mud8_fog_thirst3 --policy hybrid --agent $B/s2f_s0 --inner 8 --fog 0.75 --thirst 3 --days 5 \
         --label "g. Fog-trained brain s2f, thirst x3" \
         --description "The fog-trained brain s2f_s0 on the fog world (mud 8) with hydration decaying three times faster, 5 days. The sheep stay closer to the water still; in the evaluations 34 % die." ;;
    h) FLOCK2_TILES=72 rec --name h_scripted_river --policy scripted --willows 3 --inner 8 --river 1 --skyline 1 --tiles 72 --days 3          --label "h. Scripted flock, river meadow (72 tiles, skyline)"          --description "The scripted flock on the river meadow (spec section 21.8): 72 x 72 tiles, one straight river through the centre at a random angle, uncrossable, willows along both banks, a mud band of 8 tiles, the skyline on; clear days, 3 days. The layout only: the scripted policy knows where the water is. Nothing has trained on this world yet." ;;
    *) echo "unknown run $1 (a-h)" ;;
  esac
}

if [ $# -eq 0 ]; then set -- a b c d e f g; fi
for r in "$@"; do run_one "$r"; done
echo RECORD_ALL_DONE $(date +%H:%M:%S)
