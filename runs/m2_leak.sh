#!/usr/bin/env bash
# Diagnosis of the 2026-11-02 12:31 segmentation fault of s2s_s0 at iteration ~1,520 (host RSS 8.3 -> 10.9 GB over 1,500 iterations
# at --steps 512; WSL has 15 GB): host memory growth per iteration, measured on short runs from s2r_s0.
#   bash runs/m2_leak.sh <tag> <steps> <iters> [env assignments...]    e.g. bash runs/m2_leak.sh base512 512 60
#                                                                       bash runs/m2_leak.sh arena512 512 60 MALLOC_ARENA_MAX=2 MALLOC_TRIM_THRESHOLD_=131072
# Writes runs/meadow2/hybrid/leak_<tag>_rss.txt (seconds, RSS MB, iteration) and prints the growth per iteration after the compile.
cd /mnt/c/Users/nnels/Documents/projects/flock2 && . ~/.venvs/flock2/bin/activate && export XLA_PYTHON_CLIENT_PREALLOCATE=false
tag=$1; T=$2; iters=$3; shift 3
for kv in "$@"; do export "$kv"; done
A=~/flock2_data/m2_hybrid; n=leak_$tag; rm -rf $A/$n
R=${T}; RE=5120; [ "$T" = 128 ] && RE=4864
RECIPE="--willows 3 --kl_dyn 1.0 --replay 1024 --add_frac 0.5 --policy_input state_embed --column 256 --ppo_start 0 --fog_ramp 0,1,0.75 --skyline 2 --thirst 1.5 --stored_starts --mud_ramp 0,1,12"
python scripts/m2_hybrid.py --out $A/$n $RECIPE --steps $T --reset_every $RE --continue_from $A/s2r_s0 --iters $iters --seed 0 > runs/meadow2/hybrid/$n.log 2>&1 &
pid=$!
out=runs/meadow2/hybrid/${n}_rss.txt; : > $out; t0=$(date +%s)
while kill -0 $pid 2>/dev/null; do
  rss=$(ps -o rss= -p $pid 2>/dev/null | awk '{print int($1/1024)}')
  it=$(tail -1 $A/$n/log.jsonl 2>/dev/null | grep -o '"iter": [0-9]*' | grep -o '[0-9]*')
  echo "$(( $(date +%s) - t0 )) ${rss:-0} ${it:-0}" >> $out
  sleep 5
done
wait $pid; rc=$?
echo "LEAK $tag steps $T exit $rc $(date +%H:%M:%S)"
python - "$out" <<'PY'
import sys
rows = [tuple(int(x) for x in l.split()) for l in open(sys.argv[1]) if l.strip()]
rows = [r for r in rows if r[2] >= 10]  # after the compile and the first iterations
if len(rows) > 2:
    (t0, r0, i0), (t1, r1, i1) = rows[0], rows[-1]
    print(f"RSS {r0} MB at iteration {i0} -> {r1} MB at iteration {i1}: {(r1 - r0) / max(1, i1 - i0):.2f} MB an iteration; {(t1 - t0) / max(1, i1 - i0):.1f} s an iteration")
else:
    print("too few rows", rows[-3:] if rows else None)
PY
