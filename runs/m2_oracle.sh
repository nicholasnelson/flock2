#!/usr/bin/env bash
# Stage 0, the myopic-oracle ladder (spec § 19.2): privileged PPO with water hidden beyond a range
# and ideal memories. Usage: bash runs/m2_oracle.sh smoke | bash runs/m2_oracle.sh <arm> <seed>
#   arms: r4 r8 r16 r8_rot30 r8_place30 r8_place1000
cd /mnt/c/Users/nnels/Documents/projects/flock2 && . ~/.venvs/flock2/bin/activate && export XLA_PYTHON_CLIENT_PREALLOCATE=false
args_for() {
  case "$1" in
    r4) echo "--water_range 4" ;;
    r8) echo "--water_range 8" ;;
    r16) echo "--water_range 16" ;;
    r8_rot30) echo "--water_range 8 --memory rot --window 30" ;;
    r8_place30) echo "--water_range 8 --memory place --window 30" ;;
    r8_place1000) echo "--water_range 8 --memory place --window 1000" ;;
    *) echo "unknown arm $1" >&2; exit 2 ;;
  esac
}
if [ "$1" = smoke ]; then
  rm -rf ~/flock2_data/m2_ppo/smoke_oracle
  MAXGB=12 bash runs/watchdog.sh python scripts/m2_ppo.py --out ~/flock2_data/m2_ppo/smoke_oracle $(args_for r8_rot30) --updates 3 --meadows 4 > /tmp/oracle_smoke.log 2>&1
  grep -E '"update"|WATCHDOG|Error|Traceback' /tmp/oracle_smoke.log | cut -c1-200
  python scripts/m2_eval.py --policy ppo --agent ~/flock2_data/m2_ppo/smoke_oracle --meadows 2 --days 0.5 --out /tmp/oracle_smoke_eval.json --deaths /tmp/oracle_smoke_deaths.json 2>&1 | tail -3
  python -c "import json; d=json.load(open('/tmp/oracle_smoke_eval.json')); print({k: d[k] for k in ('death_share','thirsty_sees_water_8','information_gap_8','within_8_share_of_living_steps','forensics')})"
  exit
fi
arm=$1; seed=$2; n=${arm}_s${seed}
mkdir -p runs/meadow2/ppo runs/meadow2/eval runs/meadow2/deaths
MAXGB=12 bash runs/watchdog.sh python scripts/m2_ppo.py --out ~/flock2_data/m2_ppo/$n $(args_for $arm) --seed $seed > runs/meadow2/ppo/$n.log 2>&1 || { echo TRAIN_FAILED $n; tail -5 runs/meadow2/ppo/$n.log; exit 1; }
cp ~/flock2_data/m2_ppo/$n/log.jsonl runs/meadow2/ppo/${n}_log.jsonl
python scripts/m2_eval.py --policy ppo --agent ~/flock2_data/m2_ppo/$n --out runs/meadow2/eval/ppo_$n.json --deaths runs/meadow2/deaths/ppo_$n.json > runs/meadow2/eval/ppo_$n.log 2>&1 || echo EVAL_FAILED $n
grep WATCHDOG runs/meadow2/ppo/$n.log; echo ORACLE_DONE $n
