cd /mnt/c/Users/nnels/Documents/projects/flock2 && . ~/.venvs/flock2/bin/activate && export XLA_PYTHON_CLIENT_PREALLOCATE=false
ARGS="--kl_dyn 1.0 --replay 1024 --add_frac 0.5 --ms_weight 1.0 --ms_horizons 2,4,8,16,32 --ms_stride 16 --length 64 --batch 4 --policy_input state_embed --seed 0"
if [ "$1" = smoke ]; then
  rm -rf ~/flock2_data/m2_hybrid/smoke_mem
  MAXGB=12 bash runs/watchdog.sh python scripts/m2_hybrid.py --out ~/flock2_data/m2_hybrid/smoke_mem $ARGS --replay 64 --iters 3 --ppo_start 1 > /tmp/mem_smoke.log 2>&1
  grep -E '"iter"|WATCHDOG|Error|Traceback' /tmp/mem_smoke.log | cut -c1-250; exit
fi
n=mem_s0
MAXGB=12 bash runs/watchdog.sh python scripts/m2_hybrid.py --out ~/flock2_data/m2_hybrid/$n $ARGS > runs/meadow2/hybrid/$n.log 2>&1 || { echo TRAIN_FAILED; tail -5 runs/meadow2/hybrid/$n.log; exit 1; }
cp ~/flock2_data/m2_hybrid/$n/log.jsonl runs/meadow2/hybrid/${n}_log.jsonl
python scripts/m2_eval.py --policy hybrid --agent ~/flock2_data/m2_hybrid/$n --out runs/meadow2/eval/hybrid_$n.json > runs/meadow2/eval/hybrid_$n.log 2>&1 || echo EVAL_FAILED
python scripts/m2_diag.py --agent ~/flock2_data/m2_hybrid/$n --out runs/meadow2/diag/$n.json > runs/meadow2/diag/$n.log 2>&1 || echo DIAG_FAILED
grep WATCHDOG runs/meadow2/hybrid/$n.log; echo MEM_DONE
