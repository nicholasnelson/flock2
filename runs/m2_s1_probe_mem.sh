cd /mnt/c/Users/nnels/Documents/projects/flock2 && . ~/.venvs/flock2/bin/activate && export XLA_PYTHON_CLIENT_PREALLOCATE=false
MS=${MS:-1.0}
rm -rf ~/flock2_data/m2_hybrid/smoke_s1
MAXGB=9 bash runs/watchdog.sh python scripts/m2_hybrid.py --out ~/flock2_data/m2_hybrid/smoke_s1 --kl_dyn 1.0 --replay 256 --add_frac 0.5 --seed 0 --model spatial --ms_weight $MS --batch 4 --iters 3 --ppo_start 1 > /tmp/s1_smoke.log 2>&1
tail -3 /tmp/s1_smoke.log | cut -c1-300; grep WATCHDOG /tmp/s1_smoke.log
