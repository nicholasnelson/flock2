cd /mnt/c/Users/nnels/Documents/projects/flock2 && . ~/.venvs/flock2/bin/activate && export JAX_PLATFORMS=cpu
for m in learned beacon; do
  rm -rf /tmp/gf_$m
  python scripts/grid_flock_hybrid.py --out /tmp/gf_$m --signal $m --iters 3 --ppo_start 1 --worlds 4 --steps 32 --reset_every 64 --wm_steps 4 2>&1 | grep -v "^W\|^I0" | tail -3
done
