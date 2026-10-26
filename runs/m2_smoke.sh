cd /mnt/c/Users/nnels/Documents/projects/flock2 && . ~/.venvs/flock2/bin/activate && export JAX_PLATFORMS=cpu
python scripts/m2_eval.py --policy scripted --meadows 2 --days 0.1 2>&1 | grep -v "^W\|^I0" | head -30
rm -rf /tmp/m2ppo_smoke
python scripts/m2_ppo.py --out /tmp/m2ppo_smoke --updates 2 --meadows 2 --steps 64 --reset_every 128 2>&1 | grep -v "^W\|^I0" | tail -3
python scripts/m2_eval.py --policy ppo --agent /tmp/m2ppo_smoke --meadows 2 --days 0.05 2>&1 | grep -v "^W\|^I0" | grep -E '"reward"|death_share|Error|Traceback' 
