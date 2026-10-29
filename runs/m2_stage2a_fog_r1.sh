#!/usr/bin/env bash
# Spec section 21.6's diagnostic R1: the trained fog brains, the stage-2a brains and the scripted control on
# mud 8 + fog 0.75, 5 days, with the excursion / grazing-distance / thirsty-line readings (new files *_r1.json;
# the registered results of the learning stage are left as they are).
cd /mnt/c/Users/nnels/Documents/projects/flock2 && . ~/.venvs/flock2/bin/activate && export XLA_PYTHON_CLIENT_PREALLOCATE=false
E=runs/meadow2/eval
echo R1_START $(date +%H:%M:%S)
for n in s2f_s0 s2f_s1 s2f_s2 s2a_s0 s2a_s1 s2a_s2; do
  python scripts/m2_eval.py --policy hybrid --agent ~/flock2_data/m2_hybrid/$n --inner 8 --fog 0.75 --days 5 --out $E/hybrid_${n}_mud8_fog075_d5_r1.json > $E/hybrid_${n}_mud8_fog075_d5_r1.log 2>&1 || echo R1_FAILED $n
  echo R1_DONE $n $(date +%H:%M:%S)
done
python scripts/m2_eval.py --policy scripted --willows 3 --inner 8 --fog 0.75 --days 5 --meadows 64 --out $E/scripted_mud8_fog075_d5_m64_r1.json > $E/scripted_mud8_fog075_d5_m64_r1.log 2>&1 || echo R1_FAILED scripted
echo R1_ALL_DONE $(date +%H:%M:%S)
