#!/usr/bin/env bash
# Spec section 21.12, the owner's approval of 2026-10-31: after section 21.11 is read, in this order, one GPU job at a time:
#   the blind-leg baseline on s2m (section 21.12.3), arm W (chain5, s2m -> s2p), the thirst checks (the section 11 balance
#   rule at 3, 2 and 1.5), then arm T (chain6, s2m -> s2q) at the largest factor at which the naive explorer that turns when
#   blocked still finds water before it would die in at least half of starts on the plain world.
#   bash runs/m2_arms.sh          the whole sequence (log it: nohup bash runs/m2_arms.sh > runs/meadow2/arms.log 2>&1 &)
#   bash runs/m2_arms.sh factor   only print the factor the checks allow
cd /mnt/c/Users/nnels/Documents/projects/flock2 && . ~/.venvs/flock2/bin/activate && export XLA_PYTHON_CLIENT_PREALLOCATE=false
pick_factor() {
  python - <<'EOF'
import json, os
for F in ("3", "2", "1.5"):
    p = f"runs/meadow2/checks_thirst{F}_plain.json"
    if os.path.exists(p):
        v = json.load(open(p))["turns_when_blocked"]["before_death"]["need_scale_3"]
        print(f"# thirst {F}: naive explorer finds water before death in {100 * v:.0f} % of starts", file=__import__("sys").stderr)
        if v >= 0.5:
            print(F)
            break
else:
    print("none")
EOF
}
if [ "$1" = factor ]; then pick_factor; exit; fi
echo ARMS_START $(date +%H:%M:%S)
bash runs/m2_stage2a_sky.sh blindleg
bash runs/m2_stage2a_sky.sh chain5
bash runs/m2_stage2a_sky.sh thirstcheck
F=$(pick_factor)
echo THIRST_FACTOR $F $(date +%H:%M:%S)
if [ "$F" = none ]; then echo T_SKIPPED_RULE $(date +%H:%M:%S); echo ARMS_ALL_DONE $(date +%H:%M:%S); exit; fi
bash runs/m2_stage2a_sky.sh chain6 $F
echo ARMS_ALL_DONE $(date +%H:%M:%S)
