# wait until the first long run (and its evaluation, decoder check, probe) is done, then stop the
# runner script and whatever it started next
LOG=/mnt/c/Users/nnels/Documents/projects/flock2/runs/m1
until [ -f $LOG/hybrid_long_s0.log ] || [ -f $LOG/eval/hybrid_long_bal_s0.json -a -f $LOG/probes/latent_mlp_hybrid_long_bal_s0_m32.json ]; do sleep 20; done
sleep 5
pkill -f 'm1_hybrid_lon[g].sh'
pkill -f 'scripts/m1_hybri[d].py.*long_s0'
sleep 3
echo "stopped; next run present: $(ls $LOG/hybrid_long_s0.log 2>/dev/null)"
pgrep -af 'm1_hybri[d]|m1_eva[l]|m1_prob[e]|m1_decode[r]' || echo "nothing left running"
