#!/usr/bin/env bash
# Spec section 21.14.7, continued (2) (2026-11-02): the third launch of the --steps 512 arm died with the WSL machine when Windows ran
# out of memory (other programs, and a 12.7 GB nonpaged kernel pool). This launcher runs on the WINDOWS side (Git Bash), waits for a
# quiet machine, runs a 12-iteration health run of the trainer under a GPU-memory cap, and only then launches the chain.
#   bash runs/m2_steps512_quiet.sh            (from the repo root, in the background)
#   touch runs/meadow2/steps512_quiet.stop    cancels the wait
# Gates (every two minutes, three consecutive samples): Windows's available memory >= MINAVAIL MB and the GPU's used memory <= MAXGPU MB.
# The cap: XLA_PYTHON_CLIENT_MEM_FRACTION=$MEMFRAC bounds the trainer's GPU allocator (preallocation stays off); the trainer's own
# GPU need measured 4.9-8.9 GB with the allocator's slack, so .5 of the 16 GB card (8.2 GB) is tested by the health run first.
cd /c/Users/nnels/Documents/projects/flock2 || exit 1
LOG=runs/meadow2/steps512_quiet.log
STOP=runs/meadow2/steps512_quiet.stop
MINAVAIL=${MINAVAIL:-15000}
MAXGPU=${MAXGPU:-5000}
MEMFRAC=${MEMFRAC:-.5}
WSLREPO=/mnt/c/Users/nnels/Documents/projects/flock2
rm -f $STOP
ps1() { powershell -NoProfile -Command "$1" 2>/dev/null | tr -d '\r' | tail -1; }
echo "QUIET_WAIT $(date +%F_%T) gates avail>=${MINAVAIL}MB gpu<=${MAXGPU}MB x3; cap MEMFRAC=$MEMFRAC" >> $LOG
ok=0
while true; do
  [ -e $STOP ] && { echo "QUIET_STOPPED $(date +%F_%T)" >> $LOG; exit 2; }
  avail=$(ps1 "[int](Get-Counter '\Memory\Available MBytes').CounterSamples.CookedValue")
  pool=$(ps1 "[int]((Get-Counter '\Memory\Pool Nonpaged Bytes').CounterSamples.CookedValue/1MB)")
  gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d '\r ')
  echo "$(date +%F_%T) avail ${avail:-?} pool ${pool:-?} gpu ${gpu:-?} ok $ok" >> $LOG
  if [ -n "$avail" ] && [ -n "$gpu" ] && [ "$avail" -ge "$MINAVAIL" ] && [ "$gpu" -le "$MAXGPU" ]; then ok=$((ok + 1)); else ok=0; fi
  [ $ok -ge 3 ] && break
  sleep 120
done
echo "QUIET_OK $(date +%F_%T) avail $avail pool $pool gpu $gpu" >> $LOG
if wsl -e bash -lc "ps -eo args | grep -q '[s]cripts/m2_hybrid.py\|[m]2_steps512.sh'"; then echo "BUSY_WSL $(date +%F_%T): something of ours already runs; not launching" >> $LOG; exit 3; fi
# the health run under the cap: 12 iterations from s2r_s0 at 512 steps (runs/m2_leak.sh), the GPU's used memory sampled every 2 s
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d '\r '; sleep 2; done ) > runs/meadow2/hybrid/leak_healthcap_gpu.txt & sampler=$!
echo "HEALTH_START $(date +%F_%T)" >> $LOG
wsl -e bash -lc "cd $WSLREPO && bash runs/m2_leak.sh healthcap 512 12 XLA_PYTHON_CLIENT_MEM_FRACTION=$MEMFRAC && grep -q '\"iter\": 12,' ~/flock2_data/m2_hybrid/leak_healthcap/log.jsonl && echo HEALTH_OK" >> $LOG 2>&1
kill $sampler 2>/dev/null
echo "HEALTH_END $(date +%F_%T) GPU peak $(sort -n runs/meadow2/hybrid/leak_healthcap_gpu.txt | tail -1) MB; avail now $(ps1 "[int](Get-Counter '\Memory\Available MBytes').CounterSamples.CookedValue") MB" >> $LOG
if ! grep -q HEALTH_OK $LOG; then echo "HEALTH_FAILED $(date +%F_%T): not launching" >> $LOG; tail -5 runs/meadow2/hybrid/leak_healthcap.log >> $LOG; exit 4; fi
# the chain, as section 21.14.3 describes it, under the cap (the environment is inherited by the driver and the trainer)
wsl -e bash -lc "cd $WSLREPO && export XLA_PYTHON_CLIENT_MEM_FRACTION=$MEMFRAC && nohup setsid bash runs/m2_steps512.sh chain > runs/meadow2/steps512.log 2>&1 < /dev/null & sleep 15; cat runs/meadow2/steps512.log"  >> $LOG 2>&1
echo "CHAIN_LAUNCHED $(date +%F_%T) MEMFRAC=$MEMFRAC" >> $LOG
