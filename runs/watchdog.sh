# run a command, killing it if its resident memory passes $MAXGB (default 9) GB; prints the peak
MAXGB=${MAXGB:-9}
"$@" &
pid=$!
peak=0
while kill -0 $pid 2>/dev/null; do
  rss=$(ps -o rss= --ppid $pid -p $pid 2>/dev/null | awk '{s+=$1} END {print s+0}')
  [ "$rss" -gt "$peak" ] && peak=$rss
  if [ "$rss" -gt $((MAXGB * 1024 * 1024)) ]; then
    echo "WATCHDOG: killing pid $pid at $((rss / 1024)) MB"; kill -9 $pid; pkill -9 -P $pid
  fi
  sleep 2
done
wait $pid; rc=$?
echo "WATCHDOG: peak $((peak / 1024)) MB, exit $rc"
exit $rc
