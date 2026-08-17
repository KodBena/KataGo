#!/bin/bash
# Runs one command under a shared timing flock, an instantaneous /proc/stat idle gate,
# and a pswpin/pswpout sample taken across the WHOLE window.
#
# WHY EACH PIECE EXISTS. All three were bought with real failures on this investigation:
#
#   * The flock, because other work runs on the same cores and a timed run that shares
#     them is measuring the scheduler.
#   * The idle check is INSTANTANEOUS -- a 0.5 s /proc/stat delta -- and is taken INSIDE
#     the lock so it cannot go stale between the check and the run. The 1-minute load
#     average reads identically for a contended run and a clean one, and relying on it
#     has already cost one full measurement pass on this programme.
#   * pswpin/pswpout are sampled around the WHOLE window, never spot-checked inside it.
#     A spot check proves nothing about the window, and that error has been made and
#     reported here too.
#
# Exit 137 is treated as a SUBSTRATE VERDICT, not as flakiness: the run stops, free and
# df are recorded, and it is reported as an incident rather than blind-retried.
#
# Usage: idle_gate.sh <gatefile> <command...>
# Env:   MIN_IDLE_PCT (default 90), GATE_LOCK (default /tmp/katago-timing.lock),
#        GATE_WAIT_SECONDS (default 300)
set -u
GATEFILE="$1"; shift
MIN_IDLE_PCT="${MIN_IDLE_PCT:-90}"
LOCK="${GATE_LOCK:-/tmp/katago-timing.lock}"
WAIT_SECONDS="${GATE_WAIT_SECONDS:-300}"

idle_pct() {
  read -r _ a b c d rest < /proc/stat
  local t0=$((a+b+c+d)) i0=$d
  sleep 0.5
  read -r _ a b c d rest < /proc/stat
  local t1=$((a+b+c+d)) i1=$d
  echo $(( 100 * (i1-i0) / (t1-t0) ))
}
swap_counters() { awk '/^pswpin |^pswpout /{printf "%s ", $2}' /proc/vmstat; }

exec 9>"$LOCK"
flock 9

: > "$GATEFILE"
DEADLINE=$(( $(date +%s) + WAIT_SECONDS ))
while :; do
  IDLE=$(idle_pct)
  [ "$IDLE" -ge "$MIN_IDLE_PCT" ] && break
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "GATE VERDICT=REFUSED idle_pct=$IDLE min=$MIN_IDLE_PCT" | tee -a "$GATEFILE"
    exit 3
  fi
  sleep 5
done
echo "GATE idle_pct_before=$IDLE" >> "$GATEFILE"
echo "GATE thp=$(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || echo UNREADABLE)" >> "$GATEFILE"

read -r SIN0 SOUT0 <<< "$(swap_counters)"
START=$(date +%s)
"$@"
RC=$?
END=$(date +%s)
read -r SIN1 SOUT1 <<< "$(swap_counters)"
IDLE_AFTER=$(idle_pct)

echo "GATE pswpin_delta=$((SIN1-SIN0)) pswpout_delta=$((SOUT1-SOUT0))  (sampled across the WHOLE window)" >> "$GATEFILE"
echo "GATE wall_seconds=$((END-START))" >> "$GATEFILE"
echo "GATE idle_pct_after=$IDLE_AFTER" >> "$GATEFILE"
echo "GATE exit_code=$RC" >> "$GATEFILE"
if [ "$RC" = "137" ]; then
  echo "GATE VERDICT=SUBSTRATE_KILL (exit 137 -- presumptively OOM; this is an INCIDENT, do not retry)" >> "$GATEFILE"
  free -m >> "$GATEFILE"; df -h . >> "$GATEFILE"
elif [ "$((SIN1-SIN0))" -ne 0 ]; then
  echo "GATE VERDICT=DISQUALIFIED (pages read back from swap inside the window)" >> "$GATEFILE"
elif [ "$((SOUT1-SOUT0))" -ne 0 ]; then
  echo "GATE VERDICT=DEGRADED (reclaim wrote pages out; no measured loop stalled on a read)" >> "$GATEFILE"
elif [ "$RC" != "0" ]; then
  echo "GATE VERDICT=COMMAND_FAILED" >> "$GATEFILE"
else
  echo "GATE VERDICT=QUALIFIED" >> "$GATEFILE"
fi
cat "$GATEFILE"
exit $RC
