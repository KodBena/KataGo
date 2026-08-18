#!/bin/bash
# The NN-cache policy sweep driver.
#
# WHAT THIS SCRIPT IS FOR. KataGo's NN cache used to be one hardcoded shape: a 1-way
# direct-mapped table that overwrites on collision and stores everything offered. It is
# now configurable along four axes (how collisions are resolved, how many ways, which
# entry an eviction gives up, and whether an entry is stored at all), and the question
# this script answers is which settings are worth having. It runs in three stages.
#
#   STAGE 1  CAPTURE   Run KataGo's analysis engine once with KATAGO_NNCACHE_TRACE set,
#                      recording every cache get and set to a file. This costs one real
#                      engine run and is the only stage that needs the GPU.
#   STAGE 2  REPLAY    Replay that one recorded stream through every point of the policy
#                      matrix, reporting hit rate, resident bytes, occupancy and cache
#                      throughput per configuration into one NDJSON results file. No GPU,
#                      no neural net, minutes rather than hours.
#   STAGE 3  VISITS    For a short list of configurations, run `katago benchmark` for
#                      real visits per second. This is an AFFORDABILITY check, not the
#                      decision input: the whole cache path was previously measured at
#                      0.0077% of cycles, so no policy here can plausibly move visits/s
#                      much, and stage 2 is where the actual answer lives.
#
# Stages are selectable, because stage 1 is the expensive one and you will usually want
# to run stage 2 several times against a trace you captured once.
#
# EVERY STAGE'S OUTPUT LANDS UNDER ONE DIRECTORY, given by -o. Nothing is written
# anywhere else. Each timed stage is wrapped in idle_gate.sh, whose gate file lands beside
# the results, so a later reader can tell whether a number was taken on a quiet machine.
#
# Run with -h for the full argument list.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GATE="$SCRIPT_DIR/idle_gate.sh"

KATAGO=""
MODEL=""
CONFIG=""
OUTDIR=""
STAGES="capture,replay,visits"
TRACE=""
# MANY QUERIES x MODEST VISITS. These defaults are the inversion of the ones that produced
# a trace with 136 hits in 1.96M lookups: 4 games x ~200 turns = ~800 review queries at 500
# visits each, about 400k visits in the whole capture against the old 24 x 100000 = 2.4M.
# Fewer visits AND a usable workload; the visits were never what the cache needed.
VISITS_FOR_CAPTURE=500
CAPTURE_GAMES=4
CAPTURE_TURNS=200
GEN_VISITS=8
# Swept DOWNWARD by default now. A replacement or eviction policy can only act where the
# table cannot hold the working set; a capture of this size saturates around 2^17, so the
# sizes where the policies separate are below it. 2^21 is kept so the operator's own
# setting is on the curve -- as the right-hand end of it, not as the whole of it.
TABLE_POWS="13,14,15,16,17,19,21"
WAYS="2,4,8,16"
COLLISIONS="direct,linearprobe,quadraticprobe,chain"
EVICTIONS="random,lru,lfu"
ADMISSIONS="always,secondsighting"
REPLACEMENTS="always,keeplessseen,keepmoreseen,keepsighted"
MIN_REUSE_RATE="0.05"
# auto | table | <N>. auto sizes the sighting-count ghost from the captured trace's own
# distinct-key count, which is the quantity it should be sized to and the one thing a .cfg
# cannot know. `table` reproduces what an operator running this in production would get.
GHOST_POW="auto"
OWNERSHIP="both"
MAX_BYTES=""
BENCH_VISITS=2000
BENCH_THREADS=16
BENCH_CONFIGS="direct|direct:keepmoreseen|direct:keepsighted|linearprobe:8:lru|linearprobe:8:lru+secondsighting|chain:lru"
CAPTURE_OWNERSHIP=1

usage() {
  cat <<'USAGE'
run_policy_sweep.sh -k KATAGO -m MODEL -c CONFIG -o OUTDIR -b MAX_BYTES [options]

REQUIRED
  -k PATH   the katago binary built on THIS host
  -m PATH   the model (.bin.gz / .txt.gz) to run
  -c PATH   a KataGo .cfg for the analysis engine (used in stage 1 only)
  -o DIR    output directory; created if absent; EVERYTHING lands here
  -b BYTES  the memory budget stage 2 may not exceed, in bytes. REQUIRED, no default.
            On a host whose RAM is shared with a VM or other tasks, this is what is
            FREE, not what is installed. There is no default because the previous plan
            for this sweep was built on an assumed memory figure that was wrong.

STAGES
  -s LIST   comma-separated from capture,replay,visits (default all three)
  -t PATH   trace file. Written by capture; read by replay. Defaults to OUTDIR/trace.bin.
  -V N      visits per REVIEW query during capture (default 500). MODEST IS THE POINT.
            Visits do not buy reuse: a KataGo SearchNode holds its own NNOutput, so a
            position evaluated inside one search is never asked of the NN cache again,
            however deep that search is. The cache earns its keep ACROSS searches. What
            buys reuse is many queries over RELATED positions -- see -G and -N.
  -G N      games generated for the capture (default 4). Different seeded openings.
  -N N      moves per generated game (default 200). The review then issues one query per
            turn of each game, so queries ~= games x turns.
  -J N      visits per move while GENERATING the games (default 8). This is throwaway
            search whose only job is to produce a plausible game; it is not captured.
  -O 0|1    capture with ownership requested (default 1). Ownership roughly doubles the
            per-entry footprint and is what a reviewing GUI asks for as a matter of
            course, so 1 is the realistic setting.

REPLAY MATRIX (stage 2)
  -p LIST   table powers            (default 13,14,15,16,17,19,21 -- swept DOWNWARD,
            because a policy can only act where the table cannot hold the working set)
  -w LIST   ways                    (default 2,4,8,16)
  -C LIST   collision schemes       (default direct,linearprobe,quadraticprobe,chain)
  -e LIST   eviction policies       (default random,lru,lfu)
  -a LIST   admission policies      (default always,secondsighting)
  -R LIST   replacement rules, DIRECT MAPPING ONLY: which of the two candidates a
            collision keeps (default always,keeplessseen,keepmoreseen,keepsighted).
            The two "seen" values name the SURVIVOR, not the victim.
  -M F      minimum reuse rate the captured trace must clear, or stage 2 refuses before
            allocating anything (default 0.05). See the runbook's precondition section.
  -Q W      sighting-count ghost size: auto | table | <N>  (default auto). auto sizes it
            from the captured trace's distinct-key count so the count sketch is not
            saturated; `table` sizes it from each swept table size, which is what an
            operator gets from a .cfg that does not set nnCacheSightingGhostPowerOfTwo.
            Measured through a saturated ghost, keepmoreseen degenerates to `always` and
            keeplessseen degenerates to refusing everything, so this is not a detail.
  -W MODE   ownership: trace|off|on|both   (default both)

VISITS SUBSET (stage 3)
  -v N      visits per benchmark run (default 2000)
  -T N      threads                  (default 16)
  -B LIST   pipe-separated configurations, each of
              direct
              direct:<keeplessseen|keepmoreseen|keepsighted>
              <linearprobe|quadraticprobe>:<ways>:<random|lru|lfu>
              chain:<random|lru|lfu>
            with an optional "+secondsighting" suffix on any of them.
            (default: direct|direct:keepmoreseen|direct:keepsighted|
                      linearprobe:8:lru|linearprobe:8:lru+secondsighting|chain:lru)
USAGE
  exit 1
}

while getopts "k:m:c:o:b:s:t:V:G:N:J:O:p:w:C:e:a:R:M:Q:W:v:T:B:h" opt; do
  case "$opt" in
    k) KATAGO="$OPTARG" ;;
    m) MODEL="$OPTARG" ;;
    c) CONFIG="$OPTARG" ;;
    o) OUTDIR="$OPTARG" ;;
    b) MAX_BYTES="$OPTARG" ;;
    s) STAGES="$OPTARG" ;;
    t) TRACE="$OPTARG" ;;
    V) VISITS_FOR_CAPTURE="$OPTARG" ;;
    G) CAPTURE_GAMES="$OPTARG" ;;
    N) CAPTURE_TURNS="$OPTARG" ;;
    J) GEN_VISITS="$OPTARG" ;;
    O) CAPTURE_OWNERSHIP="$OPTARG" ;;
    p) TABLE_POWS="$OPTARG" ;;
    w) WAYS="$OPTARG" ;;
    C) COLLISIONS="$OPTARG" ;;
    e) EVICTIONS="$OPTARG" ;;
    a) ADMISSIONS="$OPTARG" ;;
    R) REPLACEMENTS="$OPTARG" ;;
    M) MIN_REUSE_RATE="$OPTARG" ;;
    Q) GHOST_POW="$OPTARG" ;;
    W) OWNERSHIP="$OPTARG" ;;
    v) BENCH_VISITS="$OPTARG" ;;
    T) BENCH_THREADS="$OPTARG" ;;
    B) BENCH_CONFIGS="$OPTARG" ;;
    *) usage ;;
  esac
done

[ -n "$KATAGO" ] || { echo "run_policy_sweep.sh: -k is required"; usage; }
[ -n "$OUTDIR" ] || { echo "run_policy_sweep.sh: -o is required"; usage; }
[ -x "$KATAGO" ] || { echo "run_policy_sweep.sh: not an executable: $KATAGO"; exit 1; }
mkdir -p "$OUTDIR" || exit 1
OUTDIR="$(cd "$OUTDIR" && pwd)"
[ -n "$TRACE" ] || TRACE="$OUTDIR/trace.bin"

has_stage() { case ",$STAGES," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

#-------------------------------------------------------------------------------------
# A substrate note written by the SCRIPT, separate from the one the binary writes.
# The binary stamps its own run; this stamps the host as a whole, once, at the top.
#-------------------------------------------------------------------------------------
{
  echo "=== run_policy_sweep.sh substrate ==="
  echo "date            $(date -Is)"
  echo "host            $(hostname)"
  echo "kernel          $(uname -r)"
  echo "cpu             $(grep -m1 '^model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')"
  echo "cores           $(nproc)"
  echo "mem             $(grep -m1 MemTotal /proc/meminfo) / $(grep -m1 MemAvailable /proc/meminfo)"
  echo "hugepages       $(grep -m1 HugePages_Total /proc/meminfo) $(grep -m1 Hugepagesize /proc/meminfo)"
  echo "thp_enabled     $(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || echo UNREADABLE)"
  echo "thp_defrag      $(cat /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null || echo UNREADABLE)"
  echo "katago          $KATAGO"
  echo "katago version  $("$KATAGO" version 2>&1 | head -3 | tr '\n' ' ')"
  echo "model           $MODEL"
  echo "model sha256    $( [ -f "$MODEL" ] && sha256sum "$MODEL" | cut -d' ' -f1 || echo NOT-A-FILE )"
  echo "config          $CONFIG"
  echo "args            stages=$STAGES pows=$TABLE_POWS ways=$WAYS collisions=$COLLISIONS"
  echo "                evictions=$EVICTIONS admissions=$ADMISSIONS replacements=$REPLACEMENTS"
  echo "                ownership=$OWNERSHIP min_reuse_rate=$MIN_REUSE_RATE ghost_pow=$GHOST_POW"
  echo "                capture_games=$CAPTURE_GAMES capture_turns=$CAPTURE_TURNS"
  echo "                capture_gen_visits=$GEN_VISITS capture_visits=$VISITS_FOR_CAPTURE"
  echo "                capture_ownership=$CAPTURE_OWNERSHIP"
  echo "                bench_visits=$BENCH_VISITS bench_threads=$BENCH_THREADS"
  echo "max_bytes       $MAX_BYTES"
  if [ -n "$(command -v nvidia-smi)" ]; then
    echo "nvidia-smi      $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | tr '\n' ';')"
  else
    echo "nvidia-smi      NOT PRESENT"
  fi
} > "$OUTDIR/substrate.txt"
cat "$OUTDIR/substrate.txt"

#-------------------------------------------------------------------------------------
# STAGE 1 -- CAPTURE
#-------------------------------------------------------------------------------------
# The analysis engine writes PROTOCOL RESPONSES to stdout and its LOG to stderr. They are
# directed to two different files here and never merged: a log line landing in a results
# stream is how a results file becomes unparseable, and merging them is a mistake that is
# very hard to see after the fact.
#
# The trace itself goes to its own binary file, named by KATAGO_NNCACHE_TRACE, and is not
# on either stream.
if has_stage capture; then
  [ -n "$MODEL" ]  || { echo "capture stage needs -m MODEL"; exit 1; }
  [ -n "$CONFIG" ] || { echo "capture stage needs -c CONFIG"; exit 1; }

  QUERIES="$OUTDIR/capture-queries.jsonl"
  if [ ! -f "$QUERIES" ]; then
    echo "run_policy_sweep.sh: generating games and writing per-turn review queries to $QUERIES"
    # The queries are REAL GAMES WALKED TURN BY TURN, not scattered positions. The previous
    # generator wrote 24 unrelated random-move positions and searched each at 100000 visits,
    # and the resulting trace had 136 hits in 1,957,525 lookups -- an entire GPU sweep that
    # could not answer anything. See make_capture_queries.py's own header for the two
    # mechanisms and the witnessed contrast. The knobs are inverted accordingly: many
    # queries at modest visits, not a few at enormous ones.
    python3 "$SCRIPT_DIR/make_capture_queries.py" \
      --katago "$KATAGO" --model "$MODEL" --config "$CONFIG" \
      --out "$QUERIES" --log "$OUTDIR/capture-gamegen.log" \
      --games "$CAPTURE_GAMES" --turns "$CAPTURE_TURNS" \
      --gen-visits "$GEN_VISITS" --visits "$VISITS_FOR_CAPTURE" \
      --ownership "$CAPTURE_OWNERSHIP" || {
        echo "run_policy_sweep.sh: game generation failed; see $OUTDIR/capture-gamegen.log"
        exit 1
      }
  fi

  echo "run_policy_sweep.sh: STAGE 1 capture -> $TRACE"
  # NOT gated on idleness: a capture is not a timing run, and the trace writer serialises
  # every cache operation through one lock anyway, so nothing measured here would mean
  # anything. It IS the stage that needs the GPU and the one that takes real time.
  KATAGO_NNCACHE_TRACE="$TRACE" \
    "$KATAGO" analysis -config "$CONFIG" -model "$MODEL" \
      < "$QUERIES" \
      > "$OUTDIR/capture-responses.jsonl" \
      2> "$OUTDIR/capture-engine.log"
  RC=$?
  echo "run_policy_sweep.sh: capture exit=$RC"
  if [ "$RC" = "137" ]; then
    echo "run_policy_sweep.sh: EXIT 137 -- the kernel killed the capture. This is a SUBSTRATE"
    echo "  incident, not a flake. Do not rerun it unchanged. Record 'free -m' and 'df -h',"
    echo "  lower nnCacheSizePowerOfTwo or the visit count, and say so in the results."
    free -m; df -h "$OUTDIR"
    exit 137
  fi
  [ "$RC" = "0" ] || { echo "run_policy_sweep.sh: capture failed; see $OUTDIR/capture-engine.log"; exit "$RC"; }
  ls -l "$TRACE"

  # THE PRECONDITION, applied where the capture was produced rather than an hour later.
  # It reads the trace and refuses if its REUSE RATE -- the fraction of lookups for a key
  # this stream had stored earlier, which is the hit rate a perfect unbounded cache would
  # deliver and therefore a ceiling on how far any two policies can differ -- is below
  # -M. The check that used to guard this stage asked about trace SIZE, and a 1252-second
  # GPU sweep passed it and answered nothing.
  echo "run_policy_sweep.sh: PRECHECK on $TRACE"
  # -table-pows is passed even though nothing is swept here, so the load-factor warning
  # is about the sweep that is actually going to run rather than about the flag's default.
  "$KATAGO" benchnncachepolicy -trace "$TRACE" -precheck \
    -min-reuse-rate "$MIN_REUSE_RATE" -table-pows "$TABLE_POWS" \
    2>&1 | tee "$OUTDIR/capture-precheck.txt"
  RC=${PIPESTATUS[0]}
  if [ "$RC" != "0" ]; then
    echo "run_policy_sweep.sh: THE CAPTURE CANNOT ANSWER THE POLICY QUESTION. Refused above,"
    echo "  before any sweep time was spent. The trace is still at $TRACE if you want to keep it."
    echo "  Fix the CAPTURE, not the threshold: raise -G (more games) or -N (longer games), which"
    echo "  is what produces reuse. Raising -V (visits) does NOT: a search never re-asks the cache"
    echo "  for a position its own tree already holds."
    exit "$RC"
  fi
fi

#-------------------------------------------------------------------------------------
# STAGE 2 -- REPLAY
#-------------------------------------------------------------------------------------
if has_stage replay; then
  [ -n "$MAX_BYTES" ] || { echo "replay stage needs -b MAX_BYTES; there is deliberately no default"; exit 1; }
  [ -f "$TRACE" ] || { echo "replay stage needs a trace at $TRACE (run the capture stage, or pass -t)"; exit 1; }
  echo "run_policy_sweep.sh: STAGE 2 replay -> $OUTDIR/policy-sweep.ndjson"
  "$GATE" "$OUTDIR/policy-sweep.gate" \
    "$KATAGO" benchnncachepolicy \
      -trace "$TRACE" \
      -out "$OUTDIR/policy-sweep.ndjson" \
      -table-pows "$TABLE_POWS" \
      -ways "$WAYS" \
      -collisions "$COLLISIONS" \
      -evictions "$EVICTIONS" \
      -admissions "$ADMISSIONS" \
      -replacements "$REPLACEMENTS" \
      -min-reuse-rate "$MIN_REUSE_RATE" \
      -ghost-pow "$GHOST_POW" \
      `# -reinsert-on-miss is ON here, though the binary's own default is off. Without it a` \
      `# swept policy that REFUSES a newcomer -- which only the replacement rules can do --` \
      `# is charged for every later lookup of that key and never sees the re-insert the live` \
      `# engine would have made, so the arms are handicapped unequally. The binary keeps the` \
      `# old default so a hand replay reproduces earlier runs; this script sweeps the` \
      `# replacement axis by default, so unequal handicaps would be the default condition.` \
      -reinsert-on-miss \
      -ownership "$OWNERSHIP" \
      -max-bytes "$MAX_BYTES" \
      -backend-name "$("$KATAGO" version 2>&1 | head -2 | tr '\n' ' ')" \
      -note "run_policy_sweep.sh on $(hostname); model $MODEL" \
      2> "$OUTDIR/policy-sweep.stderr"
  RC=$?
  echo "run_policy_sweep.sh: replay exit=$RC (progress lines are in $OUTDIR/policy-sweep.stderr)"
  [ "$RC" = "0" ] || exit "$RC"
fi

#-------------------------------------------------------------------------------------
# STAGE 3 -- VISITS
#-------------------------------------------------------------------------------------
# One `katago benchmark` run per named configuration, each against a generated .cfg that
# differs from the base config ONLY in the nnCache* keys. benchmark writes its report to
# stdout; anything else goes to stderr. The two are kept in separate files per config.
if has_stage visits; then
  [ -n "$MODEL" ]  || { echo "visits stage needs -m MODEL"; exit 1; }
  [ -n "$CONFIG" ] || { echo "visits stage needs -c CONFIG"; exit 1; }
  mkdir -p "$OUTDIR/visits"
  IFS='|' read -ra SPECS <<< "$BENCH_CONFIGS"
  for spec in "${SPECS[@]}"; do
    name="$(echo "$spec" | tr ':+' '__')"
    cfg="$OUTDIR/visits/$name.cfg"
    base="$spec"; adm="always"
    case "$spec" in *"+secondsighting") base="${spec%+secondsighting}"; adm="secondsighting" ;; esac
    cp "$CONFIG" "$cfg"
    {
      echo ""
      echo "# --- appended by run_policy_sweep.sh for configuration: $spec ---"
      echo "nnCacheAdmission = $adm"
      case "$base" in
        direct)
          echo "nnCacheCollision = direct" ;;
        direct:*)
          # The replacement axis: which of the two candidates a direct-mapped collision
          # keeps. Only direct mapping has two candidates, so only this arm can carry it.
          echo "nnCacheCollision = direct"
          echo "nnCacheReplacement = ${base#direct:}" ;;
        chain:*)
          echo "nnCacheCollision = chain"
          echo "nnCacheEviction = ${base#chain:}"
          # A chained table is bounded by BYTES, not by slots, so it needs a budget. This
          # one is chosen to be comparable to what a direct table of the configured slot
          # count would hold; adjust it if you are testing a memory ceiling rather than a
          # policy.
          echo "nnCacheMaxBytes = 4000000000" ;;
        *:*:*)
          echo "nnCacheCollision = $(echo "$base" | cut -d: -f1)"
          echo "nnCacheWays = $(echo "$base" | cut -d: -f2)"
          echo "nnCacheEviction = $(echo "$base" | cut -d: -f3)" ;;
        *)
          echo "# UNPARSEABLE SPEC: $spec" ;;
      esac
    } >> "$cfg"

    echo "run_policy_sweep.sh: STAGE 3 visits [$spec]"
    "$GATE" "$OUTDIR/visits/$name.gate" \
      "$KATAGO" benchmark -config "$cfg" -model "$MODEL" \
        -visits "$BENCH_VISITS" -threads "$BENCH_THREADS" \
        > "$OUTDIR/visits/$name.stdout" 2> "$OUTDIR/visits/$name.stderr"
    RC=$?
    echo "  exit=$RC  report: $OUTDIR/visits/$name.stdout"
    if [ "$RC" != "0" ]; then
      echo "  FAILED. The most common cause is that the config keys were REFUSED, which is"
      echo "  deliberate -- KataGo refuses an incoherent cache configuration rather than"
      echo "  silently ignoring it. The refusal names what to change; it is at the end of:"
      echo "    $OUTDIR/visits/$name.stderr"
      tail -5 "$OUTDIR/visits/$name.stderr"
    fi
  done
fi

echo ""
echo "run_policy_sweep.sh: done. Everything is under $OUTDIR"
echo "  substrate.txt          the host, as it was, at the top of the run"
echo "  policy-sweep.ndjson    THE RESULTS FILE -- read it with summarize_sweep.py"
echo "  policy-sweep.gate      whether the machine was quiet while stage 2 ran"
echo "  visits/*.stdout        one katago benchmark report per named configuration"
