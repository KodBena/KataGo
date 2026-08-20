#!/bin/sh
# DERIVES EVERY COUNT A REPORT ON THE CACHE ARC QUOTES, so none of them is transcribed by hand.
#
# WHY THIS EXISTS. Three successive reports on this arc shipped with incidental arithmetic wrong
# while every substantive claim held -- suite line counts off by one because each suite's own
# "Done" marker was counted as an observation, a grep-by-eye call-site count off by one. A rate
# like that says the defect is TRANSCRIPTION, not carelessness, and ADR-0011 Rule 2 says a
# recurrence converts to a mechanism rather than to more care. This is the mechanism: the report
# quotes this script's output and names this script beside the number, so the next reader re-runs
# it instead of trusting a number somebody read off a screen.
#
#   usage:  cpp/tests/report_census.sh <runtests.log> <runoutputtests.log>
#   e.g.    ./cpp/build/katago runtests > /tmp/rt.log 2>&1
#           (cd cpp && ./build/katago runoutputtests) > /tmp/ro.log 2>&1
#           cpp/tests/report_census.sh /tmp/rt.log /tmp/ro.log
#
# Run from the repository root (the source censuses below read cpp/tests/ by relative path).
set -e

if [ $# -ne 2 ]; then
  sed -n '2,20p' "$0"
  exit 2
fi
RUNTESTS="$1"
RUNOUTPUT="$2"

# The seven suites of this arc that live in `katago runtests`, in the order the report lists them.
SUITES="nn eval container
nn cache level-0 loader
NN cache context attribution
analysis engine model name space
two-level ordered resolution list
NN cache dump admission and persistence
analysis engine cache action"

# Which source file registers each suite, for the registered-test column.
suite_source() {
  case "$1" in
    "nn eval container") echo cpp/tests/testnnevalcontainer.cpp ;;
    "nn cache level-0 loader") echo cpp/tests/testnncachelevelzero.cpp ;;
    "NN cache context attribution") echo cpp/tests/testnncachecontext.cpp ;;
    "analysis engine model name space") echo cpp/tests/testanalysismodels.cpp ;;
    "two-level ordered resolution list") echo cpp/tests/testnncachetwolevel.cpp ;;
    "NN cache dump admission and persistence") echo cpp/tests/testnncachedump.cpp ;;
    "analysis engine cache action") echo cpp/tests/testanalysiscacheactions.cpp ;;
  esac
}

# REPORTED LINES PER SUITE. A suite's block runs from its "Running <name> tests" header to the
# next such header or to its own "Done" marker, whichever comes first. Blank lines and the "Done"
# marker itself are NOT observations -- counting "Done" is precisely the off-by-one this script
# exists to foreclose.
lines_for_suite() {
  awk -v want="$1" '
    /^Running /{ if(s==want) { print n; s=""; exit } ; s=substr($0,9); sub(/ tests$/,"",s); n=0; next }
    s==""{ next }
    /^Done$/{ if(s==want) { print n; s=""; exit } ; s=""; next }
    NF{ n++ }
    END{ if(s==want) print n }
  ' "$RUNTESTS"
}

# REGISTERED TESTS PER SUITE: the calls inside the file's own `void Tests::run...()` entry point.
tests_for_suite() {
  awk '/^void Tests::run/,/^}/' "$1" | grep -cE '^[[:space:]]+test[A-Za-z][A-Za-z0-9]*\('
}

echo "SUITES IN runtests (source: $RUNTESTS)"
printf '%-42s %14s %17s\n' "suite" "reported lines" "registered tests"
echo "$SUITES" | while IFS= read -r suite; do
  [ -n "$suite" ] || continue
  present=$(grep -c "^Running $suite tests\$" "$RUNTESTS" || true)
  if [ "$present" -eq 0 ]; then
    printf '%-42s %14s %17s\n' "$suite" "ABSENT" "-"
    continue
  fi
  printf '%-42s %14s %17s\n' "$suite" "$(lines_for_suite "$suite")" "$(tests_for_suite "$(suite_source "$suite")")"
done

echo
echo "SUITE IN runoutputtests (source: $RUNOUTPUT)"
EVAL_LEGS=$(awk '/^Eval cache keys depend on the models$/{f=1;next} f&&/^===/{next} f&&/: [01]$/{n++} f&&/^$/{if(n>0){print n;n=0;exit}} END{if(n>0)print n}' "$RUNOUTPUT")
printf '%-42s %14s\n' "Eval cache keys depend on the models" "${EVAL_LEGS:-ABSENT} legs"

echo
echo "CALL-SITE CENSUS (permit-gated acts)"
printf '%-52s %6s\n' "table attachLevelZero calls in testnncachetwolevel.cpp" \
  "$(grep -c -- '->attachLevelZero(' cpp/tests/testnncachetwolevel.cpp)"
printf '%-52s %6s\n' "table detachLevelZero calls in testnncachetwolevel.cpp" \
  "$(grep -c -- '->detachLevelZero(' cpp/tests/testnncachetwolevel.cpp)"
printf '%-52s %6s\n' "table attach/detach calls in testnncachetwolevelbench.cpp" \
  "$(grep -cE -- '->(attach|detach)LevelZero\(' cpp/tests/testnncachetwolevelbench.cpp)"
printf '%-52s %6s\n' "permit mint sites outside the three mints" \
  "$(grep -rn 'NNCacheLevelZeroSwapPermit()' cpp --include='*.cpp' --include='*.h' \
      | grep -v 'cpp/external/' \
      | grep -vE 'nncache\.cpp|nncachetwolevel\.h|analysiscacheactions\.h|testcacheswapseam\.h' \
      | grep -c . || true)"
