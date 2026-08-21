#!/usr/bin/env python3
"""End-to-end witness for the COMPOSED property: persistence, and the model keying on it.

  usage: cpp/tests/e2e/persistence_e2e_witness.py <katago> <config-template> <workdir>
                                                  [<netA> <netB>] [--only S,P,A,K,K5,K6] [--keep]
  e.g.   cpp/tests/e2e/persistence_e2e_witness.py ./cpp/build/katago \\
             cpp/tests/e2e/persistence_e2e.cfg /tmp/pe2e

  Exit 0 iff every leg held. Every leg prints PASS/FAIL, what it OBSERVED and what it
  EXPECTED, so a red says WHICH property broke rather than only that one did. The workdir is
  created, used and removed (--keep leaves it). The two nets default to the operator's
  /home/bork/kg/14.gz and /home/bork/kg/18.gz and may be given as arguments.

  Group S always runs: it establishes the per-net fingerprints every other group reads, and a
  group that read them without S having proved they discriminate would be asserting nothing.
  --only selects which of the rest run, for a seen-red pass that wants one group under a
  mutated build.

  The files: this one is the entry point and the map. e2e_harness.py is the machinery a leg
  is run with and read through. persistence_legs.py and keying_legs.py hold the legs of the
  two halves. analysis_engine_driver.py drives one engine process and reads its row counter.

-----------------------------------------------------------------------------------------
WHY THIS IS A PROCESS DRIVER AND NOT A C++ SUITE IN `katago runtests`

The property has two halves and both are claims about things a single in-process test cannot
be. The first half -- "a NEW process is handed what an OLD process earned and performs ZERO
neural net work" -- is a claim about a process boundary and about a counter that exists only
because a real NNEvaluator ran real batches. The second half -- "a key persisted under one
model is served to THAT model and not to the other" -- is a claim about an engine started
with `-model A -extra-model B` from a command line, answering a client that names a model per
request over stdin. `runtests` has no engine, no stdin protocol, no second process and no
second hosted net. The library seams underneath ARE covered there and are not repeated here:
eight suites in `katago runtests` (nn eval container; nn cache level-0 loader; NN cache
context attribution; analysis engine model name space; two-level ordered resolution list; NN
cache dump admission and persistence; analysis engine cache action) plus "Eval cache keys
depend on the models" in `runoutputtests`. This witness is their composition, observed from
outside the process, and it is the only place that composition is observed at all.

-----------------------------------------------------------------------------------------
WHAT IS OBSERVED, AND WHERE (ADR-0021 Rule 1: observe the property, at the site of the claim)

Three observation surfaces, and deliberately NOT the response field that would be easiest:

  1. THE ENGINE'S OWN NEURAL-NET ROW COUNTER, PER HOSTED MODEL, from its end-of-run log
     (analysis_engine_driver.nn_rows_for). "Zero evaluations" and "a real evaluation was paid
     for" are claims about WORK DONE; a response saying "20 entries attached" is the engine's
     opinion of itself. It is read PER MODEL because a witness hosting two nets that summed
     them could not tell "the second net did no work" from "the first net did the second's
     work instead" -- which is exactly the cross-model question.

  2. THE ROOT'S RAW NEURAL NET EVALUATION, as a per-net FINGERPRINT. With nnRandomize off
     each net's evaluation of one position is reproducible, and the two nets' evaluations
     differ (leg S3 proves both). So "18.gz was NOT served 14.gz's evaluation" is not argued
     from an entry count -- it is read off the numbers that came back. A second, independent
     axis: a bug that served the wrong bytes while still counting a row would pass axis 1 and
     fail this. It is the RAW root evaluation and not the SEARCHED winrate, because the
     searched value is downstream of a 20-visit tree; see e2e_harness.fingerprint.

  3. THE BYTES ON DISK -- size and sha256 -- for what was persisted and for what a second
     dump did or did not rewrite. Not the dump response's own byte counters.

-----------------------------------------------------------------------------------------
THE GROUPS

  S  fingerprints   each net's answer on an engine that touches no cache directory, and the
                    proof that the two answers differ so the fingerprint discriminates.
  P  persistence    P1 cold: attach an empty context, analyze, dump, detach; files on disk.
                    P2 warm: a NEW PROCESS re-attaches and pays ZERO rows for the same query
                       and returns the identical answer.
                    P3 the persisted mark, then attach -> dump -> detach -> re-attach -> dump
                       with NO query in the process at all: the container byte-identical, the
                       count log's content unmoved, no key's recorded observations moved. P3e
                       records the one place the file is NOT byte-identical, and why that is
                       FILED rather than fixed in this increment.
  A  accounting     THE OPERATOR'S OWN PRINCIPLE, pinned: the same query, with and without a
                    "cacheContext" field, does IDENTICAL neural net work and returns an
                    IDENTICAL search -- cacheContext is bookkeeping alongside the work, never
                    an input to it. The one place a difference is expected: the context's own
                    attribution counter in cache_stats.
  K  the keying     K0 seed, K1 the own container re-attached (zero rows), K2 the other model
                    gets nothing and pays real work, K3 THE DISTINGUISHER, K4 the capability
                    control that forecloses "the machinery simply cannot serve across models".
  K5 eval salt      the eval-cache model salt alone, with no file in the story -- read off the
                    SEARCH RESULT, because an eval-cache hit does not avoid a neural net
                    evaluation and the salt is therefore invisible on the row counter. K5c is
                    the leg that says so, and it is why the salt cannot be an alternative
                    explanation for K2 and K3.
  K6 independence   K2 and K3 repeated with useEvalCache on and the eval cache actually
                    populated: identical observations, so the salt is not what produces them.

-----------------------------------------------------------------------------------------
--only IS A SEEN-RED CONVENIENCE, NOT A MERGE GATE: RE-RUN THE WHOLE SUITE ON A SIBLING CHANGE

`--only` exists so a worker chasing one group's red can re-run just that group under a
mutated build without paying for the rest. It is NOT evidence that a change scoped to one
group's area cannot affect another group's legs. P1d went red for two days after
`f9be19cc` (the cross-process context lock, `<context>.nnlock`) merged into this line,
because nobody re-ran group P (or the full suite) against the merge that carried it -- the
lock touches every context's file footprint, which is exactly what group P observes,
but the branch that added it was reviewed and tested as a locking change, not as a
persistence-e2e change (audit-reports/p1d-diagnosis.md). Before trusting a green result
from `--only` after ANY change to `cpp/neuralnet/nncachefileformat.*`,
`nncachecountlog.*`, `nnevalcontainer.*`, or the cache-dir layout more generally, run the
FULL suite (all groups, no `--only`) at least once -- a change under one of those files is a
sibling of every group here, not a member of the one that happens to be under active work.

-----------------------------------------------------------------------------------------
A KNOWN RACE THIS WITNESS DOES NOT WORK AROUND

cpp/command/analysis.cpp writes a query's response BEFORE erasing that query from
openRequests (there is a bot->clearSearch() in between). So a client that WAITED for the
response can still meet "Refusing cache_attach/cache_detach while 1 request(s) are open".
Legs P1 and P3's seed session send a dump and a detach after a query and are exposed to it.
When it happens the leg is reported FAIL with the disposition KNOWN-RACE and the engine's own
refusal text, so it is neither hidden, nor retried away, nor mistaken for the property
failing. A fix is in flight on another branch and is not on this one.
"""

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e2e_harness import DEFAULT_NET_A, DEFAULT_NET_B, Harness, Witness  # noqa: E402
from keying_legs import run_eval_cache_salt, run_keying, run_salt_independence  # noqa: E402
from persistence_legs import run_accounting, run_fingerprints, run_persistence  # noqa: E402


def parse_argv(argv):
  """(positional, groups-or-None, keep). Raises ValueError on a flag it does not know."""
  only = None
  keep = False
  positional = []
  i = 0
  while i < len(argv):
    if argv[i] == "--only" and i + 1 < len(argv):
      only = set(argv[i + 1].split(","))
      i += 2
    elif argv[i] == "--keep":
      keep = True
      i += 1
    elif argv[i].startswith("--"):
      raise ValueError("unknown flag %r" % argv[i])
    else:
      positional.append(argv[i])
      i += 1
  return positional, only, keep


def main():
  try:
    positional, only, keep = parse_argv(sys.argv[1:])
  except ValueError as e:
    print("%s\n\n%s" % (e, __doc__))
    return 2
  if len(positional) < 3:
    print(__doc__)
    return 2
  binary, template, workdir = positional[0], positional[1], positional[2]
  net_a = positional[3] if len(positional) > 3 else DEFAULT_NET_A
  net_b = positional[4] if len(positional) > 4 else DEFAULT_NET_B

  if os.path.exists(workdir):
    shutil.rmtree(workdir)
  os.makedirs(workdir)

  h = Harness(binary, template, workdir, net_a, net_b)
  w = Witness()

  name_a, name_b = h.model_names()
  print("MODELS   %s = %s" % (net_a, name_a))
  print("MODELS   %s = %s" % (net_b, name_b))
  print("GROUPS   %s" % (",".join(sorted(only)) if only else "S,P,A,K,K5,K6 (all)"))
  print()

  def running(group):
    return only is None or group in only

  fps, own_cost = run_fingerprints(w, h, name_a, name_b)
  fp_a, fp_b = fps[net_a], fps[net_b]

  if running("P"):
    run_persistence(w, h, name_a)
  if running("A"):
    run_accounting(w, h, name_a)
  if running("K"):
    run_keying(w, h, name_a, name_b, fp_a, fp_b)
  if running("K5"):
    run_eval_cache_salt(w, h, name_a, name_b, fp_a, fp_b, own_cost)
  if running("K6"):
    run_salt_independence(w, h, name_a, name_b, fp_a, fp_b)

  print("=" * 88)
  print("legs run: %d" % w.legs)
  if w.failures:
    print("FAILED LEGS (%d):" % len(w.failures))
    for f in w.failures:
      print("  - " + f)
    rc = 1
  else:
    print("ALL LEGS HELD")
    rc = 0
  if keep:
    print("workdir kept:    %s" % workdir)
  else:
    shutil.rmtree(workdir, ignore_errors=True)
    print("workdir removed: %s" % workdir)
  return rc


if __name__ == "__main__":
  sys.exit(main())
