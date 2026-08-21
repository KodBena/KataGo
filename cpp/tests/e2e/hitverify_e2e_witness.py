#!/usr/bin/env python3
"""DOES A PERSISTED-CACHE HIT STILL MEAN WHAT THE NET COMPUTES? The seen-red witness.

  usage: cpp/tests/e2e/hitverify_e2e_witness.py <katago> <config-template> <workdir> [<net>]
  e.g.   cpp/tests/e2e/hitverify_e2e_witness.py /home/bork/kg/.wt/hitverify-on/cpp/build/katago \\
             cpp/tests/e2e/persistence_e2e.cfg /home/bork/kg/.scratch-hitverify

WHAT THIS OBSERVES, AND WHY IT CANNOT BE A C++ SUITE. The property is "an entry that came off
disk, through the container decoder, into a level-0 source, and out to a search, still carries
the numbers the net would compute for that position". Every one of those words is about a
PROCESS BOUNDARY: one process must write the store and exit, an external actor must then forge
a corruption into the file, and a SECOND process must be served from it. Inside one process
there is no moment at which the forgery could happen, and no level-0 source that was ever
actually loaded from a file. cpp/tests/testnncacheverifyhits.cpp covers the comparison itself;
this covers the composition, from outside.

THE THREE LEGS.

  HV1  CLEAN. Process A analyzes N positions with an attached context, dumps, exits. Process B
       attaches the same context and re-analyzes the same positions. Every one of those is a
       LEVEL-0 hit, and cache_stats must report verifiedHits > 0 with mismatches == 0.
       verifiedHits > 0 is half the leg and not a formality: a run where the verifier never
       fired reports mismatches == 0 too, and would pass a leg that only looked at mismatches.

  HV2  SEEN RED. Between two processes, `katago nncachecorruptpayload` flips ONE payload scalar
       of ONE persisted evaluation and RE-MINTS both block checksums through the format's own
       functions. The file is valid; its meaning is wrong. Process C attaches it and
       re-analyzes; cache_stats must now report mismatches > 0.

       WHY THE CHECKSUM RE-MINT IS THE WHOLE METHOD: a corruption that left the checksums
       stale would witness nothing at all. The block would be discarded at load, no such entry
       would ever be served, and this leg would go green with the verifier deleted.

  HV3  THE INSTRUMENT STAYS OUTSIDE THE INSTRUMENT. Process C still answers its queries and
       exits cleanly with a mismatch outstanding. A verifier that refused, corrected, or
       crashed on a mismatch would be changing the thing it is there to observe.

REFUSES RATHER THAN PASSING VACUOUSLY. The first thing it does is ask the binary whether it
carries the verifier at all (the nncachecorruptpayload subcommand exists only in a
KATAGO_NNCACHE_VERIFY_HITS build). Against a default build every leg here would be trivially
green -- there would be no hitVerification block to disagree with -- so it stops, with exit 2
and a plain sentence, rather than reporting a pass it did not earn (ADR-0015).

EXIT STATUS: 0 every leg held, 1 a leg failed, 2 the witness could not run.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis_engine_driver import Engine, nn_rows_for  # noqa: E402
from e2e_harness import Witness  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_NET = "/home/bork/kg/14.gz"
CONTEXT = "hitverify"


def positions(count, visits, ownership=True):
  """`count` distinct 7x7 positions. Distinct BY CONSTRUCTION -- a different move sequence is a
  different board and so a different cache key -- rather than by hope. Small and cheap because
  a verify build pays a SECOND forward pass for every hit, so this suite does twice the neural
  net work of an ordinary one by design. See --visits for why one visit is load-bearing."""
  coords = ["c3", "e5", "c5", "e3", "d4", "b2", "f6", "b6", "f2", "d2", "d6", "b4"]
  out = []
  for i in range(count):
    moves = [["b", coords[i % len(coords)]], ["w", coords[(i + 3) % len(coords)]],
             ["b", coords[(i + 5) % len(coords)]]]
    out.append({
      "boardXSize": 7, "boardYSize": 7, "rules": "tromp-taylor", "komi": 7.5,
      "initialPlayer": "b", "moves": moves, "analyzeTurns": [3],
      "maxVisits": visits, "includeOwnership": ownership,
    })
  return out


def write_config(template, cache_dir, out_path):
  """The template's placeholders filled in for this run. Substituted, not appended: the engine
  refuses a config that states a key twice."""
  with open(template) as f:
    text = f.read()
  for placeholder, value in (("@CACHE_DIR@", cache_dir), ("@USE_EVAL_CACHE@", "false")):
    if placeholder not in text:
      raise RuntimeError("the config template %s holds no %s" % (template, placeholder))
    text = text.replace(placeholder, value)
  with open(out_path, "w") as f:
    f.write(text)
  return out_path


def binary_carries_the_verifier(binary):
  """Is this a KATAGO_NNCACHE_VERIFY_HITS build?

  ASKED OF THE BINARY, not of the build directory or a flag the caller passed. The instrument
  subcommand exists only under the option, and a build without it answers with the engine's
  own unknown-subcommand refusal -- so this is the property itself and not a proxy for it
  (ADR-0021 Rule 1)."""
  proc = subprocess.run([binary, "nncachecorruptpayload"], capture_output=True, text=True)
  blob = (proc.stdout or "") + (proc.stderr or "")
  # The verify build reaches the wrapper's own argument refusal, which names the subcommand and
  # what it wants. A default build never reaches it.
  return "expected exactly one argument, the path of an eval container file" in blob


def corrupt(binary, container):
  proc = subprocess.run([binary, "nncachecorruptpayload", container], capture_output=True, text=True)
  if proc.returncode != 0:
    raise RuntimeError("nncachecorruptpayload failed: %s%s" % (proc.stdout, proc.stderr))
  return proc.stdout


class Suite:
  def __init__(self, binary, config, workdir, net, w):
    self.binary, self.config, self.workdir, self.net, self.w = binary, config, workdir, net, w

  def session(self, requests, timeout=1800):
    e = Engine(self.binary, self.config, [self.net], timeout=timeout)
    out = {}
    try:
      for req in requests:
        out[req["id"]] = e.ask(req)
      rc, _, err = e.finish()
    finally:
      e.kill()
    return out, nn_rows_for(err, self.net), rc, err

  def analyze_session(self, ctx, queries, dump):
    reqs = [{"id": "a", "action": "cache_attach", "context": ctx}]
    reqs += [dict(q, id="q%d" % i) for i, q in enumerate(queries)]
    if dump:
      reqs += [{"id": "d", "action": "cache_dump", "context": ctx, "what": "both",
                "admission": {"all": True}}]
    reqs += [{"id": "s", "action": "cache_stats"},
             {"id": "x", "action": "cache_detach", "context": ctx}]
    return self.session(reqs)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("katago")
  ap.add_argument("config_template")
  ap.add_argument("workdir")
  ap.add_argument("net", nargs="?", default=DEFAULT_NET)
  ap.add_argument("--positions", type=int, default=6)
  # ONE VISIT PER QUERY, AND THAT IS LOAD-BEARING, NOT A SPEED CHOICE. HV3 reads the CLIENT's
  # own answer to prove the corrupt value reached it, and it can only do that if the entry the
  # instrument corrupts is one the client can see. At eight visits a query evaluates its root
  # plus seven interior nodes, the instrument corrupts whichever entry the dump wrote first, and
  # that is almost always an interior node whose value never surfaces in rootInfo -- observed:
  # 32 entries written, zero client-visible movement. At one visit the store holds exactly one
  # entry per query, every one of them a ROOT, so the corrupted entry is necessarily one the
  # client is shown.
  ap.add_argument("--visits", type=int, default=1)
  ap.add_argument("--keep", action="store_true")
  args = ap.parse_args()

  if not binary_carries_the_verifier(args.katago):
    print("REFUSING TO RUN: %s does not carry the hit verifier." % args.katago)
    print("Every leg of this witness would be trivially green against a default build, because")
    print("there would be no hitVerification block for a leg to disagree with. Build with")
    print("-DKATAGO_NNCACHE_VERIFY_HITS=ON and point this at that binary.")
    return 2

  if os.path.isdir(args.workdir) and not args.keep:
    shutil.rmtree(args.workdir)
  cache_dir = os.path.join(args.workdir, "cache")
  os.makedirs(cache_dir, exist_ok=True)
  config = write_config(args.config_template, cache_dir, os.path.join(args.workdir, "hv.cfg"))

  w = Witness()
  suite = Suite(args.katago, config, args.workdir, args.net, w)
  queries = positions(args.positions, args.visits)

  # ---- Process A: earn the entries and put them on disk. -------------------------------
  print("== HV: process A evaluates and dumps ==")
  a_out, a_rows, a_rc, _ = suite.analyze_session(CONTEXT, queries, dump=True)
  written = a_out["d"].get("evaluations", {}).get("entriesWritten")
  w.check(
    "HV0 process A pays real neural net work and writes it",
    a_rc == 0 and a_rows > 0 and written and written > 0,
    "rc=%d NN rows=%d entriesWritten=%s" % (a_rc, a_rows, written),
    "rc==0, NN rows>0, entriesWritten>0",
  )

  containers = sorted(glob.glob(os.path.join(cache_dir, CONTEXT + ".*.nnevals")))
  w.check(
    "HV0b exactly one container file for this context is on disk",
    len(containers) == 1,
    "%s holds %s" % (cache_dir, [os.path.basename(p) for p in containers]),
    "one .nnevals file",
  )
  if len(containers) != 1:
    return 1 if w.failures else 0

  # ---- Process B: the CLEAN leg. -------------------------------------------------------
  print("\n== HV1: process B is served from disk, and every hit is verified against the net ==")
  b_out, b_rows, b_rc, _ = suite.analyze_session(CONTEXT, queries, dump=False)
  b_verify = b_out["s"].get("hitVerification")
  w.check(
    "HV1a cache_stats carries a hitVerification block at all",
    b_verify is not None,
    "cache_stats keys: %s" % sorted(b_out["s"].keys()),
    "a hitVerification block (this is a verify build)",
  )
  if b_verify is None:
    return 1
  w.check(
    "HV1b *** every level-0 hit was RECOMPUTED and compared -- verifiedHits > 0 ***",
    b_verify.get("verifiedHits", 0) > 0,
    "verifiedHits=%s skippedNondeterministicSymmetry=%s skippedResidentOrigin=%s"
    % (b_verify.get("verifiedHits"), b_verify.get("skippedNondeterministicSymmetry"),
       b_verify.get("skippedResidentOrigin")),
    "verifiedHits > 0 -- a run that verified nothing reports mismatches==0 too",
  )
  w.check(
    "HV1c *** and every one of them AGREED with the net: mismatches == 0 ***",
    b_verify.get("mismatches") == 0,
    "mismatches=%s worstDeviationRatio=%s worstChannel=%r"
    % (b_verify.get("mismatches"), b_verify.get("worstDeviationRatio"), b_verify.get("worstChannel")),
    "mismatches == 0",
  )
  w.check(
    "HV1d the worst deviation is INSIDE the allowance, with headroom stated rather than implied",
    isinstance(b_verify.get("worstDeviationRatio"), (int, float))
    and b_verify["worstDeviationRatio"] <= 1.0,
    "worstDeviationRatio=%s (1.0 is the allowance) on %r"
    % (b_verify.get("worstDeviationRatio"), b_verify.get("worstChannel")),
    "worstDeviationRatio <= 1.0",
  )
  w.check(
    "HV1e the symmetry was PINNED, so nothing was skipped for want of one",
    b_verify.get("skippedNondeterministicSymmetry") == 0,
    "skippedNondeterministicSymmetry=%s (config has nnRandomize=false)"
    % b_verify.get("skippedNondeterministicSymmetry"),
    "0 -- with nnRandomize=false every hit is comparable",
  )

  # ---- The forgery. --------------------------------------------------------------------
  print("\n== HV2: one persisted payload is corrupted CHECKSUM-VALIDLY, between processes ==")
  report = corrupt(args.katago, containers[0])
  print(report.rstrip())
  corrupted_key = None
  for line in report.splitlines():
    if "corruptedKey" in line:
      corrupted_key = line.split()[-1]
  w.check(
    "HV2a the instrument names the key it corrupted, so the catch can be matched to it",
    corrupted_key is not None,
    "instrument output: %r" % report.strip()[:200],
    "a corruptedKey line",
  )

  # ---- Process C: the RED leg. ---------------------------------------------------------
  c_out, c_rows, c_rc, c_err = suite.analyze_session(CONTEXT, queries, dump=False)
  c_verify = c_out["s"].get("hitVerification") or {}
  w.check(
    "HV2b the corrupted container still LOADS -- the forgery is checksum-valid, so nothing "
    "upstream of the verifier caught it",
    c_out["a"].get("containerTail") == "intact" and c_out["a"].get("entriesInLevelZero") == written,
    "containerTail=%s entriesInLevelZero=%s (A wrote %s)"
    % (c_out["a"].get("containerTail"), c_out["a"].get("entriesInLevelZero"), written),
    "tail intact and the full key set attached -- otherwise this leg witnesses the checksum, "
    "not the verifier",
  )
  w.check(
    "HV2c *** the verifier CATCHES it at serve: mismatches > 0 ***",
    c_verify.get("mismatches", 0) > 0,
    "verifiedHits=%s mismatches=%s worstDeviationRatio=%s worstChannel=%r worstKey=%r"
    % (c_verify.get("verifiedHits"), c_verify.get("mismatches"),
       c_verify.get("worstDeviationRatio"), c_verify.get("worstChannel"), c_verify.get("worstKey")),
    "mismatches > 0",
  )
  w.check(
    "HV2d and it caught THE ENTRY THAT WAS CORRUPTED, not merely something",
    corrupted_key is not None and c_verify.get("worstKey") == corrupted_key,
    "worstKey=%r, the instrument corrupted %r" % (c_verify.get("worstKey"), corrupted_key),
    "the two keys are the same",
  )
  w.check(
    "HV2e on the channel that was corrupted",
    c_verify.get("worstChannel") == "whiteWinProb",
    "worstChannel=%r (the instrument moved whiteWinProb)" % c_verify.get("worstChannel"),
    "whiteWinProb",
  )
  w.check(
    "HV2f the mismatch is LOUD: the engine's own log carries the structured line",
    "NNCACHE HIT VERIFY MISMATCH" in c_err and (corrupted_key or "") in c_err,
    "the phrase and the key are%s in the engine's stderr"
    % ("" if "NNCACHE HIT VERIFY MISMATCH" in c_err else " NOT"),
    "a NNCACHE HIT VERIFY MISMATCH line naming the corrupted key",
  )

  # ---- HV3: the observer did not become a corrector. -----------------------------------
  #
  # OBSERVED ON THE CLIENT'S OWN ANSWER, not on the exit code. An earlier draft of this leg
  # checked rc==0 and no query error, which a verifier that silently substituted the RECOMPUTED
  # value for the served one would also pass -- a symptom, not the property (ADR-0021 Rule 1).
  # The property is that the corrupt value reached the caller unchanged, so that is what is read:
  # the raw root evaluation of each query, before and after the corruption, from the two engines'
  # own responses.
  print()
  b_roots = [b_out["q%d" % i]["rootInfo"]["rawWinrate"] for i in range(len(queries))]
  c_roots = [c_out["q%d" % i]["rootInfo"]["rawWinrate"] for i in range(len(queries))]
  moved = [i for i in range(len(queries)) if abs(b_roots[i] - c_roots[i]) > 1e-9]
  w.check(
    "HV3a *** the CORRUPT value reached the client: exactly one query's raw root evaluation "
    "moved, and nothing was silently corrected ***",
    len(moved) == 1,
    "queries whose rawWinrate moved between the clean and corrupted runs: %s (clean %s, "
    "corrupted %s)" % (moved, [round(x, 6) for x in b_roots], [round(x, 6) for x in c_roots]),
    "exactly one -- zero would mean the verifier substituted its recompute, more than one would "
    "mean something other than the corruption moved",
  )
  # THE EXPECTED MOVEMENT IS DERIVED, NOT EYEBALLED. The instrument moves whiteWinProb by 0.25
  # and leaves whiteLossProb alone; the analysis engine reports rawWinrate as
  # 0.5 + (win - loss)/2, so a 0.25 move on one term of that difference surfaces as exactly
  # HALF of it at the client. Asserting 0.25 here would be asserting the wrong number and would
  # have made this leg unfalsifiable in the other direction -- it went red on the real value
  # first, which is how the relationship was pinned down rather than assumed.
  expected_delta = 0.25 / 2.0
  w.check(
    "HV3b and it moved by exactly the corruption's own magnitude, not by numerical noise",
    len(moved) == 1 and abs(abs(b_roots[moved[0]] - c_roots[moved[0]]) - expected_delta) < 1e-6,
    "delta=%s, expected %s (whiteWinProb moved 0.25; rawWinrate is 0.5+(win-loss)/2, so the "
    "client sees half)"
    % (abs(b_roots[moved[0]] - c_roots[moved[0]]) if len(moved) == 1 else None, expected_delta),
    "|delta| == %s to within 1e-6" % expected_delta,
  )
  w.check(
    "HV3c and the engine still answered every query and exited cleanly with the mismatch "
    "outstanding",
    c_rc == 0 and all(("error" not in c_out["q%d" % i]) for i in range(len(queries))),
    "rc=%d, %d queries answered without error, mismatches=%s"
    % (c_rc, len(queries), c_verify.get("mismatches")),
    "rc==0 and no query carried an error -- a verifier that refused or crashed would be "
    "changing what it is there to observe",
  )

  # ---- HV4: the OTHER arm that serves deserialized bytes. -------------------------------
  #
  # NNEvaluator::evaluate's cache-hit branch has two arms, and an audit found the first draft of
  # this feature verified only one. The second is the OWNERMAP FALL-THROUGH: the cached entry
  # answers everything except an ownership map the caller has now asked for, so the net runs
  # anyway and the cached entry's scalars and whole policy array are copied back over the fresh
  # result -- reaching the client exactly as they would through the early return, AND being
  # stored into level 1, where a corrupt value becomes resident.
  #
  # REACHING THAT ARM TAKES CONSTRUCTION, AND THE FIRST TWO ATTEMPTS AT THIS LEG DID NOT REACH IT.
  # An analysis query's ROOT always requests an ownership map -- search/searchnnhelpers.cpp:72,
  # `includeOwnerMap = isRoot || alwaysIncludeOwnerMap` -- so a store built only from roots holds
  # entries that already carry one whatever includeOwnership says, and the fall-through is never
  # reached; that draft passed with the fall-through's verification deleted, which is what a leg
  # asserting nothing looks like. Nor can an interior position simply be re-sent as a root: the
  # root of a query carries different MiscNNInputParams (root policy temperature, playout
  # doubling advantage), which go into the cache key, so it is a different key and a plain miss.
  #
  # WHAT DOES REACH IT: the SAME queries, run twice, with the ownership request flipped. Process
  # D searches with includeOwnership off, so its INTERIOR evaluations are stored with no map
  # (only its roots have one). Process E sends the identical queries with includeOwnership on,
  # which sets alwaysIncludeOwnerMap and makes every interior evaluation ask for a map -- against
  # the very entries D stored without one, under identical keys. That is the deployment shape as
  # well: two leaves on one shared directory, configured differently.
  #
  # AND THE LEG COUNTS THE ARM, NOT THE TOTAL. verifiedFallThroughHits moves only when the
  # fall-through arm compares something, so deleting that call makes this leg red and nothing
  # else does.
  print("\n== HV4: the ownermap fall-through arm -- stored without a map, read wanting one ==")
  ctx2 = CONTEXT + "-fallthrough"
  d_queries = positions(args.positions, visits=8, ownership=False)
  e_queries = positions(args.positions, visits=8, ownership=True)
  d_out, d_rows, d_rc, _ = suite.analyze_session(ctx2, d_queries, dump=True)
  d_written = d_out["d"].get("evaluations", {}).get("entriesWritten")
  w.check(
    "HV4a process D stores a search tree, whose INTERIOR entries carry no ownership map",
    d_rc == 0 and d_written and d_written > args.positions,
    "rc=%d NN rows=%d entriesWritten=%s (more than the %d roots, so interior nodes are in there)"
    % (d_rc, d_rows, d_written, args.positions),
    "entriesWritten > one per query",
  )

  containers2 = sorted(glob.glob(os.path.join(cache_dir, ctx2 + ".*.nnevals")))
  if len(containers2) != 1:
    w.check("HV4 could be run at all", False,
            "containers=%s" % [os.path.basename(x) for x in containers2],
            "exactly one container")
    return 1

  e_out, e_rows, e_rc, _ = suite.analyze_session(ctx2, e_queries, dump=False)
  e_verify = e_out["s"].get("hitVerification") or {}
  w.check(
    "HV4c *** the fall-through arm IS verified: verifiedFallThroughHits > 0 with no mismatch ***",
    e_verify.get("verifiedFallThroughHits", 0) > 0 and e_verify.get("mismatches") == 0,
    "verifiedFallThroughHits=%s of verifiedHits=%s, mismatches=%s, NN rows=%d"
    % (e_verify.get("verifiedFallThroughHits"), e_verify.get("verifiedHits"),
       e_verify.get("mismatches"), e_rows),
    "verifiedFallThroughHits > 0 and mismatches == 0 -- this counter is zero if the "
    "fall-through's verification is removed, which is the whole point of counting the arms apart",
  )

  print(corrupt(args.katago, containers2[0]).rstrip())
  f_out, f_rows, f_rc, f_err = suite.analyze_session(ctx2, e_queries, dump=False)
  f_verify = f_out["s"].get("hitVerification") or {}
  w.check(
    "HV4d *** and a corruption served through THIS arm is caught, on the right key and channel ***",
    f_verify.get("mismatches", 0) > 0 and f_verify.get("worstChannel") == "whiteWinProb"
    and f_verify.get("verifiedFallThroughHits", 0) > 0,
    "mismatches=%s worstChannel=%r worstKey=%r verifiedFallThroughHits=%s"
    % (f_verify.get("mismatches"), f_verify.get("worstChannel"), f_verify.get("worstKey"),
       f_verify.get("verifiedFallThroughHits")),
    "mismatches > 0 on whiteWinProb, with the fall-through arm having done comparisons",
  )
  w.check(
    "HV4e and the engine survives a corrupted store on this arm too, reporting rather than "
    "refusing",
    f_rc == 0 and all(("error" not in f_out["q%d" % i]) for i in range(len(e_queries))),
    "rc=%d, %d queries answered without error, mismatches=%s verifiedFallThroughHits=%s"
    % (f_rc, len(e_queries), f_verify.get("mismatches"), f_verify.get("verifiedFallThroughHits")),
    "rc==0 and no query carried an error",
  )

  print()
  print("%d legs, %d failed" % (w.legs, len(w.failures)))
  for f in w.failures:
    print("  FAILED: %s" % f)
  if not args.keep:
    shutil.rmtree(args.workdir, ignore_errors=True)
  return 1 if w.failures else 0


if __name__ == "__main__":
  sys.exit(main())
