"""TWO ENGINE PROCESSES, ONE CACHE DIRECTORY: does the sharing actually work, and does it
stay whole while both are using it?

    python3 cpp/tests/e2e/sharedcache_e2e_witness.py --cache-dir ~/nncache

WHAT THIS IS FOR, AND WHY IT IS NOT persistence_e2e_witness.py. That witness proves the
FORMAT: a process writes, a later process reads it back, and the numbers line up. Every one
of its legs runs one engine at a time, and it makes each leg its OWN fresh cache directory.
The deployment this suite is for is the other shape -- a KataProxy fanning queries across
several same-model KataGo leaves ON ONE HOST, all pointed at ONE directory, all live at
once. Two things only that shape can answer:

  * that a leaf really is served by a sibling's work rather than merely tolerating its files;
  * that a reader attaching WHILE a writer dumps sees a whole state and never a partial one.

THE DIRECTORY IS A PARAMETER, and that is the point of the whole suite. The operator's plan
is to point the shared cache at an sshfs mount so it can be watched live, and sshfs is a FUSE
filesystem -- exactly where advisory locking is least certain. So nothing here hardcodes a
path: the same suite runs unchanged against /tmp, against a local disk, and against the
mount, and comparing the three is the bring-up procedure.

WHICH IS WHY IT REFUSES TO RUN BEFORE IT KNOWS THE LOCK EXCLUDES. The first thing this does
is run `katago lockfsprobe` against the target directory. On a filesystem where flock cannot
exclude, two engines would each believe they hold the lock, their appends would interleave
mid-file, and the reader would discard every block after the first torn one -- so the suite
says so in plain text and STOPS, with no engine started and no byte written. It is not a
suite that hangs there, and it is not a suite that produces a green run over a directory
where the property under test is absent (ADR-0015: a result is only as good as the
environment that produced it).

WHAT IS ASSERTED IS A COUNTER, NEVER "IT DID NOT CRASH". Three independent numbers witness
one process being served by another's work, and they are independent on purpose -- one is the
engine's opinion of its own state, one is the work the neural net did or did not do, and one
is what the second process had left to write:

  cache_attach's  entriesInLevelZero   == the first process's cache_dump entriesWritten
  the engine's    NN rows              == 0 in the second process, for the same query
  cache_dump's    entriesWritten/bytesAppended == 0 in the second process, having added nothing

EXIT STATUS: 0 every scenario passed, 1 a scenario failed, 2 the suite could not run (the
lock probe refused the directory, a file was missing, an engine would not start).
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis_engine_driver import Engine, nn_rows_for  # noqa: E402
from e2e_harness import Witness, disposition  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
# cpp/tests/e2e -> cpp/tests -> cpp -> the repository root.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))

DEFAULT_BINARY = os.path.join(REPO, "cpp", "build", "katago")
DEFAULT_TEMPLATE = os.path.join(HERE, "persistence_e2e.cfg")
DEFAULT_NET = "/home/bork/kg/14.gz"

# Sampling gap, between one attach-and-read and the next, in the scenario where a reader runs
# alongside a writer. It is not decoration. flock offers no fairness guarantee, so a reader
# that re-takes the shared lock with NO gap can starve a writer for its whole 20-second
# deadline -- that starvation was witnessed during the locking work itself. A suite whose
# reader left no gap would be measuring its own sampling loop rather than the engine.
SAMPLE_GAP_SECONDS = 0.015


def positions(count, visits=8):
  """`count` distinct 7x7 positions, each cheap, each a different set of cache keys.

  Distinct by CONSTRUCTION -- a different move sequence is a different board, so a different
  hash -- rather than by hope. Small boards and few visits because the suite's job is to be
  run repeatedly at bring-up, including over a network mount.
  """
  coords = ["c3", "e5", "c5", "e3", "d4", "b2", "f6", "b6", "f2", "d2", "d6", "b4"]
  out = []
  for i in range(count):
    moves = [["b", coords[i % len(coords)]], ["w", coords[(i + 3) % len(coords)]],
             ["b", coords[(i + 5) % len(coords)]]]
    out.append({
      "boardXSize": 7, "boardYSize": 7, "rules": "tromp-taylor", "komi": 7.5,
      "initialPlayer": "b", "moves": moves, "analyzeTurns": [3],
      "maxVisits": visits, "includeOwnership": True,
    })
  return out


def write_config(template, cache_dir, out_path):
  """The template's placeholders filled in for THIS run's directory.

  Substituted into the template's one statement of each key rather than appended, because the
  engine refuses a config that states a key twice and a second statement would be a second
  home for the setting (ADR-0012 P1).
  """
  with open(template) as f:
    text = f.read()
  for placeholder, value in (("@CACHE_DIR@", cache_dir), ("@USE_EVAL_CACHE@", "false")):
    if placeholder not in text:
      raise RuntimeError("the config template %s holds no %s" % (template, placeholder))
    text = text.replace(placeholder, value)
  with open(out_path, "w") as f:
    f.write(text)
  return out_path


def probe_lock(binary, cache_dir):
  """`katago lockfsprobe` over the target directory. Returns (rc, combined output).

  Run through the SUBCOMMAND rather than reimplemented here, so what the suite gates on is
  the same NNCacheFileLock the engines it starts will use (ADR-0021: the witness observes the
  property, not a symptom).
  """
  proc = subprocess.run([binary, "lockfsprobe", cache_dir],
                        capture_output=True, text=True, timeout=300)
  return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def context_files(cache_dir, context):
  """Every file this run's context owns, including its lock file."""
  return sorted(glob.glob(os.path.join(cache_dir, context + ".*")))


# WHAT THE CONFIG-DRIVEN LIFECYCLE SAYS ABOUT ITSELF, in the engine's own log. These two lines
# are the only surface a leaf's operator has for "did the attach happen, and what did the
# shutdown write", precisely because the whole point of the feature is that no client ever asked
# for either. Reading them here rather than eyeballing them is what makes SC5 a witness.
_ATTACH_LINE = re.compile(
  r'nnCacheAttachContext: model "(?P<model>[^"]*)" attached context "(?P<context>[^"]*)": '
  r'(?P<entries>\d+) entries in level 0'
)
_SHUTDOWN_LINE = re.compile(
  r'nnCacheAttachContext: model "(?P<model>[^"]*)" dumped context "(?P<context>[^"]*)" '
  r'at shutdown: (?P<result>\{.*\})$'
)


def startup_attaches(stderr, context):
  """[{model, entries}] for every startup attach of `context` the engine logged, in order."""
  out = []
  for line in stderr.splitlines():
    m = _ATTACH_LINE.search(line)
    if m and m.group("context") == context:
      out.append({"model": m.group("model"), "entries": int(m.group("entries"))})
  return out


def shutdown_dumps(stderr, context):
  """[{model, result}] for every shutdown dump of `context`, the result being the dump's own
  response object -- the same one a client that had sent cache_dump by hand would have read."""
  out = []
  for line in stderr.splitlines():
    m = _SHUTDOWN_LINE.search(line)
    if m and m.group("context") == context:
      out.append({"model": m.group("model"), "result": json.loads(m.group("result"))})
  return out


class Suite:
  def __init__(self, w, binary, config, net, cache_dir, context, workdir):
    self.w, self.binary, self.config, self.net = w, binary, config, net
    self.cache_dir, self.context, self.workdir = cache_dir, context, workdir

  def session(self, requests, timeout=1800, config=None):
    """One engine process, one request at a time. Returns (responses by id, NN rows, rc)."""
    e = Engine(self.binary, config or self.config, [self.net], timeout=timeout)
    out = {}
    try:
      for req in requests:
        out[req["id"]] = e.ask(req)
      rc, _, err = e.finish()
    finally:
      e.kill()
    return out, nn_rows_for(err, self.net), rc

  def lifecycle_config(self, name, cache_dir, extra_lines):
    """This run's config plus the lifecycle keys, as its own file.

    A separate file rather than an -override-config flag because the claim SC5 makes is about
    what an OPERATOR puts in a leaf's config file; a witness that only ever exercised the
    override path would leave the config-file path untested (ADR-0021 Rule 1).
    """
    path = os.path.join(self.workdir, name)
    with open(self.config) as f:
      text = f.read()
    # The cache directory may differ from the suite's own -- the refusal leg points at one that
    # is not there -- so it is substituted rather than inherited.
    text = re.sub(r"(?m)^nnCacheDir = .*$", "nnCacheDir = " + cache_dir, text)
    with open(path, "w") as f:
      f.write(text + "\n" + "\n".join(extra_lines) + "\n")
    return path

  # ---------------------------------------------------------------------------------------
  # SC1: one process writes, a LATER process is served by it.
  # ---------------------------------------------------------------------------------------
  def sc1_sequential_reuse(self):
    print("== SC1: process A dumps; a LATER process B is served entirely by A's keys ==")
    ctx = self.context + "-sc1"
    queries = positions(3)

    a_reqs = [{"id": "a", "action": "cache_attach", "context": ctx}]
    a_reqs += [dict(q, id="q%d" % i) for i, q in enumerate(queries)]
    a_reqs += [{"id": "d", "action": "cache_dump", "context": ctx, "what": "both", "admission": {"all": True}},
               {"id": "x", "action": "cache_detach", "context": ctx}]
    a_out, a_rows, a_rc = self.session(a_reqs)
    written = a_out["d"].get("evaluations", {}).get("entriesWritten")

    self.w.check(
      "SC1a process A pays real neural net work and writes ALL of it to the shared directory",
      a_rc == 0 and a_rows > 0 and written == a_rows,
      "rc=%d NN rows=%d entriesWritten=%s bytesAppended=%s %s"
      % (a_rc, a_rows, written, a_out["d"].get("evaluations", {}).get("bytesAppended"),
         disposition(a_out["d"])),
      "rc==0, NN rows>0, entriesWritten == NN rows",
    )
    on_disk = [os.path.basename(p) for p in context_files(self.cache_dir, ctx)]
    self.w.check(
      "SC1b the files landed in the directory the OPERATOR named, not somewhere derived",
      any(n.endswith(".nncounts") for n in on_disk) and any(n.endswith(".nnevals") for n in on_disk),
      "%s holds %s" % (self.cache_dir, on_disk),
      "a .nncounts and a .nnevals for this context under --cache-dir",
    )

    b_reqs = [{"id": "a", "action": "cache_attach", "context": ctx}]
    b_reqs += [dict(q, id="q%d" % i) for i, q in enumerate(queries)]
    b_reqs += [{"id": "d", "action": "cache_dump", "context": ctx, "what": "both", "admission": {"all": True}},
               {"id": "s", "action": "cache_stats"},
               {"id": "x", "action": "cache_detach", "context": ctx}]
    b_out, b_rows, b_rc = self.session(b_reqs)
    b_attach, b_dump = b_out["a"], b_out["d"]
    b_evals = b_dump.get("evaluations", {})

    self.w.check(
      "SC1c B's attach counts EXACTLY the keys A wrote",
      b_attach.get("entriesInLevelZero") == written and b_attach.get("containerTail") == "intact"
      and b_attach.get("countLogTail") == "intact",
      "entriesInLevelZero=%s (A wrote %s) containerTail=%s countLogTail=%s %s"
      % (b_attach.get("entriesInLevelZero"), written, b_attach.get("containerTail"),
         b_attach.get("countLogTail"), disposition(b_attach)),
      "entriesInLevelZero == %s, both tails intact" % written,
    )
    self.w.check(
      "SC1d *** B pays ZERO neural net rows for the queries A already paid for ***",
      b_rows == 0,
      "NN rows=%d in B; A paid %d for the same %d queries" % (b_rows, a_rows, len(queries)),
      "NN rows == 0 exactly -- not fewer, none",
    )
    self.w.check(
      "SC1e and B has NOTHING to add: it does not rewrite what A already stored",
      "error" not in b_dump and b_evals.get("entriesWritten") == 0
      and b_evals.get("bytesAppended") == 0,
      "entriesWritten=%s bytesAppended=%s alreadyPersisted=%s notResident=%s %s"
      % (b_evals.get("entriesWritten"), b_evals.get("bytesAppended"),
         b_evals.get("alreadyPersisted"), b_evals.get("notResident"), disposition(b_dump)),
      "entriesWritten==0 and bytesAppended==0",
    )
    stats_dir = b_out["s"].get("cacheDirectory")
    self.w.check(
      "SC1f the engine agrees, in its own words, which directory it is sharing",
      stats_dir is not None and os.path.realpath(stats_dir) == os.path.realpath(self.cache_dir),
      "cache_stats cacheDirectory=%r, --cache-dir was %r" % (stats_dir, self.cache_dir),
      "the two are the same directory",
    )
    self.w.check("SC1g both engines exit cleanly", a_rc == 0 and b_rc == 0,
                 "A rc=%d B rc=%d" % (a_rc, b_rc), "both rc==0")
    print()
    return written

  # ---------------------------------------------------------------------------------------
  # SC2: the deployment's actual shape -- both processes ALIVE at the same time.
  # ---------------------------------------------------------------------------------------
  def sc2_two_live_engines(self):
    print("== SC2: two engines LIVE AT ONCE on one directory; B is served by A's dump ==")
    ctx = self.context + "-sc2"
    queries = positions(3)

    a = Engine(self.binary, self.config, [self.net])
    b = Engine(self.binary, self.config, [self.net])
    try:
      # B is up and running BEFORE A writes anything: this is the leaf that was already
      # serving when its sibling learned something, not a process started afterwards.
      b_first = b.ask({"id": "b0", "action": "cache_attach", "context": ctx})
      self.w.check(
        "SC2a B attaches an empty context while A is also running",
        b_first.get("entriesInLevelZero") == 0,
        "entriesInLevelZero=%s %s" % (b_first.get("entriesInLevelZero"), disposition(b_first)),
        "entriesInLevelZero == 0",
      )
      b.ask({"id": "b1", "action": "cache_detach", "context": ctx})

      a.ask({"id": "a", "action": "cache_attach", "context": ctx})
      for i, q in enumerate(queries):
        a.ask(dict(q, id="aq%d" % i))
      a_dump = a.ask({"id": "ad", "action": "cache_dump", "context": ctx, "what": "both", "admission": {"all": True}})
      written = a_dump.get("evaluations", {}).get("entriesWritten")

      # A stays UP and attached. B now re-attaches, with a live sibling holding the context.
      b_attach = b.ask({"id": "b2", "action": "cache_attach", "context": ctx})
      self.w.check(
        "SC2b B re-attaches while A is still live and attached, and sees A's keys",
        b_attach.get("entriesInLevelZero") == written and written > 0
        and b_attach.get("containerTail") == "intact",
        "entriesInLevelZero=%s (A wrote %s) containerTail=%s %s"
        % (b_attach.get("entriesInLevelZero"), written, b_attach.get("containerTail"),
           disposition(b_attach)),
        "entriesInLevelZero == A's entriesWritten (> 0), tail intact",
      )
      for i, q in enumerate(queries):
        b.ask(dict(q, id="bq%d" % i))
      b.ask({"id": "b3", "action": "cache_detach", "context": ctx})
      a.ask({"id": "ax", "action": "cache_detach", "context": ctx})

      a_rc, _, a_err = a.finish()
      b_rc, _, b_err = b.finish()
    finally:
      a.kill()
      b.kill()

    a_rows, b_rows = nn_rows_for(a_err, self.net), nn_rows_for(b_err, self.net)
    self.w.check(
      "SC2c *** the live sibling pays ZERO neural net rows for its sibling's queries ***",
      a_rows > 0 and b_rows == 0,
      "A NN rows=%d, B NN rows=%d over the same %d queries" % (a_rows, b_rows, len(queries)),
      "A > 0 and B == 0",
    )
    self.w.check("SC2d both live engines exit cleanly", a_rc == 0 and b_rc == 0,
                 "A rc=%d B rc=%d" % (a_rc, b_rc), "both rc==0")
    print()

  # ---------------------------------------------------------------------------------------
  # SC4: the DEFAULT admission, and the cross-session bootstrap it exists for.
  # ---------------------------------------------------------------------------------------
  def sc4_default_admission_bootstrap(self):
    """cache_dump with NO "admission" field, across two processes over one directory.

    THIS IS THE LEG THE WHOLE CURRENCY CHANGE IS FOR, and it is here rather than in the C++
    suite because the claim is about two ENGINE PROCESSES and a real directory: an in-process
    test can witness the arithmetic but not that a second engine's admission decision reads
    what the first engine's dump wrote (ADR-0021 Rule 1).

    The default is minObservations(2): store only positions that have come up more than once.
    So the shape under test is three-legged and each leg is a different assertion:

      A first session evaluates each position once. MOST of its cache keys are therefore below
      the default threshold and their evaluations are refused -- and the COUNTS dump writes a
      row for every one of them anyway, carrying 1. Writing those rows is the entire mechanism;
      a dump that filtered once-seen keys out of the count log would make the threshold
      permanently unreachable, and on the evaluations side alone that failure would look
      identical to this leg passing.

      NOT ALL of them: three 7x7 searches share sub-positions, so a handful of keys really are
      presented twice inside the first session and really do clear the default there. That is
      the currency working, not the fixture leaking, and the leg asserts the discrimination
      (most refused, some admitted) rather than a clean zero that would be a fact about the
      query set rather than about admission.

      A second session asks for the SAME positions. The keys that were at 1 are now at 2, clear
      the default, and are written.

      A third session is served by the UNION of both dumps, having paid no neural net work.
    """
    print("== SC4: the default admission (seen twice) bootstraps across two processes ==")
    ctx = self.context + "-sc4"
    queries = positions(3)

    # SESSION ONE: every position evaluated exactly once. No "admission" field at all.
    one_reqs = [{"id": "a", "action": "cache_attach", "context": ctx}]
    one_reqs += [dict(q, id="q%d" % i) for i, q in enumerate(queries)]
    one_reqs += [{"id": "d", "action": "cache_dump", "context": ctx, "what": "both"},
                 {"id": "s", "action": "cache_stats"},
                 {"id": "x", "action": "cache_detach", "context": ctx}]
    one_out, one_rows, one_rc = self.session(one_reqs)
    one_dump = one_out["d"]
    one_evals = one_dump.get("evaluations", {})
    one_counts = one_dump.get("counts", {})

    self.w.check(
      "SC4a a dump with NO \"admission\" field is accepted, and defaults to seen-at-least-twice",
      one_rc == 0 and "error" not in one_dump and "2" in str(one_evals.get("admission", "")),
      "rc=%d admission=%s %s" % (one_rc, one_evals.get("admission"), disposition(one_dump)),
      "the request is accepted with the field absent, and the admission it reports names 2",
    )
    one_written = one_evals.get("entriesWritten")
    self.w.check(
      "SC4b the default DISCRIMINATES: most of a first session's keys are refused as once-seen",
      one_rc == 0 and one_rows > 0 and one_evals.get("belowThreshold", 0) > 0
      and one_written is not None and one_written < one_rows
      and one_evals.get("belowThreshold", 0) + one_written == one_rows,
      "NN rows=%d entriesWritten=%s belowThreshold=%s"
      % (one_rows, one_written, one_evals.get("belowThreshold")),
      "real neural net work was done; the refused and the written partition it exactly, and "
      "the refused are the majority -- the handful written are keys these three searches "
      "genuinely presented twice",
    )
    self.w.check(
      "SC4c the COUNT log still gets a row for every once-seen position -- the bootstrap itself",
      one_rc == 0 and one_counts.get("bytesAppended", 0) > 0 and one_counts.get("rowsInLog", 0) > 0,
      "counts bytesAppended=%s rowsInLog=%s (observationsThisSession=%s)"
      % (one_counts.get("bytesAppended"), one_counts.get("rowsInLog"),
         one_out["s"].get("observationsThisSession")),
      "rows were written for keys whose evaluations were NOT written; without them no second "
      "session could ever raise one to 2",
    )

    # SESSION TWO: the same positions again, in a new process. Each reaches 2.
    two_reqs = [{"id": "a", "action": "cache_attach", "context": ctx}]
    two_reqs += [dict(q, id="q%d" % i) for i, q in enumerate(queries)]
    two_reqs += [{"id": "d", "action": "cache_dump", "context": ctx, "what": "both"},
                 {"id": "x", "action": "cache_detach", "context": ctx}]
    two_out, two_rows, two_rc = self.session(two_reqs)
    two_evals = two_out["d"].get("evaluations", {})

    self.w.check(
      "SC4d the SECOND session's dump writes them: the counts accumulated across processes",
      two_rc == 0 and two_evals.get("entriesWritten", 0) > 0,
      "attach entriesInLevelZero=%s NN rows=%d entriesWritten=%s belowThreshold=%s %s"
      % (two_out["a"].get("entriesInLevelZero"), two_rows, two_evals.get("entriesWritten"),
         two_evals.get("belowThreshold"), disposition(two_out["d"])),
      "the same default that admitted nothing in session one admits in session two, because "
      "the count each key carries is now 2 -- one from each process",
    )

    # SESSION THREE: served by what session two wrote, paying no neural net work.
    three_reqs = [{"id": "a", "action": "cache_attach", "context": ctx}]
    three_reqs += [dict(q, id="q%d" % i) for i, q in enumerate(queries)]
    three_reqs += [{"id": "x", "action": "cache_detach", "context": ctx, "discardUndumped": True}]
    three_out, three_rows, three_rc = self.session(three_reqs)

    self.w.check(
      "SC4e a third process is served entirely by what the bootstrap admitted",
      three_rc == 0 and three_rows == 0
      and three_out["a"].get("entriesInLevelZero") == one_written + two_evals.get("entriesWritten", 0),
      "entriesInLevelZero=%s (session one wrote %s, session two %s) NN rows=%d"
      % (three_out["a"].get("entriesInLevelZero"), one_written,
         two_evals.get("entriesWritten"), three_rows),
      "zero neural net rows, and the attach counts the UNION of both dumps -- the container is "
      "cumulative, so the figure to match is the sum and not session two's own number",
    )
    print()

  # ---------------------------------------------------------------------------------------
  # SC5: the same sharing as SC1, driven ENTIRELY BY THE CONFIG FILE. No wire cache verb.
  # ---------------------------------------------------------------------------------------
  def sc5_config_driven_lifecycle(self):
    """SC1's shape, with every cache_attach and cache_dump deleted from the client.

    WHY THIS LEG EXISTS AND SC1 DOES NOT COVER IT. SC1 proves the STORE works across processes;
    every one of its sessions drives the store by hand, sending cache_attach before its queries
    and cache_dump after them. The deployment this feature is for has no such client: a proxy
    fans queries at a pool of leaves and knows nothing about cards, contexts or dumps. So what
    has to be witnessed here is that the LEAF does it -- from its own config file, at its own
    startup and its own exit -- and the way to witness that is to send the engine nothing but
    analysis queries and see the sharing happen anyway.

    The requests below are therefore ORDINARY QUERIES AND NOTHING ELSE, with one exception that
    is deliberately not part of the claim: SC5e sends wire verbs on purpose, to witness that a
    config-attached context is an ordinary attached context and the client-driven surface still
    composes with it.
    """
    print("== SC5: the SAME sharing, driven purely by the config file -- ZERO cache verbs ==")
    ctx = self.context + "-sc5"
    queries = positions(3)
    # minObservations 0 is accept-all, so one session's work is enough to serve the next -- the
    # same thing SC1 asks for with "admission": {"all": true}, here as the operator's config key.
    cfg = self.lifecycle_config(
      "sharedcache-sc5.cfg", self.cache_dir,
      ["nnCacheAttachContext = " + ctx, "nnCacheDumpMinObservations = 0",
       # OFF for SC5, on purpose: this scenario's whole claim is about the STARTUP attach and the
       # SHUTDOWN dump, and a periodic pass firing mid-scenario would make "exactly one dump line"
       # a fact about how long the engine happened to live. SC6 is where the interval is the claim.
       "nnCacheDumpIntervalMinutes = 0"],
    )

    a = Engine(self.binary, cfg, [self.net])
    try:
      for i, q in enumerate(queries):
        a.ask(dict(q, id="q%d" % i))
      a_rc, _, a_err = a.finish()
    finally:
      a.kill()
    a_rows = nn_rows_for(a_err, self.net)
    a_attaches = startup_attaches(a_err, ctx)
    a_dumps = shutdown_dumps(a_err, ctx)
    a_written = a_dumps[0]["result"]["evaluations"]["entriesWritten"] if a_dumps else None

    self.w.check(
      "SC5a engine A attaches its context AT STARTUP from the config, with nothing sent to it",
      a_rc == 0 and len(a_attaches) == 1 and a_attaches[0]["entries"] == 0,
      "rc=%d startup attaches=%s" % (a_rc, a_attaches),
      "exactly one startup attach of this context, reporting 0 entries -- the context is new",
    )
    self.w.check(
      "SC5b *** and DUMPS it at clean shutdown, on stdin closing, with nothing sent to it ***",
      a_rc == 0 and len(a_dumps) == 1 and a_rows > 0 and a_written == a_rows,
      "rc=%d NN rows=%d shutdown dumps=%d entriesWritten=%s bytesAppended=%s"
      % (a_rc, a_rows, len(a_dumps), a_written,
         a_dumps[0]["result"]["evaluations"]["bytesAppended"] if a_dumps else None),
      "exactly one shutdown dump, and it wrote every key the neural net paid for",
    )
    on_disk = [os.path.basename(p) for p in context_files(self.cache_dir, ctx)]
    self.w.check(
      "SC5c the files landed under the configured nnCacheDir",
      any(n.endswith(".nncounts") for n in on_disk) and any(n.endswith(".nnevals") for n in on_disk),
      "%s holds %s" % (self.cache_dir, on_disk),
      "a .nncounts and a .nnevals for this context",
    )

    b = Engine(self.binary, cfg, [self.net])
    try:
      for i, q in enumerate(queries):
        b.ask(dict(q, id="q%d" % i))
      b_rc, _, b_err = b.finish()
    finally:
      b.kill()
    b_rows = nn_rows_for(b_err, self.net)
    b_attaches = startup_attaches(b_err, ctx)

    self.w.check(
      "SC5d *** engine B pays ZERO neural net rows for A's queries, having sent no cache verb "
      "and been sent none ***",
      b_rc == 0 and b_rows == 0 and len(b_attaches) == 1 and b_attaches[0]["entries"] == a_written,
      "rc=%d NN rows=%d (A paid %d) startup attach entries=%s (A wrote %s)"
      % (b_rc, b_rows, a_rows, b_attaches[0]["entries"] if b_attaches else None, a_written),
      "NN rows == 0 exactly, and B's startup attach counts exactly the keys A's shutdown wrote",
    )

    # SC5e: the wire verbs still compose. A config-attached context is an ordinary attached one.
    c = Engine(self.binary, cfg, [self.net])
    try:
      c_stats = c.ask({"id": "s", "action": "cache_stats"})
      for i, q in enumerate(positions(5)[3:]):
        c.ask(dict(q, id="cq%d" % i))
      c_dump = c.ask({"id": "d", "action": "cache_dump", "context": ctx, "what": "both",
                      "admission": {"all": True}})
      c_detach = c.ask({"id": "x", "action": "cache_detach", "context": ctx})
      c_rc, _, c_err = c.finish()
    finally:
      c.kill()
    c_contexts = [entry.get("context") for entry in c_stats.get("contexts", [])]
    self.w.check(
      "SC5e a config-attached context is an ORDINARY attached context: cache_stats sees it, "
      "cache_dump writes it, cache_detach takes it",
      c_rc == 0 and ctx in c_contexts and "error" not in c_dump and "error" not in c_detach
      and c_dump.get("evaluations", {}).get("entriesWritten", 0) > 0
      and c_detach.get("sourcesDetached") == 1,
      "cache_stats contexts=%s dump entriesWritten=%s detach sourcesDetached=%s %s %s"
      % (c_contexts, c_dump.get("evaluations", {}).get("entriesWritten"),
         c_detach.get("sourcesDetached"), disposition(c_dump), disposition(c_detach)),
      "the context appears in cache_stats without ever being attached by wire, the wire dump "
      "writes the new keys, and the wire detach takes it",
    )
    self.w.check(
      "SC5f and the shutdown then dumps NOTHING for a context the client detached -- the "
      "lifecycle does not overrule the client",
      not shutdown_dumps(c_err, ctx),
      "shutdown dumps of %s after the wire detach: %s" % (ctx, shutdown_dumps(c_err, ctx)),
      "no shutdown dump line for this context at all",
    )

    # SC5g: the startup refusal. A leaf pointed at a directory that is not there must NOT come
    # up serving an empty cache while the operator believes it is attached.
    missing = os.path.join(self.cache_dir, "definitely-not-a-directory-" + str(os.getpid()))
    bad_cfg = self.lifecycle_config(
      "sharedcache-sc5-baddir.cfg", missing, ["nnCacheAttachContext = " + ctx],
    )
    proc = subprocess.run([self.binary, "analysis", "-config", bad_cfg, "-model", self.net],
                          input="", capture_output=True, text=True, timeout=600)
    combined = (proc.stdout or "") + (proc.stderr or "")
    self.w.check(
      "SC5g an engine whose configured cache directory is not there REFUSES TO START, and says "
      "which key",
      proc.returncode != 0 and "nnCacheDir" in combined and not os.path.exists(missing),
      "rc=%d, message names nnCacheDir=%s, the directory was NOT created=%s; %r"
      % (proc.returncode, "nnCacheDir" in combined, not os.path.exists(missing),
         combined.strip()[-220:]),
      "nonzero exit, the refusal names nnCacheDir, and the engine did not silently create it",
    )
    print()

  # ---------------------------------------------------------------------------------------
  # SC6: the engine is KILLED. Its work is on disk anyway, put there by the periodic dump.
  # ---------------------------------------------------------------------------------------
  def sc6_periodic_dump_survives_a_kill(self):
    """The failure mode the operator says is NORMAL, witnessed rather than assumed.

    SC5 ends its engines by closing stdin, which is the clean exit. These leaves are not usually
    stopped that way: they are long-lived, and they end at machine shutdown or at an arbitrary
    kill when the box wants the memory back. A process that is SIGKILLed runs no shutdown code at
    all - no atexit, no destructor, no dump - so an engine whose only persistence was the
    shutdown dump would lose everything in exactly the case that happens most.

    So this leg never lets engine A exit cleanly. It sends queries, waits for the CONTAINER FILE
    to appear on disk while A is still running - which nothing but the periodic dump could have
    done, since no cache verb was ever sent and A has not exited - and then kills A with SIGKILL.
    Whether the work survived is then read off a SECOND engine, which pays for it or does not.

    The first observation is deliberately of the FILESYSTEM and not of A's log: A's own log line
    is A's opinion of itself, and the claim is about bytes (ADR-0021 Rule 1).
    """
    print("== SC6: engine A is KILLED, never exits cleanly, and its work is on disk anyway ==")
    ctx = self.context + "-sc6"
    queries = positions(3)
    cfg = self.lifecycle_config(
      "sharedcache-sc6.cfg", self.cache_dir,
      ["nnCacheAttachContext = " + ctx,
       "nnCacheDumpMinObservations = 0",
       "nnCacheDumpIntervalMinutes = 0.05"],   # 3 seconds
    )

    def evals_bytes():
      total = 0
      for path in context_files(self.cache_dir, ctx):
        if path.endswith(".nnevals"):
          total += os.path.getsize(path)
      return total

    a = Engine(self.binary, cfg, [self.net])
    grew_at = None
    try:
      for i, q in enumerate(queries):
        a.ask(dict(q, id="q%d" % i))
      # Wait on the PROPERTY with a deadline, never a fixed sleep: a fixed sleep is a leg that is
      # green on an idle box and red on a loaded one and says nothing either way.
      waiting_since = time.time()
      deadline = waiting_since + 120.0
      while time.time() < deadline:
        if evals_bytes() > 0:
          grew_at = time.time()
          break
        time.sleep(0.1)
      alive_when_written = a.proc.poll() is None
      written_bytes = evals_bytes()
      # SIGKILL. Not stdin-close, not terminate: nothing in the engine gets to run after this.
      a.proc.kill()
      _, a_err = a.proc.communicate(timeout=60)
      a_rc = a.proc.returncode
    finally:
      a.kill()

    self.w.check(
      "SC6a *** the container file GREW while engine A was still running, with no cache verb "
      "ever sent -- only the periodic dump can have written it ***",
      written_bytes > 0 and alive_when_written,
      "%s bytes of .nnevals on disk, engine A still alive at that moment=%s, after %s"
      % (written_bytes, alive_when_written,
         ("%.1f s" % (grew_at - waiting_since)) if grew_at else "never"),
      "the evaluations file is non-empty while the writing process is still running",
    )
    self.w.check(
      "SC6b and engine A was KILLED, so no shutdown dump can have contributed anything",
      a_rc is not None and a_rc < 0 and not shutdown_dumps(a_err, ctx),
      "exit status=%s (negative means killed by a signal), shutdown dump lines=%d, "
      "interval dump lines=%d"
      % (a_rc, len(shutdown_dumps(a_err, ctx)),
         len([l for l in a_err.splitlines() if "on the dump interval" in l])),
      "killed by a signal, and NOT one shutdown dump line in its log",
    )

    b_reqs = [dict(q, id="q%d" % i) for i, q in enumerate(queries)]
    _, b_rows, b_rc = self.session(b_reqs, config=cfg)
    self.w.check(
      "SC6c *** a later engine pays ZERO neural net rows for the killed engine's queries ***",
      b_rc == 0 and b_rows == 0,
      "rc=%d NN rows=%d over the same %d queries the killed engine had paid for"
      % (b_rc, b_rows, len(queries)),
      "NN rows == 0 exactly -- the killed process's work was already on disk",
    )
    print()

  # ---------------------------------------------------------------------------------------
  # SC3: a reader attaching WHILE a writer dumps. State-based, never timing-based.
  # ---------------------------------------------------------------------------------------
  def sc3_attach_during_dump(self):
    print("== SC3: B attaches over and over WHILE A dumps -- every observation is whole ==")
    ctx = self.context + "-sc3"

    # Round one establishes the PRE state: a context with real content already in it, so that
    # the dump SC3 overlaps is an append to an existing file rather than a first write.
    pre_reqs = [{"id": "a", "action": "cache_attach", "context": ctx}]
    pre_reqs += [dict(q, id="q%d" % i) for i, q in enumerate(positions(4))]
    pre_reqs += [{"id": "d", "action": "cache_dump", "context": ctx, "what": "both", "admission": {"all": True}},
                 {"id": "x", "action": "cache_detach", "context": ctx}]
    pre_out, _, pre_rc = self.session(pre_reqs)
    pre_written = pre_out["d"].get("evaluations", {}).get("entriesWritten")
    self.w.check(
      "SC3a the context holds real content before the overlapped dump",
      pre_rc == 0 and pre_written > 0,
      "rc=%d entriesWritten=%s %s" % (pre_rc, pre_written, disposition(pre_out["d"])),
      "rc==0 and entriesWritten > 0",
    )

    samples = []
    sampler_error = []
    stop = threading.Event()

    def sample():
      """B, attaching and reading and detaching, for as long as A is working.

      Every sample is a WHOLE-STATE observation: entriesInLevelZero plus both tails, read
      through the engine's own reader. A partial state would show up here as a count between
      the two whole ones, or as a tail reported truncated, or as a refusal -- and all three
      are recorded rather than tolerated.
      """
      b = Engine(self.binary, self.config, [self.net])
      try:
        n = 0
        while not stop.is_set():
          n += 1
          r = b.ask({"id": "s%d" % n, "action": "cache_attach", "context": ctx})
          samples.append({
            "entries": r.get("entriesInLevelZero"),
            "containerTail": r.get("containerTail"),
            "countLogTail": r.get("countLogTail"),
            "error": r.get("error"),
            "at": time.time(),
          })
          b.ask({"id": "d%d" % n, "action": "cache_detach", "context": ctx})
          time.sleep(SAMPLE_GAP_SECONDS)
        b.finish()
      except Exception as e:  # noqa: BLE001 -- the thread's failure must reach the report
        sampler_error.append("%s: %s" % (type(e).__name__, e))
      finally:
        b.kill()

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    # Let the sampler take at least one reading of the PRE state before A starts.
    time.sleep(0.5)

    a = Engine(self.binary, self.config, [self.net])
    dump_started = dump_finished = None
    try:
      a.ask({"id": "a", "action": "cache_attach", "context": ctx})
      # Positions 4..9: six keys the PRE dump above (which held 0..3) never saw.
      for i, q in enumerate(positions(10)[4:]):
        a.ask(dict(q, id="q%d" % i))
      dump_started = time.time()
      a_dump = a.ask({"id": "d", "action": "cache_dump", "context": ctx, "what": "both", "admission": {"all": True}})
      dump_finished = time.time()
      a.ask({"id": "x", "action": "cache_detach", "context": ctx})
      a_rc, _, _ = a.finish()
    finally:
      a.kill()

    # Keep sampling past the dump, so the POST state is observed by the same sampler that
    # observed the PRE state, in the same process, through the same reader.
    time.sleep(0.5)
    stop.set()
    thread.join(timeout=120)

    added = a_dump.get("evaluations", {}).get("entriesWritten")
    post = pre_written + (added or 0)

    self.w.check(
      "SC3b the overlapped dump really CHANGED the state, so the check below is not vacuous",
      a_rc == 0 and added is not None and added > 0,
      "the overlapped dump wrote %s more entries (%s -> %s); dump took %.0f ms; %s"
      % (added, pre_written, post,
         (dump_finished - dump_started) * 1000.0 if dump_started else -1.0,
         disposition(a_dump)),
      "entriesWritten > 0 -- PRE and POST are different states",
    )
    self.w.check(
      "SC3c the sampler ran without an error of its own",
      not sampler_error and thread.is_alive() is False and len(samples) >= 2,
      "%d samples, %d thread errors %s" % (len(samples), len(sampler_error), sampler_error or ""),
      "at least 2 samples and no thread error -- otherwise the leg below saw nothing",
    )

    counts = [s["entries"] for s in samples]
    refused = [s for s in samples if s["error"]]
    torn = [s for s in samples if s["containerTail"] != "intact" or s["countLogTail"] != "intact"]
    other = sorted({c for c in counts if c not in (pre_written, post)})
    self.w.check(
      "SC3d *** EVERY attach overlapping the dump saw the WHOLE pre-state or the WHOLE "
      "post-state, never anything between ***",
      not refused and not torn and not other,
      "%d samples: %d saw pre(%s), %d saw post(%s), %d saw something else %s; "
      "%d refused, %d reported a torn tail"
      % (len(counts), counts.count(pre_written), pre_written, counts.count(post), post,
         len(other), other, len(refused), len(torn)),
      "every sample in {%s, %s}, no refusal, every tail intact" % (pre_written, post),
    )
    self.w.check(
      "SC3e and the sampling really did STRADDLE the dump -- both states were observed",
      counts.count(pre_written) > 0 and counts.count(post) > 0,
      "pre(%s) seen %d times, post(%s) seen %d times, over %.1f s of sampling"
      % (pre_written, counts.count(pre_written), post, counts.count(post),
         (samples[-1]["at"] - samples[0]["at"]) if len(samples) >= 2 else 0.0),
      "both the pre-state and the post-state observed by the same sampler -- without this, "
      "SC3d could pass over a sampler that never overlapped the dump at all",
    )
    print()


def main():
  p = argparse.ArgumentParser(
    description="Cross-process shared-NN-cache-directory end-to-end witness.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="The cache directory is the one thing this suite is parameterised on: point it at "
           "a local directory to establish a baseline, then at the shared mount, and compare "
           "the two runs. It must already exist -- KataGo does not create it.",
  )
  p.add_argument("--cache-dir", default=os.environ.get("KATAGO_SHARED_CACHE_DIR"),
                 help="The shared cache directory under test. Defaults to $KATAGO_SHARED_CACHE_DIR.")
  p.add_argument("--binary", default=DEFAULT_BINARY, help="The katago binary. Default %s" % DEFAULT_BINARY)
  p.add_argument("--config", default=DEFAULT_TEMPLATE,
                 help="Analysis config template holding @CACHE_DIR@ and @USE_EVAL_CACHE@.")
  p.add_argument("--model", default=DEFAULT_NET, help="Model file. Default %s" % DEFAULT_NET)
  p.add_argument("--workdir", default=None,
                 help="Where the generated config is written. Default: a fresh temp directory. "
                      "Never the cache directory, which holds cache files and nothing else.")
  p.add_argument("--context", default=None,
                 help="Context name prefix. Default: unique per run, so repeated runs against a "
                      "persistent directory never collide.")
  p.add_argument("--skip-probe", action="store_true",
                 help="Do not run lockfsprobe first. Only for a directory already probed.")
  p.add_argument("--keep", action="store_true", help="Leave this run's cache files behind.")
  p.add_argument("--only", action="append", default=None,
                 choices=["sc1", "sc2", "sc3", "sc4", "sc5", "sc6"],
                 help="Run only these scenarios. Repeatable.")
  args = p.parse_args()

  if not args.cache_dir:
    p.error("no cache directory: pass --cache-dir or set $KATAGO_SHARED_CACHE_DIR")
  cache_dir = os.path.abspath(os.path.expanduser(args.cache_dir))
  binary = os.path.abspath(os.path.expanduser(args.binary))

  print("shared-cache end-to-end witness")
  print("  cache directory: %s" % cache_dir)
  print("  binary:          %s" % binary)
  print("  model:           %s" % args.model)
  print()

  for what, path in (("katago binary", binary), ("config template", args.config),
                     ("model", args.model)):
    if not os.path.exists(path):
      print("REFUSED: the %s %s is not there." % (what, path))
      return 2
  if not os.path.isdir(cache_dir):
    print("REFUSED: the cache directory %s is not there. KataGo does not create it; make it "
          "(or mount it) first." % cache_dir)
    return 2

  if args.skip_probe:
    print("  [SKIPPED] lockfsprobe, by --skip-probe. Nothing has established that locking "
          "excludes on this directory.")
    print()
  else:
    rc, output = probe_lock(binary, cache_dir)
    print(output.rstrip())
    print()
    if rc != 0:
      print("REFUSED: locking does not exclude on %s (lockfsprobe exit %d)." % (cache_dir, rc))
      print("This suite will NOT run here. Two engine processes sharing a directory whose")
      print("filesystem cannot lock would each believe they held the exclusive lock; their")
      print("appends would interleave mid-file, and the reader would discard every block after")
      print("the first torn one. No engine was started and NOTHING was written to that")
      print("directory. Point --cache-dir at a filesystem that locks, or run the two leaves")
      print("against separate directories.")
      return 2

  workdir = args.workdir
  if workdir is None:
    workdir = tempfile.mkdtemp(prefix="sharedcache-e2e-")
  else:
    workdir = os.path.abspath(os.path.expanduser(workdir))
    os.makedirs(workdir, exist_ok=True)
  config = write_config(args.config, cache_dir, os.path.join(workdir, "sharedcache.cfg"))
  context = args.context or ("sc-%d-%d" % (int(time.time()), os.getpid()))
  print("  config:          %s" % config)
  print("  context prefix:  %s" % context)
  print()

  w = Witness()
  suite = Suite(w, binary, config, args.model, cache_dir, context, workdir)
  chosen = args.only or ["sc1", "sc2", "sc3", "sc4", "sc5", "sc6"]
  try:
    if "sc1" in chosen:
      suite.sc1_sequential_reuse()
    if "sc2" in chosen:
      suite.sc2_two_live_engines()
    if "sc3" in chosen:
      suite.sc3_attach_during_dump()
    if "sc4" in chosen:
      suite.sc4_default_admission_bootstrap()
    if "sc5" in chosen:
      suite.sc5_config_driven_lifecycle()
    if "sc6" in chosen:
      suite.sc6_periodic_dump_survives_a_kill()
  finally:
    if args.keep:
      print("kept: %s" % [os.path.basename(f) for f in context_files(cache_dir, context + "-sc1")
                          + context_files(cache_dir, context + "-sc2")
                          + context_files(cache_dir, context + "-sc3")
                          + context_files(cache_dir, context + "-sc4")
                          + context_files(cache_dir, context + "-sc5")
                          + context_files(cache_dir, context + "-sc6")])
    else:
      for suffix in ("-sc1", "-sc2", "-sc3", "-sc4", "-sc5", "-sc6"):
        for f in context_files(cache_dir, context + suffix):
          try:
            os.remove(f)
          except OSError as e:
            print("could not remove %s: %s" % (f, e))

  print("%d legs, %d failed" % (w.legs, len(w.failures)))
  if w.failures:
    print("FAILED: %s" % ", ".join(w.failures))
    return 1
  print("PASS -- every scenario green against %s" % cache_dir)
  return 0


if __name__ == "__main__":
  sys.exit(main())
