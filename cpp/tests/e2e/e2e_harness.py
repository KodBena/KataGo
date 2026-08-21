"""What a leg of the persistence witness is RUN with, and READ through.

Split out of persistence_e2e_witness.py under ADR-0007: the witness's legs are its content,
and this is the machinery under them, so they are separate files with one concern each
rather than one file a reader has to page through to find either (ADR-0012 P3).

NOTHING HERE ASSERTS ANYTHING ABOUT THE ENGINE. `Witness` only records and prints what a leg
decided; `fingerprint`, `search_shape`, `digest_dir` and `observation_profile` only say HOW an
observation is taken. The legs live in persistence_legs.py and keying_legs.py, and are the
only place a property is claimed.
"""

import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis_engine_driver import Engine, nn_rows_for, run_engine, responses  # noqa: E402

DEFAULT_NET_A = "/home/bork/kg/14.gz"
DEFAULT_NET_B = "/home/bork/kg/18.gz"

# One 7x7 position, 20 visits. Small enough that a leg costs seconds on a CPU build, big
# enough that "zero rows" is a statement about twenty evaluated positions and not about one.
POSITION = {
  "boardXSize": 7,
  "boardYSize": 7,
  "rules": "tromp-taylor",
  "komi": 7.5,
  "initialPlayer": "b",
  "moves": [["b", "d4"], ["w", "e5"], ["b", "c5"]],
  "analyzeTurns": [3],
  "maxVisits": 20,
  "includeOwnership": True,
}

# The count log's per-dump framing: one block header and no records. Named here as the
# constant P3e's expectation is denominated in, and cross-checked against the format's own
# statement of it (nncachecountlog.h: "plus 32 bytes of framing for the whole dump").
COUNT_LOG_BLOCK_HEADER_BYTES = 32

# The engine's own words when the openRequests race bites. Matching the operator-facing
# refusal text is legitimate under ADR-0021 Rule 3 -- it is the contract a client reads, from
# one home (cacheSwapConcurrencyRefusal) -- and it is used only to LABEL a failure, never to
# excuse one.
RACE_MARK = "request(s) are open"


class Witness:
  def __init__(self):
    self.failures = []
    self.legs = 0

  def check(self, name, ok, observed, expected):
    self.legs += 1
    print("%-4s  %s" % ("PASS" if ok else "FAIL", name))
    print("        observed: %s" % (observed,))
    print("        expected: %s" % (expected,))
    if not ok:
      self.failures.append(name)
    return ok


def disposition(response):
  """A refusal's text, labelled KNOWN-RACE when it is the openRequests race."""
  err = response.get("error", "")
  if RACE_MARK in err:
    return "KNOWN-RACE (analysis.cpp openRequests; the fix is not on this branch): " + err
  return err or "no error"


def digest_dir(path):
  """{filename: (size, sha256)} for every file directly under `path`."""
  out = {}
  for name in sorted(os.listdir(path)):
    full = os.path.join(path, name)
    if os.path.isfile(full):
      with open(full, "rb") as f:
        data = f.read()
      out[name] = (len(data), hashlib.sha256(data).hexdigest()[:16])
  return out


def expected_files_for_context(context, model_names):
  """The COMPLETE set of files a context's on-disk footprint is entitled to hold, for a leg
  that asserts an EXACT set against digest_dir().

  ONE HOME for this set-shape, because a leg that hand-duplicates a literal set of filenames
  drifts from the format silently -- exactly what happened to P1d (audit-reports/
  p1d-diagnosis.md): the lock file (nncachefileformat.h's NNCacheFileLock) was added to the
  C++ format two days after the literal was written, and nothing forced the two to move
  together. `nncachefileformat.h`'s own comment blocks (the count-log/container pair,
  `<context>.nncounts` and `<context>.<model>.nnevals`; and the lock, `<context>.nnlock`,
  "on a file of its own... never renamed, never truncated, never written") are the C++
  authority this function mirrors. A real codegen bridge from that header into Python was
  judged not worth building for three filenames with no independent parameters beyond
  `context` and a model name already in hand at every call site -- that is a judgment call,
  not a silent shortcut, and the reason it is safe is that this is the ONLY place in the e2e
  suite duplicating the shape (grep for `set(on_disk)` / exact digest_dir comparisons before
  assuming a second one is fine to add by hand).

  `model_names` is every model whose container has been dumped for `context` in the session
  under test -- most legs pass a single name, but the function takes an iterable so a leg
  that dumps under two models is representable without a second literal.
  """
  if isinstance(model_names, str):
    raise TypeError("model_names is an iterable of names, not a single string")
  names = set(model_names)
  return {"%s.nncounts" % context, "%s.nnlock" % context} | {
    "%s.%s.nnevals" % (context, name) for name in names
  }


def fingerprint(answer):
  """The numbers that identify WHICH NET EVALUATED the root -- the net-identity claim.

  THE ROOT'S RAW EVALUATION, and deliberately not its SEARCHED winrate. The claim every
  cross-model leg makes is about which net's EVALUATION came back, and the raw root values
  are that evaluation; the searched winrate is downstream of a 20-visit tree whose shape a
  near-tied pair of moves can change (ADR-0021 Rule 1: observe at the site of the claim, not
  downstream of it, where every confounder in between is a way for the leg to lie). That is
  not a supposition either: before this witness set forDeterministicTesting, 18.gz's searched
  winrate flipped between two values across five runs while these three raw numbers never
  moved once.

  Three numbers rather than one, because a single scalar could collide between two nets by
  accident. Taken from the response's own JSON values, so no float is re-parsed through a
  second representation.
  """
  root = answer["rootInfo"]
  return (root["rawWinrate"], root["rawScoreSelfplay"], root["rawLead"])


def search_shape(answer):
  """The whole search: the raw root evaluation, the searched values, and the visit spread.

  What the EVAL CACHE moves, and what a leg comparing two searches must read. Strictly finer
  than fingerprint(): two answers with the same fingerprint can have different shapes, which
  is exactly what the eval cache does to a second search of the same position.
  """
  root = answer["rootInfo"]
  return (fingerprint(answer), root["winrate"], root["scoreLead"],
          tuple((m["move"], m["visits"]) for m in answer["moveInfos"]))


class Harness:
  """The workdir, the config template, and the two nets: everything a leg needs to run one."""

  def __init__(self, binary, template, workdir, net_a, net_b):
    self.binary, self.template, self.workdir = binary, template, workdir
    self.net_a, self.net_b = net_a, net_b

  def config(self, name, eval_cache=False):
    """A fresh cache directory and a config pointing at it. Returns (dir, config path).

    The eval-cache setting is SUBSTITUTED into the template's one statement of the key, not
    appended: the engine refuses a config that states a key twice, by name, and a second
    statement of a setting is a second home for it (ADR-0012 P1).
    """
    d = os.path.join(self.workdir, name)
    os.makedirs(d)
    with open(self.template) as f:
      text = f.read()
    for placeholder, value in (("@CACHE_DIR@", d),
                               ("@USE_EVAL_CACHE@", "true" if eval_cache else "false")):
      if placeholder not in text:
        raise RuntimeError("the config template %s holds no %s" % (self.template, placeholder))
      text = text.replace(placeholder, value)
    path = os.path.join(self.workdir, name + ".cfg")
    with open(path, "w") as f:
      f.write(text)
    return d, path

  def session(self, config, nets, requests):
    """One engine, one request at a time, waiting for each response. Returns (responses, rows).

    `responses` is a dict keyed by each request's id; `rows` is {net path: neural net rows
    that net evaluated}, from the engine's own end-of-run log.
    """
    e = Engine(self.binary, config, nets)
    out = {}
    try:
      for req in requests:
        out[req["id"]] = e.ask(req)
      rc, _, err = e.finish()
    finally:
      e.kill()
    rows = {net: nn_rows_for(err, net) for net in nets}
    out["__rc__"] = rc
    return out, rows

  def model_names(self):
    """Each net's internalName, read from the engine rather than guessed from the file name."""
    d, cfg = self.config("names")
    rc, out, err = run_engine(self.binary, cfg, [self.net_a, self.net_b],
                              [{"id": "m", "action": "query_models"}])
    if rc != 0:
      raise RuntimeError("query_models exited %d; stderr tail:\n%s" % (rc, err[-2000:]))
    return [m["internalName"] for m in responses(out)["m"][0]["models"]]


def context_of(stats, context):
  """One context's block of a cache_stats response, or {} when it is not attached."""
  for c in stats.get("contexts", []):
    if c.get("context") == context:
      return c
  return {}


def observation_profile(h, config, net, context, thresholds=(0, 1, 2, 3, 4)):
  """{minObservations t: keys a level-0 attach admits at t}, each measured in its own process.

  This is the count log's per-key CONTENT, read through the engine's own count-log reader and
  its own level-0 selector -- not through a second hand-written decoder of the file, which
  would be a second author of the format (ADR-0012 P7). Two profiles taken around an act are
  equal exactly when that act moved no key's recorded observations.
  """
  profile = {}
  for t in thresholds:
    out, _ = h.session(config, [net], [
      {"id": "a", "action": "cache_attach", "context": context, "level0": {"minObservations": t}}
    ])
    r = out["a"]
    if "error" in r:
      raise RuntimeError("observation profile attach at minObservations=%d refused: %s" % (t, r["error"]))
    profile[t] = r.get("entriesInLevelZero")
  return profile
