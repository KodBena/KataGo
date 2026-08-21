"""Half two's legs: the model keying (K), the eval-cache model salt (K5), and the proof
that the two are independent (K6).

Read persistence_e2e_witness.py first: it states what these legs observe, where, and why the
observation is not taken from the response field that would be easiest to read. This file is
only the legs; e2e_harness.py is the machinery under them.

THE ONE THING TO CARRY INTO THIS FILE. Three different mechanisms could each, on its own,
produce "one model is not handed another's cached work", and a leg that cannot say which one
it exercised has witnessed a coincidence:

  the CONTAINER PATH   <context>.<model>.nnevals -- the other model's file simply is not
                       there (K2 exercises this, and only this)
  the CONTAINER HEADER the model internalName in the file header, refused by name (K3
                       exercises this, and ONLY this, because the leg first puts the bytes at
                       exactly the path the other model reads, so the path cannot be the
                       reason)
  the EVAL-CACHE SALT  Search::evalCacheModelHash (K5 exercises this, and K5c shows it is
                       INVISIBLE on the neural-net row counter -- which is why it is not a
                       candidate explanation for K2 or K3 at all)
"""

import json
import os
import shutil

from e2e_harness import POSITION, digest_dir, disposition, fingerprint, search_shape


def _keying_pass(w, h, tag, name_a, name_b, fp_a, fp_b, eval_cache, positive_and_capability):
  """K0-K4 (or, for K6, only the two negative legs) under one eval-cache setting."""
  net_a, net_b = h.net_a, h.net_b
  short_a, short_b = os.path.basename(net_a), os.path.basename(net_b)
  k_dir, k_cfg = h.config("k_ec" if eval_cache else "k", eval_cache)
  seed, _ = h.session(k_cfg, [net_a], [
    {"id": "a", "action": "cache_attach", "context": "card-key"},
    dict(POSITION, id="q", model=name_a),
    {"id": "d", "action": "cache_dump", "context": "card-key", "what": "both", "admission": {"all": True}},
  ])
  seeded = seed["d"].get("evaluations", {}).get("entriesWritten")
  if positive_and_capability:
    w.check(
      "K0  the seed session persisted %s of %s's evaluations" % (seeded, short_a),
      bool(seeded) and seeded > 0 and fingerprint(seed["q"]) == fp_a,
      "entriesWritten=%s fingerprint=%s %s"
      % (seeded, fingerprint(seed["q"]), disposition(seed["d"])),
      "entriesWritten>0 and the seed's answer is %s's fingerprint %s" % (short_a, fp_a),
    )

  def leg(attach_request, model):
    out, rows = h.session(k_cfg, [net_a, net_b], [
      attach_request,
      {"id": "s", "action": "cache_stats", "model": model},
      dict(POSITION, id="q", model=model),
    ])
    return out["a"], out["s"], out["q"], rows

  if positive_and_capability:
    attach, stats, ans, rows = leg(
      {"id": "a", "action": "cache_attach", "context": "card-key", "model": name_a}, name_a)
    w.check(
      "K1a %s re-attaches its OWN container in the two-net engine: ZERO rows" % short_a,
      attach.get("entriesInLevelZero") == seeded and rows[net_a] == 0 and rows[net_b] == 0,
      "entriesInLevelZero=%s rows(%s)=%d rows(%s)=%d %s"
      % (attach.get("entriesInLevelZero"), short_a, rows[net_a], short_b, rows[net_b],
         disposition(attach)),
      "entriesInLevelZero==%s, rows(%s)==0, rows(%s)==0" % (seeded, short_a, short_b),
    )
    w.check(
      "K1b and the answer is %s's own fingerprint" % short_a,
      fingerprint(ans) == fp_a,
      "returned=%s" % (fingerprint(ans),),
      "%s" % (fp_a,),
    )

  # K2 -- the default negative leg.
  attach, stats, ans, rows = leg(
    {"id": "a", "action": "cache_attach", "context": "card-key", "model": name_b}, name_b)
  w.check(
    "%s2a %s attaching the SAME context gets NOTHING" % (tag, short_b),
    "error" not in attach and attach.get("entriesInLevelZero") == 0,
    "entriesInLevelZero=%s; the directory holds %s %s"
    % (attach.get("entriesInLevelZero"), sorted(digest_dir(k_dir)), disposition(attach)),
    "entriesInLevelZero==0. MECHANISM: a container's path is <context>.<model>.nnevals and "
    "only %s's exists" % short_a,
  )
  w.check(
    "%s2b *** and %s pays a REAL evaluation for a position %s's container holds ***"
    % (tag, short_b, short_a),
    rows[net_b] > 0 and rows[net_a] == 0,
    "rows(%s)=%d rows(%s)=%d" % (short_b, rows[net_b], short_a, rows[net_a]),
    "rows(%s)>0 (real work) and rows(%s)==0 (the other net did not do it either)"
    % (short_b, short_a),
  )
  w.check(
    "%s2c and the number returned is %s's own, not %s's" % (tag, short_b, short_a),
    fingerprint(ans) == fp_b and fingerprint(ans) != fp_a,
    "returned=%s   %s=%s   %s=%s" % (fingerprint(ans), short_b, fp_b, short_a, fp_a),
    "the returned triple is %s's fingerprint and is not %s's" % (short_b, short_a),
  )

  # K3 -- THE DISTINGUISHER. The first net's container bytes at exactly the path the second
  # reads, so the PATH cannot be what refuses it. Only the model name in the FILE HEADER can.
  src = os.path.join(k_dir, "card-key.%s.nnevals" % name_a)
  dst = os.path.join(k_dir, "card-key.%s.nnevals" % name_b)
  # The setup itself is a claim -- that a container lives at a per-model path -- so it is
  # checked rather than assumed. A build whose paths carried no model name would otherwise
  # make this leg die in the harness, where a stack trace says nothing about the property.
  if not w.check(
    "%s3  SETUP: %s's container is at its own per-model path" % (tag, short_a),
    os.path.exists(src),
    "%s %s" % (os.path.basename(src), "exists" if os.path.exists(src) else "IS ABSENT"),
    "the file exists; the K3 legs below copy it to %s and cannot run without it"
    % os.path.basename(dst),
  ):
    return
  shutil.copyfile(src, dst)
  try:
    attach, stats, ans, rows = leg(
      {"id": "a", "action": "cache_attach", "context": "card-key", "model": name_b}, name_b)
    err = attach.get("error", "")
    w.check(
      "%s3a with %s's BYTES AT %s's OWN PATH, the attach is REFUSED, naming both models"
      % (tag, short_a, short_b),
      "error" in attach and name_a in err and name_b in err and "never read as another's" in err,
      (err or json.dumps(attach))[:420],
      "an error naming '%s' (what the file holds) and '%s' (what it was opened for). "
      "MECHANISM: the model internalName in the container's FILE HEADER. Path keying CANNOT "
      "explain this leg -- the file is exactly where %s looks." % (name_a, name_b, short_b),
    )
    w.check(
      "%s3b and nothing was attached" % tag,
      stats.get("levelZeroSourcesAttached") == 0 and stats.get("contexts") == [],
      "levelZeroSourcesAttached=%s contexts=%s"
      % (stats.get("levelZeroSourcesAttached"), stats.get("contexts")),
      "levelZeroSourcesAttached==0 and no context attached",
    )
    w.check(
      "%s3c *** and %s still pays a REAL evaluation, and returns its own number ***"
      % (tag, short_b),
      rows[net_b] > 0 and rows[net_a] == 0 and fingerprint(ans) == fp_b,
      "rows(%s)=%d rows(%s)=%d returned=%s"
      % (short_b, rows[net_b], short_a, rows[net_a], fingerprint(ans)),
      "rows(%s)>0, rows(%s)==0, and the returned triple is %s's %s"
      % (short_b, short_a, short_b, fp_b),
    )
  finally:
    os.remove(dst)

  if positive_and_capability:
    # K4 -- the capability control. The machinery CAN serve across models when the client
    # names the foreign source, so K2 and K3 are a decision and not an inability.
    attach, stats, ans, rows = leg(
      {"id": "a", "action": "cache_attach", "context": "card-key", "model": name_b,
       "foreignModelSources": [name_a]}, name_b)
    sources = attach.get("sources", [])
    w.check(
      "K4a with foreignModelSources naming %s, %s IS served %s's container"
      % (short_a, short_b, short_a),
      "error" not in attach and len(sources) == 2
      and sources[0].get("model") == name_b and sources[0].get("entriesInLevelZero") == 0
      and sources[1].get("model") == name_a and sources[1].get("entriesInLevelZero") == seeded,
      json.dumps(sources) + " " + disposition(attach),
      "two sources in resolution order: %s's own with 0 entries, then %s's with %s"
      % (short_b, short_a, seeded),
    )
    w.check(
      "K4b *** and NOW %s pays ZERO rows -- so K2/K3 are the keying's DECISION, not the "
      "machinery's inability ***" % short_b,
      rows[net_b] == 0 and rows[net_a] == 0,
      "rows(%s)=%d rows(%s)=%d" % (short_b, rows[net_b], short_a, rows[net_a]),
      "rows(%s)==0 and rows(%s)==0" % (short_b, short_a),
    )
    w.check(
      "K4c and the number returned is %s's, so the bytes served really are %s's"
      % (short_a, short_a),
      fingerprint(ans) == fp_a,
      "returned=%s   %s=%s   %s=%s" % (fingerprint(ans), short_a, fp_a, short_b, fp_b),
      "the returned triple is %s's fingerprint %s" % (short_a, fp_a),
    )


def run_keying(w, h, name_a, name_b, fp_a, fp_b):
  """Group K: the container keying, with the eval cache off."""
  print("== K: the model keying, useEvalCache = false (the config's own default) ==")
  _keying_pass(w, h, "K", name_a, name_b, fp_a, fp_b, False, True)
  print()


def run_eval_cache_salt(w, h, name_a, name_b, fp_a, fp_b, own_cost):
  """Group K5: the eval-cache model salt, on the surface the eval cache actually caches."""
  net_a, net_b = h.net_a, h.net_b
  short_a, short_b = os.path.basename(net_a), os.path.basename(net_b)
  print("== K5: the eval-cache model salt -- a SEARCH RESULT, not an evaluation ==")
  print("        The eval cache does NOT hold NNOutputs and an eval-cache hit does NOT skip")
  print("        a neural net evaluation: an entry is a node's average utilities plus")
  print("        first-explore estimates, and the exploring code reads the node's policy")
  print("        anyway (searchexplorehelpers.cpp). So the salt is INVISIBLE on the row")
  print("        counter -- which is why these legs are read off the SEARCH RESULT, and why")
  print("        the salt cannot be an alternative explanation for K2/K3, which are read")
  print("        off the row counter.")
  _, k5_cfg = h.config("k5", eval_cache=True)
  # Each net's answer with NOTHING else in the process: the baseline the contamination legs
  # below are measured against.
  alone = {}
  for label, model in (("A", name_a), ("B", name_b)):
    out, _ = h.session(k5_cfg, [net_a, net_b], [dict(POSITION, id="q", model=model)])
    alone[label] = out["q"]
  # And now one process where A answers first, twice, and then B answers.
  shared, rows = h.session(k5_cfg, [net_a, net_b], [
    dict(POSITION, id="q1", model=name_a),
    dict(POSITION, id="q2", model=name_a),
    dict(POSITION, id="q3", model=name_b),
  ])

  # THE CONTROL, FIRST, because without it every identity below is the identity of a cache
  # that never held anything -- which is exactly the state this witness's first draft of this
  # group was in, and a seen-red pass that deleted the model salt outright could not make
  # that draft go red.
  w.check(
    "K5a control -- the eval cache IS LIVE: %s's SECOND search of the same position differs "
    "from its first" % short_a,
    search_shape(shared["q1"]) != search_shape(shared["q2"])
    and search_shape(shared["q1"]) == search_shape(alone["A"]),
    "first=%s   second=%s   (and the first matches %s alone: %s)"
    % (search_shape(shared["q1"]), search_shape(shared["q2"]), short_a,
       search_shape(shared["q1"]) == search_shape(alone["A"])),
    "the two searches DIFFER -- the second was steered by the entry the first left in the "
    "eval cache -- and the first equals the same net alone. Without this, K5b asserts nothing "
    "and its red is unavailable",
  )
  w.check(
    "K5b *** %s's search is UNAFFECTED by %s having just searched the same position ***"
    % (short_b, short_a),
    search_shape(shared["q3"]) == search_shape(alone["B"]),
    "after %s ran: %s\n                  %s alone:   %s"
    % (short_a, search_shape(shared["q3"]), short_b, search_shape(alone["B"])),
    "identical to %s's own search with nothing else in the process. MECHANISM: "
    "Search::evalCacheModelHash, and only that -- this leg configures a cache directory but "
    "never attaches, dumps or reads anything from it, and the only structure the two searches "
    "share is the process's one EvalCacheTable." % short_b,
  )
  w.check(
    "K5c and the eval cache never spared %s a single neural net row" % short_b,
    rows[net_b] == own_cost[net_b],
    "rows(%s)=%d in the shared process; %s alone costs %d"
    % (short_b, rows[net_b], short_b, own_cost[net_b]),
    "rows(%s)==%d -- the full price. This is the leg that says the salt is NOT a candidate "
    "explanation for K2 and K3: an eval-cache hit does not avoid an evaluation, so nothing "
    "the eval cache does can move the counter K2 and K3 are read from"
    % (short_b, own_cost[net_b]),
  )
  print()


def run_salt_independence(w, h, name_a, name_b, fp_a, fp_b):
  """Group K6: the container legs again with the eval cache on and actually populated."""
  print("== K6: K2 and K3 repeated with useEvalCache = true ==")
  _keying_pass(w, h, "K6/K", name_a, name_b, fp_a, fp_b, True, False)
  print("        note:     K6's K2/K3 legs observe exactly what K's did with the eval cache")
  print("                  off. The container keying does not depend on that setting, so the")
  print("                  eval-cache model salt is not what produces K2 and K3.")
  print()
