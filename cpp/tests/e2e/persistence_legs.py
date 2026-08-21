"""Group S (the fingerprints) and group P (half one: persistence).

Read persistence_e2e_witness.py first: it states what these legs observe, where, and why the
observation is not taken from the response field that would be easiest to read. This file is
only the legs; e2e_harness.py is the machinery under them.
"""

import json
import os

from e2e_harness import (
  COUNT_LOG_BLOCK_HEADER_BYTES, POSITION, context_of, digest_dir, disposition, fingerprint,
  observation_profile, search_shape,
)


def run_fingerprints(w, h, name_a, name_b):
  """Group S. Returns ({net: fingerprint}, {net: its own neural-net cost for one query}).

  ALWAYS RUN, whatever --only selects: every group reads these, and a group that read them
  without S having proved they discriminate would be asserting nothing.
  """
  net_a, net_b = h.net_a, h.net_b
  short_a, short_b = os.path.basename(net_a), os.path.basename(net_b)
  print("== S: each net's own answer, on an engine that touches no cache directory ==")
  fps = {}
  own_cost = {}
  for label, net, model in (("S1", net_a, name_a), ("S2", net_b, name_b)):
    _, cfg = h.config("s_" + label)
    out, rows = h.session(cfg, [net_a, net_b], [dict(POSITION, id="q", model=model)])
    ans = out["q"]
    other = net_b if net == net_a else net_a
    fps[net] = fingerprint(ans)
    own_cost[net] = rows[net]
    w.check(
      "%s  %s answers the position, and ONLY %s did the work"
      % (label, os.path.basename(net), os.path.basename(net)),
      rows[net] > 0 and rows[other] == 0 and ans["rootInfo"]["visits"] == POSITION["maxVisits"],
      "rows(%s)=%d rows(%s)=%d visits=%s fingerprint=%s"
      % (os.path.basename(net), rows[net], os.path.basename(other), rows[other],
         ans["rootInfo"]["visits"], fps[net]),
      "rows(this)>0, rows(the other net)==0, visits==%d" % POSITION["maxVisits"],
    )
  w.check(
    "S3  the two nets' answers DIFFER, so the fingerprint discriminates",
    fps[net_a] != fps[net_b],
    "%s=%s   %s=%s" % (short_a, fps[net_a], short_b, fps[net_b]),
    "two different triples -- every leg below that reads a fingerprint is void without this",
  )
  print()
  return fps, own_cost


def run_persistence(w, h, name_a):
  """Group P. attach -> analyze -> dump -> detach, re-attach with ZERO evaluations, and the
  no-query attach/dump/detach/re-attach/dump cycle."""
  net_a = h.net_a
  print("== P1: cold -- attach an empty context, analyze, dump, detach ==")
  p_dir, p_cfg = h.config("p")
  out, rows = h.session(p_cfg, [net_a], [
    {"id": "a", "action": "cache_attach", "context": "card-5455"},
    dict(POSITION, id="q"),
    {"id": "d", "action": "cache_dump", "context": "card-5455", "what": "both", "admission": {"all": True}},
    {"id": "x", "action": "cache_detach", "context": "card-5455"},
  ])
  attach, cold, dump, detach = out["a"], out["q"], out["d"], out["x"]
  cold_rows = rows[net_a]
  written = dump.get("evaluations", {}).get("entriesWritten")
  w.check(
    "P1a attaching a context nothing ever wrote loads an EMPTY level 0",
    attach.get("entriesInLevelZero") == 0 and attach.get("containerTail") == "intact",
    "entriesInLevelZero=%s containerTail=%s %s"
    % (attach.get("entriesInLevelZero"), attach.get("containerTail"), disposition(attach)),
    "entriesInLevelZero==0 and containerTail=='intact'",
  )
  w.check(
    "P1b the cold session evaluates real positions and dumps every one of them",
    cold_rows > 0 and written == cold_rows,
    "NN rows=%d entriesWritten=%s countLogBytesAppended=%s %s"
    % (cold_rows, written, dump.get("counts", {}).get("bytesAppended"), disposition(dump)),
    "NN rows > 0 and entriesWritten == NN rows",
  )
  w.check(
    "P1c the detach at the end of the session is accepted",
    "error" not in detach and detach.get("sourcesDetached") == 1,
    disposition(detach) if "error" in detach else json.dumps(detach),
    "no error and sourcesDetached==1",
  )
  on_disk = digest_dir(p_dir)
  w.check(
    "P1d both files are on disk, the container under the MODEL'S OWN name",
    set(on_disk) == {"card-5455.nncounts", "card-5455.%s.nnevals" % name_a},
    str(sorted(on_disk)),
    "exactly card-5455.nncounts and card-5455.%s.nnevals" % name_a,
  )
  w.check("P1e the engine exits cleanly", out["__rc__"] == 0, "rc=%d" % out["__rc__"], "rc==0")
  print()

  print("== P2: warm -- a NEW PROCESS re-attaches and pays ZERO neural net rows ==")
  out2, rows2 = h.session(p_cfg, [net_a], [
    {"id": "a", "action": "cache_attach", "context": "card-5455"},
    dict(POSITION, id="q"),
  ])
  attach2, warm = out2["a"], out2["q"]
  warm_rows = rows2[net_a]
  w.check(
    "P2a the new process attaches exactly what the old one wrote",
    attach2.get("entriesInLevelZero") == written,
    "entriesInLevelZero=%s (the first session wrote %s) %s"
    % (attach2.get("entriesInLevelZero"), written, disposition(attach2)),
    "entriesInLevelZero == %s" % written,
  )
  w.check(
    "P2b *** ZERO neural net rows on the re-served keys ***",
    warm_rows == 0,
    "NN rows=%d in the warm process; the cold process paid %d for the same query"
    % (warm_rows, cold_rows),
    "NN rows == 0 exactly -- not fewer, none",
  )
  w.check(
    "P2c and the warm answer is IDENTICAL to the cold one, search and all",
    search_shape(warm) == search_shape(cold),
    "warm=%s\n                  cold=%s" % (search_shape(warm), search_shape(cold)),
    "the same whole search shape -- what was served is the evaluations that were stored, "
    "bit for bit, and the search they drove reproduced move for move. NOTE this leg alone "
    "does NOT witness the cache: the same net recomputing the same position produces the "
    "same numbers too. P2b is the leg that witnesses it",
  )
  print()

  print("== P3: the persisted mark, and then attach -> dump -> detach -> re-attach -> dump ==")
  q_dir, q_cfg = h.config("p3")
  # P3-I: the entries are earned HERE, so the second dump's alreadyPersisted is a statement
  # about entries this very session owns -- the number the persisted mark exists to produce.
  # (In P3-II below the process earns nothing, so nothing is owed and nothing is
  # already-persisted either; that leg is about the FILE, this one about the MARK.)
  seed, _ = h.session(q_cfg, [net_a], [
    {"id": "a", "action": "cache_attach", "context": "card-5455"},
    dict(POSITION, id="q"),
    {"id": "d", "action": "cache_dump", "context": "card-5455", "what": "both", "admission": {"all": True}},
    {"id": "d0", "action": "cache_dump", "context": "card-5455", "what": "both", "admission": {"all": True}},
    {"id": "x", "action": "cache_detach", "context": "card-5455"},
  ])
  seeded = seed["d"].get("evaluations", {}).get("entriesWritten")
  second = seed["d0"].get("evaluations", {})
  w.check(
    "P3-I the PERSISTED MARK: a second dump of entries this session earned owes nothing",
    "error" not in seed["d0"] and seeded > 0 and second.get("entriesWritten") == 0
    and second.get("alreadyPersisted") == seeded and second.get("bytesAppended") == 0
    and second.get("belowThreshold") == 0 and second.get("notResident") == 0,
    "first dump wrote %s; second dump: written=%s alreadyPersisted=%s bytesAppended=%s "
    "belowThreshold=%s notResident=%s %s"
    % (seeded, second.get("entriesWritten"), second.get("alreadyPersisted"),
       second.get("bytesAppended"), second.get("belowThreshold"), second.get("notResident"),
       disposition(seed["d0"])),
    "entriesWritten==0, alreadyPersisted==%s, bytesAppended==0, and NOTHING dropped for any "
    "other reason -- 'already on disk' is why, not 'below threshold' or 'gone'" % seeded,
  )
  state1 = digest_dir(q_dir)
  profile1 = observation_profile(h, q_cfg, net_a, "card-5455")

  out3, rows3 = h.session(q_cfg, [net_a], [
    {"id": "a1", "action": "cache_attach", "context": "card-5455"},
    {"id": "s1", "action": "cache_stats"},
    {"id": "d1", "action": "cache_dump", "context": "card-5455", "what": "both", "admission": {"all": True}},
    {"id": "x1", "action": "cache_detach", "context": "card-5455"},
    {"id": "a2", "action": "cache_attach", "context": "card-5455"},
    {"id": "d2", "action": "cache_dump", "context": "card-5455", "what": "both", "admission": {"all": True}},
    {"id": "s2", "action": "cache_stats"},
  ])
  state2 = digest_dir(q_dir)
  profile2 = observation_profile(h, q_cfg, net_a, "card-5455")
  steps = [out3[k] for k in ("a1", "d1", "x1", "a2", "d2")]
  ok_seq = all("error" not in r for r in steps)
  w.check(
    "P3  the whole attach/dump/detach/re-attach/dump sequence is accepted",
    ok_seq and rows3[net_a] == 0,
    "%s ; NN rows in the process=%d"
    % (" | ".join(disposition(r) for r in steps if "error" in r) or "no refusals", rows3[net_a]),
    "no refusals, and zero neural net rows -- the process asked no query. A refusal here is "
    "either the property or the KNOWN-RACE labelled in the witness's own header",
  )
  evals = "card-5455.%s.nnevals" % name_a
  w.check(
    "P3a the CONTAINER is byte-identical across BOTH no-op dumps",
    ok_seq and state1.get(evals) == state2.get(evals),
    "before=%s after=%s" % (state1.get(evals), state2.get(evals)),
    "the same (size, sha256) -- the persisted mark is what gives this",
  )
  w.check(
    "P3b both no-op dumps write NO evaluation bytes and drop nothing for any other reason",
    ok_seq and all(
      r.get("evaluations", {}).get("entriesWritten") == 0
      and r.get("evaluations", {}).get("bytesAppended") == 0
      and r.get("evaluations", {}).get("belowThreshold") == 0
      and r.get("evaluations", {}).get("notResident") == 0
      for r in (out3["d1"], out3["d2"])),
    " | ".join(
      "%s: written=%s bytes=%s belowThreshold=%s notResident=%s alreadyPersisted=%s"
      % (k, out3[k].get("evaluations", {}).get("entriesWritten"),
         out3[k].get("evaluations", {}).get("bytesAppended"),
         out3[k].get("evaluations", {}).get("belowThreshold"),
         out3[k].get("evaluations", {}).get("notResident"),
         out3[k].get("evaluations", {}).get("alreadyPersisted"))
      for k in ("d1", "d2")),
    "entriesWritten==0 and bytesAppended==0 in both, with belowThreshold and notResident "
    "both 0 -- a dump that wrote nothing because it LOST the entries would look the same in "
    "entriesWritten alone",
  )
  rows_b4 = context_of(out3["s1"], "card-5455").get("countLogRows")
  rows_af = context_of(out3["s2"], "card-5455").get("countLogRows")
  w.check(
    "P3c the count log's ROW COUNT is unchanged by the no-op dumps",
    ok_seq and rows_b4 is not None and rows_b4 == rows_af,
    "countLogRows %s -> %s (read through the engine's own count-log reader)" % (rows_b4, rows_af),
    "the same row count -- the deltas the two dumps took were empty",
  )
  w.check(
    "P3d NO key's recorded observations moved: every minObservations threshold admits the same keys",
    ok_seq and profile1 == profile2 and profile1.get(0) == seeded,
    "before=%s after=%s (the container holds %s keys)" % (profile1, profile2, seeded),
    "the two profiles are equal, and minObservations=0 admits all %s. Had a dump re-appended "
    "counts it already wrote, every key's total would have risen and the profile would have "
    "shifted right" % seeded,
  )
  counts = "card-5455.nncounts"
  grew = state2.get(counts, (0,))[0] - state1.get(counts, (0,))[0]
  blocks_b4 = context_of(out3["s1"], "card-5455").get("countLogBlocks")
  blocks_af = context_of(out3["s2"], "card-5455").get("countLogBlocks")
  w.check(
    "P3e the count log is NOT byte-identical: each no-op dump appended an EMPTY BLOCK",
    ok_seq and grew == 2 * COUNT_LOG_BLOCK_HEADER_BYTES and blocks_af == blocks_b4 + 2
    and rows_b4 == rows_af,
    "%s grew %d bytes over two no-op dumps; countLogBlocks %s -> %s; countLogRows %s -> %s"
    % (counts, grew, blocks_b4, blocks_af, rows_b4, rows_af),
    "grew by exactly 2 x %d (one block header each, zero records) and by no content. "
    "nncachedump.h states the byte-identity rule for the SIBLING container ('An empty plan "
    "appends NOTHING AT ALL -- not a zero-entry block') and NNCacheCountLog::appendDump does "
    "not apply it. FILED, not fixed in this increment -- see "
    "audit-reports/impl-persistence-e2e.md. This leg goes RED if the growth ever becomes "
    "proportional to content, which is what a lost persisted mark looks like."
    % COUNT_LOG_BLOCK_HEADER_BYTES,
  )
  print()
