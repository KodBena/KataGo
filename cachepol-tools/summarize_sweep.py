#!/usr/bin/env python3
"""Read a policy-sweep results file and print it as a table.

WHAT THE INPUT IS. `katago benchnncachepolicy` writes NDJSON: one JSON object per line.
The first line is a "substrate" record describing the machine, the build and the trace;
every later line with record="config" is one cache configuration's result; the last line
is a "substrate_after" record carrying the swap counters across the whole run. The file
is designed to be readable on a machine that is not the one that produced it, so this
script needs nothing except the file.

WHAT IT PRINTS. One row per configuration, with occupancy beside every hit rate. That
pairing is the point of the whole exercise: a replacement policy has nothing to do in a
nearly-empty table, so a hit rate quoted without the occupancy it was measured at cannot
be compared to anything.

Usage:
  summarize_sweep.py <results.ndjson>                 # every configuration
  summarize_sweep.py <results.ndjson> --substrate     # just the machine and the trace
  summarize_sweep.py <results.ndjson> --best          # best hit rate at each (pow,ownership)
  summarize_sweep.py <results.ndjson> --csv           # a flat CSV of the config rows
"""
import json
import sys


def load(path):
    substrate, after, rows = None, None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            kind = d.get("record")
            if kind == "substrate":
                substrate = d
            elif kind == "substrate_after":
                after = d
            elif kind == "config":
                rows.append(d)
    if substrate is None:
        raise SystemExit("%s: no substrate record; this is not a policy-sweep results file" % path)
    return substrate, after, rows


def shape_name(r):
    c = r["collision"]
    if c == "direct":
        # Four different tables answer to "direct" now, distinguished by which of the two
        # candidates a collision keeps. Labelling them all "direct" would silently collapse
        # a swept axis into one row-name; the status quo keeps the bare name.
        rep = r.get("replacement", "always")
        base = "direct" if rep in ("always", None) else "direct/%s" % rep
    elif c == "chain":
        base = "chain/%s" % r["eviction"]
    else:
        base = "%s/%dway/%s" % (c, r["ways"], r["eviction"])
    if r["admission"] == "secondsighting":
        base += "+2nd"
    return base


def print_substrate(s, after):
    print("=== substrate ===")
    for k in ("written_at", "host", "kernel", "cpu_model", "nproc",
              "meminfo_memtotal", "meminfo_memavailable",
              "thp_enabled", "thp_defrag", "hugepages_total",
              "ulimit_v", "ulimit_l", "idle_pct_before",
              "katago_version", "git_revision", "backend_that_built_the_trace",
              "trace_path", "trace_format", "trace_records", "trace_gets",
              "trace_hits_as_recorded", "trace_sets", "trace_sets_with_ownermap",
              "trace_distinct_set_keys", "trace_set_repeats", "trace_set_repeat_rate",
              "trace_reusable_gets", "trace_reuse_rate", "min_reuse_rate_floor",
              "trace_mean_set_bytes",
              "sizeof_NNOutput", "ownership_mode_arg",
              "max_bytes_budget", "preflight_peak_bytes", "chain_budget_bytes", "note"):
        if k in s:
            print("  %-30s %s" % (k, s[k]))
    if not s.get("trace_footprints_are_measured", True):
        print("  !! this trace is v1: per-entry footprints were ASSUMED at %s bytes, not measured"
              % s.get("assumed_entry_bytes_for_v1"))
    if after:
        print("  %-30s %s" % ("pswpin_delta", after.get("pswpin_delta")))
        print("  %-30s %s" % ("pswpout_delta", after.get("pswpout_delta")))
        print("  %-30s %s" % ("idle_pct_after", after.get("idle_pct_after")))
        if after.get("pswpin_delta", 0) not in (0, -1):
            print("  !! nonzero pswpin across the run: every cache_ops_per_sec figure here is")
            print("     DISQUALIFIED. Hit rate, occupancy and the byte figures are not timing")
            print("     quantities and survive it.")


def fmt_rows(rows):
    hdr = ("%-34s %5s %4s %9s %9s %9s %11s %11s %10s"
           % ("shape", "2^k", "own", "hit", "occupancy", "re-miss", "payloadMiB", "fixedMiB", "kops/s"))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r.get("status") != "OK":
            print("%-34s %5d %4s   %s: %s"
                  % (shape_name(r), r["table_pow"], r["ownership_mode"],
                     r.get("status"), (r.get("refusal") or "")[:90]))
            continue
        occ = r.get("occupancy")
        occs = ("%9.4f" % occ) if isinstance(occ, (int, float)) else "%9s" % "n/a"
        print("%-34s %5d %4s %9.4f %s %9.5f %11.1f %11.1f %10.1f"
              % (shape_name(r), r["table_pow"], r["ownership_mode"],
                 r["hit_rate"], occs, r["re_miss_rate_of_gets"],
                 r["resident_payload_bytes"] / 1048576.0,
                 r["fixed_structure_bytes"] / 1048576.0,
                 r["cache_ops_per_sec"] / 1000.0))


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    path = sys.argv[1]
    flags = sys.argv[2:]
    substrate, after, rows = load(path)

    if "--substrate" in flags:
        print_substrate(substrate, after)
        return

    if "--csv" in flags:
        cols = ["collision", "ways", "eviction", "admission", "replacement", "table_pow", "mutex_pool_pow",
                "ownership_mode", "status", "hit_rate", "occupancy", "re_miss_rate_of_gets",
                "resident_entries", "resident_payload_bytes", "fixed_structure_bytes",
                "cache_ops_per_sec"]
        print(",".join(cols))
        for r in rows:
            print(",".join(str(r.get(c, "")) for c in cols))
        return

    print_substrate(substrate, after)
    print()
    ok = [r for r in rows if r.get("status") == "OK"]
    if "--best" in flags:
        # The baseline every row should be read against is the direct-mapped table at the
        # same table size and ownership mode -- that is what KataGo does today, and a
        # policy that does not beat it is not worth its extra bytes.
        groups = {}
        for r in ok:
            groups.setdefault((r["table_pow"], r["ownership_mode"]), []).append(r)
        for key in sorted(groups):
            g = groups[key]
            # The baseline is the STATUS QUO direct table, so it must also carry the
            # status-quo replacement rule -- otherwise a run that sweeps the replacement
            # axis picks whichever direct row happened to come first as its baseline.
            base = [r for r in g
                    if r["collision"] == "direct" and r["admission"] == "always"
                    and r.get("replacement", "always") == "always"]
            print("=== 2^%d, ownership %s ===" % key)
            if base:
                b = base[0]
                print("  baseline direct: hit %.4f at occupancy %.4f, %.1f MiB payload"
                      % (b["hit_rate"], b["occupancy"], b["resident_payload_bytes"] / 1048576.0))
            fmt_rows(sorted(g, key=lambda r: -r["hit_rate"])[:8])
            print()
        return

    fmt_rows(rows)


if __name__ == "__main__":
    main()
