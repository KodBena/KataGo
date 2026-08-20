#!/usr/bin/env python3
"""make_manifest.py -- write vectors/MANIFEST.tsv.

One row per conformance case: outcome, sizes, sha256 of every data file, the provenance of
the key set, and the acceptance classification.

The acceptance classification matters. Two of the refusal cases record a refusal that is a
property of the archived prototype's particular hash use, not a requirement on a new
implementation: an implementation that ACCEPTS those key sets is strictly better and must not
be failed for it. Those rows are marked PROTOTYPE-SPECIFIC and cpp/spec/chd/tools/
check_vectors.py reads the column rather than carrying the list in its own logic.

Usage: python3 make_manifest.py [vectors-dir]
"""

import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.normpath(os.path.join(HERE, "..", "vectors"))

# case -> (classification, provenance of the key set)
CLASSIFY = {
    "n0_empty":                     ("ACCEPTANCE", "synthetic: empty"),
    "absent_foreign_corpus":        ("ACCEPTANCE", "synthetic: splitmix64 seed 0x1111, queried with seed 0x2222"),
    "absent_onebit_only":           ("ACCEPTANCE", "synthetic: splitmix64 seed 0x3333"),
    "shared_hash1_distinct_buckets":("ACCEPTANCE", "synthetic: splitmix64 seed 0x7770000, two entries overwritten"),
    "shared_hash0_one_bucket":      ("ACCEPTANCE", "synthetic: constructed, one shared hash0"),
    "extremal_keys":                ("ACCEPTANCE", "synthetic: hand-written extremal values"),
    "dup_exact":                    ("ACCEPTANCE", "synthetic: splitmix64 seed 0xD00D with position 7 repeated"),
    "dup_hash1_same_bucket":        ("PROTOTYPE-SPECIFIC", "synthetic: two distinct keys sharing hash1"),
    "hash1_degenerate_corpus":      ("PROTOTYPE-SPECIFIC", "synthetic: 64 distinct keys sharing hash1"),
    "real_card5074":                ("ACCEPTANCE", "real: operator DB, card 5074, num_refs>1, num_refs DESC then hash_id"),
    "real_card5455":                ("ACCEPTANCE", "real: operator DB, card 5455 (the median-size card), num_refs>1, num_refs DESC then hash_id"),
}

NOTE_PROTOTYPE_SPECIFIC = (
    "# PROTOTYPE-SPECIFIC rows record a construction refusal caused by a weakness of the\n"
    "# archived prototype's particular hash use, not by anything a new implementation must\n"
    "# reproduce. An implementation that constructs these key sets successfully is BETTER\n"
    "# and is conforming. check_vectors.py does not require them.\n"
)


def sha(path):
    if not os.path.exists(path):
        return "-"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_meta(path):
    m = {}
    with open(path) as f:
        for line in f:
            if "\t" in line:
                k, v = line.rstrip("\n").split("\t", 1)
                m[k] = v
    return m


def main(argv):
    vecdir = argv[1] if len(argv) > 1 else DEFAULT
    names = sorted(f[:-5] for f in os.listdir(vecdir) if f.endswith(".meta"))
    rows = []
    for n in names:
        meta = read_meta(os.path.join(vecdir, n + ".meta"))
        cls, prov = CLASSIFY.get(n, ("ACCEPTANCE", "synthetic: splitmix64, see gen_vectors.cpp"))
        rows.append([
            n,
            meta.get("outcome", "?"),
            cls,
            meta.get("n_keys", "?"),
            meta.get("n_queries", "0"),
            meta.get("n_expect_member", "0"),
            meta.get("n_expect_absent", "0"),
            sha(os.path.join(vecdir, n + ".keys")),
            sha(os.path.join(vecdir, n + ".queries")),
            sha(os.path.join(vecdir, n + ".expected")),
            prov,
        ])

    out = os.path.join(vecdir, "MANIFEST.tsv")
    with open(out, "w") as f:
        f.write("# Conformance vectors for cpp/spec/chd/SPEC.md.\n")
        f.write("# Observed by running the archived prototype; see chd-spec.wiki for the run.\n")
        f.write(NOTE_PROTOTYPE_SPECIFIC)
        f.write("case\toutcome\tclassification\tn_keys\tn_queries\tn_expect_member\t"
                "n_expect_absent\tsha256_keys\tsha256_queries\tsha256_expected\tkey_set_provenance\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    print("wrote %s (%d cases)" % (out, len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
