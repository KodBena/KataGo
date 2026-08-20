#!/usr/bin/env python3
"""check_vectors.py -- run a frozen-cache implementation against the conformance vectors.

WHO THIS IS FOR
    Whoever implements the frozen read-only cache described in cpp/spec/chd/SPEC.md.
    Running this script requires nothing except this repository. It does not read, need,
    or contain any part of the archived prototype the vectors were observed from.

WHAT IT CHECKS
    For every case in cpp/spec/chd/vectors/, one of two things:

      outcome=BUILD_OK    Your implementation must construct successfully from the case's
                          key list, then answer every query in the case's query list with
                          exactly the line recorded in the case's expected list.
      outcome=BUILD_FAIL  Your implementation must REFUSE to construct from the case's key
                          list: non-zero exit status and a non-empty diagnostic on stderr.
                          Exiting zero is a failure of this check even if nothing crashed.

WHAT YOUR DRIVER MUST DO
    You supply one executable. It is invoked in exactly two forms.

      <driver> build-and-query <keys-file> <queries-file>
          Build the structure over the keys in <keys-file>, IN FILE ORDER: the key on line
          1 is entry 0, the key on line 2 is entry 1, and so on. Then, for each query in
          <queries-file> in order, print one line to stdout:
              "MEMBER <i>"   if the query key is the key at entry i
              "ABSENT"       if the query key is not in the key set
          Exit 0. Print nothing else to stdout.

      <driver> build-only <keys-file>
          Attempt construction only. Exit 0 on success. On refusal, exit non-zero and write
          a diagnostic naming the reason to stderr.

    Key file line format, both files: two lowercase 16-digit hex fields separated by one
    space -- the first is hash0, the second is hash1. A file with zero lines is a valid
    empty key set.

USAGE
      python3 check_vectors.py <driver> [vectors-dir]
      python3 check_vectors.py --self-test [vectors-dir]

    --self-test runs three deliberately broken stub drivers and requires each to FAIL, which
    is how this script demonstrates its own checks can go red for the right reason rather
    than passing vacuously.

EXIT STATUS
    0 if every case passed. 1 otherwise, with a per-case report on stdout.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VECTORS = os.path.normpath(os.path.join(HERE, "..", "vectors"))
TIMEOUT_S = 900


def read_meta(path):
    meta = {}
    with open(path) as f:
        for line in f:
            if "\t" in line:
                k, v = line.rstrip("\n").split("\t", 1)
                meta[k] = v
    return meta


def classifications(vecdir):
    """case -> ACCEPTANCE | PROTOTYPE-SPECIFIC, read from MANIFEST.tsv.

    PROTOTYPE-SPECIFIC cases record behaviour caused by an internal choice of the archived
    prototype that a new implementation is explicitly NOT required to reproduce. They are run
    and reported, but they cannot fail the check.
    """
    path = os.path.join(vecdir, "MANIFEST.tsv")
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            if line.startswith("#") or line.startswith("case\t"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) > 2:
                out[parts[0]] = parts[2]
    return out


def cases(vecdir):
    names = sorted(
        f[:-5] for f in os.listdir(vecdir) if f.endswith(".meta")
    )
    return [(n, read_meta(os.path.join(vecdir, n + ".meta"))) for n in names]


def run(driver, args, want_stdout):
    try:
        p = subprocess.run(
            [driver] + args,
            capture_output=True, text=True, timeout=TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return None, "", "driver timed out after %d s" % TIMEOUT_S
    return p.returncode, p.stdout if want_stdout else "", p.stderr


def check_case(driver, vecdir, name, meta):
    """Returns (passed, detail)."""
    keys = os.path.join(vecdir, name + ".keys")

    if meta.get("outcome") == "BUILD_FAIL":
        rc, _, err = run(driver, ["build-only", keys], False)
        if rc is None:
            return False, err
        if rc == 0:
            return False, ("construction was expected to be REFUSED and instead "
                           "succeeded (exit 0)")
        if not err.strip():
            return False, ("construction was refused (exit %d) but wrote no diagnostic "
                           "to stderr; a silent refusal is not an acceptable refusal" % rc)
        return True, "refused: exit %d, %r" % (rc, err.strip().splitlines()[0][:120])

    queries = os.path.join(vecdir, name + ".queries")
    expected_path = os.path.join(vecdir, name + ".expected")
    rc, out, err = run(driver, ["build-and-query", keys, queries], True)
    if rc is None:
        return False, err
    if rc != 0:
        return False, "driver exited %d; stderr: %s" % (rc, err.strip()[:200])

    with open(expected_path) as f:
        exp = f.read().split("\n")
    if exp and exp[-1] == "":
        exp.pop()
    got = out.split("\n")
    if got and got[-1] == "":
        got.pop()

    if len(got) != len(exp):
        return False, "produced %d answer lines, expected %d" % (len(got), len(exp))

    bad_member = bad_absent = 0
    first = None
    for i, (g, e) in enumerate(zip(got, exp)):
        if g.strip() != e.strip():
            if e.strip() == "ABSENT":
                bad_absent += 1
            else:
                bad_member += 1
            if first is None:
                with open(queries) as qf:
                    qline = qf.read().split("\n")[i]
                first = "query line %d (%s): expected %r, got %r" % (i + 1, qline, e.strip(), g.strip())
    if first is None:
        return True, "%s answers matched" % len(exp)
    return False, ("%d wrong (absent-key contract: %d, member-lookup contract: %d); first: %s"
                   % (bad_absent + bad_member, bad_absent, bad_member, first))


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    if argv[1] == "--self-test":
        return self_test(argv[2] if len(argv) > 2 else DEFAULT_VECTORS)

    driver = argv[1]
    vecdir = argv[2] if len(argv) > 2 else DEFAULT_VECTORS
    if not os.path.isdir(vecdir):
        print("no such vectors directory: %s" % vecdir)
        return 2

    cls = classifications(vecdir)
    failures = 0
    advisory = 0
    graded = 0
    for name, meta in cases(vecdir):
        ok, detail = check_case(driver, vecdir, name, meta)
        if cls.get(name) == "PROTOTYPE-SPECIFIC":
            advisory += 1
            print("%-4s %-32s %s" % ("adv?" if ok else "adv-", name,
                                     detail + "   [advisory only: not required]"))
            continue
        graded += 1
        print("%-4s %-32s %s" % ("PASS" if ok else "FAIL", name, detail))
        if not ok:
            failures += 1
    print("\n%d/%d graded cases passed (%d advisory case(s) not graded)"
          % (graded - failures, graded, advisory))
    return 0 if failures == 0 else 1


# ---------------------------------------------------------------------------------------
# Self-test. Three broken stub drivers, each of which MUST fail, and each of which fails a
# different clause of the contract. This is the evidence that a PASS from this script is
# not vacuous. None of these stubs is a hash-table implementation.
# ---------------------------------------------------------------------------------------
STUBS = {
    # The first two stubs refuse every build-only invocation, which is the CORRECT answer for
    # every case the checker issues build-only against. That isolates each stub's rejections
    # to the one query-answering leg it breaks, rather than letting the refusal leg mask it.
    "always-absent": (
        "answers ABSENT to every query; refusal leg correct",
        '#!/bin/sh\n'
        'if [ "$1" = "build-only" ]; then echo "stub refuses" >&2; exit 1; fi\n'
        'while read -r _; do echo ABSENT; done < "$3"\n',
        "the member-lookup leg, in isolation",
    ),
    "always-member-0": (
        "answers MEMBER 0 to every query; refusal leg correct",
        '#!/bin/sh\n'
        'if [ "$1" = "build-only" ]; then echo "stub refuses" >&2; exit 1; fi\n'
        'while read -r _; do echo "MEMBER 0"; done < "$3"\n',
        "the absent-key leg, in isolation",
    ),
    "never-refuses": (
        "echoes the correct answers but never refuses a construction",
        None,  # built specially below
        "the refusal leg",
    ),
}


def self_test(vecdir):
    import stat
    import tempfile

    tmp = tempfile.mkdtemp(prefix="chdvec-selftest-")
    results = []

    def write_stub(name, body):
        p = os.path.join(tmp, name)
        with open(p, "w") as f:
            f.write(body)
        os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        return p

    # The "never-refuses" stub replays the frozen expected files verbatim for BUILD_OK
    # cases -- so it is correct on every clause EXCEPT refusal, isolating that leg.
    replay = (
        '#!/bin/sh\n'
        'V=%s\n'
        'if [ "$1" = "build-only" ]; then exit 0; fi\n'
        'B=$(basename "$2" .keys)\n'
        'cat "$V/$B.expected"\n' % vecdir
    )
    STUBS["never-refuses"] = (STUBS["never-refuses"][0], replay, STUBS["never-refuses"][2])

    cls = classifications(vecdir)
    graded = [(n, m) for n, m in cases(vecdir) if cls.get(n) != "PROTOTYPE-SPECIFIC"]

    for name, (desc, body, leg) in STUBS.items():
        drv = write_stub(name, body)
        failed = []
        for cname, meta in graded:
            ok, detail = check_case(drv, vecdir, cname, meta)
            if not ok:
                failed.append((cname, detail))
        verdict = "RED (correct)" if failed else "GREEN (WRONG -- the check is vacuous)"
        print("stub %-16s -- %s" % (name, desc))
        print("     exercises: %s" % leg)
        print("     %d/%d graded cases rejected -> %s" % (len(failed), len(graded), verdict))
        if failed:
            print("     first rejection: %s: %s" % (failed[0][0], failed[0][1][:160]))
        results.append(bool(failed))
        print()

    # Positive control. A stub that replays the frozen answers AND refuses every build-only
    # invocation is correct on every clause the checker grades, so it must be ACCEPTED. If it
    # is not, the checker rejects conforming behaviour and its reds above mean nothing.
    good = write_stub("faithful-replay",
                      '#!/bin/sh\n'
                      'V=%s\n'
                      'if [ "$1" = "build-only" ]; then echo "stub refuses" >&2; exit 1; fi\n'
                      'B=$(basename "$2" .keys)\n'
                      'cat "$V/$B.expected"\n' % vecdir)
    bad = [c for c, m in graded if not check_case(good, vecdir, c, m)[0]]
    print("stub faithful-replay -- replays the frozen answers and refuses every construction "
          "it is asked to refuse")
    print("     exercises: the positive control -- conforming behaviour must be ACCEPTED")
    print("     %d/%d graded cases rejected -> %s"
          % (len(bad), len(graded),
             "GREEN (correct)" if not bad else "RED (WRONG -- the checker rejects conforming behaviour)"))
    if bad:
        print("     wrongly rejected: %s" % ", ".join(c for c in bad[:5]))
    print()
    positive_ok = not bad

    if all(results) and positive_ok:
        print("SELF-TEST PASSED: every deliberately broken stub was rejected, each for the "
              "clause it breaks, and conforming behaviour was accepted.")
        return 0
    print("SELF-TEST FAILED: a broken stub was accepted, or conforming behaviour was "
          "rejected. The checker is not observing what it claims to observe.")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
