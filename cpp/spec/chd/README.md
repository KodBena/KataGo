# cpp/spec/chd — frozen read-only NN cache: specification and conformance vectors

This directory is the complete brief for building a **frozen read-only neural-net output
cache** for this fork: a structure built once from a fixed set of position hashes and their
evaluations, never modified afterwards, answering "do you hold an evaluation for this
position, and which one" as fast as possible. The operator wants it built as a CHD
(compress-hash-displace) minimal perfect hash.

A working prototype of it exists in an archive at `/home/bork/kata_fork_archived`.
**Do not open that path.** The operator's instruction is that its behaviour is to be carried
over and its internal shape is not. Everything you need was extracted from it into this
directory; if something is missing, report the gap rather than going to look.

## Read in this order

1. **`SPEC.md`** — the behavioural specification. Start at §0 (the acceptance bar, two
   sentences long) and read §2 (the absent-key contract) before writing anything; it is the
   clause most likely to be got wrong and it fails silently.
2. **`vectors/MANIFEST.tsv`** — what the conformance vectors cover.
3. **`SPEC.md` §7 and §8** — how you are checked.

## The acceptance loop

Write one small driver that wraps your implementation and accepts two invocations
(`build-and-query` and `build-only`; the exact contract is `SPEC.md` §7.2). Then:

```sh
# prove the checker's own checks can fail, before trusting a pass from it
python3 cpp/spec/chd/tools/check_vectors.py --self-test

# check your implementation
python3 cpp/spec/chd/tools/check_vectors.py /path/to/your-driver
```

Passing every graded vector is one of the two halves of acceptance. The other is the lookup
performance floor in `SPEC.md` §8, which the vectors do not test and which you must measure.

## What is in here

| path | |
|---|---|
| `SPEC.md` | the specification; every property labelled CONTRACT or INCIDENTAL |
| `vectors/*.keys` `.queries` `.expected` `.meta` | 23 frozen cases, observed by running the prototype |
| `vectors/MANIFEST.tsv` | case index: sizes, sha256s, key-set provenance, graded vs advisory |
| `vectors/DIAGNOSTIC-internals.tsv` | prototype internals, **not** an acceptance artifact, nothing checks it |
| `tools/check_vectors.py` | the checker, plus its own self-test. Needs nothing but this repo |
| `tools/make_manifest.py` | regenerates `MANIFEST.tsv` |
| `tools/gen_vectors.cpp`, `tools/build.sh` | **the instrument that read the archive.** Do not build or run these |

`tools/gen_vectors.cpp` includes a header from the archive by absolute path. It is committed
so the vectors' provenance is auditable, not so anyone under the clean-room constraint runs
it. Nothing else in this directory reaches outside the repository.

## Where the measured numbers came from

`SPEC.md` §12 lists the source of every measured figure quoted in the specification. The
report covering this extraction pass is `/home/bork/kg/audit-reports/chd-spec.wiki`.
