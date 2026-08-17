# Runbook: running the NN-cache policy sweep on the GPU host

## READ THIS FIRST: this runbook is UNEXERCISED

**Nobody has run any command in this document.** It was written inside a virtual machine
that has no GPU, cannot build a CUDA or TensorRT binary, and cannot reach the host these
steps are for. Every command below is therefore a **claim awaiting your execution**, not a
transcript of something that worked.

What *has* been exercised, on the VM, with an Eigen CPU build:

- the C++ code these commands invoke compiles, its tests pass, and `runoutputtests`
  differs from its committed golden by exactly the one known unrelated line;
- the replay stage (Stage 2 below) ran to completion against a real recorded trace and
  produced a results file;
- the driver script and the summariser were written against that results file.

What has **not** been exercised: the CUDA/TensorRT build, the capture stage against a GPU
backend, the benchmark stage, and every path and assumption about your host. Places where
this document is **guessing** about your machine are marked **GUESS** inline.

---

## Orientation: what this sweep is, in plain words

KataGo keeps a cache of neural-net evaluations, so that a position it has already
evaluated during a search does not have to be evaluated again. Until recently that cache
had exactly one shape, with no settings: a table of N slots, each position landing in
exactly one slot determined by its hash, and a new position simply overwriting whatever
was in its slot.

That shape is now one option among several, chosen by four keys in your KataGo `.cfg`:

| key | what it chooses |
|---|---|
| `nnCacheCollision` | what happens when two positions want the same slot: `direct` (overwrite, today's behaviour), `linearprobe` or `quadraticprobe` (try several nearby slots), or `chain` (no fixed slots at all; bounded by a byte budget) |
| `nnCacheWays` | under the two probing schemes, how many slots a position may try |
| `nnCacheEviction` | when something must be thrown out, which entry goes: `random`, `lru` (least recently used), or `lfu` (least frequently used) |
| `nnCacheAdmission` | whether an evaluation offered to the cache is stored at all: `always`, or `secondsighting` (store a position only from the second time it is offered) |

**The question this sweep answers** is which of those settings is worth having: for each
combination, what fraction of lookups hit, how many bytes the cache is actually holding to
achieve that, and how full the table was at the time.

**Why "how full" matters so much.** An eviction policy only does anything when something
has to be evicted. In a table that is 2% full, nothing is ever evicted, so `lru`, `lfu`
and `random` are all *identical* and every number looks reassuring. Every measurement
taken on this cache before this sweep sat at about 1.7% occupancy. So this sweep is
organised around reaching **high occupancy** and reporting occupancy beside every hit rate.

**How it reaches high occupancy cheaply.** Filling a large table means holding millions of
evaluations in memory, which is expensive. Filling a *small* table with the same workload
is the same experiment for a fraction of the memory. So the sweep runs the same recorded
workload against a range of table sizes: the small ones are saturated, the large ones are
sparse, and the result is a *curve* of policy benefit against occupancy on which you can
locate whatever table size you actually run.

---

## The three stages

| stage | what it does | needs the GPU? | roughly how long |
|---|---|---|---|
| 1. capture | runs KataGo's analysis engine once, recording every cache operation to a file | **yes** | tens of minutes to hours, your choice |
| 2. replay | replays that one recording through every cache configuration | no | minutes |
| 3. visits | runs `katago benchmark` for a few named configurations, for visits/s | **yes** | minutes each |

Stage 2 is where the answer is. Stage 1 exists to give stage 2 a realistic workload.
Stage 3 is an **affordability check only**: the cache path was previously measured at
0.0077% of total cycles, so no setting here can plausibly move visits/s much, and stage 3
is there to confirm that rather than to rank anything.

---

## Step 0 — the source tree to copy, and why it is not the one you might expect

**A git worktree is not self-contained.** The tree this work was done in has a `.git` that
is a *file* containing a path back into a parent repository, not a directory. Copying it
over sshfs produces a tree that no git command will work in and that may not even
configure, because the build reads the git revision.

A self-contained export has therefore been made for you. It is a plain directory with a
real `.git` inside it:

```
/home/bork/kg/cachepol-export/KataGo
```

Its provenance — which branch and commit it was cut from, and by what command — is written
in a file inside it:

```
/home/bork/kg/cachepol-export/KataGo/EXPORT-PROVENANCE.txt
```

**Copy that directory**, not the worktree. From the guest, over your sshfs mount:

```sh
# GUESS: substitute your own mount point for /mnt/host.
cp -a /home/bork/kg/cachepol-export/KataGo /mnt/host/katago-cachepol
```

If you would rather pull it on the host directly, the export is a normal git repository, so
this works too and is cheaper:

```sh
git clone /path/to/the/mounted/cachepol-export/KataGo /opt/katago-cachepol
```

**From here on, this runbook calls the copied tree `$KG` on the host.** Set it once:

```sh
export KG=/opt/katago-cachepol      # GUESS: wherever you actually put it
```

---

## Step 1 — build on the host, with a GPU backend

This VM's build uses `-DUSE_BACKEND=EIGEN`, which is a CPU backend and is **not** what you
want on the host. The two GPU options in this tree's `cpp/CMakeLists.txt` are
`-DUSE_BACKEND=TENSORRT` and `-DUSE_BACKEND=CUDA`. Prefer TensorRT if you have it; it is
faster and it is what the analysis-engine configs in this tree assume by default.

**TensorRT:**

```sh
cd $KG
cmake -S cpp -B cpp/build-trt \
      -DUSE_BACKEND=TENSORRT \
      -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build-trt -j"$(nproc)"
```

**CUDA (use this if TensorRT is not installed):**

```sh
cd $KG
cmake -S cpp -B cpp/build-cuda \
      -DUSE_BACKEND=CUDA \
      -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build-cuda -j"$(nproc)"
```

Set the binary path once, and use it everywhere below:

```sh
export KATAGO=$KG/cpp/build-trt/katago      # or build-cuda/katago
"$KATAGO" version
```

Notes, each of them a real thing that can bite:

- `-DUSE_AVX2=1` is a **CPU-backend** option. It does nothing useful for a GPU build; leave
  it off.
- If you want the TensorRT plan cache, it is `-DUSE_CACHE_TENSORRT_PLAN=1`, and this tree
  will refuse to combine it with `-DBUILD_DISTRIBUTED=1`. You do not need either for this.
- **GUESS:** if your host has no `nvcc` on `PATH`, the CUDA build will fail at
  `enable_language(CUDA)`. That failure is about your toolchain, not about this branch.
- Confirm the build really is the GPU one before going further; `"$KATAGO" version` prints
  the backend. A silently-Eigen binary would make every number in stage 3 meaningless.

---

## Step 2 — which model to use

This VM has one model:

```
/home/bork/kg/14.gz
sha256 c1ec21a19069b265373df27018f28dcb07993e75a3a750d45c3b2dcc05dd6ed8
size   4,973,824 bytes
net    b6c96-s69427456-d10051148   (a 6-block, 96-channel network)
```

**Plain answer: it is the right net for stage 1 and the wrong net for stage 3.**

- **Stage 1 (capture): 14.gz is fine, and arguably better than a big net.** All that stage
  1 produces is the *sequence of positions* the search visited. A 6-block net gives you far
  more visited positions per minute of GPU time than a 40-block net would, and more
  positions is exactly what you need — see "How big a trace do I need" below. What changes
  if you use a larger net: the positions the search chooses will differ somewhat, because a
  stronger net shapes the tree differently. Whether that changes the *reuse structure* the
  cache sees is **UNEXERCISED** — nobody has measured a b6 trace against a b18 trace. If
  you have the GPU time, capturing with the net you actually review with is the more
  faithful choice, and the results file records which net was used either way.
- **Stage 3 (visits/s): 14.gz will flatter every configuration and should not be used.**
  Stage 3 asks whether a cache policy costs measurable throughput. A 6-block net evaluates
  so cheaply that the cache path is a larger fraction of the total than it is in your real
  workload, so a b6 measurement is a *pessimistic* one — it exaggerates any cache cost. If
  it shows no cost, that is a strong result. But the number itself will not be your
  number. **Use the net you actually run** for stage 3, and put it in the same directory as
  the copied tree so the paths below are simple.

Put whichever model you use somewhere with an absolute path and set it:

```sh
export MODEL=/opt/katago-nets/your-model.bin.gz    # GUESS: your own path
sha256sum "$MODEL"      # record this; the driver script records it for you too
```

---

## Step 3 — check transparent hugepages, and record the answer

```sh
cat /sys/kernel/mm/transparent_hugepage/enabled
cat /sys/kernel/mm/transparent_hugepage/defrag
grep -i huge /proc/meminfo
```

The first file prints one of `[always] madvise never`, `always [madvise] never`, or
`always madvise [never]`; the bracketed word is the current setting.

**Why it is worth a step of its own.** Earlier work on this cache measured transparent
hugepages worth **17–28%** of a large-table lookup — larger than the cost of *doubling the
table* — and nothing in KataGo's configuration controls whether it gets them. It also
silently desynchronised two measurement sweeps in that work, which is why it is recorded
here rather than assumed.

**What NOT to conclude, and this is important.** That 17–28% was measured *inside the
guest VM*, whose memory is itself backed by statically preallocated hugepages on your host.
Guest-level and host-level page sizing are different questions, and **the guest figure is
not claimed to reproduce on the host — that transfer is UNEXERCISED.** Record the state;
do not budget against the number.

**What to do about it:** nothing, unless you want to. The point is that it is *recorded*,
so that if two of your runs disagree you can check whether this is why. The driver script
and the results file both capture it automatically. If you do want to change it (root
required, affects the whole machine):

```sh
echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled   # or: never
```

If you change it, run the whole sweep on one setting. A sweep split across two settings is
not one sweep.

---

## Step 4 — decide the memory budget, and say it out loud

Stage 2 **requires** you to state a memory budget in bytes, with `-b`. There is
deliberately no default: the previous plan for this sweep was built on an assumed memory
figure that turned out to be wrong, and that assumption is what cost the plan.

The budget is what is **free**, not what is installed. On your host:

```sh
free -m
grep -i huge /proc/meminfo
```

**GUESS, from what was described to me:** the host has 32 GB, of which roughly 10 GB is
pinned as statically preallocated hugepages for the VM, plus whatever else you are running.
So a budget of a few GB is realistic and 32 GB is not. Something like:

```sh
export MAXBYTES=6000000000     # 6 GB. Raise or lower it against your own `free -m`.
```

Stage 2 computes the largest single configuration's footprint before it allocates
anything, prints the arithmetic, and **refuses to start** with that arithmetic in the
message if it exceeds your budget. It refuses rather than attempts because a run that
swaps produces a lookup curve that looks like a memory-hierarchy effect and is not one.

If it refuses, the fixes in order of preference are: drop the largest entry from
`-p`/`-table-pows`; pass `-W off` to run ownership-off only; or raise the budget if the
machine genuinely has the headroom.

---

## Step 5 — run the sweep

One script does all three stages:

```sh
mkdir -p /opt/katago-cachepol-results        # GUESS: pick your own, on real disk

$KG/cachepol-tools/run_policy_sweep.sh \
    -k "$KATAGO" \
    -m "$MODEL" \
    -c $KG/cpp/configs/analysis_example.cfg \
    -o /opt/katago-cachepol-results/run1 \
    -b "$MAXBYTES" \
    -V 20000 \
    -p 17,18,19,20,21
```

Run `$KG/cachepol-tools/run_policy_sweep.sh -h` for every option.

To run the stages separately — which you will want, because stage 1 is the expensive one
and you will re-run stage 2 several times against one capture:

```sh
# just the capture
$KG/cachepol-tools/run_policy_sweep.sh -s capture -k "$KATAGO" -m "$MODEL" \
    -c $KG/cpp/configs/analysis_example.cfg -o /opt/katago-cachepol-results/run1 -b "$MAXBYTES"

# just the replay, against the trace the capture left behind
$KG/cachepol-tools/run_policy_sweep.sh -s replay -k "$KATAGO" \
    -o /opt/katago-cachepol-results/run1 -b "$MAXBYTES" -p 17,18,19,20,21

# just the visits check
$KG/cachepol-tools/run_policy_sweep.sh -s visits -k "$KATAGO" -m "$MODEL" \
    -c $KG/cpp/configs/analysis_example.cfg -o /opt/katago-cachepol-results/run1 -b "$MAXBYTES"
```

### How big a trace do I need?

This is the one parameter that decides whether the sweep answers anything.

The sweep can only saturate a table of `2^k` slots if the trace contains **more than `2^k`
distinct positions**. Roughly one distinct position per visit:

| you want to saturate | distinct positions needed | with `-V 20000`, queries needed |
|---|---|---|
| 2^17 = 131,072 | ~131k | ~7 |
| 2^18 = 262,144 | ~262k | ~14 |
| 2^19 = 524,288 | ~524k | ~27 |
| 2^20 = 1,048,576 | ~1.05M | ~53 |
| 2^21 = 2,097,152 | ~2.1M | ~105 |

The capture stage writes 24 queries by default. **Raise `-V` rather than editing the
queries** if you want a bigger trace; the script regenerates the query file only if it is
absent, so delete `run1/capture-queries.jsonl` if you want it rewritten.

A trace of 2 million distinct positions is about **48 MB** on disk (24 bytes per operation,
and there are more operations than positions), so disk is not the constraint; GPU time is.

**The honest floor:** if your trace has fewer distinct positions than the smallest table
you sweep, every eviction policy will score identically and the sweep will tell you
nothing. The results file records `trace_distinct_set_keys` so you can check this
afterwards — but check it before you spend the GPU time.

### Which stream carries what

This matters and it is easy to get wrong.

- **KataGo's analysis engine writes protocol responses to `stdout` and its log to
  `stderr`.** They are two different things and must go to two different places. The
  driver script already does this: responses to `run1/capture-responses.jsonl`, log to
  `run1/capture-engine.log`. They are never merged.
- **The trace is on neither stream.** It is a binary file written directly to the path in
  the `KATAGO_NNCACHE_TRACE` environment variable. It cannot be corrupted by a log line
  because no log line ever goes near it.
- **`katago benchmark` writes its report to `stdout`.** The script sends it to
  `run1/visits/<name>.stdout`, with anything else to `run1/visits/<name>.stderr`.
- **The results file is written by the binary directly to the path you gave `-out`**, never
  to a stream. `benchnncachepolicy` writes only progress lines to stderr; the script sends
  those to `run1/policy-sweep.stderr`. **So a log line cannot end up inside a results
  file.** That is by construction, not by redirection care.

If you ever run these commands by hand rather than through the script, the rule is: never
write `2>&1` on any command whose stdout you intend to keep.

---

## Step 6 — where the results are, and getting them back to the guest

Everything from one run lands under the `-o` directory. For the example above:

```
/opt/katago-cachepol-results/run1/
    substrate.txt              the host as it was at the top of the run
    capture-queries.jsonl      the queries the capture ran
    capture-responses.jsonl    the engine's protocol output (stdout)
    capture-engine.log         the engine's log (stderr)
    trace.bin                  the recorded cache operation stream
    policy-sweep.ndjson        <-- THE RESULTS FILE
    policy-sweep.gate          whether the machine was quiet while stage 2 ran
    policy-sweep.stderr        stage 2 progress lines
    visits/<name>.cfg          the exact config each benchmark ran
    visits/<name>.stdout       each benchmark report
    visits/<name>.gate         whether the machine was quiet for each
```

**The results file is the deliverable.** It is one NDJSON file: the first line describes
the machine, the build, the model and the trace; every later line is one cache
configuration's result; the last line carries the swap counters across the whole run. It is
built to be read on a machine that is not the one that produced it and with no other
context, which is exactly the situation everyone downstream is in.

**Getting it back to the guest.** It is small — a few hundred KB — so the sshfs mount is
enough. `trace.bin` is the only large file and you do **not** need to copy it back.

```sh
# on the host, writing into the guest-visible mount   (GUESS: your mount point)
cp /opt/katago-cachepol-results/run1/policy-sweep.ndjson  /mnt/host/from-gpu/
cp /opt/katago-cachepol-results/run1/policy-sweep.gate    /mnt/host/from-gpu/
cp /opt/katago-cachepol-results/run1/substrate.txt        /mnt/host/from-gpu/
cp -r /opt/katago-cachepol-results/run1/visits            /mnt/host/from-gpu/
```

Read it, on either machine:

```sh
$KG/cachepol-tools/summarize_sweep.py /path/to/policy-sweep.ndjson --substrate
$KG/cachepol-tools/summarize_sweep.py /path/to/policy-sweep.ndjson --best
$KG/cachepol-tools/summarize_sweep.py /path/to/policy-sweep.ndjson --csv > sweep.csv
```

---

## Step 7 — checking that the numbers are worth anything

Three checks, in order. Each has a specific failure it catches.

**1. Was the machine quiet?** Look at `policy-sweep.gate`:

```
GATE idle_pct_before=98
GATE pswpin_delta=0 pswpout_delta=0   (sampled across the WHOLE window)
GATE idle_pct_after=99
GATE VERDICT=QUALIFIED
```

- `VERDICT=QUALIFIED` — the throughput figures are usable.
- `VERDICT=DEGRADED` — the kernel wrote pages out but nothing stalled reading them back.
  Throughput figures are soft; hit rate and byte figures are unaffected.
- `VERDICT=DISQUALIFIED` — pages were read back from swap during the window. **Every
  throughput figure in that run is void.** Hit rate, occupancy and the byte figures are
  *not* timing quantities and survive it.
- `VERDICT=REFUSED` — the machine never got quiet enough to start. Nothing ran. This is
  the gate doing its job, not a failure.
- `VERDICT=SUBSTRATE_KILL` — exit 137, meaning the kernel killed the job, almost certainly
  out of memory. **Do not rerun it unchanged.** Record `free -m` and `df -h`, lower the
  table sizes or the budget, and note it. A retry under memory pressure makes the pressure
  worse.

**2. Did the trace actually fill anything?** In the summariser's substrate output, check
`trace_distinct_set_keys` against the table sizes you swept. If the largest table you swept
has more slots than the trace has distinct keys, that table never filled and its eviction
rows are all measuring the same thing. That is fine and expected at the sparse end of the
curve — it is the *point* of sweeping a range — but it means those rows say nothing about
eviction.

**3. Read occupancy, always, beside the hit rate.** The summariser prints them in the same
row for exactly this reason. Two hit rates at different occupancies are not comparable.

---

## Reference: the config keys, and what each refuses

You do not need this to run the sweep — the driver script writes the configs for you — but
you will need it to put a chosen setting into your real `.cfg`, and KataGo **refuses**
incoherent combinations rather than ignoring them, so knowing why is useful.

```
nnCacheCollision = direct | linearprobe | quadraticprobe | chain
nnCacheWays      = 2..1024        (probing schemes only)
nnCacheEviction  = random | lru | lfu
nnCacheAdmission = always | secondsighting
nnCacheMaxBytes  = <bytes>        (chain only; required there)
```

Refusals you may hit, and what each means:

| you wrote | what happens | why |
|---|---|---|
| `direct` with `nnCacheEviction` or `nnCacheWays` or `nnCacheMaxBytes` | refused | a 1-way table gives a collision exactly one candidate victim; there is no policy to choose and no associativity to set |
| `linearprobe`/`quadraticprobe` with no `nnCacheEviction` | refused | a probed table has several candidates and the choice must be stated |
| `nnCacheWays` larger than `2^(nnCacheSizePowerOfTwo - nnMutexPoolSizePowerOfTwo)` | refused, naming all three keys | a position's whole probe sequence is confined to one lock region so that a lookup still takes exactly one lock; that caps the ways |
| `chain` with no `nnCacheMaxBytes` | refused | a chained table has no fixed capacity and is bounded only by its byte budget |
| `chain` with `nnCacheEviction = none` | **refused, and this is new** | see below |
| `nnCacheMaxBytes` too small to hold one entry per lock region | refused, with the arithmetic | a table that can never retain anything is broken, not slow |

**The one accepted value that was removed.** Before this branch, `chain` *required*
`nnCacheEviction = none`. That was wrong: a chained table's byte budget is always enforced,
so an entry *is* always given up when a region goes over — the order was recency, in force,
under a key that asserted no policy was chosen. `nnCacheEviction` under `chain` now names
that capacity victim and accepts `random`, `lru` and `lfu`. `none` is refused rather than
quietly reinterpreted, and the refusal message says so and names `lru` as the value that
reproduces the old behaviour exactly.

**If you have an existing config saying `nnCacheCollision = chain` and
`nnCacheEviction = none`, change `none` to `lru` and nothing else changes.** Omitting the
key entirely also gives you `lru`.

---

## If something goes wrong

| symptom | most likely cause | what to do |
|---|---|---|
| cmake fails at `enable_language(CUDA)` | no `nvcc` on `PATH` | a host toolchain matter, not this branch; try `-DUSE_BACKEND=TENSORRT` |
| `"$KATAGO" version` does not mention a GPU backend | you built or are running the wrong binary | check which `build-*` directory `$KATAGO` points at |
| capture produces a zero-byte or tiny `trace.bin` | the engine failed early | read `run1/capture-engine.log`; the engine's errors are on stderr, and this is what that file is |
| the engine prints "NNCache: KATAGO_NNCACHE_TRACE is set..." and then is slow | expected | tracing serialises every cache operation through one lock. It is a capture run. Do not read timings from it |
| stage 2 refuses with a memory message | the matrix does not fit your budget | the message contains the arithmetic; drop the largest `-p` entry, or use `-W off`, or raise `-b` |
| stage 2 rows say `REFUSED` | a shape is not honourable at that table size | this is deliberate — the row records *which* shape and *why*, so a hole in the matrix is never mistaken for an unrun cell. The commonest is `ways` exceeding a lock region at small table sizes |
| a benchmark in stage 3 fails immediately | a config key was refused | the refusal names what to change; it is at the end of `run1/visits/<name>.stderr` |
| exit 137 anywhere | the kernel killed it, presumptively out of memory | stop. Record `free -m` and `df -h`. Do not retry unchanged |

---

## What this runbook does not cover, and will not pretend to

- **It has never been run.** See the top of this document.
- **Whether your host's THP setting reproduces the 17–28% measured in the guest.**
  UNEXERCISED, and specifically *not* claimed, because the guest's memory is backed by
  host hugepages and the two are different questions.
- **Multi-threaded cache contention under any policy.** Nothing here measures it. The
  replay is single-threaded by design, and `katago benchmark` measures visits/s under
  threads without attributing any of it to the cache.
- **Whether a hit rate measured by replaying a recorded stream equals the hit rate a live
  multi-threaded search would get.** Replay is exact for a single-threaded fixed-visit
  search, because a cache hit returns the identical evaluation a miss would have computed
  and so cannot change which position is visited next. Under threads the operation order is
  already nondeterministic run to run. So a replayed hit rate is a claim about the recorded
  workload; its transfer to your live engine is an inference.
