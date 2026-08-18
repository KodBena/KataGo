# Frozen read-only NN cache — behavioural specification

## What this document is, and who it is for

This is the complete behavioural specification for a **frozen read-only neural-net output
cache**: a structure that is built once from a fixed set of position hashes together with
their evaluations, is never modified afterwards, and answers the single question *"do you
hold an evaluation for this position, and if so which one"* as fast as possible.

It exists because a working prototype of this structure lives in an archive the implementer
of the new version is **not permitted to read**. The operator's reason is that he does not
want the prototype's internal shape reproduced — only its behaviour. This document and the
frozen vectors beside it are therefore the implementer's *only* source. If something is not
in here, it is not a requirement; if you find yourself needing to look the prototype up, that
is a defect in this document and it should be reported rather than worked around.

The structure the operator has asked for is a **CHD (compress-hash-displace) minimal perfect
hash**, and this specification is written for that. Note in passing that the acceptance
criteria below are stated in terms of observable behaviour and measured speed, so they do not
themselves depend on CHD being the chosen construction. That is a property of how the
criteria are written, not an argument against the choice.

**Terms used throughout.** A *key* is a 128-bit position hash, made of two 64-bit halves
named `hash0` and `hash1`. An *entry* is one (key, evaluation) pair held by the structure. The
*key set* is the ordered list of keys the structure was built from; *entry i* is the entry at
position `i` of that list, counting from zero. A *member* query is a lookup for a key that is
in the key set; an *absent* query is a lookup for any other key. A *minimal perfect hash* is a
function that maps the n keys of a known key set onto the integers 0..n−1 with no two keys
sharing a value — and, critically for §2, says nothing useful about keys outside that set.

---

## 0. The acceptance bar, in one place

The operator has ruled that parity with the prototype is **behavioural**, and explicitly
*weaker* than key-level. In his words: *"behavioral parity, might not even need to be
key-level parity. Basically, it should work and it should not be worse in lookup
performance."*

That is the whole bar, and it has exactly two halves:

1. **It works** — the behaviour specified in §1–§6 below, demonstrated by the vectors in
   `cpp/spec/chd/vectors/` via the procedure in §7.
2. **It is not slower** — the lookup-latency floor in §8.

Everything the prototype does that is not named as CONTRACT below is yours to decide. In
particular you are **free to choose different internal parameters, a different search
strategy, different seeds, and a different assignment of keys to internal slots**, and doing
so is conforming, not a deviation.

Each property in this document is labelled:

- **CONTRACT** — a caller can observe it; you must match it.
- **INCIDENTAL** — the prototype does this, no caller can observe it, and you may do
  otherwise. Recorded so you know it was considered and released, not overlooked.

---

## 1. Construction

### 1.1 Inputs — CONTRACT

Construction takes, in one call:

- **An ordered list of keys**, length n ≥ 0. Order is supplied by the caller and is
  meaningful (§1.3).
- **An ordered list of evaluations**, of the same length n, positionally aligned: the
  evaluation at position i belongs to the key at position i.

The caller in the prototype's system derives both lists from a database query that selects the
positions worth caching for one flash-card and orders them by descending observed reference
count. Nothing about that selection is your concern; you receive the lists already ordered.

**Precondition, CONTRACT: the two lists must be the same length.** The prototype does not
check this and can read past the end of the shorter list. Do not reproduce that. If the
lengths differ, refuse (§5).

**Precondition, CONTRACT: keys must be pairwise distinct.** See §4.

### 1.2 Outputs — CONTRACT

A structure that supports the query operations of §2 and §3 and is thereafter immutable
except for the per-entry hit counter of §3.

### 1.3 Entry order — CONTRACT

**Entry i is the pair supplied at position i of the input lists, for every i in 0..n−1.** A
member query for the key at input position i must resolve to entry i, and reading entry i must
yield the evaluation supplied at input position i.

This is contract for two reasons, both of them things a caller does:

- The caller pairs a returned entry with its own parallel arrays by index.
- The caller iterates entries **in index order**, 0 upward, when it harvests per-entry hit
  counts at the end of a session (§3.2). Because the input is ordered by descending observed
  reference count, index order *is* descending-popularity order, and the caller relies on that
  when it decides what to keep.

What is **INCIDENTAL** is any internal slot numbering. You may lay entries out in memory in
any order you like; only the index the caller sees is fixed.

### 1.4 Ordering as a performance hint — INCIDENTAL, with a note

The input arriving in descending-popularity order was chosen by the prototype's author to get
memory locality: the most-queried entries end up adjacent at low indices. It measurably works —
under an access pattern where 90% of queries hit the first 10% of entries, the prototype's
lookup got 40% faster. But *you* are not required to exploit it, and no test checks that you
do. It is a property of the input, not an obligation on you. It is recorded here only so that,
if you are choosing a memory layout, you know that laying entries out in index order is
worth something.

The ordering is also not even fully determined: the query that produces it orders by reference
count with **no tie-break**, so keys with equal counts arrive in an arbitrary order that can
differ between two runs over identical data. Anything that depended on the exact order would
therefore be depending on something the prototype itself does not pin down.

### 1.5 Sizing constants — INCIDENTAL

The prototype groups keys four to a bucket, sizes its position table at 1.1× the key count
plus one, and places larger buckets before smaller ones. These are tuning choices with no
externally observable consequence beyond speed and memory; §8 and §6 constrain those directly.
Choose your own.

### 1.6 Construction cost — not a contract, but a budget

Not an acceptance criterion. For planning only, measured on the prototype on this class of
machine: construction is mildly superlinear at roughly n^1.21, running about 165 ns per key at
n = 10,000 and about 500 ns per key at n = 2,000,000. For the corpus sizes that actually occur
(§8.1) that is single-digit to low-double-digit milliseconds per card. If your construction is
in that neighbourhood it is fine; if it is orders of magnitude worse, that is worth raising
before you go further, because a bulk pre-warm builds one of these per card.

---

## 2. Lookup, and the absent-key contract

**This section is the one most likely to be got subtly wrong, and its failure mode is
silent.** Read it before writing anything.

### 2.1 The trap

A minimal perfect hash is a function over a *known* key set. Feed it a key that was never
inserted and it does not report an error and does not return "nothing" — it happily computes
*some* position, because computing a position is all it does. That position will, most of the
time, be a position holding a completely unrelated real entry.

How often is "most of the time"? It follows from the sizing: with a position table around 1.1
times the key count, roughly n/1.1n ≈ 91% of positions are occupied, so roughly **91% of
never-inserted keys resolve onto some real entry's position.** That is not a corner case. An
implementation that trusts the computed position will return the wrong evaluation for about
nine out of every ten absent queries.

That 91% is arithmetic on the sizing; the *observed* rate is a little higher. Across the
vector cases in `cpp/spec/chd/vectors/`, the `absent_landing_on_occupied_slot` column of
`DIAGNOSTIC-internals.tsv` records, per case, how many never-inserted query keys resolved onto
an occupied position in the prototype. On every case with at least a thousand absent queries
it runs between **90.9% and 95.3%** — including 92.5% on the real median-sized card corpus
(36,987 of 40,000) and 95.3% on the one-bit-perturbation case, which is the worst of them
because a near-miss key is disproportionately likely to land where its near-neighbour lives.

And in this application, returning the wrong evaluation is worse than returning nothing: the
caller takes the answer as a cached neural-net evaluation of a Go position and searches on it.
A miss costs one evaluation. A wrong hit corrupts a search.

### 2.2 The obligation — CONTRACT

**Every lookup must verify that the key stored at the resolved entry equals the key that was
asked for, before reporting a member.** There must be no path by which a computed position is
believed without that comparison — not for a "fast path", not behind a flag, not when the
caller "knows" the key is present.

This obliges you to store the full 128-bit key of every entry, or something from which the
full key can be recovered exactly. A fingerprint short enough to admit collisions is not
sufficient: a collision here is exactly the silent wrong-evaluation failure above. (For scale:
the prototype spends 16 bytes per entry on this and it is 39% of its total per-entry
footprint. It is not optional and §6's budget already includes it.)

You must also **bounds-check the resolved position before using it.** A computed position can
be out of range — the prototype's own table contains a sentinel meaning "nothing here", and
using it as an index without checking reads out of bounds. Whatever your representation, a
resolved position that does not name a real entry must be treated as absent, not dereferenced.

### 2.3 Lookup semantics — CONTRACT

For a key q:

| case | result |
|---|---|
| q equals the key of entry i | **member**, entry i; the caller can obtain the evaluation supplied at input position i |
| q is not any entry's key | **absent**; no evaluation is produced, and no entry's state is disturbed |
| the structure was built from an empty key set | **absent**, always, for every q |

A lookup never fails, never throws, and never blocks on anything a concurrent lookup is doing
(§3.3). "Absent" is a normal answer, not an error.

### 2.4 Miss cost — CONTRACT (weak)

Rejecting an absent key must be **cheap** — the same order of work as a member lookup, not a
scan. The prototype rejects in one position computation and one key comparison, and its
measured miss path is no slower than its hit path. Any implementation whose absent-key cost
grows with n has misunderstood §2.2.

---

## 3. The other operations the caller uses

### 3.1 Membership without retrieval — CONTRACT

A caller must be able to ask *is this key present* without obtaining the evaluation and
without disturbing the entry's hit counter. Same answer as §2.3 minus the payload.

### 3.2 Per-entry hit counting — CONTRACT

Each entry carries an unsigned counter.

- Every counter starts at **zero** when the structure is built. It is *not* seeded from any
  historical count the database may hold; the counter measures this session only, and the
  caller adds it to the historical total itself when it writes back.
- A lookup that **retrieves** an evaluation increments that entry's counter by one.
- A membership-only test (§3.1) and a counter read do **not** increment.
- There must be a way to read an entry's counter without incrementing it.
- There must be a way to add an arbitrary amount to a key's counter without retrieving the
  evaluation. The caller uses this to fold in counts from elsewhere.
- Counter overflow is not specified because it does not arise: the largest lifetime count in
  the operator's entire database is 11,997, against a 32-bit counter in the prototype.

The counters are the feedback loop that decides what gets cached next session, so an
implementation that skipped them would work and would silently degrade the operator's cache
policy over time.

### 3.3 Concurrency — CONTRACT

Lookups run from many search threads at once. The structure must be safe for **unlimited
concurrent lookups with no external locking**, including the counter increments of §3.2. The
counters need no ordering guarantee beyond not being lost or torn — a relaxed atomic increment
is what the prototype uses and is sufficient.

There is no concurrent mutation to worry about: the structure is built by one thread before
any lookup happens, and never changes afterwards.

**INCIDENTAL:** the prototype hands out all its evaluations as pointers sharing a *single*
reference count over one contiguous allocation. That turns out to be faster than the
alternative on this hardware — one always-hot cache line instead of a scattered refcount per
entry — but it is a representation choice, and it is a choice whose behaviour at high core
counts is unmeasured above four threads. Do what you like here; §8 is the only constraint.

### 3.4 Harvesting — CONTRACT

At the end of a session the caller walks **all entries in index order** and reads each
counter, to decide what to persist. Two consequences you must honour:

- Entries must be enumerable in index order, 0..n−1, including entries never looked up.
- Entry i's key, evaluation and counter must all be reachable from index i and must agree —
  in particular the evaluation stored at entry i must be the one whose position hash is the
  key at entry i. The prototype relies on this without checking it; if you can make the two
  impossible to separate, do.

The caller then sorts what it harvested by counter, descending, and takes a prefix. **Tie
order among equal counters is unspecified** and no caller depends on it — the prototype's own
tie order is not even deterministic between runs.

---

## 4. What key sets are constructible

This is the part of the contract most likely to surprise you, and it was found by running the
prototype rather than reading it.

### 4.1 Distinctness — CONTRACT

**Duplicate keys are not constructible.** If the same key appears twice in the input,
construction must be refused (§5). It cannot be silently deduplicated: the caller's index
alignment (§1.3) assumes n entries for n inputs.

WITNESSED on the prototype: a 101-key list whose position-7 key is repeated at the end is
refused. The vector case `dup_exact` is that list.

### 4.2 A second, non-obvious refusal — CONTRACT for *your* construction, INCIDENTAL as a mechanism

The prototype also refuses certain key sets in which **all keys are distinct**. Specifically,
it derives a key's position from `hash1` alone once the key's bucket is fixed by `hash0`, so
two *distinct* keys that share a `hash1` value and land in the same bucket collide for every
seed it can try, and construction is refused. WITNESSED: a two-key set differing only in
`hash0` is refused; a 64-key set all sharing one `hash1` is refused.

**How to read this.** The specific precondition — "within a bucket, `hash1` values must be
distinct" — is a consequence of the prototype's particular hash use and is **INCIDENTAL**; you
are not required to have the same weakness, and if your construction accepts those key sets,
that is strictly better and is conforming. Two vector cases (`dup_hash1_same_bucket`,
`hash1_degenerate_corpus`) record the prototype's refusal; they are marked in the vector
manifest as **PROTOTYPE-SPECIFIC** and the checker does not require you to match them.

What **is** contract is the general shape: **if your construction cannot produce a valid
structure over the key set it was given, for any reason, it says so loudly and produces
nothing (§5).** It never returns a structure that answers some keys wrongly.

Whether this can bite on real data: the operator's keys come from a Zobrist-style mix and
behave indistinguishably from uniform random draws, and no such refusal was observed on either
real corpus in the vectors. It is a theoretical hazard on real input, not an observed one.

### 4.3 Empty and singleton key sets — CONTRACT

n = 0 must construct successfully and answer absent to everything (§2.3). It must not be an
error, because the caller hits it routinely: a card with no cacheable positions yet.

n = 1 must construct and behave normally.

### 4.4 Scale — CONTRACT (weak)

The largest real per-card key set in the operator's database is 291,129. Construction must
work up to at least 2,000,000 keys, which is where the prototype was measured. Above about
4×10^9 keys the prototype's 32-bit internal arithmetic overflows; you are not required to
support that, but you are required not to *silently* misbehave there (§5).

---

## 5. Failure — loudly, always

**CONTRACT.** Every one of the following is a construction failure. A construction failure
must:

- **abort the construction and yield no structure** — never a partially built one, never one
  that answers some keys and not others;
- **raise an exception (or otherwise transfer control to the caller unmistakably)** carrying a
  message that names *what* failed and, where applicable, *which* key or key set position was
  involved;
- **never be reported only by a log line, a return code the caller can ignore, or a flag on
  an otherwise-usable object.**

The failure list:

| condition | |
|---|---|
| duplicate key in the input (§4.1) | refuse |
| key list and evaluation list lengths differ (§1.1) | refuse |
| the search for a valid arrangement is exhausted (§5.1) | refuse |
| any internal self-check fails (§5.2) | refuse |
| key count exceeds what the implementation's arithmetic supports (§4.4) | refuse |

### 5.1 Search exhaustion — CONTRACT

However you place keys, your construction may involve a search that can fail. **That search
must have a bound, reaching the bound must refuse construction, and the refusal must name the
bound.** It must not loop forever, and it must not fall back to a degraded structure.

The prototype's specific bound is 65,535 attempts per bucket. **That number is INCIDENTAL** —
it is a consequence of how it stores what it found, not a requirement. Pick your own bound.
What is contract is that a bound exists and that hitting it is loud.

For calibration, so you can tell a sane bound from a paranoid one: across 60 measured builds
spanning key counts from 10,000 to 2,000,000, the prototype's *hardest* bucket anywhere needed
1,859 attempts, against its ceiling of 65,535 — a factor of about 35 of headroom — and it
never once failed. The mean cost was about 38.9 attempts per bucket, and that mean was flat to
within 0.3% across a 200× range of key counts. Exhaustion is not a practical risk at these
sizes; the requirement is that if it ever happens, the operator hears about it.

### 5.2 Self-verification — CONTRACT

**Before returning, construction must verify its own result: every key in the input must
resolve to its own entry.** If any does not, refuse (§5).

The prototype does this and it costs a few percent of build time. Keep it. It is the check
that makes a construction bug loud instead of silent, and this whole subsystem's worst failure
mode is silence.

### 5.3 What the *caller* does with a refusal — CONTRACT on the integration

The prototype's caller catches the refusal, prints it to stderr, and carries on with an empty
frozen cache — so a failed construction degrades to "no cache for this card" with nothing but
a log line to show for it. **Do not reproduce that.** A construction that was expected to
succeed and did not is a real defect and must reach the operator: at minimum it must be
distinguishable, at the call site, from the legitimate empty case of §4.3.

---

## 6. Memory footprint

**CONTRACT (budget, not exact).** Resident bytes per entry for the *structure* — everything
except the evaluations themselves — must not exceed **48 bytes per entry**, and transient
peak during construction must not exceed **60 bytes per entry**.

Those ceilings are the prototype's measured figures (40.9 resident, ~49.3 peak) with headroom.
The measured breakdown, so you can see where it goes and what is compressible:

| component | B/entry | |
|---|---|---|
| the perfect-hash machinery itself | 4.9 | your business, likely compressible |
| the stored 128-bit keys | 16.0 | **required by §2.2** |
| the per-entry hit counter | 4.0 | required by §3.2 |
| a handle to each evaluation | 16.0 | representation-dependent |
| **total resident** | **40.9** | |

For context on whether this matters: at the median real corpus size of 45,664 entries that is
about 1.9 MB of structure, against about 70 MB for the evaluations themselves. The structure
is under 3% of the cost. Do not contort the design to shave it.

**A warning about how you report this figure.** The prototype has a method that reports its
own memory use, and it returns 4.9 B/entry — it counts only the first row of that table,
because the other three live on a different object. It **under-reports resident structure by
8.3×**. If you provide a self-reported memory figure, make it cover everything, or do not
provide one.

**INCIDENTAL:** the prototype allocates one contiguous block for all n evaluations up front,
sized to the *requested* key count rather than the count actually found in the backing files,
and never releases the difference. Where the two differ, the excess is held for the lifetime
of the structure. You need not copy this.

---

## 7. Conformance vectors and how to check against them

The vectors are in `cpp/spec/chd/vectors/`. They were produced by **running** the archived
prototype, not by transcribing what its code appears to do. The generator that ran it is
`cpp/spec/chd/tools/gen_vectors.cpp`; it reaches into the archive and you must not build or
run it.

### 7.1 What a case is

Each case `NAME` consists of:

- `NAME.meta` — tab-separated fields; `outcome` is `BUILD_OK` or `BUILD_FAIL`.
- `NAME.keys` — the key set, one key per line, in input order. Line 1 is entry 0. Two
  lowercase 16-digit hex fields separated by a single space: `hash0` then `hash1`. A file
  with zero lines is the empty key set.
- `NAME.queries` — the lookups to perform, same line format, in order. (`BUILD_OK` only.)
- `NAME.expected` — one line per query, in the same order: `MEMBER <i>` or `ABSENT`.
  (`BUILD_OK` only.)

`MANIFEST.tsv` lists every case with its sha256, the provenance of its key set, and whether it
is an acceptance case or a prototype-specific one you are not required to match.

`DIAGNOSTIC-internals.tsv` records internals of the prototype's construction — bucket counts,
table sizes, how many absent queries landed on an occupied position. **It is not an acceptance
artifact and nothing checks it.** It is there so that, while developing, you can see whether
your absent-key queries are actually exercising the verification path or trivially missing.

### 7.2 The comparison procedure

Build one executable — the *driver* — that wraps your implementation and accepts exactly two
invocations:

```
<driver> build-and-query <keys-file> <queries-file>
```
Build the structure over the keys in file order (line 1 → entry 0). For each query in order,
print exactly one line to stdout: `MEMBER <i>` if the query key is the key of entry i, or
`ABSENT` otherwise. Print nothing else to stdout. Exit 0.

```
<driver> build-only <keys-file>
```
Attempt construction only. Exit 0 on success; on refusal exit non-zero **and** write a
diagnostic naming the reason to stderr.

Then run:

```
python3 cpp/spec/chd/tools/check_vectors.py <driver>
```

It reports one line per case and exits 0 only if every **graded** case passed. Cases marked
`PROTOTYPE-SPECIFIC` in `MANIFEST.tsv` are run and reported but are **advisory** — they record
the prototype refusing key sets whose keys are all distinct (§4.2), and an implementation that
constructs them successfully is better, not worse. There are two of them; the other 21 cases
are graded.

For a `BUILD_OK` case it
requires your stdout to match `NAME.expected` **line for line, exactly**. For a `BUILD_FAIL`
case it requires non-zero exit and a non-empty stderr — a refusal that says nothing is not an
acceptable refusal (§5).

The evaluations are not part of the vectors, because the vectors test resolution, not payload
carriage. §3.4's key/evaluation agreement is yours to test in your own unit tests.

### 7.3 Convincing yourself the check is not vacuous

```
python3 cpp/spec/chd/tools/check_vectors.py --self-test
```

runs four stub drivers. Three are deliberately broken and must each be **rejected**, each for
a different clause: one answers `ABSENT` to everything (fails the member leg), one answers
`MEMBER 0` to everything (fails the absent-key leg), and one replays the correct answers but
never refuses a construction (fails the refusal leg). The fourth is a positive control that
is correct on every graded clause and must be **accepted** — without it, a checker that
rejected everything would look identical to a working one.

If any of the first three passes, or the fourth fails, the checker is not observing what it
claims to and the failure is in the checker, not in you.

Witnessed on this corpus at the time the vectors were frozen: 17/21, 20/21 and 1/21 graded
cases rejected for the three broken stubs respectively, and 0/21 for the positive control.

### 7.4 The vectors are not the whole bar

Passing every vector shows resolution is right. It does not show §3's counters, §3.4's
harvest order, §6's footprint or §8's speed. Those need your own tests and the measurement in
§8.

---

## 8. The performance floor

**CONTRACT, and an acceptance criterion.** Single-threaded lookup latency must **not exceed**
the prototype's, at the corpus sizes that actually occur.

### 8.1 The sizes that occur

Per-card key sets in the operator's real database, measured over the 251 cards that have any:
minimum 1,203; 10th percentile 9,592; **median 45,664**; 75th percentile 75,822; 90th
percentile 121,796; maximum 291,129. Sessions build one structure per card rather than one
over all of them, so the per-card figures are the relevant ones and 291,129 is the practical
ceiling.

### 8.2 The floor

Measured on the prototype, single-threaded, on this class of machine, retrieving an evaluation
(not membership-only), all queries hitting resident keys:

| key count | prototype ns/op |
|---|---|
| 4,096 | 13.255 |
| 32,768 | 27.526 |
| 262,144 | 73.884 |
| 524,288 | 109.745 |

Multi-thread figures at the same sizes exist and are in `chd-cost.wiki` §2.2 if you need them;
they are noisy above two threads and are not part of the floor.

**Interpolating onto the real sizes**, and this is arithmetic on the table above rather than a
new measurement: the median 45,664 sits between the 32,768 and 262,144 rows, the 90th
percentile 121,796 likewise, and the maximum 291,129 sits just above 262,144. So the floor you
must meet in practice is roughly **30–35 ns/op at the median and roughly 80 ns/op at the
largest card.**

Two things to know about those numbers before you measure against them. They were taken with
a 1,528-byte stand-in for the evaluation rather than a real one, at the same size and with the
key at the same offset, so the memory behaviour is faithful but the payload contents are not.
And they came from a verified-idle window on a contended 4-core box; the dispersion in the
original table is real and is reported there rather than smoothed.

### 8.3 How to compare

- **Same machine, same session, both arms in one process** wherever you can — build both
  structures over the same key set in one binary and alternate the measured loops. If you
  cannot have both arms (you cannot: you may not build the prototype), compare against the
  table above and say so explicitly, because a cross-session comparison against a published
  number is materially weaker evidence than a paired one and the report should not pretend
  otherwise.
- **Corpus sizes: 4,096 / 32,768 / 45,664 / 121,796 / 262,144 / 291,129.** The first, second
  and fifth line up with the published table; the others are the real percentiles.
- **Queries drawn uniformly from the resident key set**, one million per repetition, at least
  15 repetitions, reporting mean, standard deviation, min and max — not the mean alone. The
  published table's dispersion at one point is 28% of its mean, driven by a single repetition
  in fifteen.
- **Idle-gate the run.** Take `flock /home/bork/kg/audit-reports/.timing.lock` and verify the
  machine is actually quiet from inside the lock; on this box the 1-minute load average was
  measured to be unable to tell a contended run from an idle one. One lock hold for the whole
  sweep is better than one per point.
- **State a detection floor** so a null result means something. With 15 repetitions at the
  dispersions in the published table, a difference below roughly 10% at the larger sizes is
  not distinguishable from noise. Say that, rather than reporting "no difference".
- Also report the **absent-key** path at the same sizes (§2.4). It is not part of the floor
  but a pathological miss path is a defect the hit-path numbers will hide.

### 8.4 What is *not* required

You are not required to be *faster*. You are not required to match the prototype's shape of
degradation with corpus size, nor its multi-threaded scaling, nor its cache-miss or
instruction counts. The floor is single-threaded hit latency at the sizes in §8.1, and
nothing else.

---

## 9. Explicitly out of scope

- **Reading the prototype's files.** Format compatibility with the existing `.nncache`
  containers and the compressed archives is **not** required — see §10. You are building an
  in-memory structure; nothing in this specification is serialized to disk.
- **The database.** How the key set is chosen, how counters are written back, what a card is:
  all caller concerns.
- **Where the evaluations come from.** You receive them.

---

## 10. A note on the archives, recorded once

The operator ruled that file-format parity is not required. One consequence is worth having on
the record because it was visible at the time the decision was made: format parity is what
would have let new code read his **existing** archives, and those archives hold 25% of his
cached entries — entries that carry **50.71% of all lifetime references** in his database.
Discarding readability of them means re-earning that content by re-evaluating the positions.
He has accepted that. It is noted here, not re-argued.

A related fact that makes the decision cheaper than it sounds: **the prototype never
serialized this structure at all.** Its files hold evaluations; the perfect hash is rebuilt in
memory every time a card is loaded. So there is no on-disk format of *this* structure to be
compatible with, and the archive question is entirely about the separate loader that reads
evaluation files — a different component, and not this one.

---

## 11. Everything classified, in one table

| property | class | why |
|---|---|---|
| entry i is the caller's input position i | CONTRACT | caller indexes and iterates by it |
| member query resolves to the right entry | CONTRACT | the point of the structure |
| **absent key must be rejected by stored-key comparison** | **CONTRACT** | **~91% of absent keys land on a real entry; failing this returns a wrong evaluation, silently** |
| resolved position is bounds-checked before use | CONTRACT | out-of-range positions are reachable |
| empty key set constructs and answers absent | CONTRACT | routine caller case |
| duplicate keys refuse construction | CONTRACT | index alignment assumes n entries |
| any construction that cannot complete refuses, loudly, yielding nothing | CONTRACT | ADR-0002; the alternative is a structure that lies |
| construction self-verifies before returning | CONTRACT | makes a construction bug loud |
| search bound exists and hitting it refuses | CONTRACT | no unbounded loop, no degraded fallback |
| hit counters: start at zero, increment on retrieval only, readable, addable | CONTRACT | drives the operator's cache policy |
| entries enumerable in index order | CONTRACT | end-of-session harvest |
| lock-free concurrent lookup | CONTRACT | many search threads |
| ≤48 B/entry resident, ≤60 B/entry peak | CONTRACT | budget with headroom over measured 40.9/49.3 |
| single-thread hit latency ≤ §8.2 at §8.1 sizes | CONTRACT | the operator's second acceptance half |
| **deterministic sequential seed scan** | **INCIDENTAL** | **no caller observes which seed was chosen; key-level parity is explicitly not required** |
| the specific mixing and range-reduction functions | INCIDENTAL | internal |
| four keys per bucket; 1.1× table sizing | INCIDENTAL | tuning |
| largest-bucket-first placement | INCIDENTAL | tuning |
| 65,535 as the search ceiling | INCIDENTAL | a consequence of its storage width; only the *existence* of a bound is contract |
| internal slot numbering / memory layout | INCIDENTAL | only the caller's index is fixed |
| one shared reference count over one block | INCIDENTAL | representation; §8 constrains the effect |
| refusal on distinct keys sharing hash1 in a bucket | INCIDENTAL | a weakness of one hash scheme; accepting them is better |
| tie order among equal hit counts at harvest | INCIDENTAL | not deterministic in the prototype either |
| descending-popularity input order as a locality win | INCIDENTAL | a property of the input, not an obligation |

---

## 12. Where the numbers in this document come from

Every measured figure quoted here was produced by earlier work on this programme and is not
re-derived:

- Lookup latencies (§8.2), build cost (§1.6), seed-search statistics (§5.1), the 40.9 and
  ~49.3 B/entry footprints and the 8.3× self-report defect (§6), the locality result (§1.4):
  `/home/bork/kg/audit-reports/chd-cost.wiki`.
- Per-card corpus sizes (§8.1), the archive share and its reference weight (§10), the
  11,997 maximum reference count (§3.2): `/home/bork/kg/audit-reports/cache-corpus-stats.wiki`.
- Real keys behaving indistinguishably from uniform (§4.2):
  `/home/bork/kg/audit-reports/hash-uniformity.wiki`.
- The absent-key landing rate (§2.1), the construction refusals (§4.1, §4.2) and every
  expected answer in the vectors: observed in this pass by running the prototype; the run is
  reported in `/home/bork/kg/audit-reports/chd-spec.wiki`.
