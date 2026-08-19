#ifndef NEURALNET_NNCACHEFROZEN_H_
#define NEURALNET_NNCACHEFROZEN_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "../core/hash.h"
#include "../neuralnet/nncache.h"
#include "../neuralnet/nninputs.h"

// LEVEL 0 of the two-level cache: a frozen, read-only structure built once from a fixed
// set of positions and their evaluations, never modified afterwards, answering "do you
// hold an evaluation for this position, and which one" as fast as a perfect hash can.
//
// It is built ENTIRELY IN PROCESS, from a supplied vector of evaluations. There is no
// file format, no loader, no schema and no serialization anywhere in this header or its
// implementation, deliberately: persistence is a separate, deferred piece of work, and a
// container format invented here to make level 0 loadable would be that work done badly
// and in the wrong place.
//
// The specification this implements is cpp/spec/chd/SPEC.md; section references below are
// to it. It was written from that document and the frozen vectors beside it and from
// nothing else.

//-------------------------------------------------------------------------------------
// The perfect hash itself
//-------------------------------------------------------------------------------------

// A frozen CHD (compress-hash-displace) perfect hash over a fixed, ordered key set: it
// resolves a 128-bit position hash to the caller's own input position for that key, or to
// absent.
//
// THE ABSENT-KEY CONTRACT (SPEC.md 2.2), which is the whole reason find() is written the
// way it is. A perfect hash is a function over a KNOWN key set. Handed a key that was
// never inserted it does not fail and does not return nothing -- it computes some slot,
// because computing a slot is all it does, and at this structure's occupancy roughly nine
// out of ten never-inserted keys land on a slot holding a real, unrelated entry. So find()
// compares the FULL 128-bit key stored at the resolved slot against the key asked for, and
// there is no path around that comparison: no fast path, no flag, no "the caller knows it
// is present". Omitting it would return one position's neural-net evaluation for a
// different position, silently, into a running search -- a miss costs one evaluation, a
// wrong hit corrupts a search.
//
// Immutable after build(), and therefore safe for unlimited concurrent lookups with no
// locking at all (SPEC.md 3.3): find() only reads.
class NNCacheFrozenIndex {
 public:
  // Builds the index over `keys` IN THE GIVEN ORDER: the key at position i is entry i, and
  // find() returns exactly that i (SPEC.md 1.3). n = 0 is legal and yields an index that
  // answers absent to everything (SPEC.md 4.3).
  //
  // Throws StringError, naming what failed and which input position was involved, if the
  // key set is not constructible: duplicate keys (SPEC.md 4.1), a key count past what this
  // implementation's 32-bit slot arithmetic supports (SPEC.md 4.4), a displacement search
  // that exhausts its bound (SPEC.md 5.1), or a self-verification failure (SPEC.md 5.2).
  // It yields NO structure in any of those cases -- never a partially built one, and never
  // one that answers some keys wrongly (SPEC.md 5).
  //
  // Refusal is by exception rather than by a returned error type. That is what SPEC.md 5
  // asks for ("raise an exception, or otherwise transfer control to the caller
  // unmistakably") and what every other construction refusal in this cache already does;
  // the std::expected shape ADR-0012 P9 rule 5 would prefer is a C++23 type and this tree
  // is built at C++17.
  static NNCacheFrozenIndex build(std::vector<Hash128> keys);

  NNCacheFrozenIndex(NNCacheFrozenIndex&& other) = default;
  NNCacheFrozenIndex& operator=(NNCacheFrozenIndex&& other) = default;
  NNCacheFrozenIndex(const NNCacheFrozenIndex& other) = delete;
  NNCacheFrozenIndex& operator=(const NNCacheFrozenIndex& other) = delete;

  // The caller's input position for `key`, or nullopt if this index does not hold it.
  // Absence is a normal answer and not an error (SPEC.md 2.3). Costs one bucket hash, one
  // slot hash, one indirection and one 128-bit comparison, whether it hits or misses --
  // rejecting an absent key is the same order of work as finding a present one and never
  // a scan (SPEC.md 2.4).
  [[nodiscard]] std::optional<uint32_t> find(Hash128 key) const;

  uint32_t numEntries() const { return (uint32_t)records_.size(); }
  // Entry i's key. Entries are enumerable in index order 0..n-1 through this (SPEC.md 3.4).
  Hash128 keyAt(uint32_t i) const { return records_[i].key; }

  // Entry i's auxiliary 32-bit state word, which this class stores and does NOT interpret.
  //
  // It lives here, in the SAME record as the key, for a measured reason rather than a
  // tidiness one: whoever owns an entry's mutable state has to touch it on every
  // retrieval, and an atomic read-modify-write on a line that is not already resident
  // costs a full miss AND exclusive ownership, so it does not overlap the way a plain load
  // does. With the state word in a separate array, a retrieval at n = 262,144 measured
  // 113.1 ns against 39.6 ns for the same resolution without it -- the retrieval half
  // alone cost more than the whole lookup the performance floor allows. Placed beside the
  // key, the increment lands on the line the key comparison just brought in.
  //
  // Non-const on a const index because the word is logically mutable state about an entry,
  // not part of the frozen mapping: nothing this class does depends on its value.
  std::atomic<uint32_t>& stateAt(uint32_t i) const { return records_[i].state; }

  // Entry i's caller-owned payload pointer, stored here for the same measured reason as
  // the state word: with it in a separate array a retrieval fetched a THIRD cache line and
  // cost 99.6 ns at n = 291,129 against 38.2 for the same resolution without it. In the
  // record, key, state and payload arrive together in one 32-byte record, two to a cache
  // line. This class never dereferences it and does not own it.
  NNOutput* payloadAt(uint32_t i) const { return records_[i].payload; }

  // Entry i's PERSISTED MARK: how much of its counter has already been written to the count
  // log. It lives in the 4 bytes this record has always reserved beside the state word, so
  // the delta surface costs the structure not one byte and not one extra cache line -- the
  // mark is in the same 32-byte record the lookup already touched. See
  // NNCacheFrozen::takeUnpersistedHits.
  std::atomic<uint32_t>& persistedCountAt(uint32_t i) const { return records_[i].persistedCount; }
  void setPayloadAt(uint32_t i, NNOutput* payload) { records_[i].payload = payload; }

  // Every byte this index holds resident, including the key array -- not just the
  // displacement machinery. SPEC.md 6 warns specifically about a self-reported footprint
  // that counts only the machinery and under-reports by 8x; this counts everything the
  // object owns.
  size_t structureBytes() const;

  // The displacement-search bound a build is allowed. Named so a refusal can quote it and
  // a test can assert against the implementation rather than a second copy of the number
  // (ADR-0012 P1).
  static uint32_t searchBound();
  // The largest key count this implementation's arithmetic supports (SPEC.md 4.4).
  static uint32_t maxEntries();
  // The exact bytes one entry's record occupies, read from the record type rather than
  // re-typed. SPEC.md 8's whole two-records-to-a-cache-line argument is a claim about this
  // number, so it is a number a test can assert against the implementation (ADR-0012 P1).
  static size_t recordBytes();

 private:
  NNCacheFrozenIndex();

  // One record per entry, in entry order: 16 bytes of key, a 4-byte caller state word, a
  // 4-byte persisted mark, and an 8-byte caller payload pointer. 32 bytes, two to a cache
  // line, so a resolved lookup gets the key it must compare, the counter it must bump and
  // the payload it must hand back in ONE access rather than three. That layout is the whole
  // difference between meeting and missing SPEC.md 8's floor; see stateAt and payloadAt.
  //
  // The persisted mark occupies the 4 bytes this record reserved from the start. It is not a
  // new cost and it does not change the record's size, which testnncachedump.cpp asserts
  // against sizeof rather than trusting from here.
  struct Record {
    Hash128 key;
    mutable std::atomic<uint32_t> state;
    mutable std::atomic<uint32_t> persistedCount;
    NNOutput* payload;
    Record() : key(), state(0), persistedCount(0), payload(nullptr) {}
  };

  std::vector<Record> records_;    // entry order: records_[i] is entry i
  std::vector<uint32_t> posTable_; // slot -> entry index, or EMPTY_SLOT
  std::vector<uint16_t> disp_;     // bucket -> the displacement its keys are placed under
  uint32_t numBuckets_;
  uint32_t tableSize_;
};

//-------------------------------------------------------------------------------------
// Where a level 0's evaluations live
//-------------------------------------------------------------------------------------

// THE EVALUATIONS OF ONE LEVEL 0 TOGETHER WITH THE STORAGE THEY OCCUPY, as one type with
// one lifetime.
//
// WHY THIS TYPE EXISTS AT ALL, since a vector of unique_ptr already worked. There are two
// ways to hold a level 0's evaluations and they have incompatible destruction rules. The
// in-process build path allocates each NNOutput with `new` and each ownership map with
// `new[]`, so ~NNOutput's `delete[] whiteOwnerMap` and unique_ptr's `delete` are both
// exactly right. The LOADER carves both out of an arena -- one contiguous block of
// NNOutputs, one of ownership-map floats -- so both of those are exactly wrong: they would
// hand the allocator an interior pointer it never issued. A std::unique_ptr<NNOutput>
// pointing at arena memory is therefore not a shape to be handled carefully, it is a shape
// that must not EXIST -- so no store ever hands out an owning handle to an evaluation, and
// the only owner of the storage is the store itself (ADR-0000 Rule 2a).
//
// It is not on the lookup path. NNCacheFrozenIndex's records carry the payload pointer and
// a get resolves through the record; this interface is touched at build time and by the
// accounting calls, which are already O(n).
class NNCacheEvaluationStore {
 public:
  virtual ~NNCacheEvaluationStore();

  NNCacheEvaluationStore(const NNCacheEvaluationStore&) = delete;
  NNCacheEvaluationStore& operator=(const NNCacheEvaluationStore&) = delete;

  virtual size_t numEvaluations() const = 0;
  // Evaluation i, owned by this store and never null -- a store that would hold a null
  // refuses to be constructed instead.
  virtual NNOutput* evaluationAt(size_t i) const = 0;
  // The bytes this store spends on PER-ENTRY BOOKKEEPING, the evaluations themselves
  // excluded. It is what NNCacheFrozen::structureBytes adds to the index's own footprint,
  // and it is deliberately not the payload question: the evaluations are counted once, as
  // payload, by nnOutputFootprintBytes.
  virtual size_t handleBytes() const = 0;

 protected:
  NNCacheEvaluationStore();
};

// Evaluations each allocated separately on the heap: the in-process build path, and the
// shape every caller before the loader used.
//
// It costs one pointer per entry of bookkeeping, which is what puts the published level-0
// structure figure at 44.9 B/entry rather than the index's own 36.9.
class NNCacheHeapEvaluationStore final : public NNCacheEvaluationStore {
 public:
  // Throws StringError, naming the position, if any evaluation is null: a null carries no
  // position hash to index it by.
  static std::shared_ptr<NNCacheEvaluationStore> of(std::vector<std::unique_ptr<NNOutput>> evaluations);

  size_t numEvaluations() const override;
  NNOutput* evaluationAt(size_t i) const override;
  size_t handleBytes() const override;

 private:
  explicit NNCacheHeapEvaluationStore(std::vector<std::unique_ptr<NNOutput>> evaluations);

  std::vector<std::unique_ptr<NNOutput>> evaluations_;
};

//-------------------------------------------------------------------------------------
// The level-0 structure: the index, plus evaluations and per-entry hit counters
//-------------------------------------------------------------------------------------

// The frozen level-0 cache: NNCacheFrozenIndex plus the evaluations and the per-entry hit
// counters, built in one call from a supplied vector of evaluations.
//
// KEY AND EVALUATION CANNOT DISAGREE, BY CONSTRUCTION. SPEC.md 3.4 requires that the
// evaluation stored at entry i is the one whose position hash is entry i's key, notes that
// the prototype relies on this without checking it, and invites making the two impossible
// to separate. They are: build() takes ONE list -- the evaluations -- and derives every
// key from its own NNOutput::nnHash. There is no second list to fall out of alignment
// with, so SPEC.md 1.1's "the two lists must be the same length" precondition describes a
// shape this API cannot express (ADR-0000 Rule 2a: the type forecloses the class rather
// than a guard catching each instance of it).
//
// SHADOWING, and why a frozen structure has a mutable bit at all. Level 0 cannot be
// updated in place, but the engine does re-offer keys level 0 already holds -- a caller
// passing skipCache=true evaluates and stores unconditionally, and the ownership-map
// fall-through in NNEvaluator::evaluate deliberately re-evaluates a hit that lacks a
// requested ownership map and then stores the fuller result. If those stores were dropped
// the engine would re-evaluate the same position on every such query forever; if they went
// to level 1 while level 0 kept answering first, the fuller entry would be unreachable AND
// one key would sit in two levels with two counters. So a set for a level-0 key SHADOWS
// its entry: the entry stops being resolvable, its accrued count is handed out for the
// level-1 counter to absorb, and ownership of the key transfers to level 1 permanently.
// The invariant "at most one level owns any key" then holds at every instant rather than
// being assumed of the normal path.
class NNCacheFrozen {
 public:
  // Builds level 0 from `evaluations`, in the given order: position i of the vector is
  // entry i (SPEC.md 1.3). Every counter starts at zero and measures this session only;
  // it is not seeded from anything (SPEC.md 3.2).
  //
  // Throws StringError, yielding no structure, for everything NNCacheFrozenIndex::build
  // refuses and additionally for a null evaluation, which carries no key to index by.
  static std::unique_ptr<NNCacheFrozen> build(std::vector<std::unique_ptr<NNOutput>> evaluations);

  // The same, over evaluations that already live somewhere -- an arena, for the loader.
  // The store is shared rather than owned outright because a get hands out its result
  // through shared_ptr's aliasing constructor against exactly this owner.
  //
  // Throws StringError, yielding no structure, if the store is null.
  static std::unique_ptr<NNCacheFrozen> build(std::shared_ptr<NNCacheEvaluationStore> evaluations);

  NNCacheFrozen(const NNCacheFrozen& other) = delete;
  NNCacheFrozen& operator=(const NNCacheFrozen& other) = delete;

  // Retrieves the evaluation for `key` and counts one hit against its entry (SPEC.md 2.3,
  // 3.2). Returns false and leaves `ret` null if the key is absent or its entry has been
  // shadowed. Thread-safe with no locking.
  bool get(Hash128 key, std::shared_ptr<NNOutput>& ret);

  // Is this key resolvable here, without retrieving the evaluation and without touching
  // any counter (SPEC.md 3.1)? False for a shadowed entry: a shadowed key is no longer
  // level 0's.
  [[nodiscard]] bool contains(Hash128 key) const;

  // Transfers ownership of `key` to level 1: the entry stops resolving, and the hits it
  // accrued AND HAS NOT YET PERSISTED are returned so the caller can fold them into the
  // counter that takes over. Returns nullopt if this index never held the key, or if it was
  // already shadowed -- in which case nothing is transferred, so a racing second caller
  // cannot duplicate a count. Idempotent and thread-safe.
  //
  // UNPERSISTED, not total, and the difference is load-bearing. A count that has already
  // been written to the count log belongs to the log now; handing it to level 1 as though it
  // were new would put it in the next dump too, and appendDump adds. The two figures are the
  // same number until a take has happened, which is why nothing observed this distinction
  // before the delta surface existed.
  [[nodiscard]] std::optional<uint32_t> shadow(Hash128 key);

  // Adds `amount` to a key's counter without retrieving its evaluation, for folding in
  // counts from elsewhere (SPEC.md 3.2). Returns false if the key is absent or shadowed.
  bool addHits(Hash128 key, uint32_t amount);

  const NNCacheFrozenIndex& index() const { return index_; }
  uint32_t numEntries() const { return index_.numEntries(); }
  // Entry i's counter, read without incrementing it (SPEC.md 3.2). A shadowed entry reads
  // zero: its count left with its ownership.
  uint32_t hitCountAt(uint32_t i) const;
  bool isShadowedAt(uint32_t i) const;
  // Entry i's evaluation, without counting a hit. Null once entry i is shadowed.
  std::shared_ptr<NNOutput> evaluationAt(uint32_t i) const;

  // Every unshadowed entry, in index order 0..n-1, with its counter -- the end-of-session
  // harvest of SPEC.md 3.4. Because the input arrives in descending-reference-count order
  // this is descending-popularity order, which the caller relies on. Shadowed entries are
  // omitted: they are level 1's keys now and their counts are in level 1's ledger, so
  // including them here would put one key on the surface twice.
  [[nodiscard]] std::vector<NNCacheHitCount> harvest() const;

  // THE HITS THAT HAVE NOT REACHED THE COUNT LOG YET, taking them as it reports them.
  //
  // Per unshadowed entry: the counter minus this entry's persisted mark, with the mark then
  // advanced to the counter. An entry whose delta is zero yields NO ROW -- see
  // NNCacheTable::takeUnpersistedHitCounts for why a dump must not write a row for a key
  // with nothing to say, and harvest() above for the surface that deliberately does.
  //
  // Not const, and not a reporting call: it MOVES the mark. Take it once per dump, from the
  // dump path, at rest.
  [[nodiscard]] std::vector<NNCacheHitCount> takeUnpersistedHits();

  // Resident bytes: the index, the counters, the evaluation handles. Not the evaluations
  // themselves -- those are the two payload figures below.
  size_t structureBytes() const;
  // The footprint of the evaluations still reachable, and of those held by shadowed
  // entries -- the second is memory this structure owns and can no longer hand out.
  int64_t reachablePayloadBytes() const;
  int64_t shadowedPayloadBytes() const;
  int64_t numReachableEntries() const;

  // The store the evaluations live in. Exposed so that a caller releasing a level 0 can
  // hold a weak reference to it and OBSERVE whether the storage actually went, rather than
  // inferring it from having dropped its own handle -- an outstanding aliased evaluation
  // handed out by get() keeps the whole store alive by construction, and that is a fact a
  // detach must be able to report rather than assume away (ADR-0021 Rule 1).
  const std::shared_ptr<NNCacheEvaluationStore>& evaluationStore() const { return evaluations_; }

 private:
  NNCacheFrozen(NNCacheFrozenIndex&& index, std::shared_ptr<NNCacheEvaluationStore>&& evaluations);

  static const uint32_t SHADOW_BIT = 0x80000000u;
  static const uint32_t COUNT_MASK = 0x7FFFFFFFu;

  NNCacheFrozenIndex index_;
  // The caller's evaluations and the storage they live in, MOVED in rather than copied --
  // so no moment of construction holds two sets of handles, and SPEC.md 6's transient-peak
  // ceiling is met with the same margin as its resident one.
  //
  // ONE SHARED REFERENCE COUNT OVER THE WHOLE SET, and this is a performance contract
  // rather than a detail. A get hands out its result through shared_ptr's ALIASING
  // constructor against this single owner, so the atomic increment every lookup performs
  // lands on ONE always-hot control block instead of on a per-entry control block
  // scattered through hundreds of megabytes of payload. The first version of this class
  // gave every entry its own shared_ptr and paid a DRAM miss plus two atomic
  // read-modify-writes on a cold line per lookup, which put it 2 to 2.5 times over
  // SPEC.md 8's floor. SPEC.md 3.3 records the prototype making exactly this choice and
  // says why -- "one always-hot cache line instead of a scattered refcount per entry" --
  // and marks it INCIDENTAL, which it is as behaviour and is not as speed.
  //
  // Two consequences, both the prototype's too. A caller holding one returned evaluation
  // keeps the WHOLE level-0 payload alive -- which costs nothing, because level 0 is built
  // for a session and lives for it, and which a release therefore OBSERVES rather than
  // assumes away (see evaluationStore). And level 0 OWNS its evaluations: no store hands
  // out an owning handle to one, so a caller cannot become a second owner. Nothing in the
  // engine wants to, because level 0 is built once from a supplied set and is never set
  // into.
  //
  // This store is never touched on the lookup path -- the record carries the pointer.
  std::shared_ptr<NNCacheEvaluationStore> evaluations_;
  // The per-entry count and shadow flag live in the index's own record, beside the key --
  // see NNCacheFrozenIndex::stateAt for the measurement that put them there. They share a
  // single 32-bit word so that shadowing and reading out the accrued count are ONE atomic
  // exchange: a count cannot be transferred twice and cannot be split across the transfer.
  // The top bit is the shadow flag and the low 31 bits are the count, which is ample -- the
  // largest lifetime reference count in the operator's whole database is 11,997
  // (SPEC.md 3.2).
};

//-------------------------------------------------------------------------------------
// The two-level resolution strategy
//-------------------------------------------------------------------------------------

// Composes a frozen level 0 over an ordinary level 1 and presents them as one
// NNCacheTable, with one unified hit-count surface across both.
//
// RESOLUTION. A get asks level 0 first, and falls through to level 1 only on a level-0
// miss. A set never reaches level 0 -- it is frozen -- so it goes to level 1, and if level
// 0 held that key the level-0 entry is shadowed first so that ownership of the key, and
// the count it accrued, move to level 1 together. clear() empties level 1 and shadows
// nothing: level 0 is the pre-warmed content a session was given and is not the session's
// to discard.
//
// LEVEL 0 IS OPTIONAL, AND ABSENT IS THE DEFAULT -- but absence is represented by not
// having one of these at all, rather than by one of these holding a null level 0.
// NNCacheTable::create is untouched and still builds a single-level table for every
// configuration; this factory is reachable only by handing it a level 0, and it refuses a
// null one. So the no-level-0 path allocates nothing, tests nothing and branches on
// nothing (ADR-0000 Rule 2a).
//
// `hitLedgerPowerOfTwo` sizes the level-1 hit ledger at 2^k rows. Level 1 has no per-entry
// counter of its own and adding one would touch four table implementations and change the
// default table's memory, so the counts for level-1-owned keys live in one table here,
// holding the full 128-bit key beside each count so a count is never attributed to the
// wrong key. Throws if either argument cannot be honored.
std::unique_ptr<NNCacheTable> makeTwoLevelNNCacheTable(
  std::unique_ptr<NNCacheFrozen> levelZero,
  std::unique_ptr<NNCacheTable> levelOne,
  int hitLedgerPowerOfTwo
);

// The exact resident cost of the level-1 hit ledger that factory allocates. Named here so
// the bound is stated in one place and can be asserted rather than estimated (ADR-0012 P1).
size_t twoLevelHitLedgerBytes(int hitLedgerPowerOfTwo);

#endif  // NEURALNET_NNCACHEFROZEN_H_
