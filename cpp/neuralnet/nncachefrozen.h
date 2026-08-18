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

  uint32_t numEntries() const { return (uint32_t)keys_.size(); }
  // Entry i's key. Entries are enumerable in index order 0..n-1 through this (SPEC.md 3.4).
  Hash128 keyAt(uint32_t i) const { return keys_[i]; }

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

 private:
  NNCacheFrozenIndex();

  std::vector<Hash128> keys_;      // entry order: keys_[i] is entry i's key
  std::vector<uint32_t> posTable_; // slot -> entry index, or EMPTY_SLOT
  std::vector<uint16_t> disp_;     // bucket -> the displacement its keys are placed under
  uint32_t numBuckets_;
  uint32_t tableSize_;
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
  static std::unique_ptr<NNCacheFrozen> build(std::vector<std::shared_ptr<NNOutput>> evaluations);

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
  // accrued are returned so the caller can fold them into the counter that takes over.
  // Returns nullopt if this index never held the key, or if it was already shadowed --
  // in which case nothing is transferred, so a racing second caller cannot duplicate a
  // count. Idempotent and thread-safe.
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

  // Resident bytes: the index, the counters, the evaluation handles. Not the evaluations
  // themselves -- those are the two payload figures below.
  size_t structureBytes() const;
  // The footprint of the evaluations still reachable, and of those held by shadowed
  // entries -- the second is memory this structure owns and can no longer hand out.
  int64_t reachablePayloadBytes() const;
  int64_t shadowedPayloadBytes() const;
  int64_t numReachableEntries() const;

 private:
  NNCacheFrozen(NNCacheFrozenIndex&& index, std::vector<std::shared_ptr<NNOutput>>&& evaluations);

  static const uint32_t SHADOW_BIT = 0x80000000u;
  static const uint32_t COUNT_MASK = 0x7FFFFFFFu;

  NNCacheFrozenIndex index_;
  // The caller's evaluation vector, MOVED in rather than copied -- so no moment of
  // construction holds two sets of handles, and SPEC.md 6's transient-peak ceiling is met
  // with the same margin as its resident one.
  std::vector<std::shared_ptr<NNOutput>> evaluations_;
  // One entry's mutable state, kept separate from the handles for the reason above. The
  // count and the shadow flag share a single 32-bit word so that shadowing and reading out
  // the accrued count are ONE atomic exchange: a count cannot be transferred twice and
  // cannot be split across the transfer. The top bit is the shadow flag and the low 31 bits
  // are the count, which is ample -- the largest lifetime reference count in the operator's
  // whole database is 11,997 (SPEC.md 3.2).
  std::vector<std::atomic<uint32_t>> states_;
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
