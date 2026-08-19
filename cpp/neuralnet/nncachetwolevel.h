#ifndef NEURALNET_NNCACHETWOLEVEL_H_
#define NEURALNET_NNCACHETWOLEVEL_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "../core/hash.h"
#include "../neuralnet/nncache.h"
#include "../neuralnet/nncachefrozen.h"
#include "../neuralnet/nninputs.h"

// THE TWO-LEVEL RESOLUTION STRATEGY: an ORDERED LIST of frozen level-0 sources over one
// ordinary level 1, presented as a single NNCacheTable with one hit-count surface across
// all of them.
//
// RESOLUTION, in one sentence, because everything else in this header is a consequence of
// it: a get tries each attached source IN ATTACH ORDER and returns the FIRST that resolves
// the key, and only a miss in every one of them falls through to level 1.
//
// THE ORDER IS THE PRIORITY, AND THERE IS NO OTHER PRIORITY. This is the design's §10
// answer to the operator's "make sure the strongest net replaces the weaker ones -- or by
// policy ensure they never overlap", and it is a TYPE decision rather than a policy knob.
// Model identity is a property of the CONTAINER a source was loaded from (one model per
// file), so two nets' evaluations of one position are never candidates for one slot by
// accident and NON-OVERLAP IS THE STRUCTURAL DEFAULT -- it costs nothing and there is no
// knob to get wrong. "Strongest replaces weaker" is then an OPT-IN the client expresses by
// attaching the strong model's source first and the weaker ones after it. So there is no
// per-source priority field here, and deliberately so: a numeric priority beside an
// ordered list is a second home for one fact and the two can disagree (ADR-0012 P1). The
// list order IS the fact, and a reader of this header cannot find a second copy of it.
//
// WHAT AN OVERLAP COSTS, stated rather than left to be discovered: at the ~3.7% cross-card
// key sharing the corpus actually shows, a key can sit in two attached sources. The first
// in order serves it and counts it; the second's entry is resident memory that will never
// be reached (it is reported as such by stats(), which sums what every source holds) and
// never counted (it is suppressed from the harvest, see NNCacheLevelZeroSources::harvest).
//
// THE MISS COST IS ONE CHD PROBE PER ATTACHED SOURCE, ~40-110 ns at real sizes, against
// the 2.4-2.8 ms an avoided evaluation costs. Three orders of magnitude of headroom is not
// a licence to be careless, so the walk allocates nothing, locks nothing and touches no
// shared counter; it is a bounded loop over a contiguous vector. Measured on this box at
// the median card size: ~28 ns for the second attached source and ~41 ns for the third
// (runnncachetwolevelbench, minima of 11 reps of 1e6 lookups).
//
// WHAT THE LIST COSTS AT ONE SOURCE, MEASURED AND NOT YET REMOVED. Against the single-level-0
// shape this replaced, the miss path is about 5 ns slower at one attached source (39.5-40.0
// ns against 44.6-45.5, minima of 21 interleaved rep-pairs, three runs, paired in one
// process against the pre-change table compiled from d9b6e591 into the same binary; the hit
// path differs by about 1 ns). The cause is structural and is one extra dependent cache
// line: the sources live in a heap-allocated array, so resolving the first one loads the
// array base out of this object and then the source pointer out of that block, where a
// single owned member was one load inside the object already in cache.
//
// THE OPTION THAT WOULD REMOVE IT, named so this is a filed item and not an apology: hold
// the first few source pointers in a fixed inline array inside this object and spill to the
// heap vector beyond it. It is not built here because it introduces a SECOND storage shape
// for the resolution order -- the one fact this class exists to own once -- to buy 5 ns
// against the 2.4-2.8 ms per avoided evaluation the whole structure exists to save, which
// is the complexity-without-a-witnessed-need the design's §7 declines by name for the
// lock-free swap. Build it when a measurement shows the 5 ns mattering to something, not
// before; the bench that would show it is runnncachetwolevelbench.
//
// CONCURRENCY, honestly. A get is lock-free and takes no snapshot. ATTACH AND DETACH ARE
// SESSION-BOUNDARY ACTS AND ARE NOT SAFE AGAINST A CONCURRENT GET -- they mutate the
// vector the resolution walk reads. That is the design's own ratified v1 contract (§7,
// "attach and detach are refused while any request is open"), and the refusal that
// enforces it belongs to the protocol layer that knows the open-request count, not here.
// The same posture NNCacheTable::attachCacheContext already carries for the same reason.

//-------------------------------------------------------------------------------------
// The handle
//-------------------------------------------------------------------------------------

// ONE ATTACHED LEVEL-0 SOURCE, as a value that can only mean that.
//
// Minted by NNCacheLevelZeroSources::attach and by nothing else: the constructor is
// private and the list is its only friend. There is no conversion from a string, an
// integer, or a position, so an index a loop produced cannot become a detach argument.
//
// WHAT THIS TYPE FORECLOSES, which is the whole reason it is not a size_t.
//
//   A STALE HANDLE CANNOT ADDRESS A DIFFERENT SOURCE. The serial is drawn from a counter
//   that only ever increases and is NEVER REUSED, so an id left over from a detached
//   source names nothing, permanently -- it cannot come to name whatever was attached
//   next. Contrast NNCacheContextId, whose payload is a POSITION: that is right there
//   because contexts never detach, and it would be wrong here because sources do. A
//   position-carrying handle in a list that shrinks is the wrong-source-served defect
//   class, and it is foreclosed here by construction rather than checked for
//   (ADR-0000 Rule 2a).
//
//   AN ID FROM ANOTHER TABLE CANNOT BE SPENT HERE. Every id carries the identity of the
//   list that minted it, so an id from one model's cache used against another's is refused
//   BY NAME rather than silently addressing this list's own source with the same serial --
//   the same rule NNCacheContextId carries for the same hazard.
//
//   THERE IS NO LOOKUP AGAINST A SOURCE AT ALL. This header exposes no "get from source
//   X": resolution is a property of the LIST, not of a source a caller holds a handle to.
//   So "a lookup against a detached source" is not an error case that is checked; it is a
//   sentence that cannot be written against this interface.
class NNCacheLevelZeroSourceId {
 public:
  // Which list minted this, so a list can refuse an id that is not its own.
  [[nodiscard]] uint64_t listId() const { return listId_; }
  // Its mint order within that list. Unique for the lifetime of the process and never
  // reused after a detach; NOT a position in the resolution order, which shifts.
  [[nodiscard]] uint64_t serial() const { return serial_; }

  [[nodiscard]] bool operator==(const NNCacheLevelZeroSourceId& other) const {
    return listId_ == other.listId_ && serial_ == other.serial_;
  }
  [[nodiscard]] bool operator!=(const NNCacheLevelZeroSourceId& other) const { return !(*this == other); }

 private:
  friend class NNCacheLevelZeroSources;
  NNCacheLevelZeroSourceId(uint64_t listId, uint64_t serial) : listId_(listId), serial_(serial) {}

  uint64_t listId_;
  uint64_t serial_;
};

//-------------------------------------------------------------------------------------
// The ordered resolution list
//-------------------------------------------------------------------------------------

// THE ATTACHED LEVEL-0 SOURCES, IN RESOLUTION ORDER, together with the resolution rule
// itself. One object owns both, because "which source answers first" is not a fact about
// any one source -- it is the list's own fact, and giving it a second home is exactly the
// SSOT failure ADR-0012 P1 names.
//
// The list therefore exposes NO positional accessor. Everything a caller could want a
// position for -- resolve this key, shadow this key everywhere, harvest, account for the
// memory -- is a method here, phrased over the whole list. A caller cannot walk it out of
// order, cannot walk a stale copy of it, and cannot address a source that has left it.
class NNCacheLevelZeroSources {
 public:
  NNCacheLevelZeroSources();
  ~NNCacheLevelZeroSources();

  NNCacheLevelZeroSources(const NNCacheLevelZeroSources&) = delete;
  NNCacheLevelZeroSources& operator=(const NNCacheLevelZeroSources&) = delete;

  //---- The list -------------------------------------------------------------------

  // Appends `source` at the END of the resolution order and returns the only kind of value
  // that can address it. LAST ATTACHED IS LAST TRIED: a client that wants a source to win
  // a key attaches it earlier, which is the whole of the priority mechanism (§10).
  //
  // Throws StringError for a null source: "no source" is represented by this list being
  // shorter, never by it holding a null (ADR-0012 P11).
  [[nodiscard]] NNCacheLevelZeroSourceId attach(std::unique_ptr<NNCacheFrozen> source);

  // Removes the source `id` names and HANDS IT BACK, leaving the relative order of every
  // other source exactly as it was. Returning it rather than destroying it is what lets
  // the caller pass it to nnCacheReleaseLevelZero, which observes whether the storage
  // actually went instead of assuming it (nncachelevelzero.h); and taking it out of the
  // list in the same statement is what stops the list holding an entry nobody owns.
  //
  // Throws StringError, naming the cause, for an id this list did not mint and for an id
  // whose source has already been detached -- never returning a null, and never falling
  // back to some other source (ADR-0002).
  [[nodiscard]] std::unique_ptr<NNCacheFrozen> detach(const NNCacheLevelZeroSourceId& id);

  [[nodiscard]] size_t size() const;
  // The attached sources' ids, in resolution order -- for a report, and for a test to
  // assert the order against rather than infer it from behaviour.
  //
  // There is deliberately no owns()/holds() query beside it, and no accessor for the list's
  // own identity: detach already refuses a foreign or spent id BY NAME, so a caller has
  // nothing to ask that the act itself does not answer, and a second way to ask would be
  // surface with no consumer (ADR-0012's cancer E). Add one when something needs it.
  [[nodiscard]] std::vector<NNCacheLevelZeroSourceId> resolutionOrder() const;

  //---- The resolution rule --------------------------------------------------------

  // The FIRST attached source, in resolution order, that resolves `key`. Counts one hit
  // against that source's entry and no other's. Returns false, leaving `ret` null, when no
  // attached source holds the key.
  bool get(Hash128 key, std::shared_ptr<NNOutput>& ret);

  // Shadows `key` in EVERY attached source that holds it, and returns the total count they
  // gave up.
  //
  // EVERY source, not the first: level 1 is about to own this key, and a lower-priority
  // source that kept resolving it would answer with the superseded evaluation before the
  // fall-through to level 1 could ever happen -- the exact defect shadowing exists to
  // prevent, reintroduced by the list. So the invariant is "at most ONE level owns any
  // key" across the whole list, and it is upheld by shadowing across the whole list.
  uint64_t shadowAllHolders(Hash128 key);

  // Every unshadowed entry of every attached source, each source in resolution order and
  // each source's entries in its own index order, WITH ONE ROW PER KEY.
  //
  // A key held by two sources appears ONCE, from the source that would resolve it -- the
  // earlier one. The suppressed row's count is zero and cannot be anything else: the later
  // source's entry is unreachable while the earlier one resolves, and a set for the key
  // shadows it in both at once (shadowAllHolders), so no path exists by which it accrues a
  // hit. Nothing is therefore lost by suppressing it, and what is gained is the one-row-
  // per-key property a persistence layer must never have to repair.
  //
  // THAT PROOF LEANS ON A DIFFERENT METHOD, so it is CHECKED HERE rather than trusted: a
  // suppressed row carrying a non-zero count means the one-owner invariant has broken
  // somewhere else, and this throws, naming the key and the count, instead of dropping
  // retrievals a persistence layer would never see again.
  [[nodiscard]] std::vector<NNCacheHitCount> harvest() const;

  //---- What the list holds, for stats() -------------------------------------------

  [[nodiscard]] int64_t numReachableEntries() const;
  [[nodiscard]] int64_t reachablePayloadBytes() const;
  [[nodiscard]] int64_t shadowedPayloadBytes() const;
  [[nodiscard]] int64_t structureBytes() const;
  // Entries across every attached source, shadowed ones included -- the capacity figure.
  [[nodiscard]] int64_t numEntries() const;

 private:
  struct Entry {
    uint64_t serial;
    std::unique_ptr<NNCacheFrozen> source;
  };

  uint64_t listId_;
  // Only ever increases. A detach does not give a serial back, which is what makes a stale
  // id permanently harmless rather than dangerous once the list has churned.
  uint64_t nextSerial_;
  // In resolution order. The single home of "which source answers first".
  std::vector<Entry> entries_;
};

//-------------------------------------------------------------------------------------
// The table
//-------------------------------------------------------------------------------------

// The two-level table as its OWNER sees it: an NNCacheTable to every consumer on the
// get/set path, plus the attach/detach surface the protocol layer drives.
//
// It is a distinct type from NNCacheTable rather than a widening of it because attaching a
// frozen source is not something an arbitrary cache table can be asked to do -- a
// single-level table has nowhere to put one. A caller that holds this type can attach; a
// caller that holds an NNCacheTable& cannot, and does not have to be told at run time.
class NNCacheTwoLevelTable : public NNCacheTable {
 public:
  ~NNCacheTwoLevelTable() override;

  // See NNCacheLevelZeroSources for all four; these are the same acts, on this table's own
  // list, and they carry the same refusals.
  [[nodiscard]] virtual NNCacheLevelZeroSourceId attachLevelZero(std::unique_ptr<NNCacheFrozen> source) = 0;
  [[nodiscard]] virtual std::unique_ptr<NNCacheFrozen> detachLevelZero(const NNCacheLevelZeroSourceId& id) = 0;
  [[nodiscard]] virtual size_t numLevelZeroSources() const = 0;
  [[nodiscard]] virtual std::vector<NNCacheLevelZeroSourceId> levelZeroResolutionOrder() const = 0;

 protected:
  NNCacheTwoLevelTable();
};

// Composes a first frozen level-0 source over an ordinary level 1.
//
// LEVEL 0 IS OPTIONAL, AND ABSENT IS THE DEFAULT -- but absence is represented by not
// having one of these at all, rather than by one of these holding a null or being built
// empty. NNCacheTable::create is untouched and still builds a single-level table for every
// configuration; this factory is reachable only by handing it a source, and it refuses a
// null one. So the no-level-0 path allocates nothing, tests nothing and branches on
// nothing (ADR-0000 Rule 2a).
//
// That refusal is about WHICH TABLE SHAPE a configuration wants, decided once, and it is
// deliberately not a claim about how many sources the table carries afterwards: the
// protocol drives that, and detaching the last source is legal and leaves a table that
// serves everything from level 1 until the next attach.
//
// `hitLedgerPowerOfTwo` sizes the level-1 hit ledger at 2^k rows. Level 1 has no per-entry
// counter of its own and adding one would touch four table implementations and change the
// default table's memory, so the counts for level-1-owned keys live in one table here,
// holding the full 128-bit key beside each count so a count is never attributed to the
// wrong key. Throws if either argument cannot be honored.
std::unique_ptr<NNCacheTwoLevelTable> makeTwoLevelNNCacheTable(
  std::unique_ptr<NNCacheFrozen> levelZero,
  std::unique_ptr<NNCacheTable> levelOne,
  int hitLedgerPowerOfTwo
);

// The exact resident cost of the level-1 hit ledger that factory allocates. Named here so
// the bound is stated in one place and can be asserted rather than estimated (ADR-0012 P1).
size_t twoLevelHitLedgerBytes(int hitLedgerPowerOfTwo);

#endif  // NEURALNET_NNCACHETWOLEVEL_H_
