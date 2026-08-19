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
// be reached while the order stands (it is reported as such by stats(), which sums what every
// source holds). It is reported ONCE on the count surfaces -- both of them fold every
// holder's count into one row per key by summing, so a holder that earned its count under an
// earlier order keeps it (NNCacheLevelZeroSources::harvest).
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
// path differs by about 1 ns). Independently re-measured by review at 4.3-6.5 ns, same sign
// every time.
//
// THE EFFECT IS MEASURED; THE EXPLANATION BELOW IS NOT, and the two are marked apart on
// purpose (amended 2026-08-19 after review drew the line). The HYPOTHESIS, consistent with
// the effect and with the probe arms that ruled out the loop and the bounds reload, is one
// extra dependent cache line: the sources live in a heap-allocated array, so resolving the
// first one loads the array base out of this object and then the source pointer out of that
// block, where a single owned member was one load inside the object already in cache. NO
// PROFILER WAS RUN, so no cache-miss count was ever observed and this is not a measured
// cause. Anyone acting on it should measure the mechanism first.
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
  //
  // KNOWN DEFECT, WITNESSED AND FILED 2026-08-19 -- RE-ATTACHING A SOURCE THAT SAT OUT A SET.
  // A set shadows the key in every ATTACHED holder (shadowAllHolders); a source detached at
  // that moment is not one, so it comes back still holding the key unshadowed and resolves it
  // ahead of the level 1 that now owns it. Witnessed through this surface alone: attach A over
  // an empty level 1, serve the key, attach B, detach A, serve it from B, set the key, then
  // re-attach A -- the table serves A's superseded evaluation while level 1 holds the fresh
  // one, and the key appears on harvestHitCounts TWICE, once from A and once from the level-1
  // ledger, breaking the one-row-per-key property a dump depends on. NOT FIXED HERE: the cure
  // is a decision about the attach contract (shadow on re-attach against level 1, or refuse a
  // source carrying a key level 1 owns), it costs a probe per entry at attach, and it is a
  // different defect from the harvest folding rule this increment was chartered for. It is
  // recorded here, at the act that reintroduces the stale holder, rather than narrated
  // elsewhere and left (ADR-0013 Rule 4).
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
  // A key held by several sources yields ONE row, positioned where the EARLIEST holder's
  // entry falls, carrying the SUM of what every holder counted for it. Summing double-counts
  // nothing: a get increments exactly one source's counter, every counter starts at zero when
  // its source is built or loaded, and a set takes the key out of level 0 entirely
  // (shadowAllHolders), so two holders' counts for one key are disjoint sets of real
  // retrievals. It is the same rule takeUnpersistedHits below folds by, through the same
  // function, and the same rule shadowAllHolders already uses at this seam.
  //
  // WHY A LEVEL SUMS HERE, since a level is normally read off the one entry that owns it, and
  // this call reports one (this session's running total for a key, where takeUnpersistedHits
  // reports the flow not yet written). The level is a per-KEY quantity, and its ACCRUAL is
  // spread across entries the moment the resolution order moves: attach A, let it serve the
  // key twice, attach B which also holds it, detach A, let B serve it once, re-attach A --
  // attach appends, so A returns at the BACK holding two real retrievals of a key it no
  // longer resolves. Read off the resolving entry alone, the level answers 1 for a key that
  // was retrieved 3 times. An entry is where part of a level is KEPT, not what the level is
  // ABOUT, so the level is the sum over the entries that kept part of it. Levels de-duplicate
  // only when they live in one place; this one does not.
  //
  // RESOLVED 2026-08-19, closing the KNOWN DIVERGENCE this call carried against
  // takeUnpersistedHits. It previously emitted the earliest holder's row alone and THREW if a
  // suppressed row carried a count, on the argument that an unreachable entry can accrue
  // nothing. The sequence above reaches that state through this class's own public surface
  // with no poke, so the refusal fired on a legitimate history; the two surfaces now fold by
  // one rule and the one-owner argument is retired rather than left standing beside a
  // counter-example.
  //
  // THE ONE REFUSAL LEFT is a sum that will not fit the 32-bit row a count log record carries.
  // That is a limit of the OUTPUT TYPE -- no state of this list is legitimate and also
  // unrepresentable in the row it must be written to -- so it cannot fire on a good state the
  // way the one-owner tripwire could, and it is refused rather than wrapped (ADR-0002).
  [[nodiscard]] std::vector<NNCacheHitCount> harvest() const;

  // THE HITS THAT HAVE NOT REACHED THE COUNT LOG YET, across every attached source, taking
  // them as it reports them. The delta twin of harvest() above.
  //
  // IT FOLDS BY THE SAME RULE, through the same function: a key held by several sources yields
  // ONE row whose hits are the SUM of every holder's unpersisted delta. What differs is the
  // QUANTITY, not the folding -- this reports a FLOW (the retrievals not yet written to the
  // count log) where harvest reports a LEVEL (the session's running total), and a flow is
  // CONSERVED: appendDump ADDS each row's lookups to the key's running total, so a delta
  // dropped here is retrievals gone from the record forever with no honesty counter to show
  // it. (Recorded 2026-08-19: harvest used to suppress-and-refuse instead, and the divergence
  // noted here is closed -- see harvest's own comment for the sequence that closed it.)
  //
  // THE TWO REAL DIFFERENCES, both about what this call DOES rather than how it folds:
  //
  //   EVERY SOURCE IS TAKEN FROM, including one whose rows merge into an earlier source's.
  //   This call is CONSUMING -- it advances each entry's persisted mark -- so a source skipped
  //   for being "already covered" would keep its mark behind and re-report the same delta on
  //   the next take. There is no early exit anywhere in it, and the single-source fast path
  //   harvest() has is deliberately absent here.
  //
  //   A KEY WITH NOTHING TO SAY YIELDS NO ROW, where harvest() deliberately reports a
  //   pre-warmed entry that earned nothing as a zero. See NNCacheTable::takeUnpersistedHitCounts.
  //
  // Not const, and not a reporting call: it MOVES the marks. Take it once per dump, at rest.
  [[nodiscard]] std::vector<NNCacheHitCount> takeUnpersistedHits();

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
