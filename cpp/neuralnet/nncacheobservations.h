#ifndef NEURALNET_NNCACHEOBSERVATIONS_H_
#define NEURALNET_NNCACHEOBSERVATIONS_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "../core/hash.h"
#include "../neuralnet/nncachecontext.h"

// THE OBSERVATION LEDGER: per (key, context), how many times this session PRESENTED that
// position to the cache -- which is the currency the count log records and the currency disk
// admission gates on.
//
// WHAT AN OBSERVATION IS, IN ONE SENTENCE, because everything else here is a consequence of
// it: one evaluation request for one position, under one context, counted once whether the
// cache answered it or a forward pass had to. It is a WOULD-HAVE-BEEN-COMPUTED FORWARD PASS,
// not a cache hit.
//
// WHY THE CURRENCY CHANGED, stated once here so no reader has to reconstruct it. The count
// log used to record RETRIEVALS: a hit for a key some earlier dump had already stored. That
// number answers "how well is the cache working", and the question a persistence layer
// actually asks is "how much work would carrying this position save" -- which is how often
// the position COMES UP, hit or miss. Under retrievals a position freshly evaluated in a
// session earned no row at all, so it could never accumulate toward an admission threshold
// and the seen-twice policy the operator runs could never bootstrap: nothing was ever seen
// once. Under observations it earns a row of 1, the next session's presentation makes it 2,
// and the threshold is reachable. (Ledger rows 1651/1654 named the confusion; 1717/1722
// ratified this currency.)
//
// WHERE AN OBSERVATION IS COUNTED, AND WHY THERE IS EXACTLY ONE DOOR. It is counted by
// NNCacheTable::observe, called once per evaluation request by NNEvaluator::evaluate,
// immediately after the position's hash is final and before any level is consulted. It is
// deliberately NOT counted inside get() and set():
//
//   A REQUEST IS ONE PRESENTATION EVEN WHEN IT MAKES TWO CALLS. An ordinary miss makes a get
//   and then a set of the evaluation it computed; the ownership-map fall-through in
//   NNEvaluator::evaluate makes a get, rejects the hit, and sets a fuller result. Counting
//   both calls would make one fresh evaluation read as two observations -- and the whole
//   point of the currency is that ONE fresh evaluation is ONE observation, so that a second
//   SESSION is what carries a key over a seen-twice threshold.
//
//   AND A REQUEST IS ONE PRESENTATION EVEN WHEN IT MAKES NO GET. A caller passing skipCache
//   consults no level and sets unconditionally. Counting only gets would lose it.
//
//   MEANWHILE NOT EVERY set() IS A PRESENTATION AT ALL. An attach FILLS level 1 from the
//   context's own container (NNCacheEntryProvenance::LoadedFromContainer); nobody asked for
//   those positions. Counting them would make attach -> detach -> attach with no traffic
//   raise every loaded key's count by one per cycle, when the ratified contract is that such
//   a cycle is a tracking NO-OP.
//
// WHAT THIS STRUCTURE HOLDS IS THE UNPERSISTED DELTA, NOT THE LIFETIME TOTAL. The lifetime
// total's one home is the context's .nncounts file, which is an additive log: a dump appends
// what has accrued since the last one, and a load sums every block (ADR-0012 P1). So a
// session that attaches, observes nothing and detaches appends nothing and changes no total,
// and a key observed once in session A and once in session B reads as 2 from the file in
// session B -- which is what makes the cross-session bootstrap above work without this
// structure ever holding a number it did not itself count.
//
// EXACT, NOT A SKETCH. Every row carries the FULL 128-bit key and the context index beside
// its count, so a count is never attributed to the wrong key or the wrong card, and an
// observation that cannot be given a row is REFUSED and counted in unrecordedObservations()
// rather than overwriting somebody else's row (ADR-0002). The same posture the attribution
// recorder and the two-level hit ledger already carry, for the same reason.

//-------------------------------------------------------------------------------------
// The values
//-------------------------------------------------------------------------------------

// One key's observations under one context, this session.
struct NNCacheObservationCount {
  Hash128 key;
  // Presentations of this key under this context since the last dump. At least 1: a row
  // exists exactly because the key was presented.
  uint32_t observations;
};

// Whether a table observes at all.
//
// A typed disposition rather than an empty row vector standing for two different facts
// (ADR-0012 P11): "no context was ever attached here, so this table observes nothing" and
// "a context is attached and nothing was presented" are not the same answer, and a dump that
// read the first as the second would append a block claiming a whole session found nothing.
enum class NNCacheObservationLedgerDisposition {
  // No context has ever been attached to this table, so there is no ledger. The default
  // configuration -- plain play, no persisted cache -- is here, and pays one predictable
  // branch and not one byte of memory.
  NotObserved,
  Observed,
};

class NNCacheObservationLedger {
 public:
  static NNCacheObservationLedger notObserved();
  static NNCacheObservationLedger observed(
    std::vector<NNCacheObservationCount> entries, int64_t unrecordedObservations
  );

  [[nodiscard]] NNCacheObservationLedgerDisposition disposition() const { return disposition_; }
  [[nodiscard]] bool isObserved() const { return disposition_ == NNCacheObservationLedgerDisposition::Observed; }

  // The rows. Throws under NotObserved rather than handing back an empty vector a caller
  // could read as "nothing was presented".
  [[nodiscard]] const std::vector<NNCacheObservationCount>& entries() const;

  // Observations that happened and could NOT be given a row, because the recorder's probe
  // window was full. Zero in every ordinary run; nonzero means this ledger is short and says
  // by how much, rather than being silently incomplete (ADR-0002). Throws under NotObserved.
  [[nodiscard]] int64_t unrecordedObservations() const;

 private:
  NNCacheObservationLedger(
    NNCacheObservationLedgerDisposition disposition,
    std::vector<NNCacheObservationCount> entries,
    int64_t unrecordedObservations
  );

  NNCacheObservationLedgerDisposition disposition_;
  std::vector<NNCacheObservationCount> entries_;
  int64_t unrecordedObservations_;
};

//-------------------------------------------------------------------------------------
// The recorder
//-------------------------------------------------------------------------------------

// The per-(key, context) observation counts a cache table writes on its observe path.
//
// Shaped exactly like NNCacheAttributionRecorder and the two-level table's hit ledger, for
// the same reasons: a fixed row array with a bounded probe window and a mutex pool, so an
// observation costs one lock and O(1) work and never a rehash; the full key beside every
// row; and an overflow that is counted rather than absorbed.
//
// KEYED ON (KEY, CONTEXT) AND NOT ON KEY ALONE, because two attached cards really do share
// about 3.7% of their keys, and a shared position presented under each of them is two
// observations that belong in two different files. The home slot therefore mixes the context
// index into the key's own hash, so the two rows are independent from the first probe rather
// than colliding on one home and burning window on each other.
//
// ALLOCATED ONLY WHEN A CONTEXT IS ATTACHED, alongside the attribution recorder and in the
// same statement, so that "attribution ledger exists" and "observation ledger exists" cannot
// disagree. A table with no attached context constructs neither.
class NNCacheObservationRecorder {
 public:
  // The default size, as a power of two rows.
  //
  // A WORKING SIZE, NOT A PROOF OF SUFFICIENCY, exactly as
  // NNCacheAttributionRecorder::defaultPowerOfTwo() is and for the same reason: the quantity
  // that fills this structure is the number of DISTINCT (position, card) pairs a session
  // PRESENTS, and no corpus figure bounds that. What makes it safe is not the constant but
  // that overflow past it is counted and reported.
  //
  // It is the SAME size as the attribution recorder, 2^20 rows, and that is a decision rather
  // than a coincidence. The population here is a superset of that one -- every key a session
  // earns is also a key it presented, plus every key it presented and got an answer for --
  // but both are bounded in reference by the session touching one largest real card
  // (sizingReferenceKeys(), 291,129 keys), which sits at 27.8% occupancy here, well under
  // maxLoadFactorPercent(). REJECTED: 2^21, which would carry twice the reference at the same
  // occupancy; it doubles a 33.5 MB resident cost in a feature whose entire premise is that
  // the operator's processes already hold too much, to buy headroom against a session shape
  // no corpus figure witnesses. The honest disposition for a session that outgrows this is
  // the overflow counter, which says so.
  static int defaultPowerOfTwo();
  // The corpus scale defaultPowerOfTwo() is sized against, read from the one home that already
  // states it rather than re-typed here (ADR-0012 P1).
  static int64_t sizingReferenceKeys();
  // The occupancy defaultPowerOfTwo() keeps this under at sizingReferenceKeys().
  static int maxLoadFactorPercent();
  // The exact per-row resident cost, read from the row type itself rather than re-typed here.
  static size_t rowBytes();

  NNCacheObservationRecorder(int powerOfTwo, int mutexPoolSizePowerOfTwo);
  ~NNCacheObservationRecorder();
  NNCacheObservationRecorder(const NNCacheObservationRecorder&) = delete;
  NNCacheObservationRecorder& operator=(const NNCacheObservationRecorder&) = delete;

  // RECORDS ONE PRESENTATION of `key` under `context`. Thread-safe, and this is the hot-path
  // call: one mutex from a pool, one bounded probe, one increment. Saturating at 2^32-1, which
  // the 32-bit count-log record is the reason for and which no real session approaches (the
  // largest lifetime count in the operator's corpus is 11,997).
  //
  // `contextIndex` is NNCacheContextId::index(), already checked by the caller against the set
  // that minted it -- the table's own ownership refusal is the one home of that check, and
  // repeating it per observation would put a second copy of it on the hot path.
  void observe(Hash128 key, uint32_t contextIndex);

  // Every row of `context`, with its running count this session. Reports without taking, so no
  // mark moves. O(rows): a reporting call taken between sessions, never in a search.
  [[nodiscard]] std::vector<NNCacheObservationCount> harvestFor(uint32_t contextIndex) const;

  // THE OBSERVATIONS OF `context` THAT HAVE NOT REACHED THE COUNT LOG YET, taking them as it
  // reports them: per row, the count minus its persisted mark, with the mark then advanced.
  //
  // A KEY WITH NOTHING TO SAY YIELDS NO ROW. A count-log record is an INCREMENT and a record's
  // presence raises that key's `sessions`, so a row for a key that was not presented since the
  // last dump would credit it with a session it did not have.
  //
  // THE MARK IS WHY THE COUNT IS NOT RESET. A count of zero is this structure's free-row
  // marker, so zeroing an occupied row would cut the probe chain of every key placed behind
  // it. The mark lives in bytes the row's alignment already spent.
  [[nodiscard]] std::vector<NNCacheObservationCount> takeUnpersistedFor(uint32_t contextIndex);

  // Would takeUnpersistedFor yield anything? Asked without taking and without moving a mark.
  // True exactly when that take would yield at least one row, so the two cannot disagree about
  // a state (ADR-0012 P1). This is what a detach refusal reads: taking the delta is what makes
  // it safe to append, so a refusal may not ask the consuming question.
  [[nodiscard]] bool anyUnpersistedFor(uint32_t contextIndex) const;

  // Every row of every context, for a whole-table one-shot report (cache_stats). Rows of two
  // contexts for one key are two rows, deliberately: they are two presentations of one
  // position under two cards, and summing them would report a number belonging to neither.
  [[nodiscard]] std::vector<NNCacheObservationCount> harvestAll() const;

  [[nodiscard]] int64_t unrecordedObservations() const;
  [[nodiscard]] size_t structureBytes() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

// The exact resident cost of the row array a recorder of this size allocates, excluding its
// mutex pool. Named here so the bound is stated in one place and can be asserted rather than
// estimated (ADR-0012 P1).
size_t observationRecorderBytes(int powerOfTwo);

#endif  // NEURALNET_NNCACHEOBSERVATIONS_H_
