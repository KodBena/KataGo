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
// it: one DEMAND for one position, under one context, counted once whether the cache answered
// it or a forward pass had to.
//
// A DEMAND, NOT A FORWARD PASS, and the distinction is the operator's own ruling (ledger row
// 1814) rather than a reading chosen here. The two coincide almost everywhere and come apart on
// exactly one shipped path: averageMultipleSymmetries runs N forward passes of ONE key for one
// root query. Under "forward passes" that is N; under "demands" it is 1. The ruling is 1 --
// one query asked about one position once, and the N passes are how the engine chose to answer
// it -- because the number this log exists to feed is "how much work would carrying this
// position save the NEXT session", and carrying it saves that query once however many passes it
// was averaged over. In-search TRANSPOSITIONS are a different matter and remain separate
// demands: each is a genuinely separate arrival at the position.
//
// It is emphatically not a CACHE HIT, which is what this log used to count; see below.
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
// WHERE AN OBSERVATION IS COUNTED. At the MINT of an NNCachePresentation (see below) --
// NNCacheTable::present -- which NNEvaluator::evaluate performs once, immediately after the
// position's hash is final and before any level is consulted. The presentation is then the only
// thing the request-path get and set will accept, so on that surface presenting a position and
// counting it are one act rather than two that a caller must remember to pair.
//
// IT IS DELIBERATELY NOT COUNTED INSIDE get() AND set(), and the three reasons are worth
// keeping because they are what the mint had to satisfy:
//
//   A REQUEST IS ONE PRESENTATION EVEN WHEN IT MAKES TWO CALLS. An ordinary miss makes a get
//   and then a set of the evaluation it computed; the ownership-map fall-through in
//   NNEvaluator::evaluate makes a get, rejects the hit, and sets a fuller result. Counting
//   both calls would make one fresh evaluation read as two observations -- and the whole
//   point of the currency is that ONE fresh demand is ONE observation, so that a second
//   SESSION is what carries a key over a seen-twice threshold.
//
//   AND A REQUEST IS ONE PRESENTATION EVEN WHEN IT MAKES NO GET. A caller passing skipCache
//   consults no level and sets unconditionally. Counting only gets would lose it.
//
//   MEANWHILE NOT EVERY set() IS A PRESENTATION AT ALL. An attach FILLS level 1 from the
//   context's own container (NNCacheEntryProvenance::LoadedFromContainer); nobody asked for
//   those positions. Counting them would make attach -> detach -> attach with no traffic
//   raise every loaded key's count by one per cycle, when the ratified contract is that such
//   a cycle is a tracking NO-OP. Under the mint this one is foreclosed rather than avoided:
//   that fill holds no presentation and cannot obtain one, because minting is the table's own
//   act and the fill has nothing to present.
//
// WHAT THE FIRST VERSION OF THIS FILE GOT WRONG, kept here because the correction is the
// reason the type exists. It counted in a single line inside evaluate() and asserted, in
// capitals, "COUNTED HERE AND NOWHERE ELSE". An out-of-frame audit reproduced from the code
// that this was already false: averageMultipleSymmetries re-enters evaluate() once per symmetry
// for one root query, all N iterations compute the identical key, and each one counted. The
// three arguments above were all correct and all insufficient -- they established that get/set
// is the wrong seam, and stopped one step short of making the right seam unforgeable. That step
// is NNCachePresentation, and the two named mints are where a fan-out now has to say which it
// is (ledger row 1814 rules that answer: one demand).
//
// WHAT THIS STRUCTURE HOLDS IS THE UNPERSISTED DELTA, NOT THE LIFETIME TOTAL. The lifetime
// total's one home is the context's .nncounts file, which is an additive log: a dump appends
// what has accrued since the last one, and a load sums every block (ADR-0012 P1). So a
// session that attaches, observes nothing and detaches appends nothing and changes no total,
// and a key observed once in session A and once in session B reads as 2 from the file in
// session B -- which is what makes the cross-session bootstrap above work without this
// structure ever holding a number it did not itself count.
//
// WHAT IT COSTS, MEASURED, AND WHERE THE COST GOES. runnncachetwolevelbench's arm D, on this
// box: three rounds, each the minimum of 11 interleaved rep-pairs of 1e6 requests, both arms
// in one process against the same table and the same key stream (so no cross-build flag
// mismatch can enter the difference -- this project has been handed one fake 2x regression
// that way). Ranges are across the three rounds, not error bars:
//
//   NO CONTEXT ATTACHED -- plain play, the overwhelmingly common configuration:
//     +0.4 to +0.5 ns per evaluation request, against an ~10.8 ns lookup loop. That is the
//     inlined null test on a pointer that stays null for the life of the process, and it is
//     the whole of what the default configuration pays.
//   A CONTEXT ATTACHED, THE REQUEST NAMING NONE: +1.8 to +2.1 ns. One more test and no work.
//   A CONTEXT ATTACHED AND NAMED: +53.9 to +56.7 ns. This is the real cost, and it is paid
//     only by the deployment this feature exists for.
//
// THESE ARE THE POST-MINT FIGURES. An earlier revision counted through a bare
// NNCacheTable::observe(hash, attribution) call and measured +0.9-1.4 / +1.9-2.1 / +64.6-67.7 on
// the same three arms; introducing NNCachePresentation and routing get/set through it moved
// nothing outside the run-to-run spread these arms already show. That is the expected result and
// it is stated because it was not assumed: the presentation is a 16-byte move-only value the
// compiler elides into the same register the hash was already in, and the request-path get/set
// are inline forwards to the same virtuals.
//
// THE 65 ns IS ATTRIBUTED RATHER THAN LEFT AS A LUMP (arm D3, the recorder alone, same three
// rounds): at a table size that fits in L2, with the same lock and the same probe length, one
// observe costs 17.3-17.5 ns; at the production 2^20 rows it costs 35.5-36.8 ns. So roughly
// 18-19 ns is the random access into 33.5 MB -- irreducible for an exact per-key count -- and
// roughly 17 ns is the pooled mutex plus the mix and the compare.
//
// THE ~19 ns BETWEEN D3's 36 AND ARM D's 55 IS NOT ACCOUNTED FOR, AND THAT IS SAID RATHER THAN
// EXPLAINED AWAY. The HYPOTHESIS is the two structures evicting each other's lines -- arm D
// walks a cache table and the ledger in the same loop where D3 walks the ledger alone -- and it
// is consistent with the gap's size. NOTHING MEASURED IT: no profiler was run, no cache-miss
// count was ever observed, and no arm isolates it. The same posture nncachetwolevel.h already
// takes for its own 5 ns ("the effect is measured; the explanation below is not"), for the same
// reason: an explanation stated with the confidence of a measurement is the thing a later
// reader acts on without re-checking. The arm that would settle it runs the two loops against
// deliberately disjoint memory and compares; anyone acting on the hypothesis should build it
// first.
//
// WHY IT IS PAID RATHER THAN ENGINEERED AWAY, and what would change that. It buys the whole
// mechanism: 55 ns against the 2.4-2.8 ms forward pass a carried position avoids, which is a
// ratio of about 1:45,000. The ~17 ns of lock IS removable, by a lock-free row protocol -- a
// 32-bit tag word claimed by CAS, published release/acquire after the key and the context
// index are written, with the count an atomic. THE REASON IT IS FILED AND NOT BUILT is not the
// arithmetic: it is that this structure must be EXACT and its consuming take
// (takeUnpersistedFor) is not guaranteed to be at rest -- cache_dump reports its own
// openRequestsAtDump, so a dump can run with requests in flight. A mutex makes the take/observe
// race safe by construction; the lock-free version has to get the take's read-modify-write of
// the persisted mark right against a concurrent increment, which is the band-aid-on-band-aid
// shape ADR-0012 P5 declines without a witnessed need. Build it when a measurement shows 17 ns
// per request mattering to something; the bench that would show it is arm D3.
//
// EXACT, NOT A SKETCH. Every row carries the FULL 128-bit key and the context index beside
// its count, so a count is never attributed to the wrong key or the wrong card, and an
// observation that cannot be given a row is REFUSED and counted in unrecordedObservations()
// rather than overwriting somebody else's row (ADR-0002). The same posture the attribution
// recorder and the two-level hit ledger already carry, for the same reason.

//-------------------------------------------------------------------------------------
// The presentation itself, as a value
//-------------------------------------------------------------------------------------

// ONE POSITION, PRESENTED ONCE, AS A VALUE THAT HAS ALREADY BEEN COUNTED.
//
// WHY THE COUNTED ACT IS A VALUE AND NOT A CALL. The first version of this file counted in a
// single line inside NNEvaluator::evaluate and asserted, in capitals, that this was "the one
// door". An out-of-frame audit found the assertion already false in shipped code -- see
// NNCachePresentationRole below for the defect -- and the reason it could be false is that
// nothing but prose held the door. A counted call is a convention every future presenting path
// must be told about; a counted VALUE is one the compiler asks for. The request-path
// NNCacheTable::get and NNCacheTable::set take one of these and nothing else, so "present a
// position without counting it" is not an expression on that surface (ADR-0000 Rule 2a).
//
// MOVE-ONLY, so one presentation cannot be silently duplicated into two. It is minted by
// exactly two NNCacheTable calls and by nothing else -- present(), which counts, and
// presentAgainForSameRequest(), which does not -- and having two named mints is the point: a
// caller that fans one request out over several evaluations has to WRITE which one each of them
// is, at the call site, in a line a reviewer can see (NNEvaluator::averageMultipleSymmetries is
// the only such caller today).
//
// WHAT IT DOES NOT CLOSE, stated rather than left for the next audit. The raw
// NNCacheTable::get(Hash128, ...) / set(shared_ptr) virtuals remain public: they are how the
// decorator shapes compose one another (nncacheadmission.cpp, nncachetrace.cpp, and the
// two-level table's calls into level 1) and how ~144 test call sites drive the tables directly.
// A request-path caller could still reach them -- but only by re-deriving a bare Hash128 it has
// no other reason to hold, which is the "wrong word written at the call site" standard this
// codebase already accepts for NNCacheObservationDelta::ofDeltaRows. Closing it outright means
// a composition permit threaded through every table shape and a test seam for the 144 sites,
// which is a hot-path signature change and is filed rather than done here.
class NNCachePresentation {
 public:
  NNCachePresentation(NNCachePresentation&&) noexcept = default;
  NNCachePresentation& operator=(NNCachePresentation&&) noexcept = default;
  NNCachePresentation(const NNCachePresentation&) = delete;
  NNCachePresentation& operator=(const NNCachePresentation&) = delete;

  // The position this presentation is of. Handing out the key is not a hole: what the type
  // protects is the ACT of presenting, and a caller holding this has already performed it.
  [[nodiscard]] Hash128 key() const { return key_; }

 private:
  friend class NNCacheTable;
  explicit NNCachePresentation(Hash128 key) : key_(key) {}
  Hash128 key_;
};

//-------------------------------------------------------------------------------------
// Whether an evaluation IS the presentation, or only serves one
//-------------------------------------------------------------------------------------

// A CLOSED TWO-CASE DISPOSITION, because "one request is one observation" is a claim about
// REQUESTS and NNEvaluator::evaluate is not always one.
//
// THE DEFECT THIS TYPE EXISTS FOR, found by an out-of-frame audit of the first version of this
// file and reproduced from the code rather than argued: NNEvaluator::averageMultipleSymmetries
// re-enters evaluate() once PER SYMMETRY for a single root query, with skipCache on and with
// the same cache attribution deliberately re-supplied each time. NNInputs::getHash folds in
// board, rules, komi and the search-shape flags and NOT symmetry -- which is exactly WHY that
// path passes skipCache, since the cache cannot tell the symmetries apart -- so all N
// iterations present the IDENTICAL key. Counting each one made a single root visit at
// rootNumSymmetriesToSample = 8 write eight observations of one position in one session, which
// clears the default minObservations(2) threshold inside the very session that evaluated it:
// verbatim the outcome this file says the currency exists to prevent. The old retrieval
// currency was accidentally immune (skipCache means no get(), so no retrieval), so the
// inflation was introduced by the currency change and by nothing before it.
//
// WHY THIS AND NOT A BOOL, and why not "just don't count skipCache". A bool on the buffer would
// be a second fact remembered by convention with no name; and a skipCache-shaped rule would be
// wrong in the other direction, because a lone skipCache caller IS a presentation and must
// count. The distinction is not about the cache at all -- it is about whether this evaluation
// is the request or is one of several serving one request -- and only the caller that fans out
// knows, so only that caller can say.
//
// WHAT IT DOES NOT CLOSE, named rather than left for the next audit to find. The safe case is
// the DEFAULT, so a future fan-out path that forgets to say so over-counts rather than
// under-counts -- loud in the file, but wrong. The construction that would foreclose the class
// outright is a move-only presentation VALUE minted where the hash is computed and required by
// type to reach get()/set(), so that a fan-out has to decide explicitly whether its N
// evaluations are one presentation or N. That is the fix this type stops one step short of; it
// is not built here because it changes the signature of evaluate() at the fifteen call sites
// NNResultBuf::cacheAttribution's own comment enumerates, for no behavioural change at any of
// them. REVISIT the moment a second fan-out path exists: two callers with the same obligation
// and no compiler holding them to it is where this stops being adequate.
enum class NNCachePresentationRole {
  // This evaluation IS the request. The default, and what every ordinary path is.
  ThePresentation,
  // This evaluation is one of several the caller is running for ONE request, and that request
  // has already been counted. Set by NNEvaluator::averageMultipleSymmetries for every iteration
  // after its first, and by nothing else today.
  ServesACountedPresentation,
};

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
