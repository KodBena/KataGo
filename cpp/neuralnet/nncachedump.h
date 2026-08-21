#ifndef NEURALNET_NNCACHEDUMP_H_
#define NEURALNET_NNCACHEDUMP_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "../core/hash.h"
#include "../neuralnet/nncache.h"
#include "../neuralnet/nncachecontext.h"
#include "../neuralnet/nncachecountlog.h"
#include "../neuralnet/nnevalcontainer.h"

// THE WRITE SIDE OF THE CACHE PROTOCOL: which of a context's level-1 entries reach disk, and
// which of them are owed at all.
//
// WHAT THIS FILE IS FOR, in one sentence: nnevalcontainer.h can append any block of entries a
// caller hands it; this file is where the block comes from, and it is the one place that
// answers the two questions a caller must not be trusted to answer for itself.
//
// QUESTION ONE, THE POLICY: IS THIS ENTRY WORTH KEEPING? The read side already puts this
// judgment in the client's hands -- nnCacheLevelZeroBound is a client-supplied bound the
// engine enforces, "mechanism, not policy", because what belongs in a study session's cache
// is the client's business and custody of the files is KataGo's. The write side had NO such
// predicate at all: every entry a context earned was owed to disk. NNCacheDiskAdmission is
// that predicate, in the same stance and in the same currency -- recorded OBSERVATIONS, read
// from the count log, compared through the one shared NNCacheObservationThreshold the read
// side's minObservations bound uses. "Store only what has been seen at least twice" is then
// NNCacheDiskAdmission::minObservations(2): the operator's own typical policy, and the
// DEFAULT a cache_dump that names no admission gets -- expressed as an argument, never
// compiled in as a constant here.
//
// QUESTION TWO, THE FACT: IS THIS ENTRY ALREADY ON DISK? This is not a policy and no client
// gets to answer it. An attach fills level 1 with the container keys its level-0 bound did
// not take, and those entries are byte-for-byte in the very file a dump appends to; a dump
// specified as "the context's level-1-owned entries" therefore re-appends the whole filled
// remainder on every attach-dump cycle. The answer is the persisted mark recorded at
// admission time (NNCacheEntryProvenance) and read here -- never a containment probe against
// level 0, which cannot answer it, because a selection bound makes container membership
// strictly larger than level-0 membership.
//
// THE TWO QUESTIONS ARE ASKED IN THIS ORDER AND THE ORDER IS NOT ARBITRARY. Already-on-disk
// is decided first, because it is a fact and it is free; the threshold is applied only to
// entries a dump would otherwise owe. An entry that is on disk and below the threshold is
// reported as already-persisted, not as rejected: this predicate governs what a dump WRITES,
// and it is not an eviction policy for what a container already holds. Removing content is
// what compaction and a client rewriting its own card are for.

//-------------------------------------------------------------------------------------
// The admission predicate
//-------------------------------------------------------------------------------------

// WHAT A CLIENT LETS ONTO DISK, in the client's own currency.
//
// The closed set of kinds is the same shape NNCacheLevelZeroBound has, for the same reason:
// one predicate per dump, applied by the predicate itself rather than by a caller switching
// over its kind, so each kind's rule has exactly one home (ADR-0012 P1/P8).
class NNCacheDiskAdmission {
 public:
  // Every entry the context earned and does not already have on disk. The default, and what
  // the write side did before a predicate existed.
  static NNCacheDiskAdmission all();
  // Every entry whose count-log record clears `observations` observations. minObservations(2)
  // is "store only entries seen at least twice"; minObservations(0) is all().
  static NNCacheDiskAdmission minObservations(uint64_t observations);

  // Whether an entry the count log records `recordedObservations` observations for is
  // admitted. A key the count log does not mention is passed as zero -- see
  // NNCacheObservationThreshold, which owns that rule for both sides.
  [[nodiscard]] bool admits(uint64_t recordedObservations) const;

  // For a report or a refusal: what was asked for, in words.
  [[nodiscard]] std::string describe() const;

 private:
  enum class Kind { All, MinObservations };
  NNCacheDiskAdmission(Kind kind, uint64_t observations);

  Kind kind_;
  uint64_t observations_;
};

//-------------------------------------------------------------------------------------
// The plan
//-------------------------------------------------------------------------------------

// What a dump of one context would write, and what it would leave out and why.
//
// EVERY EXCLUSION IS COUNTED AND NAMED. A dump that wrote 12 of a context's 40,000 earned
// keys and reported only "12 written" would be indistinguishable from a dump that lost
// 39,988 of them. Each figure below is a different fact with a different remedy, so they are
// separate numbers rather than one "skipped" total (ADR-0002; ADR-0012 P11).
struct NNCacheEvaluationDumpPlan {
  // The entries to append, and their keys in the same order. Kept as two parallel lists
  // because appendBlock takes the entries and markPersisted takes the keys, and deriving
  // either from the other at the call site would be a second walk over ~69 MB of payload.
  std::vector<std::shared_ptr<const NNOutput>> entries;
  std::vector<Hash128> keys;

  // Earned keys whose bytes are already in the container: filled from it by an attach, or
  // written by an earlier dump of this attachment. THE NUMBER THIS WHOLE FILE EXISTS FOR.
  int64_t alreadyPersisted;
  // Earned, not on disk, and refused by the admission predicate.
  int64_t belowThreshold;
  // Earned, not on disk, admitted -- and no longer in the table. A capacity sweep dropped it
  // after it was earned, or a SecondSighting admission declined it on its first sighting.
  // The attribution ledger is a SUPERSET of what the table holds and says so; this is the
  // difference, measured rather than assumed.
  int64_t notResident;
};

//-------------------------------------------------------------------------------------
// The acts
//-------------------------------------------------------------------------------------

// Decides what a dump of `context` owes, without writing anything and without counting a
// single retrieval against any entry it reads (NNCacheTable::peek).
//
// `observations` is the count log's own rows for this context -- NNCacheCountLogContents
//::rows(). IT IS READ AFTER THIS DUMP'S COUNTS HAVE BEEN APPENDED, so an entry earned and
// retrieved twice within this very session is admitted on the strength of this session
// rather than having to wait for the next one. That ordering is a property of the caller,
// which is why it is stated here rather than assumed: counts first, reload, then evaluations.
//
// Throws StringError for a context this table did not attach.
[[nodiscard]] NNCacheEvaluationDumpPlan nnCachePlanEvaluationDump(
  NNCacheTable& table,
  const NNCacheContextId& context,
  const NNCacheDiskAdmission& admission,
  const std::vector<NNCacheCountRow>& observations
);

// What one evaluation dump did.
struct NNCacheEvaluationDumpResult {
  NNCacheEvaluationDumpPlan plan;
  NNEvalContainerAppendResult append;
  // Rows the persisted mark was actually set on. Equal to plan.keys.size() unless a key was
  // earned while the recorder's probe window was full, in which case it has no row to mark
  // and will be offered again next dump -- reported rather than absorbed.
  int64_t marked;
};

// Plans, appends, and only then marks what it wrote.
//
// THE ORDER IS THE CRASH CONTRACT. The mark is set after appendBlock has fsync'd and
// returned; if the append throws, nothing is marked and the entries are offered again. The
// converse order would drop an entry from every future dump on the strength of a write that
// did not happen.
//
// An empty plan appends NOTHING AT ALL -- not a zero-entry block. A second dump with no
// intervening evaluation therefore leaves the container byte-identical, which is the property
// the persisted mark exists to give.
//
// Throws StringError for whatever the container refuses; see NNEvalContainer::appendBlock.
[[nodiscard]] NNCacheEvaluationDumpResult nnCacheDumpEvaluations(
  const NNEvalContainer& container,
  NNCacheTable& table,
  const NNCacheContextId& context,
  const NNCacheDiskAdmission& admission,
  const std::vector<NNCacheCountRow>& observations
);

#endif  // NEURALNET_NNCACHEDUMP_H_
