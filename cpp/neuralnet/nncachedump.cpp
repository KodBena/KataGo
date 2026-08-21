#include "../neuralnet/nncachedump.h"

#include <unordered_map>

#include "../core/global.h"

// See nncachedump.h for the two questions this file answers and why only one of them is the
// client's to answer.

using namespace std;

//-------------------------------------------------------------------------------------
// The admission predicate
//-------------------------------------------------------------------------------------

NNCacheDiskAdmission::NNCacheDiskAdmission(Kind kind, uint64_t observations)
  :kind_(kind), observations_(observations)
{}

NNCacheDiskAdmission NNCacheDiskAdmission::all() {
  return NNCacheDiskAdmission(Kind::All, 0);
}

NNCacheDiskAdmission NNCacheDiskAdmission::minObservations(uint64_t observations) {
  return NNCacheDiskAdmission(Kind::MinObservations, observations);
}

bool NNCacheDiskAdmission::admits(uint64_t recordedObservations) const {
  switch(kind_) {
  case Kind::All:
    return true;
  case Kind::MinObservations:
    // THE COMPARISON IS NOT WRITTEN HERE. NNCacheObservationThreshold owns "this key has been seen
    // often enough" for the read side's level-0 bound and for this predicate alike, so the two
    // cannot drift on the uncounted-key boundary case (ADR-0012 P1).
    return NNCacheObservationThreshold::of(observations_).admits(recordedObservations);
  default:
    break;
  }
  throw StringError("NNCacheDiskAdmission: an admission kind this build does not implement.");
}

string NNCacheDiskAdmission::describe() const {
  switch(kind_) {
  case Kind::All:
    return "every entry this context earned and has not already stored";
  case Kind::MinObservations:
    return NNCacheObservationThreshold::of(observations_).describe();
  default:
    break;
  }
  throw StringError("NNCacheDiskAdmission: an admission kind this build does not implement.");
}

//-------------------------------------------------------------------------------------
// The plan
//-------------------------------------------------------------------------------------

namespace {

// The count log's rows as a lookup by key. Built once per dump rather than scanned per key:
// at the operator's largest card that is 291,129 keys against 291,129 candidates, and the
// quadratic form of the same join is the difference between a dump and a stall.
unordered_map<uint64_t, uint64_t> observationsByKeyHash(const vector<NNCacheCountRow>& observations) {
  unordered_map<uint64_t, uint64_t> out;
  out.reserve(observations.size() * 2);
  for(size_t i = 0; i < observations.size(); i++) {
    // Keyed on hash0 alone, and the full key is NOT re-checked against it, because it does
    // not need to be: both sides of this join are 128-bit NN cache keys drawn from the same
    // space, and a hash0 collision between two of them is the same event as a cache key
    // collision, which the whole cache already treats as not happening. What a collision
    // would cost here is one entry admitted or refused on its colliding twin's count -- a
    // policy misapplication to one entry, not a wrong evaluation served.
    out[observations[i].key.hash0] = observations[i].observations;
  }
  return out;
}

}  // namespace

NNCacheEvaluationDumpPlan nnCachePlanEvaluationDump(
  NNCacheTable& table,
  const NNCacheContextId& context,
  const NNCacheDiskAdmission& admission,
  const vector<NNCacheCountRow>& observations
) {
  NNCacheEvaluationDumpPlan plan;
  plan.alreadyPersisted = 0;
  plan.belowThreshold = 0;
  plan.notResident = 0;

  // Both queries refuse a context this table did not attach, from their one home. The
  // already-persisted figure is the difference between the two lists rather than a second
  // walk that could disagree with either.
  const vector<Hash128> earned = table.attributedKeysFor(context);
  const vector<Hash128> owed = table.unpersistedKeysFor(context);
  plan.alreadyPersisted = (int64_t)earned.size() - (int64_t)owed.size();

  const unordered_map<uint64_t, uint64_t> recordedByKeyHash = observationsByKeyHash(observations);

  plan.entries.reserve(owed.size());
  plan.keys.reserve(owed.size());
  for(size_t i = 0; i < owed.size(); i++) {
    const Hash128 key = owed[i];
    const unordered_map<uint64_t, uint64_t>::const_iterator it = recordedByKeyHash.find(key.hash0);
    // A key the count log has never mentioned is passed as zero, not dropped and not guessed
    // at: NNCacheObservationThreshold owns what a threshold makes of that.
    const uint64_t recorded = (it == recordedByKeyHash.end()) ? 0 : it->second;
    if(!admission.admits(recorded)) {
      plan.belowThreshold += 1;
      continue;
    }
    shared_ptr<NNOutput> entry;
    // peek, never get: a dump must not count itself into the counts it is dumping, and must
    // not be served an entry out of the pre-warmed level 0 whose bytes are already in the
    // file it is appending to.
    if(!table.peek(key, entry) || entry == nullptr) {
      plan.notResident += 1;
      continue;
    }
    plan.entries.push_back(shared_ptr<const NNOutput>(entry));
    plan.keys.push_back(key);
  }
  return plan;
}

NNCacheEvaluationDumpResult nnCacheDumpEvaluations(
  const NNEvalContainer& container,
  NNCacheTable& table,
  const NNCacheContextId& context,
  const NNCacheDiskAdmission& admission,
  const vector<NNCacheCountRow>& observations
) {
  NNCacheEvaluationDumpResult result;
  result.plan = nnCachePlanEvaluationDump(table, context, admission, observations);
  result.marked = 0;
  if(result.plan.entries.empty()) {
    // Nothing owed. NOT a zero-entry block: an empty block would still be 32 bytes of framing
    // and one more block for every future load to walk, so a second dump with nothing in
    // between would grow the file forever while adding nothing -- a smaller instance of the
    // very defect this file exists to close.
    result.append.bytesAppended = 0;
    result.append.tornTailBytesDiscarded = 0;
    result.append.rewroteTheFile = false;
    return result;
  }
  // Marked only after this returns: see the header's crash contract.
  result.append = container.appendBlock(result.plan.entries);
  result.marked = table.markPersisted(context, result.plan.keys);
  return result;
}
