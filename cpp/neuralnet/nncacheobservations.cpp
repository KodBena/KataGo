#include "../neuralnet/nncacheobservations.h"

#include <atomic>
#include <mutex>

#include "../core/global.h"
#include "../search/mutexpool.h"

// See nncacheobservations.h for what an observation is, where the one door that counts one
// lives, and why the lifetime total is the count log file's fact and not this structure's.

using namespace std;

//-------------------------------------------------------------------------------------
// The ledger value
//-------------------------------------------------------------------------------------

NNCacheObservationLedger::NNCacheObservationLedger(
  NNCacheObservationLedgerDisposition disposition,
  vector<NNCacheObservationCount> entries,
  int64_t unrecordedObservations
)
  :disposition_(disposition), entries_(std::move(entries)), unrecordedObservations_(unrecordedObservations)
{}

NNCacheObservationLedger NNCacheObservationLedger::notObserved() {
  return NNCacheObservationLedger(
    NNCacheObservationLedgerDisposition::NotObserved, vector<NNCacheObservationCount>(), 0
  );
}

NNCacheObservationLedger NNCacheObservationLedger::observed(
  vector<NNCacheObservationCount> entries, int64_t unrecordedObservations
) {
  return NNCacheObservationLedger(
    NNCacheObservationLedgerDisposition::Observed, std::move(entries), unrecordedObservations
  );
}

const vector<NNCacheObservationCount>& NNCacheObservationLedger::entries() const {
  if(disposition_ != NNCacheObservationLedgerDisposition::Observed)
    throw StringError(
      "NNCacheObservationLedger: no context is attached to this cache, so it observes nothing "
      "and has no rows to hand out. Check disposition() before asking; an empty row list would "
      "be indistinguishable from a session in which nothing was presented."
    );
  return entries_;
}

int64_t NNCacheObservationLedger::unrecordedObservations() const {
  if(disposition_ != NNCacheObservationLedgerDisposition::Observed)
    throw StringError(
      "NNCacheObservationLedger: no context is attached to this cache, so it has no unrecorded "
      "observation count to report either."
    );
  return unrecordedObservations_;
}

//-------------------------------------------------------------------------------------
// The recorder
//-------------------------------------------------------------------------------------

class NNCacheObservationRecorder::Impl {
 public:
  Impl(int powerOfTwo, int mutexPoolSizePowerOfTwo)
    :rows_(((size_t)1) << powerOfTwo),
     mask_((((uint64_t)1) << powerOfTwo) - 1),
     mutexPool_(((uint32_t)1) << mutexPoolSizePowerOfTwo),
     mutexMask_((((uint32_t)1) << mutexPoolSizePowerOfTwo) - 1),
     unrecorded_(0)
  {}

  // How many slots forward of home a key may be placed. A bounded window keeps the write O(1)
  // and keeps a nearly-full structure from degrading into a scan; overflowing it is reported,
  // not absorbed. The same window the attribution recorder and the hit ledger use.
  static const uint32_t PROBE_WINDOW = 16;

  // THE HOT-PATH CALL. One mixed home, one pooled mutex, one bounded probe over 32-byte rows
  // -- two to a cache line -- and one increment.
  void observe(Hash128 key, uint32_t contextIndex) {
    const uint64_t home = homeOf(key, contextIndex);
    std::mutex& mutex = mutexPool_.getMutex((uint32_t)home & mutexMask_);
    std::lock_guard<std::mutex> lock(mutex);
    for(uint32_t step = 0; step < PROBE_WINDOW; step++) {
      Row& row = rows_[(home + step) & mask_];
      // count == 0 marks a free row: a row is only ever written with a count of at least one,
      // so no occupied row can be mistaken for a free one. It is the same marker HitLedger
      // uses, and the persisted mark below is why an occupied row is never zeroed back.
      if(row.count == 0) {
        row.key = key;
        row.contextIndex = contextIndex;
        row.count = 1;
        row.persisted = 0;
        return;
      }
      if(row.contextIndex == contextIndex && row.key == key) {
        // Saturating rather than wrapping. The count-log record is 32-bit, and the largest
        // lifetime count in the operator's whole corpus is 11,997, so this is a refusal about
        // an unreachable state rather than a policy about a likely one -- but a wrapped count
        // would be four billion presentations lost with nothing to show for it (ADR-0002).
        if(row.count != 0xFFFFFFFFu)
          row.count += 1;
        return;
      }
    }
    unrecorded_.fetch_add(1, std::memory_order_relaxed);
  }

  vector<NNCacheObservationCount> harvestFor(uint32_t contextIndex) const {
    vector<NNCacheObservationCount> out;
    for(size_t i = 0; i < rows_.size(); i++) {
      std::mutex& mutex = mutexPool_.getMutex((uint32_t)i & mutexMask_);
      std::lock_guard<std::mutex> lock(mutex);
      if(rows_[i].count == 0 || rows_[i].contextIndex != contextIndex)
        continue;
      NNCacheObservationCount row;
      row.key = rows_[i].key;
      row.observations = rows_[i].count;
      out.push_back(row);
    }
    return out;
  }

  vector<NNCacheObservationCount> takeUnpersistedFor(uint32_t contextIndex) {
    vector<NNCacheObservationCount> out;
    for(size_t i = 0; i < rows_.size(); i++) {
      std::mutex& mutex = mutexPool_.getMutex((uint32_t)i & mutexMask_);
      std::lock_guard<std::mutex> lock(mutex);
      Row& row = rows_[i];
      if(row.count == 0 || row.contextIndex != contextIndex)
        continue;
      if(row.count <= row.persisted)
        continue;
      NNCacheObservationCount outRow;
      outRow.key = row.key;
      outRow.observations = row.count - row.persisted;
      row.persisted = row.count;
      out.push_back(outRow);
    }
    return out;
  }

  bool anyUnpersistedFor(uint32_t contextIndex) const {
    for(size_t i = 0; i < rows_.size(); i++) {
      std::mutex& mutex = mutexPool_.getMutex((uint32_t)i & mutexMask_);
      std::lock_guard<std::mutex> lock(mutex);
      // The same two comparisons takeUnpersistedFor makes, in the same order, against the same
      // two words under the same lock -- and no assignment, so no mark moves (ADR-0012 P1).
      if(rows_[i].count == 0 || rows_[i].contextIndex != contextIndex)
        continue;
      if(rows_[i].count > rows_[i].persisted)
        return true;
    }
    return false;
  }

  vector<NNCacheObservationCount> harvestAll() const {
    vector<NNCacheObservationCount> out;
    for(size_t i = 0; i < rows_.size(); i++) {
      std::mutex& mutex = mutexPool_.getMutex((uint32_t)i & mutexMask_);
      std::lock_guard<std::mutex> lock(mutex);
      if(rows_[i].count == 0)
        continue;
      NNCacheObservationCount row;
      row.key = rows_[i].key;
      row.observations = rows_[i].count;
      out.push_back(row);
    }
    return out;
  }

  int64_t unrecordedObservations() const { return unrecorded_.load(std::memory_order_relaxed); }

  static size_t rowBytes() { return sizeof(Row); }

  size_t structureBytes() const {
    return rows_.size() * rowBytes() + ((size_t)mutexMask_ + 1) * sizeof(std::mutex);
  }

 private:
  // WHY THE CONTEXT IS MIXED IN RATHER THAN MASKED ALONGSIDE. Two cards share about 3.7% of
  // their keys, and a shared position presented under each is two rows that must both exist.
  // Landing them on one home would make each burn the other's probe window at exactly the
  // occupancy where the window is scarce. One odd-multiplier mix spreads them independently
  // and costs one multiply and one xor -- the golden-ratio constant, which is the house
  // splitmix/fibonacci multiplier and is used here for its avalanche, not for any claim of
  // cryptographic strength.
  uint64_t homeOf(Hash128 key, uint32_t contextIndex) const {
    return (key.hash0 ^ ((uint64_t)contextIndex * 0x9E3779B97F4A7C15ull)) & mask_;
  }

  // 32 BYTES, AND THE PERSISTED MARK COSTS NONE OF THEM. 16 of key, 4 of count, 4 of mark and
  // 4 of context index is 28 bytes of fields in a structure whose alignment is 8, so it
  // occupies 32 either way; the mark and the context index live in bytes this row was already
  // paying for. Two rows to a 64-byte cache line, which is what keeps a probe walk cheap. The
  // relation is asserted against sizeof(Row) in the test suite rather than trusted from here.
  struct Row {
    Hash128 key;
    uint32_t count;
    // How much of `count` has already been written to the count log. See takeUnpersistedFor.
    uint32_t persisted;
    uint32_t contextIndex;
    Row() : key(), count(0), persisted(0), contextIndex(0) {}
  };

  std::vector<Row> rows_;
  uint64_t mask_;
  mutable MutexPool mutexPool_;
  uint32_t mutexMask_;
  std::atomic<int64_t> unrecorded_;
};

int NNCacheObservationRecorder::defaultPowerOfTwo() {
  return 20;
}

int64_t NNCacheObservationRecorder::sizingReferenceKeys() {
  // ONE HOME FOR THE CORPUS FIGURE. It is the attribution recorder's, read rather than
  // re-typed: both structures are sized against the same fact, the largest real card in the
  // operator's corpus, and a second copy of 291129 here is the drift ADR-0012 P1 names.
  return NNCacheAttributionRecorder::sizingReferenceKeys();
}

int NNCacheObservationRecorder::maxLoadFactorPercent() {
  return NNCacheAttributionRecorder::maxLoadFactorPercent();
}

size_t NNCacheObservationRecorder::rowBytes() {
  return Impl::rowBytes();
}

size_t observationRecorderBytes(int powerOfTwo) {
  return (((size_t)1) << powerOfTwo) * NNCacheObservationRecorder::rowBytes();
}

NNCacheObservationRecorder::NNCacheObservationRecorder(int powerOfTwo, int mutexPoolSizePowerOfTwo)
  :impl_(nullptr)
{
  if(powerOfTwo < 0 || powerOfTwo > 40)
    throw StringError(
      "NNCacheObservationRecorder: powerOfTwo must be between 0 and 40; got " +
      Global::intToString(powerOfTwo) + "."
    );
  if(mutexPoolSizePowerOfTwo < 0 || mutexPoolSizePowerOfTwo > powerOfTwo)
    throw StringError(
      "NNCacheObservationRecorder: mutexPoolSizePowerOfTwo must be between 0 and powerOfTwo (" +
      Global::intToString(powerOfTwo) + "); got " + Global::intToString(mutexPoolSizePowerOfTwo) + "."
    );
  impl_.reset(new Impl(powerOfTwo, mutexPoolSizePowerOfTwo));
}

NNCacheObservationRecorder::~NNCacheObservationRecorder() {}

void NNCacheObservationRecorder::observe(Hash128 key, uint32_t contextIndex) {
  impl_->observe(key, contextIndex);
}

vector<NNCacheObservationCount> NNCacheObservationRecorder::harvestFor(uint32_t contextIndex) const {
  return impl_->harvestFor(contextIndex);
}

vector<NNCacheObservationCount> NNCacheObservationRecorder::takeUnpersistedFor(uint32_t contextIndex) {
  return impl_->takeUnpersistedFor(contextIndex);
}

bool NNCacheObservationRecorder::anyUnpersistedFor(uint32_t contextIndex) const {
  return impl_->anyUnpersistedFor(contextIndex);
}

vector<NNCacheObservationCount> NNCacheObservationRecorder::harvestAll() const {
  return impl_->harvestAll();
}

int64_t NNCacheObservationRecorder::unrecordedObservations() const {
  return impl_->unrecordedObservations();
}

size_t NNCacheObservationRecorder::structureBytes() const {
  return impl_->structureBytes();
}
