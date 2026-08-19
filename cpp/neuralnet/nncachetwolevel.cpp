#include <mutex>

#include "../core/global.h"
#include "../neuralnet/nncachefrozen.h"
#include "../search/mutexpool.h"

// The two-level resolution strategy: a frozen level 0 over an ordinary level 1, presented
// as one NNCacheTable with one hit-count surface across both.
//
// See nncachefrozen.h for the semantics this file implements. The one thing worth
// restating here, because it is what the file is shaped around: LEVEL 0 IS FROZEN, so the
// only way a key can move is from level 0 to level 1, once, irreversibly, and it takes its
// hit count with it. There is no path by which a key comes to be resolvable in both.

namespace {

//-------------------------------------------------------------------------------------
// The level-1 hit ledger
//-------------------------------------------------------------------------------------

// Per-key hit counts for the keys level 1 owns.
//
// Level 1 is whatever shape the operator configured -- direct, probed or chained -- and
// none of those carries a per-entry counter. Giving them one would mean touching four
// table implementations and changing the default table's per-slot memory, which the
// byte-identical-default bar forbids. So the counts live here instead, in one structure
// the two-level table owns, and level 1's own code is untouched.
//
// IT IS EXACT, NOT A SKETCH. Every row holds the FULL 128-bit key beside its count, so a
// count is never attributed to the wrong key, and a key that cannot be given a row is
// REFUSED and counted in unrecorded_ rather than silently overwriting somebody else's row.
// A structure that forgets counts, or that quietly hands a count to a different key, is
// exactly what a persistence layer must not be handed.
//
// A ROW PERSISTS AFTER LEVEL 1 EVICTS ITS KEY, deliberately: "this key was hot this
// session" is the fact a persistence layer wants, and it does not stop being true when a
// capacity sweep drops the entry. The consequence is that the ledger's occupancy is
// bounded by the number of DISTINCT keys hit in a session rather than by level 1's
// capacity, so a long enough session can fill it -- at which point unrecorded_ rises and
// the harvest says by how much, rather than being silently short (ADR-0002).
class HitLedger {
 public:
  HitLedger(int powerOfTwo, int mutexPoolSizePowerOfTwo)
    :rows_(((size_t)1) << powerOfTwo),
     mask_((((uint64_t)1) << powerOfTwo) - 1),
     mutexPool_(((uint32_t)1) << mutexPoolSizePowerOfTwo),
     mutexMask_((((uint32_t)1) << mutexPoolSizePowerOfTwo) - 1),
     unrecorded_(0)
  {}

  // How many slots forward of home a key may be placed. A bounded window keeps the write
  // O(1) and keeps a nearly-full ledger from degrading into a scan; overflowing it is
  // reported, not absorbed.
  static const uint32_t PROBE_WINDOW = 16;

  void add(Hash128 key, uint32_t amount) {
    if(amount == 0)
      return;
    const uint64_t home = key.hash0 & mask_;
    std::mutex& mutex = mutexPool_.getMutex((uint32_t)home & mutexMask_);
    std::lock_guard<std::mutex> lock(mutex);
    for(uint32_t step = 0; step < PROBE_WINDOW; step++) {
      Row& row = rows_[(home + step) & mask_];
      // count == 0 marks a free row: a row is only ever written with a count of at least
      // one, so no occupied row can be mistaken for a free one.
      if(row.count == 0) {
        row.key = key;
        row.count = amount;
        return;
      }
      if(row.key == key) {
        const uint64_t sum = (uint64_t)row.count + (uint64_t)amount;
        row.count = sum > 0xFFFFFFFFull ? 0xFFFFFFFFu : (uint32_t)sum;
        return;
      }
    }
    unrecorded_.fetch_add((int64_t)amount, std::memory_order_relaxed);
  }

  // This key's count, or zero if it has no row. The zero is not a sentinel: a key with no row
  // was never hit, and "never hit" and "hit zero times" are one fact for a counter that only
  // ever counts up. A row is never written with a count below one, so the two cases cannot be
  // told apart by any observer either.
  uint32_t hitsFor(Hash128 key) const {
    const uint64_t home = key.hash0 & mask_;
    std::mutex& mutex = mutexPool_.getMutex((uint32_t)home & mutexMask_);
    std::lock_guard<std::mutex> lock(mutex);
    for(uint32_t step = 0; step < PROBE_WINDOW; step++) {
      const Row& row = rows_[(home + step) & mask_];
      if(row.count == 0)
        return 0;
      if(row.key == key)
        return row.count;
    }
    return 0;
  }

  std::vector<NNCacheHitCount> harvest() const {
    std::vector<NNCacheHitCount> out;
    for(size_t i = 0; i < rows_.size(); i++) {
      std::mutex& mutex = mutexPool_.getMutex((uint32_t)i & mutexMask_);
      std::lock_guard<std::mutex> lock(mutex);
      if(rows_[i].count == 0)
        continue;
      NNCacheHitCount row;
      row.key = rows_[i].key;
      row.hits = rows_[i].count;
      out.push_back(row);
    }
    return out;
  }

  // The hits that have not reached the count log yet, taking them as it reports them: per
  // occupied row, the count minus its persisted mark, with the mark then advanced.
  //
  // THE MARK IS WHY THE COUNT IS NOT SIMPLY RESET TO ZERO. A count of zero is this
  // structure's FREE-ROW marker, so zeroing an occupied row would cut the probe chain of
  // every key placed behind it and lose their counts silently. The mark is a second word in
  // the 4 bytes Row already padded away -- twoLevelHitLedgerBytes has always counted them --
  // so this costs the ledger nothing and forecloses that whole failure by not needing to
  // touch count at all.
  std::vector<NNCacheHitCount> takeUnpersisted() {
    std::vector<NNCacheHitCount> out;
    for(size_t i = 0; i < rows_.size(); i++) {
      std::mutex& mutex = mutexPool_.getMutex((uint32_t)i & mutexMask_);
      std::lock_guard<std::mutex> lock(mutex);
      Row& row = rows_[i];
      if(row.count == 0)
        continue;
      if(row.count <= row.persisted)
        continue;
      NNCacheHitCount out_row;
      out_row.key = row.key;
      out_row.hits = row.count - row.persisted;
      row.persisted = row.count;
      out.push_back(out_row);
    }
    return out;
  }

  int64_t unrecordedHits() const { return unrecorded_.load(std::memory_order_relaxed); }
  size_t structureBytes() const {
    return rows_.size() * sizeof(Row) + ((size_t)mutexMask_ + 1) * sizeof(std::mutex);
  }

 private:
  // 24 bytes: 16 of key, 4 of count, and 4 that alignment has always spent on this row
  // whether they held anything or not -- twoLevelHitLedgerBytes counts them explicitly. The
  // persisted mark is those 4 bytes, so the delta surface costs this structure nothing.
  struct Row {
    Hash128 key;
    uint32_t count;
    // How much of `count` has already been written to the count log. See takeUnpersisted.
    uint32_t persisted;
    Row() : key(), count(0), persisted(0) {}
  };

  std::vector<Row> rows_;
  uint64_t mask_;
  mutable MutexPool mutexPool_;
  uint32_t mutexMask_;
  std::atomic<int64_t> unrecorded_;
};

//-------------------------------------------------------------------------------------
// The table
//-------------------------------------------------------------------------------------

class NNCacheTableTwoLevel final : public NNCacheTable {
 public:
  NNCacheTableTwoLevel(
    std::unique_ptr<NNCacheFrozen> levelZero,
    std::unique_ptr<NNCacheTable> levelOne,
    int hitLedgerPowerOfTwo,
    int hitLedgerMutexPowerOfTwo
  )
    :levelZero_(std::move(levelZero)),
     levelOne_(std::move(levelOne)),
     ledger_(hitLedgerPowerOfTwo, hitLedgerMutexPowerOfTwo)
  {}

  bool get(Hash128 nnHash, std::shared_ptr<NNOutput>& ret) override {
    // Level 0 first. A hit here counts itself, in the same 32-bit word the lookup's own
    // entry read already brought into cache, so counting costs the level-0 path nothing
    // beyond the increment.
    if(levelZero_->get(nnHash, ret))
      return true;
    // Fall through. A level-1 hit is counted in the ledger; that is one extra random
    // access on the level-1 hit path, and it is the only place in this design where
    // counting costs a memory access it would not otherwise make.
    if(levelOne_->get(nnHash, ret)) {
      ledger_.add(nnHash, 1);
      return true;
    }
    return false;
  }

  void set(const std::shared_ptr<NNOutput>& p) override {
    // Level 0 is frozen, so every set is level 1's. If level 0 holds this key, retire its
    // entry FIRST and fold the count it accrued into the ledger, so the key and its count
    // change hands together and are never resolvable in two places at once.
    //
    // This is reached for real: a caller passing skipCache=true consults no level and sets
    // unconditionally, and NNEvaluator::evaluate's ownership-map fall-through deliberately
    // re-evaluates a hit that lacked a requested ownership map and then sets the fuller
    // result. Dropping those sets would leave level 0 answering with an entry the caller
    // has already rejected, forcing a full re-evaluation of that position on every
    // subsequent such query.
    const std::optional<uint32_t> transferred = levelZero_->shadow(p->nnHash);
    if(transferred.has_value())
      ledger_.add(p->nnHash, transferred.value());
    levelOne_->set(p);
  }

  // LEVEL 1 ONLY, AND WITHOUT COUNTING. Both halves are the point. Not counting is what
  // keeps a dump from appearing in the counts it is dumping (ADR-0009); skipping level 0 is
  // what keeps a dump from being handed an entry whose bytes are already in the file it is
  // appending to, which is the duplicate the whole persisted mark exists to foreclose.
  bool peek(Hash128 nnHash, std::shared_ptr<NNOutput>& ret) override {
    return levelOne_->get(nnHash, ret);
  }

  // Level 1 only. Level 0 is the pre-warmed content this session was handed; it cannot be
  // rebuilt in process, and clearing it would discard the whole point of having it. The
  // hit counts survive too: clear() is not the end of a session.
  void clear() override {
    levelOne_->clear();
  }

  NNCacheStats stats() const override {
    NNCacheStats s = levelOne_->stats();
    s.residentEntries += levelZero_->numReachableEntries();
    s.residentPayloadBytes += levelZero_->reachablePayloadBytes();
    // Level 0's own structure, the ledger, and -- because it is real memory this table
    // holds and can no longer hand out -- the evaluations of shadowed entries.
    s.fixedStructureBytes +=
      (int64_t)levelZero_->structureBytes() +
      (int64_t)ledger_.structureBytes() +
      levelZero_->shadowedPayloadBytes();
    // A chained level 1 reports no slot capacity because it is bounded by bytes; adding
    // level 0's fixed count to a zero would fabricate a ratio, so the zero is preserved.
    if(s.capacitySlots != 0)
      s.capacitySlots += (int64_t)levelZero_->numEntries();
    return s;
  }

  NNCacheHitLedger harvestHitCounts() const override {
    // Level 0's unshadowed entries first, in index order 0..n-1, which is the harvest
    // order SPEC.md 3.4 requires and -- because the input arrived in descending reference
    // count -- descending-popularity order. Then level 1's keys. No key appears in both
    // lists: shadowing removes a key from level 0 at the same moment its count arrives in
    // the ledger.
    std::vector<NNCacheHitCount> rows = levelZero_->harvest();
    const std::vector<NNCacheHitCount> levelOneRows = ledger_.harvest();
    rows.insert(rows.end(), levelOneRows.begin(), levelOneRows.end());
    return NNCacheHitLedger::counted(std::move(rows), ledger_.unrecordedHits());
  }

  // What a dump of one context writes. The two facts are owned in two places and joined
  // here, once: WHICH keys this context earned is the base table's attribution ledger, and
  // HOW OFTEN each was retrieved is this table's hit ledger. Neither is re-derived from the
  // other -- an earned key with no hit row appears with zero, which is a fact a dump wants,
  // not a row to drop.
  // The delta twin of harvestHitCounts above, composed from the same two owners and in the
  // same order. See NNCacheTable::takeUnpersistedHitCounts for what it is for.
  //
  // unrecordedHits is reported WHOLE rather than as a delta, and that is deliberate: it is a
  // count of hits whose KEY was lost, so there is no row to hold a mark against and no way to
  // divide it. A caller reading it twice sees the same running figure; it is a health number,
  // not a payload.
  NNCacheHitLedger takeUnpersistedHitCounts() override {
    std::vector<NNCacheHitCount> rows = levelZero_->takeUnpersistedHits();
    const std::vector<NNCacheHitCount> levelOneRows = ledger_.takeUnpersisted();
    rows.insert(rows.end(), levelOneRows.begin(), levelOneRows.end());
    return NNCacheHitLedger::counted(std::move(rows), ledger_.unrecordedHits());
  }

  NNCacheHitLedger harvestHitCountsFor(const NNCacheContextId& context) const override {
    const std::vector<Hash128> keys = attributedKeysFor(context);
    std::vector<NNCacheHitCount> rows;
    rows.reserve(keys.size());
    for(size_t i = 0; i < keys.size(); i++) {
      NNCacheHitCount row;
      row.key = keys[i];
      row.hits = ledger_.hitsFor(keys[i]);
      rows.push_back(row);
    }
    // Zero, and see NNCacheTable::harvestHitCountsFor for why the residue is not divided.
    return NNCacheHitLedger::counted(std::move(rows), 0);
  }

 private:
  std::unique_ptr<NNCacheFrozen> levelZero_;
  std::unique_ptr<NNCacheTable> levelOne_;
  mutable HitLedger ledger_;
};

}  // namespace

size_t twoLevelHitLedgerBytes(int hitLedgerPowerOfTwo) {
  // Kept in step with HitLedger::Row by construction rather than by a second copy of the
  // arithmetic (ADR-0012 P1). The mutex pool is not included: it is sized independently
  // and is reported through stats(), where the whole figure is what matters.
  // The second uint32_t is the persisted mark, which before it existed was the padding this
  // arithmetic already charged for. The figure is unchanged by its arrival, which is the
  // point (ADR-0012 P1: the row type decides, this reads it).
  return (((size_t)1) << hitLedgerPowerOfTwo) * (sizeof(Hash128) + sizeof(uint32_t) + sizeof(uint32_t));
}

std::unique_ptr<NNCacheTable> makeTwoLevelNNCacheTable(
  std::unique_ptr<NNCacheFrozen> levelZero,
  std::unique_ptr<NNCacheTable> levelOne,
  int hitLedgerPowerOfTwo
) {
  // "No level 0" is represented by not building one of these, never by one of these
  // holding a null. Refusing here keeps that the only representation there is.
  if(levelZero == nullptr)
    throw StringError(
      "makeTwoLevelNNCacheTable: no level 0 was supplied. A configuration with no frozen "
      "level 0 is served by NNCacheTable::create alone; it is not this table holding a null."
    );
  if(levelOne == nullptr)
    throw StringError("makeTwoLevelNNCacheTable: no level 1 was supplied.");
  if(hitLedgerPowerOfTwo < 0 || hitLedgerPowerOfTwo > 40)
    throw StringError(
      "makeTwoLevelNNCacheTable: hitLedgerPowerOfTwo must be between 0 and 40; got " +
      Global::intToString(hitLedgerPowerOfTwo) + "."
    );
  // The ledger's mutex pool: enough to keep concurrent level-1 hits off one lock, never
  // more mutexes than rows.
  const int mutexPow = hitLedgerPowerOfTwo < 10 ? hitLedgerPowerOfTwo : 10;
  return std::unique_ptr<NNCacheTable>(
    new NNCacheTableTwoLevel(std::move(levelZero), std::move(levelOne), hitLedgerPowerOfTwo, mutexPow)
  );
}
