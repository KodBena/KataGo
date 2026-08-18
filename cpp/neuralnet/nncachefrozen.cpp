#include "../neuralnet/nncachefrozen.h"

#include <algorithm>
#include <cstring>

#include "../core/global.h"

// The frozen level-0 cache. See nncachefrozen.h for the contract and cpp/spec/chd/SPEC.md
// for the specification this was written from; section references are to that document.

namespace {

//-------------------------------------------------------------------------------------
// Sizing and mixing
//-------------------------------------------------------------------------------------

// All INCIDENTAL under SPEC.md 11 -- no caller can observe any of them, and 1.5 says to
// choose our own. These are chosen to match the shape the specification's own calibration
// figures were measured on, so those figures apply here: with four keys per bucket and a
// position table around 1.1x the key count, SPEC.md 5.1 measured a mean of 38.9
// displacement attempts per bucket, flat to within 0.3% across a 200x range of key counts,
// and a hardest bucket anywhere of 1859 attempts across 60 builds.
const uint32_t KEYS_PER_BUCKET = 4;

// The displacement search bound. A bound must exist, reaching it must refuse, and the
// refusal must name it (SPEC.md 5.1); the specific number is INCIDENTAL. 65535 is what a
// uint16 displacement can carry, and against the 1859-attempt worst case the spec measured
// that is a factor of about 35 of headroom.
const uint32_t SEARCH_BOUND = 65535;

// posTable_ entries are entry indices, so the largest representable index is one below
// this marker (SPEC.md 4.4: a key count past what the arithmetic supports must refuse
// rather than silently misbehave).
const uint32_t EMPTY_SLOT = 0xFFFFFFFFu;

// Reduce a 64-bit hash to [0,range) without a division: Lemire's multiply-shift. Unbiased
// enough for hash placement and about twenty cycles cheaper than a modulo on the hot path.
// The modulo fallback is correct too -- it just assigns different slots, which no caller
// can observe (SPEC.md 0: different internal parameters and a different assignment of keys
// to internal slots are conforming).
inline uint32_t reduceToRange(uint64_t hash, uint32_t range) {
#ifdef __SIZEOF_INT128__
  return (uint32_t)(((__uint128_t)hash * (__uint128_t)range) >> 64);
#else
  return (uint32_t)(hash % (uint64_t)range);
#endif
}

// The murmur3 64-bit finalizer: two multiply/shift-xor rounds, full avalanche.
inline uint64_t avalanche(uint64_t x) {
  x ^= x >> 33;
  x *= 0xFF51AFD7ED558CCDULL;
  x ^= x >> 33;
  x *= 0xC4CEB9FE1A85EC53ULL;
  x ^= x >> 33;
  return x;
}

// Both halves of the key, mixed down to one 64-bit value that the bucket and the slot are
// both read off. Computed once per lookup.
//
// hash0 IS SCRAMBLED BEFORE hash1 IS FOLDED IN, and that ordering is the whole point --
// a LINEAR combine of the two halves (hash0 * K + hash1, or hash1 * K + hash0) lets a
// structured pair of bit flips in the two halves CANCEL. Multiplication mod 2^64 by an odd
// constant maps 2^63 to 2^63, so flipping the top bit of both halves changes such a
// combine by 2^63 twice, which is zero: the two distinct keys 0000...0/0000...0 and
// 8000...0/8000...0 get one identical hash under EVERY displacement, and so do ffff...f/
// ffff...f and 7fff...f/7fff...f. The extremal_keys conformance vector contains exactly
// those four keys and refused to construct against the linear version of this function.
// With hash0 multiplied and shift-xored first, no such cancellation exists; two distinct
// keys can share a base only by a genuine 128-to-64 collision, which is about n^2/2^64 and
// is a loud construction refusal rather than a wrong answer if it ever happens.
inline uint64_t mixKey(Hash128 key) {
  uint64_t a = key.hash0 ^ 0xD1B54A32D192ED03ULL;
  a *= 0xFF51AFD7ED558CCDULL;
  a ^= a >> 32;
  a += key.hash1;
  a *= 0xC4CEB9FE1A85EC53ULL;
  a ^= a >> 29;
  a *= 0xD6E8FEB86659FD93ULL;
  a ^= a >> 32;
  return a;
}

// Which slot a key's bucket sends it to under `displacement`, decorrelated from the bucket
// the same base picked.
//
// BOTH halves of the key reach this, through the base, and that is a deliberate departure
// from the prototype: SPEC.md 4.2 records that the prototype derives a key's slot from
// hash1 alone once hash0 has fixed the bucket, so two distinct keys sharing a hash1 within
// one bucket collide under every displacement and construction is refused. That
// precondition is explicitly INCIDENTAL and the spec says an implementation that accepts
// those key sets is strictly better and conforming. The two vector cases recording the
// prototype's refusal (dup_hash1_same_bucket, hash1_degenerate_corpus) are marked
// PROTOTYPE-SPECIFIC and construct successfully here.
inline uint64_t slotHash(uint64_t base, uint32_t displacement) {
  return avalanche(base ^ (((uint64_t)displacement + 1) * 0x9E3779B97F4A7C15ULL));
}

std::string keyToString(Hash128 key) {
  return key.toString();
}

}  // namespace

//-------------------------------------------------------------------------------------
// NNCacheFrozenIndex
//-------------------------------------------------------------------------------------

uint32_t NNCacheFrozenIndex::searchBound() {
  return SEARCH_BOUND;
}
uint32_t NNCacheFrozenIndex::maxEntries() {
  return EMPTY_SLOT - 1;
}

NNCacheFrozenIndex::NNCacheFrozenIndex()
  :keys_(), posTable_(), disp_(), numBuckets_(0), tableSize_(0)
{}

std::optional<uint32_t> NNCacheFrozenIndex::find(Hash128 key) const {
  // An index built from an empty key set answers absent to everything (SPEC.md 4.3), and
  // this test is also what keeps the range reductions below away from a zero range.
  if(tableSize_ == 0)
    return std::nullopt;
  const uint64_t base = mixKey(key);
  const uint32_t bucket = reduceToRange(base, numBuckets_);
  const uint32_t slot = reduceToRange(slotHash(base, disp_[bucket]), tableSize_);
  const uint32_t idx = posTable_[slot];
  // The bounds check SPEC.md 2.2 requires before a resolved position is used. A slot that
  // holds no entry carries EMPTY_SLOT, which is not a valid index and must never be
  // dereferenced; reduceToRange already confines `slot` itself to the table.
  if(idx == EMPTY_SLOT)
    return std::nullopt;
  // The stored-key comparison SPEC.md 2.2 requires, on every lookup, with no path around
  // it. Roughly nine in ten never-inserted keys reach this line with a real entry's index
  // in hand; without the comparison every one of them would be answered with another
  // position's evaluation.
  if(!(keys_[idx] == key))
    return std::nullopt;
  return idx;
}

size_t NNCacheFrozenIndex::structureBytes() const {
  return
    keys_.size() * sizeof(Hash128) +
    posTable_.size() * sizeof(uint32_t) +
    disp_.size() * sizeof(uint16_t);
}

NNCacheFrozenIndex NNCacheFrozenIndex::build(const std::vector<Hash128>& keys) {
  NNCacheFrozenIndex index;

  if(keys.size() > (size_t)maxEntries())
    throw StringError(
      "NNCacheFrozenIndex: key count " + Global::uint64ToString((uint64_t)keys.size()) +
      " exceeds what this implementation's 32-bit slot arithmetic supports (max " +
      Global::uint64ToString((uint64_t)maxEntries()) + ")."
    );

  const uint32_t n = (uint32_t)keys.size();
  index.keys_ = keys;
  if(n == 0) {
    index.numBuckets_ = 0;
    index.tableSize_ = 0;
    return index;
  }

  index.numBuckets_ = (n + KEYS_PER_BUCKET - 1) / KEYS_PER_BUCKET;
  // 1.1x the key count, plus one so that n = 1 still has a slot to spare.
  index.tableSize_ = n + n / 10 + 1;
  index.posTable_.assign(index.tableSize_, EMPTY_SLOT);
  index.disp_.assign(index.numBuckets_, 0);

  // Group the keys by bucket with a counting sort: bucketOf, then offsets, then the
  // membership array. All three are transient and freed before build() returns.
  std::vector<uint64_t> baseOf(n);
  std::vector<uint32_t> bucketOf(n);
  std::vector<uint32_t> bucketStart(index.numBuckets_ + 1, 0);
  for(uint32_t i = 0; i < n; i++) {
    baseOf[i] = mixKey(keys[i]);
    const uint32_t b = reduceToRange(baseOf[i], index.numBuckets_);
    bucketOf[i] = b;
    bucketStart[b + 1] += 1;
  }
  for(uint32_t b = 0; b < index.numBuckets_; b++)
    bucketStart[b + 1] += bucketStart[b];
  std::vector<uint32_t> members(n);
  {
    std::vector<uint32_t> fill(bucketStart.begin(), bucketStart.end() - 1);
    for(uint32_t i = 0; i < n; i++)
      members[fill[bucketOf[i]]++] = i;
  }

  // Duplicate keys are not constructible (SPEC.md 4.1) and must be refused naming the key
  // set position involved (SPEC.md 5). Two equal keys always share a bucket, so sorting
  // each bucket's members by key and comparing neighbours finds every duplicate in
  // O(n log(bucket size)) with no pathological case -- a bucket of any size stays a sort
  // rather than becoming a quadratic scan.
  for(uint32_t b = 0; b < index.numBuckets_; b++) {
    const uint32_t lo = bucketStart[b];
    const uint32_t hi = bucketStart[b + 1];
    if(hi - lo < 2)
      continue;
    std::sort(members.begin() + lo, members.begin() + hi, [&keys](uint32_t x, uint32_t y) {
      if(keys[x] != keys[y])
        return keys[x] < keys[y];
      return x < y;
    });
    for(uint32_t j = lo + 1; j < hi; j++) {
      if(keys[members[j]] == keys[members[j - 1]]) {
        const uint32_t first = std::min(members[j], members[j - 1]);
        const uint32_t second = std::max(members[j], members[j - 1]);
        throw StringError(
          "NNCacheFrozenIndex: duplicate key " + keyToString(keys[first]) +
          " at key set positions " + Global::intToString((int)first) + " and " +
          Global::intToString((int)second) +
          "; duplicate keys are not constructible because the caller's index alignment "
          "assumes one entry per input."
        );
      }
    }
  }

  // Largest buckets first: the classic CHD placement order, and INCIDENTAL (SPEC.md 1.5).
  // A big bucket placed while the table is nearly empty is cheap; the same bucket placed
  // last would be the one that exhausts the search.
  std::vector<uint32_t> order(index.numBuckets_);
  for(uint32_t b = 0; b < index.numBuckets_; b++)
    order[b] = b;
  std::sort(order.begin(), order.end(), [&bucketStart](uint32_t x, uint32_t y) {
    const uint32_t sx = bucketStart[x + 1] - bucketStart[x];
    const uint32_t sy = bucketStart[y + 1] - bucketStart[y];
    if(sx != sy)
      return sx > sy;
    return x < y;
  });

  std::vector<uint32_t> trial;
  trial.reserve(16);
  for(uint32_t oi = 0; oi < index.numBuckets_; oi++) {
    const uint32_t b = order[oi];
    const uint32_t lo = bucketStart[b];
    const uint32_t hi = bucketStart[b + 1];
    if(lo == hi)
      continue;

    bool placed = false;
    for(uint32_t d = 0; d <= SEARCH_BOUND; d++) {
      trial.clear();
      bool ok = true;
      for(uint32_t j = lo; j < hi && ok; j++) {
        const uint32_t slot = reduceToRange(slotHash(baseOf[members[j]], d), index.tableSize_);
        if(index.posTable_[slot] != EMPTY_SLOT) {
          ok = false;
          break;
        }
        // Two keys of this same bucket landing on one slot under this displacement.
        for(size_t t = 0; t < trial.size(); t++) {
          if(trial[t] == slot) {
            ok = false;
            break;
          }
        }
        if(ok)
          trial.push_back(slot);
      }
      if(!ok)
        continue;
      for(uint32_t j = lo; j < hi; j++)
        index.posTable_[trial[j - lo]] = members[j];
      index.disp_[b] = (uint16_t)d;
      placed = true;
      break;
    }
    // The search is bounded, hitting the bound refuses, and the refusal names the bound
    // (SPEC.md 5.1). It never loops forever and never falls back to a degraded structure.
    if(!placed)
      throw StringError(
        "NNCacheFrozenIndex: the displacement search for bucket " + Global::intToString((int)b) +
        " (holding " + Global::intToString((int)(hi - lo)) + " of " + Global::intToString((int)n) +
        " keys, first at key set position " + Global::intToString((int)members[lo]) +
        ") was exhausted at its bound of " + Global::uint64ToString((uint64_t)SEARCH_BOUND) +
        " attempts; no structure was produced."
      );
  }

  // Self-verification before returning (SPEC.md 5.2): every key in the input must resolve
  // to its own entry. This is the check that makes a construction bug loud instead of
  // silent, and silence is this subsystem's worst failure mode.
  for(uint32_t i = 0; i < n; i++) {
    const std::optional<uint32_t> got = index.find(keys[i]);
    if(!got.has_value() || got.value() != i)
      throw StringError(
        "NNCacheFrozenIndex: self-verification failed -- the key at key set position " +
        Global::intToString((int)i) + " (" + keyToString(keys[i]) + ") resolves to " +
        (got.has_value() ? Global::intToString((int)got.value()) : std::string("absent")) +
        "; no structure was produced."
      );
  }

  return index;
}

//-------------------------------------------------------------------------------------
// NNCacheFrozen
//-------------------------------------------------------------------------------------

NNCacheFrozen::NNCacheFrozen(NNCacheFrozenIndex&& index, std::vector<std::shared_ptr<NNOutput>>&& evaluations)
  :index_(std::move(index)), entries_(evaluations.size())
{
  for(size_t i = 0; i < evaluations.size(); i++)
    entries_[i].evaluation = std::move(evaluations[i]);
}

std::unique_ptr<NNCacheFrozen> NNCacheFrozen::build(std::vector<std::shared_ptr<NNOutput>> evaluations) {
  // The key set is DERIVED from the evaluations rather than supplied beside them, so entry
  // i's key and entry i's evaluation cannot disagree (SPEC.md 3.4) and there is no pair of
  // lengths to check (SPEC.md 1.1).
  std::vector<Hash128> keys;
  keys.reserve(evaluations.size());
  for(size_t i = 0; i < evaluations.size(); i++) {
    if(evaluations[i] == nullptr)
      throw StringError(
        "NNCacheFrozen: the evaluation at key set position " + Global::uint64ToString((uint64_t)i) +
        " is null and carries no position hash to index it by; no structure was produced."
      );
    keys.push_back(evaluations[i]->nnHash);
  }
  NNCacheFrozenIndex index = NNCacheFrozenIndex::build(keys);
  return std::unique_ptr<NNCacheFrozen>(new NNCacheFrozen(std::move(index), std::move(evaluations)));
}

bool NNCacheFrozen::get(Hash128 key, std::shared_ptr<NNOutput>& ret) {
  if(ret != nullptr)
    ret.reset();
  const std::optional<uint32_t> idx = index_.find(key);
  if(!idx.has_value())
    return false;
  Entry& entry = entries_[idx.value()];
  // Count the retrieval, and refuse to serve a shadowed entry in the same operation. A
  // relaxed increment is all SPEC.md 3.2/3.3 ask for: the counters need no ordering, only
  // not to be lost or torn.
  const uint32_t prev = entry.state.fetch_add(1, std::memory_order_relaxed);
  if((prev & SHADOW_BIT) != 0) {
    // Level 1 owns this key now. Undo the increment so the transferred count stays exact.
    entry.state.fetch_sub(1, std::memory_order_relaxed);
    return false;
  }
  ret = entry.evaluation;
  return true;
}

bool NNCacheFrozen::contains(Hash128 key) const {
  const std::optional<uint32_t> idx = index_.find(key);
  if(!idx.has_value())
    return false;
  return (entries_[idx.value()].state.load(std::memory_order_relaxed) & SHADOW_BIT) == 0;
}

std::optional<uint32_t> NNCacheFrozen::shadow(Hash128 key) {
  const std::optional<uint32_t> idx = index_.find(key);
  if(!idx.has_value())
    return std::nullopt;
  Entry& entry = entries_[idx.value()];
  // One atomic exchange both retires the entry and reads out everything it accrued, so the
  // count can be neither split across the transfer nor transferred twice: a racing second
  // caller sees SHADOW_BIT already set and hands out nothing.
  const uint32_t prev = entry.state.exchange(SHADOW_BIT, std::memory_order_relaxed);
  if((prev & SHADOW_BIT) != 0)
    return std::nullopt;
  return prev & COUNT_MASK;
}

bool NNCacheFrozen::addHits(Hash128 key, uint32_t amount) {
  const std::optional<uint32_t> idx = index_.find(key);
  if(!idx.has_value())
    return false;
  Entry& entry = entries_[idx.value()];
  uint32_t cur = entry.state.load(std::memory_order_relaxed);
  while(true) {
    if((cur & SHADOW_BIT) != 0)
      return false;
    const uint32_t next = (cur + amount) & COUNT_MASK;
    if(entry.state.compare_exchange_weak(cur, next, std::memory_order_relaxed))
      return true;
  }
}

uint32_t NNCacheFrozen::hitCountAt(uint32_t i) const {
  const uint32_t state = entries_[i].state.load(std::memory_order_relaxed);
  if((state & SHADOW_BIT) != 0)
    return 0;
  return state & COUNT_MASK;
}

bool NNCacheFrozen::isShadowedAt(uint32_t i) const {
  return (entries_[i].state.load(std::memory_order_relaxed) & SHADOW_BIT) != 0;
}

std::shared_ptr<NNOutput> NNCacheFrozen::evaluationAt(uint32_t i) const {
  if(isShadowedAt(i))
    return nullptr;
  return entries_[i].evaluation;
}

std::vector<NNCacheHitCount> NNCacheFrozen::harvest() const {
  std::vector<NNCacheHitCount> out;
  const uint32_t n = index_.numEntries();
  out.reserve(n);
  for(uint32_t i = 0; i < n; i++) {
    const uint32_t state = entries_[i].state.load(std::memory_order_relaxed);
    if((state & SHADOW_BIT) != 0)
      continue;
    NNCacheHitCount row;
    row.key = index_.keyAt(i);
    row.hits = state & COUNT_MASK;
    out.push_back(row);
  }
  return out;
}

size_t NNCacheFrozen::structureBytes() const {
  return index_.structureBytes() + entries_.size() * sizeof(Entry);
}

int64_t NNCacheFrozen::reachablePayloadBytes() const {
  int64_t bytes = 0;
  for(size_t i = 0; i < entries_.size(); i++) {
    if((entries_[i].state.load(std::memory_order_relaxed) & SHADOW_BIT) != 0)
      continue;
    if(entries_[i].evaluation != nullptr)
      bytes += (int64_t)nnOutputFootprintBytes(*entries_[i].evaluation);
  }
  return bytes;
}

int64_t NNCacheFrozen::shadowedPayloadBytes() const {
  int64_t bytes = 0;
  for(size_t i = 0; i < entries_.size(); i++) {
    if((entries_[i].state.load(std::memory_order_relaxed) & SHADOW_BIT) == 0)
      continue;
    if(entries_[i].evaluation != nullptr)
      bytes += (int64_t)nnOutputFootprintBytes(*entries_[i].evaluation);
  }
  return bytes;
}

int64_t NNCacheFrozen::numReachableEntries() const {
  int64_t count = 0;
  for(size_t i = 0; i < entries_.size(); i++) {
    if((entries_[i].state.load(std::memory_order_relaxed) & SHADOW_BIT) == 0)
      count += 1;
  }
  return count;
}
