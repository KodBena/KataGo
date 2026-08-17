#include "../neuralnet/nncacheimpl.h"

#include <vector>

#include "../search/mutexpool.h"

using namespace std;

// Separate chaining, held to a byte budget.
//
// ---------------------------------------------------------------------------
// The byte budget, and why it is not a global counter
// ---------------------------------------------------------------------------
// A single global byte counter would have to be read and written on every insert and
// every eviction, under something -- a lock, or an atomic that every core fights over
// -- which reintroduces exactly the serialisation the mutex pool exists to avoid.
//
// Instead the budget is PARTITIONED. The bucket array is split into
// 2^mutexPoolSizePowerOfTwo lock regions exactly as the probed table splits its slot
// array, each region gets maxBytes / numRegions of the budget, and each region tracks
// its own usage under its own lock. Nothing is shared, so nothing contends.
//
// The global bound this buys is HARD, not approximate: the per-region budgets are a
// floor-divided partition, so they sum to at most maxBytes and the table can never
// exceed it. What the partition costs is utilisation, not safety -- a region whose
// share of the keys is below average leaves part of its budget unused. With uniform
// bucket mapping the per-region occupancy spread is the usual Poisson one, so the
// shortfall is on the order of 1/sqrt(entries per region) and always in the safe
// direction.
//
// ---------------------------------------------------------------------------
// What a byte is
// ---------------------------------------------------------------------------
// An entry is charged sizeof(Node) + nnOutputFootprintBytes(*p): the node's own
// resident cost plus the payload's REAL footprint, which is not a constant --
// whiteOwnerMap and noisedPolicyProbs are separate heap blocks present on only some
// entries, and sizeof(NNOutput) itself is a build-time quantity (1528 bytes normally,
// 10088 under USE_BIGGER_BOARDS_EXPENSIVE). A budget denominated as count times a
// constant would be a bound in the wrong currency and wrong in two of the three modes.
// The footprint is measured before the lock is taken.
//
// NOT charged, because it is a fixed cost that exists whether the table holds anything
// or not: the bucket array, 8 bytes times 2^nnCacheSizePowerOfTwo (16 MiB at k=21).
// That is the same accounting boundary nnCacheSizePowerOfTwo already has under direct
// mapping, where the slot array is likewise outside any budget.
//
// ---------------------------------------------------------------------------
// The capacity sweep
// ---------------------------------------------------------------------------
// It is not a periodic sweep and there is no background thread. Every insert into a
// region ends by popping that region's LRU tail until the region is back under budget,
// so the work is amortised O(1) per insert -- an insert adds one entry's bytes and
// therefore removes at most one entry's bytes worth of others -- and it is bounded by
// the region, not the table. Evicted payload pointers are moved to a local buffer and
// released after the lock is dropped, so no free ever happens under the lock.
//
// The order the sweep evicts in is recency: a get() moves its node to the region's LRU
// head. That order is not currently expressible in the config -- nnCacheEviction under
// chain is required to be `none`, which means "a collision never evicts", not "nothing
// is ever evicted". Making the capacity order configurable would need a new
// NNCacheShape factory; it is deliberately not invented here.

namespace {

class NNCacheTableChained final : public NNCacheTable {
  struct Node {
    std::shared_ptr<NNOutput> ptr;
    Hash128 hash;      // inline, so a chain walk rejects without touching the payload
    size_t bytes;      // what this entry was charged, so a removal refunds exactly it
    Node* chainNext;
    Node* lruPrev;     // toward the most recently used end
    Node* lruNext;     // toward the least recently used end
  };

  struct Region {
    Node* lruHead;
    Node* lruTail;
    int64_t bytesUsed;
  };

  Node** buckets;
  Region* regions;
  MutexPool* mutexPool;
  uint64_t bucketMask;
  int regionShift;
  int64_t regionBudgetBytes;

 public:
  NNCacheTableChained(int sizePowerOfTwo, int mutexPoolSizePowerOfTwo, int64_t maxBytes);
  ~NNCacheTableChained() override;

  bool get(Hash128 nnHash, std::shared_ptr<NNOutput>& ret) override;
  void set(const std::shared_ptr<NNOutput>& p) override;
  void clear() override;

  // The one definition of what an entry costs the budget. chainedEntryBytes() below
  // re-exports it so a test asserts against this and never against its own arithmetic
  // (ADR-0012 P1).
  static size_t entryBytesFor(const NNOutput& out) { return sizeof(Node) + nnOutputFootprintBytes(out); }

 private:
  uint64_t bucketOf(Hash128 hash) const { return hash.hash0 & bucketMask; }
  uint32_t regionOf(uint64_t bucket) const { return (uint32_t)(bucket >> regionShift); }

  void lruPushFront(Region& rs, Node* n) {
    n->lruPrev = NULL;
    n->lruNext = rs.lruHead;
    if(rs.lruHead != NULL)
      rs.lruHead->lruPrev = n;
    rs.lruHead = n;
    if(rs.lruTail == NULL)
      rs.lruTail = n;
  }
  void lruUnlink(Region& rs, Node* n) {
    if(n->lruPrev != NULL) n->lruPrev->lruNext = n->lruNext; else rs.lruHead = n->lruNext;
    if(n->lruNext != NULL) n->lruNext->lruPrev = n->lruPrev; else rs.lruTail = n->lruPrev;
    n->lruPrev = NULL;
    n->lruNext = NULL;
  }
  void lruTouch(Region& rs, Node* n) {
    if(rs.lruHead == n)
      return;
    lruUnlink(rs,n);
    lruPushFront(rs,n);
  }
  void chainUnlink(uint64_t bucket, Node* n) {
    Node** link = &buckets[bucket];
    while(*link != NULL) {
      if(*link == n) { *link = n->chainNext; return; }
      link = &((*link)->chainNext);
    }
  }
};

NNCacheTableChained::NNCacheTableChained(int sizePowerOfTwo, int mutexPoolSizePowerOfTwo, int64_t maxBytes)
  :buckets(NULL),regions(NULL),mutexPool(NULL),bucketMask(0),regionShift(0),regionBudgetBytes(0)
{
  if(sizePowerOfTwo < 0 || sizePowerOfTwo > 63)
    throw StringError("NNCacheTable: Invalid sizePowerOfTwo: " + Global::intToString(sizePowerOfTwo));
  if(mutexPoolSizePowerOfTwo < 0 || mutexPoolSizePowerOfTwo > 31)
    throw StringError("NNCacheTable: Invalid mutexPoolSizePowerOfTwo: " + Global::intToString(mutexPoolSizePowerOfTwo));
  if(mutexPoolSizePowerOfTwo > sizePowerOfTwo)
    mutexPoolSizePowerOfTwo = sizePowerOfTwo;

  const uint64_t numBuckets = ((uint64_t)1) << sizePowerOfTwo;
  const uint32_t numRegions = ((uint32_t)1) << mutexPoolSizePowerOfTwo;
  regionShift = sizePowerOfTwo - mutexPoolSizePowerOfTwo;
  bucketMask = numBuckets - 1;
  regionBudgetBytes = maxBytes / (int64_t)numRegions;

  // The smallest entry this table can possibly hold. A budget that cannot fit one of
  // these in a region is a table that can never retain anything, so it is refused
  // rather than left to behave as a very slow no-op (ADR-0002 rung 1).
  const int64_t minEntryBytes = (int64_t)(sizeof(Node) + sizeof(NNOutput));
  if(regionBudgetBytes < minEntryBytes)
    throw StringError(
      "Key '" + string(NNCacheConfig::KEY_MAX_BYTES) + "' = " + Global::int64ToString(maxBytes) +
      " leaves only " + Global::int64ToString(regionBudgetBytes) + " bytes per lock region, but the "
      "smallest entry this table can hold is " + Global::int64ToString(minEntryBytes) + " bytes. "
      "The budget is partitioned across the 2^nnMutexPoolSizePowerOfTwo = " +
      Global::uint64ToString((uint64_t)numRegions) + " regions so that no global counter has to be "
      "contended on. Either raise '" + string(NNCacheConfig::KEY_MAX_BYTES) + "' to at least " +
      Global::int64ToString(minEntryBytes * (int64_t)numRegions) +
      ", or lower nnMutexPoolSizePowerOfTwo."
    );

  buckets = new Node*[numBuckets]();
  regions = new Region[numRegions]();
  mutexPool = new MutexPool(numRegions);
}

NNCacheTableChained::~NNCacheTableChained() {
  if(regions != NULL && mutexPool != NULL) {
    const uint32_t numRegions = mutexPool->getNumMutexes();
    for(uint32_t r = 0; r < numRegions; r++) {
      Node* n = regions[r].lruHead;
      while(n != NULL) { Node* next = n->lruNext; delete n; n = next; }
    }
  }
  delete[] buckets;
  delete[] regions;
  delete mutexPool;
}

bool NNCacheTableChained::get(Hash128 nnHash, shared_ptr<NNOutput>& ret) {
  // Free ret BEFORE locking, to avoid any expensive operations while locked.
  if(ret != nullptr)
    ret.reset();

  const uint64_t bucket = bucketOf(nnHash);
  const uint32_t regionIdx = regionOf(bucket);

  std::lock_guard<std::mutex> lock(mutexPool->getMutex(regionIdx));
  for(Node* n = buckets[bucket]; n != NULL; n = n->chainNext) {
    // The full 128-bit key is inline in the node, so rejecting a chain member costs
    // no dereference of the payload at all -- the inline-tag question, answered by
    // construction on this shape rather than by a 32-bit approximation.
    if(n->hash == nnHash) {
      lruTouch(regions[regionIdx], n);
      ret = n->ptr;
      return true;
    }
  }
  return false;
}

void NNCacheTableChained::set(const shared_ptr<NNOutput>& p) {
  const Hash128 nnHash = p->nnHash;
  const size_t entryBytes = entryBytesFor(*p);   // measured before the lock
  const uint64_t bucket = bucketOf(nnHash);
  const uint32_t regionIdx = regionOf(bucket);

  // Everything expensive happens outside the lock: the payload copy (whose eventual
  // release frees 1.5-3 KB plus its heap blocks), the node allocation, and the release
  // of whatever the insert displaces.
  std::shared_ptr<NNOutput> buf(p);
  Node* spare = new Node();
  std::vector<std::shared_ptr<NNOutput>> freeAfterUnlock;
  freeAfterUnlock.reserve(4);

  {
    std::lock_guard<std::mutex> lock(mutexPool->getMutex(regionIdx));
    Region& rs = regions[regionIdx];

    Node* found = NULL;
    for(Node* n = buckets[bucket]; n != NULL; n = n->chainNext) {
      if(n->hash == nnHash) { found = n; break; }
    }

    if(found != NULL) {
      // The same key offered again, possibly with a different footprint -- nneval's
      // ownermap upgrade path replaces an entry with a strictly larger one. Charge the
      // delta rather than re-charging the whole entry.
      found->ptr.swap(buf);
      rs.bytesUsed += (int64_t)entryBytes - (int64_t)found->bytes;
      found->bytes = entryBytes;
      lruTouch(rs,found);
    }
    else {
      Node* n = spare;
      spare = NULL;
      n->ptr.swap(buf);
      n->hash = nnHash;
      n->bytes = entryBytes;
      n->chainNext = buckets[bucket];
      buckets[bucket] = n;
      lruPushFront(rs,n);
      rs.bytesUsed += (int64_t)entryBytes;
    }

    while(rs.bytesUsed > regionBudgetBytes && rs.lruTail != NULL) {
      Node* victim = rs.lruTail;
      lruUnlink(rs,victim);
      chainUnlink(bucketOf(victim->hash), victim);
      rs.bytesUsed -= (int64_t)victim->bytes;
      freeAfterUnlock.emplace_back();
      victim->ptr.swap(freeAfterUnlock.back());
      delete victim;
    }
  }

  // No longer locked; the displaced payloads are freed as buf and freeAfterUnlock go
  // out of scope, and the spare node is freed if the key turned out to be resident.
  delete spare;
}

void NNCacheTableChained::clear() {
  const uint32_t numRegions = mutexPool->getNumMutexes();
  std::vector<std::shared_ptr<NNOutput>> freeAfterUnlock;
  for(uint32_t r = 0; r < numRegions; r++) {
    freeAfterUnlock.clear();
    {
      std::lock_guard<std::mutex> lock(mutexPool->getMutex(r));
      Region& rs = regions[r];
      Node* n = rs.lruHead;
      while(n != NULL) {
        Node* next = n->lruNext;
        buckets[bucketOf(n->hash)] = NULL;
        freeAfterUnlock.emplace_back();
        n->ptr.swap(freeAfterUnlock.back());
        delete n;
        n = next;
      }
      rs.lruHead = NULL;
      rs.lruTail = NULL;
      rs.bytesUsed = 0;
    }
    freeAfterUnlock.clear();
  }
}

}  // namespace

int64_t chainedRegionBudgetBytes(int64_t maxBytes, int sizePowerOfTwo, int mutexPoolSizePowerOfTwo) {
  if(mutexPoolSizePowerOfTwo > sizePowerOfTwo)
    mutexPoolSizePowerOfTwo = sizePowerOfTwo;
  return maxBytes / (int64_t)(((uint64_t)1) << mutexPoolSizePowerOfTwo);
}

size_t chainedEntryBytes(const NNOutput& out) {
  return NNCacheTableChained::entryBytesFor(out);
}

unique_ptr<NNCacheTable> makeChainedNNCacheTable(const NNCacheConfig& config) {
  if(config.shape.scheme() != NNCacheCollisionScheme::Chain)
    throw StringError("makeChainedNNCacheTable: '" + config.shape.toString() + "' is not a chained shape.");
  return unique_ptr<NNCacheTable>(
    new NNCacheTableChained(config.sizePowerOfTwo, config.mutexPoolSizePowerOfTwo, config.shape.maxBytes())
  );
}
