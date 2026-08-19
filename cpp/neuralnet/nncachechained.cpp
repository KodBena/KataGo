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
// region ends by giving up victims until the region is back under budget,
// so the work is amortised O(1) per insert -- an insert adds one entry's bytes and
// therefore removes at most one entry's bytes worth of others -- and it is bounded by
// the region, not the table. Evicted payload pointers are moved to a local buffer and
// released after the lock is dropped, so no free ever happens under the lock.
//
// ---------------------------------------------------------------------------
// Which resident entry the sweep gives up
// ---------------------------------------------------------------------------
// This is nnCacheEviction under nnCacheCollision = chain, and it is a real choice with
// three answers, not a formality. It is a DIFFERENT TRIGGER from the probed table's
// eviction -- there it fires because a bucket's ways are all taken, here because a
// region is over its byte budget -- but it is the same question, "which resident entry
// is given up", so it is the same config axis.
//
// Each of the three is exact where exactness is cheap, and the one place it is not is
// named rather than hidden:
//
//   lru     EXACT. The region's node list is kept in recency order -- a get() moves its
//           node to the head -- so the victim is the tail, in O(1). This is the order
//           that was already in force before the axis was expressible, so `lru` is the
//           value that changes nothing.
//   random  EXACT and uniform. The region's roster is a flat vector of its nodes, so a
//           uniform victim is one modulo away, in O(1).
//   lfu     SAMPLED, not exact. LFU with Dynamic Aging as on the probed table -- a
//           region floor equal to the last victim's count, newcomers admitted at
//           floor+1 -- but the victim is the least-frequent of LFU_SAMPLE_SIZE nodes
//           drawn uniformly from the roster, not of the whole region. The probed table
//           can be exact because a bucket offers `ways` candidates and `ways` is small;
//           a chained region holds however many entries its budget buys, which at a 4GB
//           budget over 2^13 regions is on the order of 165, and an exact minimum would
//           be a linear scan per eviction. A bucket-list min structure would restore
//           exactness in O(1); it was rejected as a large structure to build before the
//           sweep has said whether lfu is worth having at all. When a region holds
//           LFU_SAMPLE_SIZE nodes or fewer the sample IS the region and the choice is
//           exact, which is the case every test below constructs.
//
// What the two structures cost, stated rather than buried: the roster adds 8 bytes per
// resident entry plus one vector header per region, and the roster index adds 8 bytes
// to every Node after alignment -- paid by all three policies, including lru, which
// does not use the roster. Keeping one Node layout rather than templating the table
// over its policy is the deliberate trade: against ~1592 bytes for a bare entry and
// ~3036 with an ownership map, 8 bytes is under 0.6%, and one table is one thing to get
// right instead of three.

namespace {

// Redis's sampled-eviction constant, and the same reasoning: a sample of a handful
// picks a near-minimum with high probability, and the quality gain from a larger sample
// falls off fast. It is a constant here rather than a config key because a knob nobody
// can calibrate is worse than a documented number (ADR-0002 rung 1 is about refusing,
// not about offering every dial).
static const int LFU_SAMPLE_SIZE = 8;

inline uint64_t chainedSplitMix64(uint64_t& state) {
  state += 0x9E3779B97F4A7C15ULL;
  uint64_t z = state;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

class NNCacheTableChained final : public NNCacheTable {
  struct Node {
    std::shared_ptr<NNOutput> ptr;
    Hash128 hash;      // inline, so a chain walk rejects without touching the payload
    size_t bytes;      // what this entry was charged, so a removal refunds exactly it
    Node* chainNext;
    Node* lruPrev;     // toward the most recently used end
    Node* lruNext;     // toward the least recently used end
    uint32_t rosterIdx;  // its own position in the region's roster, so removal is O(1)
    uint32_t count;      // LFUDA reference count; unused under lru and random
  };

  struct Region {
    Node* lruHead;
    Node* lruTail;
    int64_t bytesUsed;
    std::vector<Node*> roster;  // every node in the region, in no order; for O(1) sampling
    uint64_t rng;               // splitmix64 state, seeded from the region index
    uint32_t floor;             // LFUDA aging floor: the last victim's count
  };

  Node** buckets;
  Region* regions;
  MutexPool* mutexPool;
  uint64_t bucketMask;
  int regionShift;
  int64_t regionBudgetBytes;
  NNCacheEvictionPolicy eviction;

 public:
  NNCacheTableChained(
    int sizePowerOfTwo, int mutexPoolSizePowerOfTwo, int64_t maxBytes, NNCacheEvictionPolicy evictionArg
  );
  ~NNCacheTableChained() override;

  bool get(Hash128 nnHash, std::shared_ptr<NNOutput>& ret) override;
  void set(const std::shared_ptr<NNOutput>& p) override;
  void clear() override;
  bool contains(Hash128 nnHash) const override;
  NNCacheStats stats() const override;

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
  // Roster membership: push_back to join, swap-with-last to leave, so both are O(1) and
  // the vector never has a hole. The moved node's stored index is fixed up, which is the
  // whole reason the index lives in the node.
  void rosterPush(Region& rs, Node* n) {
    n->rosterIdx = (uint32_t)rs.roster.size();
    rs.roster.push_back(n);
  }
  void rosterRemove(Region& rs, Node* n) {
    const size_t idx = (size_t)n->rosterIdx;
    Node* moved = rs.roster.back();
    rs.roster[idx] = moved;
    moved->rosterIdx = (uint32_t)idx;
    rs.roster.pop_back();
  }

  // The one place the eviction axis is read. Returns the node this region gives up, or
  // NULL when it holds nothing.
  Node* chooseVictim(Region& rs) {
    if(rs.roster.empty())
      return NULL;
    switch(eviction) {
    case NNCacheEvictionPolicy::Lru:
      return rs.lruTail;
    case NNCacheEvictionPolicy::Random:
      return rs.roster[(size_t)(chainedSplitMix64(rs.rng) % (uint64_t)rs.roster.size())];
    case NNCacheEvictionPolicy::Lfu: {
      const size_t n = rs.roster.size();
      if(n <= (size_t)LFU_SAMPLE_SIZE) {
        // The sample is the whole region, so this arm is exact.
        Node* best = rs.roster[0];
        for(size_t i = 1; i<n; i++)
          if(rs.roster[i]->count < best->count)
            best = rs.roster[i];
        return best;
      }
      Node* best = NULL;
      for(int i = 0; i<LFU_SAMPLE_SIZE; i++) {
        Node* cand = rs.roster[(size_t)(chainedSplitMix64(rs.rng) % (uint64_t)n)];
        if(best == NULL || cand->count < best->count)
          best = cand;
      }
      return best;
    }
    case NNCacheEvictionPolicy::None:
    default:
      break;
    }
    // NNCacheShape::chained already refuses None, so reaching here means the shape type
    // was bypassed -- or a policy was added to the enum and not to this switch. Refuse
    // rather than silently pick an order nobody asked for.
    throw StringError(
      "NNCacheTableChained: eviction policy 'none' has no capacity-sweep implementation; a "
      "chained table's byte budget is always enforced, so a victim must be chosen."
    );
  }

  // Recording a sighting: what a get() on a hit, or a set() of a key already resident,
  // does to the policy's state. LRU reorders; LFUDA counts, saturating; Random has
  // nothing to remember, and skipping the list reorder there is not an optimization but
  // the honest thing -- a recency order Random never reads would be a lie in the data
  // structure.
  void onSighting(Region& rs, Node* n) {
    switch(eviction) {
    case NNCacheEvictionPolicy::Lru: lruTouch(rs,n); break;
    case NNCacheEvictionPolicy::Lfu: if(n->count != 0xFFFFFFFFu) n->count += 1; break;
    case NNCacheEvictionPolicy::Random: break;
    case NNCacheEvictionPolicy::None: break;
    // A policy added to the enum and not here would silently record no sighting at all,
    // which is the quiet half of the failure chooseVictim's throw catches loudly. Refuse
    // in both halves rather than in one (ADR-0002).
    default:
      throw StringError(
        "NNCacheTableChained::onSighting: no sighting behaviour for this eviction policy; "
        "a policy was added to NNCacheEvictionPolicy without teaching the chained table what "
        "a hit means under it."
      );
    }
  }
};

NNCacheTableChained::NNCacheTableChained(
  int sizePowerOfTwo, int mutexPoolSizePowerOfTwo, int64_t maxBytes, NNCacheEvictionPolicy evictionArg
)
  :buckets(NULL),regions(NULL),mutexPool(NULL),bucketMask(0),regionShift(0),regionBudgetBytes(0),
   eviction(evictionArg)
{
  if(evictionArg == NNCacheEvictionPolicy::None)
    throw StringError(
      "NNCacheTableChained: eviction policy 'none' has no capacity-sweep implementation; a "
      "chained table's byte budget is always enforced, so a victim must be chosen."
    );
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
  for(uint32_t r = 0; r < numRegions; r++) {
    // Seeded from the region index and nothing else: run-deterministic on purpose, for
    // the same reason the probed table's Random is -- a victim distribution that changed
    // run to run could not be asserted against at all.
    uint64_t seed = (uint64_t)r * 0x2545F4914F6CDD1DULL + 0x9E3779B97F4A7C15ULL;
    regions[r].rng = chainedSplitMix64(seed);
    regions[r].floor = 0;
  }
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
      onSighting(regions[regionIdx], n);
      ret = n->ptr;
      return true;
    }
  }
  return false;
}

// THE SAME CHAIN WALK, WITHOUT onSighting. That one call is the whole difference and it is the
// whole point: under lru it moves the node to the front of its region's recency list and under
// lfu it raises the node's reference count, both of which are the correct record of a RETRIEVAL
// and a false record of an ownership question -- and this shape is the one the design recommends
// for a real session, so it is the one where a get-shaped probe at attach would do the most
// damage. See NNCacheTable::contains.
bool NNCacheTableChained::contains(Hash128 nnHash) const {
  const uint64_t bucket = bucketOf(nnHash);
  const uint32_t regionIdx = regionOf(bucket);
  std::lock_guard<std::mutex> lock(mutexPool->getMutex(regionIdx));
  for(const Node* n = buckets[bucket]; n != NULL; n = n->chainNext) {
    if(n->hash == nnHash)
      return true;
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
      onSighting(rs,found);
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
      rosterPush(rs,n);
      // LFUDA admits a newcomer at the aging floor plus one, so an entry whose count
      // froze long ago is eventually overtaken instead of becoming immortal. Under lru
      // and random this field is never read.
      n->count = rs.floor == 0xFFFFFFFFu ? 0xFFFFFFFFu : rs.floor + 1;
      rs.bytesUsed += (int64_t)entryBytes;
    }

    while(rs.bytesUsed > regionBudgetBytes) {
      Node* victim = chooseVictim(rs);
      if(victim == NULL)
        break;
      // The floor only ever rises, which is what makes the aging monotone.
      if(victim->count > rs.floor)
        rs.floor = victim->count;
      lruUnlink(rs,victim);
      rosterRemove(rs,victim);
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
      // Back to exactly what a fresh table has, including the policy state: a surviving
      // aging floor would let a cleared table go on evicting newcomers as if they were
      // stale, and a surviving rng stream would make clear() a hidden input to the
      // victim sequence.
      rs.roster.clear();
      rs.floor = 0;
      uint64_t seed = (uint64_t)r * 0x2545F4914F6CDD1DULL + 0x9E3779B97F4A7C15ULL;
      rs.rng = chainedSplitMix64(seed);
    }
    freeAfterUnlock.clear();
  }
}

// A snapshot taken one region at a time; see NNCacheTableDirect::stats for why there is
// no global lock and what that costs under live traffic.
//
// capacitySlots is 0 and deliberately so: a chained table has no slot capacity to be a
// fraction of. Its occupancy question is bytesUsed against the budget, which is
// residentPayloadBytes plus the node bytes inside fixedStructureBytes -- so a reader who
// wants "how full is it" reads bytes here, never a count over a made-up denominator.
NNCacheStats NNCacheTableChained::stats() const {
  const uint32_t numRegions = mutexPool->getNumMutexes();
  NNCacheStats s = {0,0,0,0};
  int64_t nodeBytes = 0;
  for(uint32_t r = 0; r < numRegions; r++) {
    std::lock_guard<std::mutex> lock(mutexPool->getMutex(r));
    const Region& rs = regions[r];
    for(const Node* n = rs.lruHead; n != NULL; n = n->lruNext) {
      s.residentEntries += 1;
      s.residentPayloadBytes += (int64_t)nnOutputFootprintBytes(*n->ptr);
      nodeBytes += (int64_t)sizeof(Node);
    }
    nodeBytes += (int64_t)(rs.roster.capacity() * sizeof(Node*));
  }
  s.fixedStructureBytes =
    (int64_t)((bucketMask+1) * sizeof(Node*)) +
    (int64_t)numRegions * (int64_t)(sizeof(std::mutex) + sizeof(Region)) +
    nodeBytes;
  return s;
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
    new NNCacheTableChained(
      config.sizePowerOfTwo, config.mutexPoolSizePowerOfTwo,
      config.shape.maxBytes(), config.shape.eviction()
    )
  );
}
