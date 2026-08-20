#ifndef NEURALNET_NNCACHEPROBED_H_
#define NEURALNET_NNCACHEPROBED_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "../neuralnet/nncache.h"
#include "../search/mutexpool.h"

// The open-addressed ("probed") cache table, as a template over its three axes:
// the probe sequence, the eviction policy, and whether an inline hash tag is kept
// beside each slot.
//
// This is a header rather than a body in nncacheprobed.cpp for exactly one reason:
// tests/testnncachebench.cpp instantiates the UseTag=false variant so the inline-tag
// question can be measured on the real table in one process. The shipped factory
// instantiates UseTag=true only. Nothing else includes this file.
//
// ---------------------------------------------------------------------------
// Concurrency
// ---------------------------------------------------------------------------
// The slot array is partitioned into 2^mutexPoolSizePowerOfTwo CONTIGUOUS regions of
// 2^regionShift slots each, and a key's entire probe sequence is confined to the
// region its home slot falls in. Therefore:
//
//   * a key maps to exactly one mutex, whichever thread asks -- unchanged from the
//     direct-mapped table;
//   * every slot a key can ever occupy is under that one mutex, so two threads can
//     never touch one slot while holding two different locks;
//   * an operation takes exactly one lock, so there is no lock order to get wrong
//     and no deadlock to have.
//
// LRU and LFU turn a get() into a write: a recency stamp or a frequency bump. That
// write lands inside the region lock get() already holds and inside the cache line
// the probe just read, so it costs no extra synchronisation and no extra miss. There
// is no global recency list and hence no single lock every search thread queues on.
//
// The price of confinement is that a key's candidate slots number regionSlots, not
// the whole table, so `ways` may not exceed regionSlots. That is refused at
// construction, naming both keys involved, rather than silently clamped (ADR-0002).

namespace NNCacheProbed {

// A slot's inline tag. 0 means "this slot has never been written". Derived from bits
// of hash1, which the table's index (low bits of hash0) does not use, and forced
// nonzero so that "empty" stays distinguishable from a real key -- at the cost of one
// bit, i.e. a 2^-31 chance that a tag comparison lets a non-matching key through to
// the full 128-bit comparison that then rejects it.
inline uint32_t tagOf(Hash128 hash) {
  return (uint32_t)(hash.hash1 >> 32) | 1u;
}

inline uint64_t splitMix64(uint64_t& state) {
  state += 0x9E3779B97F4A7C15ULL;
  uint64_t z = state;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

//-------------------------------------------------------------------------------------
// Probe sequences
//-------------------------------------------------------------------------------------
// offset(j) is added to the home slot's position WITHIN its region, modulo the region
// size. For a power-of-two region both sequences below are permutations of the
// region's positions, so `ways` probes always name `ways` DISTINCT slots -- asserted
// in testnncachepolicy.cpp rather than assumed here.

struct LinearProbe {
  static const char* name() { return "linearprobe"; }
  static uint64_t offset(int j) { return (uint64_t)j; }
};

struct QuadraticProbe {
  // Triangular numbers. The standard power-of-two quadratic-probing sequence: unlike
  // j*j (which covers only half the residues) (j*j+j)/2 covers every one of them.
  static const char* name() { return "quadraticprobe"; }
  static uint64_t offset(int j) { return ((uint64_t)j * (uint64_t)(j + 1)) / 2; }
};

//-------------------------------------------------------------------------------------
// Eviction policies
//-------------------------------------------------------------------------------------
// A policy owns two things as TYPES: the per-slot metadata it needs (Meta, which a
// slot inherits, so a policy needing none costs zero bytes by empty-base
// optimization -- C++17 has no [[no_unique_address]]) and the per-region state it
// needs (RegionState). It never sees the table.
//
// preferVictim(cand, best, candRank, region) answers "is cand a better victim than
// the best candidate so far", where candRank is how many candidates were considered
// before cand. Ranking pairwise rather than over an array means the table never
// builds a temporary of `ways` entries (ways may be up to 1024).

class RandomEviction {
 public:
  static const char* name() { return "random"; }
  struct Meta {};
  struct RegionState { uint64_t rng; };

  static void initRegion(RegionState& r, uint64_t regionIdx) {
    // Run-deterministic and independent per region. Deterministic on purpose: a
    // victim distribution that changes run to run cannot be tested at all.
    uint64_t seed = regionIdx * 0x2545F4914F6CDD1DULL + 0x9E3779B97F4A7C15ULL;
    r.rng = splitMix64(seed);
  }
  template<class Slot> static void onHit(Slot&, RegionState&) {}
  template<class Slot> static void onInsert(Slot&, RegionState&) {}
  template<class Slot> static void onEvict(const Slot&, RegionState&) {}
  template<class Slot> static bool preferVictim(const Slot&, const Slot&, int candRank, RegionState& r) {
    // Reservoir sampling: taking candidate k with probability 1/(k+1) leaves every
    // candidate equally likely to be the final choice, in one pass, with no array.
    return splitMix64(r.rng) % (uint64_t)(candRank + 1) == 0;
  }
};

class LruEviction {
 public:
  static const char* name() { return "lru"; }
  struct Meta { uint64_t stamp; };
  struct RegionState { uint64_t clock; };

  static void initRegion(RegionState& r, uint64_t) { r.clock = 0; }
  template<class Slot> static void onHit(Slot& s, RegionState& r) { s.stamp = ++r.clock; }
  template<class Slot> static void onInsert(Slot& s, RegionState& r) { s.stamp = ++r.clock; }
  template<class Slot> static void onEvict(const Slot&, RegionState&) {}
  template<class Slot> static bool preferVictim(const Slot& cand, const Slot& best, int, RegionState&) {
    return cand.stamp < best.stamp;
  }
};

// LFU with Dynamic Aging (LFUDA), not naive LFU.
//
// Naive LFU's classic failure is that an entry which accumulated a high count and is
// then never referenced again is immortal: every newcomer starts at 1 and is evicted
// first, forever. The aging term fixes that without a sweep and without a tunable
// constant: the region keeps a floor equal to the count of the last victim it gave
// up, a new entry is admitted at floor+1, and the floor only rises. A stale entry is
// therefore overtaken once the floor climbs past its frozen count, and dies.
//
// Counts are uint32 with a saturating increment. The floor saturates after ~4e9
// evictions in ONE region; at the measured order of 1000 cache ops/second spread over
// 2^17 regions that is not reachable, and it is stated here rather than left implicit.
class LfuEviction {
 public:
  static const char* name() { return "lfu"; }
  struct Meta { uint32_t count; };
  struct RegionState { uint32_t floor; };

  static void initRegion(RegionState& r, uint64_t) { r.floor = 0; }
  template<class Slot> static void onHit(Slot& s, RegionState&) {
    if(s.count != 0xFFFFFFFFu)
      s.count += 1;
  }
  template<class Slot> static void onInsert(Slot& s, RegionState& r) {
    s.count = r.floor == 0xFFFFFFFFu ? 0xFFFFFFFFu : r.floor + 1;
  }
  template<class Slot> static void onEvict(const Slot& s, RegionState& r) { r.floor = s.count; }
  template<class Slot> static bool preferVictim(const Slot& cand, const Slot& best, int, RegionState&) {
    return cand.count < best.count;
  }
};

//-------------------------------------------------------------------------------------
// The table
//-------------------------------------------------------------------------------------

template<class Probe, class Evict, bool UseTag>
class NNCacheTableProbed final : public NNCacheTable {
  struct Slot : Evict::Meta {
    std::shared_ptr<NNOutput> ptr;
    uint32_t tag;
  };

  Slot* slots;
  typename Evict::RegionState* regionStates;
  MutexPool* mutexPool;
  uint64_t tableMask;
  uint64_t regionMask;
  int regionShift;
  int ways;

 public:
  NNCacheTableProbed(int sizePowerOfTwo, int mutexPoolSizePowerOfTwo, int waysArg)
    :slots(NULL), regionStates(NULL), mutexPool(NULL), tableMask(0), regionMask(0), regionShift(0), ways(waysArg)
  {
    if(sizePowerOfTwo < 0 || sizePowerOfTwo > 63)
      throw StringError("NNCacheTable: Invalid sizePowerOfTwo: " + Global::intToString(sizePowerOfTwo));
    if(mutexPoolSizePowerOfTwo < 0 || mutexPoolSizePowerOfTwo > 31)
      throw StringError("NNCacheTable: Invalid mutexPoolSizePowerOfTwo: " + Global::intToString(mutexPoolSizePowerOfTwo));
    if(mutexPoolSizePowerOfTwo > sizePowerOfTwo)
      mutexPoolSizePowerOfTwo = sizePowerOfTwo;

    regionShift = sizePowerOfTwo - mutexPoolSizePowerOfTwo;
    const uint64_t regionSlots = ((uint64_t)1) << regionShift;
    if((uint64_t)ways > regionSlots)
      throw StringError(
        "Key '" + std::string(NNCacheConfig::KEY_WAYS) + "' = " + Global::intToString(ways) +
        " exceeds the " + Global::uint64ToString(regionSlots) + " slots one lock region holds. "
        "A probed table confines a key's whole probe sequence to one lock region, so that every "
        "slot the key can occupy is under the single mutex the key already maps to; that caps "
        "ways at 2^nnCacheSizePowerOfTwo / 2^nnMutexPoolSizePowerOfTwo = 2^" +
        Global::intToString(regionShift) + " = " + Global::uint64ToString(regionSlots) +
        " here. Either lower '" + std::string(NNCacheConfig::KEY_WAYS) +
        "', or lower nnMutexPoolSizePowerOfTwo (fewer, larger regions -- more contention), "
        "or raise nnCacheSizePowerOfTwo (a bigger table -- more memory)."
      );

    const uint64_t tableSize = ((uint64_t)1) << sizePowerOfTwo;
    tableMask = tableSize - 1;
    regionMask = regionSlots - 1;
    const uint32_t numRegions = ((uint32_t)1) << mutexPoolSizePowerOfTwo;

    // () is load-bearing: it value-initializes, which zeroes tag and the policy's
    // Meta. Without it those are indeterminate and an empty slot could read as full.
    slots = new Slot[tableSize]();
    regionStates = new typename Evict::RegionState[numRegions]();
    for(uint32_t r = 0; r < numRegions; r++)
      Evict::initRegion(regionStates[r], r);
    mutexPool = new MutexPool(numRegions);
  }

  // The real, compiler-decided size of one slot -- the memory the associativity costs
  // on top of the payload. Reported by the benchmark rather than asserted, because it
  // is a layout fact and stating it from a comment would be a second copy of it.
  static size_t slotBytes() { return sizeof(Slot); }

  ~NNCacheTableProbed() override {
    delete[] slots;
    delete[] regionStates;
    delete mutexPool;
  }

  // THE SAME PROBE WALK, WITHOUT Evict::onHit. That one call is the whole difference and it is
  // the whole point: under lru or lfu it moves the slot in its region's eviction order, which is
  // the correct record of a RETRIEVAL and a false record of an ownership question. See
  // NNCacheTable::contains.
  bool contains(Hash128 nnHash) const override {
    const uint64_t home = nnHash.hash0 & tableMask;
    const uint64_t region = home >> regionShift;
    const uint64_t base = region << regionShift;
    const uint64_t within = home & regionMask;
    const uint32_t tag = tagOf(nnHash);

    std::lock_guard<std::mutex> lock(mutexPool->getMutex((uint32_t)region));
    for(int j = 0; j < ways; j++) {
      const Slot& s = slots[base + ((within + Probe::offset(j)) & regionMask)];
      if constexpr(UseTag) {
        if(s.tag != tag)
          continue;
      }
      if(s.ptr != nullptr && s.ptr->nnHash == nnHash)
        return true;
    }
    return false;
  }

  bool get(Hash128 nnHash, std::shared_ptr<NNOutput>& ret) override {
    // Free ret BEFORE locking, to avoid any expensive operations while locked.
    if(ret != nullptr)
      ret.reset();

    const uint64_t home = nnHash.hash0 & tableMask;
    const uint64_t region = home >> regionShift;
    const uint64_t base = region << regionShift;
    const uint64_t within = home & regionMask;
    const uint32_t tag = tagOf(nnHash);

    std::lock_guard<std::mutex> lock(mutexPool->getMutex((uint32_t)region));
    for(int j = 0; j < ways; j++) {
      Slot& s = slots[base + ((within + Probe::offset(j)) & regionMask)];
      // Deliberately no early exit on an empty slot. Because slots only ever go from
      // empty to full, an early exit would in fact be correct today -- but it would
      // make correctness depend on that invariant surviving every future edit, and
      // the whole probe set is 2-3 cache lines, in a path measured at 0.0077% of
      // cycles. Robustness is the better buy.
      if constexpr(UseTag) {
        if(s.tag != tag)
          continue;
      }
      if(s.ptr != nullptr && s.ptr->nnHash == nnHash) {
        Evict::onHit(s, regionStates[region]);
        ret = s.ptr;
        return true;
      }
    }
    return false;
  }

  void set(const std::shared_ptr<NNOutput>& p) override {
    // Immediately copy p right now, before locking, to avoid any expensive operations
    // while locked. Whatever this ends up swapping out is freed after the unlock.
    std::shared_ptr<NNOutput> buf(p);

    const Hash128 nnHash = p->nnHash;
    const uint64_t home = nnHash.hash0 & tableMask;
    const uint64_t region = home >> regionShift;
    const uint64_t base = region << regionShift;
    const uint64_t within = home & regionMask;
    const uint32_t tag = tagOf(nnHash);

    {
      std::lock_guard<std::mutex> lock(mutexPool->getMutex((uint32_t)region));
      typename Evict::RegionState& rs = regionStates[region];

      bool placed = false;
      int64_t emptyIdx = -1;
      int64_t victimIdx = -1;
      int candRank = 0;

      for(int j = 0; j < ways; j++) {
        const uint64_t idx = base + ((within + Probe::offset(j)) & regionMask);
        Slot& s = slots[idx];
        if(s.ptr == nullptr) {
          if(emptyIdx < 0)
            emptyIdx = (int64_t)idx;
          continue;
        }
        bool same = true;
        if constexpr(UseTag)
          same = (s.tag == tag);
        same = same && s.ptr->nnHash == nnHash;
        if(same) {
          // The same key offered again -- the ownermap upgrade path in nneval.cpp is
          // exactly this. It is a sighting of a live entry, not a fresh insert, so the
          // policy sees onHit: an LFU count earned by this key must not be reset.
          s.ptr.swap(buf);
          Evict::onHit(s, rs);
          placed = true;
          break;
        }
        if(victimIdx < 0)
          victimIdx = (int64_t)idx;
        else if(Evict::preferVictim(s, slots[victimIdx], candRank, rs))
          victimIdx = (int64_t)idx;
        candRank += 1;
      }

      if(!placed) {
        if(emptyIdx >= 0) {
          Slot& s = slots[emptyIdx];
          s.tag = tag;
          s.ptr.swap(buf);
          Evict::onInsert(s, rs);
        }
        else {
          // Every way was occupied by some other key, so victimIdx was set.
          Slot& s = slots[victimIdx];
          Evict::onEvict(s, rs);
          s.tag = tag;
          s.ptr.swap(buf);
          Evict::onInsert(s, rs);
        }
      }
    }

    // No longer locked; buf falls out of scope and frees whatever it swapped out.
  }

  // A snapshot taken one region at a time; see NNCacheTableDirect::stats for why there
  // is no global lock and what that costs under live traffic.
  //
  // fixedStructureBytes is where the associativity's real memory price shows up: the
  // eviction policy's per-slot metadata is inside sizeof(Slot), so an LRU stamp or an
  // LFU count is counted here at its true, alignment-inflated cost rather than at the
  // width of its field.
  NNCacheStats stats() const override {
    const uint32_t numRegions = mutexPool->getNumMutexes();
    NNCacheStats s = {0,0,0,(int64_t)(tableMask+1)};
    for(uint32_t r = 0; r < numRegions; r++) {
      const uint64_t base = ((uint64_t)r) << regionShift;
      std::lock_guard<std::mutex> lock(mutexPool->getMutex(r));
      for(uint64_t k = 0; k <= regionMask; k++) {
        const Slot& slot = slots[base + k];
        if(slot.ptr != nullptr) {
          s.residentEntries += 1;
          s.residentPayloadBytes += (int64_t)nnOutputFootprintBytes(*slot.ptr);
        }
      }
    }
    s.fixedStructureBytes =
      (int64_t)((tableMask+1) * sizeof(Slot)) +
      (int64_t)numRegions * (int64_t)(sizeof(std::mutex) + sizeof(typename Evict::RegionState));
    return s;
  }

  void clear() override {
    const uint32_t numRegions = mutexPool->getNumMutexes();
    std::vector<std::shared_ptr<NNOutput>> freeAfterUnlock;
    for(uint32_t r = 0; r < numRegions; r++) {
      const uint64_t base = ((uint64_t)r) << regionShift;
      freeAfterUnlock.clear();
      {
        std::lock_guard<std::mutex> lock(mutexPool->getMutex(r));
        for(uint64_t k = 0; k <= regionMask; k++) {
          Slot& s = slots[base + k];
          if(s.ptr != nullptr) {
            freeAfterUnlock.emplace_back();
            s.ptr.swap(freeAfterUnlock.back());
          }
          s.tag = 0;
          // Reset the policy's per-slot metadata to exactly what a fresh table has.
          static_cast<typename Evict::Meta&>(s) = typename Evict::Meta();
        }
        Evict::initRegion(regionStates[r], r);
      }
      freeAfterUnlock.clear();
    }
  }
};

}  // namespace NNCacheProbed

#endif  // NEURALNET_NNCACHEPROBED_H_
