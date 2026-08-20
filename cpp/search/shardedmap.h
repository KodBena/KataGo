#ifndef SEARCH_SHARDEDMAP_H_
#define SEARCH_SHARDEDMAP_H_

#include "../core/global.h"
#include "../core/hash.h"
#include "../core/multithread.h"
#include "../search/mutexpool.h"

//A Hash128-keyed map split into numShards independently-mutexed shards, which ALSO remembers which
//shards have been handed out for mutation, so that a full walk of the table's contents costs
//O(number of shards actually used) rather than O(numShards).
//
//Why the occupancy list exists: these tables are sized by geometry (tens of thousands of shards, so
//that lock traffic stays spread out at high thread counts) while a single search populates only a few
//hundred shards. A teardown that loops over every shard therefore costs the same whether the table
//holds one entry or a million - the cost tracks the table's SHAPE instead of its CONTENTS.
//
//The hazard this type is shaped against: a shard that holds entries but is NOT listed as occupied
//would be silently skipped by a walk - entries leak, and nothing fails. So the tracking is not a
//discipline that callers must remember. There is no way to reach a shard's map at all except:
//  (1) lockShard(idx), which registers idx as occupied BEFORE it hands back the locked shard, and
//  (2) getOccupiedShard(pos), which is indexed by position in the occupancy list and so can only
//      ever name a shard that is already registered.
//The maps themselves are private, so "insert into a shard without registering it" is not a mistake a
//caller can make - it is not expressible.
//
//Marking at LOCK time rather than at insert time makes the tracking an over-approximation: a shard
//locked for a lookup that inserted nothing is still listed. Over-approximation is the safe direction
//(a walk visits it and finds it empty); the dangerous direction cannot occur.
//
//Thread-safety: lockShard may be called concurrently by any number of threads, and takes exactly one
//lock - the shard's own, from the pool, exactly as the callers did before this type existed. The
//occupancy bookkeeping is done with atomics into a preallocated slot table, so no second lock enters
//any path a search takes. The occupancy queries are for use when no other thread is mutating the
//table, and each says so; getOccupiedShard is safe to call concurrently for distinct positions, which
//is what the multithreaded teardowns do.
//This type does NOT own the mapped values - what a Value's lifetime is stays the caller's business.
template<typename Value>
class ShardedMap {
 public:
  typedef std::map<Hash128,Value> Shard;

  //Handle to one locked shard. Constructing one is what registers the shard as occupied, so holding
  //this handle is itself the proof that the shard will be visited by any later occupancy walk.
  class LockedShard {
   public:
    Shard& map() const { return *shard; }

    LockedShard(const LockedShard&) = delete;
    LockedShard& operator=(const LockedShard&) = delete;

   private:
    friend class ShardedMap<Value>;
    LockedShard(std::mutex& mutexToLock, Shard& shardToUse)
      :lock(mutexToLock), shard(&shardToUse)
    {}

    std::lock_guard<std::mutex> lock;
    Shard* shard;
  };

  //Refuses a nonsensical shard count at construction rather than anywhere later (ADR-0002: the
  //strongest rung that fits - the table cannot come into existence in an unusable shape).
  explicit ShardedMap(int64_t numShardsToUse)
    :shards(checkedNumShards(numShardsToUse)),
     occupiedFlags(checkedNumShards(numShardsToUse)),
     mutexPool(new MutexPool((uint32_t)checkedNumShards(numShardsToUse))),
     //Sized once, up front, so that registering a shard is a store into a slot this thread owns and
     //never a reallocation - no lock is needed on the insertion path, and none is taken there.
     occupiedShards(checkedNumShards(numShardsToUse),0),
     numOccupied(0)
  {
    for(size_t i = 0; i<occupiedFlags.size(); i++)
      occupiedFlags[i].store(false,std::memory_order_relaxed);
  }

  ShardedMap(const ShardedMap&) = delete;
  ShardedMap& operator=(const ShardedMap&) = delete;

  uint32_t getNumShards() const {
    return (uint32_t)shards.size();
  }

  //Locks the shard and marks it occupied. This is the only way to obtain a shard by index, and
  //therefore the only way an entry can ever enter this table.
  LockedShard lockShard(uint32_t shardIdx) {
    if(shardIdx >= shards.size())
      throw StringError("ShardedMap::lockShard: shard index out of range");
    //Registering costs one atomic exchange per call and, only on a shard's FIRST use, one atomic
    //increment - no lock is taken here, so this path acquires exactly the mutexes it acquired before
    //(the shard's own, below) and no others. Exactly one thread sees the flag go false->true, so
    //exactly one slot of occupiedShards is written per occupied shard.
    if(!occupiedFlags[shardIdx].exchange(true,std::memory_order_acq_rel)) {
      size_t slot = numOccupied.fetch_add(1,std::memory_order_acq_rel);
      occupiedShards[slot] = shardIdx;
    }
    //Marked before the shard's mutex is taken, so the shard is registered no later than the first
    //moment anyone could insert into it.
    return LockedShard(mutexPool->getMutex(shardIdx), shards[shardIdx]);
  }

  //Number of shards that may hold entries. Every shard not counted here is empty. Also puts the
  //occupancy list into ascending shard order, so that a walk over it visits shards in exactly the
  //order a walk over the whole table would have - the occupancy list is a subsequence of 0..numShards-1,
  //not a differently-ordered set. That matters because some walks accumulate floating-point sums whose
  //rounding depends on visit order, and this keeps such a walk's result identical to the full walk's.
  //ASSUMES no other thread is calling lockShard concurrently - the registering threads' writes reach
  //this one by whatever joined or synchronized them, exactly as the shard contents themselves do.
  size_t getNumOccupiedShards() {
    size_t n = numOccupied.load(std::memory_order_acquire);
    std::sort(occupiedShards.begin(),occupiedShards.begin()+(ptrdiff_t)n);
    return n;
  }

  //The pos'th shard that may hold entries, 0 <= pos < getNumOccupiedShards(). Indexing by POSITION in
  //the occupancy list rather than by shard index is the point: a caller cannot name a shard the table
  //does not already know about, so a walk over the table's contents cannot be written to miss one.
  //Positions are in ascending shard order as of the last getNumOccupiedShards() call, which every walk
  //makes to get its bound.
  //ASSUMES no other thread is calling lockShard concurrently.
  Shard& getOccupiedShard(size_t pos) {
    if(pos >= numOccupied.load(std::memory_order_acquire))
      throw StringError("ShardedMap::getOccupiedShard: position out of range");
    return shards[occupiedShards[pos]];
  }

  //Forget the occupancy list, after the caller has emptied every occupied shard. Refuses if any of
  //them still holds entries, since that would be the caller discarding the only record of where its
  //remaining entries live.
  //ASSUMES no other thread is calling lockShard concurrently.
  void clearOccupancyTrackingAfterEmptyingAllShards() {
    size_t n = numOccupied.load(std::memory_order_acquire);
    for(size_t pos = 0; pos<n; pos++) {
      if(!shards[occupiedShards[pos]].empty())
        throw StringError("ShardedMap: occupancy tracking cleared while a shard still holds entries");
    }
    for(size_t pos = 0; pos<n; pos++)
      occupiedFlags[occupiedShards[pos]].store(false,std::memory_order_relaxed);
    numOccupied.store(0,std::memory_order_release);
  }

  //Direct check of the invariant this type exists to maintain: no shard holds entries without being
  //listed as occupied. O(numShards), so this is a verification tool, not something a search calls.
  //ASSUMES no other thread is calling lockShard concurrently.
  void checkOccupancyTrackingIsComplete() const {
    size_t n = numOccupied.load(std::memory_order_acquire);
    std::vector<bool> listed(shards.size(),false);
    for(size_t pos = 0; pos<n; pos++)
      listed[occupiedShards[pos]] = true;
    for(size_t i = 0; i<shards.size(); i++) {
      if(!shards[i].empty() && !listed[i])
        throw StringError(
          "ShardedMap: shard " + Global::uint64ToString((uint64_t)i) + " holds entries but is not tracked as occupied"
        );
    }
  }

 private:
  static size_t checkedNumShards(int64_t numShardsToUse) {
    if(numShardsToUse <= 0 || numShardsToUse > ((int64_t)1 << 31))
      throw StringError("ShardedMap: numShards out of range: " + Global::int64ToString(numShardsToUse));
    return (size_t)numShardsToUse;
  }

  std::vector<Shard> shards;
  //One flag per shard (1 byte each) and one slot per shard (4 bytes each) - 5 bytes per shard of
  //bookkeeping, against the ~48 bytes of std::map header plus ~40 bytes of std::mutex a shard already
  //costs. At the shipped 65,536 shards that is 320 KiB per table.
  std::vector<std::atomic<bool>> occupiedFlags;
  std::unique_ptr<MutexPool> mutexPool;
  std::vector<uint32_t> occupiedShards;
  std::atomic<size_t> numOccupied;
};

#endif  // SEARCH_SHARDEDMAP_H_
