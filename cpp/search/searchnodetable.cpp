#include "../search/searchnodetable.h"

#include "../core/rand.h"
#include "../search/localpattern.h"

static int64_t numShardsFromPowerOfTwo(int numShardsPowerOfTwo) {
  if(numShardsPowerOfTwo < 0 || numShardsPowerOfTwo > 30)
    throw StringError("SearchNodeTable: numShardsPowerOfTwo out of range: " + Global::intToString(numShardsPowerOfTwo));
  return (int64_t)1 << numShardsPowerOfTwo;
}

SearchNodeTable::SearchNodeTable(int numShardsPowerOfTwo)
  :shards(numShardsFromPowerOfTwo(numShardsPowerOfTwo)),
   numShards(shards.getNumShards())
{}
SearchNodeTable::~SearchNodeTable() {
}

uint32_t SearchNodeTable::getIndex(uint64_t hash) const {
  uint32_t shardMask = numShards-1; //Always a power of two
  return (uint32_t)(hash & shardMask);
}
