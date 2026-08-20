#ifndef SEARCH_SEARCHNODETABLE_H
#define SEARCH_SEARCHNODETABLE_H

#include "../core/global.h"
#include "../core/hash.h"
#include "../core/multithread.h"
#include "../game/board.h"
#include "../search/shardedmap.h"

struct SearchNode;

//Transposition table for graph search: hash -> node, sharded so that concurrent playouts allocating
//different nodes rarely contend. The shards themselves live in the ShardedMap, which also tracks
//which shards have been used, so that deleting the tree costs time proportional to the number of
//shards that actually hold nodes rather than to the shard count. See shardedmap.h for why reaching a
//shard's map any other way is not possible.
//This table does not own the nodes; the search deletes them.
struct SearchNodeTable {
  ShardedMap<SearchNode*> shards;
  uint32_t numShards;

  SearchNodeTable(int numShardsPowerOfTwo);
  ~SearchNodeTable();

  uint32_t getIndex(uint64_t hash) const;
};

#endif
