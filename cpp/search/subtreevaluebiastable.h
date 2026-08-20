#ifndef SEARCH_SUBTREEVALUEBIASTABLE_H
#define SEARCH_SUBTREEVALUEBIASTABLE_H

#include "../core/global.h"
#include "../core/hash.h"
#include "../core/multithread.h"
#include "../game/board.h"
#include "../search/shardedmap.h"

struct SubtreeValueBiasEntry {
  double deltaUtilitySum = 0.0;
  double weightSum = 0.0;
  mutable std::atomic_flag entryLock = ATOMIC_FLAG_INIT;
};

//Table of subtree value bias entries, keyed by a local-pattern hash of the position around the move.
//Sharded and occupancy-tracked exactly like the node table (see shardedmap.h): clearUnusedSynchronous
//runs once per search, and its cost tracks the number of shards the search actually touched rather
//than the shard count.
struct SubtreeValueBiasTable {
  ShardedMap<std::shared_ptr<SubtreeValueBiasEntry>> shards;

  SubtreeValueBiasTable(int32_t numShards);
  ~SubtreeValueBiasTable();

  // ASSUMES there is no concurrent multithreading of this table or any of its entries,
  // and that all past mutations on this table or any of its entries are now visible to this thread.
  void clearUnusedSynchronous();

  // The board specified here is expected to be the board BEFORE the move is played.
  std::shared_ptr<SubtreeValueBiasEntry> get(Player pla, Loc parentPrevMoveLoc, Loc prevMoveLoc, const Board& prevBoard);
};

#endif


