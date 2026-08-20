/*
 * TEMPORARY INSTRUMENTATION -- LT-9 ladder/area redundancy census.
 *
 * This is scaffolding for a bounded measurement experiment, not shippable code. It exists
 * to answer one question: how much of the per-NN-eval work done by iterLadders (nninputs.cpp)
 * and Board::calculateAreaForPla (board.cpp) is redundant across the three per-eval passes
 * (current board, prev board, prev-prev board) and across successive evals, per
 * audit-reports/host-microarch-analysis.md section LT-9 and
 * audit-reports/exp-lt9-ladder-census.md.
 *
 * Entirely inert unless compiled with -DKATAGO_LT9_CENSUS=1 (a CMake option, off by default).
 * What would have to change for any of this to ship: see the "What would have to change to
 * ship" section of exp-lt9-ladder-census.md -- in short, none of it is intended to ship as-is;
 * a real cache (if warranted) would be a from-scratch design, not a promotion of this counter
 * scaffolding.
 */
#ifndef GAME_LT9_CENSUS_H
#define GAME_LT9_CENSUS_H

#ifdef KATAGO_LT9_CENSUS

#include <cstdint>
#include <vector>

namespace lt9census {

  // ---- Ladder side (Board::searchIsLadderCaptured / searchIsLadderCapturedAttackerFirst2Libs,
  //      board.cpp; dispatched from the single shared iterLadders in nninputs.cpp) ----

  // Called at each return point of Board::searchIsLadderCaptured (board.cpp) with the
  // searchNodeCount accumulated on THIS call. Thread-local, no locking on the hot path.
  void addLadderNodeExpansions(uint64_t n);

  // Called by iterLadders around a top-level dispatch (a call to searchIsLadderCaptured or
  // searchIsLadderCapturedAttackerFirst2Libs) to read and zero the thread-local expansion
  // counter accumulated since the last call -- i.e. the cost of exactly that one dispatch,
  // including any nested searchIsLadderCaptured calls AttackerFirst2Libs makes internally.
  uint64_t takeLadderNodeExpansions();

  // Records one ladder chain-search dispatch (a chain iterLadders decided needed a real search,
  // i.e. NOT already deduped by iterLadders's own within-call chainHeadsSolved cache).
  //   passIndex: 0 = current board, 1 = prev board, 2 = prev-prev board (see nninputs.cpp
  //     call sites -- iterLadders is always invoked exactly 3x per eval, in this order, on
  //     the calling thread, with no other caller in the tree).
  //   contentKey: an FNV-1a hash of the chain's member locations + liberty locations (see
  //     lt9census::hashLocSet below) -- the content-addressed key a (board hash, chain head)
  //     cache from the finding's remedy sketch would need, approximated at chain-shape
  //     granularity since a literal whole-board Zobrist hash differs on every ply by
  //     construction and would trivially show ~0% redundancy, telling us nothing about
  //     whether the remedy's actual intended key (same chain, unrelated distant board content)
  //     would hit. This is a heuristic surrogate, documented as such in the report: it can
  //     OVERSTATE true cacheable redundancy, because it ignores far-field board content that
  //     could (rarely) change a ladder's outcome despite the pursued chain being unchanged.
  //   expansions: node-expansion cost of this dispatch (from takeLadderNodeExpansions()).
  void recordLadderDispatch(int passIndex, uint64_t contentKey, uint64_t expansions);

  // ---- Area side (Board::calculateAreaForPla, board.cpp) ----

  // Called once per region-queue push inside calculateAreaForPla's BFS (thread-local).
  void addAreaQueueNodes(uint64_t n);
  uint64_t takeAreaQueueNodes();

  // Records one calculateAreaForPla call.
  //   pla: 0 = BLACK, 1 = WHITE (the two calls calculateArea always makes per invocation).
  //   boardHash0/1: the board's own Hash128 pos_hash (exact-board identity; see header comment
  //     above recordLadderDispatch -- no finer-grained decomposition of Benson's algorithm is
  //     exercised here, so this measures only exact-board repeat, not partial/chain-level reuse).
  //   queueNodes: work units for this call (from takeAreaQueueNodes()).
  void recordAreaCall(int pla, uint64_t boardHash0, uint64_t boardHash1, uint64_t queueNodes);

  // FNV-1a 64-bit hash of a sorted list of ints (Locs). Sorts a copy; caller passes the raw list.
  uint64_t hashLocSet(std::vector<int> locs);

  // Dumps all accumulated counters to $LT9_CENSUS_OUT (or ./lt9_census_out.txt if unset) and
  // clears them. Safe to call multiple times (e.g. once at process exit via atexit, installed
  // lazily on first use). No-op if nothing was recorded.
  void dumpAndReset();

  // Installs an atexit hook that calls dumpAndReset(). Idempotent.
  void installDumpAtExit();

} // namespace lt9census

#define LT9_CENSUS_LADDER_EXPANSIONS(n) ::lt9census::addLadderNodeExpansions(n)
#define LT9_CENSUS_TAKE_LADDER_EXPANSIONS() ::lt9census::takeLadderNodeExpansions()
#define LT9_CENSUS_LADDER_DISPATCH(passIdx,key,exp) ::lt9census::recordLadderDispatch(passIdx,key,exp)
#define LT9_CENSUS_AREA_QUEUE_NODES(n) ::lt9census::addAreaQueueNodes(n)
#define LT9_CENSUS_TAKE_AREA_QUEUE_NODES() ::lt9census::takeAreaQueueNodes()
#define LT9_CENSUS_AREA_CALL(pla,h0,h1,q) ::lt9census::recordAreaCall(pla,h0,h1,q)
#define LT9_CENSUS_INSTALL_ATEXIT() ::lt9census::installDumpAtExit()

#else // !KATAGO_LT9_CENSUS

#define LT9_CENSUS_LADDER_EXPANSIONS(n) ((void)0)
#define LT9_CENSUS_TAKE_LADDER_EXPANSIONS() (0)
#define LT9_CENSUS_LADDER_DISPATCH(passIdx,key,exp) ((void)0)
#define LT9_CENSUS_AREA_QUEUE_NODES(n) ((void)0)
#define LT9_CENSUS_TAKE_AREA_QUEUE_NODES() (0)
#define LT9_CENSUS_AREA_CALL(pla,h0,h1,q) ((void)0)
#define LT9_CENSUS_INSTALL_ATEXIT() ((void)0)

#endif // KATAGO_LT9_CENSUS

#endif // GAME_LT9_CENSUS_H
