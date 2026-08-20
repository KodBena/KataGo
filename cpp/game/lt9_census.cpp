/*
 * TEMPORARY INSTRUMENTATION -- LT-9 ladder/area redundancy census. See lt9_census.h.
 * Compiled only when KATAGO_LT9_CENSUS is defined (a CMake option, off by default).
 */
#ifdef KATAGO_LT9_CENSUS

#include "../game/lt9_census.h"
#include "../game/laddercache.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace lt9census {

namespace {

// ---- thread-local per-dispatch expansion accumulators (no locking on the hot path) ----
thread_local uint64_t t_ladderNodeExpansionsSinceTake = 0;
thread_local uint64_t t_areaQueueNodesSinceTake = 0;

// ---- global state, mutex-guarded (instrumentation only; no timing claims made anywhere
//      in this project depend on this mutex's cost) ----
std::mutex g_mutex;

// Ladder side.
// key -> (firstSeenEvalId, timesSeen)
struct LadderKeyInfo { uint64_t firstSeenEvalId; uint64_t timesSeen; };
std::unordered_map<uint64_t, LadderKeyInfo> g_ladderKeySeen;
uint64_t g_evalIdCounter = 0; // bumped whenever passIndex==0 arrives (new eval)
uint64_t g_curEvalId = 0;

uint64_t g_ladderDispatchesTotal = 0;
uint64_t g_ladderDispatchesByPass[3] = {0,0,0};
uint64_t g_ladderExpansionsTotal = 0;
uint64_t g_ladderExpansionsByPass[3] = {0,0,0};

// Redundant = key already seen anywhere before this dispatch (globally, across passes/evals).
uint64_t g_ladderRedundantDispatchesTotal = 0;
uint64_t g_ladderRedundantDispatchesByPass[3] = {0,0,0};
uint64_t g_ladderRedundantExpansionsTotal = 0;
uint64_t g_ladderRedundantExpansionsByPass[3] = {0,0,0};

// Within-eval redundant = key's first sighting was in THIS SAME eval (an earlier pass, 0 or 1).
uint64_t g_ladderWithinEvalRedundantDispatches = 0;
uint64_t g_ladderWithinEvalRedundantExpansions = 0;
// Cross-eval redundant = key's first sighting was in a strictly earlier eval.
uint64_t g_ladderCrossEvalRedundantDispatches = 0;
uint64_t g_ladderCrossEvalRedundantExpansions = 0;

// Area side.
struct AreaKeyInfo { uint64_t timesSeen; };
std::unordered_map<uint64_t, AreaKeyInfo> g_areaKeySeen; // key = hash(boardHash0,boardHash1,pla)
uint64_t g_areaCallsTotal = 0;
uint64_t g_areaCallsByPla[2] = {0,0};
uint64_t g_areaQueueNodesTotal = 0;
uint64_t g_areaQueueNodesByPla[2] = {0,0};
uint64_t g_areaRedundantCallsTotal = 0; // exact (board,pla) already seen before
uint64_t g_areaRedundantQueueNodesTotal = 0;

uint64_t mixHash(uint64_t a, uint64_t b) {
  // Simple 64-bit mix (splitmix64-style finalizer), enough for a diagnostic key -- collisions
  // would only ever make this instrumentation UNDERSTATE distinct-key counts, never overstate,
  // and none of that arithmetic is on a hot/measured path per the project's no-timing constraint.
  uint64_t x = a ^ (b + 0x9E3779B97F4A7C15ULL + (a<<6) + (a>>2));
  x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return x;
}

bool g_atexitInstalled = false;

} // anonymous namespace

void addLadderNodeExpansions(uint64_t n) {
  t_ladderNodeExpansionsSinceTake += n;
}
uint64_t takeLadderNodeExpansions() {
  uint64_t v = t_ladderNodeExpansionsSinceTake;
  t_ladderNodeExpansionsSinceTake = 0;
  return v;
}

void addAreaQueueNodes(uint64_t n) {
  t_areaQueueNodesSinceTake += n;
}
uint64_t takeAreaQueueNodes() {
  uint64_t v = t_areaQueueNodesSinceTake;
  t_areaQueueNodesSinceTake = 0;
  return v;
}

uint64_t hashLocSet(std::vector<int> locs) {
  std::sort(locs.begin(), locs.end());
  uint64_t h = 1469598103934665603ULL; // FNV-1a offset basis
  for(int l : locs) {
    h ^= (uint64_t)(uint32_t)l;
    h *= 1099511628211ULL; // FNV prime
  }
  return h;
}

void recordLadderDispatch(int passIndex, uint64_t contentKey, uint64_t expansions) {
  std::lock_guard<std::mutex> lock(g_mutex);

  if(passIndex == 0) {
    g_evalIdCounter += 1;
    g_curEvalId = g_evalIdCounter;
  }

  g_ladderDispatchesTotal += 1;
  g_ladderDispatchesByPass[passIndex] += 1;
  g_ladderExpansionsTotal += expansions;
  g_ladderExpansionsByPass[passIndex] += expansions;

  auto it = g_ladderKeySeen.find(contentKey);
  if(it == g_ladderKeySeen.end()) {
    g_ladderKeySeen.emplace(contentKey, LadderKeyInfo{g_curEvalId, 1});
  } else {
    it->second.timesSeen += 1;
    g_ladderRedundantDispatchesTotal += 1;
    g_ladderRedundantDispatchesByPass[passIndex] += 1;
    g_ladderRedundantExpansionsTotal += expansions;
    g_ladderRedundantExpansionsByPass[passIndex] += expansions;
    if(it->second.firstSeenEvalId == g_curEvalId) {
      g_ladderWithinEvalRedundantDispatches += 1;
      g_ladderWithinEvalRedundantExpansions += expansions;
    } else {
      g_ladderCrossEvalRedundantDispatches += 1;
      g_ladderCrossEvalRedundantExpansions += expansions;
    }
  }
}

void recordAreaCall(int pla, uint64_t boardHash0, uint64_t boardHash1, uint64_t queueNodes) {
  std::lock_guard<std::mutex> lock(g_mutex);

  g_areaCallsTotal += 1;
  g_areaCallsByPla[pla] += 1;
  g_areaQueueNodesTotal += queueNodes;
  g_areaQueueNodesByPla[pla] += queueNodes;

  uint64_t key = mixHash(mixHash(boardHash0, boardHash1), (uint64_t)pla);
  auto it = g_areaKeySeen.find(key);
  if(it == g_areaKeySeen.end()) {
    g_areaKeySeen.emplace(key, AreaKeyInfo{1});
  } else {
    it->second.timesSeen += 1;
    g_areaRedundantCallsTotal += 1;
    g_areaRedundantQueueNodesTotal += queueNodes;
  }
}

void dumpAndReset() {
  std::lock_guard<std::mutex> lock(g_mutex);

  if(g_ladderDispatchesTotal == 0 && g_areaCallsTotal == 0)
    return;

  const char* outPathEnv = std::getenv("LT9_CENSUS_OUT");
  std::string outPath = outPathEnv != nullptr ? std::string(outPathEnv) : std::string("lt9_census_out.txt");

  FILE* f = std::fopen(outPath.c_str(), "a");
  if(f == nullptr)
    return;

  std::fprintf(f, "== lt9_census dump ==\n");
  std::fprintf(f, "ladder.dispatches.total=%llu\n", (unsigned long long)g_ladderDispatchesTotal);
  std::fprintf(f, "ladder.dispatches.pass0=%llu\n", (unsigned long long)g_ladderDispatchesByPass[0]);
  std::fprintf(f, "ladder.dispatches.pass1=%llu\n", (unsigned long long)g_ladderDispatchesByPass[1]);
  std::fprintf(f, "ladder.dispatches.pass2=%llu\n", (unsigned long long)g_ladderDispatchesByPass[2]);
  std::fprintf(f, "ladder.expansions.total=%llu\n", (unsigned long long)g_ladderExpansionsTotal);
  std::fprintf(f, "ladder.expansions.pass0=%llu\n", (unsigned long long)g_ladderExpansionsByPass[0]);
  std::fprintf(f, "ladder.expansions.pass1=%llu\n", (unsigned long long)g_ladderExpansionsByPass[1]);
  std::fprintf(f, "ladder.expansions.pass2=%llu\n", (unsigned long long)g_ladderExpansionsByPass[2]);
  std::fprintf(f, "ladder.redundant.dispatches.total=%llu\n", (unsigned long long)g_ladderRedundantDispatchesTotal);
  std::fprintf(f, "ladder.redundant.dispatches.pass0=%llu\n", (unsigned long long)g_ladderRedundantDispatchesByPass[0]);
  std::fprintf(f, "ladder.redundant.dispatches.pass1=%llu\n", (unsigned long long)g_ladderRedundantDispatchesByPass[1]);
  std::fprintf(f, "ladder.redundant.dispatches.pass2=%llu\n", (unsigned long long)g_ladderRedundantDispatchesByPass[2]);
  std::fprintf(f, "ladder.redundant.expansions.total=%llu\n", (unsigned long long)g_ladderRedundantExpansionsTotal);
  std::fprintf(f, "ladder.redundant.expansions.pass0=%llu\n", (unsigned long long)g_ladderRedundantExpansionsByPass[0]);
  std::fprintf(f, "ladder.redundant.expansions.pass1=%llu\n", (unsigned long long)g_ladderRedundantExpansionsByPass[1]);
  std::fprintf(f, "ladder.redundant.expansions.pass2=%llu\n", (unsigned long long)g_ladderRedundantExpansionsByPass[2]);
  std::fprintf(f, "ladder.withinEval.redundant.dispatches=%llu\n", (unsigned long long)g_ladderWithinEvalRedundantDispatches);
  std::fprintf(f, "ladder.withinEval.redundant.expansions=%llu\n", (unsigned long long)g_ladderWithinEvalRedundantExpansions);
  std::fprintf(f, "ladder.crossEval.redundant.dispatches=%llu\n", (unsigned long long)g_ladderCrossEvalRedundantDispatches);
  std::fprintf(f, "ladder.crossEval.redundant.expansions=%llu\n", (unsigned long long)g_ladderCrossEvalRedundantExpansions);
  std::fprintf(f, "ladder.distinctKeys=%llu\n", (unsigned long long)g_ladderKeySeen.size());
  std::fprintf(f, "ladder.evals.total=%llu\n", (unsigned long long)g_evalIdCounter);

  std::fprintf(f, "area.calls.total=%llu\n", (unsigned long long)g_areaCallsTotal);
  std::fprintf(f, "area.calls.black=%llu\n", (unsigned long long)g_areaCallsByPla[0]);
  std::fprintf(f, "area.calls.white=%llu\n", (unsigned long long)g_areaCallsByPla[1]);
  std::fprintf(f, "area.queueNodes.total=%llu\n", (unsigned long long)g_areaQueueNodesTotal);
  std::fprintf(f, "area.queueNodes.black=%llu\n", (unsigned long long)g_areaQueueNodesByPla[0]);
  std::fprintf(f, "area.queueNodes.white=%llu\n", (unsigned long long)g_areaQueueNodesByPla[1]);
  std::fprintf(f, "area.redundant.calls.total=%llu\n", (unsigned long long)g_areaRedundantCallsTotal);
  std::fprintf(f, "area.redundant.queueNodes.total=%llu\n", (unsigned long long)g_areaRedundantQueueNodesTotal);
  std::fprintf(f, "area.distinctKeys=%llu\n", (unsigned long long)g_areaKeySeen.size());

  // Distribution detail: histogram of ladder-key multiplicity (how many distinct chain keys
  // were seen exactly N times), to distinguish "a few chains dominate" from "uniformly spread".
  std::fprintf(f, "-- ladder key multiplicity histogram (multiplicity: count of distinct keys) --\n");
  std::unordered_map<uint64_t,uint64_t> multiplicityHist;
  for(auto& kv : g_ladderKeySeen)
    multiplicityHist[kv.second.timesSeen] += 1;
  std::vector<std::pair<uint64_t,uint64_t>> sortedHist(multiplicityHist.begin(), multiplicityHist.end());
  std::sort(sortedHist.begin(), sortedHist.end());
  for(auto& kv : sortedHist)
    std::fprintf(f, "mult=%llu distinctKeys=%llu\n", (unsigned long long)kv.first, (unsigned long long)kv.second);

  // Top-20 hottest ladder keys by times-seen, to see whether a small set dominates dispatches.
  std::fprintf(f, "-- top ladder keys by timesSeen --\n");
  std::vector<std::pair<uint64_t,uint64_t>> byCount; // (timesSeen, key)
  byCount.reserve(g_ladderKeySeen.size());
  for(auto& kv : g_ladderKeySeen)
    byCount.emplace_back(kv.second.timesSeen, kv.first);
  std::sort(byCount.begin(), byCount.end(), std::greater<>());
  size_t topN = std::min<size_t>(20, byCount.size());
  for(size_t i = 0; i < topN; i++)
    std::fprintf(f, "key=%llu timesSeen=%llu\n", (unsigned long long)byCount[i].second, (unsigned long long)byCount[i].first);

  //TEMPORARY -- the SHIPPED ladder memo's own hit/miss totals ride in the same dump block as the
  //chain-shape census numbers above, so the two are read off one identical run.
  {
    uint64_t cacheHits = 0, cacheMisses = 0;
    LadderCache::censusTotals(cacheHits, cacheMisses);
    std::fprintf(f, "laddercache.hits=%llu\n", (unsigned long long)cacheHits);
    std::fprintf(f, "laddercache.misses=%llu\n", (unsigned long long)cacheMisses);
  }

  std::fclose(f);

  // Reset for the next dump call (e.g. multiple processes appending to the same file across
  // corpus runs; each dumpAndReset() call emits one self-contained block).
  g_ladderKeySeen.clear();
  g_evalIdCounter = 0;
  g_curEvalId = 0;
  g_ladderDispatchesTotal = 0;
  for(int i=0;i<3;i++) g_ladderDispatchesByPass[i]=0;
  g_ladderExpansionsTotal = 0;
  for(int i=0;i<3;i++) g_ladderExpansionsByPass[i]=0;
  g_ladderRedundantDispatchesTotal = 0;
  for(int i=0;i<3;i++) g_ladderRedundantDispatchesByPass[i]=0;
  g_ladderRedundantExpansionsTotal = 0;
  for(int i=0;i<3;i++) g_ladderRedundantExpansionsByPass[i]=0;
  g_ladderWithinEvalRedundantDispatches = 0;
  g_ladderWithinEvalRedundantExpansions = 0;
  g_ladderCrossEvalRedundantDispatches = 0;
  g_ladderCrossEvalRedundantExpansions = 0;
  g_areaKeySeen.clear();
  g_areaCallsTotal = 0;
  for(int i=0;i<2;i++) g_areaCallsByPla[i]=0;
  g_areaQueueNodesTotal = 0;
  for(int i=0;i<2;i++) g_areaQueueNodesByPla[i]=0;
  g_areaRedundantCallsTotal = 0;
  g_areaRedundantQueueNodesTotal = 0;
}

void installDumpAtExit() {
  std::lock_guard<std::mutex> lock(g_mutex);
  if(g_atexitInstalled)
    return;
  g_atexitInstalled = true;
  std::atexit([](){ dumpAndReset(); });
}

} // namespace lt9census

#endif // KATAGO_LT9_CENSUS
