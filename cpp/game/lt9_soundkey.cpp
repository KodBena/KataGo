/*
 * TEMPORARY INSTRUMENTATION -- LT-9 sound ladder-cache key measurement. See lt9_soundkey.h for
 * the key, the two closure rules, and the soundness argument.
 * Compiled only when KATAGO_LT9_CENSUS is defined (a CMake option, off by default).
 */
#ifdef KATAGO_LT9_CENSUS

#include "../game/lt9_soundkey.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace lt9soundkey {

namespace {

// Board::MAX_ARR_SIZE for the largest board this project compiles for is well under this; the
// bound is asserted-by-clamp at every marking site rather than trusted, since including board.h
// from here would be a circular include (board.cpp includes this header).
constexpr int LOC_ARRAY_BOUND = 4096;

// The most entries one bucket will hold and scan. A bucket's entries differ only in FAR-FIELD
// board content around one identical chain, so a deep bucket means the far field genuinely
// varies. The cap is reported (soundkey.bucketCapBites) so a capacity effect can never be
// mistaken for a key effect; run with LT9_SOUNDKEY_BUCKET_CAP to change it.
constexpr size_t DEFAULT_BUCKET_CAP = 512;

// ---- per-dispatch recording state (thread-local; the ladder path is single-threaded per
//      search thread and never re-enters itself) ----
thread_local bool t_active = false;
thread_local bool t_truncated = false;  // snapshot of g_truncatedMode, taken under the lock
thread_local std::vector<int> t_readLocs;
thread_local std::vector<uint32_t> t_stamp;   // LOC_ARRAY_BOUND entries, dedup by stamp id
thread_local uint32_t t_stampId = 0;

thread_local int t_rootLoc = 0;
thread_local int t_variant = 0;
thread_local int t_koLoc = 0;
thread_local uint64_t t_chainContentKey = 0;
thread_local uint64_t t_boardPosHash = 0;

// What lookup() found, carried to endDispatch().
thread_local bool t_lookupHit = false;
thread_local bool t_lookupResult = false;
thread_local std::vector<int> t_lookupWorkingMoves;
thread_local uint64_t t_lookupSrcPosHash = 0;
thread_local size_t t_lookupEntryIdx = 0;
thread_local uint64_t t_lookupBucketKey = 0;

// ---- search-lifetime store, mutex-guarded ----
struct Entry {
  int koLoc = 0;
  std::vector<int> readLocs;     // parallel to readColors
  std::vector<int8_t> readColors;
  bool result = false;
  std::vector<int> workingMoves;
  uint64_t srcPosHash = 0;
};

std::mutex g_mutex;
std::unordered_map<uint64_t, std::vector<Entry>> g_buckets;

// Counters.
uint64_t g_dispatches = 0;
uint64_t g_expansions = 0;
uint64_t g_hits = 0;
uint64_t g_hitExpansions = 0;
// A hit whose stored entry came from a board with a DIFFERENT pos_hash: the boards genuinely
// differ somewhere, so the cached-vs-recomputed comparison below is exercised on non-identical
// boards rather than trivially on the same board.
uint64_t g_hitsDifferentBoard = 0;
uint64_t g_hitExpansionsDifferentBoard = 0;
// THE WITNESS. A validated entry whose stored answer differs from the answer the real search
// just produced. Under correct closure rules this is 0 by construction; under the red leg
// (LT9_SOUNDKEY_MODE=truncated) it is expected to fire.
uint64_t g_mismatchResult = 0;
uint64_t g_mismatchWorkingMoves = 0;
uint64_t g_bucketCapBites = 0;
uint64_t g_distinctBuckets = 0;

// Read-set size distribution.
uint64_t g_readSetSizeSum = 0;
uint64_t g_readSetSizeMax = 0;
std::vector<uint64_t> g_readSetSizeHist; // index = |R|, value = count of dispatches

bool g_truncatedMode = false;
size_t g_bucketCap = DEFAULT_BUCKET_CAP;
bool g_configRead = false;

void readConfigLocked() {
  if(g_configRead)
    return;
  g_configRead = true;
  const char* mode = std::getenv("LT9_SOUNDKEY_MODE");
  g_truncatedMode = (mode != nullptr && std::string(mode) == "truncated");
  const char* cap = std::getenv("LT9_SOUNDKEY_BUCKET_CAP");
  if(cap != nullptr) {
    long v = std::strtol(cap, nullptr, 10);
    if(v > 0)
      g_bucketCap = (size_t)v;
  }
}

uint64_t mixHash(uint64_t a, uint64_t b) {
  uint64_t x = a ^ (b + 0x9E3779B97F4A7C15ULL + (a<<6) + (a>>2));
  x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return x;
}

} // anonymous namespace

bool recording() {
  return t_active;
}

namespace {
// The unconditional form: adds a point regardless of red-leg state. Only the seed uses it.
void markPointRaw(int loc) {
  if(loc < 0 || loc >= LOC_ARRAY_BOUND)
    return;
  if(t_stamp[(size_t)loc] == t_stampId)
    return;
  t_stamp[(size_t)loc] = t_stampId;
  t_readLocs.push_back(loc);
}
} // anonymous namespace

void markPoint(int loc) {
  if(!t_active)
    return;
  if(t_truncated)
    return; // red leg: record only the seed (the census's chain-shape key), nothing the search reads
  markPointRaw(loc);
}

void beginDispatch(
  int rootLoc,
  int variant,
  int koLoc,
  uint64_t chainContentKey,
  uint64_t boardPosHash,
  const std::vector<int>& seedLocs
) {
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    readConfigLocked();
    t_truncated = g_truncatedMode;
  }
  if(t_stamp.size() != (size_t)LOC_ARRAY_BOUND)
    t_stamp.assign((size_t)LOC_ARRAY_BOUND, 0u);
  t_stampId += 1;
  if(t_stampId == 0) { // wrapped: clear so a stale stamp cannot alias
    t_stamp.assign((size_t)LOC_ARRAY_BOUND, 0u);
    t_stampId = 1;
  }
  t_readLocs.clear();
  t_active = true;
  t_rootLoc = rootLoc;
  t_variant = variant;
  t_koLoc = koLoc;
  t_chainContentKey = chainContentKey;
  t_boardPosHash = boardPosHash;
  t_lookupHit = false;
  t_lookupResult = false;
  t_lookupWorkingMoves.clear();
  t_lookupSrcPosHash = 0;
  t_lookupEntryIdx = 0;
  t_lookupBucketKey = mixHash(mixHash(chainContentKey, (uint64_t)(uint32_t)rootLoc), (uint64_t)variant);

  // The seed is the census's own key material (chain members + liberties). It enters the read set
  // unconditionally -- including under the red leg, where it is the ONLY thing recorded, so the
  // truncated key is exactly the census's chain-shape key.
  for(int l : seedLocs)
    markPointRaw(l);
}

bool lookup(const int8_t* colorsArr, bool& cachedResult, std::vector<int>& cachedWorkingMoves) {
  if(!t_active)
    return false;
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_buckets.find(t_lookupBucketKey);
  if(it == g_buckets.end())
    return false;
  std::vector<Entry>& bucket = it->second;
  if(bucket.size() >= g_bucketCap)
    g_bucketCapBites += 1;
  for(size_t i = 0; i < bucket.size(); i++) {
    const Entry& e = bucket[i];
    if(e.koLoc != t_koLoc)
      continue;
    bool ok = true;
    for(size_t j = 0; j < e.readLocs.size(); j++) {
      if(colorsArr[e.readLocs[j]] != e.readColors[j]) { ok = false; break; }
    }
    if(!ok)
      continue;
    t_lookupHit = true;
    t_lookupResult = e.result;
    t_lookupWorkingMoves = e.workingMoves;
    t_lookupSrcPosHash = e.srcPosHash;
    t_lookupEntryIdx = i;
    cachedResult = e.result;
    cachedWorkingMoves = e.workingMoves;
    return true;
  }
  return false;
}

void endDispatch(
  const int8_t* pristineColors,
  bool realResult,
  const std::vector<int>& realWorkingMoves,
  uint64_t expansions
) {
  if(!t_active)
    return;
  t_active = false;

  std::lock_guard<std::mutex> lock(g_mutex);

  g_dispatches += 1;
  g_expansions += expansions;

  size_t rSize = t_readLocs.size();
  g_readSetSizeSum += (uint64_t)rSize;
  if((uint64_t)rSize > g_readSetSizeMax)
    g_readSetSizeMax = (uint64_t)rSize;
  if(g_readSetSizeHist.size() <= rSize)
    g_readSetSizeHist.resize(rSize + 1, 0);
  g_readSetSizeHist[rSize] += 1;

  if(t_lookupHit) {
    g_hits += 1;
    g_hitExpansions += expansions;
    if(t_lookupSrcPosHash != t_boardPosHash) {
      g_hitsDifferentBoard += 1;
      g_hitExpansionsDifferentBoard += expansions;
    }
    // THE WITNESS: a validated entry must carry the answer the real search just produced.
    if(t_lookupResult != realResult)
      g_mismatchResult += 1;
    else if(t_lookupWorkingMoves != realWorkingMoves)
      g_mismatchWorkingMoves += 1;
    return; // a hit refreshes nothing: the entry it hit already holds this exact read set
  }

  // Miss: store the entry, snapshotting the PRISTINE board's colors over the accumulated read set.
  std::vector<Entry>& bucket = g_buckets[t_lookupBucketKey];
  if(bucket.empty())
    g_distinctBuckets += 1;
  if(bucket.size() >= g_bucketCap)
    bucket.erase(bucket.begin()); // oldest out; the cap's bite is already counted in lookup()
  Entry e;
  e.koLoc = t_koLoc;
  e.readLocs = t_readLocs;
  std::sort(e.readLocs.begin(), e.readLocs.end());
  e.readColors.resize(e.readLocs.size());
  for(size_t j = 0; j < e.readLocs.size(); j++)
    e.readColors[j] = pristineColors[e.readLocs[j]];
  e.result = realResult;
  e.workingMoves = realWorkingMoves;
  e.srcPosHash = t_boardPosHash;
  bucket.push_back(std::move(e));
}

std::string renderDump() {
  std::lock_guard<std::mutex> lock(g_mutex);
  readConfigLocked();

  char buf[512];
  std::string out;
  auto line = [&](const char* fmt, unsigned long long v) {
    std::snprintf(buf, sizeof(buf), fmt, v);
    out += buf;
  };

  out += "-- lt9 SOUND-KEY (read-set) measurement --\n";
  out += std::string("soundkey.mode=") + (g_truncatedMode ? "truncated(RED LEG)" : "full") + "\n";
  line("soundkey.bucketCap=%llu\n", (unsigned long long)g_bucketCap);
  line("soundkey.dispatches=%llu\n", (unsigned long long)g_dispatches);
  line("soundkey.expansions=%llu\n", (unsigned long long)g_expansions);
  line("soundkey.hits=%llu\n", (unsigned long long)g_hits);
  line("soundkey.hitExpansions=%llu\n", (unsigned long long)g_hitExpansions);
  line("soundkey.hitsDifferentBoard=%llu\n", (unsigned long long)g_hitsDifferentBoard);
  line("soundkey.hitExpansionsDifferentBoard=%llu\n", (unsigned long long)g_hitExpansionsDifferentBoard);
  line("soundkey.MISMATCH.result=%llu\n", (unsigned long long)g_mismatchResult);
  line("soundkey.MISMATCH.workingMoves=%llu\n", (unsigned long long)g_mismatchWorkingMoves);
  line("soundkey.bucketCapBites=%llu\n", (unsigned long long)g_bucketCapBites);
  line("soundkey.distinctBuckets=%llu\n", (unsigned long long)g_distinctBuckets);
  line("soundkey.readSetSize.sum=%llu\n", (unsigned long long)g_readSetSizeSum);
  line("soundkey.readSetSize.max=%llu\n", (unsigned long long)g_readSetSizeMax);
  out += "-- soundkey read-set size histogram (size: dispatches) --\n";
  for(size_t i = 0; i < g_readSetSizeHist.size(); i++) {
    if(g_readSetSizeHist[i] == 0)
      continue;
    std::snprintf(buf, sizeof(buf), "rsize=%llu dispatches=%llu\n",
                  (unsigned long long)i, (unsigned long long)g_readSetSizeHist[i]);
    out += buf;
  }
  return out;
}

void reset() {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_buckets.clear();
  g_dispatches = 0;
  g_expansions = 0;
  g_hits = 0;
  g_hitExpansions = 0;
  g_hitsDifferentBoard = 0;
  g_hitExpansionsDifferentBoard = 0;
  g_mismatchResult = 0;
  g_mismatchWorkingMoves = 0;
  g_bucketCapBites = 0;
  g_distinctBuckets = 0;
  g_readSetSizeSum = 0;
  g_readSetSizeMax = 0;
  g_readSetSizeHist.clear();
}

} // namespace lt9soundkey

#endif // KATAGO_LT9_CENSUS
