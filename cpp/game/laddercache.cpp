/*
 * Implementation of the sound, search-lifetime ladder memo. See laddercache.h for the key, the
 * two closure rules, and what the types do and do not guarantee.
 */
#include "../game/laddercache.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <string>

#include "../core/global.h"

//The packed representation puts the color in the low two bits, so a Loc must fit in the rest.
static_assert(Board::MAX_ARR_SIZE <= (1 << 14), "packed (loc<<2)|color needs Loc to fit in 14 bits");
static_assert(NUM_BOARD_COLORS <= 4, "packed (loc<<2)|color needs a color to fit in 2 bits");

namespace LadderRead {
  thread_local Accum* t_accum = nullptr;
}

namespace {
  //One accumulator per thread, reused across dispatches: arming a scope bumps a stamp id rather
  //than clearing an array of MAX_ARR_SIZE entries.
  thread_local LadderRead::Accum t_ownAccum;

  //Verification mode is a process-wide arming switch, read on the hit path. It is a relaxed
  //atomic rather than a plain bool because a test arms it from one thread and feature generation
  //reads it from another.
  //Verification arming. KATAGO_LADDER_CACHE_VERIFY=1 turns it on for a whole process without a
  //patched binary, which is what lets the soundness re-check run against the SHIPPED mechanism
  //over a real corpus (ADR-0011 Rule 3's shipped-binding requirement). It changes cost only: with
  //it armed the search runs on hits too and its answer is compared against the cached one. No
  //path can return a different answer with it on than with it off.
  std::atomic<bool> g_verifyEnabled{[]() {
    const char* env = std::getenv("KATAGO_LADDER_CACHE_VERIFY");
    return env != nullptr && std::string(env) != "0" && std::string(env) != "";
  }()};

#ifdef KATAGO_LT9_CENSUS
  //TEMPORARY -- LT-9 census totals; see laddercache.h.
  std::atomic<uint64_t> g_censusHits{0};
  std::atomic<uint64_t> g_censusMisses{0};
#endif

  uint64_t mixHash(uint64_t a, uint64_t b) {
    uint64_t x = a ^ (b + 0x9E3779B97F4A7C15ULL + (a<<6) + (a>>2));
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
  }
}

// ------------------------------------------------------------------------------------------
// LadderReadSet
// ------------------------------------------------------------------------------------------

bool LadderReadSet::validates(const Board& board) const {
  if(board.ko_loc != koLoc_)
    return false;
  for(uint16_t p : packed_) {
    if(board.colors[p >> 2] != (Color)(p & 3))
      return false;
  }
  return true;
}

// ------------------------------------------------------------------------------------------
// LadderReadScope
// ------------------------------------------------------------------------------------------

uint64_t ladderBucketKey(const Board& board, Loc rootLoc, int variant) {
  uint64_t h = 1469598103934665603ULL; //FNV-1a offset basis
  auto absorb = [&h](uint64_t v) { h ^= v; h *= 1099511628211ULL; };
  //The pursued chain's member locations, each followed by its own empty neighbours. The walk
  //order is the board's circular chain list from rootLoc, and iterLadders always dispatches a
  //given chain from the same rootLoc (the first stone of it in scan order), so it is stable.
  Loc cur = rootLoc;
  do {
    absorb((uint64_t)(uint16_t)cur);
    for(int i = 0; i < 4; i++) {
      Loc adj = cur + board.adj_offsets[i];
      if(board.colors[adj] == C_EMPTY)
        absorb(0x10000ULL | (uint64_t)(uint16_t)adj);
    }
    cur = board.next_in_chain[cur];
  } while(cur != rootLoc);
  return mixHash(mixHash(h, (uint64_t)(uint32_t)rootLoc), (uint64_t)variant);
}

LadderReadScope::LadderReadScope(const Board& board, uint64_t bucketKey)
  : board_(board), bucketKey_(bucketKey)
{
  LadderRead::Accum& a = t_ownAccum;
  if(a.stamp.size() != (size_t)Board::MAX_ARR_SIZE)
    a.stamp.assign((size_t)Board::MAX_ARR_SIZE, 0u);
  a.stampId += 1;
  if(a.stampId == 0) { //wrapped: clear, so a stale stamp cannot alias a fresh one
    std::fill(a.stamp.begin(), a.stamp.end(), 0u);
    a.stampId = 1;
  }
  a.locs.clear();
  LadderRead::t_accum = &a;
}

LadderReadScope::~LadderReadScope() {
  LadderRead::t_accum = nullptr;
}

LadderReadSet LadderReadScope::finish() && {
  LadderRead::Accum& a = t_ownAccum;

  LadderReadSet rs;
  rs.packed_.reserve(a.locs.size());
  for(uint16_t loc : a.locs)
    rs.packed_.push_back((uint16_t)((loc << 2) | (uint16_t)(uint8_t)board_.colors[loc]));
  //Sorted so validates() walks the colors array forward rather than jumping around it, and so two
  //read sets over the same points compare in a canonical order.
  std::sort(rs.packed_.begin(), rs.packed_.end());
  rs.koLoc_ = (int16_t)board_.ko_loc;
  rs.bucketKey_ = bucketKey_;

  LadderRead::t_accum = nullptr;
  return rs;
}

// ------------------------------------------------------------------------------------------
// LadderCache
// ------------------------------------------------------------------------------------------

LadderCache::LadderCache(size_t bucketCap, size_t maxBytes)
  : bucketCap_(bucketCap), maxBytes_(maxBytes)
{}

#ifdef KATAGO_LT9_CENSUS
#define LT9_CACHE_COUNT_HIT()  (g_censusHits.fetch_add(1, std::memory_order_relaxed))
#define LT9_CACHE_COUNT_MISS() (g_censusMisses.fetch_add(1, std::memory_order_relaxed))
#else
#define LT9_CACHE_COUNT_HIT()  ((void)0)
#define LT9_CACHE_COUNT_MISS() ((void)0)
#endif

std::optional<LadderAnswer> LadderCache::lookup(const Board& board, uint64_t bucketKey) {
  auto it = buckets_.find(bucketKey);
  if(it == buckets_.end()) {
    misses_ += 1;
    LT9_CACHE_COUNT_MISS();
    return std::nullopt;
  }
  std::vector<Entry>& bucket = it->second;
  for(size_t i = 0; i < bucket.size(); i++) {
    if(!bucket[i].readSet.validates(board))
      continue;
    LadderAnswer answer = bucket[i].answer;
    //MRU: the next dispatch on this chain is overwhelmingly the same position one played move
    //later, so the entry that just validated is the one it should examine first.
    if(i != 0)
      std::rotate(bucket.begin(), bucket.begin() + (long)i, bucket.begin() + (long)i + 1);
    hits_ += 1;
    LT9_CACHE_COUNT_HIT();
    return answer;
  }
  misses_ += 1;
  LT9_CACHE_COUNT_MISS();
  return std::nullopt;
}

#ifdef KATAGO_LT9_CENSUS
void LadderCache::censusTotals(uint64_t& hits, uint64_t& misses) {
  hits = g_censusHits.load(std::memory_order_relaxed);
  misses = g_censusMisses.load(std::memory_order_relaxed);
}
#endif

void LadderCache::insert(LadderReadSet&& readSet, const LadderAnswer& answer) {
  //An empty read set constrains nothing, so an entry carrying one would validate against every
  //board. That can only arise if no scope was armed while the search ran, which is a caller
  //error rather than a board state, so it is refused here rather than stored.
  if(readSet.isEmpty())
    return;

  //The store is bounded in bytes, the resource that actually exhausts. When the budget is spent
  //the whole store is dropped rather than partially evicted: every entry is independently valid,
  //so dropping any subset costs hit rate and nothing else, and one wholesale drop keeps the
  //accounting exact instead of tracking per-entry sizes through evictions.
  if(bytesUsed_ + readSet.heapBytes() > maxBytes_)
    clear();

  std::vector<Entry>& bucket = buckets_[readSet.bucketKey()];
  if(bucket.size() >= bucketCap_) {
    bytesUsed_ -= std::min(bytesUsed_, bucket.back().readSet.heapBytes());
    bucket.pop_back(); //LRU out: the back is the least recently validated
  }
  bytesUsed_ += readSet.heapBytes();
  bucket.insert(bucket.begin(), Entry{std::move(readSet), answer});
}

void LadderCache::clear() {
  buckets_.clear();
  bytesUsed_ = 0;
}

void LadderCache::setVerify(bool on) {
  g_verifyEnabled.store(on, std::memory_order_relaxed);
}

bool LadderCache::verifyEnabled() {
  return g_verifyEnabled.load(std::memory_order_relaxed);
}

void LadderCache::reportVerification(const LadderAnswer& cached, const LadderAnswer& recomputed) {
  bool same = cached.laddered == recomputed.laddered
    && cached.numWorkingMoves == recomputed.numWorkingMoves;
  if(same) {
    for(uint8_t i = 0; i < cached.numWorkingMoves; i++) {
      if(cached.workingMoves[i] != recomputed.workingMoves[i]) { same = false; break; }
    }
  }
  if(same)
    return;
  //The soundness argument says this cannot happen: a validated read set means the search would
  //have followed the identical execution. A disagreement therefore means the argument is wrong or
  //a consultation site lost its mark -- an invariant violation, not a recoverable condition, so
  //it aborts loudly rather than quietly preferring one of the two answers.
  Global::fatalError(
    "LadderCache verification failed: a validated read set carried laddered=" +
    Global::boolToString(cached.laddered) + " (" + Global::intToString(cached.numWorkingMoves) +
    " working moves) but recomputation gave laddered=" + Global::boolToString(recomputed.laddered) +
    " (" + Global::intToString(recomputed.numWorkingMoves) + " working moves). " +
    "The ladder-cache key is unsound or a board-state consultation site in the ladder search "
    "lost its read-set mark. See game/laddercache.h and audit-reports/impl-lt9-ladder-cache.md."
  );
}
