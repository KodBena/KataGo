/*
 * A sound, search-lifetime memo for the ladder searches NN feature generation runs.
 *
 * WHY. nninputs.cpp's iterLadders runs a ladder search on the current board and on both previous
 * boards for every evaluated position. On a 60-game opening corpus at 500 visits, 99.7% of those
 * searches repeat work an earlier search already did (audit-reports/exp-lt9-ladder-census.md).
 * The redundancy is CROSS-EVAL, not within one eval, so only a memo that outlives a single
 * evaluation captures it.
 *
 * THE KEY, AND WHY IT IS SOUND. Chain shape is NOT a sound key -- the ladder search plays moves,
 * and its move generation reads whatever opposing chains happen to sit around the ones it
 * touches, so two identically shaped chains can genuinely ladder differently. Measured: keying on
 * chain shape returns a wrong verdict on 13.2% of its own hits
 * (audit-reports/impl-lt9-ladder-cache.md). The key here is instead the set of board points the
 * search's execution actually DEPENDS ON, recorded while it runs, under two closure rules:
 *
 *   RULE-A (a point read)  -- a read of colors[p] contributes {p}.
 *   RULE-B (a chain read)  -- consulting chain-level state at p (chain_head[p],
 *      chain_data[chain_head[p]].{num_liberties,num_locs,owner}, a next_in_chain walk,
 *      getNumLiberties(p)) contributes chain(p) UNION adj(chain(p)).
 *
 * RULE-B's closure is what makes colors-agreement over the recorded set pin both the chain's
 * EXTENT (its boundary points are in the set, so it cannot extend differently) and its LIBERTY
 * SET (its liberties are exactly the empty points of that boundary) -- which is all chain_data is
 * ever consulted for. Chain heads enter the algorithm only through equality comparisons among
 * heads, never as addresses with meaning, so two boards agreeing over the recorded set induce the
 * same head-equality relation and hence the same execution even where the representative Loc
 * differs. Every point the search WRITES is read on the same execution before or as it is
 * written, so writes need no rule of their own. Board::ladderMarkChain / Board::ladderMarkMove
 * apply the two rules at the consultation sites inside the ladder search.
 *
 * WHAT THE TYPES BUY, AND WHAT THEY DO NOT. LadderReadSet has no public constructor and exactly
 * one producer -- LadderReadScope::finish(), rvalue-qualified so it consumes the scope that
 * bracketed a real search. LadderCache::insert accepts only a LadderReadSet, so storing an answer
 * under an invented key has no expressible form; LadderCache::lookup validates internally and
 * returns std::optional, so using an unvalidated answer likewise has none. What that does NOT buy
 * is completeness of the MARKING: a consultation site that reads board state without a
 * corresponding mark would be a silent unsoundness no type here catches, because Board::colors is
 * a public array. The net for that residue is setVerify(): with it armed, every hit is checked
 * against a full recomputation and a disagreement is a loud refusal, not a silent wrong answer.
 */
#ifndef GAME_LADDERCACHE_H
#define GAME_LADDERCACHE_H

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

#include "../game/board.h"

// ------------------------------------------------------------------------------------------
// Read-set accumulation. Board's marking sites call LadderRead::mark, which is a single
// thread-local null-check when no LadderReadScope is active -- the state every path that is not
// generating NN features takes.
// ------------------------------------------------------------------------------------------
namespace LadderRead {
  struct Accum {
    std::vector<uint16_t> locs;   // points recorded so far, in first-touch order
    std::vector<uint32_t> stamp;  // Board::MAX_ARR_SIZE entries; dedup by stamp id
    uint32_t stampId = 0;
  };

  // The accumulator the current thread is recording into, or nullptr.
  extern thread_local Accum* t_accum;

  inline void mark(int loc) {
    Accum* a = t_accum;
    if(a == nullptr)
      return;
    if(loc < 0 || loc >= Board::MAX_ARR_SIZE)
      return;
    if(a->stamp[(size_t)loc] == a->stampId)
      return;
    a->stamp[(size_t)loc] = a->stampId;
    a->locs.push_back((uint16_t)loc);
  }
}

// ------------------------------------------------------------------------------------------
// The recorded key. Constructible only by LadderReadScope.
// ------------------------------------------------------------------------------------------
class LadderReadSet {
public:
  LadderReadSet(const LadderReadSet&) = default;
  LadderReadSet(LadderReadSet&&) noexcept = default;
  LadderReadSet& operator=(const LadderReadSet&) = default;
  LadderReadSet& operator=(LadderReadSet&&) noexcept = default;

  //Does this read set still describe `board`? This is the whole validity test: if every recorded
  //point still carries the color it carried when the answer was computed, and the simple-ko point
  //is the same, then the search would follow the identical execution and produce the identical
  //answer.
  bool validates(const Board& board) const;

  //Which bucket this read set belongs to. Two read sets can only ever validate against the same
  //board state if they agree here, so this is a pure narrowing of the linear scan, never part of
  //the soundness argument.
  uint64_t bucketKey() const { return bucketKey_; }

  size_t numPoints() const { return packed_.size(); }
  //Bytes this read set occupies in a cache entry, for the store's byte budget.
  size_t heapBytes() const { return packed_.capacity() * sizeof(uint16_t); }

  //True when nothing was recorded, which can only happen if the search consulted no board state
  //under an armed scope. Such a read set constrains nothing and would validate against every
  //board, so the cache refuses to store it rather than trusting that it cannot arise.
  bool isEmpty() const { return packed_.empty(); }

private:
  LadderReadSet() = default;
  friend class LadderReadScope;

  //(loc << 2) | color, sorted by loc. Loc needs ceil(log2(MAX_ARR_SIZE)) bits and color needs 2.
  std::vector<uint16_t> packed_;
  int16_t koLoc_ = 0;
  uint64_t bucketKey_ = 0;
};

//The bucket a dispatch on (rootLoc, variant) belongs to, computed from the board BEFORE the
//search runs -- the pursued chain's member and liberty locations, which the search itself is
//about to walk anyway. This has exactly one home so a lookup and the insert that follows it
//cannot disagree (ADR-0012 P1). It is a pure narrowing of the linear scan and carries no part of
//the soundness argument: a wrong bucket key costs hits, never correctness.
//variant: 1 = searchIsLadderCaptured(defenderFirst), 2 = ...AttackerFirst2Libs.
uint64_t ladderBucketKey(const Board& board, Loc rootLoc, int variant);

// ------------------------------------------------------------------------------------------
// The only producer of a LadderReadSet: an RAII scope that arms recording for one dispatch.
// ------------------------------------------------------------------------------------------
class LadderReadScope {
public:
  //Arms read-set recording on the calling thread for the lifetime of this object. `board` is the
  //board the dispatch is about to run on; its colors are the snapshot finish() takes, so it must
  //be the board in its pristine pre-search state. `bucketKey` is ladderBucketKey's value for
  //this dispatch.
  LadderReadScope(const Board& board, uint64_t bucketKey);
  ~LadderReadScope();
  LadderReadScope(const LadderReadScope&) = delete;
  LadderReadScope& operator=(const LadderReadScope&) = delete;

  //Consumes the scope and yields the read set accumulated over it, snapshotted against the board
  //the scope was constructed with. Rvalue-qualified: a scope that is still going to record cannot
  //hand out a key describing only part of what it will record.
  LadderReadSet finish() &&;

private:
  const Board& board_;
  uint64_t bucketKey_;
};

// ------------------------------------------------------------------------------------------
// The answer a ladder dispatch produces.
// ------------------------------------------------------------------------------------------
struct LadderAnswer {
  bool laddered = false;
  //searchIsLadderCapturedAttackerFirst2Libs reports at most its two candidate moves; the
  //one-liberty variant reports none.
  uint8_t numWorkingMoves = 0;
  Loc workingMoves[2] = {0,0};
};

// ------------------------------------------------------------------------------------------
// The memo itself. One per thread of feature generation; see the ownership discussion in
// audit-reports/impl-lt9-ladder-cache.md.
// ------------------------------------------------------------------------------------------
class LadderCache {
public:
  //maxBytes bounds the store in the resource that actually exhausts (bytes of recorded points),
  //not in an entry count that says nothing about how large each entry is.
  LadderCache(size_t bucketCap, size_t maxBytes);

  //Returns the memoised answer if some stored entry's read set still validates against `board`.
  //There is no way to obtain an answer from this cache without that validation having succeeded.
  std::optional<LadderAnswer> lookup(const Board& board, uint64_t bucketKey);

  //Stores `answer` under `readSet`. The only key this accepts is one a LadderReadScope produced
  //over a real search, so a fabricated key has no expressible form. An empty read set (no scope
  //was armed) is refused.
  void insert(LadderReadSet&& readSet, const LadderAnswer& answer);

  //Verification mode. When armed, the caller is expected to recompute on every hit and call
  //reportVerification with what it got; a disagreement is a loud refusal (ADR-0002 rung 3),
  //because a silently wrong ladder feature is precisely the failure this cache must never have.
  //Off by default: this is the net for the marking-completeness residue, not a normal-path cost.
  static void setVerify(bool on);
  static bool verifyEnabled();
  //Compares a cached answer against a freshly recomputed one and aborts loudly if they differ.
  static void reportVerification(const LadderAnswer& cached, const LadderAnswer& recomputed);

  //Diagnostics for tests and for the measurement build.
  uint64_t numHits() const { return hits_; }
  uint64_t numMisses() const { return misses_; }
#ifdef KATAGO_LT9_CENSUS
  //TEMPORARY -- process-wide totals across every per-thread cache, for the LT-9 census dump.
  //Behind the census's own default-OFF option because it costs a shared atomic on the hit path.
  static void censusTotals(uint64_t& hits, uint64_t& misses);
#endif
  size_t numBuckets() const { return buckets_.size(); }
  void clear();

  //The bucket cap this build ships, chosen at the knee of a measured sweep -- see
  //audit-reports/impl-lt9-ladder-cache.md. Not a round literal.
  static constexpr size_t DEFAULT_BUCKET_CAP = 32;
  //Derived from that same run: 809 distinct buckets x 32 entries x a mean read set of 31 points
  //at 2 bytes per point, rounded up to the next power of two for the allocator's benefit.
  static constexpr size_t DEFAULT_MAX_BYTES = 2u * 1024u * 1024u;

private:
  struct Entry {
    LadderReadSet readSet;
    LadderAnswer answer;
  };

  size_t bucketCap_;
  size_t maxBytes_;
  size_t bytesUsed_ = 0;
  uint64_t hits_ = 0;
  uint64_t misses_ = 0;
  std::unordered_map<uint64_t, std::vector<Entry>> buckets_;
};

#endif // GAME_LADDERCACHE_H
