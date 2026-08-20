#include "../game/superkocandidates.h"

#include <cstring>

//The two facts the candidate predicate reads, packed into one byte per point so that one snapshot
//comparison covers both. Colors are 0..3 (C_EMPTY..C_WALL), so bit 2 is free for the flag and the
//value can never be NEVER_OBSERVED.
static inline uint8_t packState(Color color, bool wasEverOccupiedOrPlayed) {
  return (uint8_t)((uint8_t)color | (wasEverOccupiedOrPlayed ? 4u : 0u));
}

//Eight packed points at a time. Reading through memcpy rather than a cast keeps this free of
//aliasing assumptions about Color and bool arrays; every compiler turns it back into one load.
static inline uint64_t loadEightBytes(const void* p) {
  uint64_t v;
  std::memcpy(&v,p,sizeof(uint64_t));
  return v;
}

static inline void setBit(uint64_t* words, int idx, bool value) {
  const uint64_t mask = (uint64_t)1 << (idx & 63);
  if(value)
    words[idx >> 6] |= mask;
  else
    words[idx >> 6] &= ~mask;
}

static inline int indexOfLowestSetBit(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_ctzll(x);
#elif defined(_MSC_VER)
  unsigned long idx;
  _BitScanForward64(&idx,x);
  return (int)idx;
#else
  int n = 0;
  while((x & 1) == 0) { x >>= 1; n += 1; }
  return n;
#endif
}

//Bit loc of the result is bit loc-shift of src, i.e. the neighbor "shift" points earlier in the
//board array. Bits shifted in from before the start of the array are zero, which is correct: those
//indices are off the board and off-board points are never empty.
static inline uint64_t wordShiftedUp(const uint64_t* src, int w, int shift) {
  const int wordShift = shift >> 6;
  const int bitShift = shift & 63;
  const uint64_t lo = (w - wordShift >= 0) ? src[w - wordShift] : (uint64_t)0;
  if(bitShift == 0)
    return lo;
  const uint64_t hi = (w - wordShift - 1 >= 0) ? src[w - wordShift - 1] : (uint64_t)0;
  return (lo << bitShift) | (hi >> (64 - bitShift));
}

//Bit loc of the result is bit loc+shift of src, i.e. the neighbor "shift" points later in the board
//array. Same reasoning about what gets shifted in.
static inline uint64_t wordShiftedDown(const uint64_t* src, int w, int shift, int numWords) {
  const int wordShift = shift >> 6;
  const int bitShift = shift & 63;
  const uint64_t lo = (w + wordShift < numWords) ? src[w + wordShift] : (uint64_t)0;
  if(bitShift == 0)
    return lo;
  const uint64_t hi = (w + wordShift + 1 < numWords) ? src[w + wordShift + 1] : (uint64_t)0;
  return (lo >> bitShift) | (hi << (64 - bitShift));
}

SuperKoCandidates::SuperKoCandidates() {
  //Deliberately NOT a state that agrees with any real board: the first resync must find every point
  //different and rebuild the bitsets, rather than inherit whatever a caller happened to leave.
  std::memset(snapshot,NEVER_OBSERVED,sizeof(snapshot));
  std::memset(isEmptyBits,0,sizeof(isEmptyBits));
  std::memset(wasEverBits,0,sizeof(wasEverBits));
}

int SuperKoCandidates::resync(
  const Board& board,
  const bool (&wasEverOccupiedOrPlayed)[Board::MAX_ARR_SIZE],
  Loc (&outCandidates)[Board::MAX_ARR_SIZE]
) {
  static_assert(sizeof(Color) == 1, "the packed snapshot assumes one byte per color");
  static_assert(sizeof(bool) == 1, "the packed snapshot assumes one byte per wasEverOccupiedOrPlayed");
  static_assert(C_EMPTY == 0 && C_WALL == 3, "the packed snapshot assumes colors fit in two bits");

  //Phase 1: repair the bitsets against reality. Every point is compared, so it does not matter what
  //changed the board or whether this object was told about it.
  const int numWholeWords = Board::MAX_ARR_SIZE / 8;
  for(int wordIdx = 0; wordIdx < numWholeWords; wordIdx++) {
    const int base = wordIdx * 8;
    const uint64_t colorBytes = loadEightBytes(board.colors + base);
    //Each bool byte is 0 or 1, so shifting the whole word left by 2 shifts each byte's bit into that
    //same byte's bit 2 with no carry across byte boundaries.
    const uint64_t wasEverBytes = loadEightBytes(wasEverOccupiedOrPlayed + base) << 2;
    const uint64_t current = colorBytes | wasEverBytes;
    if(current == loadEightBytes(snapshot + base))
      continue;
    std::memcpy(snapshot + base,&current,sizeof(uint64_t));
    //Redo all eight points of a word in which anything changed, rather than picking out which byte
    //differed. Words that change at all are rare, so the seven redundant point updates cost nothing,
    //and not mapping bit positions back to byte positions keeps this free of any assumption about
    //byte order.
    for(int offset = 0; offset < 8; offset++) {
      const int loc = base + offset;
      setBit(isEmptyBits,loc,board.colors[loc] == C_EMPTY);
      setBit(wasEverBits,loc,wasEverOccupiedOrPlayed[loc]);
    }
  }
  for(int loc = numWholeWords * 8; loc < Board::MAX_ARR_SIZE; loc++) {
    const uint8_t current = packState(board.colors[loc],wasEverOccupiedOrPlayed[loc]);
    if(current == snapshot[loc])
      continue;
    snapshot[loc] = current;
    setBit(isEmptyBits,loc,board.colors[loc] == C_EMPTY);
    setBit(wasEverBits,loc,wasEverOccupiedOrPlayed[loc]);
  }

  //Phase 2: derive the candidates. "Has an adjacent empty point" is the union of the empty-point
  //bitset shifted by the four adjacency offsets, which needs no per-point branching and no edge
  //cases - the ring of wall points around the board is never empty, so nothing wraps between rows.
  const int rowStride = board.x_size + 1;
  int numCandidates = 0;
  for(int wordIdx = 0; wordIdx < NUM_WORDS; wordIdx++) {
    const uint64_t hasAdjacentEmpty =
      wordShiftedUp(isEmptyBits,wordIdx,1) |
      wordShiftedDown(isEmptyBits,wordIdx,1,NUM_WORDS) |
      wordShiftedUp(isEmptyBits,wordIdx,rowStride) |
      wordShiftedDown(isEmptyBits,wordIdx,rowStride,NUM_WORDS);
    uint64_t candidates = isEmptyBits[wordIdx] & (wasEverBits[wordIdx] | ~hasAdjacentEmpty);
    while(candidates != 0) {
      outCandidates[numCandidates++] = (Loc)(wordIdx * 64 + indexOfLowestSetBit(candidates));
      candidates &= candidates - 1;
    }
  }
  return numCandidates;
}
