/*
 * superkocandidates.h
 * The board points that the per-move superko sweep still has to examine, kept small.
 *
 * WHY THIS EXISTS. After every move, BoardHistory decides for each point whether the next player is
 * forbidden from playing there by positional/situational superko. Deciding that for one point costs
 * an isSuicide call and a Zobrist probe of the ko-hash history, and doing it for all 361 points of a
 * 19x19 board every move made that sweep the largest single cycles site in the engine. Nearly every
 * point is dismissed by one of two cheap tests, both of which read only the state this class tracks,
 * so this class answers "which points are NOT dismissed by the cheap tests" and the sweep runs over
 * those alone - a few tens of points in a typical position rather than all of them.
 *
 * THE PREDICATE. A point can be superko-banned only if it is empty, and only if either a stone was
 * once on it (or a move was once played there) or playing there would be suicide. Board::isSuicide
 * returns false the moment it sees an adjacent empty point, so "no adjacent point is empty" holds
 * whenever suicide does, and
 *
 *     candidate(loc) = empty(loc) AND ( wasEverOccupiedOrPlayed(loc) OR no adjacent point is empty )
 *
 * covers every point the sweep could ban, while calling isSuicide for none of them. It is a
 * SUPERSET on purpose. The sweep still applies its own full test to each candidate, so this class
 * being too generous costs only cycles and can never change a verdict; only being too NARROW could,
 * which is what the rest of this comment, and the differential witness in tests/testsuperkohash.cpp,
 * are about.
 *
 * WHY IT CANNOT GO STALE. The obvious implementation - update the set at each place and capture in
 * BoardHistory::makeBoardMoveAssumeLegal - is not merely fragile here, it is unsound. BoardHistory
 * does not own the Board it sweeps; the Board is a parameter held by the caller, and Board::setStone,
 * a direct call to Board::playMoveAssumeLegal, or handing over a different Board entirely all change
 * the state this predicate reads without BoardHistory seeing anything. No enumeration of
 * BoardHistory's own write sites can be complete against that.
 *
 * So this class does not observe mutations at all. It keeps a byte-per-point snapshot of exactly the
 * state its predicate reads, and every resync compares that whole snapshot against reality - eight
 * points per 64-bit word - and repairs whatever differs. What is compared is the state itself, not a
 * record of who touched it, so every mutation path is covered by construction: the ones that exist,
 * the ones in code this class has never heard of, and the ones someone adds next year without
 * reading this file. The snapshot is initialized to a byte value the packing below cannot produce
 * (NEVER_OBSERVED), so a freshly constructed set differs everywhere on its first resync and rebuilds
 * itself completely rather than trusting anyone to have initialized it.
 *
 * Consequently the set is never read except through resync(): there is no accessor that returns the
 * candidates without first bringing them up to date. Reading a set computed against state other than
 * the state about to be swept is not a mistake this interface can express.
 */

#ifndef GAME_SUPERKOCANDIDATES_H_
#define GAME_SUPERKOCANDIDATES_H_

#include "../game/board.h"

struct SuperKoCandidates {
  //Number of 64-bit words needed to hold one bit per board array index.
  static constexpr int NUM_WORDS = (Board::MAX_ARR_SIZE + 63) / 64;

  SuperKoCandidates();

  //Bring this set up to date against the given board and wasEverOccupiedOrPlayed array, write the
  //candidate locations into outCandidates, and return how many were written.
  //
  //This is the ONLY member that hands out candidates. There is deliberately no accessor that
  //returns them without resyncing first, so no caller can sweep a set that was computed against
  //different state than the state it is sweeping.
  //
  //Both array parameters are taken by reference-to-array of exactly Board::MAX_ARR_SIZE, so passing
  //a buffer that could overflow, or a wasEverOccupiedOrPlayed array of the wrong extent, is a
  //compile error rather than a bounds hazard. Candidates come out in increasing loc order.
  int resync(
    const Board& board,
    const bool (&wasEverOccupiedOrPlayed)[Board::MAX_ARR_SIZE],
    Loc (&outCandidates)[Board::MAX_ARR_SIZE]
  );

 private:
  //A byte value the packing cannot produce: pack() uses only the low three bits (two for the color,
  //one for wasEverOccupiedOrPlayed). Starting the snapshot here makes the first resync differ at
  //every point and therefore rebuild the bitsets from nothing.
  static constexpr uint8_t NEVER_OBSERVED = 0xFF;

  //snapshot[loc] is pack(board.colors[loc], wasEverOccupiedOrPlayed[loc]) as of the last resync.
  uint8_t snapshot[Board::MAX_ARR_SIZE];
  //Bit loc of isEmptyBits is set iff board.colors[loc] == C_EMPTY, as of the last resync.
  uint64_t isEmptyBits[NUM_WORDS];
  //Bit loc of wasEverBits is set iff wasEverOccupiedOrPlayed[loc], as of the last resync.
  uint64_t wasEverBits[NUM_WORDS];
};

#endif  // GAME_SUPERKOCANDIDATES_H_
