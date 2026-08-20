/*
 * TEMPORARY INSTRUMENTATION -- LT-9 SOUND ladder-cache key measurement.
 *
 * Sibling of lt9_census.{h,cpp}. The census measured redundancy under a CHAIN-SHAPE key
 * (member + liberty locations), which is NOT a sound cache key: the ladder search plays moves
 * and its move generation reads arbitrary adjacent opposing chains, so its result can depend on
 * board content the chain shape does not mention. The census says so and left it UNEXERCISED.
 * This file measures the hit rate of a key that IS sound.
 *
 * THE KEY. One dispatch's key is (root loc, variant, ko_loc, {(p, colors[p]) : p in R}), where R
 * is the set of board points the search's execution depends on, accumulated during the search
 * under two closure rules read off board.cpp:
 *
 *   RULE-A (a point read)  -- a direct read of colors[p] contributes {p}.
 *   RULE-B (a chain read)  -- any consultation of chain-level state at p (chain_head[p],
 *      chain_data[chain_head[p]].{num_liberties,num_locs,owner}, a next_in_chain walk,
 *      getNumLiberties(p)) contributes chain(p) UNION adj(chain(p)) -- every stone of p's chain
 *      and every point orthogonally adjacent to one of them. That closure is what makes
 *      colors-agreement on R pin both the chain's EXTENT (its boundary points are in R, so it
 *      cannot extend differently) and its LIBERTY SET (its liberties are exactly the empty
 *      points in adj(chain)), which is all chain_data is ever consulted for.
 *
 * Board::lt9MarkChain / Board::lt9MarkMove (board.cpp, same #ifdef) apply those two rules at the
 * consultation sites inside searchIsLadderCaptured and searchIsLadderCapturedAttackerFirst2Libs.
 *
 * WHY A SNAPSHOT OF R IS A COMPLETE KEY. Chain heads enter the algorithm only through equality
 * comparisons among heads, never as addresses with meaning, so two boards agreeing on R induce
 * the same head-equality relation and hence the same behaviour even where the representative Loc
 * differs. Writes need no rule of their own: every point the search writes (colors[move] in
 * playMoveAssumeLegal, removeChain's clears, undo's addChain refills) is read on the same
 * execution before or as it is written, so it is already in R; and the snapshot stored is the
 * PRISTINE board's colors, taken from the board the dispatch started on.
 *
 * WHAT THE MEASUREMENT DOES. A dispatch looks up a search-lifetime map, keyed on
 * (root loc, variant, chain-content hash), for an earlier entry whose stored read-set snapshot
 * VALIDATES against the live board -- exactly what a real cache would have to do. The real search
 * then runs anyway, and its answer is compared against the entry's. That comparison is the
 * witness: if the closure rules above are wrong, a validated entry can carry a different answer
 * and the mismatch counter fires. LT9_SOUNDKEY_MODE=truncated is the red leg -- it records only
 * the census's unsound chain-shape read set, under which the mismatch counter is expected to
 * fire.
 *
 * Entirely inert unless compiled with -DKATAGO_LT9_CENSUS=1 (a CMake option, off by default).
 * Not shippable; see audit-reports/impl-lt9-ladder-cache.md.
 */
#ifndef GAME_LT9_SOUNDKEY_H
#define GAME_LT9_SOUNDKEY_H

#ifdef KATAGO_LT9_CENSUS

#include <cstdint>
#include <string>
#include <vector>

namespace lt9soundkey {

  // ---- Read-set recording (driven from board.cpp's marking sites) ----

  // True while a dispatch is being recorded. Board's marking sites check this first so that
  // ladder searches dispatched from anywhere other than iterLadders record nothing.
  bool recording();

  // RULE-A: contribute one board point to the current dispatch's read set. No-op when the red
  // leg is armed (LT9_SOUNDKEY_MODE=truncated), so the recorded read set collapses to the
  // chain-shape set iterLadders seeds -- i.e. to the census's own unsound key.
  void markPoint(int loc);

  // ---- Per-dispatch bracket (driven from iterLadders in nninputs.cpp) ----

  // Begin recording a dispatch. seedLocs are the chain's member + liberty locations, which
  // iterLadders has already computed for the census key; they seed the read set so the truncated
  // (red-leg) mode has exactly the census's key and nothing else.
  //   variant: 1 = searchIsLadderCaptured(defenderFirst), 2 = ...AttackerFirst2Libs.
  void beginDispatch(
    int rootLoc,
    int variant,
    int koLoc,
    uint64_t chainContentKey,
    uint64_t boardPosHash,
    const std::vector<int>& seedLocs
  );

  // Look up the search-lifetime map for an entry whose stored read-set snapshot validates against
  // the live board. Returns true and fills cachedResult/cachedWorkingMoves on a hit. Called
  // BEFORE the real search runs, exactly where a real cache's lookup would sit.
  //   colorsArr: the live board's colors array (Board::colors), indexed by Loc.
  bool lookup(const int8_t* colorsArr, bool& cachedResult, std::vector<int>& cachedWorkingMoves);

  // End recording. Snapshots the accumulated read set against pristineColors, compares the real
  // answer against any hit reported by lookup(), and stores/refreshes the entry.
  void endDispatch(
    const int8_t* pristineColors,
    bool realResult,
    const std::vector<int>& realWorkingMoves,
    uint64_t expansions
  );

  // Renders this run's sound-key block, appended to the census dump by lt9census::dumpAndReset.
  std::string renderDump();

  // Clears all sound-key state (called from the same dumpAndReset).
  void reset();

} // namespace lt9soundkey

#define LT9_SK_MARK_POINT(loc) ::lt9soundkey::markPoint(loc)
#define LT9_SK_RECORDING() ::lt9soundkey::recording()

#else // !KATAGO_LT9_CENSUS

#define LT9_SK_MARK_POINT(loc) ((void)0)
#define LT9_SK_RECORDING() (false)

#endif // KATAGO_LT9_CENSUS

#endif // GAME_LT9_SOUNDKEY_H
