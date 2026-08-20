/*
 * nnposgeometry.h
 * The board <-> policy-vector coordinate mapping of one search, owned as a single value.
 */

#ifndef NEURALNET_NNPOSGEOMETRY_H_
#define NEURALNET_NNPOSGEOMETRY_H_

#include <cassert>
#include <cstdint>

#include "../game/board.h"
#include "../neuralnet/nninputs.h"

class NNEvaluator;
//Defined only in the geometry's own test translation unit; see the friend declaration below.
struct NNPosGeometryTesting;

//The mapping between a policy-vector index ("pos") and a board location ("Loc") is a pure
//function of four numbers - the board size and the neural net's spatial extent - and all four
//are constant for the whole of a search. Passing them around as four loose ints has two costs:
//a caller can transpose them silently, and the compiler sees a runtime divisor at every use, so
//NNPos::posToLoc pays a hardware integer division (and NNPos::locToPos another, inside
//Location::getY) on every call.
//
//NNPosGeometry gives those four numbers one home. A call site names one thing instead of four,
//and the mapping is precomputed both ways at construction, so a lookup replaces the division.
//
//The tables are filled by calling NNPos::posToLoc / NNPos::locToPos, which remain the single
//authoritative definition of the mapping: this type caches that derivation on the value that
//owns the geometry, it is not a second implementation of it. Nothing here is specialized to any
//particular board size - the table bounds come from the compiled-in Board::MAX_LEN and the
//entries are filled from the actual runtime sizes.
struct NNPosGeometry {
  //Build the geometry for a board searched with the evaluator whose extent defines it. Every
  //number arrives from a typed source, so there is no argument order for a caller to get wrong.
  //Throws StringError if a size is outside the range the compiled-in bounds can represent -
  //a geometry whose tables would not cover the board it is asked about is not constructible.
  //
  //This is the only way to build one from outside. The underlying loose-int form is private
  //below: it takes nnXLen and nnYLen as two same-typed positional ints, which is exactly the
  //transposable shape this type exists to remove, so it is not offered as public API.
  static NNPosGeometry of(const Board& board, const NNEvaluator& nnEval);

  inline int getBoardXSize() const { return boardXSize; }
  inline int getBoardYSize() const { return boardYSize; }
  inline int getNNXLen() const { return nnXLen; }
  inline int getNNYLen() const { return nnYLen; }
  //Length of the policy vector, i.e. one entry per nn-extent point plus one for the pass move.
  inline int getPolicySize() const { return policySize; }
  inline int getPassPos() const { return policySize - 1; }

  //The policy index for a location: NNPos::locToPos(loc,boardXSize,nnXLen,nnYLen), precomputed.
  inline int locToPos(Loc loc) const {
    assert(loc >= 0 && loc < Board::MAX_ARR_SIZE);
    return locToPosTable[loc];
  }
  //The location for a policy index: NNPos::posToLoc(pos,boardXSize,boardYSize,nnXLen,nnYLen),
  //precomputed. Board::PASS_LOC for the pass index, Board::NULL_LOC for an index off the board.
  inline Loc posToLoc(int pos) const {
    assert(pos >= 0 && pos < policySize);
    return posToLocTable[pos];
  }

  //A word-per-position restatement of "posToLoc(pos) == Board::NULL_LOC": OFF_BOARD_POS_MASK, which is
  //-1, i.e. a word with every bit set, for a policy index that is not a board point (off the board
  //within the neural net's extent, or past the end of the policy vector), and 0 for one that is. The
  //same fact as posToLocTable, derived from it at construction rather than defined a second time, in
  //the shape a whole-policy-vector scan can consume as arithmetic instead of a per-element branch - see
  //the new-child scan in Search::selectBestChildToDescend, the one caller. The whole table is handed
  //out rather than one entry because that scan wants every entry.
  static constexpr int32_t OFF_BOARD_POS_MASK = -1;
  inline const int32_t* getOffBoardPosMaskTable() const {
    return offBoardPosMaskTable;
  }

  //Whether this geometry is the one for this board and this extent - i.e. whether the cached
  //tables still describe the live geometry. False means some writer changed the geometry
  //without rebuilding, and every lookup after that point would be answering about a board that
  //is no longer there.
  inline bool matches(const Board& board, int nnXLen_, int nnYLen_) const {
    return board.x_size == boardXSize && board.y_size == boardYSize && nnXLen_ == nnXLen && nnYLen_ == nnYLen;
  }
  inline bool matchesBoardSize(const Board& board) const {
    return board.x_size == boardXSize && board.y_size == boardYSize;
  }

private:
  //The loose-int form. Reachable only from the evaluator overload above (which fixes the
  //argument order once, here, in one hand-checked expression) and from the geometry's own
  //tests, which must sweep nnXLen/nnYLen pairs no evaluator would produce - including the
  //out-of-range ones whose refusal is under test.
  static NNPosGeometry of(const Board& board, int nnXLen, int nnYLen);
  friend struct NNPosGeometryTesting;

  NNPosGeometry() = default;

  int boardXSize;
  int boardYSize;
  int nnXLen;
  int nnYLen;
  int policySize;

  Loc posToLocTable[NNPos::MAX_NN_POLICY_SIZE];
  int32_t offBoardPosMaskTable[NNPos::MAX_NN_POLICY_SIZE];
  int locToPosTable[Board::MAX_ARR_SIZE];
};

#endif  // NEURALNET_NNPOSGEOMETRY_H_
