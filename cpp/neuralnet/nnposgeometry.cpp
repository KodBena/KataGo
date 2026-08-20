#include "../neuralnet/nnposgeometry.h"

#include "../neuralnet/nneval.h"

static void requireInRange(const char* name, int value, int maxValue) {
  if(value < 1 || value > maxValue)
    throw StringError(
      std::string("NNPosGeometry: ") + name + " = " + Global::intToString(value) +
      " but must be between 1 and " + Global::intToString(maxValue) +
      " (the compiled-in bound; rebuild with a larger COMPILE_MAX_BOARD_LEN to go higher)"
    );
}

NNPosGeometry NNPosGeometry::of(const Board& board, int nnXLen, int nnYLen) {
  requireInRange("boardXSize", board.x_size, Board::MAX_LEN);
  requireInRange("boardYSize", board.y_size, Board::MAX_LEN);
  requireInRange("nnXLen", nnXLen, NNPos::MAX_BOARD_LEN);
  requireInRange("nnYLen", nnYLen, NNPos::MAX_BOARD_LEN);

  NNPosGeometry geometry;
  geometry.boardXSize = board.x_size;
  geometry.boardYSize = board.y_size;
  geometry.nnXLen = nnXLen;
  geometry.nnYLen = nnYLen;
  geometry.policySize = NNPos::getPolicySize(nnXLen,nnYLen);

  //The free functions stay the one authoritative definition of the mapping; these tables are
  //that derivation, evaluated once per search instead of once per lookup.
  for(int pos = 0; pos<NNPos::MAX_NN_POLICY_SIZE; pos++) {
    Loc loc =
      pos < geometry.policySize ? NNPos::posToLoc(pos,board.x_size,board.y_size,nnXLen,nnYLen) : Board::NULL_LOC;
    geometry.posToLocTable[pos] = loc;
    //Derived from the Loc just computed, in the same step, so the two tables cannot drift apart.
    geometry.offBoardPosMaskTable[pos] = loc == Board::NULL_LOC ? OFF_BOARD_POS_MASK : 0;
  }
  for(int loc = 0; loc<Board::MAX_ARR_SIZE; loc++)
    geometry.locToPosTable[loc] = NNPos::locToPos((Loc)loc,board.x_size,nnXLen,nnYLen);

  return geometry;
}

NNPosGeometry NNPosGeometry::of(const Board& board, const NNEvaluator& nnEval) {
  return NNPosGeometry::of(board,nnEval.getNNXLen(),nnEval.getNNYLen());
}
