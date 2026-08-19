#include "../tests/tests.h"

#include "../neuralnet/nnposgeometry.h"

using namespace std;
using namespace TestCommon;

//The geometry's tables are a cache of NNPos::posToLoc / NNPos::locToPos. This checks the cache
//against its authority exhaustively - every policy index and every location, for every board
//size the compiled-in bound allows, under two different neural net extents - and checks that a
//geometry that could not answer correctly is refused at construction rather than built.
//The mapping is discrete, so equality is the bar; a tolerance here would be a category error.
//
//These checks need to build geometries from raw nnXLen/nnYLen pairs that no NNEvaluator would
//hand out - rectangles no net is configured for, and the out-of-range values whose refusal is
//the point of the last block. That loose-int factory is private precisely so ordinary code
//cannot reach it; this struct is the single declared exception (NNPosGeometry befriends it),
//and it exists only in this translation unit.
struct NNPosGeometryTesting {
  static NNPosGeometry of(const Board& board, int nnXLen, int nnYLen) {
    return NNPosGeometry::of(board,nnXLen,nnYLen);
  }
};

static void checkAgreesWithNNPos(const Board& board, int nnXLen, int nnYLen) {
  const NNPosGeometry geometry = NNPosGeometryTesting::of(board,nnXLen,nnYLen);

  testAssert(geometry.getBoardXSize() == board.x_size);
  testAssert(geometry.getBoardYSize() == board.y_size);
  testAssert(geometry.getNNXLen() == nnXLen);
  testAssert(geometry.getNNYLen() == nnYLen);
  testAssert(geometry.getPolicySize() == NNPos::getPolicySize(nnXLen,nnYLen));
  testAssert(geometry.getPassPos() == NNPos::getPassPos(nnXLen,nnYLen));
  testAssert(geometry.posToLoc(geometry.getPassPos()) == Board::PASS_LOC);

  for(int pos = 0; pos<geometry.getPolicySize(); pos++)
    testAssert(geometry.posToLoc(pos) == NNPos::posToLoc(pos,board.x_size,board.y_size,nnXLen,nnYLen));
  for(int loc = 0; loc<Board::MAX_ARR_SIZE; loc++)
    testAssert(geometry.locToPos((Loc)loc) == NNPos::locToPos((Loc)loc,board.x_size,nnXLen,nnYLen));
}

static void expectRefused(const char* what, const Board& board, int nnXLen, int nnYLen) {
  bool refused = false;
  try {
    NNPosGeometryTesting::of(board,nnXLen,nnYLen);
  }
  catch(const StringError& e) {
    refused = true;
    cout << what << ": " << e.what() << endl;
  }
  testAssert(refused);
}

void Tests::runNNPosGeometryTests() {
  cout << "Running nn pos geometry tests" << endl;

  for(int boardXSize = 1; boardXSize<=Board::MAX_LEN; boardXSize++) {
    for(int boardYSize = 1; boardYSize<=Board::MAX_LEN; boardYSize++) {
      Board board(boardXSize,boardYSize);
      //Once with the net sized exactly to the board, once with the net at its full extent, so
      //both the square case and the case where the board is a sub-rectangle of the net are covered.
      checkAgreesWithNNPos(board,boardXSize,boardYSize);
      checkAgreesWithNNPos(board,Board::MAX_LEN,Board::MAX_LEN);
    }
  }

  //A geometry is only about the board and the extent it was built for. Asking it whether it is
  //still the right one is how a stale geometry is caught before it mis-indexes anything.
  {
    Board board(19,19);
    Board smallerBoard(9,9);
    Board rectangularBoard(19,9);
    const NNPosGeometry geometry = NNPosGeometryTesting::of(board,19,19);
    testAssert(geometry.matches(board,19,19));
    testAssert(geometry.matchesBoardSize(board));
    testAssert(!geometry.matches(smallerBoard,19,19));
    testAssert(!geometry.matchesBoardSize(smallerBoard));
    testAssert(!geometry.matches(rectangularBoard,19,19));
    testAssert(!geometry.matchesBoardSize(rectangularBoard));
    testAssert(!geometry.matches(board,9,19));
    testAssert(!geometry.matches(board,19,9));
  }

  //Sizes the tables could not cover are refused at construction, loudly, rather than silently
  //mis-indexing later. (A board of an impossible size cannot be constructed at all - Board::init
  //refuses it - so the board sizes reaching here are already validated by the Board type.)
  {
    Board board(19,19);
    expectRefused("nnXLen too large",board,Board::MAX_LEN+1,19);
    expectRefused("nnYLen too large",board,19,Board::MAX_LEN+1);
    expectRefused("nnXLen zero",board,0,19);
    expectRefused("nnYLen negative",board,19,-1);
  }

  cout << "Done" << endl;
}
