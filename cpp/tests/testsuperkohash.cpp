#include "../tests/tests.h"

#include "../program/playutils.h"

using namespace std;
using namespace TestCommon;

//Differential witness for BoardHistory's incrementally-maintained superKoBannedHash.
//
//getSituationRulesAndKoHash used to derive the ko-mark contribution of its hash by sweeping every
//point of the board on every call. It now consumes two hashes that BoardHistory maintains at the
//write sites of the mark arrays instead. The two tests below observe that change directly:
//
//  W1 - the maintained hash equals a fold recomputed over the whole array. This fails if any write
//       to superKoBanned bypasses setSuperKoBanned/clearSuperKoBanned, i.e. if a write site was
//       missed. Observed on the history object itself, which is where the claim lives.
//
//  W2 - the SHIPPED getSituationRulesAndKoHash returns bit-identical results to a transcription of
//       the swept version it replaces. The oracle is a full copy of the old function, so what is
//       observed is the function callers actually reach, not a restatement of its intended formula.
//       (An earlier form of this test compared two test-local expressions instead; deleting the
//       shipped function's O(1) ko_loc correction left it green, which is how that was found.)
//
//Both are checked at every position of a randomized corpus that is deliberately weighted toward the
//states where the two could differ: small boards so positions repeat, positional and situational
//superko so bans actually arise, territory scoring so the encore is entered, and ko fights so ko
//recapture blocks and simple ko locs are live. The corpus counts how often it reached each of those
//states and asserts floors on the counts, because a corpus of quiet openings would pass both checks
//while witnessing nothing. The floors sit at 63-65% of the counts this fixed seed produces, i.e.
//tight enough that any single category dropping by more than about half again - a 2x regression
//included - fails here rather than silently reducing this test to a tautology. The remaining third
//is headroom for the counts to shift under an unrelated legitimate change to the rules code the
//corpus plays against; it is not slack for coverage to decay into.

namespace {

  struct SuperKoHashCorpusStats {
    int64_t numPositionsChecked = 0;
    int64_t numWithAnySuperKoBan = 0;
    int64_t numWithKoLocAlsoBanned = 0;
    int64_t numInEncore1 = 0;
    int64_t numInEncore2 = 0;
    int64_t numWithKoRecapBlock = 0;
    int64_t numWithSimpleKoLoc = 0;
    int64_t numKoLocBannedPairsChecked = 0;
  };

  //The fold over the whole array, recomputed from scratch. This is what the maintained hash claims
  //to be equal to at all times.
  Hash128 recomputeSuperKoBannedHash(const BoardHistory& hist) {
    Hash128 h;
    for(int i = 0; i<Board::MAX_ARR_SIZE; i++) {
      if(hist.isSuperKoBanned((Loc)i))
        h ^= Board::ZOBRIST_KO_LOC_HASH[i];
    }
    return h;
  }

  //A full transcription of getSituationRulesAndKoHash as it stood BEFORE this change: the ko-mark
  //contribution derived by sweeping the board, everything else identical. This is the oracle the
  //shipped function is compared against, so the witness observes the shipped function itself rather
  //than a re-statement of what it is supposed to do.
  Hash128 referenceSituationRulesAndKoHashBySweep(
    const Board& board, const BoardHistory& hist, Player nextPlayer, double drawEquivalentWinsForWhite,
    const BoardHistoryModes& modes
  ) {
    int xSize = board.x_size;
    int ySize = board.y_size;

    Hash128 hash = board.pos_hash;
    hash ^= Board::ZOBRIST_PLAYER_HASH[nextPlayer];

    testAssert(hist.encorePhase >= 0 && hist.encorePhase <= 2);
    hash ^= Board::ZOBRIST_ENCORE_HASH[hist.encorePhase];

    if(hist.encorePhase == 0) {
      if(board.ko_loc != Board::NULL_LOC)
        hash ^= Board::ZOBRIST_KO_LOC_HASH[board.ko_loc];
      for(int y = 0; y<ySize; y++) {
        for(int x = 0; x<xSize; x++) {
          Loc loc = Location::getLoc(x,y,xSize);
          if(hist.isSuperKoBanned(loc) && loc != board.ko_loc)
            hash ^= Board::ZOBRIST_KO_LOC_HASH[loc];
        }
      }
    }
    else {
      for(int y = 0; y<ySize; y++) {
        for(int x = 0; x<xSize; x++) {
          Loc loc = Location::getLoc(x,y,xSize);
          if(hist.isSuperKoBanned(loc))
            hash ^= Board::ZOBRIST_KO_LOC_HASH[loc];
          if(hist.koRecapBlocked[loc])
            hash ^= Board::ZOBRIST_KO_MARK_HASH[loc][P_BLACK] ^ Board::ZOBRIST_KO_MARK_HASH[loc][P_WHITE];
        }
      }
      if(hist.encorePhase == 2) {
        for(int y = 0; y<ySize; y++) {
          for(int x = 0; x<xSize; x++) {
            Loc loc = Location::getLoc(x,y,xSize);
            Color c = hist.secondEncoreStartColors[loc];
            if(c != C_EMPTY)
              hash ^= Board::ZOBRIST_SECOND_ENCORE_START_HASH[loc][c];
          }
        }
      }
    }

    float selfKomi = hist.currentSelfKomi(nextPlayer,drawEquivalentWinsForWhite);
    int64_t komiDiscretized = (int64_t)(selfKomi*256.0f);
    uint64_t komiHash = Hash::murmurMix((uint64_t)komiDiscretized);
    hash.hash0 ^= komiHash;
    hash.hash1 ^= Hash::basicLCong(komiHash);

    hash ^= Rules::ZOBRIST_KO_RULE_HASH[hist.rules.koRule];
    hash ^= Rules::ZOBRIST_SCORING_RULE_HASH[hist.rules.scoringRule];
    hash ^= Rules::ZOBRIST_TAX_RULE_HASH[hist.rules.taxRule];
    if(hist.rules.multiStoneSuicideLegal)
      hash ^= Rules::ZOBRIST_MULTI_STONE_SUICIDE_HASH;
    if(hist.hasButton)
      hash ^= Rules::ZOBRIST_BUTTON_HASH;
    if(hist.rules.friendlyPassOk)
      hash ^= Rules::ZOBRIST_FRIENDLY_PASS_OK_HASH;

    if(modes.alwaysComputePassAliveUnderSuicideRules && !hist.rules.multiStoneSuicideLegal)
      hash ^= Rules::ZOBRIST_PASS_ALIVE_UNDER_SUICIDE_HASH;
    if(modes.excludeTerritoryAdjacentToAtari && hist.rules.scoringRule == Rules::SCORING_TERRITORY && hist.rules.taxRule == Rules::TAX_NONE)
      hash ^= Rules::ZOBRIST_EXCLUDE_TERRITORY_ADJ_ATARI_HASH;

    return hash;
  }

  void checkPosition(
    const Board& board, const BoardHistory& hist, Player nextPlayer, double drawEquivalentWinsForWhite,
    SuperKoHashCorpusStats& stats
  ) {
    //W1: the maintained hash is the fold over the array it claims to summarize. Observed on the
    //history object, which is where a missed write site would show up.
    testAssert(hist.getSuperKoBannedHash() == recomputeSuperKoBannedHash(hist));
    //W2: the shipped hash function agrees bit for bit with the swept version it replaces.
    testAssert(
      BoardHistory::getSituationRulesAndKoHash(board,hist,nextPlayer,drawEquivalentWinsForWhite,hist.modes) ==
      referenceSituationRulesAndKoHashBySweep(board,hist,nextPlayer,drawEquivalentWinsForWhite,hist.modes)
    );

    stats.numPositionsChecked += 1;
    if(hist.getSuperKoBannedHash() != Hash128())
      stats.numWithAnySuperKoBan += 1;
    if(board.ko_loc != Board::NULL_LOC) {
      stats.numWithSimpleKoLoc += 1;
      if(hist.isSuperKoBanned(board.ko_loc))
        stats.numWithKoLocAlsoBanned += 1;
    }
    if(hist.encorePhase == 1)
      stats.numInEncore1 += 1;
    if(hist.encorePhase == 2)
      stats.numInEncore2 += 1;
    if(hist.koRecapBlockHash != Hash128())
      stats.numWithKoRecapBlock += 1;
  }

  //Play out one randomized game, checking both witnesses before the first move and after every move.
  void playAndCheckOneGame(Rand& rand, SuperKoHashCorpusStats& stats) {
    //Small boards so that positions repeat often enough for superko to actually trigger.
    static const int SIZES[10][2] = {{2,2},{3,2},{3,3},{4,3},{4,4},{5,4},{5,5},{6,6},{7,7},{9,5}};
    int sizeIdx = rand.nextInt(0,9);
    int xSize = SIZES[sizeIdx][0];
    int ySize = SIZES[sizeIdx][1];

    Rules rules;
    rules.koRule = rand.nextInt(0,3) == 0 ? Rules::KO_SIMPLE : rand.nextBool(0.5) ? Rules::KO_POSITIONAL : Rules::KO_SITUATIONAL;
    rules.scoringRule = rand.nextBool(0.5) ? Rules::SCORING_AREA : Rules::SCORING_TERRITORY;
    rules.taxRule = rand.nextBool(0.5) ? Rules::TAX_NONE : rand.nextBool(0.5) ? Rules::TAX_SEKI : Rules::TAX_ALL;
    rules.multiStoneSuicideLegal = rand.nextBool(0.5);
    rules.hasButton = rules.scoringRule == Rules::SCORING_AREA && rand.nextBool(0.25);
    rules.komi = 7.0f - (float)rand.nextInt(0,14) * 0.5f;

    int initialEncorePhase = 0;
    if(rules.scoringRule == Rules::SCORING_TERRITORY)
      initialEncorePhase = rand.nextBool(0.5) ? 0 : rand.nextBool(0.5) ? 1 : 2;

    BoardHistoryModes modes(rand.nextBool(0.5),rand.nextBool(0.5));

    Board board(xSize,ySize);
    Player pla = rand.nextBool(0.5) ? P_BLACK : P_WHITE;
    BoardHistory hist(board,pla,rules,initialEncorePhase,modes);

    double drawEquivalentWinsForWhite = 0.5;
    double passProb = rand.nextDouble(0.005,0.08);
    int numSteps = rand.nextInt(60,260);
    bool steerTowardBans = rand.nextBool(0.5);

    checkPosition(board,hist,pla,drawEquivalentWinsForWhite,stats);

    for(int i = 0; i<numSteps; i++) {
      if(hist.isGameFinished)
        break;
      Loc moveLoc;
      if(rand.nextBool(passProb))
        moveLoc = Board::PASS_LOC;
      else {
        moveLoc = PlayUtils::chooseRandomLegalMove(board,hist,pla,rand,Board::NULL_LOC);
        //Resample toward captures, which is what produces kos, ko recapture blocks and repeats.
        for(int resample = 0; resample<2; resample++) {
          if(moveLoc != Board::NULL_LOC && !board.wouldBeCapture(moveLoc,pla))
            moveLoc = PlayUtils::chooseRandomLegalMove(board,hist,pla,rand,Board::NULL_LOC);
        }
        if(moveLoc == Board::NULL_LOC)
          moveLoc = Board::PASS_LOC;
      }
      bool preventEncore = rand.nextBool(0.1);
      //Positions that actually carry superko bans are rare under purely random play, and they are
      //exactly the states this witness exists to cover. In some games, look one move ahead over a
      //few candidates and take whichever leaves the most locations banned.
      if(steerTowardBans && moveLoc != Board::PASS_LOC && moveLoc != Board::NULL_LOC) {
        int bestNumBanned = -1;
        Loc bestLoc = moveLoc;
        for(int cand = 0; cand<4; cand++) {
          Loc candLoc = cand == 0 ? moveLoc : PlayUtils::chooseRandomLegalMove(board,hist,pla,rand,Board::NULL_LOC);
          if(candLoc == Board::NULL_LOC || !hist.isLegal(board,candLoc,pla))
            continue;
          Board candBoard = board;
          BoardHistory candHist = hist;
          candHist.makeBoardMoveAssumeLegal(candBoard,candLoc,pla,NULL,preventEncore);
          int numBanned = 0;
          for(int j = 0; j<Board::MAX_ARR_SIZE; j++) {
            if(candHist.isSuperKoBanned((Loc)j))
              numBanned += 1;
          }
          if(numBanned > bestNumBanned) {
            bestNumBanned = numBanned;
            bestLoc = candLoc;
          }
        }
        moveLoc = bestLoc;
      }
      if(!hist.isLegal(board,moveLoc,pla))
        continue;
      hist.makeBoardMoveAssumeLegal(board,moveLoc,pla,NULL,preventEncore);
      pla = getOpp(pla);

      checkPosition(board,hist,pla,drawEquivalentWinsForWhite,stats);
      //Both players' perspectives, since the hash folds in the player to move and self komi.
      checkPosition(board,hist,getOpp(pla),drawEquivalentWinsForWhite,stats);

      //A copy must carry the maintained hash, and must still satisfy the invariant on its own.
      if(rand.nextBool(0.05)) {
        BoardHistory histCopy = hist;
        testAssert(histCopy.getSuperKoBannedHash() == hist.getSuperKoBannedHash());
        checkPosition(board,histCopy,pla,drawEquivalentWinsForWhite,stats);
      }
      //As must a history rebuilt by replaying the same moves from the initial position.
      if(rand.nextBool(0.02)) {
        BoardHistory replayed = hist.copyToInitial();
        Board replayBoard = replayed.getRecentBoard(0);
        for(size_t j = 0; j<hist.moveHistory.size(); j++)
          replayed.makeBoardMoveAssumeLegal(replayBoard,hist.moveHistory[j].loc,hist.moveHistory[j].pla,NULL,hist.preventEncoreHistory[j]);
        testAssert(replayed.getSuperKoBannedHash() == hist.getSuperKoBannedHash());
        checkPosition(replayBoard,replayed,pla,drawEquivalentWinsForWhite,stats);
      }
      //And clearing must zero the hash along with the array.
      if(rand.nextBool(0.01)) {
        BoardHistory cleared = hist;
        cleared.clear(board,pla,rules,initialEncorePhase);
        testAssert(cleared.getSuperKoBannedHash() == Hash128());
        checkPosition(board,cleared,pla,drawEquivalentWinsForWhite,stats);
      }

      //The non-encore branch corrects for the case where the simple ko loc is itself superko-banned.
      //A history's own recompute always clears the ban at its own board's ko loc, so that case cannot
      //arise from play - it only arises because the function is static over an arbitrary (board,hist)
      //pair and a caller may pass a board the marks were not computed against. Construct exactly that
      //pair here, so the correction is observed rather than assumed dead.
      if(hist.encorePhase == 0) {
        for(int y = 0; y<ySize; y++) {
          for(int x = 0; x<xSize; x++) {
            Loc loc = Location::getLoc(x,y,xSize);
            if(!hist.isSuperKoBanned(loc))
              continue;
            Board boardWithKoAtBannedLoc = board;
            boardWithKoAtBannedLoc.ko_loc = loc;
            testAssert(
              BoardHistory::getSituationRulesAndKoHash(boardWithKoAtBannedLoc,hist,pla,drawEquivalentWinsForWhite,hist.modes) ==
              referenceSituationRulesAndKoHashBySweep(boardWithKoAtBannedLoc,hist,pla,drawEquivalentWinsForWhite,hist.modes)
            );
            stats.numKoLocBannedPairsChecked += 1;
          }
        }
      }

    }
  }
}

void Tests::runSuperKoBannedHashTests() {
  cout << "Running superko banned hash differential tests" << endl;
  Rand rand("runSuperKoBannedHashTests");
  SuperKoHashCorpusStats stats;

  for(int game = 0; game<3000; game++)
    playAndCheckOneGame(rand,stats);

  //Each count is printed WITH the floor it must clear, so the tightness of the guard is observed
  //rather than described. A prose claim about how tight a floor is can drift from the floor; a
  //printed pair cannot.
  auto report = [](const char* what, int64_t count, int64_t floor) {
    cout << what << " " << count << " (floor " << floor << ")" << endl;
    testAssert(count > floor);
  };

  cout << "Coverage of the differential corpus:" << endl;
  report("  positions checked                ", stats.numPositionsChecked, 200000);
  report("  with a superko ban               ", stats.numWithAnySuperKoBan, 800);
  report("  with a simple ko loc             ", stats.numWithSimpleKoLoc, 1300);
  report("  in encore phase 1                ", stats.numInEncore1, 41000);
  report("  in encore phase 2                ", stats.numInEncore2, 47000);
  report("  with a ko recapture block        ", stats.numWithKoRecapBlock, 5900);
  report("  constructed ko-loc-is-banned pairs", stats.numKoLocBannedPairsChecked, 360);
  //Not floored, and printed without one on purpose: this count is 0 by construction, because a
  //history's own recompute always clears the ban at its own board's ko loc. The constructed pairs
  //on the line above are what cover that branch instead.
  cout << "  with the ko loc itself banned    " << " " << stats.numWithKoLocAlsoBanned
       << " (0 by construction, not floored)" << endl;
}
