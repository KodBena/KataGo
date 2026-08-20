#include "../tests/tests.h"

#include "../program/playutils.h"

using namespace std;
using namespace TestCommon;

//Differential witness for BoardHistory's superko marks: the incrementally-maintained
//superKoBannedHash, and the marks themselves now that they are computed over a candidate set rather
//than over the whole board.
//
//getSituationRulesAndKoHash used to derive the ko-mark contribution of its hash by sweeping every
//point of the board on every call. It now consumes two hashes that BoardHistory maintains at the
//write sites of the mark arrays instead. Separately, makeBoardMoveAssumeLegal used to decide the
//marks themselves by testing every point of the board after every move; it now tests only the
//points SuperKoCandidates hands it. The tests below observe both changes directly:
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
//  W3 - the SHIPPED superKoBanned array is bit-identical to what the whole-board sweep would put
//       there. That sweep no longer runs over the whole board: it runs over a candidate set that
//       SuperKoCandidates resyncs against the board (see game/superkocandidates.h), and the failure
//       mode of a carried-over set is staleness - a point that should be examined is not, and a move
//       that superko forbids is silently permitted. Staleness produces no crash and no wrong hash;
//       W1 and W2 would both stay green through it, because a stale set still writes a
//       self-consistent array and hash. Only comparing against a full sweep observes it. The oracle
//       is a transcription of the sweep as it stood before the change, so what is observed is the
//       shipped marks against the old whole-board answer, not against a restatement of the new one.
//
//  W4 - W3 still holds after the Board has been changed behind the history's back. BoardHistory does
//       not own the Board it sweeps - it is a caller-held parameter - so a candidate set maintained
//       at BoardHistory's own place/capture sites would be correct through ordinary play and wrong
//       here. Two flavors are constructed: scrambling stones onto and off the board with
//       Board::setStonesTolerant, and substituting an entirely different (earlier) Board, in both
//       cases then making a move through the history and checking W3 on the result. This is the case
//       the pre-change code could not fail and the post-change code can. It is insurance against a
//       future rewrite toward the incrementally-maintained design, not against the shipped one: no
//       defect has been found that W4 catches and W3 does not, because the shipped implementation
//       carries nothing that ordinary play would not already expose. Stated plainly here, because
//       "there is a test for that" is exactly the claim that stops such a rewrite being scrutinized.
//
//  W6 - W3 still holds after a warm history is cleared onto a board of a DIFFERENT SIZE. Both pieces
//       of carried state are size-dependent: the snapshot is indexed by a board-array index whose
//       meaning depends on the row stride, and the adjacency shifts are BY the stride. The shipped
//       design carries no stride and re-compares the snapshot in full, so it needs nothing special
//       here - but that is a derivation, and this is what observes it.
//
//W1, W2 and W3 are checked at every position of a randomized corpus that is deliberately weighted
//toward the states in which the checked things could differ: small boards so positions repeat,
//positional and situational superko so bans actually arise, territory scoring so the encore is
//entered, and ko fights so ko recapture blocks and simple ko locs are live. W4 is constructed on top
//of that corpus at intervals. The corpus counts how often it reached each of those states and
//asserts floors on the counts, because a corpus of quiet openings would pass every check above
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
    int64_t numFullSweepComparisons = 0;
    int64_t numWithNeverOccupiedSuicidePoint = 0;
    int64_t numWithNeverOccupiedCandidateBanned = 0;
    int64_t numOutOfBandScrambleChecks = 0;
    int64_t numOutOfBandSubstitutedBoardChecks = 0;
    int64_t numResizedOntoDifferentBoardChecks = 0;
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

  //Transcriptions of the two BoardHistory internals the pre-change sweep called. koHashOccursInHistory
  //is transcribed for the rootKoHashTable == NULL case, which is the case this corpus produces: every
  //makeBoardMoveAssumeLegal below passes NULL, so the shipped function reduces to exactly this scan.
  Hash128 referenceKoHashAfterMoveNonEncore(const Rules& rules, Hash128 posHashAfterMove, Player pla) {
    if(rules.koRule == Rules::KO_SITUATIONAL || rules.koRule == Rules::KO_SIMPLE)
      return posHashAfterMove ^ Board::ZOBRIST_PLAYER_HASH[pla];
    else
      return posHashAfterMove;
  }
  bool referenceKoHashOccursInHistory(const BoardHistory& hist, Hash128 koHash) {
    for(size_t i = 0; i<hist.koHashHistory.size(); i++)
      if(hist.koHashHistory[i] == koHash)
        return true;
    return false;
  }

  //A full transcription of the superko marking sweep as it stood BEFORE this change: every point of
  //the board tested in turn, no candidate set anywhere. This is the oracle for W3. It answers for
  //hist.presumedNextMovePla, because that is the player the shipped marks were computed for.
  void referenceSuperKoBannedByFullSweep(const Board& board, const BoardHistory& hist, bool out[Board::MAX_ARR_SIZE]) {
    for(int i = 0; i<Board::MAX_ARR_SIZE; i++)
      out[i] = false;
    Player nextPla = hist.presumedNextMovePla;
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        if(board.colors[loc] != C_EMPTY)
          out[loc] = false;
        else if(!hist.wasEverOccupiedOrPlayed[loc] && !board.isSuicide(loc,nextPla))
          out[loc] = false;
        else if(board.isIllegalSuicide(loc,nextPla,hist.rules.multiStoneSuicideLegal) || loc == board.ko_loc)
          out[loc] = false;
        else {
          Hash128 posHashAfterMove = board.getPosHashAfterMove(loc,nextPla);
          Hash128 koHashAfterMove = referenceKoHashAfterMoveNonEncore(hist.rules,posHashAfterMove,getOpp(nextPla));
          out[loc] = referenceKoHashOccursInHistory(hist,koHashAfterMove);
        }
      }
    }
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

  //Returns whether the W3 full-sweep comparison actually ran at this position. Callers that count
  //constructed cases use that, so their counts measure comparisons performed rather than cases
  //attempted - a count of attempts would clear its floor while the thing it stands for never ran.
  bool checkPosition(
    const Board& board, const BoardHistory& hist, Player nextPlayer, double drawEquivalentWinsForWhite,
    SuperKoHashCorpusStats& stats
  ) {
    bool fullSweepComparisonRan = false;
    //W1: the maintained hash is the fold over the array it claims to summarize. Observed on the
    //history object, which is where a missed write site would show up.
    testAssert(hist.getSuperKoBannedHash() == recomputeSuperKoBannedHash(hist));
    //W2: the shipped hash function agrees bit for bit with the swept version it replaces.
    testAssert(
      BoardHistory::getSituationRulesAndKoHash(board,hist,nextPlayer,drawEquivalentWinsForWhite,hist.modes) ==
      referenceSituationRulesAndKoHashBySweep(board,hist,nextPlayer,drawEquivalentWinsForWhite,hist.modes)
    );

    //W3: the shipped marks are what the whole-board sweep would have produced. Restricted to the
    //state in which the sweep is what put them there: in the encore a different rule writes them, and
    //at a position with no moves played yet nothing has written them at all. Under simple ko in the
    //main phase nothing writes them either, and that they are consequently all clear is checked
    //rather than assumed.
    if(hist.encorePhase == 0 && hist.moveHistory.size() > 0) {
      if(hist.rules.koRule != Rules::KO_SIMPLE) {
        bool expected[Board::MAX_ARR_SIZE];
        referenceSuperKoBannedByFullSweep(board,hist,expected);
        for(int i = 0; i<Board::MAX_ARR_SIZE; i++)
          testAssert(hist.isSuperKoBanned((Loc)i) == expected[i]);
        stats.numFullSweepComparisons += 1;
        fullSweepComparisonRan = true;

        //Coverage of the one point class for which "empty and a stone was once here" - the obvious
        //and WRONG candidate predicate - is too narrow: a point that never held a stone, where
        //playing would nonetheless be suicide, so the move can still repeat a position. Two counts,
        //because they say different things and both are floored. The first is how often such a point
        //exists at all, i.e. how often the non-obvious half of the predicate is exercised. The second
        //is how often one of them is ACTUALLY superko-banned - each of those positions is one where
        //the narrow predicate would have permitted a move superko forbids, so a nonzero floor on it
        //is what keeps this corpus able to reject that predicate rather than merely visit it.
        bool anyNeverOccupiedSuicide = false;
        bool anyNeverOccupiedBanned = false;
        for(int y = 0; y<board.y_size; y++) {
          for(int x = 0; x<board.x_size; x++) {
            Loc loc = Location::getLoc(x,y,board.x_size);
            if(board.colors[loc] != C_EMPTY || hist.wasEverOccupiedOrPlayed[loc])
              continue;
            if(board.isSuicide(loc,hist.presumedNextMovePla))
              anyNeverOccupiedSuicide = true;
            if(hist.isSuperKoBanned(loc))
              anyNeverOccupiedBanned = true;
          }
        }
        if(anyNeverOccupiedSuicide)
          stats.numWithNeverOccupiedSuicidePoint += 1;
        if(anyNeverOccupiedBanned)
          stats.numWithNeverOccupiedCandidateBanned += 1;
      }
      else {
        for(int i = 0; i<Board::MAX_ARR_SIZE; i++)
          testAssert(!hist.isSuperKoBanned((Loc)i));
      }
    }

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
    return fullSweepComparisonRan;
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

      //W4: change the Board behind the history's back, then make a move through the history and
      //check W3 on the result. The history is handed a Board by its caller and has no hook into it,
      //so a candidate set maintained at the history's own place/capture sites would be stale exactly
      //here - and stale in the silent direction, permitting a move superko forbids.
      if(rand.nextBool(0.05)) {
        bool substituteAnEarlierBoard = rand.nextBool(0.5);
        Board oobBoard = board;
        BoardHistory oobHist = hist;
        if(substituteAnEarlierBoard) {
          //Not a mutation of the board at all: a different Board object entirely, one the history's
          //marks were never computed against.
          oobBoard = hist.getRecentBoard(rand.nextInt(1,3));
        }
        else {
          //Stones appearing and disappearing at points the history was never told about, including
          //points it has never seen a stone on.
          vector<Move> placements;
          int numPlacements = rand.nextInt(1,4);
          for(int p = 0; p<numPlacements; p++) {
            Loc loc = Location::getLoc(rand.nextInt(0,xSize-1),rand.nextInt(0,ySize-1),xSize);
            int roll = rand.nextInt(0,2);
            placements.push_back(Move(loc,roll == 0 ? C_EMPTY : roll == 1 ? P_BLACK : P_WHITE));
          }
          oobBoard.setStonesTolerant(placements);
        }
        //Any move that the BOARD accepts - the history's own legality is not the point here, and its
        //superko marks are stale by construction at this instant.
        Loc oobLoc = Board::PASS_LOC;
        int startIdx = rand.nextInt(0,xSize*ySize-1);
        for(int k = 0; k<xSize*ySize; k++) {
          int idx = (startIdx + k) % (xSize*ySize);
          Loc loc = Location::getLoc(idx % xSize,idx / xSize,xSize);
          if(oobBoard.isLegal(loc,pla,rules.multiStoneSuicideLegal)) {
            oobLoc = loc;
            break;
          }
        }
        oobHist.makeBoardMoveAssumeLegal(oobBoard,oobLoc,pla,NULL,true);
        //Counted only if the resulting position was one where W3 actually compared against the full
        //sweep. Roughly half of this corpus is in an encore, where the marks come from a different
        //rule and W3 does not apply, so counting the construction rather than the comparison would
        //put a floor on cases that never checked anything.
        if(checkPosition(oobBoard,oobHist,getOpp(pla),drawEquivalentWinsForWhite,stats)) {
          if(substituteAnEarlierBoard)
            stats.numOutOfBandSubstitutedBoardChecks += 1;
          else
            stats.numOutOfBandScrambleChecks += 1;
        }
      }

      //W6: clear a warm history onto a board of a DIFFERENT SIZE and play on. Both pieces of state
      //the candidate set carries across moves are size-dependent in different ways - the snapshot is
      //indexed by a board-array index whose meaning depends on the stride, and the adjacency is
      //computed from the stride itself - so a resize is the case where carrying either one would
      //show. Nothing in the shipped design carries the stride and the snapshot is re-compared in
      //full, but that is a derivation, and this is the corpus position that observes it.
      if(rand.nextBool(0.03)) {
        int resizedIdx = rand.nextInt(0,9);
        int resizedX = SIZES[resizedIdx][0];
        int resizedY = SIZES[resizedIdx][1];
        if(resizedX != xSize || resizedY != ySize) {
          //Deliberately NOT a fresh history: this one has already swept many positions on the old
          //size, so its snapshot and bitsets are warm and describe a board that no longer exists.
          BoardHistory resizedHist = hist;
          Board resizedBoard(resizedX,resizedY);
          Player resizedPla = rand.nextBool(0.5) ? P_BLACK : P_WHITE;
          resizedHist.clear(resizedBoard,resizedPla,rules,initialEncorePhase);
          for(int step = 0; step<12; step++) {
            if(resizedHist.isGameFinished)
              break;
            Loc resizedLoc = PlayUtils::chooseRandomLegalMove(resizedBoard,resizedHist,resizedPla,rand,Board::NULL_LOC);
            if(resizedLoc == Board::NULL_LOC)
              resizedLoc = Board::PASS_LOC;
            if(!resizedHist.isLegal(resizedBoard,resizedLoc,resizedPla))
              continue;
            resizedHist.makeBoardMoveAssumeLegal(resizedBoard,resizedLoc,resizedPla,NULL,true);
            resizedPla = getOpp(resizedPla);
            if(checkPosition(resizedBoard,resizedHist,resizedPla,drawEquivalentWinsForWhite,stats))
              stats.numResizedOntoDifferentBoardChecks += 1;
          }
        }
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
  report("  compared against a full sweep    ", stats.numFullSweepComparisons, 83000);
  report("  with a never-occupied suicide pt ", stats.numWithNeverOccupiedSuicidePoint, 15000);
  report("  never-occupied point actually banned", stats.numWithNeverOccupiedCandidateBanned, 35);
  report("  out-of-band board scrambles      ", stats.numOutOfBandScrambleChecks, 950);
  report("  out-of-band substituted boards   ", stats.numOutOfBandSubstitutedBoardChecks, 880);
  report("  resized onto a different board   ", stats.numResizedOntoDifferentBoardChecks, 16000);
  //Not floored, and printed without one on purpose: this count is 0 by construction, because a
  //history's own recompute always clears the ban at its own board's ko loc. The constructed
  //ko-loc-is-banned pairs counted above are what cover that branch instead.
  cout << "  with the ko loc itself banned    " << " " << stats.numWithKoLocAlsoBanned
       << " (0 by construction, not floored)" << endl;
}
