/*
 * The seen-red witness for the sound ladder memo (game/laddercache.h).
 *
 * The claim is negative -- "a validated cache entry never carries a different answer than the
 * search would produce" -- and absence cannot be watched, so it is converted into a positive
 * observation: LadderCache::reportVerification is a tripwire at the exact site where a cached
 * answer and a recomputed answer meet, and its FIRING is the observation.
 *
 * Byte-identity of runoutputtests is necessary and not sufficient for this claim: it covers only
 * the positions those tests happen to use. The substrate here is a LADDER-BREAKER pair, chosen
 * precisely to break a chain-shape key -- two boards carrying the same chain with the same
 * liberties, where the ladder resolves differently because of a stone far away in the ladder's
 * diagonal path. That is the far-field dependence exp-lt9-ladder-census.md named and left
 * UNEXERCISED, and impl-lt9-ladder-cache.md measured at 13.2% of hits under the census's key.
 *
 *   runladdercachetests stale  -- RED. Insert under a hand-truncated read set carrying only the
 *     pursued chain's own closure -- the census's chain-shape key, built through the ordinary
 *     public API with no production-only hook -- then look up on the broken board. The truncated
 *     read set validates, hands back the wrong answer, and reportVerification REFUSES. This
 *     subcommand is EXPECTED TO DIE; a zero exit status is the failure.
 *   runladdercachetests sound  -- GREEN. Identical boards, identical code path, only the read set
 *     changed: record what the marking sites actually produce, and the entry declines to validate
 *     against a board it does not describe. Green here is not the absence of a crash -- it is a
 *     miss where the red leg had a wrong hit.
 */
#include "../main.h"

#include <cstdlib>
#include <iostream>

#include "../core/global.h"
#include "../game/board.h"
#include "../game/laddercache.h"

using namespace std;

namespace {

  constexpr int BOARD_LEN = 19;

  //A ladder start, built rather than drawn so its geometry is stated as the rule it is: a lone
  //defender stone in atari on three orthogonal sides, plus one attacker stone on the diagonal
  //that turns the chase into a ladder instead of a two-move escape. Everything else on the board
  //is empty, so the ladder's own diagonal is free and a breaker can be placed far from the chain.
  //
  //The stone is left with ONE liberty on purpose. The two-liberty shape has a ladder running in
  //BOTH diagonal directions, and searchIsLadderCapturedAttackerFirst2Libs answers true if either
  //works, so a single distant breaker cannot flip it and every candidate would be unbreakable --
  //witnessed while building this file, and the reason the one-liberty variant is used instead.
  //
  //Whether this actually ladders is not asserted here -- buildSubstrate below verifies it against
  //the real search and refuses if it does not.
  Board makeLadderBoard(int px, int py, int dx, int dy, Player defender) {
    Board board(BOARD_LEN,BOARD_LEN);
    const Player attacker = getOpp(defender);
    board.setStone(Location::getLoc(px,py,BOARD_LEN), defender);
    board.setStone(Location::getLoc(px-dx,py,BOARD_LEN), attacker);
    board.setStone(Location::getLoc(px,py-dy,BOARD_LEN), attacker);
    board.setStone(Location::getLoc(px+dx,py,BOARD_LEN), attacker);
    board.setStone(Location::getLoc(px+dx,py+dy,BOARD_LEN), attacker);
    return board;
  }

  //Variant 1 -- the one-liberty ladder search, run on a copy exactly as iterLadders does. The
  //one-liberty variant reports no working moves, so the answer under test is the verdict itself.
  constexpr int LADDER_VARIANT = 1;

  LadderAnswer runLadderSearch(const Board& board, Loc target) {
    Board copy(board);
    vector<Loc> buf;
    LadderAnswer answer;
    answer.laddered = copy.searchIsLadderCaptured(target,true,buf);
    return answer;
  }

  //Finds a point far from the pursued chain whose occupation FLIPS the ladder verdict while
  //leaving the chain and its liberties untouched -- i.e. a genuine ladder breaker. Searching for
  //one rather than asserting a hand-picked coordinate keeps the witness honest: if no such point
  //exists on this position, the whole exercise is vacuous and must say so rather than pass.
  //Returns the FARTHEST such point, not the first in scan order: the claim being witnessed is
  //about board content the pursued chain does not mention, so the more distant the breaker the
  //less room there is to read the red as some local effect the chain-shape key could have seen.
  Loc findLadderBreaker(const Board& base, Loc target, bool baseLaddered) {
    const Player defender = base.colors[target];
    Loc best = Board::NULL_LOC;
    int bestDist = -1;
    for(int y = 0; y < BOARD_LEN; y++) {
      for(int x = 0; x < BOARD_LEN; x++) {
        Loc loc = Location::getLoc(x,y,BOARD_LEN);
        if(base.colors[loc] != C_EMPTY)
          continue;
        //Leave the chain's own neighbourhood alone: a breaker adjacent to the chain would change
        //its liberties, which even a chain-shape key already sees, and would witness nothing.
        if(Location::distance(loc,target,BOARD_LEN) <= 2)
          continue;
        Board broken(base);
        if(!broken.setStoneFailIfNoLibs(loc,defender))
          continue;
        //A breaker must leave the pursued chain and its liberties untouched, or the two boards
        //would not agree on the chain-shape key and the red leg would observe the bucket key
        //rather than the read set.
        if(broken.getNumLiberties(target) != base.getNumLiberties(target))
          continue;
        const int dist = Location::distance(loc,target,BOARD_LEN);
        if(dist <= bestDist)
          continue;
        if(runLadderSearch(broken,target).laddered != baseLaddered) {
          best = loc;
          bestDist = dist;
        }
      }
    }
    return best;
  }

  //Every stone with exactly two liberties whose ladder verdict is TRUE -- the only chains a
  //breaker can flip, since a breaker makes a working ladder fail rather than the reverse.
  vector<Loc> workingLadderTargets(const Board& board) {
    vector<Loc> targets;
    for(int y = 0; y < BOARD_LEN; y++) {
      for(int x = 0; x < BOARD_LEN; x++) {
        Loc loc = Location::getLoc(x,y,BOARD_LEN);
        if(board.colors[loc] != C_BLACK && board.colors[loc] != C_WHITE)
          continue;
        if(board.getNumLiberties(loc) != 2)
          continue;
        if(runLadderSearch(board,loc).laddered)
          targets.push_back(loc);
      }
    }
    return targets;
  }

  struct Substrate {
    Board base;
    Board broken;
    Loc target = Board::NULL_LOC;
    Loc breaker = Board::NULL_LOC;
    LadderAnswer baseAnswer;
    LadderAnswer brokenAnswer;
  };

  Substrate buildSubstrate() {
    Substrate s;

    //Search over the four diagonal orientations for a board carrying a working ladder together
    //with a DISTANT breaker, rather than asserting hand-picked coordinates. If no such pair
    //exists the witness observes nothing, and it says so and refuses rather than reporting a
    //green it did not earn.
    //Candidate placements are visited in an order that puts the LONGEST ladders first. From the
    //shape above, a ladder built with diagonal (dx,dy) runs in direction (-dx,+dy), so seating the
    //defender in the corner opposite that run gives the chase the whole board to cross -- and the
    //longer the chase, the farther away a breaker can sit and still decide it.
    int numCandidates = 0;
    int numLaddered = 0;
    const int dirs[4][2] = {{1,1},{1,-1},{-1,1},{-1,-1}};
    for(int d = 0; d < 4 && s.breaker == Board::NULL_LOC; d++) {
      const int startPx = dirs[d][0] > 0 ? BOARD_LEN-3 : 2;
      const int startPy = dirs[d][1] > 0 ? 2 : BOARD_LEN-3;
      const int stepPx = dirs[d][0] > 0 ? -1 : 1;
      const int stepPy = dirs[d][1] > 0 ? 1 : -1;
      for(int yi = 0; yi < BOARD_LEN-4 && s.breaker == Board::NULL_LOC; yi++) {
        for(int xi = 0; xi < BOARD_LEN-4 && s.breaker == Board::NULL_LOC; xi++) {
          const int px = startPx + stepPx*xi;
          const int py = startPy + stepPy*yi;
          if(px < 2 || px >= BOARD_LEN-2 || py < 2 || py >= BOARD_LEN-2)
            continue;
          Board candidate = makeLadderBoard(px,py,dirs[d][0],dirs[d][1],P_BLACK);
          Loc target = Location::getLoc(px,py,BOARD_LEN);
          if(candidate.colors[target] != C_BLACK || candidate.getNumLiberties(target) != 1)
            continue;
          numCandidates += 1;
          LadderAnswer answer = runLadderSearch(candidate,target);
          if(!answer.laddered)
            continue;
          numLaddered += 1;
          Loc breaker = findLadderBreaker(candidate,target,true);
          if(breaker == Board::NULL_LOC)
            continue;
          s.base = candidate;
          s.target = target;
          s.breaker = breaker;
          s.baseAnswer = answer;
        }
      }
    }
    if(s.breaker == Board::NULL_LOC)
      Global::fatalError(
        "laddercachetests: no (working ladder, distant breaker) pair was found among " +
        Global::intToString(numCandidates) + " candidate ladder shapes, of which " +
        Global::intToString(numLaddered) + " actually laddered. There is nothing for the witness "
        "to observe; it is vacuous and refuses to report a result."
      );

    s.broken = s.base;
    s.broken.setStoneFailIfNoLibs(s.breaker,s.base.colors[s.target]);
    s.brokenAnswer = runLadderSearch(s.broken,s.target);

    cout << "substrate: pursued chain at " << Location::toString(s.target,s.base)
         << ", ladder breaker at " << Location::toString(s.breaker,s.base)
         << " (distance " << Location::distance(s.breaker,s.target,BOARD_LEN) << ")" << endl;
    cout << "  laddered without the breaker: " << (s.baseAnswer.laddered ? "true" : "false") << endl;
    cout << "  laddered with the breaker:    " << (s.brokenAnswer.laddered ? "true" : "false") << endl;
    return s;
  }

  //The read set the marking sites actually produce over a real search: the honest key.
  LadderReadSet recordFullReadSet(const Board& board, Loc target, uint64_t key) {
    Board copy(board);
    vector<Loc> buf;
    LadderReadScope scope(board,key);
    (void)copy.searchIsLadderCaptured(target,true,buf);
    return std::move(scope).finish();
  }

  //A deliberately TRUNCATED read set: the pursued chain's members and liberties, and nothing the
  //search goes on to read. That is exactly the census's chain-shape key. It is built through the
  //ordinary public API -- open a scope, contribute those points via LadderRead::mark, close it
  //before any search runs -- so the red leg needs no production-only hook to exist.
  LadderReadSet recordTruncatedReadSet(const Board& board, Loc target, uint64_t key) {
    LadderReadScope scope(board,key);
    Loc cur = target;
    do {
      LadderRead::mark((int)cur);
      for(int i = 0; i < 4; i++) {
        Loc adj = cur + board.adj_offsets[i];
        if(board.colors[adj] == C_EMPTY)
          LadderRead::mark((int)adj);
      }
      cur = board.next_in_chain[cur];
    } while(cur != target);
    return std::move(scope).finish();
  }
}

int MainCmds::runladdercachetests(const vector<string>& args) {
  //handleSubcommand hands each subcommand a vector whose [0] is the subcommand name itself, so
  //the leg selector is [1].
  string mode = args.size() >= 2 ? args[1] : string("");
  if(mode != "sound" && mode != "stale") {
    cerr << "Usage: katago runladdercachetests <sound|stale>" << endl;
    cerr << "  sound -- the green leg: the full read set declines to validate against a board it" << endl;
    cerr << "           does not describe, so the wrong answer is never handed back." << endl;
    cerr << "  stale -- the red leg: a truncated read set validates against that same board and" << endl;
    cerr << "           hands back the wrong answer, and the verification tripwire refuses." << endl;
    cerr << "           THIS MODE IS EXPECTED TO DIE. A zero exit status is the failure." << endl;
    return 1;
  }

  Board::initHash();
  Substrate s = buildSubstrate();

  const uint64_t key = ladderBucketKey(s.base,s.target,LADDER_VARIANT);
  //The bucket key is deliberately identical for the two boards -- it is computed from the chain
  //and its liberties, which the breaker does not touch. If it differed, the lookup below would
  //miss for a reason having nothing to do with the read set, and the witness would observe the
  //bucket key rather than the property it claims to observe.
  const uint64_t brokenKey = ladderBucketKey(s.broken,s.target,LADDER_VARIANT);
  if(key != brokenKey)
    Global::fatalError(
      "laddercachetests: the two boards fall in different buckets, so this witness would observe "
      "the bucket key rather than the read set. Refusing to report a result."
    );
  cout << "bucket key identical across both boards: yes" << endl;

  LadderCache cache(LadderCache::DEFAULT_BUCKET_CAP, LadderCache::DEFAULT_MAX_BYTES);
  LadderReadSet readSet = (mode == "stale")
    ? recordTruncatedReadSet(s.base,s.target,key)
    : recordFullReadSet(s.base,s.target,key);
  cout << "read set recorded on the unbroken board: " << readSet.numPoints() << " points"
       << (mode == "stale" ? " (TRUNCATED -- the census's chain-shape key)" : " (full closure)") << endl;
  cache.insert(std::move(readSet), s.baseAnswer);

  //Now ask the cache about the BROKEN board, whose ladder resolves the other way.
  std::optional<LadderAnswer> cached = cache.lookup(s.broken,key);

  if(mode == "sound") {
    if(cached.has_value()) {
      cout << "GREEN LEG FAILED: the full read set validated against the broken board" << endl;
      return 1;
    }
    cout << "GREEN: the full read set did NOT validate against the broken board -- a miss, so the "
         << "wrong answer was never available to hand back." << endl;
    //And the same entry must still validate against the board it was recorded on, or the "miss"
    //above would be explained by the entry being useless rather than by the breaker.
    LadderCache cache2(LadderCache::DEFAULT_BUCKET_CAP, LadderCache::DEFAULT_MAX_BYTES);
    cache2.insert(recordFullReadSet(s.base,s.target,key), s.baseAnswer);
    std::optional<LadderAnswer> self = cache2.lookup(s.base,key);
    if(!self.has_value() || self->laddered != s.baseAnswer.laddered) {
      cout << "GREEN LEG FAILED: the full read set did not validate against its own board either, "
           << "so the miss above says nothing about the breaker" << endl;
      return 1;
    }
    cout << "GREEN: the same entry DOES validate against its own board and returns laddered="
         << (self->laddered ? "true" : "false") << " -- so the miss above is the breaker, not a "
         << "useless entry." << endl;
    cout << "All ladder cache tests passed" << endl;
    return 0;
  }

  //Red leg.
  if(!cached.has_value()) {
    cout << "RED LEG FAILED: the truncated read set did not even validate, so the tripwire was "
         << "never reached and nothing was observed" << endl;
    return 1;
  }
  cout << "RED: the truncated read set VALIDATED against the broken board and handed back "
       << "laddered=" << (cached->laddered ? "true" : "false")
       << " where the search on that board gives laddered="
       << (s.brokenAnswer.laddered ? "true" : "false") << endl;
  cout << "RED: handing both to the verification tripwire now; it must refuse." << endl;
  cout.flush();
  LadderCache::reportVerification(*cached, s.brokenAnswer);

  cout << "RED LEG FAILED: reportVerification returned instead of refusing" << endl;
  return 1;
}
