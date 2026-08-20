#ifndef TESTS_TESTCACHESWAPSEAM_H_
#define TESTS_TESTCACHESWAPSEAM_H_

#include "../neuralnet/nncachetwolevel.h"

// THE DECLARED TEST SEAM FOR THE LEVEL-0 SWAP DOOR.
//
// NNCacheLevelZeroSwapPermit is the key to every act that mutates the level-0 resolution list --
// NNCacheTwoLevelTable::attachLevelZero and detachLevelZero, and NNEvaluator's two forwarding
// surfaces. It cannot be constructed outside the three mints it names, so a caller that has not
// declared itself one of them cannot write those calls: they do not compile.
//
// The unit suites DO need to write them: attaching, detaching, re-attaching and probing the
// resolution order in one thread is most of what tests/testnncachetwolevel.cpp exists to do, and
// none of it is the hazard the permit exists against -- there is no evaluation in flight in a
// single-threaded unit test. So the seam is real and it is HERE, in one header under tests/,
// rather than as a relaxation of the door itself.
//
// A production translation unit that included this header would mint too. That is deliberately
// left as a review surface rather than closed further: it is one greppable line, in a file that
// has no business including a tests/ header, and it is a categorically different act from the
// ordinary-looking method call the permit replaced.
class NNCacheLevelZeroSwapTestSeam {
 public:
  [[nodiscard]] static NNCacheLevelZeroSwapPermit permit() { return NNCacheLevelZeroSwapPermit(); }
};

#endif  // TESTS_TESTCACHESWAPSEAM_H_
