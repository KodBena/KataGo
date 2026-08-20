/*
 * policymaskedargmax.h
 * Which policy index a whole-policy-vector scan picks, as one named operation.
 */

#ifndef SEARCH_POLICYMASKEDARGMAX_H_
#define SEARCH_POLICYMASKEDARGMAX_H_

#include <cstdint>
#include <cstring>
#include <limits>

#include "../core/global.h"

//Search's new-child scan asks one question of the whole policy vector: of the positions that are
//still candidates, which has the highest policy prob? Written as a loop of conditional skips that
//question costs a data-dependent, unpredictable branch per position, once per descend step, over
//every position on the board - which profiling identifies as the largest single branch-mispredict
//site in the engine. Written as arithmetic over a sort key it costs no branch at all, and becomes
//something a compiler's own vectoriser will take.
//
//This is that operation, named and separated from its caller so that the property that actually
//matters - it picks the SAME index a conditional scan would pick, for every float, not just the
//ones a game happens to produce - can be observed by a test rather than argued in a comment. See
//Tests::runPolicyMaskedArgmaxTests, which checks it against a direct transcription of the
//conditional form over randomised vectors including negative zero, NaN, denormals and infinities.
namespace PolicyMaskedArgmax {

  //The word that marks a position as no longer a candidate, and equally the sort key of such a
  //position: all bits set, i.e. -1. It is one value for both jobs on purpose - that is what lets a
  //single OR per position do the whole exclusion. Callers building the exclusion array must write
  //this exact word, not merely "something nonzero".
  static constexpr int32_t EXCLUDED = -1;

  //What find() returns when no position was a candidate.
  static constexpr int NO_POS = -1;

  //The order-preserving reading of a float's bits as an integer is a fact about IEEE-754 binary32
  //and two's-complement integers, not about C++ in general, so say so where a port would trip over
  //it rather than leaving it to be discovered as a wrong move choice.
  static_assert(
    std::numeric_limits<float>::is_iec559 && sizeof(float) == sizeof(int32_t),
    "PolicyMaskedArgmax reads a float's bits as an integer, which orders correctly only for IEEE-754 binary32"
  );
  //The two properties EXCLUDED is relied on for: OR-ing it over any sort key yields it (so exclusion
  //is one OR), and it loses to every candidate key (candidate keys are non-negative).
  static_assert(
    (EXCLUDED | 0x7FFFFFFF) == EXCLUDED && EXCLUDED < 0,
    "EXCLUDED must be all bits set, and negative so that every candidate key beats it"
  );

  //The index of the highest-prob candidate position in policyProbs[0,policySize), or NO_POS if there
  //is none. A position is a candidate iff posExcluded[pos] == 0 and its prob is not negative.
  //
  //Equivalent, position for position, to:
  //
  //    best = -1.0f; bestPos = NO_POS;
  //    for(pos...) {
  //      if(posExcluded[pos] != 0)          continue;
  //      if(policyProbs[pos] < 0)           continue;
  //      if(policyProbs[pos] > best) { best = policyProbs[pos]; bestPos = pos; }
  //    }
  //
  //including its two non-obvious cases:
  //  - TIES GO TO THE LOWEST INDEX. The reference form's comparison is a strict greater-than, so a
  //    later position whose prob equals the running best does not displace the earlier one. Move
  //    selection is order-sensitive downstream, so this is behaviour, not a detail.
  //  - A NaN PROB IS NEVER PICKED. The reference form does not skip it - NaN < 0 is false - but it
  //    can never win either, because every ordered comparison against a NaN is false. Here it is
  //    excluded outright by failing "not negative", which reaches the identical outcome: not picked,
  //    not recorded.
  //And one case that makes the difference between this and a naive bit-pattern key:
  //  - A NEGATIVE ZERO PROB IS A CANDIDATE. -0.0f < 0 is false, so the reference form admits it, and
  //    -0.0f > -1.0f is true, so it can win. Its bit pattern read as an integer is however the
  //    SMALLEST int32 there is. Masking bit 31 off the key gives it +0.0f's key, which is right
  //    because -0.0f and +0.0f compare equal as floats. The mask is a no-op for every other
  //    candidate, whose sign bit is already clear. This is done on the bits and not by adding 0.0f
  //    to the value, because that addition normalises only under round-to-nearest and flushes a
  //    denormal to zero under MXCSR's flush-to-zero - it would make the answer depend on
  //    floating-point ENVIRONMENT state, which the reference form, doing no arithmetic at all, does
  //    not depend on.
  //
  //keyScratch is caller-provided working space of at least policySize entries; its contents on
  //return are unspecified. It is a parameter rather than a local so the caller can keep one buffer
  //per thread instead of one per call.
  inline int find(
    int policySize,
    const float* policyProbs,
    const int32_t* posExcluded,
    int32_t* keyScratch
  ) {
    int32_t bestKey = EXCLUDED;
    for(int pos = 0; pos<policySize; pos++) {
      float policyProb = policyProbs[pos];
      int32_t key;
      static_assert(sizeof(key) == sizeof(policyProb), "The sort key must be as wide as the prob whose bits it is");
      std::memcpy(&key,&policyProb,sizeof(key));
      //Bit 31 off: negative zero sorts where positive zero does. A no-op for any other candidate.
      key &= 0x7FFFFFFF;
      //One OR excludes: EXCLUDED is all bits set, so it swallows the key whole.
      key |= posExcluded[pos] | -(int32_t)(policyProb >= 0.0f ? 0 : 1);
      keyScratch[pos] = key;
      bestKey = key > bestKey ? key : bestKey;
    }
    if(bestKey == EXCLUDED)
      return NO_POS;

    //The lowest index attaining the best key. Equal probs have equal keys - the only two distinct
    //bit patterns that compare equal as floats are the two zeroes, and the mask above merged them -
    //so this is the lowest index attaining the best prob.
    int bestPos = 0;
    while(bestPos < policySize && keyScratch[bestPos] != bestKey)
      bestPos += 1;
    //bestKey was taken out of this buffer over this range, so it is in it. Reaching the end would
    //mean the two loops disagreed about what they scanned; returning NO_POS there would be a silently
    //skipped move, so refuse instead.
    if(bestPos >= policySize)
      throw StringError("PolicyMaskedArgmax::find: the best key is absent from the buffer it was taken from");
    return bestPos;
  }

}

#endif  // SEARCH_POLICYMASKEDARGMAX_H_
