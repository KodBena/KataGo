#include "../tests/tests.h"

#include <algorithm>
#include <limits>
#include <vector>

#include "../neuralnet/nninputs.h"
#include "../search/policymaskedargmax.h"

using namespace std;
using namespace TestCommon;

//PolicyMaskedArgmax::find replaces a loop of conditional skips with branchless arithmetic over a sort
//key derived from each policy prob's bit pattern. The claim that earns that rewrite is not "it is
//fast" - it is "it picks the SAME index the conditional loop picks, for every float". This file is
//that claim's witness, and it is built so that its red really is red for that reason:
//
//  - The reference below spells the ORIGINAL predicate ("skip if excluded; skip if prob < 0;
//    take if prob > best"), not the implementation's ("candidate if prob >= 0"). Writing the
//    implementation's own rule here would make the test agree with itself no matter what; writing
//    the original's makes disagreement on NaN or on negative zero a failure, which is the whole
//    point of checking.
//  - The inputs deliberately include the values a real search never produces and the byte-identity
//    output tests can therefore never see: negative and positive zero, quiet NaN, denormals of both
//    signs, both infinities, and exact ties. Those are exactly the cases where a bit-pattern key and
//    a float comparison can disagree.
//  - Exactness is the bar. These are argmax indices, not measurements; a tolerance here would be a
//    category error.

//A direct transcription of the loop PolicyMaskedArgmax::find replaced, kept in the original's own
//terms on purpose. Do not "simplify" it to match the implementation - that would delete the test.
static int referenceFind(int policySize, const float* policyProbs, const int32_t* posExcluded) {
  float best = -1.0f;
  int bestPos = -1;
  for(int pos = 0; pos<policySize; pos++) {
    if(posExcluded[pos] != 0)
      continue;
    float prob = policyProbs[pos];
    if(prob < 0)
      continue;
    if(prob > best) {
      best = prob;
      bestPos = pos;
    }
  }
  return bestPos;
}

static void checkAgrees(const char* what, int policySize, const float* policyProbs, const int32_t* posExcluded) {
  vector<int32_t> keyScratch((size_t)policySize);
  int expected = referenceFind(policySize,policyProbs,posExcluded);
  int actual = PolicyMaskedArgmax::find(policySize,policyProbs,posExcluded,keyScratch.data());
  if(expected != actual) {
    ostringstream out;
    out << "PolicyMaskedArgmax disagrees with the conditional scan it replaced (" << what
        << "): conditional scan picked " << expected << ", masked argmax picked " << actual
        << ", policySize " << policySize;
    if(expected >= 0)
      out << ", prob there " << policyProbs[expected];
    if(actual >= 0)
      out << ", prob at the other " << policyProbs[actual];
    throw StringError(out.str());
  }
}

//The floats that make this test worth running. Every one of them is a case where reading a float's
//bits as an integer and comparing floats can part company.
static vector<float> awkwardProbs() {
  return {
    0.0f,
    -0.0f,
    std::numeric_limits<float>::quiet_NaN(),
    std::numeric_limits<float>::denorm_min(),
    -std::numeric_limits<float>::denorm_min(),
    std::numeric_limits<float>::min(),
    std::numeric_limits<float>::infinity(),
    -std::numeric_limits<float>::infinity(),
    1.0f,
    -1.0f,
    0.5f,
    std::numeric_limits<float>::max(),
  };
}

void Tests::runPolicyMaskedArgmaxTests() {
  cout << "Running policy masked argmax tests" << endl;

  const vector<float> awkward = awkwardProbs();

  //Every ordered pair of awkward values, at every pair of positions, both excluded and not. Small
  //enough to be exhaustive, which beats sampling for cases this pointed.
  for(size_t a = 0; a<awkward.size(); a++) {
    for(size_t b = 0; b<awkward.size(); b++) {
      for(int excludeMask = 0; excludeMask<4; excludeMask++) {
        float probs[2] = {awkward[a], awkward[b]};
        int32_t excluded[2] = {
          (excludeMask & 1) ? PolicyMaskedArgmax::EXCLUDED : 0,
          (excludeMask & 2) ? PolicyMaskedArgmax::EXCLUDED : 0
        };
        checkAgrees("exhaustive awkward pair",2,probs,excluded);
      }
    }
  }

  //Every awkward value alone, and the empty vector.
  for(size_t a = 0; a<awkward.size(); a++) {
    float probs[1] = {awkward[a]};
    int32_t notExcluded[1] = {0};
    int32_t excluded[1] = {PolicyMaskedArgmax::EXCLUDED};
    checkAgrees("single value",1,probs,notExcluded);
    checkAgrees("single excluded value",1,probs,excluded);
  }
  {
    testAssert(PolicyMaskedArgmax::find(0,NULL,NULL,NULL) == PolicyMaskedArgmax::NO_POS);
  }

  //Ties: the rule is that the lowest index wins, and it is behaviour, not a detail. Build vectors
  //that are nothing but ties and check the index, not just the agreement.
  {
    const int policySize = 37;
    vector<float> probs((size_t)policySize, 0.25f);
    vector<int32_t> excluded((size_t)policySize, 0);
    vector<int32_t> keyScratch((size_t)policySize);
    testAssert(PolicyMaskedArgmax::find(policySize,probs.data(),excluded.data(),keyScratch.data()) == 0);
    for(int firstOpen = 0; firstOpen<policySize; firstOpen++) {
      std::fill(excluded.begin(),excluded.end(),PolicyMaskedArgmax::EXCLUDED);
      for(int pos = firstOpen; pos<policySize; pos++)
        excluded[(size_t)pos] = 0;
      testAssert(PolicyMaskedArgmax::find(policySize,probs.data(),excluded.data(),keyScratch.data()) == firstOpen);
      checkAgrees("all-ties, lowest open index",policySize,probs.data(),excluded.data());
    }
    //A tie between the two zeroes must resolve the same way as a tie between two ordinary equals.
    vector<float> zeroes((size_t)policySize);
    for(int pos = 0; pos<policySize; pos++)
      zeroes[(size_t)pos] = (pos % 2 == 0) ? 0.0f : -0.0f;
    std::fill(excluded.begin(),excluded.end(),0);
    checkAgrees("positive and negative zeroes tied",policySize,zeroes.data(),excluded.data());
    testAssert(PolicyMaskedArgmax::find(policySize,zeroes.data(),excluded.data(),keyScratch.data()) == 0);
    excluded[0] = PolicyMaskedArgmax::EXCLUDED;
    checkAgrees("negative zero first among zeroes",policySize,zeroes.data(),excluded.data());
    testAssert(PolicyMaskedArgmax::find(policySize,zeroes.data(),excluded.data(),keyScratch.data()) == 1);
  }

  //Nothing at all is a candidate: every position excluded, and every position illegal.
  {
    const int policySize = 41;
    vector<float> probs((size_t)policySize, 0.5f);
    vector<int32_t> excluded((size_t)policySize, PolicyMaskedArgmax::EXCLUDED);
    vector<int32_t> keyScratch((size_t)policySize);
    testAssert(PolicyMaskedArgmax::find(policySize,probs.data(),excluded.data(),keyScratch.data()) == PolicyMaskedArgmax::NO_POS);
    std::fill(excluded.begin(),excluded.end(),0);
    std::fill(probs.begin(),probs.end(),-1.0f);
    testAssert(PolicyMaskedArgmax::find(policySize,probs.data(),excluded.data(),keyScratch.data()) == PolicyMaskedArgmax::NO_POS);
    std::fill(probs.begin(),probs.end(),std::numeric_limits<float>::quiet_NaN());
    checkAgrees("every prob a NaN",policySize,probs.data(),excluded.data());
    testAssert(PolicyMaskedArgmax::find(policySize,probs.data(),excluded.data(),keyScratch.data()) == PolicyMaskedArgmax::NO_POS);
  }

  //Randomised vectors at the real policy size, mixing ordinary probs with the awkward values and a
  //sparse exclusion set - the shape a descend step actually sees, plus the values it never sees.
  {
    Rand rand("runPolicyMaskedArgmaxTests");
    const int policySize = NNPos::MAX_NN_POLICY_SIZE;
    vector<float> probs((size_t)policySize);
    vector<int32_t> excluded((size_t)policySize);
    for(int trial = 0; trial<2000; trial++) {
      for(int pos = 0; pos<policySize; pos++) {
        double roll = rand.nextDouble();
        if(roll < 0.25)
          probs[(size_t)pos] = -1.0f;                                  //illegal, as the net writes it
        else if(roll < 0.35)
          probs[(size_t)pos] = awkward[rand.nextUInt((uint32_t)awkward.size())];
        else if(roll < 0.55)
          probs[(size_t)pos] = 0.0f;                                   //underflowed to zero
        else if(roll < 0.75)
          probs[(size_t)pos] = (float)(rand.nextUInt(8) * 0.125);      //deliberate ties
        else
          probs[(size_t)pos] = (float)rand.nextDouble();
        excluded[(size_t)pos] = rand.nextDouble() < 0.1 ? PolicyMaskedArgmax::EXCLUDED : 0;
      }
      checkAgrees("randomised full-size vector",policySize,probs.data(),excluded.data());
      //And at a shorter length, so the vectorised body's tail is exercised at every remainder.
      int shortSize = 1 + (int)rand.nextUInt((uint32_t)policySize);
      checkAgrees("randomised partial-length vector",shortSize,probs.data(),excluded.data());
    }
  }

  cout << "Done" << endl;
}
