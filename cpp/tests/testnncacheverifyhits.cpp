#include "../tests/tests.h"

// EMPTY TRANSLATION UNIT IN THE DEFAULT BUILD, for the same reason nncacheverifyhits.cpp is:
// the hit verifier does not exist there, so neither does anything that tests it or feeds it.
#ifdef KATAGO_NNCACHE_VERIFY_HITS

#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>

#include "../core/global.h"
#include "../core/test.h"
#include "../neuralnet/nncachefileformat.h"
#include "../neuralnet/nncacheverifyhits.h"
#include "../neuralnet/nnevalcontainer.h"
#include "../neuralnet/nninputs.h"

using namespace std;

// WHAT IS ASSERTED HERE AND WHAT IS NOT.
//
// This suite covers the COMPARISON -- that the allowance is the one the header describes, that
// a channel outside it is reported by name, that the counters say what happened. It cannot
// cover the thing the feature exists for, which is a real level-0 hit compared against a real
// forward pass in a real engine: that needs a net, a cache directory, two processes and a
// corrupted store, and it lives in the out-of-process witness the corruption instrument at the
// bottom of this file feeds.
//
// So this file has two halves and they are different kinds of thing: a suite, and an
// instrument. The instrument is here rather than in a bench because what it does -- forge a
// checksum-VALID semantic corruption -- is the same act testnnevalcontainer.cpp already does
// for the later-version refusal, using the same production checksum functions, and putting it
// beside a second hand-rolled copy of the format's framing is exactly what that test's own
// comment refuses.

namespace {

// AN NNOutput FILLED FROM ITS SERIAL, so no two channels hold the same number and a comparison
// that read the wrong field would land on a visibly wrong value rather than on a coincidence.
void fillOutput(NNOutput& out, int nnXLen, int nnYLen, bool withOwnerMap, double bias) {
  out.nnHash = Hash128(0x1234567890abcdefULL, 0xfedcba0987654321ULL);
  out.whiteWinProb = (float)(0.51 + bias);
  out.whiteLossProb = (float)(0.42 + bias);
  out.whiteNoResultProb = (float)(0.07 + bias);
  out.whiteScoreMean = (float)(3.75 + bias);
  out.whiteScoreMeanSq = (float)(410.5 + bias);
  out.whiteLead = (float)(2.25 + bias);
  out.varTimeLeft = (float)(88.5 + bias);
  out.shorttermWinlossError = (float)(0.031 + bias);
  out.shorttermScoreError = (float)(1.75 + bias);
  out.policyOptimismUsed = 0.0f;
  out.nnXLen = nnXLen;
  out.nnYLen = nnYLen;
  for(int i = 0; i < NNPos::MAX_NN_POLICY_SIZE; i++)
    out.policyProbs[i] = -1.0f;
  for(int i = 0; i < nnXLen * nnYLen + 1; i++)
    out.policyProbs[i] = (float)(((i * 37) % 997) * 0.001 + bias);
  out.noisedPolicyProbs = NULL;
  if(withOwnerMap) {
    out.whiteOwnerMap = new float[nnXLen * nnYLen];
    for(int i = 0; i < nnXLen * nnYLen; i++)
      out.whiteOwnerMap[i] = (float)(((i * 53) % 2001) * 0.001 - 1.0 + bias);
  }
  else
    out.whiteOwnerMap = NULL;
}

NNCacheHitVerifyTolerances fp32Tolerances() {
  return NNCacheHitVerifyTolerances::defaults();
}

const int EDGE = 19;
const int POLICY_SIZE = EDGE * EDGE + 1;

//-------------------------------------------------------------------------------------
// The suite
//-------------------------------------------------------------------------------------

// TWO IDENTICAL EVALUATIONS HOLD, and the worst ratio is zero rather than merely "under one".
// This is the leg that would go red if the allowance were computed as zero.
void testIdenticalOutputsHoldWithNoDeviation() {
  NNCacheHitVerifier verifier(fp32Tolerances(), false, NULL);
  NNOutput a, b;
  fillOutput(a, EDGE, EDGE, true, 0.0);
  fillOutput(b, EDGE, EDGE, true, 0.0);
  testAssert(verifier.compare(a.nnHash, 0, a, b, POLICY_SIZE));
  const NNCacheHitVerifyStats stats = verifier.stats();
  testAssert(stats.verifiedHits == 1);
  testAssert(stats.mismatches == 0);
  testAssert(stats.worstDeviationRatio == 0.0);
  cout << "hit verify: two identical evaluations hold, worstDeviationRatio="
       << stats.worstDeviationRatio << endl;
}

// A PERTURBATION INSIDE THE ALLOWANCE HOLDS. Not a decoration: it is what makes the tolerance
// a tolerance rather than a bit-exactness check with extra steps, and it is the leg that goes
// red if someone "tightens" this to ==.
void testAPerturbationInsideTheAllowanceHolds() {
  NNCacheHitVerifier verifier(fp32Tolerances(), false, NULL);
  NNOutput a, b;
  fillOutput(a, EDGE, EDGE, true, 0.0);
  fillOutput(b, EDGE, EDGE, true, 0.0);
  // Half the fp32 allowance on a probability channel, whose magnitude is under the floor so
  // the allowance is exactly relative*floor = 1e-4.
  b.whiteWinProb = a.whiteWinProb + 5e-5f;
  testAssert(verifier.compare(a.nnHash, 0, a, b, POLICY_SIZE));
  const NNCacheHitVerifyStats stats = verifier.stats();
  testAssert(stats.mismatches == 0);
  testAssert(stats.worstDeviationRatio > 0.0 && stats.worstDeviationRatio < 1.0);
  cout << "hit verify: a 5e-5 move on whiteWinProb holds at ratio "
       << stats.worstDeviationRatio << " of the allowance" << endl;
}

// AND ONE OUTSIDE IT IS REPORTED, ON THE RIGHT CHANNEL. The channel name is asserted, not just
// the count: a verifier that noticed something but named the wrong field would send a reader
// looking at the wrong decoder.
void testAValueChannelOutsideTheAllowanceIsReportedByName() {
  NNCacheHitVerifier verifier(fp32Tolerances(), false, NULL);
  NNOutput a, b;
  fillOutput(a, EDGE, EDGE, true, 0.0);
  fillOutput(b, EDGE, EDGE, true, 0.0);
  b.whiteLead = a.whiteLead + 0.25f;
  testAssert(!verifier.compare(a.nnHash, 0, a, b, POLICY_SIZE));
  const NNCacheHitVerifyStats stats = verifier.stats();
  testAssert(stats.verifiedHits == 1);
  testAssert(stats.mismatches == 1);
  testAssert(stats.worstChannel == "whiteLead");
  testAssert(stats.worstDeviationRatio > 1.0);
  cout << "hit verify: a 0.25 move on whiteLead is caught, worstChannel=" << stats.worstChannel
       << " ratio=" << stats.worstDeviationRatio << endl;
}

// A POLICY SLOT IS CAUGHT AND THE SLOT INDEX IS NAMED. The policy is where a length or offset
// defect lands, so "which slot" is the whole diagnostic value.
void testAPolicySlotOutsideTheAllowanceNamesItsIndex() {
  NNCacheHitVerifier verifier(fp32Tolerances(), false, NULL);
  NNOutput a, b;
  fillOutput(a, EDGE, EDGE, true, 0.0);
  fillOutput(b, EDGE, EDGE, true, 0.0);
  b.policyProbs[17] = a.policyProbs[17] + 0.01f;
  testAssert(!verifier.compare(a.nnHash, 0, a, b, POLICY_SIZE));
  testAssert(verifier.stats().worstChannel == "policyProbs[17]");
  cout << "hit verify: a policy slot is caught by index, worstChannel="
       << verifier.stats().worstChannel << endl;
}

// AN OWNERSHIP SLOT TOO, and this is the channel a whole-map offset defect shows up in first.
void testAnOwnershipSlotOutsideTheAllowanceIsCaught() {
  NNCacheHitVerifier verifier(fp32Tolerances(), false, NULL);
  NNOutput a, b;
  fillOutput(a, EDGE, EDGE, true, 0.0);
  fillOutput(b, EDGE, EDGE, true, 0.0);
  b.whiteOwnerMap[200] = a.whiteOwnerMap[200] + 0.5f;
  testAssert(!verifier.compare(a.nnHash, 0, a, b, POLICY_SIZE));
  testAssert(verifier.stats().worstChannel == "whiteOwnerMap[200]");
  cout << "hit verify: an ownership slot is caught by index, worstChannel="
       << verifier.stats().worstChannel << endl;
}

// THE OWNERMAP-PRESENT FLAG IS ITSELF A PERSISTED FIELD, so a decoder that lost it is a
// mismatch and not a channel to skip. This is the leg that goes red if someone "helpfully"
// makes the comparison tolerate a missing map.
void testALostOwnershipMapIsAMismatchAndNotASkippedChannel() {
  NNCacheHitVerifier verifier(fp32Tolerances(), false, NULL);
  NNOutput a, b;
  fillOutput(a, EDGE, EDGE, true, 0.0);
  fillOutput(b, EDGE, EDGE, false, 0.0);
  testAssert(!verifier.compare(a.nnHash, 0, a, b, POLICY_SIZE));
  testAssert(verifier.stats().worstChannel == "whiteOwnerMap presence");
  cout << "hit verify: an ownership map present on one side only is a mismatch: "
       << verifier.stats().worstChannel << endl;
}

// A NaN ON ONE SIDE ONLY IS A MISMATCH. It has to be forced rather than computed: a NaN
// deviation compares false against every allowance and would otherwise read as "held".
void testANaNOnOneSideOnlyIsAMismatch() {
  NNCacheHitVerifier verifier(fp32Tolerances(), false, NULL);
  NNOutput a, b;
  fillOutput(a, EDGE, EDGE, true, 0.0);
  fillOutput(b, EDGE, EDGE, true, 0.0);
  b.whiteScoreMean = std::nanf("");
  testAssert(!verifier.compare(a.nnHash, 0, a, b, POLICY_SIZE));
  testAssert(verifier.stats().worstChannel == "whiteScoreMean");
  cout << "hit verify: a NaN on one side only is caught rather than passing as held" << endl;
}

// THE MAGNITUDE TERM IS LIVE, not decorative: whiteScoreMeanSq is O(100) here, so a move that
// would fail on a probability channel is inside the allowance on this one. This is the leg
// that justifies relative-with-a-floor rather than one flat absolute number.
void testTheAllowanceScalesWithTheChannelsOwnMagnitude() {
  NNCacheHitVerifier verifier(fp32Tolerances(), false, NULL);
  NNOutput a, b;
  fillOutput(a, EDGE, EDGE, true, 0.0);
  fillOutput(b, EDGE, EDGE, true, 0.0);
  // 2e-3 absolute: ten times the fp32 allowance of a probability channel, but under the
  // 1e-4 * 410.5 = 4.1e-2 that this channel's own magnitude earns it.
  b.whiteScoreMeanSq = a.whiteScoreMeanSq + 2e-3f;
  testAssert(verifier.compare(a.nnHash, 0, a, b, POLICY_SIZE));
  // The same absolute move on a small channel does NOT hold.
  NNCacheHitVerifier verifier2(fp32Tolerances(), false, NULL);
  NNOutput c, d;
  fillOutput(c, EDGE, EDGE, true, 0.0);
  fillOutput(d, EDGE, EDGE, true, 0.0);
  d.whiteWinProb = c.whiteWinProb + 2e-3f;
  testAssert(!verifier2.compare(c.nnHash, 0, c, d, POLICY_SIZE));
  cout << "hit verify: 2e-3 holds on whiteScoreMeanSq (|v|~410) and fails on whiteWinProb "
          "(|v|<1) -- the allowance follows the channel's own magnitude" << endl;
}

// THE SKIP COUNTS ARE REPORTED AND NOT FOLDED INTO verifiedHits. The whole reason they exist
// is that "mismatches: 0" from a run that verified nothing must not read like "mismatches: 0"
// from a run that verified everything.
void testTheSkipCountsAreSeparateFromTheVerifiedCount() {
  NNCacheHitVerifier verifier(fp32Tolerances(), false, NULL);
  testAssert(verifier.shouldVerify(NNCacheHitOrigin::LevelZeroPersisted));
  testAssert(!verifier.shouldVerify(NNCacheHitOrigin::LevelOneResident));
  verifier.countSkippedNondeterministicSymmetry();
  verifier.countSkippedNondeterministicSymmetry();
  const NNCacheHitVerifyStats stats = verifier.stats();
  testAssert(stats.verifiedHits == 0);
  testAssert(stats.mismatches == 0);
  testAssert(stats.skippedResidentOrigin == 1);
  testAssert(stats.skippedNondeterministicSymmetry == 2);
  cout << "hit verify: a run that verified nothing reports verifiedHits=0 beside "
          "skippedNondeterministicSymmetry=" << stats.skippedNondeterministicSymmetry
       << " skippedResidentOrigin=" << stats.skippedResidentOrigin << endl;

  // AND THE OPT-IN IS REAL: with it on, a resident hit is verified and still counted as the
  // opt-in it was, so a reader can tell an all-resident run from an all-persisted one.
  NNCacheHitVerifier withResident(fp32Tolerances(), true, NULL);
  testAssert(withResident.shouldVerify(NNCacheHitOrigin::LevelOneResident));
  testAssert(withResident.stats().skippedResidentOrigin == 1);
}

// THE RECOMPUTE SCOPE NESTS AND UNWINDS. It is what keeps a verification recompute from
// storing its own answer over the entry it is checking, so "is it still set after the scope
// ends" is a correctness question and not a style one.
void testTheRecomputeScopeNestsAndUnwinds() {
  testAssert(!NNCacheHitVerifier::inRecompute());
  {
    NNCacheHitVerifier::RecomputeScope outer;
    testAssert(NNCacheHitVerifier::inRecompute());
    {
      NNCacheHitVerifier::RecomputeScope inner;
      testAssert(NNCacheHitVerifier::inRecompute());
    }
    // The inner scope's exit must NOT clear the flag the outer one is holding.
    testAssert(NNCacheHitVerifier::inRecompute());
  }
  testAssert(!NNCacheHitVerifier::inRecompute());
  cout << "hit verify: the recompute scope nests, and an inner exit does not clear an outer "
          "scope's suppression" << endl;
}

//-------------------------------------------------------------------------------------
// The instrument: a checksum-VALID semantic corruption of a real container
//-------------------------------------------------------------------------------------

void overwriteBytesAt(const string& path, int64_t offset, const vector<uint8_t>& bytes) {
  FILE* f = fopen(path.c_str(), "r+b");
  if(f == NULL)
    throw StringError("nncachecorruptpayload: could not open for writing: " + path);
  if(fseek(f, (long)offset, SEEK_SET) != 0 || fwrite(bytes.data(), 1, bytes.size(), f) != bytes.size()) {
    fclose(f);
    throw StringError("nncachecorruptpayload: short write at offset " + Global::int64ToString(offset));
  }
  fclose(f);
}

vector<uint8_t> readBytesAt(const string& path, int64_t offset, size_t count) {
  vector<uint8_t> bytes(count);
  FILE* f = fopen(path.c_str(), "rb");
  if(f == NULL)
    throw StringError("nncachecorruptpayload: could not open for reading: " + path);
  if(fseek(f, (long)offset, SEEK_SET) != 0 || fread(bytes.data(), 1, count, f) != count) {
    fclose(f);
    throw StringError(
      "nncachecorruptpayload: short read of " + Global::uint64ToString((uint64_t)count) +
      " bytes at offset " + Global::int64ToString(offset) + " -- is this a container file?"
    );
  }
  fclose(f);
  return bytes;
}

uint32_t readU32(const uint8_t* p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
uint64_t readU64(const uint8_t* p) {
  uint64_t v = 0;
  for(int i = 0; i < 8; i++)
    v |= ((uint64_t)p[i]) << (8 * i);
  return v;
}
void writeU64(uint8_t* p, uint64_t v) {
  for(int i = 0; i < 8; i++)
    p[i] = (uint8_t)(v >> (8 * i));
}
float readF32(const uint8_t* p) {
  const uint32_t bits = readU32(p);
  float f;
  memcpy(&f, &bits, 4);
  return f;
}
void writeF32(uint8_t* p, float f) {
  uint32_t bits;
  memcpy(&bits, &f, 4);
  for(int i = 0; i < 4; i++)
    p[i] = (uint8_t)(bits >> (8 * i));
}

}  // namespace

void Tests::runNNCacheVerifyHitsTests() {
  cout << "Running NN cache hit verification tests" << endl;
  testIdenticalOutputsHoldWithNoDeviation();
  testAPerturbationInsideTheAllowanceHolds();
  testAValueChannelOutsideTheAllowanceIsReportedByName();
  testAPolicySlotOutsideTheAllowanceNamesItsIndex();
  testAnOwnershipSlotOutsideTheAllowanceIsCaught();
  testALostOwnershipMapIsAMismatchAndNotASkippedChannel();
  testANaNOnOneSideOnlyIsAMismatch();
  testTheAllowanceScalesWithTheChannelsOwnMagnitude();
  testTheSkipCountsAreSeparateFromTheVerifiedCount();
  testTheRecomputeScopeNestsAndUnwinds();
}

// FLIPS ONE PAYLOAD SCALAR OF ONE REAL, ALREADY-WRITTEN EVALUATION AND LEAVES THE FILE VALID.
//
// WHY IT MUST RE-MINT THE CHECKSUMS, which is the whole method: a corruption that left them
// stale would witness NOTHING. The block would be discarded at load as torn or corrupt, the
// engine would serve no such entry, and the verifier would never be reached -- the run would
// go green with the verifier deleted. A file whose checksums are VALID and whose MEANING is
// wrong is the only input that can reach the property under test. Same argument, same two
// checksums, same production functions as
// testnnevalcontainer.cpp's testEvalContainerAFileFromALaterVersionIsRefusedNotSilentlyRead.
//
// EVERY OFFSET IS DERIVED FROM THE FILE'S OWN HEADER, not from a constant restated here: the
// file header publishes its own length, the entry- and block-header strides, and the context
// hash both checksums are seeded with. So this cannot drift from the format the way a
// hand-copied layout would (ADR-0012 P7: one authoritative definition, every reader derives).
//
// WHICH SCALAR: whiteWinProb, the first f32 of the first entry of the first block. It is a
// value channel rather than a policy slot on purpose -- a value channel is what a search
// actually acts on, so a corruption there is the version of this defect that would do the most
// damage in the field, and it is the version this instrument should therefore be able to see.
void Tests::corruptFirstPersistedEvaluation(const string& containerPath) {
  // --- the file header, which describes its own framing ---
  const vector<uint8_t> fileHeaderPrefix = readBytesAt(containerPath, 0, 48);
  if(memcmp(fileHeaderPrefix.data(), "KGNNEVAL", 8) != 0)
    throw StringError("nncachecorruptpayload: not an eval container (bad magic): " + containerPath);
  const uint32_t fileHeaderBytes = readU32(fileHeaderPrefix.data() + 12);
  const uint32_t entryHeaderBytes = readU32(fileHeaderPrefix.data() + 16);
  const uint32_t blockHeaderBytes = readU32(fileHeaderPrefix.data() + 20);
  const uint64_t contextHash = readU64(fileHeaderPrefix.data() + 24);

  // --- the first block ---
  const int64_t blockStart = (int64_t)fileHeaderBytes;
  const vector<uint8_t> blockHeader = readBytesAt(containerPath, blockStart, blockHeaderBytes);
  const uint32_t entryCount = readU32(blockHeader.data() + 4);
  const uint64_t totalPayloadBytes = readU64(blockHeader.data() + 8);
  if(entryCount == 0)
    throw StringError("nncachecorruptpayload: the first block is empty; nothing to corrupt");

  const int64_t indexStart = blockStart + (int64_t)blockHeaderBytes;
  const int64_t payloadStart = indexStart + (int64_t)entryCount * (int64_t)entryHeaderBytes;

  // --- the first entry's key and payload location, out of its own header ---
  const vector<uint8_t> entryHeader = readBytesAt(containerPath, indexStart, entryHeaderBytes);
  const uint64_t keyHash0 = readU64(entryHeader.data() + 0);
  const uint64_t keyHash1 = readU64(entryHeader.data() + 8);
  const uint64_t payloadOffset = readU64(entryHeader.data() + 24);
  const int64_t scalarAt = payloadStart + (int64_t)payloadOffset;  // whiteWinProb is scalar 0

  // --- the corruption itself: one scalar, moved far outside any plausible tolerance ---
  vector<uint8_t> scalarBytes = readBytesAt(containerPath, scalarAt, 4);
  const float before = readF32(scalarBytes.data());
  // 0.25 is chosen to be enormous against the 1e-4 fp32 allowance AND still a legal winrate,
  // so nothing downstream refuses the file for being out of range -- the ONLY thing wrong with
  // it is that it no longer means what the net computed.
  const float after = before > 0.5f ? before - 0.25f : before + 0.25f;
  writeF32(scalarBytes.data(), after);
  overwriteBytesAt(containerPath, scalarAt, scalarBytes);

  // --- re-mint both checksums through the production function ---
  const int64_t regionBytes =
    (int64_t)entryCount * (int64_t)entryHeaderBytes + (int64_t)totalPayloadBytes;
  const vector<uint8_t> region = readBytesAt(containerPath, indexStart, (size_t)regionBytes);
  vector<uint8_t> entryChecksum(8);
  writeU64(entryChecksum.data(), NNCacheFileChecksum::of(region.data(), region.size(), contextHash));
  overwriteBytesAt(containerPath, blockStart + 16, entryChecksum);

  const vector<uint8_t> headerFirst24 = readBytesAt(containerPath, blockStart, 24);
  vector<uint8_t> headerChecksum(8);
  writeU64(headerChecksum.data(), NNCacheFileChecksum::of(headerFirst24.data(), headerFirst24.size(), contextHash));
  overwriteBytesAt(containerPath, blockStart + 24, headerChecksum);

  // SAYS EXACTLY WHAT IT DID, because the witness that reads this output has to be able to
  // assert against the key it corrupted rather than against "a mismatch happened somewhere".
  cout << "nncachecorruptpayload: " << containerPath << endl;
  cout << "  block at " << blockStart << " holds " << entryCount << " entries, "
       << totalPayloadBytes << " payload bytes" << endl;
  cout << "  corruptedKey " << Global::uint64ToHexString(keyHash0) << Global::uint64ToHexString(keyHash1) << endl;
  cout << "  channel whiteWinProb at file offset " << scalarAt
       << ": " << before << " -> " << after << endl;
  cout << "  both block checksums re-minted through NNCacheFileChecksum::of; "
          "the file is VALID and its meaning is wrong" << endl;
}

#endif  // KATAGO_NNCACHE_VERIFY_HITS
