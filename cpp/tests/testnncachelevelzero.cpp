#include "../tests/tests.h"

#include <algorithm>
#include <cstdio>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#ifdef __linux__
#include <unistd.h>
#endif
#if defined(__GLIBC__) && !defined(__UCLIBC__)
#include <malloc.h>
#endif

#include "../core/fileutils.h"
#include "../core/timer.h"
#include "../neuralnet/nncachecountlog.h"
#include "../neuralnet/nncachelevelzero.h"
#include "../neuralnet/nnevalcontainer.h"

using namespace std;
using namespace TestCommon;

// Correctness tests for the level-0 loader: the container-to-frozen-CHD attach, the
// count-log join that decides the build order, the selection bound, and the arena.
//
// Everything asserted here is a LOGIC INVARIANT and is asserted EXACTLY (ADR-0009's
// calibration): an order is an order, a key resolves or it does not, and a restored float is
// a value that was written and must come back as itself, so every float comparison is ==.
// The two figures that are MEASUREMENTS rather than invariants -- resident set size, and
// what the allocator gives back -- are printed, never asserted against a number, except for
// one deliberately loose bound whose looseness is stated where it is used.
//
// The load-bearing tests are the ordering one (including the absent-from-the-log case, which
// is the case a loader can get wrong in three different ways) and the release one (which
// observes that the storage went, not that a destructor ran).

namespace {

const char* MODEL = "kata1-b18c384nbt-s9732312320-d4245566942";
const int MODEL_VERSION = 14;
const char* CONTEXT = "card-5455";

// A distinct key per serial, spread over both halves.
Hash128 nthKey(int serial) {
  return Hash128(
    ((uint64_t)(serial + 1)) * 0x9E3779B97F4A7C15ULL,
    ((uint64_t)(serial + 1)) * 0xD6E8FEB86659FD93ULL + 0x1234567ULL
  );
}

// The value written into every policy slot beyond the board, so that a loader bringing it
// back would be unmistakable rather than plausible.
const float BEYOND_BOARD_SENTINEL = -12345.5f;

shared_ptr<NNOutput> makeOutput(int serial, int generation, int nnXLen, int nnYLen, bool withOwnerMap) {
  shared_ptr<NNOutput> out = make_shared<NNOutput>();
  out->nnHash = nthKey(serial);
  const float g = (float)generation;
  out->whiteWinProb = 0.125f * g + (float)serial * 0.0009765625f;
  out->whiteLossProb = 0.25f + g;
  out->whiteNoResultProb = 0.0f;
  out->whiteScoreMean = -3.5f * g;
  out->whiteScoreMeanSq = 17.25f + g;
  out->whiteLead = 1.75f * g;
  out->varTimeLeft = 42.5f;
  out->shorttermWinlossError = 0.03125f * g;
  out->shorttermScoreError = 0.0625f;
  out->policyOptimismUsed = 0.5f;
  out->nnXLen = nnXLen;
  out->nnYLen = nnYLen;

  const int area = nnXLen * nnYLen;
  for(int i = 0; i < NNPos::MAX_NN_POLICY_SIZE; i++)
    out->policyProbs[i] = BEYOND_BOARD_SENTINEL;
  for(int i = 0; i <= area; i++)
    out->policyProbs[i] = (i % 7 == 0) ? -1.0f : ((float)i * 0.5f + g + (float)serial);

  if(withOwnerMap) {
    out->whiteOwnerMap = new float[area];
    for(int i = 0; i < area; i++)
      out->whiteOwnerMap[i] = ((float)i * 0.015625f) - 1.0f + g + (float)serial;
  }
  return out;
}

// Asserts that `got` is bit-for-bit the evaluation `expected` was, over every field the
// format carries -- and that the policy slots BEYOND the board came back zeroed rather than
// carrying whatever the arena block held.
void assertSameEvaluation(const NNOutput& got, const NNOutput& expected) {
  testAssert(got.nnHash == expected.nnHash);
  testAssert(got.nnXLen == expected.nnXLen && got.nnYLen == expected.nnYLen);
  testAssert(got.whiteWinProb == expected.whiteWinProb);
  testAssert(got.whiteLossProb == expected.whiteLossProb);
  testAssert(got.whiteNoResultProb == expected.whiteNoResultProb);
  testAssert(got.whiteScoreMean == expected.whiteScoreMean);
  testAssert(got.whiteScoreMeanSq == expected.whiteScoreMeanSq);
  testAssert(got.whiteLead == expected.whiteLead);
  testAssert(got.varTimeLeft == expected.varTimeLeft);
  testAssert(got.shorttermWinlossError == expected.shorttermWinlossError);
  testAssert(got.shorttermScoreError == expected.shorttermScoreError);
  testAssert(got.policyOptimismUsed == expected.policyOptimismUsed);
  testAssert(got.noisedPolicyProbs == NULL);
  const int area = expected.nnXLen * expected.nnYLen;
  for(int i = 0; i <= area; i++)
    testAssert(got.policyProbs[i] == expected.policyProbs[i]);
  for(int i = area + 1; i < NNPos::MAX_NN_POLICY_SIZE; i++)
    testAssert(got.policyProbs[i] == 0.0f);
  testAssert((got.whiteOwnerMap != NULL) == (expected.whiteOwnerMap != NULL));
  if(expected.whiteOwnerMap != NULL) {
    for(int i = 0; i < area; i++)
      testAssert(got.whiteOwnerMap[i] == expected.whiteOwnerMap[i]);
  }
}

void writeContainerBlock(const string& dir, const vector<shared_ptr<const NNOutput>>& entries) {
  const NNEvalContainer container =
    NNEvalContainer::forContextAndModel(dir, CONTEXT, MODEL, MODEL_VERSION);
  const NNEvalContainerAppendResult r = container.appendBlock(entries);
  testAssert(r.bytesAppended > 0);
}

void writeCountDump(const string& dir, const vector<NNCacheHitCount>& rows) {
  const NNCacheCountLog log = NNCacheCountLog::forContext(dir, CONTEXT);
  const NNCacheCountLogAppendResult r = log.appendDump(NNCacheHitLedger::counted(rows, 0));
  testAssert(r.bytesAppended > 0);
}

NNCacheLevelZeroLoadRequest requestFor(const string& dir, NNCacheLevelZeroBound bound) {
  NNCacheLevelZeroLoadRequest request{dir, CONTEXT, MODEL, MODEL_VERSION, bound};
  return request;
}

NNCacheHitCount countRow(int serial, uint32_t hits) {
  NNCacheHitCount row;
  row.key = nthKey(serial);
  row.hits = hits;
  return row;
}

bool refused(const std::function<void()>& f, const string& mustSay) {
  try {
    f();
  }
  catch(const StringError& e) {
    const string what = e.what();
    if(what.find(mustSay) != string::npos)
      return true;
    cout << "Refused, but not for the stated reason: " << what << endl;
    return false;
  }
  return false;
}

// The resident set size of this process in bytes, or -1 where this build cannot read it.
//
// It is the figure the operator's complaint is denominated in -- "my instances grow to 10 GB"
// is an RSS statement -- so it is the figure the release test observes, rather than a
// destructor call or an internal counter that would agree with the code being tested by
// construction (ADR-0021 Rule 1).
// Returns the heap to the operating system, for the SETUP of a measurement only.
//
// It deliberately does NOT call nnCacheReclaimFreedHeap, even though that is exactly what
// that function does. A witness whose setup routes through the code it observes cannot go
// red: the first version of the release test below called the production function to
// establish its baseline, and the "the detach never trims" seen-red leg passed green,
// because the defect disabled the setup and the observation together and the confounder the
// setup existed to remove came straight back (ADR-0021 Rule 1: the witness observes at the
// site of the claim, through nothing that the claim is about).
void trimForMeasurementSetup() {
#if defined(__GLIBC__) && !defined(__UCLIBC__)
  (void)malloc_trim(0);
#endif
}

int64_t residentBytes() {
#ifdef __linux__
  FILE* f = fopen("/proc/self/statm", "r");
  if(f == NULL)
    return -1;
  long long totalPages = 0;
  long long residentPages = 0;
  const int got = fscanf(f, "%lld %lld", &totalPages, &residentPages);
  fclose(f);
  if(got != 2)
    return -1;
  return (int64_t)residentPages * (int64_t)sysconf(_SC_PAGESIZE);
#else
  return -1;
#endif
}

//-------------------------------------------------------------------------------------
// The round trip
//-------------------------------------------------------------------------------------

void testLevelZeroLoadsEveryKeyTheContainerHolds() {
  ScopedTempDir tmp("nnlevelzero-roundtrip");

  // Mixed shapes and mixed ownership-map carriage, because a level 0 must admit exactly what
  // a live cache admits and a restored entry must never be a de-scoped subset of a live one.
  vector<shared_ptr<const NNOutput>> written;
  vector<shared_ptr<NNOutput>> kept;
  for(int i = 0; i < 24; i++) {
    const bool ownerMap = (i % 3) != 0;
    const int edge = (i % 4 == 0) ? 9 : ((i % 4 == 1) ? 13 : 19);
    shared_ptr<NNOutput> out = makeOutput(i, 1, edge, edge, ownerMap);
    kept.push_back(out);
    written.push_back(out);
  }
  writeContainerBlock(tmp.path(), written);

  vector<NNCacheHitCount> rows;
  for(int i = 0; i < 24; i++)
    rows.push_back(countRow(i, (uint32_t)(100 - i)));
  writeCountDump(tmp.path(), rows);

  NNCacheLevelZeroLoad load = nnCacheLoadLevelZero(requestFor(tmp.path(), NNCacheLevelZeroBound::all()));
  testAssert(load.report.entriesInContainer == 24);
  testAssert(load.report.entriesInLevelZero == 24);
  testAssert(load.report.entriesCounted == 24);
  testAssert(load.report.entriesUncounted == 0);
  testAssert(load.report.entriesLeftOver == 0);
  testAssert(load.report.containerTail == NNEvalContainerTail::Intact);
  testAssert(load.report.countLogTail == NNCacheCountLogTail::Intact);

  // EVERY KEY THE CONTAINER HELD RESOLVES, and resolves to its own evaluation, byte for byte.
  for(size_t i = 0; i < kept.size(); i++) {
    shared_ptr<NNOutput> got;
    testAssert(load.levelZero->get(kept[i]->nnHash, got));
    testAssert(got != nullptr);
    assertSameEvaluation(*got, *kept[i]);
  }

  // And a key the container never held is absent rather than answered with a neighbour's
  // evaluation -- the absent-key contract, exercised through the loaded structure.
  {
    shared_ptr<NNOutput> got;
    testAssert(!load.levelZero->get(nthKey(9999), got));
    testAssert(got == nullptr);
  }

  const NNCacheLevelZeroRelease released = nnCacheReleaseLevelZero(std::move(load.levelZero));
  testAssert(released.storageReleased);
}

void testLevelZeroAppliesTheContainersOwnMergeRules() {
  ScopedTempDir tmp("nnlevelzero-merge");

  // Block 1: key 0 with an ownership map, key 1 without.
  shared_ptr<NNOutput> firstWithMap = makeOutput(0, 1, 19, 19, true);
  shared_ptr<NNOutput> firstNoMap = makeOutput(1, 1, 19, 19, false);
  writeContainerBlock(tmp.path(), {firstWithMap, firstNoMap});
  // Block 2: key 0 re-evaluated WITHOUT a map -- which must not erase the map -- and key 1
  // re-evaluated, which must win because nothing is lost by it.
  shared_ptr<NNOutput> secondNoMap = makeOutput(0, 2, 19, 19, false);
  shared_ptr<NNOutput> secondNoMapForOne = makeOutput(1, 2, 19, 19, false);
  writeContainerBlock(tmp.path(), {secondNoMap, secondNoMapForOne});

  writeCountDump(tmp.path(), {countRow(0, 5), countRow(1, 4)});

  NNCacheLevelZeroLoad load = nnCacheLoadLevelZero(requestFor(tmp.path(), NNCacheLevelZeroBound::all()));
  testAssert(load.report.entriesInContainer == 2);

  shared_ptr<NNOutput> gotZero;
  testAssert(load.levelZero->get(nthKey(0), gotZero));
  assertSameEvaluation(*gotZero, *firstWithMap);   // the ownership-map entry stood
  shared_ptr<NNOutput> gotOne;
  testAssert(load.levelZero->get(nthKey(1), gotOne));
  assertSameEvaluation(*gotOne, *secondNoMapForOne);  // last-wins where nothing is lost

  (void)nnCacheReleaseLevelZero(std::move(load.levelZero));
}

//-------------------------------------------------------------------------------------
// The count-log join: the build order
//-------------------------------------------------------------------------------------

void testTheBuildOrderIsDescendingLookupsThenTheUncountedByKey() {
  ScopedTempDir tmp("nnlevelzero-order");

  // Six keys. The count log mentions three of them -- one at a count of ZERO, which is a
  // different fact from not being mentioned -- and never mentions the other three, two of
  // which are there to make the tie-by-key rule observable.
  vector<shared_ptr<const NNOutput>> written;
  for(int i = 0; i < 6; i++)
    written.push_back(makeOutput(i, 1, 9, 9, false));
  writeContainerBlock(tmp.path(), written);
  writeCountDump(tmp.path(), {countRow(3, 2), countRow(1, 9), countRow(4, 0)});

  NNCacheLevelZeroLoad load = nnCacheLoadLevelZero(requestFor(tmp.path(), NNCacheLevelZeroBound::all()));
  testAssert(load.report.entriesInContainer == 6);
  testAssert(load.report.entriesInLevelZero == 6);
  // THE ABSENT-FROM-THE-LOG KEYS ARE KEPT, not dropped and not guessed at.
  testAssert(load.report.entriesCounted == 3);
  testAssert(load.report.entriesUncounted == 3);

  // The exact order, asserted position by position, because the order IS the deliverable:
  // key 1 (9 lookups), key 3 (2 lookups), key 4 (COUNTED AT ZERO -- which still outranks a
  // key the log never saw), then the three the log never mentioned, among themselves in key
  // order. The fixture's keys are hashes and are not monotone in their serial, so the last
  // three are stated as a sorted key set rather than as a serial order.
  vector<Hash128> expected;
  expected.push_back(nthKey(1));
  expected.push_back(nthKey(3));
  expected.push_back(nthKey(4));
  vector<Hash128> uncounted;
  uncounted.push_back(nthKey(0));
  uncounted.push_back(nthKey(2));
  uncounted.push_back(nthKey(5));
  std::sort(uncounted.begin(), uncounted.end());
  expected.insert(expected.end(), uncounted.begin(), uncounted.end());
  for(uint32_t i = 0; i < 6; i++)
    testAssert(load.levelZero->index().keyAt(i) == expected[i]);

  (void)nnCacheReleaseLevelZero(std::move(load.levelZero));
}

void testAKeyTheCountLogNeverSawIsKeptRatherThanDroppedOrGuessed() {
  ScopedTempDir tmp("nnlevelzero-uncounted");

  vector<shared_ptr<const NNOutput>> written;
  vector<shared_ptr<NNOutput>> kept;
  for(int i = 0; i < 5; i++) {
    shared_ptr<NNOutput> out = makeOutput(i, 1, 9, 9, i % 2 == 0);
    kept.push_back(out);
    written.push_back(out);
  }
  writeContainerBlock(tmp.path(), written);
  // A client that dumped "evaluations" without "counts" produces exactly this: no count log
  // at all. Every key is uncounted, and every key is still loaded and still resolves.
  NNCacheLevelZeroLoad load = nnCacheLoadLevelZero(requestFor(tmp.path(), NNCacheLevelZeroBound::all()));
  testAssert(load.report.entriesInContainer == 5);
  testAssert(load.report.entriesInLevelZero == 5);
  testAssert(load.report.entriesCounted == 0);
  testAssert(load.report.entriesUncounted == 5);
  for(size_t i = 0; i < kept.size(); i++) {
    shared_ptr<NNOutput> got;
    testAssert(load.levelZero->get(kept[i]->nnHash, got));
    assertSameEvaluation(*got, *kept[i]);
  }
  // In key order, since nothing distinguishes them but their keys.
  vector<Hash128> keys;
  for(int i = 0; i < 5; i++)
    keys.push_back(nthKey(i));
  std::sort(keys.begin(), keys.end());
  for(uint32_t i = 0; i < 5; i++)
    testAssert(load.levelZero->index().keyAt(i) == keys[i]);

  (void)nnCacheReleaseLevelZero(std::move(load.levelZero));
}

//-------------------------------------------------------------------------------------
// The selection bound
//-------------------------------------------------------------------------------------

void testTheSelectionBoundsTakeAPrefixInTheirOwnCurrency() {
  ScopedTempDir tmp("nnlevelzero-bound");

  vector<shared_ptr<const NNOutput>> written;
  for(int i = 0; i < 8; i++)
    written.push_back(makeOutput(i, 1, 9, 9, false));
  writeContainerBlock(tmp.path(), written);
  // Descending counts by serial: key 0 is the most looked-up. Keys 6 and 7 are uncounted.
  vector<NNCacheHitCount> rows;
  for(int i = 0; i < 6; i++)
    rows.push_back(countRow(i, (uint32_t)(60 - 10 * i)));
  writeCountDump(tmp.path(), rows);

  {
    NNCacheLevelZeroLoad load =
      nnCacheLoadLevelZero(requestFor(tmp.path(), NNCacheLevelZeroBound::maxEntries(3)));
    testAssert(load.report.entriesInLevelZero == 3);
    testAssert(load.report.entriesLeftOver == 5);
    for(uint32_t i = 0; i < 3; i++)
      testAssert(load.levelZero->index().keyAt(i) == nthKey((int)i));
    // The remainder is the rest, still in build order -- what a level-1 fill takes next.
    testAssert(load.remainder.size() == 5);
    testAssert(load.remainder[0].key == nthKey(3));
    testAssert(load.remainder[4].key == nthKey(7) || load.remainder[4].key == nthKey(6));
    (void)nnCacheReleaseLevelZero(std::move(load.levelZero));
  }
  {
    // At least 30 lookups: keys 0, 1, 2, 3 (60, 50, 40, 30). An UNCOUNTED key is never
    // admitted by a threshold above zero: its count is not zero, it is unknown.
    NNCacheLevelZeroLoad load =
      nnCacheLoadLevelZero(requestFor(tmp.path(), NNCacheLevelZeroBound::minLookups(30)));
    testAssert(load.report.entriesInLevelZero == 4);
    for(uint32_t i = 0; i < 4; i++)
      testAssert(load.levelZero->index().keyAt(i) == nthKey((int)i));
    (void)nnCacheReleaseLevelZero(std::move(load.levelZero));
  }
  {
    // The byte bound is denominated in RESIDENT bytes -- an NNOutput plus its ownership map
    // -- so the number a caller passes is arithmetic on sizeof(NNOutput), never on the file.
    const int64_t perEntry = (int64_t)sizeof(NNOutput);
    NNCacheLevelZeroLoad load =
      nnCacheLoadLevelZero(requestFor(tmp.path(), NNCacheLevelZeroBound::maxBytes(perEntry * 2 + perEntry / 2)));
    testAssert(load.report.entriesInLevelZero == 2);
    testAssert(load.report.arenaEvaluationBytes == perEntry * 2);
    testAssert(load.report.arenaOwnerMapBytes == 0);
    (void)nnCacheReleaseLevelZero(std::move(load.levelZero));
  }
  {
    // AN UNCOUNTED KEY IS NOT ADMITTED BY A THRESHOLD ABOVE ZERO. Keys 6 and 7 are not in the
    // count log; every key that is has at least 10 lookups, so a threshold of 10 would admit
    // all six counted keys and then meet the two uncounted ones. It stops.
    NNCacheLevelZeroLoad load =
      nnCacheLoadLevelZero(requestFor(tmp.path(), NNCacheLevelZeroBound::minLookups(10)));
    testAssert(load.report.entriesInLevelZero == 6);
    testAssert(load.report.entriesCounted == 6);
    testAssert(load.report.entriesUncounted == 2);
    testAssert(load.remainder.size() == 2);
    testAssert(!load.remainder[0].counted && !load.remainder[1].counted);
    (void)nnCacheReleaseLevelZero(std::move(load.levelZero));
  }
  {
    NNCacheLevelZeroLoad load =
      nnCacheLoadLevelZero(requestFor(tmp.path(), NNCacheLevelZeroBound::maxEntries(0)));
    testAssert(load.report.entriesInLevelZero == 0);
    testAssert(load.levelZero->numEntries() == 0);
    shared_ptr<NNOutput> got;
    testAssert(!load.levelZero->get(nthKey(0), got));
    (void)nnCacheReleaseLevelZero(std::move(load.levelZero));
  }
}

//-------------------------------------------------------------------------------------
// The arena
//-------------------------------------------------------------------------------------

void testTheEvaluationsLiveInTwoContiguousBlocksAndNotInNinetyThousandAllocations() {
  ScopedTempDir tmp("nnlevelzero-arena");

  vector<shared_ptr<const NNOutput>> written;
  for(int i = 0; i < 40; i++)
    written.push_back(makeOutput(i, 1, 19, 19, i % 2 == 0));
  writeContainerBlock(tmp.path(), written);

  NNCacheLevelZeroLoad load = nnCacheLoadLevelZero(requestFor(tmp.path(), NNCacheLevelZeroBound::all()));
  testAssert(load.report.entriesInLevelZero == 40);

  // CONTIGUITY, observed as contiguity rather than trusted from the allocation call:
  // consecutive entries are consecutive NNOutputs of one block, and every ownership map
  // points inside one float block, in ascending order and without overlap.
  const NNCacheFrozenIndex& index = load.levelZero->index();
  const NNOutput* base = index.payloadAt(0);
  for(uint32_t i = 0; i < 40; i++)
    testAssert(index.payloadAt(i) == base + i);

  // The ownership maps TILE ONE BLOCK EXACTLY: sorted by address they are end-to-end with no
  // gap and no overlap, which is the property "one contiguous block" actually means. They are
  // carved in the order the file is READ, which is ascending file offset rather than index
  // order, so the check sorts rather than assuming the two coincide.
  vector<const float*> maps;
  int64_t mapFloats = 0;
  for(uint32_t i = 0; i < 40; i++) {
    const NNOutput* out = index.payloadAt(i);
    if(out->whiteOwnerMap == NULL)
      continue;
    maps.push_back(out->whiteOwnerMap);
    mapFloats += out->nnXLen * out->nnYLen;
  }
  testAssert((int)maps.size() == 20);
  testAssert(mapFloats == 20 * 19 * 19);
  std::sort(maps.begin(), maps.end());
  for(size_t i = 1; i < maps.size(); i++)
    testAssert(maps[i] == maps[i - 1] + 19 * 19);

  // The two blocks are exactly the payload the structure reports holding: no per-entry
  // overhead hiding in the arithmetic.
  testAssert(load.report.arenaEvaluationBytes == (int64_t)(40 * sizeof(NNOutput)));
  testAssert(load.report.arenaOwnerMapBytes == (int64_t)(20 * 19 * 19 * sizeof(float)));
  testAssert(load.report.arenaTotalBytes == load.levelZero->reachablePayloadBytes());

  (void)nnCacheReleaseLevelZero(std::move(load.levelZero));
}

void testAnArenaBackedLevelZeroStaysUnderTheMeasuredStructureCeiling() {
  ScopedTempDir tmp("nnlevelzero-ceiling");

  const int n = 4096;
  vector<shared_ptr<const NNOutput>> written;
  for(int i = 0; i < n; i++)
    written.push_back(makeOutput(i, 1, 9, 9, false));
  writeContainerBlock(tmp.path(), written);

  NNCacheLevelZeroLoad load = nnCacheLoadLevelZero(requestFor(tmp.path(), NNCacheLevelZeroBound::all()));
  const double perEntry = (double)load.report.levelZeroStructureBytes / (double)n;
  // The published ceiling for this branch's level-0 structure. An arena-backed level 0 is
  // BELOW it, by the 8 bytes per entry of unique_ptr the heap store spends and the arena
  // does not -- so this asserts the ceiling holds and prints where it actually landed.
  testAssert(perEntry <= 44.9);
  cout << "level-0 loader: arena-backed structure " << perEntry << " B/entry at n = " << n
       << " (ceiling 44.9; index alone "
       << ((double)load.levelZero->index().structureBytes() / (double)n) << ")" << endl;

  (void)nnCacheReleaseLevelZero(std::move(load.levelZero));
}

//-------------------------------------------------------------------------------------
// The release: the arena does not outlive the attach span
//-------------------------------------------------------------------------------------

void testTheArenaIsReleasedAtDetachAndNotOneMomentLater() {
  ScopedTempDir tmp("nnlevelzero-release");

  // Large enough that the arena is a real allocation and its going is visible in the
  // process's own resident set rather than lost in noise: 15,000 19x19 entries with
  // ownership maps is ~44 MB of payload.
  const int n = 15000;
  {
    vector<shared_ptr<const NNOutput>> written;
    written.reserve(n);
    for(int i = 0; i < n; i++)
      written.push_back(makeOutput(i, 1, 19, 19, true));
    writeContainerBlock(tmp.path(), written);
  }

  // TRIM BEFORE THE BASELINE, and this is method rather than tidiness. Writing that container
  // allocated and freed ~45 MiB of its own, and glibc does not return a free to the operating
  // system: an untrimmed baseline is therefore already carrying that free pool, the attach
  // below reuses it instead of growing the process, and the whole observation collapses to
  // noise. It collapsed exactly that way on the first attempt, and the missing-trim seen-red
  // leg passed green against it (ADR-0021 Rule 4: a green that cannot go red witnesses
  // nothing). The trim here is the test's OWN call and not the loader's -- see
  // trimForMeasurementSetup for why that distinction is the difference between a witness
  // and a tautology.
  trimForMeasurementSetup();
  const int64_t baseline = residentBytes();

  NNCacheLevelZeroLoad load = nnCacheLoadLevelZero(requestFor(tmp.path(), NNCacheLevelZeroBound::all()));
  testAssert(load.report.entriesInLevelZero == n);
  const int64_t arenaBytes = load.report.arenaTotalBytes;
  const int64_t afterAttach = residentBytes();

  // THE NEGATIVE LEG, at the site of the claim. A caller holding one evaluation the level 0
  // handed out holds the WHOLE arena, because a get returns through shared_ptr's aliasing
  // constructor against the single store. So a release under an outstanding handle must
  // report that the storage did NOT go -- and if it reported success here, the "released at
  // detach" claim would be a claim about this function's own handle and not about the memory.
  {
    NNCacheLevelZeroLoad held =
      nnCacheLoadLevelZero(requestFor(tmp.path(), NNCacheLevelZeroBound::maxEntries(64)));
    std::weak_ptr<NNCacheEvaluationStore> watch = held.levelZero->evaluationStore();
    shared_ptr<NNOutput> outstanding;
    testAssert(held.levelZero->numEntries() == 64);
    testAssert(held.levelZero->get(held.levelZero->index().keyAt(0), outstanding));
    testAssert(outstanding != nullptr);
    const NNCacheLevelZeroRelease blocked = nnCacheReleaseLevelZero(std::move(held.levelZero));
    testAssert(!blocked.storageReleased);
    testAssert(!watch.expired());
    // And when the handle goes, so does the storage -- observed on the same weak reference,
    // so the red and the green are the same observation of the same property.
    outstanding.reset();
    testAssert(watch.expired());
  }

  // The green leg. Nothing outstanding, so the storage goes.
  std::weak_ptr<NNCacheEvaluationStore> watch = load.levelZero->evaluationStore();
  const NNCacheLevelZeroRelease released = nnCacheReleaseLevelZero(std::move(load.levelZero));
  testAssert(released.storageReleased);
  testAssert(watch.expired());
  const int64_t afterDetach = residentBytes();

  if(baseline >= 0 && afterAttach >= 0 && afterDetach >= 0) {
    // THE PROPERTY, IN THE CURRENCY THE COMPLAINT IS MADE IN. The attach must be visible in
    // the process's resident set and the detach must give it back. The bound is deliberately
    // loose -- half the arena either way -- because resident set size is a measurement and
    // the page cache, the allocator's own bookkeeping and the container read all move it;
    // what is being asserted is that tens of megabytes appear and then leave, not a figure.
    testAssert(afterAttach - baseline > arenaBytes / 2);
    // THE DETACH GAVE THE PAGES BACK. Against a trimmed baseline this is the whole claim:
    // the process is within a quarter of the arena of where it started, having held all of
    // it a moment ago. The quarter is slack for the allocator's own bookkeeping and for the
    // key set and remainder this scope still holds, not a hedge -- the figure it is
    // distinguishing between is zero returned and all of it returned.
    testAssert(afterDetach - baseline < arenaBytes / 4);
    cout << "level-0 loader: RSS baseline " << (baseline / 1048576) << " MiB, after attach of "
         << (arenaBytes / 1048576) << " MiB arena " << (afterAttach / 1048576)
         << " MiB, after detach " << (afterDetach / 1048576) << " MiB" << endl;
  }
  else {
    cout << "level-0 loader: UNEXERCISED -- this build cannot read its own resident set size, "
            "so the release was witnessed only through the store's own weak reference" << endl;
  }
}

// A MEASUREMENT, not an assertion (ADR-0009): what the allocator actually gives back at a
// detach on this box, and what the trim call itself contributes on top of the release.
//
// It is separated from the test above because it must split an act that
// nnCacheReleaseLevelZero deliberately performs as one -- release, then trim -- to attribute
// the two. Nothing here is asserted against a number; the figures are printed so a report can
// quote a measurement made on the machine it is reporting about rather than a figure
// inherited from somewhere else.
void measureWhatTheDetachGivesBack() {
  ScopedTempDir tmp("nnlevelzero-reclaim");

  const int n = 15000;
  {
    vector<shared_ptr<const NNOutput>> written;
    written.reserve(n);
    for(int i = 0; i < n; i++)
      written.push_back(makeOutput(i, 1, 19, 19, true));
    writeContainerBlock(tmp.path(), written);
  }

  // TRIM FIRST, so that what is measured below is this attach and this detach and not the
  // heap every test before this one left grown. Without it the trim's figure would be the
  // whole suite's accumulated free lists attributed to one detach, which is the kind of
  // number that gets quoted for years (ADR-0009: the method is part of the claim).
  trimForMeasurementSetup();
  const int64_t baseline = residentBytes();
  NNCacheLevelZeroLoad load = nnCacheLoadLevelZero(requestFor(tmp.path(), NNCacheLevelZeroBound::all()));
  const int64_t afterAttach = residentBytes();
  load.levelZero.reset();
  load.remainder.clear();
  const int64_t afterFree = residentBytes();
  ClockTimer trimTimer;
  const NNCacheHeapReclaim reclaim = nnCacheReclaimFreedHeap();
  const double trimMs = trimTimer.getSeconds() * 1000.0;
  const int64_t afterTrim = residentBytes();

  const char* reclaimName =
    reclaim == NNCacheHeapReclaim::Trimmed ? "Trimmed" :
    (reclaim == NNCacheHeapReclaim::NothingToTrim ? "NothingToTrim" : "Unavailable");
  if(baseline < 0) {
    cout << "level-0 loader reclaim: UNEXERCISED -- this build cannot read its own resident "
            "set size. The reclaim reported " << reclaimName << " in " << trimMs << " ms." << endl;
    return;
  }
  cout << "level-0 loader reclaim measurement (this box, one run, RSS from /proc/self/statm): "
       << "arena " << (load.report.arenaTotalBytes / 1048576) << " MiB; RSS baseline "
       << (baseline / 1024) << " KiB, after attach " << (afterAttach / 1024)
       << " KiB, after release " << (afterFree / 1024)
       << " KiB, after malloc_trim " << (afterTrim / 1024) << " KiB; release returned "
       << ((afterAttach - afterFree) / 1024) << " KiB, malloc_trim returned a further "
       << ((afterFree - afterTrim) / 1024) << " KiB in " << trimMs
       << " ms and reported " << reclaimName << endl;
}

//-------------------------------------------------------------------------------------
// The refusals
//-------------------------------------------------------------------------------------

void testTheLoaderRefusesWhatItCannotHonor() {
  ScopedTempDir tmp("nnlevelzero-refusals");

  vector<shared_ptr<const NNOutput>> written;
  for(int i = 0; i < 4; i++)
    written.push_back(makeOutput(i, 1, 9, 9, false));
  writeContainerBlock(tmp.path(), written);
  writeCountDump(tmp.path(), {countRow(0, 3)});

  // A container is bound to ONE model, and the cache key names no net, so attaching under
  // another model's name must refuse rather than serve one net's evaluations as another's.
  // The file is per-(context, model), so the wrong model simply finds no file -- which is an
  // empty level 0, not a wrong one. The refusal that matters is the one against a file whose
  // HEADER names another model, which is what a renamed or hand-copied file looks like.
  {
    const string wrongPath = tmp.path() + "/" + CONTEXT + ".othernet-v1.nnevals";
    const string bytes = FileUtils::readFileBinary(tmp.path() + "/" + CONTEXT + "." + MODEL + ".nnevals");
    FILE* w = fopen(wrongPath.c_str(), "wb");
    testAssert(w != NULL);
    testAssert(fwrite(bytes.data(), 1, bytes.size(), w) == bytes.size());
    fclose(w);
    NNCacheLevelZeroLoadRequest request{tmp.path(), CONTEXT, "othernet-v1", MODEL_VERSION,
                                        NNCacheLevelZeroBound::all()};
    testAssert(refused([&]() { (void)nnCacheLoadLevelZero(request); },
                       "one net's evaluations are never read as another's"));
  }

  // A context name that is not in the closed path-component alphabet never reaches a path.
  {
    NNCacheLevelZeroLoadRequest request{tmp.path(), "../escape", MODEL, MODEL_VERSION,
                                        NNCacheLevelZeroBound::all()};
    testAssert(refused([&]() { (void)nnCacheLoadLevelZero(request); }, "context name"));
  }

  // An absent container is a normal answer: an empty level 0, distinguishable in the report
  // from a container that exists and is empty.
  {
    NNCacheLevelZeroLoadRequest request{tmp.path(), "card-never-dumped", MODEL, MODEL_VERSION,
                                        NNCacheLevelZeroBound::all()};
    NNCacheLevelZeroLoad load = nnCacheLoadLevelZero(request);
    testAssert(load.report.entriesInContainer == 0);
    testAssert(load.report.entriesInLevelZero == 0);
    testAssert(load.report.containerTail == NNEvalContainerTail::Intact);
    shared_ptr<NNOutput> got;
    testAssert(!load.levelZero->get(nthKey(0), got));
    (void)nnCacheReleaseLevelZero(std::move(load.levelZero));
  }

  // A bound whose number has no reading is refused where it is constructed, not where it is
  // applied.
  testAssert(refused([&]() { (void)NNCacheLevelZeroBound::maxEntries(-1); }, "no reading"));
  testAssert(refused([&]() { (void)NNCacheLevelZeroBound::maxBytes(-1); }, "no reading"));
}

void testAContainerRewrittenBetweenItsKeySetAndItsEntriesIsRefused() {
  ScopedTempDir tmp("nnlevelzero-changed");

  // Two blocks, so that a compaction moves entries rather than leaving them where they were.
  vector<shared_ptr<const NNOutput>> firstBlock;
  for(int i = 0; i < 6; i++)
    firstBlock.push_back(makeOutput(i, 1, 9, 9, false));
  writeContainerBlock(tmp.path(), firstBlock);
  vector<shared_ptr<const NNOutput>> secondBlock;
  for(int i = 6; i < 12; i++)
    secondBlock.push_back(makeOutput(i, 1, 9, 9, false));
  writeContainerBlock(tmp.path(), secondBlock);

  const NNEvalContainer container =
    NNEvalContainer::forContextAndModel(tmp.path(), CONTEXT, MODEL, MODEL_VERSION);
  const NNEvalContainerIndex index = container.loadIndex();
  testAssert((int)index.entries().size() == 12);

  // Now the file is rewritten under the caller: one block instead of two, so the second
  // block's entries are at different offsets. Reading the stale locations must REFUSE, not
  // hand back whatever now lives at those offsets.
  const NNEvalContainerContents compacted = container.compact();
  testAssert((int)compacted.entries().size() == 12);

  const std::shared_ptr<NNCacheLevelZeroArena> arena =
    NNCacheLevelZeroArena::reserve(index.entries().size(), 0);
  testAssert(refused(
    [&]() { container.readEntriesInto(index.entries(), *arena); },
    "changed between reading its key set and reading its entries"
  ));
}

void testATornContainerAttachesTheIntactPrefixAndSaysSo() {
  ScopedTempDir tmp("nnlevelzero-torn");

  vector<shared_ptr<const NNOutput>> firstBlock;
  vector<shared_ptr<NNOutput>> kept;
  for(int i = 0; i < 5; i++) {
    shared_ptr<NNOutput> out = makeOutput(i, 1, 9, 9, false);
    kept.push_back(out);
    firstBlock.push_back(out);
  }
  writeContainerBlock(tmp.path(), firstBlock);
  vector<shared_ptr<const NNOutput>> secondBlock;
  for(int i = 5; i < 10; i++)
    secondBlock.push_back(makeOutput(i, 1, 9, 9, false));
  writeContainerBlock(tmp.path(), secondBlock);

  // Chop the file mid-second-block, the shape a crash during a dump leaves.
  const string path = tmp.path() + "/" + CONTEXT + "." + MODEL + ".nnevals";
  const int64_t secondBlockBytes =
    (int64_t)NNEvalContainer::blockHeaderBytes() + 5 * NNEvalContainer::bytesForEntry(9, 9, false);
  {
    const string all = FileUtils::readFileBinary(path);
    const size_t keep = (size_t)((int64_t)all.size() - secondBlockBytes / 2);
    testAssert(keep < all.size());
    FILE* w = fopen(path.c_str(), "wb");
    testAssert(w != NULL);
    testAssert(fwrite(all.data(), 1, keep, w) == keep);
    fclose(w);
  }

  NNCacheLevelZeroLoad load = nnCacheLoadLevelZero(requestFor(tmp.path(), NNCacheLevelZeroBound::all()));
  testAssert(load.report.containerTail == NNEvalContainerTail::Truncated);
  testAssert(load.report.containerDiscardedTailBytes > 0);
  testAssert(load.report.entriesInLevelZero == 5);
  for(size_t i = 0; i < kept.size(); i++) {
    shared_ptr<NNOutput> got;
    testAssert(load.levelZero->get(kept[i]->nnHash, got));
    assertSameEvaluation(*got, *kept[i]);
  }
  (void)nnCacheReleaseLevelZero(std::move(load.levelZero));
}

}  // namespace

void Tests::runNNCacheLevelZeroTests() {
  cout << "Running nn cache level-0 loader tests" << endl;

  testLevelZeroLoadsEveryKeyTheContainerHolds();
  testLevelZeroAppliesTheContainersOwnMergeRules();
  testTheBuildOrderIsDescendingLookupsThenTheUncountedByKey();
  testAKeyTheCountLogNeverSawIsKeptRatherThanDroppedOrGuessed();
  testTheSelectionBoundsTakeAPrefixInTheirOwnCurrency();
  testTheEvaluationsLiveInTwoContiguousBlocksAndNotInNinetyThousandAllocations();
  testAnArenaBackedLevelZeroStaysUnderTheMeasuredStructureCeiling();
  testTheArenaIsReleasedAtDetachAndNotOneMomentLater();
  testTheLoaderRefusesWhatItCannotHonor();
  testAContainerRewrittenBetweenItsKeySetAndItsEntriesIsRefused();
  testATornContainerAttachesTheIntactPrefixAndSaysSo();

  measureWhatTheDetachGivesBack();
}
