#include "../tests/tests.h"

#include <cstdio>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../core/fileutils.h"
#include "../neuralnet/nncache.h"
#include "../neuralnet/nncachecontext.h"
#include "../neuralnet/nncachecountlog.h"
#include "../neuralnet/nncachedump.h"
#include "../neuralnet/nncachefrozen.h"
#include "../neuralnet/nnevalcontainer.h"

using namespace std;
using namespace TestCommon;

// THE WRITE SIDE OF THE CACHE PROTOCOL: the disk admission predicate, the persisted mark, and
// the count delta.
//
// THE CLAIM UNDER TEST is three sentences, and each is witnessed at the surface on which it
// is a fact rather than at whichever surface is cheapest to read (ADR-0021 Rule 1):
//
//   AN ENTRY BELOW THE ADMISSION THRESHOLD IS NOT ON DISK. Witnessed against THE FILE -- the
//   container is loaded back from the filesystem and its key set inspected -- and never
//   against the plan the planner just produced. Reading the planner's own output back would
//   witness that a vector round-trips.
//
//   AN ENTRY WHOSE BYTES ARE ALREADY ON DISK IS NOT WRITTEN AGAIN. Three transitions, three
//   witnesses: set on fill-from-container, set on dump, CLEARED on a live overwrite. The
//   third is the one that matters most and is the one a naive "have I seen this key" cache
//   would get wrong: an ownership-map upgrade is a live overwrite, so it must re-persist.
//
//   A SECOND HARVEST WITH NOTHING IN BETWEEN YIELDS NOTHING. Witnessed twice, because there
//   are two harvests and they fail differently: the evaluation dump must leave the container
//   BYTE-IDENTICAL, and the count take must leave every key's lookups and sessions in the
//   count log unmoved.
//
// WHAT IS NOT WITNESSED HERE, and why. The cache_dump action does not exist -- it is a later
// increment -- so nothing reaches these seams over the protocol. The composed session shape
// (attach, analyze, dump, detach, re-attach, dump) is exercised at the LIBRARY seam below,
// against real files, with the table rebuilt between the two attachments exactly as a detach
// and a re-attach would rebuild it; the protocol-level version of the same walk belongs to
// the action increment and to the end-to-end test.

namespace {

//-------------------------------------------------------------------------------------
// Fixtures
//-------------------------------------------------------------------------------------

const char* const TMP_DIR_PREFIX = "tmpnncachedump";
const char* const MODEL = "kata1-b18c384nbt-s9732312320-d4245566942";
const int MODEL_VERSION = 14;
const int XLEN = 9;
const int YLEN = 9;

Hash128 nthKey(int serial) {
  return Hash128(
    ((uint64_t)(serial + 1)) * 0x9E3779B97F4A7C15ULL,
    ((uint64_t)(serial + 1)) * 0xD6E8FEB86659FD93ULL + 0x1234567ULL
  );
}

// One evaluation, filled from (serial, generation) so two evaluations of the SAME key from
// different moments are distinguishable in the file -- which is what makes "the later one
// superseded the earlier" observable rather than inferred.
shared_ptr<NNOutput> makeOutput(int serial, int generation, bool withOwnerMap) {
  shared_ptr<NNOutput> out = make_shared<NNOutput>();
  out->nnHash = nthKey(serial);
  const float g = (float)generation;
  out->whiteWinProb = 0.125f * g;
  out->whiteLossProb = 0.25f + g;
  out->whiteNoResultProb = 0.0f;
  out->whiteScoreMean = -3.5f * g;
  out->whiteScoreMeanSq = 17.25f + g;
  out->whiteLead = 1.75f * g;
  out->varTimeLeft = 42.5f;
  out->shorttermWinlossError = 0.03125f * g;
  out->shorttermScoreError = 0.0625f;
  out->policyOptimismUsed = 0.5f;
  out->nnXLen = XLEN;
  out->nnYLen = YLEN;
  const int area = XLEN * YLEN;
  for(int i = 0; i < NNPos::MAX_NN_POLICY_SIZE; i++)
    out->policyProbs[i] = 0.0f;
  for(int i = 0; i <= area; i++)
    out->policyProbs[i] = (i % 7 == 0) ? -1.0f : ((float)i * 0.5f + g);
  if(withOwnerMap) {
    out->whiteOwnerMap = new float[area];
    for(int i = 0; i < area; i++)
      out->whiteOwnerMap[i] = (float)i * 0.03125f - g;
  }
  return out;
}

unique_ptr<NNOutput> makeOwnedOutput(int serial, int generation) {
  shared_ptr<NNOutput> p = makeOutput(serial, generation, false);
  return unique_ptr<NNOutput>(new NNOutput(*p));
}

unique_ptr<NNCacheTable> defaultLevelOne(int sizePowerOfTwo) {
  return NNCacheTable::create(NNCacheConfig::statusQuo(sizePowerOfTwo, 2));
}

// A level 1 that DECLINES a key on its first sighting. The shape that makes the attribution
// ledger a strict superset of what the table holds, which is what the notResident figure is a
// measurement of.
unique_ptr<NNCacheTable> secondSightingLevelOne(int sizePowerOfTwo) {
  NNCacheConfig config = NNCacheConfig::statusQuo(sizePowerOfTwo, 2);
  config.admission = NNCacheAdmissionPolicy::SecondSighting;
  return NNCacheTable::create(config);
}

// A two-level table over a frozen level 0 built from `zeroSerials`. This is the shape a
// session that attached anything actually runs, and it is the only shape that counts.
unique_ptr<NNCacheTable> twoLevelTableOver(const vector<int>& zeroSerials) {
  vector<unique_ptr<NNOutput>> zero;
  for(size_t i = 0; i < zeroSerials.size(); i++)
    zero.push_back(makeOwnedOutput(zeroSerials[i], 1));
  return makeTwoLevelNNCacheTable(NNCacheFrozen::build(std::move(zero)), defaultLevelOne(10), 10);
}

bool containerHasKey(const NNEvalContainerIndex& index, Hash128 key) {
  for(size_t i = 0; i < index.entries().size(); i++) {
    if(index.entries()[i].key == key)
      return true;
  }
  return false;
}

bool containerEntryHasOwnerMap(const NNEvalContainerIndex& index, Hash128 key) {
  for(size_t i = 0; i < index.entries().size(); i++) {
    if(index.entries()[i].key == key)
      return index.entries()[i].hasOwnerMap;
  }
  testAssert(false);
  return false;
}

// The file's exact bytes, so "unchanged" can mean unchanged rather than "the same live key
// set", which an appended duplicate block would also satisfy.
string fileContents(const string& path) {
  string contents;
  if(!FileUtils::exists(path))
    return string();
  FileUtils::loadFileIntoString(path, string(), contents);
  return contents;
}

int64_t fileBytes(const string& path) {
  if(!FileUtils::exists(path))
    return -1;
  return (int64_t)fileContents(path).size();
}

// The count log's row for a key, or a refusal. A named helper so a missing row is a test
// failure rather than a silent zero standing in for one.
NNCacheCountRow logRowFor(const NNCacheCountLogContents& contents, Hash128 key) {
  for(size_t i = 0; i < contents.rows().size(); i++) {
    if(contents.rows()[i].key == key)
      return contents.rows()[i];
  }
  NNCacheCountRow absent;
  absent.key = key;
  absent.lookups = 0;
  absent.sessions = 0;
  return absent;
}

bool logHasKey(const NNCacheCountLogContents& contents, Hash128 key) {
  for(size_t i = 0; i < contents.rows().size(); i++) {
    if(contents.rows()[i].key == key)
      return true;
  }
  return false;
}

vector<NNCacheCountRow> observations(const vector<pair<int, uint64_t> >& serialAndLookups) {
  vector<NNCacheCountRow> out;
  for(size_t i = 0; i < serialAndLookups.size(); i++) {
    NNCacheCountRow row;
    row.key = nthKey(serialAndLookups[i].first);
    row.lookups = serialAndLookups[i].second;
    row.sessions = 1;
    out.push_back(row);
  }
  return out;
}

bool refused(const std::function<void()>& f, const string& mustMention) {
  try {
    f();
  }
  catch(const StringError& e) {
    testAssert(string(e.what()).find(mustMention) != string::npos);
    return true;
  }
  return false;
}

//-------------------------------------------------------------------------------------
// The shared threshold: one home for "seen often enough"
//-------------------------------------------------------------------------------------

// The read side's level-0 bound and the write side's disk admission ask ONE question of a
// key. This asserts they answer it identically across the whole boundary -- including the
// case that makes the rule non-obvious, a key the count log has never mentioned -- so a
// future edit to one that does not reach the other fails here rather than in a card that
// quietly stops being loadable at the size it is dumped at.
void testTheReadAndWriteSidesShareOneLookupThresholdRule() {
  for(uint64_t threshold = 0; threshold <= 4; threshold++) {
    const NNCacheLookupThreshold shared = NNCacheLookupThreshold::of(threshold);
    const NNCacheDiskAdmission write = NNCacheDiskAdmission::minLookups(threshold);
    for(uint64_t recorded = 0; recorded <= 6; recorded++)
      testAssert(write.admits(recorded) == shared.admits(recorded));

    // The boundary case in full: a key the log never mentioned is passed as zero, so it is
    // admitted by a threshold of zero and by no other. "Admit everything" must not silently
    // become "admit everything the log happens to know about".
    const uint64_t uncounted = 0;
    testAssert(shared.admits(uncounted) == (threshold == 0));
  }
  // all() is not minLookups(0) wearing a different name, but it must agree with it, because
  // a client writing either means the same thing.
  for(uint64_t recorded = 0; recorded <= 3; recorded++)
    testAssert(NNCacheDiskAdmission::all().admits(recorded));
  testAssert(NNCacheDiskAdmission::minLookups(2).describe().find("2") != string::npos);
}

//-------------------------------------------------------------------------------------
// Claim 1: the admission threshold, witnessed against the file
//-------------------------------------------------------------------------------------

// Three keys, recorded at 0, 1 and 2 lookups. Under minLookups(2) exactly one of them may
// reach the file, and the observation point is the file -- loaded back off the filesystem
// after the dump returned, not the plan the dump produced.
void testAnEntryBelowTheAdmissionThresholdIsAbsentFromTheWrittenContainer() {
  ScopedTempDir dir(TMP_DIR_PREFIX);
  unique_ptr<NNCacheTable> table = twoLevelTableOver(vector<int>());
  const NNCacheContextId card = table->attachCacheContext("card-5455");

  const int seenTwice = 1;
  const int seenOnce = 2;
  const int neverCounted = 3;
  table->set(makeOutput(seenTwice, 1, false), NNCacheAttribution::toContext(card));
  table->set(makeOutput(seenOnce, 1, false), NNCacheAttribution::toContext(card));
  table->set(makeOutput(neverCounted, 1, false), NNCacheAttribution::toContext(card));

  vector<pair<int, uint64_t> > counts;
  counts.push_back(make_pair(seenTwice, (uint64_t)2));
  counts.push_back(make_pair(seenOnce, (uint64_t)1));
  // neverCounted is deliberately absent from the count log entirely.

  const NNEvalContainer container =
    NNEvalContainer::forContextAndModel(dir.path(), "card-5455", MODEL, MODEL_VERSION);
  const NNCacheEvaluationDumpResult result = nnCacheDumpEvaluations(
    container, *table, card, NNCacheDiskAdmission::minLookups(2), observations(counts)
  );
  // THE WITNESS COMES FIRST, deliberately: the claim is about the file, so the file is what
  // fails when the claim fails. Asserting the planner's own report before it would let a
  // regression fire on a bookkeeping number and never reach the surface that matters.
  const NNEvalContainerIndex onDisk = container.loadIndex();
  testAssert(!containerHasKey(onDisk, nthKey(seenOnce)));
  testAssert(!containerHasKey(onDisk, nthKey(neverCounted)));
  testAssert(containerHasKey(onDisk, nthKey(seenTwice)));
  testAssert(onDisk.entries().size() == 1);

  // And the report agrees with it, so a client is told the same thing the disk says.
  testAssert(result.plan.keys.size() == 1);
  testAssert(result.plan.belowThreshold == 2);
  testAssert(result.plan.alreadyPersisted == 0);
  testAssert(result.plan.notResident == 0);
  cout << "  admission minLookups(2): 3 earned, " << onDisk.entries().size()
       << " on disk; seen-twice present=" << containerHasKey(onDisk, nthKey(seenTwice))
       << " seen-once present=" << containerHasKey(onDisk, nthKey(seenOnce))
       << " uncounted present=" << containerHasKey(onDisk, nthKey(neverCounted)) << endl;
}

// The same three keys and the same session under all(), so the absence above is attributable
// to the predicate and to nothing else about the fixture. Without this leg, a dump that wrote
// only its first entry for an unrelated reason would pass the test above.
void testTheSameSessionUnderAllAdmitsEveryEntryTheFirstOneRefused() {
  ScopedTempDir dir(TMP_DIR_PREFIX);
  unique_ptr<NNCacheTable> table = twoLevelTableOver(vector<int>());
  const NNCacheContextId card = table->attachCacheContext("card-5455");
  for(int serial = 1; serial <= 3; serial++)
    table->set(makeOutput(serial, 1, false), NNCacheAttribution::toContext(card));

  vector<pair<int, uint64_t> > counts;
  counts.push_back(make_pair(1, (uint64_t)2));
  counts.push_back(make_pair(2, (uint64_t)1));

  const NNEvalContainer container =
    NNEvalContainer::forContextAndModel(dir.path(), "card-5455", MODEL, MODEL_VERSION);
  const NNCacheEvaluationDumpResult result = nnCacheDumpEvaluations(
    container, *table, card, NNCacheDiskAdmission::all(), observations(counts)
  );
  testAssert(result.plan.belowThreshold == 0);
  const NNEvalContainerIndex onDisk = container.loadIndex();
  testAssert(onDisk.entries().size() == 3);
  for(int serial = 1; serial <= 3; serial++)
    testAssert(containerHasKey(onDisk, nthKey(serial)));
  cout << "  same session under all(): " << onDisk.entries().size()
       << " on disk, belowThreshold=" << result.plan.belowThreshold << endl;
}

//-------------------------------------------------------------------------------------
// Claim 2: the persisted mark, one witness per transition
//-------------------------------------------------------------------------------------

// TRANSITION ONE: SET ON FILL-FROM-CONTAINER. An attach fills level 1 with the container keys
// its level-0 bound did not take. Those entries are already in the file, so a dump owes
// nothing for them -- and the witness is the file, which must stay exactly as the fill found
// it rather than gaining a second copy of every key.
void testAnEntryFilledFromTheContainerIsNeverWrittenBackToIt() {
  ScopedTempDir dir(TMP_DIR_PREFIX);
  const NNEvalContainer container =
    NNEvalContainer::forContextAndModel(dir.path(), "card-5455", MODEL, MODEL_VERSION);

  // The card as it already exists on disk.
  vector<shared_ptr<const NNOutput> > alreadyStored;
  for(int serial = 1; serial <= 3; serial++)
    alreadyStored.push_back(shared_ptr<const NNOutput>(makeOutput(serial, 1, false)));
  (void)container.appendBlock(alreadyStored);
  const string afterFirstWrite = fileContents(container.path());

  // The attach: level 0 takes nothing, so all three are filled into level 1 FROM THE
  // CONTAINER. That provenance is the one thing this call says differently from an ordinary
  // set, and it is the whole mechanism.
  unique_ptr<NNCacheTable> table = twoLevelTableOver(vector<int>());
  const NNCacheContextId card = table->attachCacheContext("card-5455");
  for(int serial = 1; serial <= 3; serial++) {
    table->set(
      makeOutput(serial, 1, false),
      NNCacheAttribution::toContext(card),
      NNCacheEntryProvenance::LoadedFromContainer
    );
  }

  const NNCacheEvaluationDumpResult result = nnCacheDumpEvaluations(
    container, *table, card, NNCacheDiskAdmission::all(), vector<NNCacheCountRow>()
  );

  // THE WITNESS FIRST: the file is the same bytes it was before the attach. Not "the same
  // number of live keys" -- a re-append would keep that identical while doubling the file.
  testAssert(fileContents(container.path()) == afterFirstWrite);
  testAssert(container.loadIndex().blocksApplied() == 1);

  // Then the in-memory state and the report, which must agree with it.
  testAssert(table->attributedKeysFor(card).size() == 3);
  testAssert(table->unpersistedKeysFor(card).empty());
  testAssert(result.plan.entries.empty());
  testAssert(result.plan.alreadyPersisted == 3);
  testAssert(result.append.bytesAppended == 0);
  cout << "  fill-from-container: alreadyPersisted=" << result.plan.alreadyPersisted
       << " owed=" << result.plan.entries.size()
       << " bytesAppended=" << result.append.bytesAppended
       << " file bytes " << afterFirstWrite.size() << " -> " << fileContents(container.path()).size()
       << ", blocks=" << container.loadIndex().blocksApplied() << endl;
}

// The same fixture with the fill declared as an ordinary live evaluation, which is what the
// engine's every other set path is. Every entry is then owed and every entry is re-written --
// the defect, reproduced deliberately, so the leg above is attributable to the provenance
// argument rather than to the container happening to reject duplicates.
void testTheSameFillDeclaredLiveDoesReAppendTheWholeRemainder() {
  ScopedTempDir dir(TMP_DIR_PREFIX);
  const NNEvalContainer container =
    NNEvalContainer::forContextAndModel(dir.path(), "card-5455", MODEL, MODEL_VERSION);
  vector<shared_ptr<const NNOutput> > alreadyStored;
  for(int serial = 1; serial <= 3; serial++)
    alreadyStored.push_back(shared_ptr<const NNOutput>(makeOutput(serial, 1, false)));
  (void)container.appendBlock(alreadyStored);
  const int64_t afterFirstWrite = fileBytes(container.path());

  unique_ptr<NNCacheTable> table = twoLevelTableOver(vector<int>());
  const NNCacheContextId card = table->attachCacheContext("card-5455");
  for(int serial = 1; serial <= 3; serial++)
    table->set(makeOutput(serial, 1, false), NNCacheAttribution::toContext(card));

  testAssert(table->unpersistedKeysFor(card).size() == 3);
  const NNCacheEvaluationDumpResult result = nnCacheDumpEvaluations(
    container, *table, card, NNCacheDiskAdmission::all(), vector<NNCacheCountRow>()
  );
  testAssert(result.plan.entries.size() == 3);
  testAssert(result.plan.alreadyPersisted == 0);
  // The file grew by a whole second copy: three live keys, two blocks. This is the defect,
  // and it is what the provenance argument is the difference between.
  testAssert(fileBytes(container.path()) > afterFirstWrite);
  testAssert(container.loadIndex().blocksApplied() == 2);
  testAssert(container.loadIndex().entriesApplied() == 6);
  testAssert(container.loadIndex().entries().size() == 3);
  cout << "  same fill declared live: owed=" << result.plan.entries.size()
       << " file bytes " << afterFirstWrite << " -> " << fileBytes(container.path())
       << ", blocks=" << container.loadIndex().blocksApplied()
       << ", entriesApplied=" << container.loadIndex().entriesApplied()
       << " over " << container.loadIndex().entries().size() << " live keys" << endl;
}

// TRANSITION TWO: SET ON DUMP. A live entry is owed once. The second dump of the same
// attachment, with nothing in between, must leave the file BYTE-IDENTICAL -- not merely
// "logically the same set", which an appended duplicate block would also satisfy.
void testASecondDumpWithNothingInBetweenLeavesTheContainerByteIdentical() {
  ScopedTempDir dir(TMP_DIR_PREFIX);
  const NNEvalContainer container =
    NNEvalContainer::forContextAndModel(dir.path(), "card-5455", MODEL, MODEL_VERSION);
  unique_ptr<NNCacheTable> table = twoLevelTableOver(vector<int>());
  const NNCacheContextId card = table->attachCacheContext("card-5455");
  for(int serial = 1; serial <= 3; serial++)
    table->set(makeOutput(serial, 1, false), NNCacheAttribution::toContext(card));

  const NNCacheEvaluationDumpResult first = nnCacheDumpEvaluations(
    container, *table, card, NNCacheDiskAdmission::all(), vector<NNCacheCountRow>()
  );
  testAssert(first.plan.entries.size() == 3);
  testAssert(first.append.bytesAppended > 0);
  const string afterFirstDump = fileContents(container.path());
  testAssert(!afterFirstDump.empty());

  // The mark the first dump set is what makes the second one owe nothing.
  const NNCacheEvaluationDumpResult second = nnCacheDumpEvaluations(
    container, *table, card, NNCacheDiskAdmission::all(), vector<NNCacheCountRow>()
  );

  // THE WITNESS FIRST: byte-identical, which an appended duplicate block would fail while
  // still leaving the live key set at three.
  testAssert(fileContents(container.path()) == afterFirstDump);

  // Then the in-memory state and the report.
  testAssert(first.marked == 3);
  testAssert(table->unpersistedKeysFor(card).empty());
  testAssert(second.plan.entries.empty());
  testAssert(second.plan.alreadyPersisted == 3);
  testAssert(second.append.bytesAppended == 0);
  testAssert(second.marked == 0);
  cout << "  second dump, nothing in between: owed=" << second.plan.entries.size()
       << " alreadyPersisted=" << second.plan.alreadyPersisted
       << " bytesAppended=" << second.append.bytesAppended
       << " file byte-identical=" << (fileContents(container.path()) == afterFirstDump) << endl;
}

// TRANSITION THREE: CLEARED ON A LIVE OVERWRITE, and the case that makes it load-bearing.
//
// NNEvaluator's ownership-map fall-through re-evaluates a hit that lacked a requested map and
// sets the fuller result. That is a live overwrite, so the mark clears and the fuller entry is
// owed again -- which is how the store's rule that an entry WITHOUT an ownermap never
// supersedes one WITH is reached rather than assumed. A mark that survived the overwrite
// would leave the ownership knowledge in memory and never on disk, and the next attach would
// hand the session back the map-less entry and pay for the re-evaluation all over again.
void testALiveOverwriteClearsTheMarkSoAnOwnermapUpgradeReachesDisk() {
  ScopedTempDir dir(TMP_DIR_PREFIX);
  const NNEvalContainer container =
    NNEvalContainer::forContextAndModel(dir.path(), "card-5455", MODEL, MODEL_VERSION);
  unique_ptr<NNCacheTable> table = twoLevelTableOver(vector<int>());
  const NNCacheContextId card = table->attachCacheContext("card-5455");

  const int upgraded = 1;
  table->set(makeOutput(upgraded, 1, false), NNCacheAttribution::toContext(card));
  const NNCacheEvaluationDumpResult first = nnCacheDumpEvaluations(
    container, *table, card, NNCacheDiskAdmission::all(), vector<NNCacheCountRow>()
  );
  testAssert(first.plan.entries.size() == 1);
  testAssert(!containerEntryHasOwnerMap(container.loadIndex(), nthKey(upgraded)));
  testAssert(table->unpersistedKeysFor(card).empty());

  // The fall-through: the same key, re-evaluated, now carrying an ownership map.
  table->set(makeOutput(upgraded, 2, true), NNCacheAttribution::toContext(card));

  const NNCacheEvaluationDumpResult second = nnCacheDumpEvaluations(
    container, *table, card, NNCacheDiskAdmission::all(), vector<NNCacheCountRow>()
  );

  // THE WITNESS FIRST: the merged live set on disk now carries the map. One key, two blocks,
  // and the later, fuller entry is the one a load returns.
  const NNEvalContainerIndex onDisk = container.loadIndex();
  testAssert(containerEntryHasOwnerMap(onDisk, nthKey(upgraded)));
  testAssert(onDisk.entries().size() == 1);
  testAssert(onDisk.blocksApplied() == 2);

  // Then the in-memory state and the report.
  testAssert(second.plan.entries.size() == 1);
  testAssert(second.append.bytesAppended > 0);

  const NNEvalContainerContents merged = container.load();
  testAssert(merged.entries().size() == 1);
  testAssert(merged.entries()[0]->whiteOwnerMap != NULL);
  // And it is the SECOND generation's numbers, not the first's -- the upgrade superseded
  // rather than merely appended beside.
  testAssert(merged.entries()[0]->whiteWinProb == 0.25f);
  cout << "  ownermap upgrade: owed again after live overwrite=" << second.plan.entries.size()
       << ", blocks=" << onDisk.blocksApplied()
       << ", live keys=" << onDisk.entries().size()
       << ", ownermap on disk=" << containerEntryHasOwnerMap(onDisk, nthKey(upgraded)) << endl;
}

//-------------------------------------------------------------------------------------
// Claim 3: the count delta
//-------------------------------------------------------------------------------------

// A take is a DELTA and a second take with nothing in between is empty, at both levels: the
// frozen level 0's per-entry counters and level 1's hit ledger.
void testASecondCountTakeWithNoInterveningLookupsYieldsNothing() {
  unique_ptr<NNCacheTable> table = twoLevelTableOver(vector<int>(1, 100));
  const NNCacheContextId card = table->attachCacheContext("card-5455");
  table->set(makeOutput(1, 1, false), NNCacheAttribution::toContext(card));

  shared_ptr<NNOutput> got;
  testAssert(table->get(nthKey(100), got));  // a level-0 hit
  testAssert(table->get(nthKey(100), got));  // and another
  testAssert(table->get(nthKey(1), got));    // a level-1 hit

  const NNCacheHitLedger firstTake = table->takeUnpersistedHitCounts();
  testAssert(firstTake.isCounted());
  testAssert(firstTake.entries().size() == 2);
  for(size_t i = 0; i < firstTake.entries().size(); i++) {
    if(firstTake.entries()[i].key == nthKey(100))
      testAssert(firstTake.entries()[i].hits == 2);
    else {
      testAssert(firstTake.entries()[i].key == nthKey(1));
      testAssert(firstTake.entries()[i].hits == 1);
    }
  }

  // Nothing in between.
  const NNCacheHitLedger secondTake = table->takeUnpersistedHitCounts();
  testAssert(secondTake.isCounted());
  testAssert(secondTake.entries().empty());

  // And the delta resumes from the mark rather than from zero or from the total.
  testAssert(table->get(nthKey(100), got));
  const NNCacheHitLedger thirdTake = table->takeUnpersistedHitCounts();
  testAssert(thirdTake.entries().size() == 1);
  testAssert(thirdTake.entries()[0].key == nthKey(100));
  testAssert(thirdTake.entries()[0].hits == 1);

  // The pure-read surface is untouched by any of it: it still reports the session totals,
  // which is what stats and a between-searches report want. Two surfaces, two meanings.
  cout << "  count take: first=" << firstTake.entries().size()
       << " rows, second (nothing in between)=" << secondTake.entries().size()
       << " rows, third after one more lookup=" << thirdTake.entries().size()
       << " rows of " << thirdTake.entries()[0].hits << " hit" << endl;

  const NNCacheHitLedger reported = table->harvestHitCounts();
  testAssert(reported.isCounted());
  for(size_t i = 0; i < reported.entries().size(); i++) {
    if(reported.entries()[i].key == nthKey(100))
      testAssert(reported.entries()[i].hits == 3);
  }
}

// AN EARNED KEY THAT IS NO LONGER IN THE TABLE IS COUNTED, NOT WRITTEN AS A HOLE. The
// attribution ledger records what a session EARNED, which is a superset of what the table
// holds -- a SecondSighting admission declines a key on its first sighting, and a capacity
// sweep can drop one later. A dump meeting such a key has nothing to write for it, and the
// honest disposition is a named figure rather than a silently shorter block.
void testAnEarnedKeyTheTableNoLongerHoldsIsCountedAndNotWritten() {
  ScopedTempDir dir(TMP_DIR_PREFIX);
  const NNEvalContainer container =
    NNEvalContainer::forContextAndModel(dir.path(), "card-5455", MODEL, MODEL_VERSION);
  unique_ptr<NNCacheTable> table =
    makeTwoLevelNNCacheTable(NNCacheFrozen::build(vector<unique_ptr<NNOutput>>()), secondSightingLevelOne(10), 10);
  const NNCacheContextId card = table->attachCacheContext("card-5455");

  // Serial 1 is offered twice, so its second sighting stores it. Serial 2 is offered once and
  // is declined -- earned, attributed, and not resident.
  table->set(makeOutput(1, 1, false), NNCacheAttribution::toContext(card));
  table->set(makeOutput(1, 1, false), NNCacheAttribution::toContext(card));
  table->set(makeOutput(2, 1, false), NNCacheAttribution::toContext(card));

  const NNCacheEvaluationDumpResult result = nnCacheDumpEvaluations(
    container, *table, card, NNCacheDiskAdmission::all(), vector<NNCacheCountRow>()
  );

  const NNEvalContainerIndex onDisk = container.loadIndex();
  testAssert(containerHasKey(onDisk, nthKey(1)));
  testAssert(!containerHasKey(onDisk, nthKey(2)));
  testAssert(result.plan.notResident == 1);
  testAssert(result.plan.entries.size() == 1);
  cout << "  earned but not resident: " << result.plan.notResident
       << " counted, " << onDisk.entries().size() << " written" << endl;
}

// A DUMP MUST NOT APPEAR IN THE COUNTS IT IS DUMPING. Reading the entries to be written
// through get() would count each one as a retrieval, so every dump would raise the popularity
// of exactly the keys it wrote -- a self-fulfilling ranking that the next attach's build order
// then believes. The witness is a take AFTER a dump: the dump read three entries, and the take
// must still be empty.
void testADumpDoesNotCountItselfIntoTheCountsItIsDumping() {
  ScopedTempDir dir(TMP_DIR_PREFIX);
  const NNEvalContainer container =
    NNEvalContainer::forContextAndModel(dir.path(), "card-5455", MODEL, MODEL_VERSION);
  unique_ptr<NNCacheTable> table = twoLevelTableOver(vector<int>(1, 100));
  const NNCacheContextId card = table->attachCacheContext("card-5455");
  for(int serial = 1; serial <= 3; serial++)
    table->set(makeOutput(serial, 1, false), NNCacheAttribution::toContext(card));

  // Start from a clean slate so anything the dump adds is the only thing there is.
  (void)table->takeUnpersistedHitCounts();

  const NNCacheEvaluationDumpResult result = nnCacheDumpEvaluations(
    container, *table, card, NNCacheDiskAdmission::all(), vector<NNCacheCountRow>()
  );
  testAssert(result.plan.entries.size() == 3);

  const NNCacheHitLedger afterDump = table->takeUnpersistedHitCounts();
  testAssert(afterDump.isCounted());
  testAssert(afterDump.entries().empty());
  cout << "  a dump of " << result.plan.entries.size() << " entries left "
       << afterDump.entries().size() << " rows of new hits behind it" << endl;
}

// THE COMPOSED INVARIANT, at the seam this increment can reach: attach, look up, dump,
// detach, RE-ATTACH THE SAME CARD, dump again with no queries in between. The count log must
// be unchanged by the second dump -- not merely "not doubled", unchanged.
//
// The re-attach is modelled by discarding the table and building a fresh one over a level 0
// carrying the same keys, which is exactly what a detach and a re-attach do to it. What is
// NOT modelled here is the protocol: cache_attach and cache_dump do not exist yet, so the
// walk is driven through the library seams those actions will call.
void testAnAttachDumpDetachReattachDumpCycleLeavesTheCountLogUnchanged() {
  ScopedTempDir dir(TMP_DIR_PREFIX);
  const NNCacheCountLog log = NNCacheCountLog::forContext(dir.path(), "card-5455");
  const vector<int> zeroSerials(1, 100);

  {
    unique_ptr<NNCacheTable> table = twoLevelTableOver(zeroSerials);
    shared_ptr<NNOutput> got;
    for(int i = 0; i < 5; i++)
      testAssert(table->get(nthKey(100), got));
    (void)log.appendDump(table->takeUnpersistedHitCounts());
  }
  const NNCacheCountLogContents afterFirst = log.load();
  testAssert(logHasKey(afterFirst, nthKey(100)));
  testAssert(logRowFor(afterFirst, nthKey(100)).lookups == 5);
  testAssert(logRowFor(afterFirst, nthKey(100)).sessions == 1);

  // The re-attach: the same card, freshly loaded, nothing looked up.
  {
    unique_ptr<NNCacheTable> table = twoLevelTableOver(zeroSerials);
    const NNCacheHitLedger delta = table->takeUnpersistedHitCounts();
    testAssert(delta.isCounted());
    testAssert(delta.entries().empty());
    (void)log.appendDump(delta);
  }

  // THE WITNESS: every key's accumulated figures, read back off disk, are what they were.
  const NNCacheCountLogContents afterSecond = log.load();
  testAssert(afterSecond.rows().size() == afterFirst.rows().size());
  for(size_t i = 0; i < afterFirst.rows().size(); i++) {
    const NNCacheCountRow before = afterFirst.rows()[i];
    const NNCacheCountRow after = logRowFor(afterSecond, before.key);
    testAssert(after.lookups == before.lookups);
    testAssert(after.sessions == before.sessions);
  }
  cout << "  attach/dump/detach/re-attach/dump: key100 lookups "
       << logRowFor(afterFirst, nthKey(100)).lookups << " -> " << logRowFor(afterSecond, nthKey(100)).lookups
       << ", sessions " << logRowFor(afterFirst, nthKey(100)).sessions
       << " -> " << logRowFor(afterSecond, nthKey(100)).sessions << endl;
}

// The same walk with the PURE-READ surface handed to the dump instead of the delta take --
// the shape a caller reaches for if the two surfaces are confused. It is the defect, and it
// is why the delta surface exists as its own type rather than as a comment on the other one.
void testTheSameCycleThroughTheReportingSurfaceDoesInflateTheRecord() {
  ScopedTempDir dir(TMP_DIR_PREFIX);
  const NNCacheCountLog log = NNCacheCountLog::forContext(dir.path(), "card-5455");
  const vector<int> zeroSerials(1, 100);

  {
    unique_ptr<NNCacheTable> table = twoLevelTableOver(zeroSerials);
    shared_ptr<NNOutput> got;
    for(int i = 0; i < 5; i++)
      testAssert(table->get(nthKey(100), got));
    (void)log.appendDump(table->harvestHitCounts());
  }
  const NNCacheCountLogContents afterFirst = log.load();
  testAssert(logRowFor(afterFirst, nthKey(100)).lookups == 5);
  testAssert(logRowFor(afterFirst, nthKey(100)).sessions == 1);

  {
    unique_ptr<NNCacheTable> table = twoLevelTableOver(zeroSerials);
    // Not empty: the reporting surface reports a row of zero hits for a pre-warmed key that
    // earned nothing, deliberately, and appendDump raises that key's sessions for it.
    testAssert(!table->harvestHitCounts().entries().empty());
    (void)log.appendDump(table->harvestHitCounts());
  }
  const NNCacheCountLogContents afterSecond = log.load();
  testAssert(logRowFor(afterSecond, nthKey(100)).lookups == 5);
  testAssert(logRowFor(afterSecond, nthKey(100)).sessions == 2);
  cout << "  same cycle through the reporting surface: key100 sessions "
       << logRowFor(afterFirst, nthKey(100)).sessions << " -> "
       << logRowFor(afterSecond, nthKey(100)).sessions << " with nothing looked up" << endl;
}

// A shadowing set hands level 1 the hits level 0 accrued. It must hand over the UNPERSISTED
// remainder and not the total, or the part already written to the log is written again.
void testShadowingTransfersOnlyTheHitsThatHaveNotReachedTheLog() {
  unique_ptr<NNCacheTable> table = twoLevelTableOver(vector<int>(1, 100));
  shared_ptr<NNOutput> got;
  for(int i = 0; i < 4; i++)
    testAssert(table->get(nthKey(100), got));

  const NNCacheHitLedger firstTake = table->takeUnpersistedHitCounts();
  testAssert(firstTake.entries().size() == 1);
  testAssert(firstTake.entries()[0].hits == 4);

  // Two more, then the ownership-map fall-through's set, which shadows the level-0 entry and
  // transfers what it accrued into level 1's ledger.
  testAssert(table->get(nthKey(100), got));
  testAssert(table->get(nthKey(100), got));
  table->set(makeOutput(100, 2, true));

  const NNCacheHitLedger secondTake = table->takeUnpersistedHitCounts();
  testAssert(secondTake.entries().size() == 1);
  testAssert(secondTake.entries()[0].key == nthKey(100));
  // TWO, not six: the four already in the log left with the log, not with the key.
  testAssert(secondTake.entries()[0].hits == 2);
  cout << "  shadowing transfers the unpersisted remainder: 6 hits, 4 already in the log, "
       << secondTake.entries()[0].hits << " transferred" << endl;
}

//-------------------------------------------------------------------------------------
// The cost claim, mechanized
//-------------------------------------------------------------------------------------

// The persisted mark is claimed to cost nothing, in three structures. A claim about bytes is
// asserted against sizeof rather than left in a comment, because a comment about a byte
// figure in this very file has already been wrong once (see the recorder's own header).
void testThePersistedMarkCostsNoBytesInAnyStructureThatCarriesIt() {
  // The attribution recorder's row: 16 of key, 8 of set id, 4 of context index, 1 of mark,
  // in a structure aligned to 8. The mark is inside the alignment, not beside it.
  testAssert(NNCacheAttributionRecorder::rowBytes() == 32);
  // The two-level hit ledger's row: 16 + 4 + 4, the last four being what alignment always
  // spent here. twoLevelHitLedgerBytes is the one home of that arithmetic.
  testAssert(twoLevelHitLedgerBytes(0) == 24);
  testAssert(twoLevelHitLedgerBytes(10) == 24 * 1024);
  // The frozen index's record: SPEC.md 8's whole cache-line argument rests on this being 32,
  // and the mark went into the 4 bytes it already reserved.
  testAssert(NNCacheFrozenIndex::recordBytes() == 32);
  cout << "  the mark costs: recorder row " << NNCacheAttributionRecorder::rowBytes()
       << " B, hit ledger row " << (twoLevelHitLedgerBytes(10) / 1024)
       << " B, frozen record " << NNCacheFrozenIndex::recordBytes() << " B" << endl;
}

//-------------------------------------------------------------------------------------
// The refusals
//-------------------------------------------------------------------------------------

void testTheDumpSeamsRefuseAContextThisTableDidNotAttach() {
  unique_ptr<NNCacheTable> mine = twoLevelTableOver(vector<int>());
  unique_ptr<NNCacheTable> theirs = twoLevelTableOver(vector<int>());
  mine->attachCacheContext("card-5455");
  const NNCacheContextId foreign = theirs->attachCacheContext("card-9001");

  testAssert(refused([&]() { (void)mine->unpersistedKeysFor(foreign); }, "not attached to this cache"));
  testAssert(refused(
    [&]() { (void)mine->markPersisted(foreign, vector<Hash128>()); }, "not attached to this cache"
  ));
}

}  // namespace

void Tests::runNNCacheDumpTests() {
  cout << "Running NN cache dump admission and persistence tests" << endl;

  testTheReadAndWriteSidesShareOneLookupThresholdRule();
  testAnEntryBelowTheAdmissionThresholdIsAbsentFromTheWrittenContainer();
  testTheSameSessionUnderAllAdmitsEveryEntryTheFirstOneRefused();
  testAnEntryFilledFromTheContainerIsNeverWrittenBackToIt();
  testTheSameFillDeclaredLiveDoesReAppendTheWholeRemainder();
  testASecondDumpWithNothingInBetweenLeavesTheContainerByteIdentical();
  testALiveOverwriteClearsTheMarkSoAnOwnermapUpgradeReachesDisk();
  testASecondCountTakeWithNoInterveningLookupsYieldsNothing();
  testAnEarnedKeyTheTableNoLongerHoldsIsCountedAndNotWritten();
  testADumpDoesNotCountItselfIntoTheCountsItIsDumping();
  testAnAttachDumpDetachReattachDumpCycleLeavesTheCountLogUnchanged();
  testTheSameCycleThroughTheReportingSurfaceDoesInflateTheRecord();
  testShadowingTransfersOnlyTheHitsThatHaveNotReachedTheLog();
  testThePersistedMarkCostsNoBytesInAnyStructureThatCarriesIt();
  testTheDumpSeamsRefuseAContextThisTableDidNotAttach();

  cout << "Done" << endl;
}
