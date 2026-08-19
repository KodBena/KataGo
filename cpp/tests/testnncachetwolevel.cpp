#include "../tests/tests.h"

#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <vector>

#include "../neuralnet/nncache.h"
#include "../neuralnet/nncachefrozen.h"
#include "../neuralnet/nncachelevelzero.h"
#include "../neuralnet/nncachetwolevel.h"

using namespace std;
using namespace TestCommon;

// THE ORDERED RESOLUTION LIST: several attached level-0 sources serving one lookup, tried
// in attach order, then level 1.
//
// EVERY SCENARIO HERE USES DELIBERATELY OVERLAPPING KEYS, and that is the point rather than
// a detail. A test whose sources are disjoint cannot observe an ordering at all: every key
// has exactly one holder, so first-match, last-match and any-match give the same answer and
// a reversed walk would pass. The order is only observable where two sources both hold a
// key and their evaluations differ, so that is what these build (ADR-0021 Rule 1 -- the
// witness observes the claimed property, at the site of the claim).
//
// Everything asserted here is a logic invariant and is asserted exactly, with no tolerance
// (ADR-0009's Calibration): which source answered, which sources gave up a count, how many
// rows a harvest carried and in what order are discrete facts. The float the tests compare
// is a TAG written into whiteScoreMean at construction and never arithmetic upon -- it is
// an identity, not a measurement, so exact comparison is the right bar for it and a
// tolerance would be the category error that Calibration names.

namespace {

// A distinct key per serial, spread over both halves. Same generator as
// testnncachefrozen.cpp's, so a key named nthKey(3) means the same position in both files.
Hash128 nthKey(int serial) {
  return Hash128(
    ((uint64_t)(serial + 1)) * 0x9E3779B97F4A7C15ULL,
    ((uint64_t)(serial + 1)) * 0xD6E8FEB86659FD93ULL + 0x1234567ULL
  );
}

// WHICH SOURCE AN EVALUATION CAME FROM, carried in the evaluation itself.
//
// Two sources hold the same key on purpose here, so "the table answered" is not the
// observation any of these tests needs -- "WHICH source answered" is. The tag rides in
// whiteScoreMean because it is a plain float the cache copies and never touches, and it is
// read back with tagOf so no test compares the raw field and none can drift from this one
// home (ADR-0012 P1).
const float SOURCE_A = 101.0f;
const float SOURCE_B = 202.0f;
const float SOURCE_C = 303.0f;
const float SOURCE_D = 404.0f;
const float LEVEL_ONE = 909.0f;

unique_ptr<NNOutput> ownedOutputFor(Hash128 hash, float tag) {
  unique_ptr<NNOutput> p(new NNOutput());
  p->nnHash = hash;
  p->nnXLen = 19;
  p->nnYLen = 19;
  p->whiteScoreMean = tag;
  return p;
}

shared_ptr<NNOutput> sharedOutputFor(Hash128 hash, float tag) {
  shared_ptr<NNOutput> p = make_shared<NNOutput>();
  p->nnHash = hash;
  p->nnXLen = 19;
  p->nnYLen = 19;
  p->whiteScoreMean = tag;
  return p;
}

float tagOf(const shared_ptr<NNOutput>& p) {
  testAssert(p != nullptr);
  return p->whiteScoreMean;
}

// A frozen source over `keys`, every entry tagged `tag`.
unique_ptr<NNCacheFrozen> sourceOver(const vector<Hash128>& keys, float tag) {
  vector<unique_ptr<NNOutput>> outputs;
  for(size_t i = 0; i < keys.size(); i++)
    outputs.push_back(ownedOutputFor(keys[i], tag));
  return NNCacheFrozen::build(std::move(outputs));
}

unique_ptr<NNCacheTable> defaultLevelOne(int sizePowerOfTwo) {
  return NNCacheTable::create(NNCacheConfig::statusQuo(sizePowerOfTwo, 2));
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

// The tag the table served for `key`, or nullopt if nothing did.
optional<float> served(NNCacheTable& table, Hash128 key) {
  shared_ptr<NNOutput> got;
  const bool found = table.get(key, got);
  testAssert(found == (got != nullptr));
  if(!found)
    return nullopt;
  testAssert(got->nnHash == key);
  return tagOf(got);
}

uint32_t hitsForKeyIn(const NNCacheHitLedger& ledger, Hash128 key) {
  uint32_t total = 0;
  int rows = 0;
  for(size_t i = 0; i < ledger.entries().size(); i++) {
    if(ledger.entries()[i].key == key) {
      total += ledger.entries()[i].hits;
      rows += 1;
    }
  }
  // ONE ROW PER KEY across the whole surface, sources included. This is the property a
  // persistence layer depends on, so it is asserted on every read rather than once.
  testAssert(rows <= 1);
  return total;
}

//-------------------------------------------------------------------------------------
// Order
//-------------------------------------------------------------------------------------

// THE CENTRAL CLAIM: a lookup against N attached sources returns the FIRST match in
// resolution order, and resolution order is attach order.
//
// The key sets deliberately overlap in a chain -- A and B share k1, B and C share k2, all
// three share k3 -- so every position in the walk is exercised: the first source wins a key
// the second also holds, the second wins a key the third also holds, and the first wins a
// key all three hold. Nothing here would still pass under a reversed walk, under a
// last-match rule, or under "whichever source is cheapest to probe".
void testFirstMatchInAttachOrderWins() {
  const Hash128 onlyA = nthKey(0);
  const Hash128 k1 = nthKey(1);   // A and B
  const Hash128 onlyB = nthKey(2);
  const Hash128 k2 = nthKey(3);   // B and C
  const Hash128 onlyC = nthKey(4);
  const Hash128 k3 = nthKey(5);   // A, B and C
  const Hash128 nowhere = nthKey(90);

  unique_ptr<NNCacheFrozen> aOwned = sourceOver({onlyA, k1, k3}, SOURCE_A);
  unique_ptr<NNCacheFrozen> bOwned = sourceOver({k1, onlyB, k2, k3}, SOURCE_B);
  unique_ptr<NNCacheFrozen> cOwned = sourceOver({k2, onlyC, k3}, SOURCE_C);
  NNCacheFrozen* a = aOwned.get();
  NNCacheFrozen* b = bOwned.get();
  NNCacheFrozen* c = cOwned.get();

  unique_ptr<NNCacheTwoLevelTable> table =
    makeTwoLevelNNCacheTable(std::move(aOwned), defaultLevelOne(8), 8);
  const NNCacheLevelZeroSourceId idB = table->attachLevelZero(std::move(bOwned));
  const NNCacheLevelZeroSourceId idC = table->attachLevelZero(std::move(cOwned));
  testAssert(table->numLevelZeroSources() == 3);

  // Sole holders resolve to themselves.
  testAssert(served(*table, onlyA) == SOURCE_A);
  testAssert(served(*table, onlyB) == SOURCE_B);
  testAssert(served(*table, onlyC) == SOURCE_C);
  // Shared keys resolve to the EARLIER holder, at each of the three positions.
  testAssert(served(*table, k1) == SOURCE_A);
  testAssert(served(*table, k2) == SOURCE_B);
  testAssert(served(*table, k3) == SOURCE_A);
  // A key no source holds falls through, and level 1 has it once it is set.
  testAssert(served(*table, nowhere) == nullopt);
  table->set(sharedOutputFor(nowhere, LEVEL_ONE));
  testAssert(served(*table, nowhere) == LEVEL_ONE);

  // ONLY THE WINNING SOURCE COUNTED. The losers' entries for the shared keys were never
  // reached, so their counters are still zero -- which is the same fact as "first match
  // wins", read at the counter instead of at the answer.
  testAssert(a->hitCountAt(a->index().find(k3).value()) == 1);
  testAssert(b->hitCountAt(b->index().find(k3).value()) == 0);
  testAssert(c->hitCountAt(c->index().find(k3).value()) == 0);
  testAssert(b->hitCountAt(b->index().find(k1).value()) == 0);
  testAssert(c->hitCountAt(c->index().find(k2).value()) == 0);

  cout << "  first-match: 3 sources, 3 shared keys; k1->" << served(*table, k1).value()
       << " k2->" << served(*table, k2).value()
       << " k3->" << served(*table, k3).value()
       << "; loser counters for k3: B=" << b->hitCountAt(b->index().find(k3).value())
       << " C=" << c->hitCountAt(c->index().find(k3).value()) << endl;

  // Silence the unused-id warning honestly: they are used below in the detach tests, and
  // here their only job is to exist so attach's return value is not discarded.
  testAssert(idB != idC);
}

// THE ORDER IS ATTACH ORDER, INCLUDING AFTER CHURN. Detaching a source does not reorder
// what is left, and re-attaching one puts it at the BACK -- so a client that wants a source
// to win a key attaches it earlier, and re-attaching is not a way to jump the queue.
//
// FOUR SOURCES, AND THE ONE DETACHED IS NOT THE MIDDLE OF THREE. That is deliberate and it
// is the second thing this test learned: with three attached, removing the middle one gives
// the same list under an order-preserving erase and under a swap-with-the-last-and-pop, so
// a three-source version of this test passes under both and witnesses nothing about order
// preservation (witnessed -- an earlier draft did exactly that, and the swap mutation ran
// green through it). Four sources with the SECOND detached separate the two: erase leaves
// A,C,D and the swap leaves A,D,C. The difference is asserted twice, once on the ids and
// once on what a key held by both C and D actually resolves to, because the second is the
// consequence a client would feel (ADR-0021 Rule 1).
void testDetachPreservesRelativeOrderAndReAttachGoesToTheBack() {
  const Hash128 shared = nthKey(10);   // every source
  const Hash128 cd = nthKey(14);       // C and D only -- where an A,D,C order shows itself
  unique_ptr<NNCacheFrozen> aOwned = sourceOver({shared, nthKey(11)}, SOURCE_A);
  unique_ptr<NNCacheFrozen> bOwned = sourceOver({shared, nthKey(12)}, SOURCE_B);
  unique_ptr<NNCacheFrozen> cOwned = sourceOver({shared, nthKey(13), cd}, SOURCE_C);
  unique_ptr<NNCacheFrozen> dOwned = sourceOver({shared, nthKey(15), cd}, SOURCE_D);

  unique_ptr<NNCacheTwoLevelTable> table =
    makeTwoLevelNNCacheTable(std::move(aOwned), defaultLevelOne(8), 8);
  const NNCacheLevelZeroSourceId idA = table->levelZeroResolutionOrder()[0];
  const NNCacheLevelZeroSourceId idB = table->attachLevelZero(std::move(bOwned));
  const NNCacheLevelZeroSourceId idC = table->attachLevelZero(std::move(cOwned));
  const NNCacheLevelZeroSourceId idD = table->attachLevelZero(std::move(dOwned));

  testAssert(table->levelZeroResolutionOrder() == vector<NNCacheLevelZeroSourceId>({idA, idB, idC, idD}));
  testAssert(served(*table, shared) == SOURCE_A);
  testAssert(served(*table, cd) == SOURCE_C);

  // Detach the SECOND of four. A, C and D keep their relative order.
  unique_ptr<NNCacheFrozen> bBack = table->detachLevelZero(idB);
  testAssert(bBack != nullptr);
  testAssert(table->levelZeroResolutionOrder() == vector<NNCacheLevelZeroSourceId>({idA, idC, idD}));
  testAssert(served(*table, shared) == SOURCE_A);
  testAssert(served(*table, cd) == SOURCE_C);        // C still precedes D
  testAssert(served(*table, nthKey(12)) == nullopt); // B's own key left with B
  testAssert(served(*table, nthKey(13)) == SOURCE_C);
  testAssert(served(*table, nthKey(15)) == SOURCE_D);

  // Detach the FIRST. C is now the earliest holder of the shared key, so the same key
  // resolves differently -- the ordering observed from the other side.
  unique_ptr<NNCacheFrozen> aBack = table->detachLevelZero(idA);
  testAssert(table->levelZeroResolutionOrder() == vector<NNCacheLevelZeroSourceId>({idC, idD}));
  testAssert(served(*table, shared) == SOURCE_C);
  testAssert(served(*table, cd) == SOURCE_C);

  // Re-attach A. It goes to the BACK, so C keeps the shared key and A's own key returns.
  const NNCacheLevelZeroSourceId idA2 = table->attachLevelZero(std::move(aBack));
  testAssert(idA2 != idA);
  testAssert(table->levelZeroResolutionOrder() == vector<NNCacheLevelZeroSourceId>({idC, idD, idA2}));
  testAssert(served(*table, shared) == SOURCE_C);
  testAssert(served(*table, nthKey(11)) == SOURCE_A);

  // Detach everything. The table is coherent with an empty list: it serves from level 1
  // alone, and a source can be attached again afterwards.
  (void)table->detachLevelZero(idC);
  (void)table->detachLevelZero(idD);
  (void)table->detachLevelZero(idA2);
  testAssert(table->numLevelZeroSources() == 0);
  testAssert(served(*table, shared) == nullopt);
  table->set(sharedOutputFor(shared, LEVEL_ONE));
  testAssert(served(*table, shared) == LEVEL_ONE);

  cout << "  churn: A,B,C,D -> detach B -> detach A -> re-attach A; the C-and-D key served "
       << "C throughout, the all-sources key served A,A,C,C, and "
       << table->numLevelZeroSources() << " sources remain after detaching all" << endl;
}

//-------------------------------------------------------------------------------------
// The one-owner invariant, across the whole list
//-------------------------------------------------------------------------------------

// A SET SHADOWS THE KEY IN EVERY SOURCE THAT HOLDS IT, not in the first one that does.
//
// This is the invariant the list can break that a single level 0 could not. If a set
// shadowed only the winning source, the NEXT source in order would start resolving the key
// -- and would answer with the evaluation the caller has just superseded, before the
// fall-through to level 1 could ever happen. So the observation is deliberately made at
// what the table SERVES after the set, not merely at the shadow bits: serving the stale
// entry is the defect, and it is what this watches.
void testASetShadowsTheKeyInEverySourceThatHoldsIt() {
  const Hash128 shared = nthKey(20);
  unique_ptr<NNCacheFrozen> aOwned = sourceOver({shared, nthKey(21)}, SOURCE_A);
  unique_ptr<NNCacheFrozen> bOwned = sourceOver({shared, nthKey(22)}, SOURCE_B);
  unique_ptr<NNCacheFrozen> cOwned = sourceOver({shared, nthKey(23)}, SOURCE_C);
  NNCacheFrozen* a = aOwned.get();
  NNCacheFrozen* b = bOwned.get();
  NNCacheFrozen* c = cOwned.get();

  unique_ptr<NNCacheTwoLevelTable> table =
    makeTwoLevelNNCacheTable(std::move(aOwned), defaultLevelOne(8), 8);
  (void)table->attachLevelZero(std::move(bOwned));
  (void)table->attachLevelZero(std::move(cOwned));

  // Earn counts in the winner, so the transfer has something to carry.
  testAssert(served(*table, shared) == SOURCE_A);
  testAssert(served(*table, shared) == SOURCE_A);
  // And plant a count in a LOSER by hand, so the sum is over more than one source. Nothing
  // on the get path can do this while A wins, which is exactly why it is planted.
  testAssert(b->addHits(shared, 5));

  table->set(sharedOutputFor(shared, LEVEL_ONE));

  // THE OBSERVATION: what the table serves. Under a first-holder-only shadow this is
  // SOURCE_B -- a superseded evaluation, served ahead of the fresh one.
  testAssert(served(*table, shared) == LEVEL_ONE);
  // And the shadow bits agree, in all three.
  testAssert(!a->contains(shared));
  testAssert(!b->contains(shared));
  testAssert(!c->contains(shared));
  // Every source's other key is untouched.
  testAssert(a->contains(nthKey(21)));
  testAssert(b->contains(nthKey(22)));
  testAssert(c->contains(nthKey(23)));

  // THE COUNTS ARE SUMMED, NOT DROPPED AND NOT DOUBLE-COUNTED: two hits earned in A plus
  // the five planted in B, plus the one this get just took in level 1.
  const NNCacheHitLedger ledger = table->harvestHitCounts();
  testAssert(ledger.disposition() == NNCacheHitLedgerDisposition::Counted);
  testAssert(hitsForKeyIn(ledger, shared) == 2 + 5 + 1);

  cout << "  shadow-all: 3 holders of one key; after the set the table serves "
       << served(*table, shared).value() << " and the ledger carries "
       << hitsForKeyIn(ledger, shared) << " hits (2 earned + 5 planted + 1 level-1)" << endl;
}

//-------------------------------------------------------------------------------------
// Handles
//-------------------------------------------------------------------------------------

// A HANDLE NAMES A SOURCE OR NAMES NOTHING; IT NEVER COMES TO NAME A DIFFERENT ONE.
void testStaleAndForeignHandlesAreRefused() {
  unique_ptr<NNCacheTwoLevelTable> table =
    makeTwoLevelNNCacheTable(sourceOver({nthKey(30)}, SOURCE_A), defaultLevelOne(8), 8);
  unique_ptr<NNCacheTwoLevelTable> other =
    makeTwoLevelNNCacheTable(sourceOver({nthKey(30)}, SOURCE_B), defaultLevelOne(8), 8);

  const NNCacheLevelZeroSourceId idB = table->attachLevelZero(sourceOver({nthKey(31)}, SOURCE_B));
  (void)table->detachLevelZero(idB);

  // Stale: the source it named is gone, and the serial it carries is never reissued, so
  // attaching more sources cannot make this id start naming one of them.
  const NNCacheLevelZeroSourceId idC = table->attachLevelZero(sourceOver({nthKey(32)}, SOURCE_C));
  // Observed at the consequence, not at the id: the claim is that spending the stale handle
  // cannot take the source attached after it, so what is watched is the refusal and then
  // whether C is still there and still serving (ADR-0021 Rule 1).
  testAssert(refused([&]() { (void)table->detachLevelZero(idB); }, "already been detached"));
  testAssert(served(*table, nthKey(32)) == SOURCE_C);  // idB's second detach took nothing
  testAssert(idC != idB);

  // Foreign: an id minted by another table's list is refused by name rather than detaching
  // whichever source sits at the same serial here.
  const NNCacheLevelZeroSourceId foreign = other->levelZeroResolutionOrder()[0];
  testAssert(refused([&]() { (void)table->detachLevelZero(foreign); }, "different cache"));
  testAssert(table->numLevelZeroSources() == 2);

  // A null source is refused rather than becoming a hole in the resolution order.
  testAssert(refused([&]() { (void)table->attachLevelZero(nullptr); }, "no source was supplied"));
  testAssert(table->numLevelZeroSources() == 2);

  cout << "  handles: stale id refused, foreign id refused, null source refused; "
       << table->numLevelZeroSources() << " sources still attached and serving" << endl;
}

// A DETACHED SOURCE COMES BACK WHOLE, and composes with the loader's own release -- which
// is the only reason detach hands it back instead of destroying it.
void testDetachedSourceIsHandedBackAndReleasable() {
  unique_ptr<NNCacheTwoLevelTable> table =
    makeTwoLevelNNCacheTable(sourceOver({nthKey(40)}, SOURCE_A), defaultLevelOne(8), 8);
  const NNCacheLevelZeroSourceId idB = table->attachLevelZero(sourceOver({nthKey(41), nthKey(42)}, SOURCE_B));

  unique_ptr<NNCacheFrozen> back = table->detachLevelZero(idB);
  testAssert(back != nullptr);
  testAssert(back->numEntries() == 2);
  testAssert(back->contains(nthKey(41)));

  const NNCacheLevelZeroRelease released = nnCacheReleaseLevelZero(std::move(back));
  testAssert(released.storageReleased);
  testAssert(served(*table, nthKey(41)) == nullopt);
  testAssert(served(*table, nthKey(40)) == SOURCE_A);

  cout << "  detach->release: 2-entry source handed back intact and its storage observed "
       << "released (" << (released.storageReleased ? "true" : "false") << ")" << endl;
}

//-------------------------------------------------------------------------------------
// The surfaces the list aggregates
//-------------------------------------------------------------------------------------

// ONE ROW PER KEY ACROSS OVERLAPPING SOURCES, and the row that survives is the one that
// could actually have been served.
void testHarvestEmitsOneRowPerKeyInResolutionOrder() {
  const Hash128 shared = nthKey(50);
  const Hash128 aOnly = nthKey(51);
  const Hash128 bOnly = nthKey(52);
  unique_ptr<NNCacheTwoLevelTable> table =
    makeTwoLevelNNCacheTable(sourceOver({shared, aOnly}, SOURCE_A), defaultLevelOne(8), 8);
  (void)table->attachLevelZero(sourceOver({shared, bOnly}, SOURCE_B));

  testAssert(served(*table, shared) == SOURCE_A);
  testAssert(served(*table, shared) == SOURCE_A);
  testAssert(served(*table, bOnly) == SOURCE_B);

  const NNCacheHitLedger ledger = table->harvestHitCounts();
  testAssert(ledger.disposition() == NNCacheHitLedgerDisposition::Counted);

  // Three distinct keys across two sources holding four entries between them.
  set<Hash128> keys;
  for(size_t i = 0; i < ledger.entries().size(); i++)
    keys.insert(ledger.entries()[i].key);
  testAssert(ledger.entries().size() == 3);
  testAssert(keys.size() == 3);
  // The surviving row for the shared key carries the WINNER's count, not the loser's zero.
  testAssert(hitsForKeyIn(ledger, shared) == 2);
  testAssert(hitsForKeyIn(ledger, bOnly) == 1);
  testAssert(hitsForKeyIn(ledger, aOnly) == 0);
  // Resolution order: source A's entries, then source B's remaining one.
  testAssert(ledger.entries()[0].key == shared || ledger.entries()[0].key == aOnly);
  testAssert(ledger.entries()[1].key == shared || ledger.entries()[1].key == aOnly);
  testAssert(ledger.entries()[2].key == bOnly);

  cout << "  harvest: 2 sources x 2 entries with 1 key shared -> " << ledger.entries().size()
       << " rows, shared key carrying " << hitsForKeyIn(ledger, shared) << " hits" << endl;
}

// STATS SUM EVERY ATTACHED SOURCE, and a key held twice is two resident evaluations,
// because it really is two.
void testStatsSumEverySource() {
  const Hash128 shared = nthKey(60);
  unique_ptr<NNCacheFrozen> aOwned = sourceOver({shared, nthKey(61)}, SOURCE_A);
  unique_ptr<NNCacheFrozen> bOwned = sourceOver({shared, nthKey(62), nthKey(63)}, SOURCE_B);
  const int64_t aStructure = (int64_t)aOwned->structureBytes();
  const int64_t bStructure = (int64_t)bOwned->structureBytes();
  const int64_t aPayload = aOwned->reachablePayloadBytes();
  const int64_t bPayload = bOwned->reachablePayloadBytes();

  unique_ptr<NNCacheTable> levelOneAlone = defaultLevelOne(8);
  const NNCacheStats bare = levelOneAlone->stats();
  levelOneAlone.reset();

  unique_ptr<NNCacheTwoLevelTable> table =
    makeTwoLevelNNCacheTable(std::move(aOwned), defaultLevelOne(8), 8);
  const NNCacheStats one = table->stats();
  testAssert(one.residentEntries == bare.residentEntries + 2);

  (void)table->attachLevelZero(std::move(bOwned));
  const NNCacheStats two = table->stats();
  testAssert(two.residentEntries == bare.residentEntries + 5);
  testAssert(two.residentPayloadBytes == bare.residentPayloadBytes + aPayload + bPayload);
  testAssert(two.fixedStructureBytes ==
             bare.fixedStructureBytes + aStructure + bStructure +
             (int64_t)twoLevelHitLedgerBytes(8) +
             ((int64_t)(((size_t)1) << 8) * (int64_t)sizeof(std::mutex)));
  testAssert(two.capacitySlots == bare.capacitySlots + 5);

  cout << "  stats: 2 sources of 2 and 3 entries -> " << (two.residentEntries - bare.residentEntries)
       << " resident level-0 entries and " << (two.fixedStructureBytes - one.fixedStructureBytes)
       << " added structure bytes" << endl;
}

// THE SUPPRESSED-ROW TRIPWIRE FIRES. harvest() drops the later holder's row for a shared
// key on the ground that it is provably zero -- a proof that lives in shadowAllHolders, not
// here. This plants the state that proof forbids and watches the tripwire go off, which is
// the only way to know the check is live rather than dead code beside a comment (ADR-0021
// Rule 2: the negative claim "no counted row is ever suppressed" is watched by a probe whose
// FIRING is the observation, and the probe is proven able to fire).
//
// addHits is the only door to that state, and that is the point: nothing on the get path can
// reach it while the invariant holds, so a test that could produce it by ordinary means
// would be reporting a defect rather than exercising a guard.
void testASuppressedRowCarryingHitsIsRefusedRatherThanDropped() {
  const Hash128 shared = nthKey(70);
  unique_ptr<NNCacheFrozen> bOwned = sourceOver({shared, nthKey(71)}, SOURCE_B);
  NNCacheFrozen* b = bOwned.get();
  unique_ptr<NNCacheTwoLevelTable> table =
    makeTwoLevelNNCacheTable(sourceOver({shared, nthKey(72)}, SOURCE_A), defaultLevelOne(8), 8);
  (void)table->attachLevelZero(std::move(bOwned));

  // Green first: with the invariant intact the suppression is silent and the surface is
  // three rows, so the tripwire is not merely never-reached-because-nothing-works.
  testAssert(table->harvestHitCounts().entries().size() == 3);

  // Now break it, at the only place it can be broken from outside.
  testAssert(b->addHits(shared, 4));
  testAssert(refused([&]() { (void)table->harvestHitCounts(); }, "already resolves that key"));

  cout << "  suppressed-row tripwire: silent at 3 rows while the invariant holds, refuses "
       << "by name once a shadowed-out entry is given 4 hits" << endl;
}

// THE EMPTY LIST IS A FULL CITIZEN. Detaching every source leaves a table that reports what
// level 1 alone reports -- not a table that throws, and not one that still claims level-0
// bytes it no longer holds.
void testAnEmptyListReportsExactlyLevelOne() {
  unique_ptr<NNCacheTable> levelOneAlone = defaultLevelOne(8);
  const NNCacheStats bare = levelOneAlone->stats();
  levelOneAlone.reset();

  unique_ptr<NNCacheTwoLevelTable> table =
    makeTwoLevelNNCacheTable(sourceOver({nthKey(80), nthKey(81)}, SOURCE_A), defaultLevelOne(8), 8);
  const NNCacheLevelZeroSourceId idA = table->levelZeroResolutionOrder()[0];
  const NNCacheLevelZeroRelease released = nnCacheReleaseLevelZero(table->detachLevelZero(idA));
  testAssert(released.storageReleased);

  testAssert(table->numLevelZeroSources() == 0);
  testAssert(table->levelZeroResolutionOrder().empty());
  const NNCacheStats empty = table->stats();
  testAssert(empty.residentEntries == bare.residentEntries);
  testAssert(empty.residentPayloadBytes == bare.residentPayloadBytes);
  testAssert(empty.capacitySlots == bare.capacitySlots);
  // The ledger and its mutex pool remain -- they are the table's, not a source's.
  testAssert(empty.fixedStructureBytes ==
             bare.fixedStructureBytes + (int64_t)twoLevelHitLedgerBytes(8) +
             ((int64_t)(((size_t)1) << 8) * (int64_t)sizeof(std::mutex)));
  // Counted, not NotCounted: a two-level table still counts, it just has no level-0 rows.
  const NNCacheHitLedger ledger = table->harvestHitCounts();
  testAssert(ledger.disposition() == NNCacheHitLedgerDisposition::Counted);
  testAssert(ledger.entries().empty());

  cout << "  empty list: 0 sources, stats equal to level 1 alone plus the ledger, harvest "
       << "Counted with " << ledger.entries().size() << " rows" << endl;
}

// AN EARLIER SOURCE THAT RESOLVES NOTHING STILL HOLDS ITS PLACE. Every one of source A's
// entries is shadowed out, so the walk reaches B for every key -- and A is still first in
// the order, still reported in stats, and still detachable by its own handle. The state is
// reachable in a real session (a client attaches an older card and the session re-evaluates
// all of it) and nothing else here exercises it.
void testAFullyShadowedEarlierSourceStillHoldsItsPlace() {
  const Hash128 k0 = nthKey(90);
  const Hash128 k1 = nthKey(91);
  unique_ptr<NNCacheFrozen> aOwned = sourceOver({k0, k1}, SOURCE_A);
  NNCacheFrozen* a = aOwned.get();
  unique_ptr<NNCacheTwoLevelTable> table =
    makeTwoLevelNNCacheTable(std::move(aOwned), defaultLevelOne(8), 8);
  const NNCacheLevelZeroSourceId idA = table->levelZeroResolutionOrder()[0];
  const NNCacheLevelZeroSourceId idB = table->attachLevelZero(sourceOver({k0, k1}, SOURCE_B));

  testAssert(served(*table, k0) == SOURCE_A);
  // Shadow both of A's keys -- which also shadows B's, so level 1 owns them now.
  table->set(sharedOutputFor(k0, LEVEL_ONE));
  table->set(sharedOutputFor(k1, LEVEL_ONE));
  testAssert(a->numEntries() == 2);
  testAssert(!a->contains(k0) && !a->contains(k1));
  testAssert(served(*table, k0) == LEVEL_ONE);
  testAssert(served(*table, k1) == LEVEL_ONE);

  // A is still first, still counted as capacity it holds, and its handle still works.
  testAssert(table->levelZeroResolutionOrder() == vector<NNCacheLevelZeroSourceId>({idA, idB}));
  testAssert(table->stats().capacitySlots >= 4);
  testAssert(table->harvestHitCounts().entries().size() == 2);  // level 1's two keys only
  unique_ptr<NNCacheFrozen> aBack = table->detachLevelZero(idA);
  testAssert(aBack != nullptr && aBack->numEntries() == 2);
  testAssert(table->levelZeroResolutionOrder() == vector<NNCacheLevelZeroSourceId>({idB}));
  testAssert(served(*table, k0) == LEVEL_ONE);

  cout << "  fully-shadowed first source: resolves nothing, keeps position 0, still "
       << "detachable, and the harvest carries " << table->harvestHitCounts().entries().size()
       << " rows" << endl;
}

//-------------------------------------------------------------------------------------
// The delta twin
//-------------------------------------------------------------------------------------

// EVERY ATTACHED SOURCE IS TAKEN FROM, so a second take with nothing in between is empty.
//
// The claim is about CONSUMPTION, not about rows, so the second take is asserted FIRST: a
// take that skipped a source would still look right on its own row count for the sources it
// did visit, and only the next take would show the skipped one re-reporting a delta it had
// already handed over. Asserting the emptiness before the row count puts the observation on
// the property (ADR-0021 Rule 1).
//
// The two sources hold DISJOINT keys here on purpose: this test is about visiting every
// source, and summing is the next test's business. Mixing them would leave a failure
// ambiguous between the two.
void testEverySourceIsTakenFromSoASecondTakeYieldsNothing() {
  unique_ptr<NNCacheFrozen> bOwned = sourceOver({nthKey(101)}, SOURCE_B);
  NNCacheFrozen* b = bOwned.get();
  unique_ptr<NNCacheTwoLevelTable> table =
    makeTwoLevelNNCacheTable(sourceOver({nthKey(100)}, SOURCE_A), defaultLevelOne(8), 8);
  (void)table->attachLevelZero(std::move(bOwned));

  testAssert(served(*table, nthKey(100)) == SOURCE_A);
  testAssert(served(*table, nthKey(100)) == SOURCE_A);
  testAssert(served(*table, nthKey(101)) == SOURCE_B);
  // A level-1 key too, so the table's own composition of the two owners is exercised.
  table->set(sharedOutputFor(nthKey(102), LEVEL_ONE));
  testAssert(served(*table, nthKey(102)) == LEVEL_ONE);

  const NNCacheHitLedger firstTake = table->takeUnpersistedHitCounts();
  const NNCacheHitLedger secondTake = table->takeUnpersistedHitCounts();

  testAssert(secondTake.disposition() == NNCacheHitLedgerDisposition::Counted);
  testAssert(secondTake.entries().empty());
  // Read at the source itself, not only through the table: the LATER source's own mark
  // advanced, which is the thing a skipped visit would leave behind.
  testAssert(b->takeUnpersistedHits().empty());

  testAssert(firstTake.disposition() == NNCacheHitLedgerDisposition::Counted);
  testAssert(firstTake.entries().size() == 3);
  testAssert(hitsForKeyIn(firstTake, nthKey(100)) == 2);
  testAssert(hitsForKeyIn(firstTake, nthKey(101)) == 1);
  testAssert(hitsForKeyIn(firstTake, nthKey(102)) == 1);

  // And the delta resumes from the mark rather than from zero or from the total.
  testAssert(served(*table, nthKey(101)) == SOURCE_B);
  const NNCacheHitLedger thirdTake = table->takeUnpersistedHitCounts();
  testAssert(thirdTake.entries().size() == 1);
  testAssert(hitsForKeyIn(thirdTake, nthKey(101)) == 1);

  cout << "  delta twin: first take " << firstTake.entries().size()
       << " rows across 2 sources and level 1, second take " << secondTake.entries().size()
       << ", third after one more lookup " << thirdTake.entries().size() << endl;
}

// A REORDERED SOURCE'S UNWRITTEN RETRIEVALS ARE SUMMED INTO THE SURVIVING ROW, NOT DROPPED.
//
// THE STATE IS BUILT WITHOUT addHits, deliberately. The point in dispute is whether a
// suppressed source can legitimately carry a non-zero unpersisted count, and a test that
// planted one through a fold-in-counts-from-elsewhere API could be answered with "then do not
// do that". This builds it out of nothing but attach, get and detach -- the class's own
// ordinary interface -- so the state is a history a client can reach, not a poke:
//
//   attach A, which holds the key and serves it twice;
//   attach B, which also holds it and therefore serves nothing;
//   detach A, after which B serves it once;
//   re-attach A, which goes to the BACK because attach appends.
//
// A now sits behind B holding two real retrievals that never reached the count log. Under
// harvest()'s suppress-and-refuse rule those two are a broken invariant. They are not: they
// happened. appendDump ADDS a row's lookups to the key's running total, so the only rule that
// keeps them is to sum -- and summing double-counts nothing, because at most one source
// resolves a key at any instant, so A's two and B's one are disjoint sets of real retrievals.
void testAReorderedSourcesUnwrittenDeltaIsSummedRatherThanLost() {
  const Hash128 shared = nthKey(110);
  unique_ptr<NNCacheFrozen> aOwned = sourceOver({shared, nthKey(111)}, SOURCE_A);
  NNCacheFrozen* a = aOwned.get();
  unique_ptr<NNCacheTwoLevelTable> table =
    makeTwoLevelNNCacheTable(std::move(aOwned), defaultLevelOne(8), 8);
  const NNCacheLevelZeroSourceId idA = table->levelZeroResolutionOrder()[0];

  testAssert(served(*table, shared) == SOURCE_A);
  testAssert(served(*table, shared) == SOURCE_A);   // A has two unwritten retrievals

  const NNCacheLevelZeroSourceId idB = table->attachLevelZero(sourceOver({shared, nthKey(112)}, SOURCE_B));
  unique_ptr<NNCacheFrozen> aBack = table->detachLevelZero(idA);
  testAssert(served(*table, shared) == SOURCE_B);   // B now resolves it, and counts one
  const NNCacheLevelZeroSourceId idA2 = table->attachLevelZero(std::move(aBack));

  // The state under test, observed rather than assumed: A is last, still holds the key
  // unshadowed, and still carries its two.
  testAssert(table->levelZeroResolutionOrder() == vector<NNCacheLevelZeroSourceId>({idB, idA2}));
  testAssert(a->contains(shared));
  testAssert(a->hitCountAt(a->index().find(shared).value()) == 2);

  const NNCacheHitLedger take = table->takeUnpersistedHitCounts();
  // ONE row for the key -- summing must not cost the one-row-per-key property a dump needs.
  testAssert(take.entries().size() == 1);
  testAssert(take.entries()[0].key == shared);
  // THREE: B's one plus A's two. Under a suppress rule this reads 1 and two real retrievals
  // are gone from the count log forever, with no honesty counter to show it.
  testAssert(take.entries()[0].hits == 3);

  // Both marks advanced, so the retrievals are handed over exactly once.
  testAssert(table->takeUnpersistedHitCounts().entries().empty());

  cout << "  reordered source: A served 2 then went behind B which served 1; the take "
       << "carries " << take.entries().size() << " row of "
       << take.entries()[0].hits << " hits, and the second take is empty" << endl;
}

// THE ONE THING THE DELTA TWIN DOES REFUSE BY NAME: a sum that will not fit the row.
//
// A count-log row's lookups field is 32 bits and a frozen counter is 31, so two holders can
// never overflow it but three can. It is unreachable at any real scale -- the largest lifetime
// reference count in the operator's whole corpus is 11,997 -- and it is checked anyway,
// because the alternative to refusing is a silently wrapped delta, which is a count log losing
// four billion retrievals with nothing to show for it. Constructed here rather than waited
// for: addHits is the fold-in-counts-from-elsewhere door, and saturating three holders is the
// only way to reach the branch.
void testASummedDeltaThatWillNotFitARowIsRefusedByName() {
  const Hash128 shared = nthKey(120);
  const uint32_t nearlyFull = 0x7FFFFFFFu;  // NNCacheFrozen's counter is 31 bits
  unique_ptr<NNCacheFrozen> bOwned = sourceOver({shared}, SOURCE_B);
  unique_ptr<NNCacheFrozen> cOwned = sourceOver({shared}, SOURCE_C);
  NNCacheFrozen* b = bOwned.get();
  NNCacheFrozen* c = cOwned.get();
  unique_ptr<NNCacheFrozen> aOwned = sourceOver({shared}, SOURCE_A);
  NNCacheFrozen* a = aOwned.get();
  unique_ptr<NNCacheTwoLevelTable> table =
    makeTwoLevelNNCacheTable(std::move(aOwned), defaultLevelOne(8), 8);
  (void)table->attachLevelZero(std::move(bOwned));
  (void)table->attachLevelZero(std::move(cOwned));

  testAssert(a->addHits(shared, nearlyFull));
  testAssert(b->addHits(shared, nearlyFull));
  testAssert(c->addHits(shared, nearlyFull));

  testAssert(refused([&]() { (void)table->takeUnpersistedHitCounts(); }, "does not fit the 32-bit row"));

  cout << "  overflow refusal: three holders at 2^31-1 each; the take refuses by name rather "
       << "than wrapping the row" << endl;
}

}  // namespace

void Tests::runNNCacheTwoLevelTests() {
  cout << "Running two-level ordered resolution list tests" << endl;
  testFirstMatchInAttachOrderWins();
  testDetachPreservesRelativeOrderAndReAttachGoesToTheBack();
  testASetShadowsTheKeyInEverySourceThatHoldsIt();
  testStaleAndForeignHandlesAreRefused();
  testDetachedSourceIsHandedBackAndReleasable();
  testHarvestEmitsOneRowPerKeyInResolutionOrder();
  testStatsSumEverySource();
  testASuppressedRowCarryingHitsIsRefusedRatherThanDropped();
  testAnEmptyListReportsExactlyLevelOne();
  testAFullyShadowedEarlierSourceStillHoldsItsPlace();
  testEverySourceIsTakenFromSoASecondTakeYieldsNothing();
  testAReorderedSourcesUnwrittenDeltaIsSummedRatherThanLost();
  testASummedDeltaThatWillNotFitARowIsRefusedByName();
  cout << "Done" << endl;
}
