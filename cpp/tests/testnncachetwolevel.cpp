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
#include "../tests/testcacheswapseam.h"

using namespace std;

// The declared test seam for the level-0 swap door; see tests/testcacheswapseam.h. One permit,
// minted once for this file, spent at every attach and detach below.
static const NNCacheLevelZeroSwapPermit SWAP_PERMIT = NNCacheLevelZeroSwapTestSeam::permit();
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

// A SOURCE ATTACHED UNDER NO CONTEXT, which is what most of these tests want: they are about the
// resolution order and the count surfaces, and not about which card a source was loaded for. The
// per-context surfaces have their own tests below, which name a context deliberately.
optional<NNCacheContextId> noContext() {
  return optional<NNCacheContextId>();
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
  const NNCacheLevelZeroSourceId idB = table->attachLevelZero(SWAP_PERMIT, std::move(bOwned), noContext()).id;
  const NNCacheLevelZeroSourceId idC = table->attachLevelZero(SWAP_PERMIT, std::move(cOwned), noContext()).id;
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
  const NNCacheLevelZeroSourceId idB = table->attachLevelZero(SWAP_PERMIT, std::move(bOwned), noContext()).id;
  const NNCacheLevelZeroSourceId idC = table->attachLevelZero(SWAP_PERMIT, std::move(cOwned), noContext()).id;
  const NNCacheLevelZeroSourceId idD = table->attachLevelZero(SWAP_PERMIT, std::move(dOwned), noContext()).id;

  testAssert(table->levelZeroResolutionOrder() == vector<NNCacheLevelZeroSourceId>({idA, idB, idC, idD}));
  testAssert(served(*table, shared) == SOURCE_A);
  testAssert(served(*table, cd) == SOURCE_C);

  // Detach the SECOND of four. A, C and D keep their relative order.
  unique_ptr<NNCacheFrozen> bBack = table->detachLevelZero(SWAP_PERMIT, idB);
  testAssert(bBack != nullptr);
  testAssert(table->levelZeroResolutionOrder() == vector<NNCacheLevelZeroSourceId>({idA, idC, idD}));
  testAssert(served(*table, shared) == SOURCE_A);
  testAssert(served(*table, cd) == SOURCE_C);        // C still precedes D
  testAssert(served(*table, nthKey(12)) == nullopt); // B's own key left with B
  testAssert(served(*table, nthKey(13)) == SOURCE_C);
  testAssert(served(*table, nthKey(15)) == SOURCE_D);

  // Detach the FIRST. C is now the earliest holder of the shared key, so the same key
  // resolves differently -- the ordering observed from the other side.
  unique_ptr<NNCacheFrozen> aBack = table->detachLevelZero(SWAP_PERMIT, idA);
  testAssert(table->levelZeroResolutionOrder() == vector<NNCacheLevelZeroSourceId>({idC, idD}));
  testAssert(served(*table, shared) == SOURCE_C);
  testAssert(served(*table, cd) == SOURCE_C);

  // Re-attach A. It goes to the BACK, so C keeps the shared key and A's own key returns.
  const NNCacheLevelZeroSourceId idA2 = table->attachLevelZero(SWAP_PERMIT, std::move(aBack), noContext()).id;
  testAssert(idA2 != idA);
  testAssert(table->levelZeroResolutionOrder() == vector<NNCacheLevelZeroSourceId>({idC, idD, idA2}));
  testAssert(served(*table, shared) == SOURCE_C);
  testAssert(served(*table, nthKey(11)) == SOURCE_A);

  // Detach everything. The table is coherent with an empty list: it serves from level 1
  // alone, and a source can be attached again afterwards.
  (void)table->detachLevelZero(SWAP_PERMIT, idC);
  (void)table->detachLevelZero(SWAP_PERMIT, idD);
  (void)table->detachLevelZero(SWAP_PERMIT, idA2);
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
  (void)table->attachLevelZero(SWAP_PERMIT, std::move(bOwned), noContext()).id;
  (void)table->attachLevelZero(SWAP_PERMIT, std::move(cOwned), noContext()).id;

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

  const NNCacheLevelZeroSourceId idB = table->attachLevelZero(SWAP_PERMIT, sourceOver({nthKey(31)}, SOURCE_B), noContext()).id;
  (void)table->detachLevelZero(SWAP_PERMIT, idB);

  // Stale: the source it named is gone, and the serial it carries is never reissued, so
  // attaching more sources cannot make this id start naming one of them.
  const NNCacheLevelZeroSourceId idC = table->attachLevelZero(SWAP_PERMIT, sourceOver({nthKey(32)}, SOURCE_C), noContext()).id;
  // Observed at the consequence, not at the id: the claim is that spending the stale handle
  // cannot take the source attached after it, so what is watched is the refusal and then
  // whether C is still there and still serving (ADR-0021 Rule 1).
  testAssert(refused([&]() { (void)table->detachLevelZero(SWAP_PERMIT, idB); }, "already been detached"));
  testAssert(served(*table, nthKey(32)) == SOURCE_C);  // idB's second detach took nothing
  testAssert(idC != idB);

  // Foreign: an id minted by another table's list is refused by name rather than detaching
  // whichever source sits at the same serial here.
  const NNCacheLevelZeroSourceId foreign = other->levelZeroResolutionOrder()[0];
  testAssert(refused([&]() { (void)table->detachLevelZero(SWAP_PERMIT, foreign); }, "different cache"));
  testAssert(table->numLevelZeroSources() == 2);

  // A null source is refused rather than becoming a hole in the resolution order.
  testAssert(refused([&]() { (void)table->attachLevelZero(SWAP_PERMIT, nullptr, noContext()).id; }, "no source was supplied"));
  testAssert(table->numLevelZeroSources() == 2);

  cout << "  handles: stale id refused, foreign id refused, null source refused; "
       << table->numLevelZeroSources() << " sources still attached and serving" << endl;
}

// A DETACHED SOURCE COMES BACK WHOLE, and composes with the loader's own release -- which
// is the only reason detach hands it back instead of destroying it.
void testDetachedSourceIsHandedBackAndReleasable() {
  unique_ptr<NNCacheTwoLevelTable> table =
    makeTwoLevelNNCacheTable(sourceOver({nthKey(40)}, SOURCE_A), defaultLevelOne(8), 8);
  const NNCacheLevelZeroSourceId idB = table->attachLevelZero(SWAP_PERMIT, sourceOver({nthKey(41), nthKey(42)}, SOURCE_B), noContext()).id;

  unique_ptr<NNCacheFrozen> back = table->detachLevelZero(SWAP_PERMIT, idB);
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
  (void)table->attachLevelZero(SWAP_PERMIT, sourceOver({shared, bOnly}, SOURCE_B), noContext()).id;

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

  (void)table->attachLevelZero(SWAP_PERMIT, std::move(bOwned), noContext()).id;
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

// A REORDERED SOURCE'S COUNTS REACH THE HARVEST, AND THE HARVEST DOES NOT REFUSE THEM.
//
// THE SEQUENCE IS THE WITNESS, not a proxy for it (ADR-0021 Rule 1). Nothing here pokes a
// counter: the state is built out of attach, get and detach alone -- the class's own public
// surface -- because the point in dispute is whether a source sitting behind another can
// legitimately carry a count for the key they share.
//
//   attach A, which holds the key and serves it TWICE;
//   attach B, which also holds it and therefore serves nothing;
//   detach A, after which B serves it ONCE;
//   re-attach A, which goes to the BACK because attach appends.
//
// A now sits behind B holding two real retrievals of a key it no longer resolves. The key was
// retrieved three times this session and the harvest says three. Before 2026-08-19 this call
// suppressed A's row and THREW on it carrying a count, naming a one-owner invariant this
// sequence breaks without doing anything wrong -- a refusal firing on a good state.
void testAReorderedSourcesHarvestIsSummedRatherThanRefused() {
  const Hash128 shared = nthKey(70);
  const Hash128 aOnly = nthKey(71);
  const Hash128 bOnly = nthKey(72);
  unique_ptr<NNCacheFrozen> aOwned = sourceOver({shared, aOnly}, SOURCE_A);
  NNCacheFrozen* a = aOwned.get();
  unique_ptr<NNCacheTwoLevelTable> table =
    makeTwoLevelNNCacheTable(std::move(aOwned), defaultLevelOne(8), 8);
  const NNCacheLevelZeroSourceId idA = table->levelZeroResolutionOrder()[0];

  testAssert(served(*table, shared) == SOURCE_A);
  testAssert(served(*table, shared) == SOURCE_A);   // A has counted two

  const NNCacheLevelZeroSourceId idB = table->attachLevelZero(SWAP_PERMIT, sourceOver({shared, bOnly}, SOURCE_B), noContext()).id;
  unique_ptr<NNCacheFrozen> aBack = table->detachLevelZero(SWAP_PERMIT, idA);
  testAssert(served(*table, shared) == SOURCE_B);   // B resolves it now, and counts one
  const NNCacheLevelZeroSourceId idA2 = table->attachLevelZero(SWAP_PERMIT, std::move(aBack), noContext()).id;

  // The state under test, observed rather than assumed: A is last, still holds the key
  // unshadowed, and still carries its two.
  testAssert(table->levelZeroResolutionOrder() == vector<NNCacheLevelZeroSourceId>({idB, idA2}));
  testAssert(a->contains(shared));
  testAssert(a->hitCountAt(a->index().find(shared).value()) == 2);

  // THE OBSERVATION. No throw, one row for the key, and the row carries every retrieval.
  const NNCacheHitLedger ledger = table->harvestHitCounts();
  testAssert(ledger.disposition() == NNCacheHitLedgerDisposition::Counted);
  testAssert(hitsForKeyIn(ledger, shared) == 3);    // B's one plus A's two
  testAssert(ledger.entries().size() == 3);         // shared, bOnly, aOnly -- once each
  testAssert(hitsForKeyIn(ledger, bOnly) == 0);
  testAssert(hitsForKeyIn(ledger, aOnly) == 0);
  // Row position is the earliest holder's, so B's entries come first now that B is first.
  testAssert(ledger.entries()[0].key == shared || ledger.entries()[0].key == bOnly);
  testAssert(ledger.entries()[1].key == shared || ledger.entries()[1].key == bOnly);
  testAssert(ledger.entries()[2].key == aOnly);

  // And it is a report, not a take: reading it twice reads the same level.
  testAssert(hitsForKeyIn(table->harvestHitCounts(), shared) == 3);

  cout << "  reordered source, harvest: A served 2 then went behind B which served 1; the "
       << "harvest carries " << ledger.entries().size() << " rows and "
       << hitsForKeyIn(ledger, shared) << " hits for the shared key, twice running" << endl;
}

// THE ONE THING HARVEST STILL REFUSES: a sum that will not fit the row it must be written to.
//
// This is a limit of the OUTPUT TYPE rather than an invariant about the list's state, which is
// why it cannot fire on a legitimate history the way the retired one-owner tripwire could. A
// count log row's field is 32 bits and a frozen counter is 31, so two holders can never
// overflow it but three can. Unreachable at any real scale -- the largest lifetime reference
// count in the operator's whole corpus is 11,997 -- and refused anyway, because the
// alternative is a wrapped count, which is four billion retrievals lost silently (ADR-0002).
// Constructed with addHits, the fold-in-counts-from-elsewhere door, because saturating three
// holders is the only way to reach the branch.
void testASummedHarvestThatWillNotFitARowIsRefusedByName() {
  const Hash128 shared = nthKey(75);
  const uint32_t nearlyFull = 0x7FFFFFFFu;  // NNCacheFrozen's counter is 31 bits
  unique_ptr<NNCacheFrozen> aOwned = sourceOver({shared}, SOURCE_A);
  unique_ptr<NNCacheFrozen> bOwned = sourceOver({shared}, SOURCE_B);
  unique_ptr<NNCacheFrozen> cOwned = sourceOver({shared}, SOURCE_C);
  NNCacheFrozen* a = aOwned.get();
  NNCacheFrozen* b = bOwned.get();
  NNCacheFrozen* c = cOwned.get();
  unique_ptr<NNCacheTwoLevelTable> table =
    makeTwoLevelNNCacheTable(std::move(aOwned), defaultLevelOne(8), 8);
  (void)table->attachLevelZero(SWAP_PERMIT, std::move(bOwned), noContext()).id;
  (void)table->attachLevelZero(SWAP_PERMIT, std::move(cOwned), noContext()).id;

  // Two holders at the ceiling still fit: the refusal is about the row, so it must not fire
  // one addend early.
  testAssert(a->addHits(shared, nearlyFull));
  testAssert(b->addHits(shared, nearlyFull));
  testAssert(hitsForKeyIn(table->harvestHitCounts(), shared) == 0xFFFFFFFEu);

  testAssert(c->addHits(shared, nearlyFull));
  testAssert(refused([&]() { (void)table->harvestHitCounts(); }, "does not fit the 32-bit row"));

  cout << "  harvest overflow: two holders at 2^31-1 sum to " << 0xFFFFFFFEu
       << " and are reported; a third makes the harvest refuse by name rather than wrap" << endl;
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
  const NNCacheLevelZeroRelease released = nnCacheReleaseLevelZero(table->detachLevelZero(SWAP_PERMIT, idA));
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
  const NNCacheLevelZeroSourceId idB = table->attachLevelZero(SWAP_PERMIT, sourceOver({k0, k1}, SOURCE_B), noContext()).id;

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
  unique_ptr<NNCacheFrozen> aBack = table->detachLevelZero(SWAP_PERMIT, idA);
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
  (void)table->attachLevelZero(SWAP_PERMIT, std::move(bOwned), noContext()).id;

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

  const NNCacheLevelZeroSourceId idB = table->attachLevelZero(SWAP_PERMIT, sourceOver({shared, nthKey(112)}, SOURCE_B), noContext()).id;
  unique_ptr<NNCacheFrozen> aBack = table->detachLevelZero(SWAP_PERMIT, idA);
  testAssert(served(*table, shared) == SOURCE_B);   // B now resolves it, and counts one
  const NNCacheLevelZeroSourceId idA2 = table->attachLevelZero(SWAP_PERMIT, std::move(aBack), noContext()).id;

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
  (void)table->attachLevelZero(SWAP_PERMIT, std::move(bOwned), noContext()).id;
  (void)table->attachLevelZero(SWAP_PERMIT, std::move(cOwned), noContext()).id;

  testAssert(a->addHits(shared, nearlyFull));
  testAssert(b->addHits(shared, nearlyFull));
  testAssert(c->addHits(shared, nearlyFull));

  testAssert(refused([&]() { (void)table->takeUnpersistedHitCounts(); }, "does not fit the 32-bit row"));

  cout << "  overflow refusal: three holders at 2^31-1 each; the take refuses by name rather "
       << "than wrapping the row" << endl;
}

//-------------------------------------------------------------------------------------
// The attach contract: a source rejoining the list cannot serve what level 1 has superseded
//-------------------------------------------------------------------------------------

// THE DEFECT THIS CLOSES, AND THE SEQUENCE THAT REACHED IT. A set shadows the key in every
// ATTACHED holder. A source that was DETACHED at that moment is not one, so before this fix it
// came back holding the key unshadowed, sitting on the resolution list AHEAD of the level 1 that
// now owned the key -- and served the evaluation the set had superseded.
//
// THE OBSERVATION IS THE VALUE SERVED THROUGH THE PUBLIC SURFACE, not a shadow bit and not a
// count: serving a superseded evaluation into a running search is the harm, so that is what is
// watched (ADR-0021 Rule 1). The evaluations are tagged, so "which one came back" is a fact this
// test can read rather than infer.
//
// Every step is the class's own ordinary interface -- attach, get, detach, set -- so the state is
// a history a client reaches at a session boundary, not a poke.
void testAReAttachedSourceCannotServeWhatLevelOneOwns() {
  const Hash128 shared = nthKey(130);
  const Hash128 aOnly = nthKey(131);
  unique_ptr<NNCacheFrozen> aOwned = sourceOver({shared, aOnly}, SOURCE_A);
  NNCacheFrozen* a = aOwned.get();
  unique_ptr<NNCacheTwoLevelTable> table =
    makeTwoLevelNNCacheTable(std::move(aOwned), defaultLevelOne(8), 8);
  const NNCacheLevelZeroSourceId idA = table->levelZeroResolutionOrder()[0];

  // A serves the key and counts it, so the source that will be detached carries real,
  // unpersisted retrievals -- the thing the reconcile must move rather than discard.
  testAssert(served(*table, shared) == SOURCE_A);
  testAssert(served(*table, shared) == SOURCE_A);

  const NNCacheLevelZeroSourceId idB =
    table->attachLevelZero(SWAP_PERMIT, sourceOver({shared, nthKey(132)}, SOURCE_B), noContext()).id;
  unique_ptr<NNCacheFrozen> aBack = table->detachLevelZero(SWAP_PERMIT, idA);
  testAssert(served(*table, shared) == SOURCE_B);

  // THE SET A IS ABSENT FOR. It shadows the key in B, which is attached, and cannot reach A,
  // which is not. Level 1 owns the key from here on.
  table->set(sharedOutputFor(shared, LEVEL_ONE));
  testAssert(served(*table, shared) == LEVEL_ONE);

  // THE RE-ATTACH. Under the old contract A returned holding the key unshadowed.
  const NNCacheLevelZeroAttachment reattached =
    table->attachLevelZero(SWAP_PERMIT, std::move(aBack), noContext());
  testAssert(table->levelZeroResolutionOrder() ==
             vector<NNCacheLevelZeroSourceId>({idB, reattached.id}));

  // THE OBSERVATION: what the table SERVES. Under the defect this is SOURCE_A -- the frozen
  // evaluation, ahead of the fresher one level 1 holds.
  testAssert(served(*table, shared) == LEVEL_ONE);
  // And the same fact read at the source: its entry for the shared key is retired, while the key
  // it alone holds is untouched -- the reconcile shadowed what level 1 owned and nothing else.
  testAssert(!a->contains(shared));
  testAssert(a->contains(aOnly));
  testAssert(served(*table, aOnly) == SOURCE_A);

  // WHAT THE RECONCILE REPORTS, so a client is not left to infer it: one entry level 1 already
  // owned, and the two retrievals it had accrued and never written handed over.
  testAssert(reattached.entriesLevelOneAlreadyOwned == 1);
  testAssert(reattached.hitsTransferredToLevelOne == 2);

  // AND THE KEY APPEARS ONCE ACROSS THE COMPOSED HARVEST, carrying every retrieval of it:
  // A's two, B's one, and the two level 1 has served since the set. Under the defect it appeared
  // twice -- once from A, once from the level-1 ledger -- and a dump would have raised its
  // sessions twice for one dump.
  const NNCacheHitLedger ledger = table->harvestHitCounts();
  testAssert(hitsForKeyIn(ledger, shared) == 5);   // hitsForKeyIn asserts one row per key
  int rowsForShared = 0;
  for(size_t i = 0; i < ledger.entries().size(); i++)
    rowsForShared += (ledger.entries()[i].key == shared) ? 1 : 0;
  testAssert(rowsForShared == 1);

  cout << "  attach reconciles: a source detached across a set comes back with " << reattached.entriesLevelOneAlreadyOwned
       << " entry shadowed and " << reattached.hitsTransferredToLevelOne
       << " hits handed to level 1; the table serves " << served(*table, shared).value()
       << " and the key has " << rowsForShared << " row carrying " << hitsForKeyIn(ledger, shared)
       << " hits" << endl;
}

// THE AXIS THE RECONCILE DELIBERATELY DOES NOT COVER, closed at the other seam.
//
// attach asks what level 1 HOLDS, because only an entry level 1 can still serve could be served
// wrongly. A key level 1 has EVICTED is a different case: level 1 cannot answer for it, the
// arriving source can, and shadowing it would throw away pre-warmed content for nobody's benefit.
// But the level-1 LEDGER ROW OUTLIVES THE ENTRY on purpose -- "this key was hot this session" is
// the fact a persistence layer wants and a capacity sweep does not make it untrue -- so that key
// legitimately sits on both halves of the composed count surface.
//
// Concatenated, it is two rows for one key, and appendDump would raise its sessions twice for one
// dump. Folded, it is one row carrying both disjoint sets of real retrievals. This watches the
// composed surface, which is what a dump actually receives.
void testAKeyLevelOneEvictedAndThenCarriedBackInAppearsOnce() {
  const Hash128 hot = nthKey(140);
  const Hash128 other = nthKey(141);
  // A ONE-SLOT LEVEL 1, so eviction is deterministic and needs no capacity heuristics: the next
  // set displaces whatever is there.
  unique_ptr<NNCacheTwoLevelTable> table =
    makeTwoLevelNNCacheTable(sourceOver({nthKey(149)}, SOURCE_C), defaultLevelOne(0), 8);

  table->set(sharedOutputFor(hot, LEVEL_ONE));
  testAssert(served(*table, hot) == LEVEL_ONE);   // a level-1 hit, so the ledger has a row
  table->set(sharedOutputFor(other, LEVEL_ONE));  // and the one slot is now other's
  shared_ptr<NNOutput> gone;
  testAssert(!table->peek(hot, gone));            // level 1 really has evicted it

  // The card comes back, holding the key level 1 no longer has. The reconcile leaves it alone,
  // which is the point: level 0 can serve it and level 1 cannot.
  const NNCacheLevelZeroAttachment attachment =
    table->attachLevelZero(SWAP_PERMIT, sourceOver({hot}, SOURCE_A), noContext());
  testAssert(attachment.entriesLevelOneAlreadyOwned == 0);
  testAssert(served(*table, hot) == SOURCE_A);

  // THE OBSERVATION: one row on the composed surface, carrying the level-1 retrieval AND the
  // level-0 one. Two rows here is the defect.
  const NNCacheHitLedger ledger = table->harvestHitCounts();
  int rowsForHot = 0;
  for(size_t i = 0; i < ledger.entries().size(); i++)
    rowsForHot += (ledger.entries()[i].key == hot) ? 1 : 0;
  testAssert(rowsForHot == 1);
  testAssert(hitsForKeyIn(ledger, hot) == 2);

  // And the delta surface folds identically -- the one that matters most, because these rows are
  // increments a count log ADDS.
  const NNCacheHitLedger take = table->takeUnpersistedHitCounts();
  int deltaRowsForHot = 0;
  for(size_t i = 0; i < take.entries().size(); i++)
    deltaRowsForHot += (take.entries()[i].key == hot) ? 1 : 0;
  testAssert(deltaRowsForHot == 1);
  testAssert(hitsForKeyIn(take, hot) == 2);

  cout << "  evicted-then-carried-back: level 1 dropped the entry but kept its ledger row; the "
       << "composed harvest carries " << rowsForHot << " row of " << hitsForKeyIn(ledger, hot)
       << " hits and the delta " << deltaRowsForHot << " row of " << hitsForKeyIn(take, hot) << endl;
}

//-------------------------------------------------------------------------------------
// The per-context delta
//-------------------------------------------------------------------------------------

// TWO CARDS ATTACHED AT ONCE, EACH DUMPABLE ON ITS OWN. The whole-table delta cannot be divided
// by any caller -- a row carries a key, and a key names a position, never a card -- so before
// this surface existed a session with two contexts attached could not dump counts at all.
//
// The division is watched from both sides: A's take must carry A's retrievals and NOT B's, and
// B's take afterwards must still carry B's whole delta rather than one that A's take had already
// spent. Both halves are exercised, level 0 and level 1: a card's own pre-warmed entries serve
// retrievals that never call set(), and the keys it earns are recorded against it by attribution.
void testTheDeltaDividesByContextAndTakesOnlyItsOwn() {
  const Hash128 aZero = nthKey(150);   // in A's level-0 source
  const Hash128 bZero = nthKey(151);   // in B's level-0 source
  const Hash128 aEarned = nthKey(152); // set under A
  const Hash128 bEarned = nthKey(153); // set under B

  unique_ptr<NNCacheTwoLevelTable> table =
    makeTwoLevelNNCacheTable(NNCacheFrozen::build(vector<unique_ptr<NNOutput>>()), defaultLevelOne(8), 8);
  const NNCacheLevelZeroSourceId placeholder = table->levelZeroResolutionOrder()[0];
  (void)table->detachLevelZero(SWAP_PERMIT, placeholder);

  const NNCacheContextId cardA = table->attachCacheContext("card-a");
  const NNCacheContextId cardB = table->attachCacheContext("card-b");
  (void)table->attachLevelZero(SWAP_PERMIT, sourceOver({aZero}, SOURCE_A), optional<NNCacheContextId>(cardA)).id;
  (void)table->attachLevelZero(SWAP_PERMIT, sourceOver({bZero}, SOURCE_B), optional<NNCacheContextId>(cardB)).id;

  // Each card earns one key and each serves one out of its own level 0. The level-0 retrievals
  // are the ones no attribution can see: they call no set().
  table->set(sharedOutputFor(aEarned, LEVEL_ONE), NNCacheAttribution::toContext(cardA));
  table->set(sharedOutputFor(bEarned, LEVEL_ONE), NNCacheAttribution::toContext(cardB));
  testAssert(served(*table, aZero) == SOURCE_A);
  testAssert(served(*table, aZero) == SOURCE_A);
  testAssert(served(*table, aEarned) == LEVEL_ONE);
  testAssert(served(*table, bZero) == SOURCE_B);
  testAssert(served(*table, bEarned) == LEVEL_ONE);
  testAssert(served(*table, bEarned) == LEVEL_ONE);

  // A'S DUMP. Two rows, both A's, carrying both halves: the pre-warmed key retrieved twice out of
  // level 0, and the earned key retrieved once out of level 1.
  const NNCacheHitLedger takeA = table->takeUnpersistedHitCountsFor(cardA);
  testAssert(takeA.disposition() == NNCacheHitLedgerDisposition::Counted);
  testAssert(takeA.entries().size() == 2);
  testAssert(hitsForKeyIn(takeA, aZero) == 2);
  testAssert(hitsForKeyIn(takeA, aEarned) == 1);
  // AND NOT B'S. Watched as an absence made positive: B's two keys have no row here at all.
  for(size_t i = 0; i < takeA.entries().size(); i++)
    testAssert(takeA.entries()[i].key != bZero && takeA.entries()[i].key != bEarned);

  // A SECOND TAKE FOR A, WITH NOTHING IN BETWEEN, YIELDS NOTHING. Not "less" -- nothing.
  testAssert(table->takeUnpersistedHitCountsFor(cardA).entries().empty());

  // AND B'S DELTA IS STILL WHOLE. A's take advanced A's marks and no others, which is the
  // difference between a per-context take and a filter over a whole-table one: a filter would
  // have consumed B's rows and dropped them on the floor.
  const NNCacheHitLedger takeB = table->takeUnpersistedHitCountsFor(cardB);
  testAssert(takeB.entries().size() == 2);
  testAssert(hitsForKeyIn(takeB, bZero) == 1);
  testAssert(hitsForKeyIn(takeB, bEarned) == 2);
  testAssert(table->takeUnpersistedHitCountsFor(cardB).entries().empty());

  // The delta resumes from the mark rather than from zero or from the total.
  testAssert(served(*table, aZero) == SOURCE_A);
  const NNCacheHitLedger takeAgain = table->takeUnpersistedHitCountsFor(cardA);
  testAssert(takeAgain.entries().size() == 1);
  testAssert(hitsForKeyIn(takeAgain, aZero) == 1);

  // An id from another table's name space is refused rather than indexing this one's.
  unique_ptr<NNCacheTwoLevelTable> other =
    makeTwoLevelNNCacheTable(sourceOver({nthKey(159)}, SOURCE_C), defaultLevelOne(8), 8);
  const NNCacheContextId foreign = other->attachCacheContext("card-a");
  testAssert(refused([&]() { (void)table->takeUnpersistedHitCountsFor(foreign); }, "not attached to this cache"));
  testAssert(refused(
    [&]() { (void)table->attachLevelZero(SWAP_PERMIT, sourceOver({nthKey(158)}, SOURCE_C), optional<NNCacheContextId>(foreign)); },
    "attached to a different cache"
  ));

  cout << "  per-context delta: A's take " << takeA.entries().size() << " rows (level 0 and level 1, "
       << "none of B's), A's second take " << table->takeUnpersistedHitCountsFor(cardA).entries().size()
       << ", B's take still " << takeB.entries().size() << " rows" << endl;
}

//-------------------------------------------------------------------------------------
// The non-consuming question
//-------------------------------------------------------------------------------------

// A PEEK THAT REPORTS TRUTHFULLY AND ADVANCES NOTHING.
//
// The property is stated as an equality between two runs and is watched that way: a peek
// followed by a take yields EXACTLY what the take would have yielded without the peek. So the
// same history is built twice, side by side, peeked in one and not in the other, and the two
// takes are compared row for row. A peek that advanced a mark would show up as a shorter take on
// the peeked side, which no assertion about the peek's own return value could catch
// (ADR-0021 Rule 1).
//
// The state it is peeked in is deliberately the one the old protocol-layer proxy was blind to:
// retrievals served ENTIRELY OUT OF LEVEL 0, which call no set(), earn no attributed key, and
// leave the level-1 ledger untouched.
void testThePeekReportsTruthfullyAndAdvancesNothing() {
  const Hash128 zero = nthKey(160);
  const Hash128 earned = nthKey(161);

  struct Fixture {
    unique_ptr<NNCacheTwoLevelTable> table;
    optional<NNCacheContextId> card;
  };
  const std::function<Fixture()> build = [&]() {
    Fixture f;
    f.table = makeTwoLevelNNCacheTable(NNCacheFrozen::build(vector<unique_ptr<NNOutput>>()), defaultLevelOne(8), 8);
    (void)f.table->detachLevelZero(SWAP_PERMIT, f.table->levelZeroResolutionOrder()[0]);
    f.card = f.table->attachCacheContext("card-peek");
    (void)f.table->attachLevelZero(SWAP_PERMIT, sourceOver({zero}, SOURCE_A), f.card).id;
    f.table->set(sharedOutputFor(earned, LEVEL_ONE), NNCacheAttribution::toContext(f.card.value()));
    testAssert(served(*f.table, zero) == SOURCE_A);
    testAssert(served(*f.table, zero) == SOURCE_A);
    testAssert(served(*f.table, earned) == LEVEL_ONE);
    return f;
  };

  Fixture peeked = build();
  Fixture unpeeked = build();

  // The peek, on one of the two only.
  testAssert(peeked.table->hasUnpersistedHitCountsFor(peeked.card.value()));
  testAssert(peeked.table->hasUnpersistedHitCountsFor(peeked.card.value()));  // and again, idempotent

  const NNCacheHitLedger afterPeek = peeked.table->takeUnpersistedHitCountsFor(peeked.card.value());
  const NNCacheHitLedger withoutPeek = unpeeked.table->takeUnpersistedHitCountsFor(unpeeked.card.value());
  testAssert(afterPeek.entries().size() == withoutPeek.entries().size());
  testAssert(afterPeek.entries().size() == 2);
  testAssert(hitsForKeyIn(afterPeek, zero) == hitsForKeyIn(withoutPeek, zero));
  testAssert(hitsForKeyIn(afterPeek, earned) == hitsForKeyIn(withoutPeek, earned));
  testAssert(hitsForKeyIn(afterPeek, zero) == 2);
  testAssert(hitsForKeyIn(afterPeek, earned) == 1);

  // AND IT REPORTS FALSE WHEN THERE IS NOTHING, which is the half that makes the true above mean
  // something: after the take, both halves are written and the peek says so.
  testAssert(!peeked.table->hasUnpersistedHitCountsFor(peeked.card.value()));
  // A single level-0 retrieval -- no set, no earned key, no ledger row -- arms it again. This is
  // exactly the state the retired protocol proxy read as "nothing to lose".
  testAssert(served(*peeked.table, zero) == SOURCE_A);
  testAssert(peeked.table->hasUnpersistedHitCountsFor(peeked.card.value()));
  // ...and a level-1 retrieval alone arms it too, so neither half is carrying the other.
  const NNCacheHitLedger drain = peeked.table->takeUnpersistedHitCountsFor(peeked.card.value());
  testAssert(drain.entries().size() == 1);
  testAssert(!peeked.table->hasUnpersistedHitCountsFor(peeked.card.value()));
  testAssert(served(*peeked.table, earned) == LEVEL_ONE);
  testAssert(peeked.table->hasUnpersistedHitCountsFor(peeked.card.value()));

  // A single-level table keeps no counts, so it has none unpersisted -- and says so rather than
  // hedging, while still refusing a context it never attached.
  unique_ptr<NNCacheTable> single = defaultLevelOne(8);
  const NNCacheContextId singleCard = single->attachCacheContext("card-single");
  testAssert(!single->hasUnpersistedHitCountsFor(singleCard));

  cout << "  peek: true on level-0-only retrievals, and the take after it carries "
       << afterPeek.entries().size() << " rows -- identical row for row to the same history never "
       << "peeked; false once everything is written" << endl;
}

}  // namespace

// A LEVEL 1 THAT FAILS ON DEMAND, so a throw can be forced at the exact point of the attach
// reconcile that can really throw.
//
// The reconcile asks level 1 about every arriving key through NNCacheTable::contains, and that
// call is the reconcile's only fallible step: it is a virtual call onto whatever table shape the
// operator configured. This decorator wraps a real level 1 and forwards everything, except that
// the Nth contains() throws instead of answering. The fault is injected at the site the claim is
// about rather than simulated somewhere adjacent (ADR-0021 Rule 1), and it is a real throw out of
// a real call in the walk, not a flag the code under test consults.
class FaultInjectingLevelOne final : public NNCacheTable {
 public:
  explicit FaultInjectingLevelOne(unique_ptr<NNCacheTable> inner) : inner_(std::move(inner)), containsCalls_(0), throwOnCall_(0) {}

  // 1-based; zero disarms. Resets the call counter so a test can arm, attach, and arm again.
  void throwOnContainsCall(int64_t n) { throwOnCall_ = n; containsCalls_ = 0; }
  int64_t containsCalls() const { return containsCalls_; }

  bool contains(Hash128 nnHash) const override {
    containsCalls_ += 1;
    if(throwOnCall_ != 0 && containsCalls_ == throwOnCall_)
      throw StringError("forced fault: level 1 refused to answer contains() mid-reconcile");
    return inner_->contains(nnHash);
  }

  bool get(Hash128 nnHash, shared_ptr<NNOutput>& ret) override { return inner_->get(nnHash, ret); }
  void set(const shared_ptr<NNOutput>& p) override { inner_->set(p); }
  void clear() override { inner_->clear(); }
  NNCacheStats stats() const override { return inner_->stats(); }

 private:
  unique_ptr<NNCacheTable> inner_;
  mutable int64_t containsCalls_;
  int64_t throwOnCall_;
};

// THE ATTACH IS ALL-OR-NOTHING: a throw partway through the reconcile leaves the table exactly as
// it was, and the source is not attached.
//
// WHAT THE DEFECT WAS. The reconcile walks the arriving source and shadows every key level 1
// already owns, restoring "at most one level owns any key". Asking level 1 is the step that can
// throw. In one loop, a throw at entry k left the first k keys shadowed -- their evaluations
// retired, their unpersisted retrievals already moved into level 1's ledger -- and the rest
// untouched, for an attach that never happened, with nothing recording it. The operation whose
// whole purpose is restoring an invariant left it half-restored.
//
// THE OBSERVATION IS ON STATE THAT SURVIVES THE FAILED ATTACH, through the public surface. The
// arriving source is owned by the attach and dies with the throw, so the surviving witness of "an
// entry was shadowed" is where a shadowed entry's retrievals GO: level 1's counter, read through
// harvestHitCounts. Every key level 1 owns here carries retrievals in the arriving source, so
// "shadowed" and "its hits arrived in the ledger" are the same event and the ledger observes it
// exactly. Three legs, all through the ordinary interface:
//   (1) the resolution list is unchanged and the source is not on it;
//   (2) the composed hit counts are byte-for-byte what they were before the failed attach;
//   (3) a later, clean attach of an identical source transfers the FULL hit total -- so nothing
//       was skimmed off by the attempt that failed, which (2) alone could not distinguish from a
//       transfer that happened to land on a key with no row.
void testAThrowMidReconcileLeavesTheAttachAsIfItNeverHappened() {
  const Hash128 owned0 = nthKey(170);
  const Hash128 owned1 = nthKey(171);
  const Hash128 owned2 = nthKey(172);

  // A source that has really served, so its entries carry unpersisted retrievals for the
  // reconcile to move. It is built by attaching to a scratch table, serving each key, and
  // detaching -- the ordinary session-boundary history, not a poke at the counters.
  const auto sourceWithHits = [&]() {
    unique_ptr<NNCacheFrozen> owned = sourceOver({owned0, owned1, owned2}, SOURCE_B);
    unique_ptr<NNCacheTwoLevelTable> scratch = makeTwoLevelNNCacheTable(std::move(owned), defaultLevelOne(8), 8);
    const NNCacheLevelZeroSourceId id = scratch->levelZeroResolutionOrder()[0];
    testAssert(served(*scratch, owned0) == SOURCE_B);
    testAssert(served(*scratch, owned1) == SOURCE_B);
    testAssert(served(*scratch, owned1) == SOURCE_B);
    testAssert(served(*scratch, owned2) == SOURCE_B);
    return scratch->detachLevelZero(SWAP_PERMIT, id);
  };
  // 1 + 2 + 1 retrievals, all unpersisted, all on keys level 1 will own below.
  const uint32_t TOTAL_HITS_TO_TRANSFER = 4;

  // The table under test: one placeholder source, and a level 1 that can be made to fail.
  unique_ptr<FaultInjectingLevelOne> faultyOwned(new FaultInjectingLevelOne(defaultLevelOne(8)));
  FaultInjectingLevelOne* faulty = faultyOwned.get();
  unique_ptr<NNCacheTwoLevelTable> table =
    makeTwoLevelNNCacheTable(sourceOver({nthKey(179)}, SOURCE_A), std::move(faultyOwned), 8);
  const vector<NNCacheLevelZeroSourceId> orderBefore = table->levelZeroResolutionOrder();
  testAssert(orderBefore.size() == 1);

  // Level 1 takes ownership of all three keys, and earns retrievals of its own on one of them so
  // the ledger is not empty before the failed attach -- an unchanged empty ledger would be a
  // weaker observation than an unchanged populated one.
  table->set(sharedOutputFor(owned0, LEVEL_ONE));
  table->set(sharedOutputFor(owned1, LEVEL_ONE));
  table->set(sharedOutputFor(owned2, LEVEL_ONE));
  testAssert(served(*table, owned0) == LEVEL_ONE);
  testAssert(served(*table, owned0) == LEVEL_ONE);

  const NNCacheHitLedger before = table->harvestHitCounts();
  const uint32_t owned0Before = hitsForKeyIn(before, owned0);
  const uint32_t owned1Before = hitsForKeyIn(before, owned1);
  const uint32_t owned2Before = hitsForKeyIn(before, owned2);
  testAssert(owned0Before == 2);

  // THE FORCED THROW, on the SECOND key the reconcile asks about. Under one loop the first key is
  // already shadowed and its retrieval already in level 1's ledger by the time this fires.
  faulty->throwOnContainsCall(2);
  unique_ptr<NNCacheFrozen> arriving = sourceWithHits();
  testAssert(refused(
    [&]() { (void)table->attachLevelZero(SWAP_PERMIT, std::move(arriving), noContext()); },
    "forced fault"
  ));
  // Read the count BEFORE disarming, since disarming resets it: the walk got exactly as far as
  // the injected fault and no further.
  testAssert(faulty->containsCalls() == 2);
  faulty->throwOnContainsCall(0);

  // (1) NOTHING WAS ATTACHED.
  testAssert(table->numLevelZeroSources() == 1);
  testAssert(table->levelZeroResolutionOrder() == orderBefore);

  // (2) NO KEY WAS SHADOWED THAT WAS NOT SHADOWED BEFORE, read where a shadow's retrievals land.
  const NNCacheHitLedger after = table->harvestHitCounts();
  testAssert(hitsForKeyIn(after, owned0) == owned0Before);
  testAssert(hitsForKeyIn(after, owned1) == owned1Before);
  testAssert(hitsForKeyIn(after, owned2) == owned2Before);
  testAssert(after.entries().size() == before.entries().size());

  // (3) AND THE WHOLE TRANSFER IS STILL THERE TO BE MADE. An identical source, attached cleanly,
  // hands over every one of the four retrievals -- so the failed attempt took none of them.
  const NNCacheLevelZeroAttachment clean =
    table->attachLevelZero(SWAP_PERMIT, sourceWithHits(), noContext());
  testAssert(clean.entriesLevelOneAlreadyOwned == 3);
  testAssert(clean.hitsTransferredToLevelOne == (int64_t)TOTAL_HITS_TO_TRANSFER);
  testAssert(table->numLevelZeroSources() == 2);

  cout << "  a throw at contains() call 2 of the reconcile left " << table->numLevelZeroSources() - 1
       << " source(s) attached and the level-1 counts at " << hitsForKeyIn(after, owned0) << "/"
       << hitsForKeyIn(after, owned1) << "/" << hitsForKeyIn(after, owned2)
       << ", and the clean attach that followed transferred all "
       << clean.hitsTransferredToLevelOne << " retrievals" << endl;
}

void Tests::runNNCacheTwoLevelTests() {
  cout << "Running two-level ordered resolution list tests" << endl;
  testFirstMatchInAttachOrderWins();
  testDetachPreservesRelativeOrderAndReAttachGoesToTheBack();
  testASetShadowsTheKeyInEverySourceThatHoldsIt();
  testStaleAndForeignHandlesAreRefused();
  testDetachedSourceIsHandedBackAndReleasable();
  testHarvestEmitsOneRowPerKeyInResolutionOrder();
  testStatsSumEverySource();
  testAReorderedSourcesHarvestIsSummedRatherThanRefused();
  testASummedHarvestThatWillNotFitARowIsRefusedByName();
  testAnEmptyListReportsExactlyLevelOne();
  testAFullyShadowedEarlierSourceStillHoldsItsPlace();
  testEverySourceIsTakenFromSoASecondTakeYieldsNothing();
  testAReorderedSourcesUnwrittenDeltaIsSummedRatherThanLost();
  testASummedDeltaThatWillNotFitARowIsRefusedByName();
  testAReAttachedSourceCannotServeWhatLevelOneOwns();
  testAKeyLevelOneEvictedAndThenCarriedBackInAppearsOnce();
  testTheDeltaDividesByContextAndTakesOnlyItsOwn();
  testThePeekReportsTruthfullyAndAdvancesNothing();
  testAThrowMidReconcileLeavesTheAttachAsIfItNeverHappened();
  cout << "Done" << endl;
}
