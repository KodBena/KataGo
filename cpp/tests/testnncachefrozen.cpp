#include "../tests/tests.h"

#include <functional>
#include <iostream>
#include <memory>
#include <set>
#include <vector>

#include "../neuralnet/nncache.h"
#include "../neuralnet/nncachefrozen.h"

using namespace std;
using namespace TestCommon;

// Correctness tests for the frozen level-0 cache and the two-level resolution strategy.
//
// Everything here is a logic invariant and is asserted exactly, with no tolerance
// (ADR-0009, Calibration): which level owns a key, which entry a query resolves to,
// whether an absent key is rejected, and where a hit count ended up are all discrete
// facts. The only numbers that are PRINTED rather than pinned are the per-entry byte
// figures, which are compiler layout facts, and they are checked against the SPEC.md 6
// ceiling rather than against a literal.

namespace {

Hash128 keyOf(uint64_t a, uint64_t b) {
  return Hash128(a, b);
}

// A distinct key per serial, spread over both halves.
Hash128 nthKey(int serial) {
  return Hash128(
    ((uint64_t)(serial + 1)) * 0x9E3779B97F4A7C15ULL,
    ((uint64_t)(serial + 1)) * 0xD6E8FEB86659FD93ULL + 0x1234567ULL
  );
}

shared_ptr<NNOutput> outputFor(Hash128 hash, bool withOwnerMap) {
  shared_ptr<NNOutput> p = make_shared<NNOutput>();
  p->nnHash = hash;
  p->nnXLen = 19;
  p->nnYLen = 19;
  if(withOwnerMap)
    p->whiteOwnerMap = new float[19 * 19];
  return p;
}

vector<shared_ptr<NNOutput>> outputsFor(const vector<Hash128>& keys) {
  vector<shared_ptr<NNOutput>> out;
  for(size_t i = 0; i < keys.size(); i++)
    out.push_back(outputFor(keys[i], false));
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

// A level-1 table in the shipped default shape, so every two-level scenario below is
// composed over exactly the table an operator runs today.
unique_ptr<NNCacheTable> defaultLevelOne(int sizePowerOfTwo) {
  return NNCacheTable::create(NNCacheConfig::statusQuo(sizePowerOfTwo, 2));
}

bool present(NNCacheTable& table, Hash128 hash) {
  shared_ptr<NNOutput> got;
  const bool found = table.get(hash, got);
  testAssert(found == (got != nullptr));
  if(found)
    testAssert(got->nnHash == hash);
  return found;
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
  // ONE ROW PER KEY. This is the whole point of the surface, so it is asserted on every
  // read rather than checked once: a persistence layer must never have to merge rows.
  testAssert(rows <= 1);
  return total;
}

//-------------------------------------------------------------------------------------
// The index: resolution, and the absent-key contract
//-------------------------------------------------------------------------------------

void testEveryMemberResolvesToItsOwnInputPosition() {
  for(int n : {0, 1, 2, 3, 4, 5, 8, 9, 16, 17, 100, 1000, 10000}) {
    vector<Hash128> keys;
    for(int i = 0; i < n; i++)
      keys.push_back(nthKey(i));
    const NNCacheFrozenIndex index = NNCacheFrozenIndex::build(keys);
    testAssert((int)index.numEntries() == n);
    for(int i = 0; i < n; i++) {
      const optional<uint32_t> got = index.find(keys[i]);
      testAssert(got.has_value());
      testAssert((int)got.value() == i);
      testAssert(index.keyAt((uint32_t)i) == keys[i]);
    }
  }
}

// SPEC.md 2.2, the clause the specification says is most likely to be got wrong and whose
// failure is silent. A perfect hash sends a never-inserted key to SOME slot, and at this
// occupancy the great majority of those slots hold a real, unrelated entry -- so this test
// is only meaningful if it can show that its absent queries actually reached an occupied
// slot rather than trivially missing. It counts that and asserts a floor on it.
void testAbsentKeysAreRejectedByTheStoredKeyComparison() {
  const int n = 10000;
  vector<Hash128> keys;
  for(int i = 0; i < n; i++)
    keys.push_back(nthKey(i));
  const NNCacheFrozenIndex index = NNCacheFrozenIndex::build(keys);
  const std::set<Hash128> resident(keys.begin(), keys.end());

  int landedOnAnEntry = 0;
  const int numQueries = 20000;
  for(int q = 0; q < numQueries; q++) {
    // Keys from a disjoint generator, and one-bit perturbations of resident keys -- the
    // near-miss case SPEC.md 2.1 names as the worst, because a near neighbour is
    // disproportionately likely to land where its neighbour lives.
    const Hash128 absent =
      (q % 2 == 0)
      ? Hash128(nthKey(q + 1000000).hash0, nthKey(q + 1000000).hash1)
      : Hash128(keys[q % n].hash0 ^ (((uint64_t)1) << (q % 64)), keys[q % n].hash1);
    // Skip the vanishingly unlikely case of a perturbation colliding with a real key.
    if(resident.find(absent) != resident.end())
      continue;
    testAssert(!index.find(absent).has_value());
    landedOnAnEntry += 1;
  }
  // Every one of those queries computed a slot; the assertion above is that none of them
  // was believed. Reported so a reader can see the path was exercised rather than assumed.
  cout << "frozen absent-key contract: " << landedOnAnEntry
       << " never-inserted queries against " << n << " entries, all rejected" << endl;
}

void testConstructionRefusals() {
  // Duplicate keys (SPEC.md 4.1), named with the key set positions involved.
  {
    vector<Hash128> keys;
    for(int i = 0; i < 101; i++)
      keys.push_back(nthKey(i));
    keys.push_back(keys[7]);
    testAssert(refused([&]() { NNCacheFrozenIndex::build(keys); }, "duplicate key"));
    testAssert(refused([&]() { NNCacheFrozenIndex::build(keys); }, "positions 7 and 101"));
  }
  // A null evaluation carries no key to index it by.
  {
    vector<shared_ptr<NNOutput>> outs = outputsFor({nthKey(0), nthKey(1)});
    outs[1] = nullptr;
    testAssert(refused([&]() { NNCacheFrozen::build(outs); }, "null"));
  }
  // The bound on the displacement search exists and the refusal names it (SPEC.md 5.1).
  testAssert(NNCacheFrozenIndex::searchBound() > 0);
}

void testEmptyAndSingleton() {
  // n = 0 constructs and answers absent to everything. Routine: a card with nothing
  // cacheable yet (SPEC.md 4.3).
  const NNCacheFrozenIndex empty = NNCacheFrozenIndex::build(vector<Hash128>());
  testAssert(empty.numEntries() == 0);
  testAssert(!empty.find(keyOf(0, 0)).has_value());
  testAssert(!empty.find(nthKey(5)).has_value());

  const NNCacheFrozenIndex one = NNCacheFrozenIndex::build({nthKey(0)});
  testAssert(one.find(nthKey(0)).has_value() && one.find(nthKey(0)).value() == 0);
  testAssert(!one.find(nthKey(1)).has_value());

  // The extremal key, which a zeroed empty slot would wrongly match if occupancy were not
  // checked before the key comparison.
  const NNCacheFrozenIndex zeroKey = NNCacheFrozenIndex::build({keyOf(0, 0), nthKey(1)});
  testAssert(zeroKey.find(keyOf(0, 0)).value() == 0);
  testAssert(!zeroKey.find(keyOf(0, 1)).has_value());
  testAssert(!zeroKey.find(keyOf(1, 0)).has_value());
}

void testFrozenCountersAndHarvest() {
  vector<Hash128> keys;
  for(int i = 0; i < 8; i++)
    keys.push_back(nthKey(i));
  unique_ptr<NNCacheFrozen> frozen = NNCacheFrozen::build(outputsFor(keys));

  // Every counter starts at zero and is not seeded from anything (SPEC.md 3.2).
  for(uint32_t i = 0; i < frozen->numEntries(); i++)
    testAssert(frozen->hitCountAt(i) == 0);

  // A retrieval increments; a membership test and a counter read do not.
  shared_ptr<NNOutput> got;
  testAssert(frozen->get(keys[3], got));
  testAssert(got->nnHash == keys[3]);
  testAssert(frozen->get(keys[3], got));
  testAssert(frozen->hitCountAt(3) == 2);
  testAssert(frozen->contains(keys[3]));
  testAssert(frozen->hitCountAt(3) == 2);
  testAssert(!frozen->contains(nthKey(99)));

  // An arbitrary amount can be folded in without retrieving (SPEC.md 3.2).
  testAssert(frozen->addHits(keys[3], 40));
  testAssert(frozen->hitCountAt(3) == 42);
  testAssert(!frozen->addHits(nthKey(99), 1));

  // Harvest is in index order, includes entries never looked up, and entry i's key,
  // evaluation and counter all agree (SPEC.md 3.4).
  const vector<NNCacheHitCount> rows = frozen->harvest();
  testAssert(rows.size() == 8);
  for(size_t i = 0; i < rows.size(); i++) {
    testAssert(rows[i].key == keys[i]);
    testAssert(rows[i].hits == frozen->hitCountAt((uint32_t)i));
    testAssert(frozen->evaluationAt((uint32_t)i)->nnHash == keys[i]);
  }
  testAssert(rows[3].hits == 42);
  testAssert(rows[0].hits == 0);
}

//-------------------------------------------------------------------------------------
// The two-level strategy
//-------------------------------------------------------------------------------------

void testLevelZeroAnswersFirstAndLevelOneCatchesTheRest() {
  vector<Hash128> zeroKeys;
  for(int i = 0; i < 4; i++)
    zeroKeys.push_back(nthKey(i));
  unique_ptr<NNCacheTable> table = makeTwoLevelNNCacheTable(
    NNCacheFrozen::build(outputsFor(zeroKeys)), defaultLevelOne(8), 8
  );

  // Level 0's keys resolve without level 1 ever having been given them.
  for(int i = 0; i < 4; i++)
    testAssert(present(*table, zeroKeys[i]));
  // A key in neither level is absent.
  testAssert(!present(*table, nthKey(50)));
  // A set goes to level 1 and is then found by fall-through.
  table->set(outputFor(nthKey(50), false));
  testAssert(present(*table, nthKey(50)));

  // clear() empties level 1 and leaves level 0 intact: level 0 is the content this
  // session was handed and cannot be rebuilt in process.
  table->clear();
  testAssert(!present(*table, nthKey(50)));
  for(int i = 0; i < 4; i++)
    testAssert(present(*table, zeroKeys[i]));
}

// The invariant the operator ruled on, driven through the FIRST of the two engine paths
// that can offer a set for a key level 0 already holds: a caller passing skipCache=true
// consults no level at all, evaluates, and sets unconditionally.
void testSkipCachePathUpholdsTheOneOwnerInvariant() {
  const Hash128 key = nthKey(0);
  unique_ptr<NNCacheFrozen> frozenOwned = NNCacheFrozen::build(outputsFor({key, nthKey(1)}));
  NNCacheFrozen* frozen = frozenOwned.get();
  unique_ptr<NNCacheTable> table =
    makeTwoLevelNNCacheTable(std::move(frozenOwned), defaultLevelOne(8), 8);

  testAssert(frozen->contains(key));
  // No get precedes this set: that is exactly what skipCache does.
  table->set(outputFor(key, false));

  // AT MOST ONE LEVEL OWNS THE KEY, observed rather than argued: level 0 no longer
  // resolves it, and the table still does -- from level 1.
  testAssert(!frozen->contains(key));
  testAssert(frozen->isShadowedAt(0));
  testAssert(present(*table, key));
  // The other level-0 entry is untouched.
  testAssert(frozen->contains(nthKey(1)));
}

// The SECOND path, and the one where the invariant and usefulness pull against each other:
// NNEvaluator::evaluate's ownership-map fall-through. A get HITS an entry that lacks a
// requested ownership map, does not return, evaluates for real, and sets the fuller entry.
// Level 0 is frozen and cannot be upgraded in place, so the fuller entry must go to level
// 1 -- and level 0's copy must stop answering, or the caller would keep being handed the
// entry it has already rejected.
void testOwnerMapFallThroughUpholdsTheOneOwnerInvariant() {
  const Hash128 key = nthKey(0);
  vector<shared_ptr<NNOutput>> zeroOutputs;
  zeroOutputs.push_back(outputFor(key, false));  // no ownership map
  unique_ptr<NNCacheFrozen> frozenOwned = NNCacheFrozen::build(std::move(zeroOutputs));
  NNCacheFrozen* frozen = frozenOwned.get();
  unique_ptr<NNCacheTable> table =
    makeTwoLevelNNCacheTable(std::move(frozenOwned), defaultLevelOne(8), 8);

  // The get that hits level 0 and finds no ownership map. nneval stashes this and falls
  // through rather than returning.
  shared_ptr<NNOutput> hit;
  testAssert(table->get(key, hit));
  testAssert(hit->whiteOwnerMap == NULL);

  // The set of the fuller result that follows.
  table->set(outputFor(key, true));

  // One owner: level 0 has given the key up.
  testAssert(!frozen->contains(key));
  // And the caller now gets the fuller entry, rather than being handed the ownermap-less
  // one forever. This is the whole reason the set is not a no-op.
  shared_ptr<NNOutput> after;
  testAssert(table->get(key, after));
  testAssert(after->whiteOwnerMap != NULL);
}

// The count follows the key across the transfer, so the unified surface still shows one
// row per key carrying the session's whole total.
void testHitCountsSurviveTheTransferBetweenLevels() {
  const Hash128 moved = nthKey(0);
  const Hash128 stays = nthKey(1);
  unique_ptr<NNCacheTable> table = makeTwoLevelNNCacheTable(
    NNCacheFrozen::build(outputsFor({moved, stays})), defaultLevelOne(8), 8
  );

  // Three hits while level 0 owns `moved`, two on `stays`.
  for(int i = 0; i < 3; i++)
    testAssert(present(*table, moved));
  for(int i = 0; i < 2; i++)
    testAssert(present(*table, stays));

  // Ownership of `moved` transfers to level 1, carrying its three hits.
  table->set(outputFor(moved, true));
  // Two more hits, now served by level 1.
  for(int i = 0; i < 2; i++)
    testAssert(present(*table, moved));

  const NNCacheHitLedger ledger = table->harvestHitCounts();
  testAssert(ledger.isCounted());
  testAssert(ledger.unrecordedHits() == 0);
  testAssert(hitsForKeyIn(ledger, moved) == 5);
  testAssert(hitsForKeyIn(ledger, stays) == 2);

  // A key only ever served by level 1 is counted too -- "no matter which level it occurs
  // on" includes the level that has no counter of its own.
  const Hash128 levelOneOnly = nthKey(77);
  table->set(outputFor(levelOneOnly, false));
  for(int i = 0; i < 4; i++)
    testAssert(present(*table, levelOneOnly));
  const NNCacheHitLedger again = table->harvestHitCounts();
  testAssert(hitsForKeyIn(again, levelOneOnly) == 4);
  testAssert(hitsForKeyIn(again, moved) == 5);
  testAssert(hitsForKeyIn(again, stays) == 2);
}

// A single-level table -- the default configuration -- says it does not count, rather
// than handing back an empty row list a caller could read as "nothing was hit".
void testASingleLevelTableReportsNotCountedRatherThanEmpty() {
  unique_ptr<NNCacheTable> table = defaultLevelOne(8);
  table->set(outputFor(nthKey(0), false));
  testAssert(present(*table, nthKey(0)));
  const NNCacheHitLedger ledger = table->harvestHitCounts();
  testAssert(!ledger.isCounted());
  testAssert(ledger.disposition() == NNCacheHitLedgerDisposition::NotCounted);
  bool threw = false;
  try { (void)ledger.entries(); } catch(const StringError&) { threw = true; }
  testAssert(threw);

  // And the two-level factory refuses a null level 0: "absent level 0" is the absence of
  // this table, never this table holding a null.
  testAssert(refused(
    [&]() { makeTwoLevelNNCacheTable(nullptr, defaultLevelOne(8), 8); }, "no level 0"
  ));
}

// SPEC.md 6. The ceiling is 48 B/entry resident for the structure, everything except the
// evaluations. Printed rather than pinned -- per-entry bytes are a compiler layout fact --
// and asserted only against the ceiling.
void testFrozenFootprintIsUnderTheBudget() {
  const int n = 50000;
  vector<Hash128> keys;
  for(int i = 0; i < n; i++)
    keys.push_back(nthKey(i));
  unique_ptr<NNCacheFrozen> frozen = NNCacheFrozen::build(outputsFor(keys));
  const double perEntry = (double)frozen->structureBytes() / (double)n;
  cout << "frozen level 0 structure: " << perEntry << " B/entry resident at n=" << n
       << " (SPEC.md 6 ceiling 48; index alone "
       << ((double)frozen->index().structureBytes() / (double)n) << ")" << endl;
  testAssert(perEntry <= 48.0);
  cout << "two-level hit ledger: " << twoLevelHitLedgerBytes(20) << " B at 2^20 rows" << endl;
}

}  // namespace

void Tests::runNNCacheFrozenTests() {
  cout << "Running frozen level-0 cache and two-level strategy tests" << endl;
  testEveryMemberResolvesToItsOwnInputPosition();
  testAbsentKeysAreRejectedByTheStoredKeyComparison();
  testConstructionRefusals();
  testEmptyAndSingleton();
  testFrozenCountersAndHarvest();
  testLevelZeroAnswersFirstAndLevelOneCatchesTheRest();
  testSkipCachePathUpholdsTheOneOwnerInvariant();
  testOwnerMapFallThroughUpholdsTheOneOwnerInvariant();
  testHitCountsSurviveTheTransferBetweenLevels();
  testASingleLevelTableReportsNotCountedRatherThanEmpty();
  testFrozenFootprintIsUnderTheBudget();
  cout << "Done" << endl;
}
