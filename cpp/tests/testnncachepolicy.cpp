#include "../tests/tests.h"

#include <cmath>
#include <memory>
#include <vector>

#include "../neuralnet/nncache.h"
#include "../neuralnet/nncacheimpl.h"
#include "../neuralnet/nncacheprobed.h"

using namespace std;
using namespace TestCommon;

// Correctness tests for the cache policies. Nothing here measures anything: an LRU
// that evicts the wrong entry while being fast is a defect, and these tests are
// written so they can see that.
//
// Every eviction order below is CONSTRUCTED, not sampled. The tables are tiny (16
// slots in 4 lock regions), the keys are chosen so that a named group of them shares
// one home slot and therefore competes for exactly the same ways, and each scenario
// asserts by name which key survived and which did not. Those are logic invariants,
// so they are asserted exactly, with no tolerance. The one genuinely distributional
// quantity here -- which way Random picks -- is the one and only thing given a
// statistical bar instead (ADR-0009, Calibration).

//-------------------------------------------------------------------------------------
// Helpers
//-------------------------------------------------------------------------------------

// A key whose home slot is `home` (the table indexes on the low bits of hash0) and
// which is distinct from every other key made with a different serial.
static Hash128 keyAt(uint64_t home, int serial) {
  return Hash128(home, ((uint64_t)(serial + 1)) * 0x9E3779B97F4A7C15ULL);
}

static shared_ptr<NNOutput> entryFor(Hash128 hash, bool withOwnerMap) {
  shared_ptr<NNOutput> p = make_shared<NNOutput>();
  p->nnHash = hash;
  p->nnXLen = 19;
  p->nnYLen = 19;
  if(withOwnerMap)
    p->whiteOwnerMap = new float[19*19];
  return p;
}

// NOTE for every caller: this is a get(), so under LRU and LFU it is a SIGHTING and
// moves the entry's standing. Every scenario below therefore does all of its
// membership assertions after its last insert, and replays from a fresh table when it
// needs to continue a sequence.
static bool present(NNCacheTable& table, Hash128 hash) {
  shared_ptr<NNOutput> got;
  bool found = table.get(hash,got);
  testAssert(found == (got != nullptr));
  if(found)
    testAssert(got->nnHash == hash);
  return found;
}

static NNCacheConfig probedConfig(
  NNCacheCollisionScheme scheme, int ways, NNCacheEvictionPolicy eviction,
  int sizePowerOfTwo, int mutexPoolSizePowerOfTwo
) {
  NNCacheConfig config = {
    sizePowerOfTwo, mutexPoolSizePowerOfTwo,
    NNCacheShape::probed(scheme,ways,eviction),
    NNCacheAdmissionPolicy::Always
  };
  return config;
}

// The geometry every probed scenario below uses: 16 slots in 4 regions of 4 slots, so
// one lock region holds exactly the 4 ways, and the six keys made with home 0 all
// compete for slots 0..3 of region 0.
static const int SIZE_POW = 4;
static const int POOL_POW = 2;
static const int WAYS = 4;

//-------------------------------------------------------------------------------------
// Probe sequences: the slot sets are what the two schemes actually differ in
//-------------------------------------------------------------------------------------

static void testProbeSequences() {
  using namespace NNCacheProbed;

  // Both sequences must be a permutation of a power-of-two region's positions,
  // because `ways` probes that landed on the same slot twice would silently reduce
  // the associativity the operator asked for. Exact, for every region size the
  // config can produce a table with.
  for(int shift = 0; shift <= 10; shift++) {
    const uint64_t regionSlots = ((uint64_t)1) << shift;
    vector<bool> seenLinear((size_t)regionSlots,false);
    vector<bool> seenQuadratic((size_t)regionSlots,false);
    for(uint64_t j = 0; j < regionSlots; j++) {
      const uint64_t lin = LinearProbe::offset((int)j) & (regionSlots-1);
      const uint64_t quad = QuadraticProbe::offset((int)j) & (regionSlots-1);
      testAssert(!seenLinear[(size_t)lin]);
      testAssert(!seenQuadratic[(size_t)quad]);
      seenLinear[(size_t)lin] = true;
      seenQuadratic[(size_t)quad] = true;
    }
  }

  // And they must genuinely differ, or the two config values would be one value
  // wearing two names. At 8 slots per region, 3 ways: linear takes {0,1,2},
  // quadratic takes {0,1,3}.
  testAssert(LinearProbe::offset(0) == 0 && QuadraticProbe::offset(0) == 0);
  testAssert(LinearProbe::offset(1) == 1 && QuadraticProbe::offset(1) == 1);
  testAssert(LinearProbe::offset(2) == 2 && QuadraticProbe::offset(2) == 3);
  testAssert(LinearProbe::offset(3) == 3 && QuadraticProbe::offset(3) == 6);
}

//-------------------------------------------------------------------------------------
// LRU
//-------------------------------------------------------------------------------------

// Fills all four ways with A,B,C,D in that order, then re-sights A, then inserts E.
// The least recently used entry at that moment is B: A was just sighted, and C and D
// were both written after B.
static void testLruEvictsTheLeastRecentlyUsed(NNCacheCollisionScheme scheme) {
  unique_ptr<NNCacheTable> table = NNCacheTable::create(
    probedConfig(scheme,WAYS,NNCacheEvictionPolicy::Lru,SIZE_POW,POOL_POW)
  );
  const Hash128 a = keyAt(0,0), b = keyAt(0,1), c = keyAt(0,2), d = keyAt(0,3), e = keyAt(0,4);

  table->set(entryFor(a,false));
  table->set(entryFor(b,false));
  table->set(entryFor(c,false));
  table->set(entryFor(d,false));
  testAssert(present(*table,a));  // re-sighting A is the point of the scenario
  table->set(entryFor(e,false));

  testAssert(present(*table,a));
  testAssert(!present(*table,b));   // B, and nothing else, is the victim
  testAssert(present(*table,c));
  testAssert(present(*table,d));
  testAssert(present(*table,e));
}

// Replays the scenario above and continues it: after E lands, C is re-sighted, so the
// least recently used entry becomes D -- the only one of the four that has not been
// touched since it was written.
static void testLruSecondEvictionFollowsTheNewOrder(NNCacheCollisionScheme scheme) {
  unique_ptr<NNCacheTable> table = NNCacheTable::create(
    probedConfig(scheme,WAYS,NNCacheEvictionPolicy::Lru,SIZE_POW,POOL_POW)
  );
  const Hash128 a = keyAt(0,0), b = keyAt(0,1), c = keyAt(0,2), d = keyAt(0,3);
  const Hash128 e = keyAt(0,4), f = keyAt(0,5);

  table->set(entryFor(a,false));
  table->set(entryFor(b,false));
  table->set(entryFor(c,false));
  table->set(entryFor(d,false));
  testAssert(present(*table,a));
  table->set(entryFor(e,false));
  testAssert(present(*table,c));
  table->set(entryFor(f,false));

  testAssert(present(*table,a));
  testAssert(!present(*table,b));
  testAssert(present(*table,c));
  testAssert(!present(*table,d));   // D is now the oldest sighting
  testAssert(present(*table,e));
  testAssert(present(*table,f));
}

//-------------------------------------------------------------------------------------
// LFU, and the pollution question
//-------------------------------------------------------------------------------------

// A,B,C,D are given 4,3,2,1 sightings respectively. The first eviction must take D.
static void testLfuEvictsTheLeastFrequentlyUsed() {
  unique_ptr<NNCacheTable> table = NNCacheTable::create(
    probedConfig(NNCacheCollisionScheme::LinearProbe,WAYS,NNCacheEvictionPolicy::Lfu,SIZE_POW,POOL_POW)
  );
  const Hash128 a = keyAt(0,0), b = keyAt(0,1), c = keyAt(0,2), d = keyAt(0,3), e = keyAt(0,4);

  table->set(entryFor(a,false));
  table->set(entryFor(b,false));
  table->set(entryFor(c,false));
  table->set(entryFor(d,false));
  for(int i = 0; i<3; i++) testAssert(present(*table,a));
  for(int i = 0; i<2; i++) testAssert(present(*table,b));
  testAssert(present(*table,c));
  table->set(entryFor(e,false));

  testAssert(present(*table,a));
  testAssert(present(*table,b));
  testAssert(present(*table,c));
  testAssert(!present(*table,d));   // one sighting, the fewest
  testAssert(present(*table,e));
}

// The pollution test, and the reason this is LFU with dynamic aging rather than naive
// LFU. A accumulates the highest count in the set and is then NEVER referenced again.
// Under naive LFU that makes A immortal: every newcomer starts at 1 and is evicted
// first, forever. Under LFUDA the region's aging floor rises to each victim's count
// and newcomers are admitted at floor+1, so the floor overtakes A's frozen count and
// A dies. This asserts exactly that -- and asserts first that A survives the early
// evictions, so the test would still fail a policy that simply evicted A eagerly.
//
// Counts and the region's aging floor, step by step, once A,B,C,D have been given
// 4,3,2,1 sightings and none of the newcomers is ever re-referenced:
//
//   set E: victim D(1), floor->1, E:2   -- residents A4 B3 C2 E2
//   set F: victim C(2), floor->2, F:3   -- residents A4 B3 F3 E2
//   set G: victim E(2), floor->2, G:3   -- residents A4 B3 F3 G3
//   set H: victim B(3), floor->3, H:4   -- residents A4 H4 F3 G3
//   set I: victim F(3), floor->3, I:4   -- residents A4 H4 I4 G3
//   set J: victim G(3), floor->3, J:4   -- residents A4 H4 I4 J4
//   set K: every count is now 4, and A is first in the probe order, so A is the victim.
//
// A survives the first six insertions and dies on the seventh. Nothing here may READ A
// in between: a get() is a sighting and would raise its count, which is exactly the
// mechanism under test.
static unique_ptr<NNCacheTable> buildLfuAgedTable(int newcomers) {
  unique_ptr<NNCacheTable> table = NNCacheTable::create(
    probedConfig(NNCacheCollisionScheme::LinearProbe,WAYS,NNCacheEvictionPolicy::Lfu,SIZE_POW,POOL_POW)
  );
  const Hash128 a = keyAt(0,0), b = keyAt(0,1), c = keyAt(0,2), d = keyAt(0,3);
  table->set(entryFor(a,false));
  table->set(entryFor(b,false));
  table->set(entryFor(c,false));
  table->set(entryFor(d,false));
  for(int i = 0; i<3; i++) testAssert(present(*table,a));   // A: 4 sightings
  for(int i = 0; i<2; i++) testAssert(present(*table,b));   // B: 3
  testAssert(present(*table,c));                            // C: 2
                                                            // D: 1
  for(int i = 0; i<newcomers; i++)
    table->set(entryFor(keyAt(0,4+i),false));
  return table;
}

static void testLfuAgingEvictsAStaleHighFrequencyEntry() {
  // Six newcomers: A, the most frequently referenced entry in the set, is untouched.
  {
    unique_ptr<NNCacheTable> table = buildLfuAgedTable(6);
    testAssert(present(*table,keyAt(0,0)));
  }
  // The seventh is the one the rising floor pays for. Naive LFU would never do this:
  // A's count of 4 against newcomers admitted at 1 would make it immortal.
  {
    unique_ptr<NNCacheTable> table = buildLfuAgedTable(7);
    testAssert(!present(*table,keyAt(0,0)));
    testAssert(present(*table,keyAt(0,7)));    // H
    testAssert(present(*table,keyAt(0,8)));    // I
    testAssert(present(*table,keyAt(0,9)));    // J
    testAssert(present(*table,keyAt(0,10)));   // K
  }
}

//-------------------------------------------------------------------------------------
// Random
//-------------------------------------------------------------------------------------

// Two exact facts first: nothing is evicted while a way is still free, and when the
// ways are full exactly one resident dies -- not zero, not two.
static void testRandomEvictsExactlyOneResidentAndOnlyWhenFull() {
  unique_ptr<NNCacheTable> table = NNCacheTable::create(
    probedConfig(NNCacheCollisionScheme::LinearProbe,WAYS,NNCacheEvictionPolicy::Random,SIZE_POW,POOL_POW)
  );
  vector<Hash128> keys;
  for(int i = 0; i<WAYS; i++) {
    keys.push_back(keyAt(0,i));
    table->set(entryFor(keys.back(),false));
  }
  for(int i = 0; i<WAYS; i++)
    testAssert(present(*table,keys[i]));

  const Hash128 extra = keyAt(0,WAYS);
  table->set(entryFor(extra,false));
  testAssert(present(*table,extra));
  int survivors = 0;
  for(int i = 0; i<WAYS; i++)
    survivors += present(*table,keys[i]) ? 1 : 0;
  testAssert(survivors == WAYS-1);
}

// The only distributional claim in this file, and the only one given a tolerance.
// Each way is followed by tracking which key currently occupies it: a new key always
// lands in the slot its victim vacated, so the victim's WAY is observable without
// reaching inside the table. Over 4000 evictions each of the 4 ways should be chosen
// about 1000 times; the bar is 5 standard deviations of the binomial (sd = 27.4), and
// the generator is seeded deterministically per region so the result is reproducible.
static void testRandomChoosesUniformlyAmongWays() {
  unique_ptr<NNCacheTable> table = NNCacheTable::create(
    probedConfig(NNCacheCollisionScheme::LinearProbe,WAYS,NNCacheEvictionPolicy::Random,SIZE_POW,POOL_POW)
  );
  // Filling an empty table takes the ways in probe order, so way j holds key j.
  vector<Hash128> occupant;
  for(int j = 0; j<WAYS; j++) {
    occupant.push_back(keyAt(0,j));
    table->set(entryFor(occupant.back(),false));
  }

  const int trials = 4000;
  vector<int> victimCount((size_t)WAYS,0);
  for(int t = 0; t<trials; t++) {
    const Hash128 fresh = keyAt(0,WAYS+t);
    table->set(entryFor(fresh,false));
    int victimWay = -1;
    for(int j = 0; j<WAYS; j++) {
      if(!present(*table,occupant[(size_t)j])) {
        testAssert(victimWay < 0);   // exactly one resident died
        victimWay = j;
      }
    }
    testAssert(victimWay >= 0);
    victimCount[(size_t)victimWay] += 1;
    occupant[(size_t)victimWay] = fresh;   // the newcomer took the vacated slot
  }

  const double expected = (double)trials / (double)WAYS;
  const double sd = sqrt((double)trials * 0.25 * 0.75);
  cout << "  random victim counts by way:";
  for(int j = 0; j<WAYS; j++)
    cout << " " << victimCount[(size_t)j];
  cout << " (expected " << expected << " each, sd " << sd << ")" << endl;
  for(int j = 0; j<WAYS; j++)
    testAssert(fabs((double)victimCount[(size_t)j] - expected) < 5.0 * sd);
}

//-------------------------------------------------------------------------------------
// Chaining and the byte budget
//-------------------------------------------------------------------------------------

// The capacity eviction order is now an argument rather than an unnameable recency
// policy, so every chained scenario states which order it is asserting against. The
// scenarios that predate the change pass Lru, which is the order that was already in
// force, so they assert exactly what they asserted before.
static NNCacheConfig chainedConfig(
  int64_t maxBytes, int sizePowerOfTwo, int mutexPoolSizePowerOfTwo, NNCacheEvictionPolicy eviction
) {
  NNCacheConfig config = {
    sizePowerOfTwo, mutexPoolSizePowerOfTwo,
    NNCacheShape::chained(maxBytes,eviction),
    NNCacheAdmissionPolicy::Always
  };
  return config;
}

// The budget is a real byte budget, denominated in the resource that actually
// exhausts memory, so how many entries fit DEPENDS ON THE ENTRY. This inserts the same
// number of keys into the same region under the same budget, once with bare payloads
// and once with a 19x19 ownership map attached, and asserts that the two capacities
// differ by exactly the amount the footprint accounting says they should.
static void testChainedByteBudgetCountsTheRealFootprint() {
  const int sizePow = 4, poolPow = 2;   // 16 buckets in 4 regions; buckets 0..3 are region 0

  shared_ptr<NNOutput> bareProto = entryFor(keyAt(0,0),false);
  shared_ptr<NNOutput> ownerProto = entryFor(keyAt(0,0),true);
  const size_t bareBytes = chainedEntryBytes(*bareProto);
  const size_t ownerBytes = chainedEntryBytes(*ownerProto);
  // The ownermap is a separate heap block reached through a pointer, so it is invisible
  // to sizeof(NNOutput) and must be visible to the budget.
  testAssert(ownerBytes == bareBytes + (size_t)(19*19*sizeof(float)));

  const int64_t bareFit = 3;
  const int64_t maxBytes = (int64_t)bareBytes * bareFit * (((int64_t)1) << poolPow);
  testAssert(chainedRegionBudgetBytes(maxBytes,sizePow,poolPow) == (int64_t)bareBytes * bareFit);
  const int64_t ownerFit = chainedRegionBudgetBytes(maxBytes,sizePow,poolPow) / (int64_t)ownerBytes;
  testAssert(ownerFit < bareFit);   // the same budget holds strictly fewer fat entries

  cout << "  chained budget: " << bareBytes << " B/bare entry, " << ownerBytes
       << " B/ownermap entry, region budget " << chainedRegionBudgetBytes(maxBytes,sizePow,poolPow)
       << " B -> " << bareFit << " bare or " << ownerFit << " ownermap entries" << endl;

  // Bare entries: insert one more than fits. The least recently used, which is the one
  // inserted first and never sighted, is the one that goes.
  {
    unique_ptr<NNCacheTable> table = NNCacheTable::create(chainedConfig(maxBytes,sizePow,poolPow,NNCacheEvictionPolicy::Lru));
    vector<Hash128> keys;
    for(int64_t i = 0; i < bareFit+1; i++) {
      keys.push_back(keyAt((uint64_t)(i % 4), (int)i));   // buckets 0..3, all in region 0
      table->set(entryFor(keys.back(),false));
    }
    testAssert(!present(*table,keys[0]));
    for(size_t i = 1; i<keys.size(); i++)
      testAssert(present(*table,keys[i]));
  }

  // Ownermap entries under the identical budget: fewer survive, by measurement.
  {
    unique_ptr<NNCacheTable> table = NNCacheTable::create(chainedConfig(maxBytes,sizePow,poolPow,NNCacheEvictionPolicy::Lru));
    vector<Hash128> keys;
    for(int64_t i = 0; i < bareFit+1; i++) {
      keys.push_back(keyAt((uint64_t)(i % 4), (int)i));
      table->set(entryFor(keys.back(),true));
    }
    int survivors = 0;
    for(size_t i = 0; i<keys.size(); i++)
      survivors += present(*table,keys[i]) ? 1 : 0;
    testAssert(survivors == (int)ownerFit);
    // and it is the most recent ones that survive
    testAssert(present(*table,keys.back()));
    testAssert(!present(*table,keys[0]));
  }
}

// A sighting is what defers eviction, so re-reading an entry keeps it and lets an
// untouched newer one go instead.
static void testChainedSweepEvictsByRecency() {
  const int sizePow = 4, poolPow = 2;
  shared_ptr<NNOutput> bareProto = entryFor(keyAt(0,0),false);
  const int64_t maxBytes = (int64_t)chainedEntryBytes(*bareProto) * 3 * 4;

  unique_ptr<NNCacheTable> table = NNCacheTable::create(chainedConfig(maxBytes,sizePow,poolPow,NNCacheEvictionPolicy::Lru));
  const Hash128 a = keyAt(0,0), b = keyAt(1,1), c = keyAt(2,2), d = keyAt(3,3);
  table->set(entryFor(a,false));
  table->set(entryFor(b,false));
  table->set(entryFor(c,false));
  testAssert(present(*table,a));    // A is now the most recently used, B the least
  table->set(entryFor(d,false));

  testAssert(present(*table,a));
  testAssert(!present(*table,b));
  testAssert(present(*table,c));
  testAssert(present(*table,d));
}

// Re-offering a key with a bigger payload is nneval's ownermap-upgrade path. The
// budget must be re-charged by the delta, not by a second whole entry.
static void testChainedRechargesAnUpgradedEntry() {
  const int sizePow = 4, poolPow = 2;
  shared_ptr<NNOutput> bareProto = entryFor(keyAt(0,0),false);
  shared_ptr<NNOutput> ownerProto = entryFor(keyAt(0,0),true);
  // A budget of exactly two bare entries per region.
  const int64_t maxBytes = (int64_t)chainedEntryBytes(*bareProto) * 2 * 4;

  unique_ptr<NNCacheTable> table = NNCacheTable::create(chainedConfig(maxBytes,sizePow,poolPow,NNCacheEvictionPolicy::Lru));
  const Hash128 a = keyAt(0,0), b = keyAt(1,1);
  table->set(entryFor(a,false));
  table->set(entryFor(b,false));
  testAssert(present(*table,a));
  testAssert(present(*table,b));

  // Upgrading A to carry an ownermap costs more than the region has left, so the
  // upgrade is retained and B -- the least recently used -- pays for it.
  testAssert(
    chainedEntryBytes(*ownerProto) + chainedEntryBytes(*bareProto) >
    (size_t)chainedRegionBudgetBytes(maxBytes,sizePow,poolPow)
  );
  table->set(entryFor(a,true));
  testAssert(present(*table,a));
  testAssert(!present(*table,b));
}

// The chained capacity sweep under `random`. Two things are asserted, and the second is
// the one that distinguishes random from lru rather than merely exercising it.
static void testChainedRandomEvictsExactlyOneResident() {
  const int sizePow = 4, poolPow = 2;
  shared_ptr<NNOutput> bareProto = entryFor(keyAt(0,0),false);
  const int64_t maxBytes = (int64_t)chainedEntryBytes(*bareProto) * 3 * 4;
  const Hash128 keys[4] = { keyAt(0,0), keyAt(1,1), keyAt(2,2), keyAt(3,3) };

  // Exactly one victim per over-budget insert: not zero (which would break the bound)
  // and not two (which would evict more than the newcomer cost).
  {
    unique_ptr<NNCacheTable> table =
      NNCacheTable::create(chainedConfig(maxBytes,sizePow,poolPow,NNCacheEvictionPolicy::Random));
    for(int i = 0; i<3; i++)
      table->set(entryFor(keys[i],false));
    table->set(entryFor(keys[3],false));
    int survivors = 0;
    for(int i = 0; i<4; i++)
      survivors += present(*table,keys[i]) ? 1 : 0;
    testAssert(survivors == 3);
  }

  // And the order is not the recency one. Ten distinct keys are inserted into one
  // region whose budget holds three. Under `lru` the survivors are then EXACTLY the
  // three most recent, every time and by construction -- that is what
  // testChainedSweepEvictsByRecency pins. So asserting that this survivor set is not
  // {the last three} distinguishes `random` from a recency order quietly wearing its
  // name, rather than merely exercising the code path. The region rng is seeded from
  // the region index and nothing else, so this is exact and deterministic run to run,
  // not a sample given a tolerance.
  {
    const int numKeys = 10;
    unique_ptr<NNCacheTable> table =
      NNCacheTable::create(chainedConfig(maxBytes,sizePow,poolPow,NNCacheEvictionPolicy::Random));
    vector<Hash128> many;
    for(int i = 0; i<numKeys; i++) {
      many.push_back(keyAt((uint64_t)(i % 4), 200+i));   // buckets 0..3, all region 0
      table->set(entryFor(many.back(),false));
    }
    int survivors = 0;
    bool matchesRecencyOutcome = true;
    for(int i = 0; i<numKeys; i++) {
      const bool here = present(*table,many[(size_t)i]);
      survivors += here ? 1 : 0;
      if(here != (i >= numKeys-3))
        matchesRecencyOutcome = false;
    }
    testAssert(survivors == 3);            // the budget is still hard, whatever the order
    testAssert(!matchesRecencyOutcome);    // and it is not the recency order
    cout << "  chained random: 10 keys into a 3-entry region left 3 residents, and they are"
         << " NOT the last 3 inserted (which is exactly what lru would leave)" << endl;
  }
}

// The chained capacity sweep under `lfu`. Both scenarios keep the region at or below
// LFU_SAMPLE_SIZE entries, where the sample IS the region and the choice is therefore
// exact -- so these are logic invariants asserted with no tolerance, not samples.
static void testChainedLfuEvictsTheLeastFrequentlyUsed() {
  const int sizePow = 4, poolPow = 2;
  shared_ptr<NNOutput> bareProto = entryFor(keyAt(0,0),false);
  const int64_t maxBytes = (int64_t)chainedEntryBytes(*bareProto) * 3 * 4;

  unique_ptr<NNCacheTable> table =
    NNCacheTable::create(chainedConfig(maxBytes,sizePow,poolPow,NNCacheEvictionPolicy::Lfu));
  const Hash128 a = keyAt(0,0), b = keyAt(1,1), c = keyAt(2,2), d = keyAt(3,3);
  table->set(entryFor(a,false));
  table->set(entryFor(b,false));
  table->set(entryFor(c,false));
  // All three were admitted at floor+1 = 1. Sight A three times and B once, leaving C
  // the least frequent by construction; every get here is a sighting, which is the trap
  // named at the top of this file, and it is being used deliberately.
  testAssert(present(*table,a));
  testAssert(present(*table,a));
  testAssert(present(*table,a));
  testAssert(present(*table,b));

  table->set(entryFor(d,false));
  testAssert(present(*table,a));
  testAssert(present(*table,b));
  testAssert(!present(*table,c));   // C, and only C: fewest sightings
  testAssert(present(*table,d));
}

// The aging property, which is what makes this LFUDA rather than naive LFU: an entry
// with a high frozen count is eventually overtaken instead of becoming immortal.
static void testChainedLfuAgesAStaleEntry() {
  const int sizePow = 4, poolPow = 2;
  shared_ptr<NNOutput> bareProto = entryFor(keyAt(0,0),false);
  const int64_t maxBytes = (int64_t)chainedEntryBytes(*bareProto) * 2 * 4;

  const Hash128 a = keyAt(0,0);

  // A is probed only ONCE per table, at the very end, and the whole scenario is rebuilt
  // from scratch for each newcomer count. That is not fastidiousness: a get() is a
  // SIGHTING under LFU, so probing A inside the loop would bump the very count the test
  // is asserting has gone stale, and the entry would stay ahead of the aging floor
  // forever. This is the trap named at the top of this file, and it is the one that
  // would silently turn this test into a test of nothing.
  const int maxNewcomers = 40;
  int diedAfter = -1;
  for(int n = 0; n <= maxNewcomers && diedAfter < 0; n++) {
    unique_ptr<NNCacheTable> table =
      NNCacheTable::create(chainedConfig(maxBytes,sizePow,poolPow,NNCacheEvictionPolicy::Lfu));
    table->set(entryFor(a,false));
    for(int i = 0; i<4; i++)          // A reaches count 5, then is never referenced again
      testAssert(present(*table,a));
    for(int i = 0; i<n; i++)          // newcomers, none of them ever re-referenced
      table->set(entryFor(keyAt((uint64_t)(1 + (i % 3)), 100+i),false));
    if(!present(*table,a))
      diedAfter = n;
  }

  // The assertion is the aging property itself, and it is a logic invariant with no
  // tolerance: under NAIVE LFU there is no finite n at all, because every newcomer would
  // enter at count 1 against A's frozen 5 and A would be immortal. Under LFUDA the floor
  // climbs past 5 and A dies. That a finite n exists is the whole difference between the
  // algorithm that shipped and the one it replaced.
  testAssert(diedAfter > 0);
  testAssert(diedAfter <= maxNewcomers);
  cout << "  chained lfu aging: the stale high-count entry survived " << (diedAfter-1)
       << " unreferenced newcomers and died on the " << diedAfter
       << "th (naive LFU: never dies)" << endl;
}

// stats() is what every byte and occupancy figure the sweep reports comes from, so it is
// asserted against a scenario whose resident set is known by construction, not merely
// printed.
static void testChainedStatsReportWhatIsResident() {
  const int sizePow = 4, poolPow = 2;
  shared_ptr<NNOutput> bareProto = entryFor(keyAt(0,0),false);
  shared_ptr<NNOutput> ownerProto = entryFor(keyAt(0,0),true);
  const int64_t maxBytes = (int64_t)chainedEntryBytes(*bareProto) * 3 * 4;

  unique_ptr<NNCacheTable> table =
    NNCacheTable::create(chainedConfig(maxBytes,sizePow,poolPow,NNCacheEvictionPolicy::Lru));
  table->set(entryFor(keyAt(0,0),false));
  table->set(entryFor(keyAt(1,1),true));

  const NNCacheStats s = table->stats();
  testAssert(s.residentEntries == 2);
  // The payload figure counts the ownership map, which is a separate heap block: the
  // whole reason the sweep reports bytes rather than a count.
  testAssert(
    s.residentPayloadBytes ==
    (int64_t)(nnOutputFootprintBytes(*bareProto) + nnOutputFootprintBytes(*ownerProto))
  );
  // A chained table is bounded by bytes and has no slot capacity, so it reports none
  // rather than a denominator someone downstream would divide by.
  testAssert(s.capacitySlots == 0);
  testAssert(s.fixedStructureBytes > 0);
}

// The same for a probed table, where capacitySlots IS meaningful and occupancy is the
// quantity the whole sweep is organised around.
static void testProbedStatsReportOccupancy() {
  unique_ptr<NNCacheTable> table = NNCacheTable::create(
    probedConfig(NNCacheCollisionScheme::LinearProbe,WAYS,NNCacheEvictionPolicy::Lru,SIZE_POW,POOL_POW)
  );
  for(int i = 0; i<3; i++)
    table->set(entryFor(keyAt(0,i),false));

  const NNCacheStats s = table->stats();
  testAssert(s.residentEntries == 3);
  testAssert(s.capacitySlots == (((int64_t)1) << SIZE_POW));
  testAssert(s.residentPayloadBytes == 3 * (int64_t)sizeof(NNOutput));
  // Occupancy is exactly 3/16 here, which is the arithmetic the sweep does; asserting it
  // means the sweep's denominator is the table's own, not a second copy of it.
  testAssert(s.residentEntries * 16 == 3 * s.capacitySlots);
}

//-------------------------------------------------------------------------------------
// Second-sighting admission
//-------------------------------------------------------------------------------------

static NNCacheConfig withAdmission(NNCacheConfig config, NNCacheAdmissionPolicy admission) {
  config.admission = admission;
  return config;
}

static void testSecondSightingAdmission() {
  // The ghost set's memory bound is exact and derived from a knob that already
  // exists, so it can be asserted rather than estimated.
  testAssert(secondSightingGhostBytes(21) == (size_t)4 * (((size_t)1) << 21));
  testAssert(secondSightingGhostBytes(0) == 4);

  NNCacheConfig config = withAdmission(
    NNCacheConfig::statusQuo(8,4), NNCacheAdmissionPolicy::SecondSighting
  );
  unique_ptr<NNCacheTable> table = NNCacheTable::create(config);
  const Hash128 a = keyAt(3,0), b = keyAt(5,1);

  table->set(entryFor(a,false));
  testAssert(!present(*table,a));   // first sighting is remembered, not stored
  table->set(entryFor(a,false));
  testAssert(present(*table,a));    // second sighting is stored

  table->set(entryFor(b,false));
  testAssert(!present(*table,b));
  testAssert(present(*table,a));    // and B's first sighting did not disturb A
  table->set(entryFor(b,false));
  testAssert(present(*table,b));

  // clear() must clear the ghost set too, or the table would keep admitting on the
  // strength of sightings that belong to a position it no longer holds.
  table->clear();
  testAssert(!present(*table,a));
  table->set(entryFor(a,false));
  testAssert(!present(*table,a));
  table->set(entryFor(a,false));
  testAssert(present(*table,a));
}

// Admission is a decorator, so it must behave identically over every collision scheme.
static void testSecondSightingComposesWithEveryShape() {
  vector<NNCacheConfig> configs;
  configs.push_back(NNCacheConfig::statusQuo(8,4));
  configs.push_back(probedConfig(NNCacheCollisionScheme::LinearProbe,4,NNCacheEvictionPolicy::Lru,8,4));
  configs.push_back(probedConfig(NNCacheCollisionScheme::QuadraticProbe,4,NNCacheEvictionPolicy::Lfu,8,4));
  shared_ptr<NNOutput> proto = entryFor(keyAt(0,0),false);
  configs.push_back(chainedConfig((int64_t)chainedEntryBytes(*proto) * 8 * 16, 8, 4, NNCacheEvictionPolicy::Lru));

  for(size_t i = 0; i<configs.size(); i++) {
    unique_ptr<NNCacheTable> table = NNCacheTable::create(
      withAdmission(configs[i],NNCacheAdmissionPolicy::SecondSighting)
    );
    const Hash128 key = keyAt(7,(int)i);
    table->set(entryFor(key,false));
    testAssert(!present(*table,key));
    table->set(entryFor(key,false));
    testAssert(present(*table,key));
  }
}

//-------------------------------------------------------------------------------------
// Every shape builds, and honours the plain cache contract
//-------------------------------------------------------------------------------------

static void testEveryShapeConstructsAndCaches() {
  vector<NNCacheConfig> configs;
  configs.push_back(NNCacheConfig::statusQuo(8,4));
  const NNCacheCollisionScheme schemes[2] =
    {NNCacheCollisionScheme::LinearProbe, NNCacheCollisionScheme::QuadraticProbe};
  const NNCacheEvictionPolicy evictions[3] =
    {NNCacheEvictionPolicy::Random, NNCacheEvictionPolicy::Lru, NNCacheEvictionPolicy::Lfu};
  for(int s = 0; s<2; s++)
    for(int e = 0; e<3; e++)
      configs.push_back(probedConfig(schemes[s],4,evictions[e],8,4));
  shared_ptr<NNOutput> proto = entryFor(keyAt(0,0),false);
  configs.push_back(chainedConfig((int64_t)chainedEntryBytes(*proto) * 8 * 16, 8, 4, NNCacheEvictionPolicy::Lru));

  for(size_t i = 0; i<configs.size(); i++) {
    unique_ptr<NNCacheTable> table = NNCacheTable::create(configs[i]);
    testAssert(table != nullptr);
    const Hash128 key = keyAt(11,(int)i);
    const Hash128 absent = keyAt(11,1000+(int)i);
    testAssert(!present(*table,key));
    table->set(entryFor(key,false));
    testAssert(present(*table,key));
    testAssert(!present(*table,absent));
    table->clear();
    testAssert(!present(*table,key));
    table->set(entryFor(key,false));
    testAssert(present(*table,key));
  }
}

// The one constraint the probe-confinement concurrency design imposes, refused at
// construction with both the keys the operator would have to change (ADR-0002).
static void testWaysBeyondALockRegionIsRefused() {
  bool threw = false;
  string what;
  try {
    // 16 slots, 4 regions -> 4 slots per region, but 8 ways asked for.
    (void)NNCacheTable::create(
      probedConfig(NNCacheCollisionScheme::LinearProbe,8,NNCacheEvictionPolicy::Lru,4,2)
    );
  }
  catch(const StringError& e) { threw = true; what = e.what(); }
  testAssert(threw);
  testAssert(what.find("nnCacheWays") != string::npos);
  testAssert(what.find("nnMutexPoolSizePowerOfTwo") != string::npos);
  testAssert(what.find("nnCacheSizePowerOfTwo") != string::npos);

  // And exactly at the region size it is accepted, so the refusal is a real boundary
  // and not a margin someone guessed at.
  (void)NNCacheTable::create(
    probedConfig(NNCacheCollisionScheme::LinearProbe,4,NNCacheEvictionPolicy::Lru,4,2)
  );
}

static void testChainedBudgetTooSmallForARegionIsRefused() {
  bool threw = false;
  string what;
  try {
    // 2^4 buckets in 2^2 regions, with a budget that cannot hold one entry per region.
    (void)NNCacheTable::create(chainedConfig(1000,4,2,NNCacheEvictionPolicy::Lru));
  }
  catch(const StringError& e) { threw = true; what = e.what(); }
  testAssert(threw);
  testAssert(what.find("nnCacheMaxBytes") != string::npos);
  testAssert(what.find("nnMutexPoolSizePowerOfTwo") != string::npos);
  testAssert(what.find("per lock region") != string::npos);
}

//-------------------------------------------------------------------------------------

void Tests::runNNCachePolicyTests() {
  cout << "Running nn cache policy tests" << endl;

  testProbeSequences();

  testLruEvictsTheLeastRecentlyUsed(NNCacheCollisionScheme::LinearProbe);
  testLruEvictsTheLeastRecentlyUsed(NNCacheCollisionScheme::QuadraticProbe);
  testLruSecondEvictionFollowsTheNewOrder(NNCacheCollisionScheme::LinearProbe);
  testLruSecondEvictionFollowsTheNewOrder(NNCacheCollisionScheme::QuadraticProbe);

  testLfuEvictsTheLeastFrequentlyUsed();
  testLfuAgingEvictsAStaleHighFrequencyEntry();

  testRandomEvictsExactlyOneResidentAndOnlyWhenFull();
  testRandomChoosesUniformlyAmongWays();

  testChainedByteBudgetCountsTheRealFootprint();
  testChainedSweepEvictsByRecency();
  testChainedRechargesAnUpgradedEntry();
  testChainedRandomEvictsExactlyOneResident();
  testChainedLfuEvictsTheLeastFrequentlyUsed();
  testChainedLfuAgesAStaleEntry();
  testChainedStatsReportWhatIsResident();
  testProbedStatsReportOccupancy();

  testSecondSightingAdmission();
  testSecondSightingComposesWithEveryShape();

  testEveryShapeConstructsAndCaches();
  testWaysBeyondALockRegionIsRefused();
  testChainedBudgetTooSmallForARegionIsRefused();

  cout << "nn cache policy tests passed" << endl;
}
