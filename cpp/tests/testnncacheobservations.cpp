#include "../tests/tests.h"

#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "../neuralnet/nncache.h"
#include "../neuralnet/nncachecontext.h"
#include "../neuralnet/nncachecountlog.h"
#include "../neuralnet/nncachedump.h"
#include "../neuralnet/nncachefrozen.h"
#include "../neuralnet/nncacheobservations.h"
#include "../neuralnet/nncachetwolevel.h"
#include "../tests/testsearchcommon.h"

using namespace std;

// OBSERVATION CURRENCY: what the count log records, witnessed as the five separable claims it
// actually is rather than as one "it works".
//
//   ONE PRESENTATION IS ONE COUNT. Observing the same position twice counts two, and a
//   position presented and then SET -- an ordinary miss, which makes a get and a set -- counts
//   the ONE presentation it was, because the count is taken at the request and not at the
//   calls the request makes.
//
//   A MISS COUNTS. This is the whole difference from the retrieval currency it replaces: a
//   position evaluated fresh earns a row of 1 rather than no row at all, which is what makes a
//   seen-twice threshold reachable by anything.
//
//   THE COUNT IS PER (POSITION, CARD). Two cards presented the same position hold two rows,
//   because that position came up once under each and each writes its own file.
//
//   THE DELTA IS CONSUMED EXACTLY ONCE. A second take with nothing in between yields nothing;
//   a count-log record is an increment, so a running total appended twice inflates the record.
//
//   IT ACCUMULATES ACROSS SESSIONS, AND THAT IS OBSERVED ON DISK. A key observed once in one
//   engine's lifetime and once in another's reads as 2 from the file, and a threshold of 2
//   admits it -- the bootstrap the currency exists for. The observation point is the real
//   .nncounts file loaded back, not an in-memory counter, because "the count survived the
//   process" is a claim about the file (ADR-0021 Rule 1).
//
// WHAT IS NOT WITNESSED HERE. That NNEvaluator::evaluate is the caller of observe(), and that
// it calls it exactly once per request -- that is a property of an evaluation path with a real
// net behind it, and it is witnessed end to end in cpp/tests/e2e. Here the door is called
// directly, which is what makes the five claims above separable at all.

namespace {

const char* const TMP_DIR_PREFIX = "tmpnncacheobservations";

Hash128 nthKey(int serial) {
  return Hash128(
    ((uint64_t)(serial + 1)) * 0x9E3779B97F4A7C15ULL,
    ((uint64_t)(serial + 1)) * 0xD6E8FEB86659FD93ULL + 0x1234567ULL
  );
}

shared_ptr<NNOutput> outputFor(Hash128 hash) {
  shared_ptr<NNOutput> p = make_shared<NNOutput>();
  p->nnHash = hash;
  p->nnXLen = 19;
  p->nnYLen = 19;
  return p;
}

unique_ptr<NNCacheTable> plainTable() {
  return NNCacheTable::create(NNCacheConfig::statusQuo(10, 2));
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

uint32_t observationsFor(const NNCacheObservationLedger& ledger, Hash128 key) {
  for(size_t i = 0; i < ledger.entries().size(); i++) {
    if(ledger.entries()[i].key == key)
      return ledger.entries()[i].observations;
  }
  return 0;
}

bool hasRowFor(const NNCacheObservationLedger& ledger, Hash128 key) {
  for(size_t i = 0; i < ledger.entries().size(); i++) {
    if(ledger.entries()[i].key == key)
      return true;
  }
  return false;
}

uint64_t logObservationsFor(const NNCacheCountLogContents& contents, Hash128 key) {
  for(size_t i = 0; i < contents.rows().size(); i++) {
    if(contents.rows()[i].key == key)
      return contents.rows()[i].observations;
  }
  return 0;
}

uint64_t logSessionsFor(const NNCacheCountLogContents& contents, Hash128 key) {
  for(size_t i = 0; i < contents.rows().size(); i++) {
    if(contents.rows()[i].key == key)
      return contents.rows()[i].sessions;
  }
  return 0;
}

//-------------------------------------------------------------------------------------
// The no-context configuration
//-------------------------------------------------------------------------------------

// The default configuration -- plain play, no cache directory, no attached context -- must
// observe nothing, allocate nothing, and answer with a disposition rather than an empty list.
void testATableWithNoAttachedContextObservesNothingAndSaysSo() {
  const unique_ptr<NNCacheTable> table = plainTable();

  // Observing against it is legal and is a no-op: this is the call NNEvaluator::evaluate makes
  // on every single evaluation in every ordinary run of KataGo.
  table->observe(nthKey(1), NNCacheAttribution::noAttributableContext());
  table->observe(nthKey(2), NNCacheAttribution::noAttributableContext());

  const NNCacheObservationLedger ledger = table->harvestObservationCounts();
  testAssert(!ledger.isObserved());
  // NotObserved refuses its rows rather than handing back an empty vector a caller could read
  // as "this session was asked for nothing".
  testAssert(refused([&]() { (void)ledger.entries(); }, "no context is attached"));
  // And it allocated nothing at all, which is the whole memory claim.
  testAssert(table->observationStructureBytes() == 0);

  cout << "  no attached context: observes nothing, allocates nothing, refuses its rows" << endl;
}

// An unattributable presentation, with a context attached, is not filed under whichever
// context happens to be first. It is simply not counted, and the population is reported
// elsewhere (NNCacheAttributionLedger::noAttributableContextEntries).
void testAnUnattributablePresentationIsFiledUnderNobody() {
  const unique_ptr<NNCacheTable> table = plainTable();
  const NNCacheContextId a = table->attachCacheContext("card-a");
  table->attachCacheContext("card-b");

  table->observe(nthKey(1), NNCacheAttribution::noAttributableContext());
  table->observe(nthKey(1), NNCacheAttribution::toContext(a));

  const NNCacheObservationLedger forA = table->harvestObservationCountsFor(a);
  testAssert(forA.isObserved());
  testAssert(observationsFor(forA, nthKey(1)) == 1);
  // Whole-table: exactly the one attributed row, and nothing standing in for the other.
  testAssert(table->harvestObservationCounts().entries().size() == 1);

  cout << "  an unattributable presentation is counted under nobody, not under the first card" << endl;
}

//-------------------------------------------------------------------------------------
// One presentation is one count
//-------------------------------------------------------------------------------------

void testEachPresentationCountsOnceAndAMissCountsToo() {
  const unique_ptr<NNCacheTable> table = plainTable();
  const NNCacheContextId card = table->attachCacheContext("card-a");
  const NNCacheAttribution to = NNCacheAttribution::toContext(card);

  const Hash128 fresh = nthKey(1);
  const Hash128 repeated = nthKey(2);

  // A FRESH EVALUATION: the request presents the position, misses, and stores what it
  // computed. One presentation, and the set is not a second one -- observe() is called once
  // per request, by the one caller that knows a request is one request.
  table->observe(fresh, to);
  table->set(outputFor(fresh), to);

  // The same position asked for three times: three presentations, whether or not the cache
  // answered them.
  table->observe(repeated, to);
  table->set(outputFor(repeated), to);
  table->observe(repeated, to);
  table->observe(repeated, to);

  const NNCacheObservationLedger ledger = table->harvestObservationCountsFor(card);
  // THE CLAIM THE WHOLE CURRENCY TURNS ON: a position evaluated exactly once, and never
  // retrieved, carries 1 -- not nothing. Under the retrieval currency this row did not exist,
  // which is why a seen-twice threshold could never be reached by a new position.
  testAssert(observationsFor(ledger, fresh) == 1);
  testAssert(observationsFor(ledger, repeated) == 3);

  cout << "  one request is one observation; a fresh evaluation carries 1 rather than nothing" << endl;
}

void testTheCountIsPerPositionAndCard() {
  const unique_ptr<NNCacheTable> table = plainTable();
  const NNCacheContextId a = table->attachCacheContext("card-a");
  const NNCacheContextId b = table->attachCacheContext("card-b");
  // The ~3.7% of keys two real cards share: one position, presented under each.
  const Hash128 shared = nthKey(7);

  table->observe(shared, NNCacheAttribution::toContext(a));
  table->observe(shared, NNCacheAttribution::toContext(a));
  table->observe(shared, NNCacheAttribution::toContext(b));

  testAssert(observationsFor(table->harvestObservationCountsFor(a), shared) == 2);
  testAssert(observationsFor(table->harvestObservationCountsFor(b), shared) == 1);
  // TWO ROWS, NOT ONE SUMMED ROW. Each card writes its own file, and a sum would be a number
  // belonging to neither.
  testAssert(table->harvestObservationCounts().entries().size() == 2);

  cout << "  a shared position under two cards is two rows of two counts, never one sum" << endl;
}

void testAForeignContextIsRefusedByNameOnEverySurface() {
  const unique_ptr<NNCacheTable> mine = plainTable();
  const unique_ptr<NNCacheTable> theirs = plainTable();
  mine->attachCacheContext("card-a");
  const NNCacheContextId foreign = theirs->attachCacheContext("card-a");

  // Same NAME, different cache. Spending the id here would count under whichever context sits
  // at the same position in this table's own name space, which is a different card.
  testAssert(refused(
    [&]() { mine->observe(nthKey(1), NNCacheAttribution::toContext(foreign)); },
    "attached to a different cache"
  ));
  testAssert(refused([&]() { (void)mine->harvestObservationCountsFor(foreign); }, "not attached to this cache"));
  testAssert(refused(
    [&]() { (void)mine->takeUnpersistedObservationCountsFor(foreign); }, "not attached to this cache"
  ));
  testAssert(refused(
    [&]() { (void)mine->hasUnpersistedObservationCountsFor(foreign); }, "not attached to this cache"
  ));

  cout << "  an id from another cache is refused by name on all four surfaces" << endl;
}

//-------------------------------------------------------------------------------------
// The delta
//-------------------------------------------------------------------------------------

void testTheDeltaIsConsumedExactlyOnceAndOmitsAKeyWithNothingToSay() {
  const unique_ptr<NNCacheTable> table = plainTable();
  const NNCacheContextId card = table->attachCacheContext("card-a");
  const NNCacheAttribution to = NNCacheAttribution::toContext(card);

  table->observe(nthKey(1), to);
  table->observe(nthKey(1), to);
  table->observe(nthKey(2), to);

  testAssert(table->hasUnpersistedObservationCountsFor(card));
  const NNCacheObservationLedger first = table->takeUnpersistedObservationCountsFor(card);
  testAssert(first.entries().size() == 2);
  testAssert(observationsFor(first, nthKey(1)) == 2);
  testAssert(observationsFor(first, nthKey(2)) == 1);

  // A SECOND TAKE WITH NOTHING IN BETWEEN YIELDS NOTHING. A record is an increment, so a
  // running total handed to appendDump twice would inflate the file by a whole session's
  // worth of presentations that never happened.
  testAssert(!table->hasUnpersistedObservationCountsFor(card));
  const NNCacheObservationLedger second = table->takeUnpersistedObservationCountsFor(card);
  testAssert(second.isObserved());
  testAssert(second.entries().empty());

  // What accrues AFTER the take is the next delta, and only that.
  table->observe(nthKey(1), to);
  testAssert(table->hasUnpersistedObservationCountsFor(card));
  const NNCacheObservationLedger third = table->takeUnpersistedObservationCountsFor(card);
  testAssert(third.entries().size() == 1);
  testAssert(observationsFor(third, nthKey(1)) == 1);
  // And the ABSOLUTE surface still reports the running total, which is a different question.
  testAssert(observationsFor(table->harvestObservationCountsFor(card), nthKey(1)) == 3);

  cout << "  the delta is consumed once, omits a key with nothing to say, and the running "
       << "total is unmoved by taking it" << endl;
}

void testOneCardsTakeLeavesAnothersDeltaWhole() {
  const unique_ptr<NNCacheTable> table = plainTable();
  const NNCacheContextId a = table->attachCacheContext("card-a");
  const NNCacheContextId b = table->attachCacheContext("card-b");

  table->observe(nthKey(1), NNCacheAttribution::toContext(a));
  table->observe(nthKey(2), NNCacheAttribution::toContext(b));

  const NNCacheObservationLedger takeA = table->takeUnpersistedObservationCountsFor(a);
  testAssert(takeA.entries().size() == 1);
  testAssert(hasRowFor(takeA, nthKey(1)));
  // B's mark did not move: with two cards attached in one session, each one's dump still finds
  // its own delta whole.
  testAssert(table->hasUnpersistedObservationCountsFor(b));
  const NNCacheObservationLedger takeB = table->takeUnpersistedObservationCountsFor(b);
  testAssert(takeB.entries().size() == 1);
  testAssert(hasRowFor(takeB, nthKey(2)));

  cout << "  a per-card take advances that card's marks and no other card's" << endl;
}

//-------------------------------------------------------------------------------------
// Across sessions, on disk
//-------------------------------------------------------------------------------------

// THE BOOTSTRAP, END TO END THROUGH THE FILE. A position evaluated once in one engine's
// lifetime and once in another's clears a seen-twice threshold in the second. Nothing in this
// test holds state across the two halves except the directory.
void testObservationsAccumulateAcrossSessionsAndBootstrapASeenTwiceThreshold() {
  TestCommon::ScopedTempDir dir(TMP_DIR_PREFIX);
  const NNCacheCountLog log = NNCacheCountLog::forContext(dir.path(), "card-a");
  const Hash128 position = nthKey(11);

  // SESSION A: the position is evaluated fresh, exactly once, and never asked for again.
  {
    const unique_ptr<NNCacheTable> table = plainTable();
    const NNCacheContextId card = table->attachCacheContext("card-a");
    const NNCacheAttribution to = NNCacheAttribution::toContext(card);
    table->observe(position, to);
    table->set(outputFor(position), to);
    (void)log.appendDump(NNCacheObservationDelta::takeFor(*table, card));
  }
  {
    const NNCacheCountLogContents after = log.load();
    testAssert(logObservationsFor(after, position) == 1);
    testAssert(logSessionsFor(after, position) == 1);
    // A ROW EXISTS FOR A KEY THIS SESSION'S OWN ADMISSION WOULD NOT ADMIT, which is the point:
    // the count is what the next session adds to.
    testAssert(!NNCacheDiskAdmission::minObservations(2).admits(logObservationsFor(after, position)));
  }

  // SESSION B: a different table, a different context id, the same directory. The position
  // comes up once more.
  {
    const unique_ptr<NNCacheTable> table = plainTable();
    const NNCacheContextId card = table->attachCacheContext("card-a");
    table->observe(position, NNCacheAttribution::toContext(card));
    (void)log.appendDump(NNCacheObservationDelta::takeFor(*table, card));
  }
  {
    const NNCacheCountLogContents after = log.load();
    testAssert(logObservationsFor(after, position) == 2);
    testAssert(logSessionsFor(after, position) == 2);
    // AND NOW IT CLEARS THE THRESHOLD. Under the retrieval currency this key never had a row
    // at all, so this comparison could not be reached in any number of sessions.
    testAssert(NNCacheDiskAdmission::minObservations(2).admits(logObservationsFor(after, position)));
  }

  cout << "  observations accumulate across sessions on disk: 1 then 2, and 2 clears "
       << "minObservations(2)" << endl;
}

// ATTACH -> DETACH -> ATTACH WITH NO TRAFFIC IS A TRACKING NO-OP. Stated as a ratified
// contract, so it is witnessed as one: a dump with nothing observed since the last one leaves
// the file byte-identical, and no key's sessions climbs.
void testACycleWithNoTrafficChangesNothingOnDisk() {
  TestCommon::ScopedTempDir dir(TMP_DIR_PREFIX);
  const NNCacheCountLog log = NNCacheCountLog::forContext(dir.path(), "card-a");
  const Hash128 position = nthKey(12);

  const unique_ptr<NNCacheTable> table = plainTable();
  const NNCacheContextId card = table->attachCacheContext("card-a");
  table->observe(position, NNCacheAttribution::toContext(card));
  (void)log.appendDump(NNCacheObservationDelta::takeFor(*table, card));

  const NNCacheCountLogContents afterFirst = log.load();
  testAssert(logObservationsFor(afterFirst, position) == 1);
  testAssert(logSessionsFor(afterFirst, position) == 1);

  // A second dump, nothing presented in between. The delta is empty, so the block carries no
  // records and this key's sessions does not rise.
  (void)log.appendDump(NNCacheObservationDelta::takeFor(*table, card));
  const NNCacheCountLogContents afterSecond = log.load();
  testAssert(logObservationsFor(afterSecond, position) == 1);
  testAssert(logSessionsFor(afterSecond, position) == 1);

  cout << "  a dump with no traffic since the last leaves every count and every sessions "
       << "exactly where it was" << endl;
}

// The count log refuses a delta from a table that observes nothing, by name, rather than
// writing it as a dump of zero rows a later reader would take for "this session asked for
// nothing".
void testTheCountLogRefusesADeltaFromATableThatObservesNothing() {
  TestCommon::ScopedTempDir dir(TMP_DIR_PREFIX);
  const NNCacheCountLog log = NNCacheCountLog::forContext(dir.path(), "card-a");
  testAssert(refused(
    [&]() { (void)log.appendDump(NNCacheObservationDelta::notObserved()); },
    "NotObserved"
  ));
  cout << "  appendDump refuses a NotObserved delta by name" << endl;
}

//-------------------------------------------------------------------------------------
// The structure's own claims
//-------------------------------------------------------------------------------------

// The sizing comment's arithmetic, asserted against the implementation rather than trusted
// from prose -- an earlier comment in this family named a per-row byte figure that was not the
// real one (ADR-0012 P1).
void testTheRecorderSizingIsWhatItsCommentSays() {
  const int pow = NNCacheObservationRecorder::defaultPowerOfTwo();
  const int64_t rows = ((int64_t)1) << pow;
  const int64_t reference = NNCacheObservationRecorder::sizingReferenceKeys();
  // Two rows to a 64-byte cache line is the whole reason a probe walk is cheap.
  testAssert(NNCacheObservationRecorder::rowBytes() == 32);
  testAssert(observationRecorderBytes(pow) == (size_t)(rows * 32));
  // At the reference scale the structure stays under the load factor its bounded probe window
  // is held to.
  testAssert(reference * 100 < rows * NNCacheObservationRecorder::maxLoadFactorPercent());
  // One home for the corpus figure, shared with the attribution recorder.
  testAssert(reference == NNCacheAttributionRecorder::sizingReferenceKeys());

  cout << "  observation recorder: 2^" << pow << " rows of " << NNCacheObservationRecorder::rowBytes()
       << " B = " << observationRecorderBytes(pow) << " B, at "
       << (reference * 100 / rows) << "% occupancy at the corpus reference of " << reference << " keys"
       << endl;
}

// The overflow is COUNTED, not absorbed: a presentation that cannot be given a row is reported
// rather than overwriting somebody else's. Witnessed on a recorder small enough to fill.
void testAnOverflowingObservationIsCountedRatherThanAbsorbed() {
  // 16 rows and a 16-slot probe window: filling it is a handful of keys, and the seventeenth
  // distinct key has nowhere to go.
  NNCacheObservationRecorder tiny(4, 0);
  for(int i = 0; i < 16; i++)
    tiny.observe(nthKey(2000 + i), 0);
  testAssert(tiny.unrecordedObservations() == 0);
  for(int i = 16; i < 24; i++)
    tiny.observe(nthKey(2000 + i), 0);
  testAssert(tiny.unrecordedObservations() == 8);
  // And nothing already recorded was corrupted by the overflow.
  testAssert(tiny.harvestAll().size() == 16);

  cout << "  a presentation with no row is counted in unrecordedObservations (" << tiny.unrecordedObservations()
       << "), never written over another key's row" << endl;
}

}  // namespace

void Tests::runNNCacheObservationTests() {
  cout << "Running NN cache observation-currency tests" << endl;

  testATableWithNoAttachedContextObservesNothingAndSaysSo();
  testAnUnattributablePresentationIsFiledUnderNobody();
  testEachPresentationCountsOnceAndAMissCountsToo();
  testTheCountIsPerPositionAndCard();
  testAForeignContextIsRefusedByNameOnEverySurface();
  testTheDeltaIsConsumedExactlyOnceAndOmitsAKeyWithNothingToSay();
  testOneCardsTakeLeavesAnothersDeltaWhole();
  testObservationsAccumulateAcrossSessionsAndBootstrapASeenTwiceThreshold();
  testACycleWithNoTrafficChangesNothingOnDisk();
  testTheCountLogRefusesADeltaFromATableThatObservesNothing();
  testTheRecorderSizingIsWhatItsCommentSays();
  testAnOverflowingObservationIsCountedRatherThanAbsorbed();

  cout << "Done" << endl;
}
