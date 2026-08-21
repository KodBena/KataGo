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
#include "../neuralnet/nneval.h"
#include "../game/board.h"
#include "../game/boardhistory.h"
#include "../game/rules.h"
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
// THE LEGS BELOW SPLIT INTO TWO GROUPS, AND THE SPLIT IS THE POINT.
//
//   AT THE RECORDER, calling the mint directly. That is what makes the five claims above
//   SEPARABLE -- each one is a property of the counting structure and is asserted alone.
//
//   THROUGH THE DOOR, driving a real NNEvaluator::evaluate. These exist because the first
//   version of this file had only the first group, and an out-of-frame audit found that the
//   sentence the whole design rests on -- one demand is one observation, whatever the request
//   goes on to do -- was asserted in a comment and tested nowhere. It was FALSE on a shipped
//   path (averageMultipleSymmetries) and no leg here would have noticed. The four sub-cases the
//   design names (hit, miss, skipCache, ownership-map fall-through) and the multi-symmetry root
//   are each driven through evaluate() below, with the count read off the table afterwards.

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
  (void)table->present(nthKey(1), NNCacheAttribution::noAttributableContext());
  (void)table->present(nthKey(2), NNCacheAttribution::noAttributableContext());
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

  (void)table->present(nthKey(1), NNCacheAttribution::noAttributableContext());
  (void)table->present(nthKey(1), NNCacheAttribution::toContext(a));
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
  (void)table->present(fresh, to);
  table->set(outputFor(fresh), to);

  // The same position asked for three times: three presentations, whether or not the cache
  // answered them.
  (void)table->present(repeated, to);
  table->set(outputFor(repeated), to);
  (void)table->present(repeated, to);
  (void)table->present(repeated, to);
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

  (void)table->present(shared, NNCacheAttribution::toContext(a));
  (void)table->present(shared, NNCacheAttribution::toContext(a));
  (void)table->present(shared, NNCacheAttribution::toContext(b));
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
    [&]() { (void)mine->present(nthKey(1), NNCacheAttribution::toContext(foreign)); },
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

  (void)table->present(nthKey(1), to);
  (void)table->present(nthKey(1), to);
  (void)table->present(nthKey(2), to);
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
  (void)table->present(nthKey(1), to);
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

  (void)table->present(nthKey(1), NNCacheAttribution::toContext(a));
  (void)table->present(nthKey(2), NNCacheAttribution::toContext(b));
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
    (void)table->present(position, to);
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
    (void)table->present(position, NNCacheAttribution::toContext(card));
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
  (void)table->present(position, NNCacheAttribution::toContext(card));
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

// THE CONTRAST: WHAT AN ABSOLUTE SURFACE DOES TO AN ADDITIVE READER.
//
// RE-MADE HERE AFTER AN AUDIT FOUND IT DELETED WITHOUT REPLACEMENT. Its baseline form
// (testnncachedump.cpp, retrieval currency) was the NEGATIVE half of a pair: the positive leg
// shows the consuming take leaves a no-traffic dump inert, and this one shows what happens if
// the RUNNING TOTAL reaches appendDump instead. Without it the suite says the right path is
// right and never says the wrong path is wrong -- and the hazard is not hypothetical, because
// NNCacheObservationDelta::ofDeltaRows still constructs an appendable delta from arbitrary rows
// and five test files use it.
//
// THE LAUNDER IS DELIBERATE AND IS NAMED AS SUCH. `log.appendDump(table.harvestObservation...)`
// does not compile -- that is the type doing its job -- so reaching the defect requires
// unwrapping a harvest's rows and re-wrapping them under a name that says "delta". That act is
// what a caller would now have to type ON PURPOSE; before the type existed it was the obvious
// one-liner.
void testTheAbsoluteSurfaceLaunderedIntoADeltaInflatesTheRecord() {
  TestCommon::ScopedTempDir dir(TMP_DIR_PREFIX);
  const NNCacheCountLog log = NNCacheCountLog::forContext(dir.path(), "card-a");
  const Hash128 position = nthKey(21);

  const unique_ptr<NNCacheTable> table = plainTable();
  const NNCacheContextId card = table->attachCacheContext("card-a");
  (void)table->present(position, NNCacheAttribution::toContext(card));

  // The laundering, written out so the call site says what it is doing.
  const NNCacheObservationLedger runningTotals = table->harvestObservationCountsFor(card);
  (void)log.appendDump(
    NNCacheObservationDelta::ofDeltaRows(runningTotals.entries(), runningTotals.unrecordedObservations())
  );
  testAssert(logObservationsFor(log.load(), position) == 1);
  testAssert(logSessionsFor(log.load(), position) == 1);

  // AGAIN, WITH NOTHING PRESENTED IN BETWEEN. The absolute surface still reports 1, because a
  // running total does not go away; appendDump ADDS it, so the record climbs and the key is
  // credited with a session in which nothing asked for it.
  const NNCacheObservationLedger sameTotalsAgain = table->harvestObservationCountsFor(card);
  testAssert(sameTotalsAgain.entries().size() == 1);
  (void)log.appendDump(
    NNCacheObservationDelta::ofDeltaRows(sameTotalsAgain.entries(), sameTotalsAgain.unrecordedObservations())
  );
  const NNCacheCountLogContents inflated = log.load();
  testAssert(logObservationsFor(inflated, position) == 2);   // one real demand, recorded twice
  testAssert(logSessionsFor(inflated, position) == 2);       // and a session that never happened

  // THE CONTROL, so this leg is about the SURFACE and not about the fixture: the consuming take
  // over the very same state appends nothing at all.
  const NNCacheCountLog soundLog = NNCacheCountLog::forContext(dir.path(), "card-b");
  const unique_ptr<NNCacheTable> soundTable = plainTable();
  const NNCacheContextId soundCard = soundTable->attachCacheContext("card-b");
  (void)soundTable->present(position, NNCacheAttribution::toContext(soundCard));
  (void)soundLog.appendDump(NNCacheObservationDelta::takeFor(*soundTable, soundCard));
  (void)soundLog.appendDump(NNCacheObservationDelta::takeFor(*soundTable, soundCard));
  testAssert(logObservationsFor(soundLog.load(), position) == 1);
  testAssert(logSessionsFor(soundLog.load(), position) == 1);

  cout << "  laundering the running total into a delta inflates the record (1 demand -> "
       << logObservationsFor(inflated, position) << " observations, "
       << logSessionsFor(inflated, position) << " sessions) where the consuming take does not"
       << endl;
}

// THE NON-CONSUMING QUESTION ADVANCES NOTHING, AS A TWO-FIXTURE A/B CONTROL.
//
// RE-MADE HERE AFTER AN AUDIT FOUND ITS BASELINE FORM DELETED AND ONLY HALF-REPLACED. Asserting
// that hasUnpersisted... flips true->false around a take is the ARMING half; it cannot witness
// that ASKING moved nothing, because a question that consumed what it reported would flip
// exactly the same way. The property is an EQUALITY BETWEEN TWO RUNS and is watched as one: the
// same history is built twice, side by side, asked in one fixture and not in the other, and the
// two takes are then compared ROW FOR ROW. A question that advanced a mark shows up as a
// shorter take on the asked side and as nothing at all on any assertion about its own answer.
void testTheNonConsumingQuestionAdvancesNothing() {
  // The state deliberately includes a key presented but NEVER SET -- a demand the cache could
  // not serve and that no attribution records. That is the shape the retired protocol-layer
  // proxy was blind to and read as "nothing to lose", and it is the reason this question exists
  // at all rather than a count of accepted requests.
  const Hash128 servedAndSet = nthKey(31);
  const Hash128 demandedOnly = nthKey(32);

  vector<NNCacheObservationCount> takes[2];
  bool armedBeforeTake[2] = {false, false};
  for(int fixture = 0; fixture < 2; fixture++) {
    const bool askFirst = fixture == 0;
    const unique_ptr<NNCacheTable> table = plainTable();
    const NNCacheContextId card = table->attachCacheContext("card-a");
    const NNCacheAttribution to = NNCacheAttribution::toContext(card);

    const NNCachePresentation first = table->present(servedAndSet, to);
    table->set(first, outputFor(servedAndSet), to);
    (void)table->present(servedAndSet, to);
    (void)table->present(demandedOnly, to);

    // THE ONLY DIFFERENCE BETWEEN THE TWO FIXTURES.
    if(askFirst) {
      armedBeforeTake[fixture] = table->hasUnpersistedObservationCountsFor(card);
      // Asked twice, because once could not tell an idempotent question from a consuming one.
      testAssert(table->hasUnpersistedObservationCountsFor(card) == armedBeforeTake[fixture]);
    }

    const NNCacheObservationLedger taken = table->takeUnpersistedObservationCountsFor(card);
    takes[fixture] = taken.entries();
    // The question is FALSE afterwards on the asked side too -- the disarming half, kept.
    if(askFirst)
      testAssert(!table->hasUnpersistedObservationCountsFor(card));
  }
  testAssert(armedBeforeTake[0]);

  // ROW FOR ROW, in the order the two takes produced them. Both walk the same structure with
  // the same keys and the same mixing, so the orders are comparable; that they ARE identical is
  // itself part of the claim, since a mark moved on one side would drop a row.
  testAssert(takes[0].size() == takes[1].size());
  testAssert(takes[0].size() == 2);
  for(size_t i = 0; i < takes[0].size(); i++) {
    testAssert(takes[0][i].key == takes[1][i].key);
    testAssert(takes[0][i].observations == takes[1][i].observations);
  }
  // And the demanded-but-never-set key really is in there, which is what makes this the state
  // the old proxy could not see.
  testAssert(
    (takes[0][0].key == demandedOnly && takes[0][0].observations == 1) ||
    (takes[0][1].key == demandedOnly && takes[0][1].observations == 1)
  );

  cout << "  the non-consuming question advances nothing: the take after asking is identical "
       << "row for row (" << takes[0].size() << " rows) to the same history never asked, and it "
       << "arms on a demand that was never set" << endl;
}

//-------------------------------------------------------------------------------------
// Through the door: a real NNEvaluator
//-------------------------------------------------------------------------------------

// debugSkipNeuralNet, so the evaluations are stubs and the CACHE PATH is the real one -- the
// same NNResultBuf, the same evaluate(), the same mint, the same set. What is under test is the
// counting, not the net.
NNEvaluator* startStubEvaluator(Logger& logger, const string& seed) {
  return TestSearchCommon::startNNEval(
    "/dev/null", logger, seed, NNPos::MAX_BOARD_LEN, NNPos::MAX_BOARD_LEN,
    0, true, false, false, true, false
  );
}

BoardHistory sevenBySevenPosition(Board& board, Player& nextPla) {
  board = Board::parseBoard(7, 7, R"%%(
.......
.......
..x....
.......
....o..
.......
.......
)%%");
  nextPla = P_BLACK;
  return BoardHistory(board, nextPla, Rules::getTrompTaylorish(), 0, BoardHistoryModes(false, false));
}

// The total observations this evaluator's table has recorded for `card`, whatever keys they are
// under. The legs below each drive ONE position, so the total IS that position's count.
int64_t totalObservationsFor(NNEvaluator* nnEval, const NNCacheContextId& card) {
  const NNCacheObservationLedger ledger = nnEval->harvestCacheObservationCountsFor(card);
  testAssert(ledger.isObserved());
  int64_t total = 0;
  for(size_t i = 0; i < ledger.entries().size(); i++)
    total += (int64_t)ledger.entries()[i].observations;
  return total;
}

// THE FOUR SUB-CASES THE DESIGN NAMES, EACH THROUGH evaluate(). Every one of them is ONE demand
// and must read as ONE observation, and they differ in what the request does with the cache --
// which is exactly the axis a get/set-based count would have got wrong.
void testEachEvaluateIsOneObservationThroughTheDoor() {
  ScoreValue::initTables();
  Logger logger(nullptr, false, false, false);

  // (a) A MISS: the request consults the cache, finds nothing, and stores what it computed.
  //     Two cache calls, one demand.
  {
    NNEvaluator* nnEval = startStubEvaluator(logger, "obsdoor-miss");
    const NNCacheContextId card = nnEval->attachCacheContext("card-a");
    Board board; Player nextPla;
    const BoardHistory hist = sevenBySevenPosition(board, nextPla);
    NNResultBuf buf;
    MiscNNInputParams params;
    buf.cacheAttribution = NNCacheAttribution::toContext(card);
    nnEval->evaluate(board, hist, nextPla, params, buf, false, false);
    testAssert(buf.hasResult);
    testAssert(totalObservationsFor(nnEval, card) == 1);
    cout << "  through the door, miss: 1 demand -> 1 observation" << endl;
    delete nnEval;
  }

  // (b) A HIT: the same position asked for a second time. Two demands, two observations -- and
  //     this is the leg that says a hit counts at all, which the retired retrieval currency got
  //     right and a naive "count only misses" would get wrong.
  {
    NNEvaluator* nnEval = startStubEvaluator(logger, "obsdoor-hit");
    const NNCacheContextId card = nnEval->attachCacheContext("card-a");
    Board board; Player nextPla;
    const BoardHistory hist = sevenBySevenPosition(board, nextPla);
    MiscNNInputParams params;
    for(int i = 0; i < 2; i++) {
      NNResultBuf buf;
      buf.cacheAttribution = NNCacheAttribution::toContext(card);
      nnEval->evaluate(board, hist, nextPla, params, buf, false, false);
      testAssert(buf.hasResult);
    }
    testAssert(totalObservationsFor(nnEval, card) == 2);
    cout << "  through the door, hit: 2 demands -> 2 observations" << endl;
    delete nnEval;
  }

  // (c) skipCache: the request consults NO level and sets unconditionally. Zero gets, one
  //     demand. A count folded into get() would lose this one entirely.
  {
    NNEvaluator* nnEval = startStubEvaluator(logger, "obsdoor-skip");
    const NNCacheContextId card = nnEval->attachCacheContext("card-a");
    Board board; Player nextPla;
    const BoardHistory hist = sevenBySevenPosition(board, nextPla);
    NNResultBuf buf;
    MiscNNInputParams params;
    buf.cacheAttribution = NNCacheAttribution::toContext(card);
    nnEval->evaluate(board, hist, nextPla, params, buf, true, false);
    testAssert(buf.hasResult);
    testAssert(totalObservationsFor(nnEval, card) == 1);
    cout << "  through the door, skipCache: 1 demand and no get -> 1 observation" << endl;
    delete nnEval;
  }

  // (d) THE OWNERSHIP-MAP FALL-THROUGH: the request hits, REJECTS the hit for lacking an
  //     ownership map, re-evaluates and sets the fuller result. One get, one set, ONE demand.
  //     A count folded into get and set would read 2 here.
  {
    NNEvaluator* nnEval = startStubEvaluator(logger, "obsdoor-ownermap");
    const NNCacheContextId card = nnEval->attachCacheContext("card-a");
    Board board; Player nextPla;
    const BoardHistory hist = sevenBySevenPosition(board, nextPla);
    MiscNNInputParams params;
    {
      // First, store an entry WITHOUT an ownership map, so the fall-through has something to
      // reject. That is itself one demand.
      NNResultBuf buf;
      buf.cacheAttribution = NNCacheAttribution::toContext(card);
      nnEval->evaluate(board, hist, nextPla, params, buf, false, false);
      testAssert(buf.hasResult);
      testAssert(buf.result->whiteOwnerMap == NULL);
    }
    {
      // Now ask WITH an ownership map: the get hits, the result is rejected, a second
      // evaluation runs and sets the fuller entry -- all inside one evaluate().
      NNResultBuf buf;
      buf.cacheAttribution = NNCacheAttribution::toContext(card);
      nnEval->evaluate(board, hist, nextPla, params, buf, false, true);
      testAssert(buf.hasResult);
      testAssert(buf.result->whiteOwnerMap != NULL);
    }
    testAssert(totalObservationsFor(nnEval, card) == 2);
    cout << "  through the door, ownership-map fall-through: 2 demands (one of which did a get, "
         << "a rejection and a set) -> 2 observations" << endl;
    delete nnEval;
  }

  ScoreValue::freeTables();
}

// THE MULTI-SYMMETRY ROOT, WHICH IS THE LEG THE FIRST VERSION OF THIS FILE DID NOT HAVE AND
// WHICH AN OUT-OF-FRAME AUDIT FOUND THE DEFECT IN.
//
// averageMultipleSymmetries runs N forward passes for ONE root query. All N compute the SAME
// cache key -- NNInputs::getHash does not fold in symmetry, which is why that path passes
// skipCache -- so before the presentation mint every one of them counted, and one root query at
// N = 8 wrote 8 observations of one position and cleared the default seen-twice admission
// inside the session that evaluated it.
//
// THE EXPECTED COUNT IS A NAMED CONSTANT because the question it answers was a genuine fork --
// N forward passes, or one demand? -- decided by the operator (ledger row 1814: one demand).
// Flipping the ruling flips this one constant and nothing else in this test, which is what
// makes the test a statement about the ruling rather than about the implementation.
const int64_t OBSERVATIONS_PER_MULTI_SYMMETRY_ROOT_QUERY = 1;

void testAMultiSymmetryRootQueryIsOneObservation() {
  ScoreValue::initTables();
  Logger logger(nullptr, false, false, false);
  NNEvaluator* nnEval = startStubEvaluator(logger, "obsdoor-symmetry");
  const NNCacheContextId card = nnEval->attachCacheContext("card-a");
  Board board; Player nextPla;
  const BoardHistory hist = sevenBySevenPosition(board, nextPla);

  NNResultBuf buf;
  MiscNNInputParams params;
  Rand rand("obsdoor-symmetry-rand");
  const int numSymmetries = 8;
  buf.cacheAttribution = NNCacheAttribution::toContext(card);
  shared_ptr<NNOutput>* averaged = nnEval->averageMultipleSymmetries(
    board, hist, nextPla, NULL, params, buf, false, rand, numSymmetries
  );
  testAssert(averaged != NULL);
  delete averaged;

  // THE WITNESS, and it is read off the count surface rather than off any flag: N forward
  // passes of one key are ONE demand.
  const int64_t observed = totalObservationsFor(nnEval, card);
  testAssert(observed == OBSERVATIONS_PER_MULTI_SYMMETRY_ROOT_QUERY);
  // AND THE ROW EXISTS, so this is not passing because nothing was counted at all.
  testAssert(nnEval->harvestCacheObservationCountsFor(card).entries().size() == 1);
  // AND IT DOES NOT CLEAR THE DEFAULT ADMISSION IN ITS OWN SESSION, which is the property the
  // inflation destroyed and the reason the ruling matters.
  testAssert(!NNCacheDiskAdmission::minObservations(2).admits((uint64_t)observed));

  cout << "  through the door, " << numSymmetries << "-symmetry root query: " << observed
       << " observation (ledger row 1814: one demand, not " << numSymmetries
       << " forward passes), and it does not clear minObservations(2) in-session" << endl;
  delete nnEval;
  ScoreValue::freeTables();
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
  testTheAbsoluteSurfaceLaunderedIntoADeltaInflatesTheRecord();
  testTheNonConsumingQuestionAdvancesNothing();
  testEachEvaluateIsOneObservationThroughTheDoor();
  testAMultiSymmetryRootQueryIsOneObservation();

  cout << "Done" << endl;
}
