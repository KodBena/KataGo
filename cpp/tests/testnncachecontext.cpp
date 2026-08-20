#include "../tests/tests.h"

#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "../game/board.h"
#include "../game/boardhistory.h"
#include "../game/rules.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nncache.h"
#include "../neuralnet/nncachecontext.h"
#include "../neuralnet/nncachecountlog.h"
#include "../neuralnet/nncachefrozen.h"
#include "../neuralnet/nncachetwolevel.h"
#include "../neuralnet/nneval.h"
#include "../search/search.h"
#include "../tests/testsearchcommon.h"

using namespace std;
using namespace TestCommon;

// CONTEXT ATTRIBUTION OF WHAT A SESSION EARNS: the tag's whole journey, in four stages.
//
// The claim under test is one sentence -- an entry a session earns is filed under the context
// that earned it, and under no other -- and it is a claim about four different surfaces, so it
// is witnessed at four different observation points rather than once wherever it is cheapest.
//
//   THE NAME SPACE. Whether an unknown context is refused, whether a sole attached context is
//   defaulted to, and whether an id from one cache means anything in another. Observed against
//   NNCacheContextSet itself, because these are properties of the names alone and need no cache.
//
//   THE LEDGER, AGAINST THE REAL PER-CONTEXT FILE. Whether an entry earned under context A is
//   filed under A. The observation point is deliberately NOT the in-memory counter the set path
//   just wrote -- reading that back would witness that a store round-trips, which is not the
//   claim. It is the actual <context>.nncounts file the count log writes, loaded back from
//   disk: the key appears in A's file and is ABSENT from B's. That is the surface a dump
//   actually persists to and the only one on which "filed under the wrong card" is a fact
//   rather than an opinion (ADR-0021 Rule 1).
//
//   THE UNATTRIBUTABLE RESIDUE. With several contexts attached and a request naming none, the
//   entry is COUNTED and reported, never assigned to whichever context happens to be first.
//   Observed both ways: the count rises, and the entry is absent from every context's file.
//
//   THE PLUMB. Whether the tag a request carries actually reaches the set path. Observed
//   through a real NNEvaluator and a real Search running a real search -- request-shaped
//   resolution in, attribution ledger out -- rather than by asserting that an assignment
//   statement exists.
//
// WHAT IS NOT WITNESSED HERE, and why. The cache_attach action does not exist yet, so nothing
// reaches NNEvaluator::attachCacheContext over the protocol; these tests call that seam
// directly. The protocol half that DOES exist -- the "cacheContext" field being parsed,
// resolved against the model the request selected, and refused when unknown -- is witnessed
// end to end against a running engine, in audit-reports/impl-cachecontext-witness.py, because
// a refusal is a property of a served response and not of a function this suite can call.

namespace {

//-------------------------------------------------------------------------------------
// Fixtures
//-------------------------------------------------------------------------------------

const char* const TMP_DIR_PREFIX = "tmpnncachecontext";

// The per-context hit surface, as the delta type appendDump takes.
//
// harvestHitCountsFor reports a context's RUNNING TOTAL, and appendDump takes a DELTA -- two
// different quantities, which is why the type refuses the conversion and this helper has to
// say what it is doing. The tests below each dump ONCE per context, into a log with no prior
// block, so the running total IS the delta and the two coincide exactly. That coincidence is
// what makes the conversion honest here and nowhere else: there is no per-context delta
// surface, so a second dump of the same context through this route would double-count, and no
// test below takes one.
NNCacheHitCountDelta asDelta(const NNCacheHitLedger& perContextTotals) {
  return NNCacheHitCountDelta::ofDeltaRows(perContextTotals.entries(), perContextTotals.unrecordedHits());
}

// A distinct key per serial, spread over both halves.
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

unique_ptr<NNOutput> ownedOutputFor(Hash128 hash) {
  unique_ptr<NNOutput> p(new NNOutput());
  p->nnHash = hash;
  p->nnXLen = 19;
  p->nnYLen = 19;
  return p;
}

vector<unique_ptr<NNOutput>> outputsFor(const vector<Hash128>& keys) {
  vector<unique_ptr<NNOutput>> out;
  for(size_t i = 0; i < keys.size(); i++)
    out.push_back(ownedOutputFor(keys[i]));
  return out;
}

unique_ptr<NNCacheTable> defaultLevelOne(int sizePowerOfTwo) {
  return NNCacheTable::create(NNCacheConfig::statusQuo(sizePowerOfTwo, 2));
}

// A two-level table over a small frozen level 0, which is the shape a session that attached
// anything actually runs. Its keys are nthKey(1000+i), well away from the level-1 keys below.
unique_ptr<NNCacheTable> twoLevelTable() {
  vector<Hash128> zeroKeys;
  for(int i = 0; i < 4; i++)
    zeroKeys.push_back(nthKey(1000 + i));
  return makeTwoLevelNNCacheTable(NNCacheFrozen::build(outputsFor(zeroKeys)), defaultLevelOne(10), 10);
}

// True if `f` threw a StringError whose message mentions `mustMention`. A named helper so a
// refusal is asserted on its message, not merely on the fact that something went wrong: a
// refusal that does not name what it refused leaves the client with nothing to act on.
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

bool contextOwns(const NNCacheAttributionLedger& ledger, Hash128 key, const string& context) {
  for(size_t i = 0; i < ledger.rows().size(); i++) {
    if(ledger.rows()[i].key == key)
      return ledger.rows()[i].context == context;
  }
  return false;
}

bool anyRowFor(const NNCacheAttributionLedger& ledger, Hash128 key) {
  for(size_t i = 0; i < ledger.rows().size(); i++) {
    if(ledger.rows()[i].key == key)
      return true;
  }
  return false;
}

bool logHasKey(const NNCacheCountLogContents& contents, Hash128 key) {
  for(size_t i = 0; i < contents.rows().size(); i++) {
    if(contents.rows()[i].key == key)
      return true;
  }
  return false;
}

uint64_t logLookupsFor(const NNCacheCountLogContents& contents, Hash128 key) {
  for(size_t i = 0; i < contents.rows().size(); i++) {
    if(contents.rows()[i].key == key)
      return contents.rows()[i].lookups;
  }
  testAssert(false);
  return 0;
}

//-------------------------------------------------------------------------------------
// Stage 1: the name space
//-------------------------------------------------------------------------------------

void testAnAttachedNameResolvesAndAnUnknownOneIsRefusedByName() {
  NNCacheContextSet contexts;
  const NNCacheContextId a = contexts.attach("card-5455");
  const NNCacheContextId b = contexts.attach("card-9001");
  testAssert(contexts.size() == 2);

  const NNCacheContextResolution resolvedA = contexts.resolveForRequest(std::optional<string>("card-5455"));
  testAssert(resolvedA.attribution().has_value());
  testAssert(resolvedA.attribution().value().isToContext());
  testAssert(resolvedA.attribution().value().contextId() == a);
  testAssert(!resolvedA.refusal().has_value());

  const NNCacheContextResolution resolvedB = contexts.resolveForRequest(std::optional<string>("card-9001"));
  testAssert(resolvedB.attribution().value().contextId() == b);

  // The refusal names the id it refused AND lists what is attached, so a typo is actionable.
  const NNCacheContextResolution unknown = contexts.resolveForRequest(std::optional<string>("card-5456"));
  testAssert(!unknown.attribution().has_value());
  testAssert(unknown.refusal().has_value());
  testAssert(unknown.refusal().value().find("card-5456") != string::npos);
  testAssert(unknown.refusal().value().find("card-5455") != string::npos);
  testAssert(unknown.refusal().value().find("card-9001") != string::npos);
}

void testTheDefaultIsTheSoleAttachedContextAndNothingElse() {
  {
    NNCacheContextSet one;
    const NNCacheContextId only = one.attach("card-5455");
    // With exactly one attached, everything the session earns belongs to it and there is
    // nothing to infer.
    const NNCacheContextResolution defaulted = one.resolveForRequest(std::optional<string>());
    testAssert(defaulted.attribution().value().isToContext());
    testAssert(defaulted.attribution().value().contextId() == only);
  }
  {
    NNCacheContextSet two;
    two.attach("card-5455");
    two.attach("card-9001");
    // With two attached, the engine cannot know which one a query served, and does not guess.
    const NNCacheContextResolution defaulted = two.resolveForRequest(std::optional<string>());
    testAssert(defaulted.attribution().has_value());
    testAssert(!defaulted.attribution().value().isToContext());
  }
  {
    NNCacheContextSet none;
    // Nothing attached: a request that names none behaves exactly as it did before contexts
    // existed, and a request that names one is refused rather than served under a name that
    // is not there.
    testAssert(!none.resolveForRequest(std::optional<string>()).attribution().value().isToContext());
    const NNCacheContextResolution named = none.resolveForRequest(std::optional<string>("card-5455"));
    testAssert(!named.attribution().has_value());
    testAssert(named.refusal().value().find("card-5455") != string::npos);
  }
}

void testAContextNameIsValidatedAsAPathComponentAndNeverTwice() {
  NNCacheContextSet contexts;
  contexts.attach("kata1-b18c384nbt-s9732312320-d4245566942");
  // The name becomes a component of two files' paths. It is refused, not escaped.
  testAssert(refused([&]() { contexts.attach(".."); }, ".."));
  // The refusal names the offending character and where it is, which is the actionable fact
  // for a name the client believes is legal.
  testAssert(refused([&]() { contexts.attach("card/5455"); }, "'/'"));
  testAssert(refused([&]() { contexts.attach(""); }, "context name"));
  // Two attachments under one name have no single answer to "which of them earned this".
  contexts.attach("card-5455");
  testAssert(refused([&]() { contexts.attach("card-5455"); }, "card-5455"));
}

void testAnIdFromAnotherCachesNameSpaceMeansNothingHere() {
  NNCacheContextSet mine;
  NNCacheContextSet theirs;
  mine.attach("card-5455");
  const NNCacheContextId foreign = theirs.attach("card-9001");
  // Same position (index 0) in both sets, and reading it against the wrong one would hand back
  // "card-5455" -- a different card, silently. It is refused instead.
  testAssert(!mine.owns(foreign));
  testAssert(refused([&]() { const string& n = mine.nameOf(foreign); (void)n; }, "different cache"));
}

//-------------------------------------------------------------------------------------
// Stage 2: the ledger, against the real per-context .nncounts file
//-------------------------------------------------------------------------------------

// THE CENTRAL WITNESS. Two contexts are attached to one table; each earns its own entries and
// one of them is retrieved twice. The observation is the pair of real count-log files on disk
// after each context's dump: A's file holds A's key with its two lookups, and B's file does
// NOT hold it. A ledger that filed the entry under the wrong context, or under both, is a
// different pair of files.
void testEachContextsEarningsReachThatContextsCountLogAndNoOthers() {
  TestCommon::ScopedTempDir dir(TMP_DIR_PREFIX);

  unique_ptr<NNCacheTable> table = twoLevelTable();
  const NNCacheContextId cardA = table->attachCacheContext("card-5455");
  const NNCacheContextId cardB = table->attachCacheContext("card-9001");

  const Hash128 earnedByA = nthKey(1);
  const Hash128 alsoEarnedByA = nthKey(2);
  const Hash128 earnedByB = nthKey(3);

  table->set(outputFor(earnedByA), NNCacheAttribution::toContext(cardA));
  table->set(outputFor(alsoEarnedByA), NNCacheAttribution::toContext(cardA));
  table->set(outputFor(earnedByB), NNCacheAttribution::toContext(cardB));

  // Two retrievals of A's key, so the count that lands in A's file is a number and not a zero
  // that any wiring would produce.
  shared_ptr<NNOutput> got;
  testAssert(table->get(earnedByA, got));
  testAssert(table->get(earnedByA, got));

  const NNCacheCountLog logA = NNCacheCountLog::forContext(dir.path(), "card-5455");
  const NNCacheCountLog logB = NNCacheCountLog::forContext(dir.path(), "card-9001");
  logA.appendDump(asDelta(table->harvestHitCountsFor(cardA)));
  logB.appendDump(asDelta(table->harvestHitCountsFor(cardB)));

  const NNCacheCountLogContents contentsA = logA.load();
  const NNCacheCountLogContents contentsB = logB.load();
  testAssert(contentsA.tail() == NNCacheCountLogTail::Intact);
  testAssert(contentsB.tail() == NNCacheCountLogTail::Intact);

  // A's file: A's two keys, with the retrievals that actually happened.
  testAssert(contentsA.rows().size() == 2);
  testAssert(logHasKey(contentsA, earnedByA));
  testAssert(logLookupsFor(contentsA, earnedByA) == 2);
  testAssert(logHasKey(contentsA, alsoEarnedByA));
  // A key earned and never retrieved is present with zero, because "this card earned this
  // position and nothing came back for it" is the fact that says to stop carrying it.
  testAssert(logLookupsFor(contentsA, alsoEarnedByA) == 0);
  testAssert(!logHasKey(contentsA, earnedByB));

  // B's file: B's one key, and none of A's. This is the leg that fails if attribution is
  // dropped, guessed, or shared.
  testAssert(contentsB.rows().size() == 1);
  testAssert(logHasKey(contentsB, earnedByB));
  testAssert(!logHasKey(contentsB, earnedByA));
  testAssert(!logHasKey(contentsB, alsoEarnedByA));

  // And the in-memory surface a cache_dump response reports agrees with the files.
  const NNCacheAttributionLedger ledger = table->harvestAttribution();
  testAssert(ledger.isAttributed());
  testAssert(ledger.rows().size() == 3);
  testAssert(contextOwns(ledger, earnedByA, "card-5455"));
  testAssert(contextOwns(ledger, alsoEarnedByA, "card-5455"));
  testAssert(contextOwns(ledger, earnedByB, "card-9001"));
  testAssert(ledger.noAttributableContextEntries() == 0);
  testAssert(ledger.unrecordedAttributions() == 0);
}

//-------------------------------------------------------------------------------------
// Stage 3: the residue nobody can attribute
//-------------------------------------------------------------------------------------

void testAnEntryWithNoAttributableContextIsCountedAndNotGuessedIntoOne() {
  TestCommon::ScopedTempDir dir(TMP_DIR_PREFIX);

  unique_ptr<NNCacheTable> table = twoLevelTable();
  const NNCacheContextId cardA = table->attachCacheContext("card-5455");
  const NNCacheContextId cardB = table->attachCacheContext("card-9001");

  const Hash128 orphan = nthKey(7);
  const Hash128 secondOrphan = nthKey(8);
  const Hash128 attributed = nthKey(9);
  // Two contexts attached and no field supplied: this is what the resolution boundary hands
  // back, verbatim.
  const NNCacheContextResolution resolution =
    table->cacheContexts().resolveForRequest(std::optional<string>());
  testAssert(!resolution.attribution().value().isToContext());
  table->set(outputFor(orphan), resolution.attribution().value());
  table->set(outputFor(secondOrphan), resolution.attribution().value());
  table->set(outputFor(attributed), NNCacheAttribution::toContext(cardA));

  const NNCacheAttributionLedger ledger = table->harvestAttribution();
  // COUNTED: the number is reported, so a dump can say how much of the session it is not
  // writing into any card's file.
  testAssert(ledger.noAttributableContextEntries() == 2);
  // NOT GUESSED: neither orphan is filed under the first attached context, or any context.
  testAssert(!anyRowFor(ledger, orphan));
  testAssert(!anyRowFor(ledger, secondOrphan));
  testAssert(ledger.rows().size() == 1);
  testAssert(contextOwns(ledger, attributed, "card-5455"));

  // And the same, observed on the files a dump actually writes: neither card's log carries it.
  const NNCacheCountLog logA = NNCacheCountLog::forContext(dir.path(), "card-5455");
  const NNCacheCountLog logB = NNCacheCountLog::forContext(dir.path(), "card-9001");
  logA.appendDump(asDelta(table->harvestHitCountsFor(cardA)));
  logB.appendDump(asDelta(table->harvestHitCountsFor(cardB)));
  testAssert(!logHasKey(logA.load(), orphan));
  testAssert(!logHasKey(logB.load(), orphan));
  testAssert(logHasKey(logA.load(), attributed));
}

void testAnAttributionFromAnotherCacheCannotBeSpentHere() {
  unique_ptr<NNCacheTable> mine = twoLevelTable();
  unique_ptr<NNCacheTable> theirs = twoLevelTable();
  mine->attachCacheContext("card-5455");
  const NNCacheContextId foreign = theirs->attachCacheContext("card-9001");

  // Position 0 in both name spaces. Spending it here would file the entry under "card-5455".
  testAssert(refused(
    [&]() { mine->set(outputFor(nthKey(11)), NNCacheAttribution::toContext(foreign)); },
    "different cache"
  ));
  testAssert(refused([&]() { mine->harvestHitCountsFor(foreign); }, "not attached to this cache"));
}

void testACacheWithNoAttachedContextIsExactlyWhatItWasBefore() {
  unique_ptr<NNCacheTable> table = defaultLevelOne(10);
  // No context attached: the table attributes nothing and says so, rather than reporting an
  // empty row list a caller would read as "this session earned nothing".
  testAssert(table->harvestAttribution().disposition() == NNCacheAttributionDisposition::NotAttributed);
  testAssert(refused(
    [&]() { const NNCacheAttributionLedger l = table->harvestAttribution(); const size_t n = l.rows().size(); (void)n; },
    "attributes nothing"
  ));

  // And a set through the attributing entry point stores exactly what the plain one stores.
  const Hash128 key = nthKey(21);
  table->set(outputFor(key), NNCacheAttribution::noAttributableContext());
  shared_ptr<NNOutput> got;
  testAssert(table->get(key, got));
  testAssert(got->nnHash == key);
  testAssert(table->harvestAttribution().disposition() == NNCacheAttributionDisposition::NotAttributed);
}

// A table that keeps no per-key hit counts answers the per-context dump question with the same
// NotCounted it answers the whole-table one with -- rather than an empty Counted ledger, which
// a dump would write as "this context earned nothing this session".
void testASingleLevelTablesPerContextDumpIsNotCountedRatherThanEmpty() {
  unique_ptr<NNCacheTable> table = defaultLevelOne(10);
  const NNCacheContextId card = table->attachCacheContext("card-5455");
  table->set(outputFor(nthKey(31)), NNCacheAttribution::toContext(card));

  // The attribution IS recorded -- the earnings are known.
  const NNCacheAttributionLedger ledger = table->harvestAttribution();
  testAssert(ledger.isAttributed());
  testAssert(contextOwns(ledger, nthKey(31), "card-5455"));
  // But there are no counts to put beside them, and that is said rather than implied.
  const NNCacheHitLedger perContext = table->harvestHitCountsFor(card);
  testAssert(perContext.disposition() == NNCacheHitLedgerDisposition::NotCounted);
}

// THE RELATION THE DEFAULT SIZE CLAIMS, ASSERTED RATHER THAN TRUSTED FROM A COMMENT.
//
// The previous version of this code carried a comment justifying its constant by a corpus
// ceiling the constant did not actually cover -- 2^18 rows against a stated 291,129-key
// reference -- and a per-row byte figure that was not the row type's real size. Prose is
// where that drift lived, so the check is here rather than in more prose (ADR-0011 Rule 2):
// the constant, the reference scale and the load-factor bar are three accessors, and this
// test asserts the relation between them. A future edit to any one of the three that breaks
// it fails the suite instead of leaving a comment that lies.
void testTheDefaultRecorderSizeCoversTheScaleItsRationaleNames() {
  const int pow = NNCacheAttributionRecorder::defaultPowerOfTwo();
  const int64_t rows = ((int64_t)1) << pow;
  const int64_t reference = NNCacheAttributionRecorder::sizingReferenceKeys();
  const int loadPercent = NNCacheAttributionRecorder::maxLoadFactorPercent();

  // The reference scale must fit under the load factor the bounded probe window is held to --
  // NOT merely under the row count, which is the test the old comment was implicitly making.
  testAssert(rows * loadPercent / 100 >= reference);
  // And the next size down must NOT satisfy it, so the constant is the smallest that does
  // rather than a round number that happens to be large enough. This is what makes the
  // rationale load-bearing instead of decorative.
  testAssert((rows / 2) * loadPercent / 100 < reference);

  cout << "attribution recorder: " << attributionRecorderBytes(pow) << " B at 2^" << pow
       << " rows of " << NNCacheAttributionRecorder::rowBytes() << " B, "
       << (100.0 * (double)reference / (double)rows) << "% occupancy at the "
       << reference << "-key reference scale" << endl;
}

// THE OVERFLOW COUNTER, EXERCISED. Everything else in this file asserts it is zero, which
// witnesses that ordinary work does not overflow but never witnesses that the counter can
// move -- a green that would stay green with the counter wired to a constant. A recorder of
// one row cannot hold two keys, so the second earning has nowhere to go: it is counted, and
// it does NOT displace the row already there, which is the half that matters (a recorder that
// overwrote somebody else's row would report a clean harvest and a wrong context).
void testAnAttributionThatCannotBeGivenARowIsCountedAndDisplacesNothing() {
  NNCacheContextSet contexts;
  const NNCacheContextId card = contexts.attach("card-5455");

  // One row, one mutex: the probe window wraps onto the same row every time.
  NNCacheAttributionRecorder tiny(0, 0);
  testAssert(tiny.unrecordedAttributions() == 0);

  tiny.record(nthKey(41), NNCacheAttribution::toContext(card));
  testAssert(tiny.unrecordedAttributions() == 0);
  testAssert(tiny.harvest(contexts).size() == 1);

  // Re-recording the SAME key is not an overflow: it finds its own row and wins there.
  tiny.record(nthKey(41), NNCacheAttribution::toContext(card));
  testAssert(tiny.unrecordedAttributions() == 0);
  testAssert(tiny.harvest(contexts).size() == 1);

  // A second, different key has nowhere to go.
  tiny.record(nthKey(42), NNCacheAttribution::toContext(card));
  testAssert(tiny.unrecordedAttributions() == 1);
  const vector<NNCacheAttributionRow> rows = tiny.harvest(contexts);
  testAssert(rows.size() == 1);
  testAssert(rows[0].key == nthKey(41));
  testAssert(tiny.keysFor(card).size() == 1);

  // An unattributable entry never reaches a row at all, so it is never an overflow either --
  // it lands in its own counter, whatever the recorder's occupancy.
  tiny.record(nthKey(43), NNCacheAttribution::noAttributableContext());
  testAssert(tiny.unrecordedAttributions() == 1);
  testAssert(tiny.noAttributableContextEntries() == 1);
}

//-------------------------------------------------------------------------------------
// Stage 4: the plumb, through a real evaluator and a real search
//-------------------------------------------------------------------------------------

// THE PLUMB WITNESS. Everything above hands an attribution straight to a table. This one hands
// it to a Search the way an analysis request does -- resolve against the model, set it on the
// search, run the search -- and observes the evaluator's own attribution ledger afterwards.
// The observation point is the ledger of the cache the search actually filled, so a tag that
// is dropped anywhere between Search and NNEvaluator::evaluate's set call shows up here as an
// empty ledger rather than as a passing assertion about an assignment.
void testTheRequestTagReachesTheEvaluatorsSetPathThroughASearch() {
  // A real search reads the score-value tables, and runtests frees them before this block of
  // suites precisely because the rest of the block does not need them. This one does, so it
  // takes them for its own scope and gives them back. initTables refuses loudly if they are
  // already live, so a future reordering of the suites fails here rather than running on a
  // table somebody else owns.
  ScoreValue::initTables();

  const bool logToStdout = false;
  const bool logToStderr = false;
  const bool logTime = false;
  Logger logger(nullptr, logToStdout, logToStderr, logTime);

  // debugSkipNeuralNet: the evaluations are stubs, but the cache path they travel is the real
  // one -- the same NNResultBuf, the same evaluate(), the same set at the end of it.
  NNEvaluator* nnEval = TestSearchCommon::startNNEval(
    "/dev/null", logger, "cachecontext", NNPos::MAX_BOARD_LEN, NNPos::MAX_BOARD_LEN,
    0, true, false, false, true, false
  );

  const NNCacheContextId card = nnEval->attachCacheContext("card-5455");
  (void)card;

  // Resolution exactly as the analysis boundary does it, including the refusal it hands a
  // client that names something this model has not attached.
  const NNCacheContextResolution refusedResolution =
    nnEval->resolveCacheContext(std::optional<string>("card-not-attached"));
  testAssert(!refusedResolution.attribution().has_value());
  testAssert(refusedResolution.refusal().value().find("card-not-attached") != string::npos);
  testAssert(refusedResolution.refusal().value().find("card-5455") != string::npos);

  const NNCacheContextResolution resolution =
    nnEval->resolveCacheContext(std::optional<string>("card-5455"));
  testAssert(resolution.attribution().has_value());

  SearchParams params;
  params.maxVisits = 10;
  Search* search = new Search(params, nnEval, &logger, "cacheContextPlumbSeed");
  search->setCacheAttribution(resolution.attribution().value());

  Board board = Board::parseBoard(7, 7, R"%%(
.......
.......
..x....
.......
....o..
.......
.......
)%%");
  const Player nextPla = P_BLACK;
  BoardHistory hist(board, nextPla, Rules::getTrompTaylorish(), 0, BoardHistoryModes(false, false));
  search->setPosition(nextPla, board, hist);
  search->runWholeSearch(nextPla);

  const NNCacheAttributionLedger ledger = nnEval->harvestCacheAttribution();
  testAssert(ledger.isAttributed());
  // The search evaluated real positions and every one of them is filed under the context the
  // request named. An empty ledger here is the tag never arriving.
  testAssert(ledger.rows().size() > 0);
  for(size_t i = 0; i < ledger.rows().size(); i++)
    testAssert(ledger.rows()[i].context == "card-5455");
  testAssert(ledger.noAttributableContextEntries() == 0);
  testAssert(ledger.unrecordedAttributions() == 0);
  cout << "cache context plumb: " << ledger.rows().size()
       << " level-1 entries earned by 'card-5455', 0 unattributed" << endl;

  delete search;

  // A TRIPWIRE FOR THE STALE-TAG CLASS, which is the one way this plumb can mis-file rather
  // than under-file. One NNResultBuf serves every evaluation its thread ever makes, so a tag
  // left standing on it after a call would be spent by the NEXT call -- filing one query's
  // earnings under the previous query's context, silently, because both are legal values.
  // evaluate() takes the tag out of the buffer, so the second evaluation below is unattributed;
  // if it did not, that evaluation would be filed under 'card-5455' and this leg would see two
  // rows and a zero.
  NNEvaluator* second = TestSearchCommon::startNNEval(
    "/dev/null", logger, "cachecontextreuse", NNPos::MAX_BOARD_LEN, NNPos::MAX_BOARD_LEN,
    0, true, false, false, true, false
  );
  second->attachCacheContext("card-5455");
  const NNCacheContextResolution once =
    second->resolveCacheContext(std::optional<string>("card-5455"));

  NNResultBuf buf;
  MiscNNInputParams nnInputParams;
  Board firstBoard = Board::parseBoard(7, 7, R"%%(
.......
.......
..x....
.......
.......
.......
.......
)%%");
  BoardHistory firstHist(firstBoard, P_WHITE, Rules::getTrompTaylorish(), 0, BoardHistoryModes(false, false));
  Board secondBoard = Board::parseBoard(7, 7, R"%%(
.......
.......
..x....
.......
....o..
.......
.......
)%%");
  BoardHistory secondHist(secondBoard, P_BLACK, Rules::getTrompTaylorish(), 0, BoardHistoryModes(false, false));

  buf.cacheAttribution = once.attribution().value();
  second->evaluate(firstBoard, firstHist, P_WHITE, nnInputParams, buf, false, false);
  // Deliberately NOT re-supplied. This is the mistake the class is about.
  second->evaluate(secondBoard, secondHist, P_BLACK, nnInputParams, buf, false, false);

  const NNCacheAttributionLedger reuse = second->harvestCacheAttribution();
  testAssert(reuse.rows().size() == 1);
  testAssert(reuse.rows()[0].context == "card-5455");
  testAssert(reuse.noAttributableContextEntries() == 1);

  delete second;
  delete nnEval;

  ScoreValue::freeTables();
}

}  // namespace

void Tests::runNNCacheContextTests() {
  cout << "Running NN cache context attribution tests" << endl;

  testAnAttachedNameResolvesAndAnUnknownOneIsRefusedByName();
  testTheDefaultIsTheSoleAttachedContextAndNothingElse();
  testAContextNameIsValidatedAsAPathComponentAndNeverTwice();
  testAnIdFromAnotherCachesNameSpaceMeansNothingHere();
  testEachContextsEarningsReachThatContextsCountLogAndNoOthers();
  testAnEntryWithNoAttributableContextIsCountedAndNotGuessedIntoOne();
  testAnAttributionFromAnotherCacheCannotBeSpentHere();
  testACacheWithNoAttachedContextIsExactlyWhatItWasBefore();
  testASingleLevelTablesPerContextDumpIsNotCountedRatherThanEmpty();
  testTheDefaultRecorderSizeCoversTheScaleItsRationaleNames();
  testAnAttributionThatCannotBeGivenARowIsCountedAndDisplacesNothing();
  testTheRequestTagReachesTheEvaluatorsSetPathThroughASearch();

  cout << "Done" << endl;
}
