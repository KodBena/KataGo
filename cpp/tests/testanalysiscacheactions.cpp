#include "../tests/tests.h"

#include <sstream>
#include <thread>
#include <chrono>

#include "../command/analysiscacheactions.h"
#include "../core/config_parser.h"
#include "../core/fileutils.h"
#include "../neuralnet/nncachecountlog.h"
#include "../neuralnet/nnevalcontainer.h"
#include "../tests/tinymodel.h"

// THE ANALYSIS ENGINE'S PERSISTED-CACHE ACTIONS: cache_attach, cache_detach, cache_dump and
// cache_stats, plus the concurrency refusal all of them rest on.
//
// WHAT IS OBSERVED HERE, AND WHERE. Two halves, split by what each claim is actually about.
//
//   THE DECODE BOUNDARY is a property of a request object alone -- which field is refused, and
//   what the client is told -- so it is exercised against the four decoders directly, with real
//   json objects. The refusal MESSAGES are asserted on, deliberately: a refusal that does not
//   say what the valid values are leaves a client with something it cannot act on, which is the
//   whole difference between failing loudly and merely failing.
//
//   THE ACTS are a property of files on disk and of a live cache table, so they are exercised
//   against a REAL NNEvaluator -- one of the two nets embedded in tinymodel.cpp, loaded through
//   the same Setup::initializeNNEvaluator path the engine uses -- with a real nnCacheDir. The
//   observation point for "a session's work survived" is therefore the actual .nnevals and
//   .nncounts files and the actual level-0 entry count after a re-attach, never the return
//   value of the call that just wrote them (ADR-0021 Rule 1).
//
// WHAT IS NOT OBSERVED HERE, and where it is. That the five actions work over the engine's real
// JSON line protocol -- an actual process, an actual stdin, an actual response line -- is a
// property of the running engine and is witnessed end to end by
// audit-reports/impl-cache-actions-witness.py, which drives a real katago analysis process. This
// file is the seam that makes each refusal cheap to exercise; it is not a substitute for that.
//
// AND ONE CLAIM THAT NO COMPILED TEST CAN CARRY, named rather than left out: that
// NNCacheCountLog::appendDump REFUSES an absolute hit surface. That is a compile-time property,
// so a test that observed it would have to fail to build. Its witness is a compilation, and it
// is reproduced by putting the two lines below in a file under cpp/ and compiling it:
//
//   #include "../neuralnet/nncachecountlog.h"
//   void f(const NNCacheCountLog& l, NNCacheTable& t) { (void)l.appendDump(t.harvestHitCounts()); }
//
//   g++ -std=c++17 -fsyntax-only -I. that_file.cpp
//   -> error: cannot convert 'NNCacheHitLedger' to 'const NNCacheHitCountDelta&'
//
// The RUNTIME consequence of getting that composition wrong -- the record inflating -- is
// witnessed in testnncachedump.cpp, which now has to launder the harvest through
// NNCacheHitCountDelta::ofDeltaRows on purpose to reach it.

using namespace std;
using json = nlohmann::json;

namespace {

const char* const TMP_DIR_PREFIX = "tmpanalysiscacheactions";
const char* const CONTEXT = "card-5455";
const char* const OTHER_CONTEXT = "card-9001";

//-------------------------------------------------------------------------------------
// Decode: the closed key sets
//-------------------------------------------------------------------------------------

json attachRequest() {
  json request;
  request["id"] = "a1";
  request["action"] = "cache_attach";
  request["context"] = CONTEXT;
  return request;
}

json dumpRequest() {
  json request;
  request["id"] = "a3";
  request["action"] = "cache_dump";
  request["context"] = CONTEXT;
  request["what"] = "both";
  return request;
}

// THE HAZARD THIS INCREMENT REFUSES TO INHERIT. An analysis query with an unknown top-level key
// is WARNED about and then analyzed anyway, and the warning is switchable off. Every field of a
// cache action decides which bytes on disk are read or written, so the same disposition here
// would mean a misspelled "levelOneFill" silently reads as "do not fill level 1" -- and reads as
// nothing at all to a client that set warnUnusedFields=false.
//
// Both polarities, and the red leg is the point: the same request WITHOUT the typo decodes.
void testAnUnknownFieldIsRefusedAndTheActionIsNotPerformed() {
  {
    json request = attachRequest();
    request["levelOneFill"] = json::object();  // the real field is "level1Fill"
    const CacheActionDecode<CacheAttachRequest> decoded = decodeCacheAttach(request);
    testAssert(!decoded.value().has_value());
    testAssert(decoded.refusal().has_value());
    testAssert(decoded.refusal().value().field == "levelOneFill");
    // The message names the field, and lists what IS accepted, so the client can fix it.
    testAssert(decoded.refusal().value().message.find("levelOneFill") != string::npos);
    testAssert(decoded.refusal().value().message.find("level1Fill") != string::npos);
    testAssert(decoded.refusal().value().message.find("foreignModelSources") != string::npos);
  }
  // The same request without the typo. If this did not decode, the leg above would be red for
  // the wrong reason.
  testAssert(decodeCacheAttach(attachRequest()).value().has_value());

  // The other three actions carry the same rule; a per-action check, because a closed key set
  // that only one action actually applied would be exactly as leaky as no closed key set.
  {
    json request;
    request["id"] = "a2";
    request["action"] = "cache_detach";
    request["context"] = CONTEXT;
    request["discardUndumpped"] = true;  // typo
    const CacheActionDecode<CacheDetachRequest> decoded = decodeCacheDetach(request);
    testAssert(decoded.refusal().has_value() && decoded.refusal().value().field == "discardUndumpped");
  }
  {
    json request = dumpRequest();
    request["admision"] = json::object();  // typo
    const CacheActionDecode<CacheDumpRequest> decoded = decodeCacheDump(request);
    testAssert(decoded.refusal().has_value() && decoded.refusal().value().field == "admision");
  }
  {
    json request;
    request["id"] = "a4";
    request["action"] = "cache_stats";
    request["context"] = CONTEXT;  // cache_stats takes no context: it reports the whole model
    const CacheActionDecode<CacheStatsRequest> decoded = decodeCacheStats(request);
    testAssert(decoded.refusal().has_value() && decoded.refusal().value().field == "context");
  }
  cout << "  an unknown field on each of the four actions is an error, not a warning" << endl;
}

void testTheLevelZeroBoundIsExactlyOneOfThree() {
  {
    json request = attachRequest();
    request["level0"] = json::object();
    request["level0"]["minLookups"] = 2;
    const CacheActionDecode<CacheAttachRequest> decoded = decodeCacheAttach(request);
    testAssert(decoded.value().has_value());
    testAssert(decoded.value().value().levelZeroBound.describe().find("2") != string::npos);
  }
  {
    // Two bounds are two different prefixes of one ranked order, and the engine will not pick.
    json request = attachRequest();
    request["level0"] = json::object();
    request["level0"]["minLookups"] = 2;
    request["level0"]["maxBytes"] = 1000;
    const CacheActionDecode<CacheAttachRequest> decoded = decodeCacheAttach(request);
    testAssert(decoded.refusal().has_value() && decoded.refusal().value().field == "level0");
    testAssert(decoded.refusal().value().message.find("exactly one") != string::npos);
  }
  {
    json request = attachRequest();
    request["level0"] = json::object();
    request["level0"]["minLookup"] = 2;  // typo inside the nested object
    const CacheActionDecode<CacheAttachRequest> decoded = decodeCacheAttach(request);
    testAssert(decoded.refusal().has_value() && decoded.refusal().value().field == "level0");
    testAssert(decoded.refusal().value().message.find("minLookup\"") != string::npos);
  }
  {
    // An empty object is not "no bound": it is a bound that says nothing, and it is refused
    // rather than read as all().
    json request = attachRequest();
    request["level0"] = json::object();
    testAssert(decodeCacheAttach(request).refusal().has_value());
  }
  // Absent IS all(), which is the documented default.
  testAssert(decodeCacheAttach(attachRequest()).value().value().levelZeroBound.describe() ==
             NNCacheLevelZeroBound::all().describe());
  cout << "  level0 takes exactly one of minLookups/maxEntries/maxBytes, or nothing" << endl;
}

void testTheLevelOneFillIsBoundedInBytesOrNotRequested() {
  {
    json request = attachRequest();
    request["level1Fill"] = false;
    const CacheActionDecode<CacheAttachRequest> decoded = decodeCacheAttach(request);
    testAssert(decoded.value().has_value());
    testAssert(!decoded.value().value().levelOneFillMaxBytes.has_value());
  }
  {
    // true has no budget the engine could invent, so it is refused rather than defaulted.
    json request = attachRequest();
    request["level1Fill"] = true;
    const CacheActionDecode<CacheAttachRequest> decoded = decodeCacheAttach(request);
    testAssert(decoded.refusal().has_value() && decoded.refusal().value().field == "level1Fill");
    testAssert(decoded.refusal().value().message.find("maxBytes") != string::npos);
  }
  {
    json request = attachRequest();
    request["level1Fill"] = json::object();
    request["level1Fill"]["maxBytes"] = 65536;
    const CacheActionDecode<CacheAttachRequest> decoded = decodeCacheAttach(request);
    testAssert(decoded.value().value().levelOneFillMaxBytes.value() == 65536);
  }
  cout << "  level1Fill is false, or a byte budget, and never a bare true" << endl;
}

void testTheDumpTargetIsRequiredAndClosed() {
  {
    json request = dumpRequest();
    request.erase("what");
    const CacheActionDecode<CacheDumpRequest> decoded = decodeCacheDump(request);
    testAssert(decoded.refusal().has_value() && decoded.refusal().value().field == "what");
    testAssert(decoded.refusal().value().message.find("counts") != string::npos);
    testAssert(decoded.refusal().value().message.find("evaluations") != string::npos);
    testAssert(decoded.refusal().value().message.find("both") != string::npos);
  }
  {
    json request = dumpRequest();
    request["what"] = "all";
    const CacheActionDecode<CacheDumpRequest> decoded = decodeCacheDump(request);
    testAssert(decoded.refusal().has_value() && decoded.refusal().value().field == "what");
  }
  testAssert(decodeCacheDump(dumpRequest()).value().value().what == CacheDumpWhat::Both);
  {
    json request = dumpRequest();
    request["what"] = "counts";
    testAssert(decodeCacheDump(request).value().value().what == CacheDumpWhat::Counts);
    request["what"] = "evaluations";
    testAssert(decodeCacheDump(request).value().value().what == CacheDumpWhat::Evaluations);
  }
  cout << "  cache_dump's \"what\" is required and closed to three values" << endl;
}

void testAContextIsRequiredOnTheThreeActionsThatActOnOne() {
  json request = attachRequest();
  request.erase("context");
  testAssert(decodeCacheAttach(request).refusal().value().field == "context");
  request["context"] = 5;
  testAssert(decodeCacheAttach(request).refusal().value().field == "context");
  request["context"] = "";
  testAssert(decodeCacheAttach(request).refusal().value().field == "context");
  cout << "  a cache action that acts on a context must name one" << endl;
}

// A NAME THE ENGINE WILL NEVER ACCEPT IS THE REQUEST FAILING TO READ, AND IS REPORTED AS THAT.
//
// The alphabet is enforced deeper too -- by the count log, the container and the context set,
// which all call the same NNCacheFileName::verify -- and if it were left to them the client would
// get field "action", which the documented error contract reserves for a well-formed request the
// engine declined to carry out. A client author following that contract would look for "context".
// So the decoder calls the same one function and answers under the right field.
//
// Both polarities: a name in the alphabet decodes.
void testAContextNameOutsideTheAlphabetIsRefusedUnderItsOwnField() {
  json request = attachRequest();
  request["context"] = "../escape";
  const CacheActionDecode<CacheAttachRequest> decoded = decodeCacheAttach(request);
  testAssert(decoded.refusal().has_value());
  testAssert(decoded.refusal().value().field == "context");
  // The message is the one shared validator's, so it names the alphabet a client must fit.
  testAssert(decoded.refusal().value().message.find("ASCII letters, digits") != string::npos);
  // Not rewritten into something acceptable: no request came out of this at all.
  testAssert(!decoded.value().has_value());

  request["context"] = "..";
  testAssert(decodeCacheDetach(request).refusal().has_value());

  // A real internalName-shaped context, which is inside the alphabet, still decodes.
  request["context"] = "card-5455.v2_a";
  testAssert(decodeCacheAttach(request).value().has_value());
  cout << "  an illegal context name is refused under field \"context\": \""
       << decoded.refusal().value().message.substr(0, 60) << "...\"" << endl;
}

void testForeignModelSourcesAreAnOrderedListWithoutRepeats() {
  json request = attachRequest();
  request["foreignModelSources"] = json::array({"weaker-net", "weaker-net"});
  const CacheActionDecode<CacheAttachRequest> decoded = decodeCacheAttach(request);
  testAssert(decoded.refusal().has_value() && decoded.refusal().value().field == "foreignModelSources");
  // The list IS the priority order, so one model cannot hold two positions in it.
  testAssert(decoded.refusal().value().message.find("priority") != string::npos);

  request["foreignModelSources"] = json::array({"weaker-net", "weakest-net"});
  const CacheActionDecode<CacheAttachRequest> accepted = decodeCacheAttach(request);
  testAssert(accepted.value().value().foreignModelSources.size() == 2);
  testAssert(accepted.value().value().foreignModelSources[0] == "weaker-net");
  cout << "  foreignModelSources is an ordered list and refuses a repeat" << endl;
}

//-------------------------------------------------------------------------------------
// The concurrency refusal
//-------------------------------------------------------------------------------------

// A get walks the level-0 resolution list lock-free while attach and detach mutate the vector it
// walks. The refusal names the open count, because "try later" without a number is not something
// a client can act on.
void testTheSwapRefusalNamesTheOpenRequestCount() {
  const string message = cacheSwapConcurrencyRefusal("cache_attach", 3);
  testAssert(message.find("cache_attach") != string::npos);
  testAssert(message.find("3") != string::npos);
  testAssert(message.find("open") != string::npos);
  cout << "  the attach/detach refusal names the action and the open-request count: \""
       << message.substr(0, 60) << "...\"" << endl;
}

//-------------------------------------------------------------------------------------
// The acts, against a real evaluator and real files
//-------------------------------------------------------------------------------------

// A real net with a real persisted-cache directory. Real rather than the debug stub because the
// stub declares itself "random", and the container's file name and its header both carry the
// model's internal name -- so a stub would make every model's container one file.
class RealEngineCache {
 public:
  RealEngineCache()
    : dir(TMP_DIR_PREFIX),
      cacheDir(TMP_DIR_PREFIX),
      logger(nullptr, false, false, false, false),
      attachments(0)
  {
    istringstream cfgIn(
      "nnCacheSizePowerOfTwo = 12\n"
      "nnMutexPoolSizePowerOfTwo = 8\n"
      "numSearchThreads = 1\n"
      "nnCacheDir = " + cacheDir.path() + "\n"
    );
    cfg.initialize(cfgIn);
    const bool randFileName = true;
    primary = TinyModelTest::loadEmbeddedModel(
      TinyModelTest::EmbeddedModel::Rect15B2C16, dir.path(), logger, cfg, randFileName
    );
    secondary = TinyModelTest::loadEmbeddedModel(
      TinyModelTest::EmbeddedModel::B1C6Nbt, dir.path(), logger, cfg, randFileName
    );
    vector<HostedModel> searchable;
    searchable.push_back(
      HostedModel{ModelAddress{primary.eval->getInternalModelName(), "-model", ModelRole::Searchable}, primary.eval}
    );
    searchable.push_back(
      HostedModel{
        ModelAddress{secondary.eval->getInternalModelName(), "-extra-model", ModelRole::Searchable}, secondary.eval
      }
    );
    hosts.reset(new AnalysisModelHosts(AnalysisModelHosts::create(std::move(searchable), std::optional<HostedModel>())));
    // ONE registry for the whole fixture, exactly as a running engine has one. It is not a
    // convenience: a context's NAME is registered on the evaluator once for the life of the
    // process and never unregistered, so a second registry would try to register a name the
    // first one already did and be refused -- which is the engine's own correct behaviour and
    // would only be a test artifact here.
    attachments = AnalysisCacheAttachments(hosts->numSearchable());
  }
  ~RealEngineCache() {
    delete primary.eval;
    delete secondary.eval;
  }
  RealEngineCache(const RealEngineCache&) = delete;
  RealEngineCache& operator=(const RealEngineCache&) = delete;

  TestCommon::ScopedTempDir dir;
  TestCommon::ScopedTempDir cacheDir;
  ConfigParser cfg;
  Logger logger;
  TinyModelTest::LoadedTinyModel primary;
  TinyModelTest::LoadedTinyModel secondary;
  unique_ptr<AnalysisModelHosts> hosts;
  AnalysisCacheAttachments attachments;
};

Hash128 nthKey(int serial) {
  return Hash128(
    ((uint64_t)(serial + 1)) * 0x9E3779B97F4A7C15ULL,
    ((uint64_t)(serial + 1)) * 0xD6E8FEB86659FD93ULL + 0x1234567ULL
  );
}

shared_ptr<NNOutput> makeOutput(int serial, bool withOwnerMap) {
  shared_ptr<NNOutput> out = make_shared<NNOutput>();
  out->nnHash = nthKey(serial);
  out->whiteWinProb = 0.5f;
  out->whiteLossProb = 0.5f;
  out->whiteNoResultProb = 0.0f;
  out->whiteScoreMean = 0.25f;
  out->whiteScoreMeanSq = 1.0f;
  out->whiteLead = 0.25f;
  out->varTimeLeft = 1.0f;
  out->shorttermWinlossError = 0.1f;
  out->shorttermScoreError = 0.2f;
  out->policyOptimismUsed = 0.0f;
  out->nnXLen = 5;
  out->nnYLen = 5;
  const int area = 25;
  for(int i = 0; i < NNPos::MAX_NN_POLICY_SIZE; i++)
    out->policyProbs[i] = 0.0f;
  for(int i = 0; i <= area; i++)
    out->policyProbs[i] = (i % 7 == 0) ? -1.0f : ((float)i * 0.5f + (float)serial);
  if(withOwnerMap) {
    out->whiteOwnerMap = new float[area];
    for(int i = 0; i < area; i++)
      out->whiteOwnerMap[i] = (float)i * 0.03125f;
  }
  return out;
}

AnalysisEngineCounters counters(int64_t requestsAccepted, int64_t openRequests) {
  return AnalysisEngineCounters{requestsAccepted, openRequests};
}

CacheAttachRequest attachAll(const string& context) {
  return CacheAttachRequest{
    context, NNCacheLevelZeroBound::all(), std::optional<int64_t>(), vector<string>()
  };
}

// The refusal an act raised, as a value, so each site reads it rather than growing its own try.
std::optional<string> refusalOf(const std::function<void()>& act) {
  try {
    act();
    return std::nullopt;
  }
  catch(const StringError& e) {
    return string(e.what());
  }
}

// THE COMPOSED WITNESS, and the one every other act here exists for: attach an empty context,
// earn entries against it, dump, detach, RE-ATTACH, and observe the entries come back -- read off
// the files, not off the return value of the call that wrote them.
void testASessionsWorkSurvivesADumpDetachReattachCycle(RealEngineCache& engine) {
  AnalysisCacheAttachments& attachments = engine.attachments;
  const SearchableModelIdx model = AnalysisModelHosts::PRIMARY_SEARCHABLE_IDX;
  NNEvaluator& eval = *engine.hosts->searchableEval(model);

  // A first attach of a context nothing has ever written: a missing container is a NORMAL answer,
  // not an error, and it loads as an empty level 0.
  const json firstAttach = cacheAttachExecute(*engine.hosts, model, attachments, attachAll(CONTEXT));
  testAssert(firstAttach["entriesInLevelZero"].get<int64_t>() == 0);
  testAssert(firstAttach["containerTail"].get<string>() == "intact");
  testAssert(eval.numLevelZeroSources() == 1);

  // The session earns three entries under that context and retrieves one of them twice.
  const NNCacheContextId contextId = attachments.attachmentFor(model, CONTEXT).contextId;
  const NNCacheAttribution attribution = NNCacheAttribution::toContext(contextId);
  for(int serial = 1; serial <= 3; serial++)
    eval.cacheTable().set(makeOutput(serial, serial == 2), attribution);
  shared_ptr<NNOutput> got;
  testAssert(eval.cacheTable().get(nthKey(1), got));
  testAssert(eval.cacheTable().get(nthKey(1), got));

  // The dump. Counts first, then evaluations, which is the order the admission predicate needs.
  const CacheDumpRequest dumpBoth{CONTEXT, CacheDumpWhat::Both, NNCacheDiskAdmission::all()};
  const json dumped = cacheDumpExecute(*engine.hosts, model, attachments, dumpBoth, counters(10, 0));
  testAssert(dumped["evaluations"]["entriesWritten"].get<int64_t>() == 3);
  testAssert(dumped["evaluations"]["alreadyPersisted"].get<int64_t>() == 0);
  testAssert(dumped["counts"]["bytesAppended"].get<int64_t>() > 0);
  testAssert(dumped["openRequestsAtDump"].get<int64_t>() == 0);

  // A SECOND dump with nothing in between writes NOTHING. Not "not much" -- nothing: the
  // persisted mark says those bytes are already in the file, and the count delta is empty.
  const json dumpedAgain = cacheDumpExecute(*engine.hosts, model, attachments, dumpBoth, counters(10, 0));
  testAssert(dumpedAgain["evaluations"]["entriesWritten"].get<int64_t>() == 0);
  testAssert(dumpedAgain["evaluations"]["alreadyPersisted"].get<int64_t>() == 3);

  // Detach: nothing is undumped, so it does not refuse.
  const json detached = cacheDetachExecute(
    *engine.hosts, model, attachments, CacheDetachRequest{CONTEXT, false}, counters(10, 0)
  );
  testAssert(detached["sourcesDetached"].get<int64_t>() == 1);
  testAssert(detached["storageReleased"].get<bool>());
  testAssert(eval.numLevelZeroSources() == 0);
  testAssert(!attachments.isAttached(model, CONTEXT));

  // THE WITNESS: re-attach and read what came back. Three entries in level 0, and the count log
  // holds the two retrievals of key 1 -- both figures read off the files this cycle wrote.
  const json reattached = cacheAttachExecute(*engine.hosts, model, attachments, attachAll(CONTEXT));
  testAssert(reattached["entriesInLevelZero"].get<int64_t>() == 3);
  const NNCacheCountLogContents counts =
    NNCacheCountLog::forContext(engine.cacheDir.path(), CONTEXT).load();
  testAssert(counts.tail() == NNCacheCountLogTail::Intact);
  bool foundKeyOne = false;
  for(size_t i = 0; i < counts.rows().size(); i++) {
    if(counts.rows()[i].key == nthKey(1)) {
      foundKeyOne = true;
      testAssert(counts.rows()[i].lookups == 2);
      // ONE dump in which this key earned a retrieval. Not "one dump that carried it through":
      // see NNCacheCountRow::sessions.
      testAssert(counts.rows()[i].sessions == 1);
    }
  }
  testAssert(foundKeyOne);

  // And the re-attached content really resolves: a get for a key this process never set in the
  // new attachment answers, out of level 0.
  shared_ptr<NNOutput> fromDisk;
  testAssert(eval.cacheTable().get(nthKey(3), fromDisk));
  testAssert(fromDisk->nnHash == nthKey(3));

  cout << "  attach -> earn -> dump -> detach -> re-attach: " << reattached["entriesInLevelZero"].get<int64_t>()
       << " entries and " << counts.rows().size() << " count rows came back off disk" << endl;

  // Leave the model clean for the tests after this one.
  (void)cacheDetachExecute(
    *engine.hosts, model, attachments, CacheDetachRequest{CONTEXT, true}, counters(10, 0)
  );
}

// THE SPLIT AN ATTACH ACTUALLY MAKES: the level-0 bound takes a prefix of the ranked order and
// the level-1 fill takes what is left, up to a byte budget. What is witnessed is that the
// remainder really lands in the LIVE table -- a get answers for a key level 0 does not hold --
// and that it lands marked as already on disk, so the next dump does not write it back.
void testALevelOneFillAdmitsTheRemainderAndMarksItPersisted(RealEngineCache& engine) {
  AnalysisCacheAttachments& attachments = engine.attachments;
  const SearchableModelIdx model = AnalysisModelHosts::PRIMARY_SEARCHABLE_IDX;
  NNEvaluator& eval = *engine.hosts->searchableEval(model);
  const string context = "card-split";

  // Put three entries on disk for this context, then let go of them.
  (void)cacheAttachExecute(*engine.hosts, model, attachments, attachAll(context));
  const NNCacheContextId contextId = attachments.attachmentFor(model, context).contextId;
  for(int serial = 301; serial <= 303; serial++)
    eval.cacheTable().set(makeOutput(serial, false), NNCacheAttribution::toContext(contextId));
  const CacheDumpRequest dumpEvaluations{context, CacheDumpWhat::Evaluations, NNCacheDiskAdmission::all()};
  testAssert(cacheDumpExecute(*engine.hosts, model, attachments, dumpEvaluations, counters(1, 0))
               ["evaluations"]["entriesWritten"].get<int64_t>() == 3);
  (void)cacheDetachExecute(*engine.hosts, model, attachments, CacheDetachRequest{context, true}, counters(1, 0));
  eval.clearCache();

  // Re-attach taking ONE key into level 0 and letting the rest into level 1.
  CacheAttachRequest split = attachAll(context);
  split.levelZeroBound = NNCacheLevelZeroBound::maxEntries(1);
  split.levelOneFillMaxBytes = (int64_t)1 << 20;
  const json attached = cacheAttachExecute(*engine.hosts, model, attachments, split);
  testAssert(attached["entriesInLevelZero"].get<int64_t>() == 1);
  testAssert(attached["levelOneFilled"].get<int64_t>() == 2);
  testAssert(attached["levelOneFilledBytes"].get<int64_t>() > 0);

  // All three resolve, out of two different levels.
  int resolved = 0;
  for(int serial = 301; serial <= 303; serial++) {
    shared_ptr<NNOutput> got;
    if(eval.cacheTable().get(nthKey(serial), got))
      resolved += 1;
  }
  testAssert(resolved == 3);

  // AND THE FILL IS NOT OWED BACK. Its bytes are already in the file a dump appends to, so a
  // dump now writes nothing and reports the two as already persisted. Without the provenance
  // mark this is where the container would grow by the whole filled remainder every session.
  const json dumpedAfterFill =
    cacheDumpExecute(*engine.hosts, model, attachments, dumpEvaluations, counters(1, 0));
  testAssert(dumpedAfterFill["evaluations"]["entriesWritten"].get<int64_t>() == 0);
  // THREE, not two, and the third is worth naming rather than rounding off: the attribution
  // recorder is a record of what this PROCESS earned under this context, and neither a detach
  // nor clearCache erases it -- so the key that went into level 0 is still one of the context's
  // attributed keys, still marked persisted by the dump that put it in the file. What the
  // number says is "every key this context is attributed is already on disk", which is exactly
  // the claim: the fill is not owed back.
  testAssert(dumpedAfterFill["evaluations"]["alreadyPersisted"].get<int64_t>() == 3);

  cout << "  level0 maxEntries=1 + level1Fill: 1 frozen, "
       << attached["levelOneFilled"].get<int64_t>() << " filled into level 1, all "
       << resolved << " resolve, 0 owed back to disk" << endl;
  (void)cacheDetachExecute(*engine.hosts, model, attachments, CacheDetachRequest{context, true}, counters(1, 0));
  eval.clearCache();
}

// A byte budget that fits nothing takes nothing -- the bound is denominated in the resident
// bytes that actually exhaust, not in a slot count standing proxy for them.
void testALevelOneFillBudgetOfZeroAdmitsNothing(RealEngineCache& engine) {
  AnalysisCacheAttachments& attachments = engine.attachments;
  const SearchableModelIdx model = AnalysisModelHosts::PRIMARY_SEARCHABLE_IDX;
  const string context = "card-split";
  CacheAttachRequest split = attachAll(context);
  split.levelZeroBound = NNCacheLevelZeroBound::maxEntries(1);
  split.levelOneFillMaxBytes = (int64_t)0;
  const json attached = cacheAttachExecute(*engine.hosts, model, attachments, split);
  testAssert(attached["entriesInLevelZero"].get<int64_t>() == 1);
  testAssert(attached["levelOneFilled"].get<int64_t>() == 0);
  cout << "  a level1Fill budget of 0 bytes admits 0 entries" << endl;
  (void)cacheDetachExecute(*engine.hosts, model, attachments, CacheDetachRequest{context, true}, counters(1, 0));
  engine.hosts->searchableEval(model)->clearCache();
}

// Silent discard is the silent-failure shape and silent auto-dump would make cache_dump no
// longer the one verb that writes. So the refusal wins, and it names what would be lost.
void testDetachRefusesUndumpedWorkUnlessTheClientSaysDiscard(RealEngineCache& engine) {
  AnalysisCacheAttachments& attachments = engine.attachments;
  const SearchableModelIdx model = AnalysisModelHosts::PRIMARY_SEARCHABLE_IDX;
  NNEvaluator& eval = *engine.hosts->searchableEval(model);

  (void)cacheAttachExecute(*engine.hosts, model, attachments, attachAll(OTHER_CONTEXT));
  const NNCacheContextId contextId = attachments.attachmentFor(model, OTHER_CONTEXT).contextId;
  eval.cacheTable().set(makeOutput(101, false), NNCacheAttribution::toContext(contextId));

  const std::optional<string> refusal = refusalOf([&]() {
    (void)cacheDetachExecute(
      *engine.hosts, model, attachments, CacheDetachRequest{OTHER_CONTEXT, false}, counters(1, 0)
    );
  });
  testAssert(refusal.has_value());
  // Named, and countable: "1 earned entries that are not on disk".
  testAssert(refusal.value().find("1 earned entries") != string::npos);
  testAssert(refusal.value().find("discardUndumped") != string::npos);
  // AND IT REALLY DID NOT DETACH. A refusal that had already freed the structure would be worse
  // than no refusal at all.
  testAssert(attachments.isAttached(model, OTHER_CONTEXT));
  testAssert(eval.numLevelZeroSources() == 1);

  // The same detach with the client saying so goes through, and says how much it threw away.
  const json discarded = cacheDetachExecute(
    *engine.hosts, model, attachments, CacheDetachRequest{OTHER_CONTEXT, true}, counters(1, 0)
  );
  testAssert(discarded["discardedUndumpedEntries"].get<int64_t>() == 1);
  testAssert(!attachments.isAttached(model, OTHER_CONTEXT));
  cout << "  detach refuses undumped work by name, and discards it only when asked: \""
       << refusal.value().substr(0, 70) << "...\"" << endl;
}

// The one case this version refuses, and it is a limit of the surface underneath rather than a
// policy: the unpersisted count delta is kept per TABLE, so with two contexts attached it cannot
// be divided between them and is not guessed into one.
void testDumpingCountsIsRefusedWhileTwoContextsAreAttached(RealEngineCache& engine) {
  AnalysisCacheAttachments& attachments = engine.attachments;
  const SearchableModelIdx model = AnalysisModelHosts::PRIMARY_SEARCHABLE_IDX;

  (void)cacheAttachExecute(*engine.hosts, model, attachments, attachAll(CONTEXT));
  // One context: counts dump fine. The red leg below is only meaningful against this green one.
  const CacheDumpRequest countsOnly{CONTEXT, CacheDumpWhat::Counts, NNCacheDiskAdmission::all()};
  testAssert(!refusalOf([&]() {
    (void)cacheDumpExecute(*engine.hosts, model, attachments, countsOnly, counters(1, 0));
  }).has_value());

  (void)cacheAttachExecute(*engine.hosts, model, attachments, attachAll(OTHER_CONTEXT));
  const std::optional<string> refusal = refusalOf([&]() {
    (void)cacheDumpExecute(*engine.hosts, model, attachments, countsOnly, counters(1, 0));
  });
  testAssert(refusal.has_value());
  testAssert(refusal.value().find("2 contexts are attached") != string::npos);
  testAssert(refusal.value().find(OTHER_CONTEXT) != string::npos);

  // The evaluations leg IS per context and is unaffected, which is what makes the refusal a
  // boundary rather than a wall.
  const CacheDumpRequest evaluationsOnly{CONTEXT, CacheDumpWhat::Evaluations, NNCacheDiskAdmission::all()};
  testAssert(!refusalOf([&]() {
    (void)cacheDumpExecute(*engine.hosts, model, attachments, evaluationsOnly, counters(1, 0));
  }).has_value());

  cout << "  dumping counts with two contexts attached is refused, naming both" << endl;
  (void)cacheDetachExecute(*engine.hosts, model, attachments, CacheDetachRequest{CONTEXT, true}, counters(1, 0));
  (void)cacheDetachExecute(*engine.hosts, model, attachments, CacheDetachRequest{OTHER_CONTEXT, true}, counters(1, 0));
}

// A foreign source is another model's container. An unknown name is refused naming the loaded
// vocabulary, and this model's own name is refused because attaching one file twice puts resident
// memory on the list that no lookup can reach.
void testForeignModelSourcesResolveAgainstTheLoadedModels(RealEngineCache& engine) {
  AnalysisCacheAttachments& attachments = engine.attachments;
  const SearchableModelIdx model = AnalysisModelHosts::PRIMARY_SEARCHABLE_IDX;
  NNEvaluator& eval = *engine.hosts->searchableEval(model);
  const string secondaryName = engine.secondary.eval->getInternalModelName();

  {
    CacheAttachRequest request = attachAll("card-foreign");
    request.foreignModelSources.push_back("no-such-net");
    const std::optional<string> refusal = refusalOf([&]() {
      (void)cacheAttachExecute(*engine.hosts, model, attachments, request);
    });
    testAssert(refusal.has_value());
    testAssert(refusal.value().find("no-such-net") != string::npos);
    // Names what IS loaded, so the client can act on it.
    testAssert(refusal.value().find(secondaryName) != string::npos);
    // And nothing was left half-attached by the refusal.
    testAssert(eval.numLevelZeroSources() == 0);
  }
  {
    CacheAttachRequest request = attachAll("card-foreign");
    request.foreignModelSources.push_back(eval.getInternalModelName());
    const std::optional<string> refusal = refusalOf([&]() {
      (void)cacheAttachExecute(*engine.hosts, model, attachments, request);
    });
    testAssert(refusal.has_value());
    testAssert(refusal.value().find("is the model this attach is FOR") != string::npos);
  }
  {
    // A real foreign source. Both containers are empty, so what is witnessed here is the LIST:
    // two sources on the resolution order, this model's own first.
    CacheAttachRequest request = attachAll("card-foreign");
    request.foreignModelSources.push_back(secondaryName);
    const json attached = cacheAttachExecute(*engine.hosts, model, attachments, request);
    testAssert(attached["sources"].size() == 2);
    testAssert(attached["sources"][0]["model"].get<string>() == eval.getInternalModelName());
    testAssert(attached["sources"][1]["model"].get<string>() == secondaryName);
    testAssert(eval.numLevelZeroSources() == 2);
    const json detached = cacheDetachExecute(
      *engine.hosts, model, attachments, CacheDetachRequest{"card-foreign", true}, counters(1, 0)
    );
    testAssert(detached["sourcesDetached"].get<int64_t>() == 2);
    testAssert(eval.numLevelZeroSources() == 0);
  }
  cout << "  foreignModelSources resolve against the loaded models, in list order" << endl;
}

void testStatsReportsWhatIsResidentAndWhatIsAttached(RealEngineCache& engine) {
  AnalysisCacheAttachments& attachments = engine.attachments;
  const SearchableModelIdx model = AnalysisModelHosts::PRIMARY_SEARCHABLE_IDX;
  NNEvaluator& eval = *engine.hosts->searchableEval(model);

  const json before = cacheStatsExecute(*engine.hosts, model, attachments);
  testAssert(before["contexts"].size() == 0);
  testAssert(before["cacheDirectory"].get<string>() == engine.cacheDir.path());
  testAssert(before["levelZeroSourcesAttached"].get<int64_t>() == 0);

  (void)cacheAttachExecute(*engine.hosts, model, attachments, attachAll(CONTEXT));
  const NNCacheContextId contextId = attachments.attachmentFor(model, CONTEXT).contextId;
  eval.cacheTable().set(makeOutput(201, false), NNCacheAttribution::toContext(contextId));

  const json after = cacheStatsExecute(*engine.hosts, model, attachments);
  testAssert(after["contexts"].size() == 1);
  testAssert(after["contexts"][0]["context"].get<string>() == CONTEXT);
  testAssert(after["contexts"][0]["unpersistedEntries"].get<int64_t>() == 1);
  testAssert(after["levelZeroSourcesAttached"].get<int64_t>() == 1);
  testAssert(after["residentEntries"].get<int64_t>() >= before["residentEntries"].get<int64_t>() + 1);
  cout << "  cache_stats reports " << after["residentEntries"].get<int64_t>() << " resident entries and "
       << after["contexts"].size() << " attached context(s)" << endl;

  (void)cacheDetachExecute(*engine.hosts, model, attachments, CacheDetachRequest{CONTEXT, true}, counters(1, 0));
}

// THE DEBUG TRIPWIRE, EXERCISED BY BYPASSING THE PROTOCOL LAYER ON PURPOSE.
//
// The protocol refuses cache_attach and cache_detach while any request is open, which is what
// keeps a lock-free resolution walk off a vector that is being mutated. That refusal lives in the
// engine's request loop, so a caller reaching NNEvaluator directly walks straight past it -- and
// what it would corrupt is memory, silently, later. So a debug build asserts the same property AT
// THE SITE, and this is the leg that trips it.
//
// IT IS FATAL BY DESIGN and therefore cannot run inside an ordinary test process: the assertion
// calls Global::fatalError, which does not return. It is gated behind an environment variable and
// witnessed by running the binary twice -- once without it, where the suite passes, and once with
// it, where the process dies at the assertion with its message on stderr. A release build compiles
// the whole thing out and this leg finds nothing to trip, which the message below says out loud
// rather than passing silently.
//
//   KATAGO_WITNESS_CACHE_SWAP_TRIPWIRE=1 ./build/katago runtests
//
// The observation point is the act itself, not a downstream symptom: an evaluation really is in
// flight on another thread, and the attach really is the call that fires (ADR-0021 Rules 1-2).
void testTheDebugTripwireFiresWhenTheProtocolLayerIsBypassed(RealEngineCache& engine) {
  const char* env = getenv("KATAGO_WITNESS_CACHE_SWAP_TRIPWIRE");
  if(env == NULL || env[0] == '\0' || string(env) == "0")
    return;

  NNEvaluator& eval = *engine.hosts->searchableEval(AnalysisModelHosts::PRIMARY_SEARCHABLE_IDX);
#ifdef NDEBUG
  cout << "  KATAGO_WITNESS_CACHE_SWAP_TRIPWIRE is set, but this is a RELEASE build (NDEBUG): the "
          "attach/detach tripwire is compiled out and there is nothing here to fire. Rebuild with "
          "assertions (-UNDEBUG) to witness it." << endl;
  (void)eval;
  return;
#else
  cout << "  KATAGO_WITNESS_CACHE_SWAP_TRIPWIRE is set: bypassing the protocol layer, an evaluation "
          "in flight, expecting a FATAL ERROR from the attach below." << endl;
  cout.flush();

  std::atomic<bool> stop(false);
  std::thread evaluator([&]() {
    Board board(5, 5);
    BoardHistory hist(board, P_BLACK, Rules::getTrompTaylorish(), 0, BoardHistoryModes());
    MiscNNInputParams params;
    while(!stop.load(std::memory_order_acquire)) {
      NNResultBuf buf;
      // skipCache, so every iteration is a real forward pass and the thread is inside evaluate()
      // essentially all of the time rather than being served from the cache.
      eval.evaluate(board, hist, P_BLACK, params, buf, true, false);
    }
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  // The bypass. In a debug build this does not return.
  const NNCacheLevelZeroSourceId id =
    eval.attachLevelZeroSource(NNCacheFrozen::build(vector<unique_ptr<NNOutput>>()));
  (void)id;

  stop.store(true, std::memory_order_release);
  evaluator.join();
  cout << "  the tripwire did NOT fire: the attach returned while an evaluation was in flight, "
          "which is the thing it exists to catch." << endl;
  testAssert(false);
#endif
}

// Every act says the same thing about an engine started without nnCacheDir, rather than one of
// them silently doing nothing.
void testAnEngineWithoutACacheDirectoryRefusesEveryAct() {
  TestCommon::ScopedTempDir dir("tmpanalysiscacheactionsnodir");
  Logger logger(nullptr, false, false, false, false);
  ConfigParser cfg;
  istringstream cfgIn(
    "nnCacheSizePowerOfTwo = 12\n"
    "nnMutexPoolSizePowerOfTwo = 8\n"
    "numSearchThreads = 1\n"
  );
  cfg.initialize(cfgIn);
  const TinyModelTest::LoadedTinyModel loaded =
    TinyModelTest::loadEmbeddedModel(TinyModelTest::EmbeddedModel::B1C6Nbt, dir.path(), logger, cfg, true);
  vector<HostedModel> searchable;
  searchable.push_back(
    HostedModel{ModelAddress{loaded.eval->getInternalModelName(), "-model", ModelRole::Searchable}, loaded.eval}
  );
  const AnalysisModelHosts hosts =
    AnalysisModelHosts::create(std::move(searchable), std::optional<HostedModel>());
  AnalysisCacheAttachments attachments(hosts.numSearchable());
  const SearchableModelIdx model = AnalysisModelHosts::PRIMARY_SEARCHABLE_IDX;

  const std::optional<string> refusal = refusalOf([&]() {
    (void)cacheAttachExecute(hosts, model, attachments, attachAll(CONTEXT));
  });
  testAssert(refusal.has_value());
  testAssert(refusal.value().find("nnCacheDir") != string::npos);
  cout << "  an engine without nnCacheDir refuses cache_attach, naming the config key" << endl;
  delete loaded.eval;
}

}  // namespace

void Tests::runAnalysisCacheActionTests() {
  cout << "Running analysis engine cache action tests" << endl;
  testAnUnknownFieldIsRefusedAndTheActionIsNotPerformed();
  testTheLevelZeroBoundIsExactlyOneOfThree();
  testTheLevelOneFillIsBoundedInBytesOrNotRequested();
  testTheDumpTargetIsRequiredAndClosed();
  testAContextIsRequiredOnTheThreeActionsThatActOnOne();
  testAContextNameOutsideTheAlphabetIsRefusedUnderItsOwnField();
  testForeignModelSourcesAreAnOrderedListWithoutRepeats();
  testTheSwapRefusalNamesTheOpenRequestCount();
  {
    RealEngineCache engine;
    testASessionsWorkSurvivesADumpDetachReattachCycle(engine);
    testALevelOneFillAdmitsTheRemainderAndMarksItPersisted(engine);
    testALevelOneFillBudgetOfZeroAdmitsNothing(engine);
    testDetachRefusesUndumpedWorkUnlessTheClientSaysDiscard(engine);
    testDumpingCountsIsRefusedWhileTwoContextsAreAttached(engine);
    testForeignModelSourcesResolveAgainstTheLoadedModels(engine);
    testStatsReportsWhatIsResidentAndWhatIsAttached(engine);
    testTheDebugTripwireFiresWhenTheProtocolLayerIsBypassed(engine);
  }
  testAnEngineWithoutACacheDirectoryRefusesEveryAct();
  cout << "Done" << endl;
}
