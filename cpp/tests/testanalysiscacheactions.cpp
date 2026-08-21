#include "../tests/tests.h"

#include <sstream>
#include <type_traits>

#include "../command/analysiscacheactions.h"
#include "../core/config_parser.h"
#include "../neuralnet/nncachecountlog.h"
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
  request["admission"] = json::object();
  request["admission"]["all"] = true;
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
    request["level0"]["minObservations"] = 2;
    const CacheActionDecode<CacheAttachRequest> decoded = decodeCacheAttach(request);
    testAssert(decoded.value().has_value());
    testAssert(decoded.value().value().levelZeroBound.describe().find("2") != string::npos);
  }
  {
    // Two bounds are two different prefixes of one ranked order, and the engine will not pick.
    json request = attachRequest();
    request["level0"] = json::object();
    request["level0"]["minObservations"] = 2;
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
  cout << "  level0 takes exactly one of minObservations/maxEntries/maxBytes, or nothing" << endl;
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

// admission IS OPTIONAL AND ITS ABSENCE MEANS minObservations(2) (ratified spec, ledger rows
// 1717/1722). This REPLACES the required-field refusal of cd200625, whose reason -- that the
// currency an admission ought to gate on was an open question, so no default could be picked
// honestly -- is spent now that the currency is settled. What is exercised here: the default a
// missing field gets, both explicit forms, every way of getting an explicit form wrong, and
// the teaching refusal the retired "minLookups" key earns.
void testTheDumpAdmissionDefaultsToSeenTwiceAndIsOtherwiseClosed() {
  {
    // ABSENT: the default, and it is the conservative arm rather than "write everything".
    json request = dumpRequest();
    request.erase("admission");
    const CacheActionDecode<CacheDumpRequest> decoded = decodeCacheDump(request);
    testAssert(decoded.value().has_value());
    const NNCacheDiskAdmission admission = decoded.value().value().admission;
    testAssert(admission.admits(2));
    testAssert(!admission.admits(1));
    testAssert(admission.describe().find("2") != string::npos);
  }
  {
    // AN EMPTY OBJECT IS NOT THE DEFAULT. Omitting the field says "give me yours"; sending {}
    // says "here is my policy" and then names none, which is a client bug worth reporting.
    json request = dumpRequest();
    request["admission"] = json::object();
    const CacheActionDecode<CacheDumpRequest> decoded = decodeCacheDump(request);
    testAssert(decoded.refusal().has_value() && decoded.refusal().value().field == "admission");
  }
  {
    // Both keys: exactly one, not "at most one".
    json request = dumpRequest();
    request["admission"]["minObservations"] = 2;
    const CacheActionDecode<CacheDumpRequest> decoded = decodeCacheDump(request);
    testAssert(decoded.refusal().has_value() && decoded.refusal().value().field == "admission");
  }
  {
    // all:false has no meaning -- there is no false form of "everything".
    json request = dumpRequest();
    request["admission"] = json::object();
    request["admission"]["all"] = false;
    const CacheActionDecode<CacheDumpRequest> decoded = decodeCacheDump(request);
    testAssert(decoded.refusal().has_value() && decoded.refusal().value().field == "admission");
  }
  // Both explicit forms decode.
  testAssert(decodeCacheDump(dumpRequest()).value().has_value());  // "all":true, from dumpRequest()
  {
    json request = dumpRequest();
    request["admission"] = json::object();
    request["admission"]["minObservations"] = 3;
    const CacheActionDecode<CacheDumpRequest> decoded = decodeCacheDump(request);
    testAssert(decoded.value().has_value());
    testAssert(decoded.value().value().admission.describe().find("3") != string::npos);
  }
  cout << "  cache_dump's \"admission\" defaults to minObservations(2) and is otherwise closed" << endl;
}

// THE RETIRED KEY IS A TEACHING REFUSAL, NOT AN ALIAS, and the refusal has to say why -- the
// quantity changed, not just the word, so accepting it would have admitted strictly more to
// disk than the client asked for with nothing in the response saying so.
void testTheRetiredMinLookupsKeyIsRefusedAndNamesItsReplacement() {
  {
    json request = dumpRequest();
    request["admission"] = json::object();
    request["admission"]["minLookups"] = 2;
    const CacheActionDecode<CacheDumpRequest> decoded = decodeCacheDump(request);
    testAssert(decoded.refusal().has_value() && decoded.refusal().value().field == "admission");
    const string& message = decoded.refusal().value().message;
    testAssert(message.find("minObservations") != string::npos);
    testAssert(message.find("no longer accepted") != string::npos);
    // It explains the CHANGE, not just the spelling: a client that reads only the message has
    // enough to know its policy means something different now.
    testAssert(message.find("OBSERVATION") != string::npos);
    testAssert(message.find("RETRIEVAL") != string::npos);
  }
  {
    json request = attachRequest();
    request["level0"] = json::object();
    request["level0"]["minLookups"] = 2;
    const CacheActionDecode<CacheAttachRequest> decoded = decodeCacheAttach(request);
    testAssert(decoded.refusal().has_value() && decoded.refusal().value().field == "level0");
    testAssert(decoded.refusal().value().message.find("minObservations") != string::npos);
  }
  cout << "  the retired \"minLookups\" key is refused by name on both actions, naming what "
       << "replaced it and why" << endl;
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

AnalysisEngineCounters counters(int64_t openRequests) {
  return AnalysisEngineCounters{openRequests};
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

  // The session PRESENTS three positions under that context, evaluating each once, and then
  // asks for one of them twice more. observe() is the door NNEvaluator::evaluate calls once per
  // request; the get/set pair beside it is what that request then did with the cache.
  const NNCacheContextId contextId = attachments.attachmentFor(model, CONTEXT).contextId;
  const NNCacheAttribution attribution = NNCacheAttribution::toContext(contextId);
  for(int serial = 1; serial <= 3; serial++) {
    eval.cacheTable().observe(nthKey(serial), attribution);
    eval.cacheTable().set(makeOutput(serial, serial == 2), attribution);
  }
  shared_ptr<NNOutput> got;
  for(int again = 0; again < 2; again++) {
    eval.cacheTable().observe(nthKey(1), attribution);
    testAssert(eval.cacheTable().get(nthKey(1), got));
  }

  // The dump. Counts first, then evaluations, which is the order the admission predicate needs.
  const CacheDumpRequest dumpBoth{CONTEXT, CacheDumpWhat::Both, NNCacheDiskAdmission::all()};
  const json dumped = cacheDumpExecute(*engine.hosts, model, attachments, dumpBoth, counters(0));
  testAssert(dumped["evaluations"]["entriesWritten"].get<int64_t>() == 3);
  testAssert(dumped["evaluations"]["alreadyPersisted"].get<int64_t>() == 0);
  testAssert(dumped["counts"]["bytesAppended"].get<int64_t>() > 0);
  testAssert(dumped["openRequestsAtDump"].get<int64_t>() == 0);

  // A SECOND dump with nothing in between writes NOTHING. Not "not much" -- nothing: the
  // persisted mark says those bytes are already in the file, and the count delta is empty.
  const json dumpedAgain = cacheDumpExecute(*engine.hosts, model, attachments, dumpBoth, counters(0));
  testAssert(dumpedAgain["evaluations"]["entriesWritten"].get<int64_t>() == 0);
  testAssert(dumpedAgain["evaluations"]["alreadyPersisted"].get<int64_t>() == 3);

  // Detach: nothing is undumped, so it does not refuse.
  const json detached = cacheDetachExecute(
    *engine.hosts, model, attachments, CacheDetachRequest{CONTEXT, false}
  );
  testAssert(detached["sourcesDetached"].get<int64_t>() == 1);
  testAssert(detached["storageReleased"].get<bool>());
  testAssert(eval.numLevelZeroSources() == 0);
  testAssert(!attachments.isAttached(model, CONTEXT));

  // THE WITNESS: re-attach and read what came back. Three entries in level 0, and the count log
  // holds the THREE observations of key 1 -- one for the request that evaluated it and one for
  // each of the two that were answered from cache. Both figures read off the files this cycle
  // wrote.
  const json reattached = cacheAttachExecute(*engine.hosts, model, attachments, attachAll(CONTEXT));
  testAssert(reattached["entriesInLevelZero"].get<int64_t>() == 3);
  const NNCacheCountLogContents counts =
    NNCacheCountLog::forContext(engine.cacheDir.path(), CONTEXT).load();
  testAssert(counts.tail() == NNCacheCountLogTail::Intact);
  bool foundKeyOne = false;
  for(size_t i = 0; i < counts.rows().size(); i++) {
    if(counts.rows()[i].key == nthKey(1)) {
      foundKeyOne = true;
      testAssert(counts.rows()[i].observations == 3);
      // ONE dump in which this key was observed. Not "one dump that carried it through":
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
    *engine.hosts, model, attachments, CacheDetachRequest{CONTEXT, true}
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
  testAssert(cacheDumpExecute(*engine.hosts, model, attachments, dumpEvaluations, counters(0))
               ["evaluations"]["entriesWritten"].get<int64_t>() == 3);
  (void)cacheDetachExecute(*engine.hosts, model, attachments, CacheDetachRequest{context, true});
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
    cacheDumpExecute(*engine.hosts, model, attachments, dumpEvaluations, counters(0));
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
  (void)cacheDetachExecute(*engine.hosts, model, attachments, CacheDetachRequest{context, true});
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
  (void)cacheDetachExecute(*engine.hosts, model, attachments, CacheDetachRequest{context, true});
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
      *engine.hosts, model, attachments, CacheDetachRequest{OTHER_CONTEXT, false}
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
    *engine.hosts, model, attachments, CacheDetachRequest{OTHER_CONTEXT, true}
  );
  testAssert(discarded["discardedUndumpedEntries"].get<int64_t>() == 1);
  testAssert(!attachments.isAttached(model, OTHER_CONTEXT));
  cout << "  detach refuses undumped work by name, and discards it only when asked: \""
       << refusal.value().substr(0, 70) << "...\"" << endl;
}

// One key's row in a context's count log, or nothing if the log does not mention it. Read off the
// file every time, because the file is the observation point these tests exist for.
std::optional<NNCacheCountRow> countRowFor(const string& dir, const string& context, Hash128 key) {
  const NNCacheCountLogContents contents = NNCacheCountLog::forContext(dir, context).load();
  testAssert(contents.tail() == NNCacheCountLogTail::Intact);
  for(size_t i = 0; i < contents.rows().size(); i++) {
    if(contents.rows()[i].key == key)
      return contents.rows()[i];
  }
  return std::nullopt;
}

// COUNTS DUMPED PER CONTEXT, WITH TWO CARDS ATTACHED AT ONCE -- which this action refused outright
// until the table could divide its unpersisted delta by context.
//
// THE REFUSAL WAS HONEST AND THE LIMIT WAS REAL: the delta is what a count log may be handed,
// because a record is an INCREMENT, and a whole-table delta written into one card's file would
// file the other card's retrievals under it with no field of the response saying so. What has
// changed is not the policy but the surface underneath -- the division is now made where the two
// facts live (which source serves which context, which context earned which key) rather than
// attempted here, where only keys are visible and a key names a position, never a card.
//
// THE OBSERVATION IS THE .nncounts FILES, not the response: each card's own file must carry its
// own retrievals and none of the other's, and the second dump of an untouched card must leave
// every row of its file exactly as it was.
void testCountsAreDumpedPerContextWithTwoContextsAttached(RealEngineCache& engine) {
  AnalysisCacheAttachments& attachments = engine.attachments;
  const SearchableModelIdx model = AnalysisModelHosts::PRIMARY_SEARCHABLE_IDX;
  NNEvaluator& eval = *engine.hosts->searchableEval(model);
  const string cardA = "card-multi-a";
  const string cardB = "card-multi-b";
  const CacheDumpRequest countsA{cardA, CacheDumpWhat::Counts, NNCacheDiskAdmission::all()};
  const CacheDumpRequest countsB{cardB, CacheDumpWhat::Counts, NNCacheDiskAdmission::all()};

  // A FIRST SESSION FOR CARD A ALONE, so that card A has real pre-warmed content on disk for the
  // second session to serve out of level 0. That is the half no attribution can see -- a level-0
  // retrieval calls no set() -- and it is the half a mature card is made of.
  (void)cacheAttachExecute(*engine.hosts, model, attachments, attachAll(cardA));
  const NNCacheContextId idA = attachments.attachmentFor(model, cardA).contextId;
  for(int serial = 401; serial <= 403; serial++)
    eval.cacheTable().set(makeOutput(serial, false), NNCacheAttribution::toContext(idA));
  const CacheDumpRequest bothA{cardA, CacheDumpWhat::Both, NNCacheDiskAdmission::all()};
  testAssert(cacheDumpExecute(*engine.hosts, model, attachments, bothA, counters(0))
               ["evaluations"]["entriesWritten"].get<int64_t>() == 3);
  (void)cacheDetachExecute(*engine.hosts, model, attachments, CacheDetachRequest{cardA, false});
  eval.clearCache();

  // THE SECOND SESSION: both cards attached at once, which is the shape the refusal made
  // undumpable.
  const json attachedA = cacheAttachExecute(*engine.hosts, model, attachments, attachAll(cardA));
  testAssert(attachedA["entriesInLevelZero"].get<int64_t>() == 3);
  (void)cacheAttachExecute(*engine.hosts, model, attachments, attachAll(cardB));
  const NNCacheContextId idB = attachments.attachmentFor(model, cardB).contextId;
  testAssert(attachments.attachedContexts(model).size() == 2);

  // Card A earns nothing and is asked twice for one of its own pre-warmed positions. Card B is
  // asked once for a position it has to evaluate. Both halves of both cards are therefore live,
  // and BOTH are observations -- which is the currency change: A's card would once have been
  // the only one of the two with anything to write.
  shared_ptr<NNOutput> got;
  for(int again = 0; again < 2; again++) {
    eval.cacheTable().observe(nthKey(401), NNCacheAttribution::toContext(idA));
    testAssert(eval.cacheTable().get(nthKey(401), got));
  }
  eval.cacheTable().observe(nthKey(451), NNCacheAttribution::toContext(idB));
  eval.cacheTable().set(makeOutput(451, false), NNCacheAttribution::toContext(idB));

  // A'S DUMP, with B attached. It writes A's observations...
  const json dumpedA = cacheDumpExecute(*engine.hosts, model, attachments, countsA, counters(0));
  testAssert(dumpedA["counts"]["bytesAppended"].get<int64_t>() > 0);
  const std::optional<NNCacheCountRow> aRow = countRowFor(engine.cacheDir.path(), cardA, nthKey(401));
  testAssert(aRow.has_value());
  testAssert(aRow.value().observations == 2);
  testAssert(aRow.value().sessions == 1);
  // ...AND NOT B'S. Watched as a positive observation about A's file: B's key is not in it, and
  // B's file does not exist yet at all.
  testAssert(!countRowFor(engine.cacheDir.path(), cardA, nthKey(451)).has_value());
  testAssert(NNCacheCountLog::forContext(engine.cacheDir.path(), cardB).load().rows().empty());

  // A SECOND DUMP OF A, WITH NOTHING IN BETWEEN, APPENDS NOTHING. Not "little" -- the delta is
  // empty, so no row of A's file changes and no key's sessions rises. Read off the file: a dump
  // that had re-appended the running total would show observations 2 -> 4 and sessions 1 -> 2, which
  // is exactly the record inflation the delta type exists to prevent.
  const json dumpedAgain = cacheDumpExecute(*engine.hosts, model, attachments, countsA, counters(0));
  const std::optional<NNCacheCountRow> aRowAgain = countRowFor(engine.cacheDir.path(), cardA, nthKey(401));
  testAssert(aRowAgain.has_value());
  testAssert(aRowAgain.value().observations == 2);
  testAssert(aRowAgain.value().sessions == 1);
  testAssert(dumpedAgain["counts"]["rowsInLog"].get<int64_t>() == dumpedA["counts"]["rowsInLog"].get<int64_t>());

  // AND B'S DELTA IS STILL WHOLE AFTER A TOOK ITS OWN. A whole-table take dressed as a per-context
  // one would have consumed B's observation here and dropped it on the floor, leaving B's file
  // claiming its position never came up.
  const json dumpedB = cacheDumpExecute(*engine.hosts, model, attachments, countsB, counters(0));
  testAssert(dumpedB["counts"]["bytesAppended"].get<int64_t>() > 0);
  const std::optional<NNCacheCountRow> bRow = countRowFor(engine.cacheDir.path(), cardB, nthKey(451));
  testAssert(bRow.has_value());
  testAssert(bRow.value().observations == 1);
  testAssert(bRow.value().sessions == 1);
  testAssert(!countRowFor(engine.cacheDir.path(), cardB, nthKey(401)).has_value());

  cout << "  two contexts attached: card A's counts dump wrote its own level-0 observations ("
       << aRow.value().observations << " observations, " << aRow.value().sessions
       << " session) and none of B's; a second dump left them at " << aRowAgain.value().observations
       << "/" << aRowAgain.value().sessions << "; B's own dump then carried its "
       << bRow.value().observations << endl;

  (void)cacheDetachExecute(*engine.hosts, model, attachments, CacheDetachRequest{cardA, true});
  (void)cacheDetachExecute(*engine.hosts, model, attachments, CacheDetachRequest{cardB, true});
  eval.clearCache();
}

// THE UNDER-REFUSAL, EXHIBITED, AND THE QUERY THAT CLOSES IT.
//
// cache_detach refuses to drop a context holding work that is not on disk. Its evaluations half is
// exact. Its counts half was a PROXY -- "has the engine accepted any request since this model last
// dumped counts" -- documented as over-refusing and never under-refusing, WHICH WAS FALSE AS
// STATED. Two facts make the gap, and both are structural rather than unlucky:
//
//   A LEVEL-0 HIT NEVER CALLS set(). NNCacheTwoLevelTable::get bumps the frozen entry's counter
//   and returns; the attribution recorder is written on the set path alone. So a query served
//   entirely out of pre-warmed content earns no attributed key, and unpersistedKeysFor -- the
//   exact half of the refusal -- stays at zero while retrievals accrue.
//
//   THE PROXY'S COUNTER RISES AT ACCEPTANCE, NOT AT COMPLETION. A dump is legal while requests are
//   open and advances its marks at its own harvest, so a request accepted BEFORE a dump and still
//   running after it keeps recording retrievals the dump did not write while no new acceptance is
//   ever observed.
//
// THE STATE BOTH BLIND SPOTS MEET IN is built below out of the production actions alone: a fully
// pre-warmed card, re-attached, whose session only re-studies positions it already had. THE
// EXHIBIT IS THE LOSS ITSELF -- the retrievals are shown to be gone from the file after a detach
// that the old pair permitted -- and then the same history is shown refused by the query that
// replaced it.
void testDetachSeesUndumpedCountsThatTheOldProxyCouldNot(RealEngineCache& engine) {
  AnalysisCacheAttachments& attachments = engine.attachments;
  const SearchableModelIdx model = AnalysisModelHosts::PRIMARY_SEARCHABLE_IDX;
  NNEvaluator& eval = *engine.hosts->searchableEval(model);
  const string card = "card-prewarmed";
  const CacheDumpRequest dumpCounts{card, CacheDumpWhat::Counts, NNCacheDiskAdmission::all()};

  // A first session puts two positions on disk and dumps everything, so the card is mature.
  (void)cacheAttachExecute(*engine.hosts, model, attachments, attachAll(card));
  const NNCacheContextId contextId = attachments.attachmentFor(model, card).contextId;
  for(int serial = 501; serial <= 502; serial++)
    eval.cacheTable().set(makeOutput(serial, false), NNCacheAttribution::toContext(contextId));
  const CacheDumpRequest dumpBoth{card, CacheDumpWhat::Both, NNCacheDiskAdmission::all()};
  testAssert(cacheDumpExecute(*engine.hosts, model, attachments, dumpBoth, counters(0))
               ["evaluations"]["entriesWritten"].get<int64_t>() == 2);
  (void)cacheDetachExecute(*engine.hosts, model, attachments, CacheDetachRequest{card, false});
  eval.clearCache();

  // THE SESSION THE PROXY WAS BLIND TO: everything served out of level 0, nothing earned.
  const json reattached = cacheAttachExecute(*engine.hosts, model, attachments, attachAll(card));
  testAssert(reattached["entriesInLevelZero"].get<int64_t>() == 2);
  shared_ptr<NNOutput> got;
  const NNCacheContextId prewarmedId = attachments.attachmentFor(model, card).contextId;
  const int prewarmedSerials[3] = {501, 501, 502};
  shared_ptr<NNOutput> unused;
  for(int i = 0; i < 3; i++) {
    eval.cacheTable().observe(nthKey(prewarmedSerials[i]), NNCacheAttribution::toContext(prewarmedId));
    testAssert(eval.cacheTable().get(nthKey(prewarmedSerials[i]), got));
  }
  (void)unused;

  // THE EXHIBIT, HALF ONE: both quantities the OLD refusal read are silent in this state.
  //
  //   unpersistedKeysFor is empty -- observed here, and it is the very expression cacheDetachExecute
  //   still evaluates for its evaluations half.
  //
  //   The proxy's own comparison was `requestsAcceptedSoFar > requestsAcceptedAtLastCountsDump`,
  //   against a counter this test supplies exactly as the request loop did. No request has been
  //   accepted since the dump above, so both sides are the value passed at that dump and the
  //   comparison is false. It is arithmetic on numbers visible at this call site, not an inference
  //   about a mechanism.
  const NNCacheContextId liveContextId = attachments.attachmentFor(model, card).contextId;
  testAssert(eval.cacheTable().unpersistedKeysFor(liveContextId).empty());
  // And the observations really are there to be lost, read off the per-context surface rather
  // than the whole-table one -- this fixture's engine has served other cards, and a figure that
  // included them would not be about this card at all.
  const NNCacheObservationLedger held = eval.cacheTable().harvestObservationCountsFor(liveContextId);
  int64_t heldObservations = 0;
  for(size_t i = 0; i < held.entries().size(); i++)
    heldObservations += (int64_t)held.entries()[i].observations;
  testAssert(heldObservations == 3);

  // THE EXHIBIT, HALF TWO: THE LOSS. A detach that both blind checks permitted is what
  // discardUndumped:true reproduces exactly -- the same act, with the refusal stood down -- and
  // the file afterwards is the witness. The three observations are not in it, and nothing
  // anywhere says they happened.
  (void)cacheDetachExecute(*engine.hosts, model, attachments, CacheDetachRequest{card, true});
  const std::optional<NNCacheCountRow> afterLoss = countRowFor(engine.cacheDir.path(), card, nthKey(501));
  testAssert(!afterLoss.has_value());
  eval.clearCache();

  // NOW THE SAME HISTORY AGAIN, AGAINST THE QUERY THAT REPLACED THE PROXY.
  (void)cacheAttachExecute(*engine.hosts, model, attachments, attachAll(card));
  const NNCacheContextId secondId = attachments.attachmentFor(model, card).contextId;
  for(int i = 0; i < 3; i++) {
    eval.cacheTable().observe(nthKey(prewarmedSerials[i]), NNCacheAttribution::toContext(secondId));
    testAssert(eval.cacheTable().get(nthKey(prewarmedSerials[i]), got));
  }
  testAssert(eval.cacheTable().unpersistedKeysFor(secondId).empty());          // still silent
  testAssert(eval.cacheTable().hasUnpersistedObservationCountsFor(secondId));  // and this is not

  const std::optional<string> refusal = refusalOf([&]() {
    (void)cacheDetachExecute(*engine.hosts, model, attachments, CacheDetachRequest{card, false});
  });
  testAssert(refusal.has_value());
  testAssert(refusal.value().find("observation counts that have not been dumped") != string::npos);
  // Naming zero undumped ENTRIES while refusing anyway is the whole point: this is the state the
  // entries half cannot see.
  testAssert(refusal.value().find("0 earned entries") != string::npos);
  testAssert(attachments.isAttached(model, card));

  // AND THE REFUSAL IS ACTIONABLE: the dump it asks for writes exactly those observations, after
  // which the same detach goes through.
  (void)cacheDumpExecute(*engine.hosts, model, attachments, dumpCounts, counters(0));
  const std::optional<NNCacheCountRow> saved = countRowFor(engine.cacheDir.path(), card, nthKey(501));
  testAssert(saved.has_value());
  testAssert(saved.value().observations == 2);
  testAssert(!eval.cacheTable().hasUnpersistedObservationCountsFor(secondId));
  testAssert(!refusalOf([&]() {
    (void)cacheDetachExecute(*engine.hosts, model, attachments, CacheDetachRequest{card, false});
  }).has_value());

  cout << "  level-0-only session: 0 earned entries and " << heldObservations
       << " observations; a detach with the refusal stood down left the count log with no row for the key, "
       << "and the non-consuming query refuses instead: \"" << refusal.value().substr(0, 80) << "...\"" << endl;
  eval.clearCache();
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
      *engine.hosts, model, attachments, CacheDetachRequest{"card-foreign", true}
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

  (void)cacheDetachExecute(*engine.hosts, model, attachments, CacheDetachRequest{CONTEXT, true});
}

// WHAT cache_stats REPORTS ABOUT THE COUNT LOG'S OWN CURRENCY.
//
// THIS REPLACES THE GATED admissionSignalMeasurement LEGS. Those measured three candidate
// currencies side by side -- retrievals, raw presentations, presentations deduplicated per
// search -- so the operator could pick one; he picked, and the picked one is now what the
// engine counts in production and what these legs read (ledger rows 1717/1722). The
// instrument, its config key and its per-request window hook are gone: a measurement kept
// running after its question is answered is a second counter of a fact production already
// owns (ADR-0012 P1), and the deduped variant it also carried has no consumer left.
//
// The sequence below is built so OBSERVATIONS and RETRIEVALS DISAGREE, because a leg in which
// they happened to coincide would witness nothing about which one is being reported:
//
//   key A: presented 4 times -- once for the request that evaluated it, three more that hit.
//     observations = 4,  retrievals = 3
//   key B: presented 3 times -- once evaluating, two that hit.
//     observations = 3,  retrievals = 2
//
// Totals: observations 7, retrievals 5.
void testCacheStatsReportsObservationsAndTheyDifferFromRetrievals() {
  RealEngineCache engine;
  const SearchableModelIdx model = AnalysisModelHosts::PRIMARY_SEARCHABLE_IDX;
  NNEvaluator& eval = *engine.hosts->searchableEval(model);
  (void)cacheAttachExecute(*engine.hosts, model, engine.attachments, attachAll(CONTEXT));
  const NNCacheContextId contextId = engine.attachments.attachmentFor(model, CONTEXT).contextId;
  const NNCacheAttribution attribution = NNCacheAttribution::toContext(contextId);
  NNCacheTable& table = eval.cacheTable();

  shared_ptr<NNOutput> got;
  // The two requests that evaluated: one presentation each, and the set that followed is not a
  // second one -- see nncacheobservations.h for why the door is per request.
  table.observe(nthKey(401), attribution);
  table.set(makeOutput(401, false), attribution);
  table.observe(nthKey(402), attribution);
  table.set(makeOutput(402, false), attribution);
  // The requests that hit.
  for(int i = 0; i < 3; i++) {
    table.observe(nthKey(401), attribution);
    testAssert(table.get(nthKey(401), got));
  }
  for(int i = 0; i < 2; i++) {
    table.observe(nthKey(402), attribution);
    testAssert(table.get(nthKey(402), got));
  }

  const NNCacheObservationLedger observations = table.harvestObservationCountsFor(contextId);
  testAssert(observations.isObserved());
  int64_t observationsA = 0, observationsB = 0;
  for(size_t i = 0; i < observations.entries().size(); i++) {
    if(observations.entries()[i].key == nthKey(401)) observationsA = observations.entries()[i].observations;
    if(observations.entries()[i].key == nthKey(402)) observationsB = observations.entries()[i].observations;
  }
  testAssert(observationsA == 4);
  testAssert(observationsB == 3);

  // The retrieval surface still exists and still answers a DIFFERENT question, which is why it
  // was kept rather than folded in: 3 and 2 against 4 and 3.
  const NNCacheHitLedger hits = table.harvestHitCounts();
  testAssert(hits.isCounted());
  int64_t retrievalsA = 0, retrievalsB = 0;
  for(size_t i = 0; i < hits.entries().size(); i++) {
    if(hits.entries()[i].key == nthKey(401)) retrievalsA = hits.entries()[i].hits;
    if(hits.entries()[i].key == nthKey(402)) retrievalsB = hits.entries()[i].hits;
  }
  testAssert(retrievalsA == 3);
  testAssert(retrievalsB == 2);

  const json stats = cacheStatsExecute(*engine.hosts, model, engine.attachments);
  testAssert(stats["observationsThisSession"].get<int64_t>() == 7);
  testAssert(stats["observedKeys"].get<int64_t>() == 2);
  testAssert(stats["unrecordedObservations"].get<int64_t>() == 0);
  // The memory bill is reported rather than buried: this feature costs tens of megabytes the
  // moment anything is attached, and the operator's whole complaint is resident size.
  testAssert(stats["observationLedgerBytes"].get<int64_t>() > 0);
  testAssert(stats["contexts"][0]["observationsThisSession"].get<int64_t>() == 7);
  testAssert(stats["contexts"][0]["observedKeys"].get<int64_t>() == 2);
  // And the retired instrument leaves nothing behind on the wire.
  testAssert(!stats.contains("admissionSignalMeasurement"));
  testAssert(!stats["contexts"][0].contains("admissionSignalMeasurement"));

  cout << "  cache_stats: observations(7) != retrievals(5), and observationLedgerBytes="
       << stats["observationLedgerBytes"].get<int64_t>() << endl;
  (void)cacheDetachExecute(*engine.hosts, model, engine.attachments, CacheDetachRequest{CONTEXT, true});
}

// THE DEBUG TRIPWIRE THAT STOOD HERE IS GONE, AND SO IS THE BYPASS IT WITNESSED.
//
// It attached a level-0 source through NNEvaluator directly, with an evaluation in flight on
// another thread, and expected a debug-build assertion to kill the process. That leg no longer
// compiles, which is the whole of the change it is recording: NNEvaluator::attachLevelZeroSource
// now takes an NNCacheLevelZeroSwapPermit, and a caller that cannot name one of that type's three
// mints cannot write the call. The bypass is not caught -- it is unwritable, and unwritable is
// what the assertion was standing in for in the one build configuration (debug) where it existed
// at all. See the note above NNEvaluator::attachLevelZeroSource's definition for why the
// assertion was removed rather than kept beside the type.
//
// WHAT STILL WITNESSES THE PROPERTY, in this file and in the compiler:
//   - testTheSwapPermitCannotBeMintedHere below, a static_assert in a translation unit that is
//     not one of the three mints: it fails the BUILD if the permit ever becomes constructible
//     from an ordinary caller (ADR-0021 Rule 2 -- a tripwire whose firing is the observation,
//     here at compile time);
//   - the request loop's own refusal while a request is open, which is always on, is not
//     compiled out of anything, and is what the permit's holders spend a permit AFTER checking.
void testTheSwapPermitCannotBeMintedHere() {
  // This translation unit is not NNCacheTable, not AnalysisCacheSwapAuthority, and does not
  // include tests/testcacheswapseam.h. So the permit's constructor is inaccessible here, and the
  // door is closed against exactly the caller shape the deleted leg above used to embody.
  static_assert(
    !std::is_default_constructible<NNCacheLevelZeroSwapPermit>::value,
    "NNCacheLevelZeroSwapPermit became constructible from an ordinary caller: the level-0 swap "
    "door is open again, and NNEvaluator::attachLevelZeroSource can be reached past the protocol "
    "layer's open-request refusal."
  );
  // Copy-construction stays available on purpose -- a permitted caller forwards its own permit
  // down a layer -- and is not a way to obtain a first one.
  static_assert(
    std::is_copy_constructible<NNCacheLevelZeroSwapPermit>::value,
    "A permitted caller can no longer forward the permit it holds."
  );
  cout << "  the level-0 swap permit is not constructible in this translation unit (compile-time)" << endl;
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
  testTheDumpAdmissionDefaultsToSeenTwiceAndIsOtherwiseClosed();
  testTheRetiredMinLookupsKeyIsRefusedAndNamesItsReplacement();
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
    testCountsAreDumpedPerContextWithTwoContextsAttached(engine);
    testDetachSeesUndumpedCountsThatTheOldProxyCouldNot(engine);
    testForeignModelSourcesResolveAgainstTheLoadedModels(engine);
    testStatsReportsWhatIsResidentAndWhatIsAttached(engine);
  }
  testTheSwapPermitCannotBeMintedHere();
  testAnEngineWithoutACacheDirectoryRefusesEveryAct();
  testCacheStatsReportsObservationsAndTheyDifferFromRetrievals();
  cout << "Done" << endl;
}
