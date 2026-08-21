#include "../tests/tests.h"

#include <chrono>
#include <functional>
#include <mutex>
#include <sstream>
#include <thread>

#include "../core/timer.h"

#include "../command/analysiscachelifecycle.h"
#include "../core/config_parser.h"
#include "../neuralnet/nncachecountlog.h"
#include "../tests/nncachetabletestaccess.h"
#include "../tests/tinymodel.h"

// THE CONFIG-DRIVEN LIFECYCLE OF THE PERSISTED CACHE: an engine that attaches its context from
// the config file at startup and dumps it at shutdown, with no client ever sending a cache verb.
//
// WHAT IS OBSERVED HERE, AND WHERE. Two halves, split by what each claim is about.
//
//   THE CONFIG BOUNDARY is a property of .cfg text alone -- which combination is refused and what
//   the operator is told -- so it is exercised against AnalysisCacheLifecycle::fromCfg directly.
//   The refusal MESSAGES are asserted on, and specifically that they NAME THE KEYS: an operator
//   who set nnCacheAttachContext and forgot nnCacheDir needs the other key's spelling, and a
//   refusal that does not carry it is one they cannot act on (ADR-0002).
//
//   THE TWO ACTS are a property of files on disk and of a live cache table, so they run against
//   real NNEvaluators -- the nets embedded in tinymodel.cpp, loaded through the same
//   Setup::initializeNNEvaluator path the engine uses -- over a real cache directory. The
//   observation point for "a session's work survived without a wire verb" is a SECOND set of
//   evaluators over the SAME directory, whose startup attach reports what the first set's
//   shutdown dump wrote (ADR-0021 Rule 1: read the property off the files, never off the return
//   value of the call that wrote them).
//
// WHAT IS NOT OBSERVED HERE. That a real engine PROCESS started from a config file does this --
// with a real stdin closing and a real exit -- is the SC5 leg of cpp/tests/e2e/
// sharedcache_e2e_witness.py. That leg needs two processes and this suite runs in one, so the
// two witnesses are deliberately different shapes over the same claim.

using namespace std;
using json = nlohmann::json;

namespace {

const char* const TMP_DIR_PREFIX = "tmpanalysiscachelifecycle";
const char* const CONTEXT = "lifecycle-card";

std::optional<string> refusalOf(const std::function<void()>& act) {
  try {
    act();
    return std::nullopt;
  }
  catch(const StringError& e) {
    return string(e.what());
  }
}

AnalysisCacheLifecycle lifecycleFrom(const string& cfgText) {
  ConfigParser cfg;
  istringstream in(cfgText);
  cfg.initialize(in);
  return AnalysisCacheLifecycle::fromCfg(cfg);
}

// TWO REAL NETS OVER A CACHE DIRECTORY THE CALLER OWNS, which is the whole reason this fixture is
// not the one in testanalysiscacheactions.cpp: that one makes its own temporary cache directory
// and so cannot be built twice over one directory. The claim under test here is precisely that a
// SECOND engine reads what a FIRST engine's shutdown wrote, so the directory has to outlive the
// evaluators, and it does: it belongs to the test function, and this fixture is constructed and
// destroyed twice over it.
class Engine {
 public:
  Engine(const string& cacheDirPath, const string& extraCfg)
    : dir(TMP_DIR_PREFIX),
      logger(nullptr, false, false, false, false),
      attachments(0)
  {
    istringstream cfgIn(
      "nnCacheSizePowerOfTwo = 12\n"
      "nnMutexPoolSizePowerOfTwo = 8\n"
      "numSearchThreads = 1\n"
      "nnCacheDir = " + cacheDirPath + "\n" + extraCfg
    );
    cfg.initialize(cfgIn);
    // Read BEFORE the models load, exactly as MainCmds::analysis reads it, so a test that
    // configured an incoherent pair sees the same refusal at the same moment an operator would.
    lifecycle = AnalysisCacheLifecycle::fromCfg(cfg);
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
    attachments = AnalysisCacheAttachments(hosts->numSearchable());
  }
  ~Engine() {
    delete primary.eval;
    delete secondary.eval;
  }
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  TestCommon::ScopedTempDir dir;
  ConfigParser cfg;
  Logger logger;
  AnalysisCacheLifecycle lifecycle = AnalysisCacheLifecycle::none();
  TinyModelTest::LoadedTinyModel primary;
  TinyModelTest::LoadedTinyModel secondary;
  unique_ptr<AnalysisModelHosts> hosts;
  AnalysisCacheAttachments attachments;
};

Hash128 nthKey(int serial) {
  return Hash128(
    ((uint64_t)(serial + 1)) * 0x517CC1B727220A95ULL,
    ((uint64_t)(serial + 1)) * 0x2545F4914F6CDD1DULL + 0x89ABCDEFULL
  );
}

shared_ptr<NNOutput> makeOutput(int serial) {
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
  for(int i = 0; i < NNPos::MAX_NN_POLICY_SIZE; i++)
    out->policyProbs[i] = 0.0f;
  out->policyProbs[0] = 1.0f;
  out->whiteOwnerMap = NULL;
  out->noisedPolicyProbs = NULL;
  return out;
}

//-------------------------------------------------------------------------------------
// The config boundary
//-------------------------------------------------------------------------------------

void testAnEngineWithNoLifecycleKeysHasNoLifecycle() {
  const AnalysisCacheLifecycle lifecycle = lifecycleFrom("numSearchThreads = 1\n");
  testAssert(!lifecycle.isConfigured());
  // The two accessors refuse rather than hand back a default that would read as configured.
  testAssert(refusalOf([&]() { (void)lifecycle.context(); }).has_value());
  testAssert(refusalOf([&]() { (void)lifecycle.dumpAdmission(); }).has_value());
  cout << "  a config with neither lifecycle key yields no lifecycle, and its accessors refuse" << endl;
}

void testTheContextKeyWithoutACacheDirectoryIsRefusedNamingBothKeys() {
  const std::optional<string> refusal = refusalOf([&]() {
    (void)lifecycleFrom("numSearchThreads = 1\nnnCacheAttachContext = somecard\n");
  });
  testAssert(refusal.has_value());
  testAssert(refusal.value().find(AnalysisCacheLifecycle::KEY_ATTACH_CONTEXT) != string::npos);
  testAssert(refusal.value().find(NNCacheConfig::KEY_DIR) != string::npos);
  cout << "  nnCacheAttachContext without nnCacheDir is refused, naming BOTH keys" << endl;
}

void testTheShutdownAdmissionKeyWithoutAContextIsRefused() {
  const std::optional<string> refusal = refusalOf([&]() {
    (void)lifecycleFrom("numSearchThreads = 1\nnnCacheDumpMinObservations = 5\n");
  });
  testAssert(refusal.has_value());
  testAssert(refusal.value().find(AnalysisCacheLifecycle::KEY_DUMP_MIN_OBSERVATIONS) != string::npos);
  testAssert(refusal.value().find(AnalysisCacheLifecycle::KEY_ATTACH_CONTEXT) != string::npos);
  cout << "  a shutdown admission with no context to dump is refused, not silently ignored" << endl;
}

// A QUOTED RUN OF SPACES, which is the ONE spelling of "empty" that reaches this layer at all:
// ConfigParser refuses a bare `key =` with no value itself, and does not trim a double-quoted
// value. So this is the reachable empty name, and it is the one the guard is written for.
void testAnEmptyContextNameIsRefused() {
  const std::optional<string> refusal = refusalOf([&]() {
    (void)lifecycleFrom("numSearchThreads = 1\nnnCacheDir = /tmp\nnnCacheAttachContext = \"   \"\n");
  });
  testAssert(refusal.has_value());
  testAssert(refusal.value().find(AnalysisCacheLifecycle::KEY_ATTACH_CONTEXT) != string::npos);
  cout << "  an all-whitespace nnCacheAttachContext is refused, naming the key" << endl;
}

// THE DEFAULT IS NOT RESTATED HERE, it is read from the one home the wire decoder reads it from.
// A test that hardcoded 2 would keep passing after someone changed the policy in one place.
void testTheShutdownAdmissionDefaultsToTheSameNumberAnAdmissionLessDumpGets() {
  const AnalysisCacheLifecycle byDefault =
    lifecycleFrom("numSearchThreads = 1\nnnCacheDir = /tmp\nnnCacheAttachContext = somecard\n");
  testAssert(byDefault.isConfigured());
  testAssert(byDefault.context() == "somecard");
  testAssert(
    byDefault.dumpAdmission().describe() ==
    NNCacheDiskAdmission::minObservations(cacheDumpDefaultAdmissionObservations()).describe()
  );

  const AnalysisCacheLifecycle overridden = lifecycleFrom(
    "numSearchThreads = 1\nnnCacheDir = /tmp\nnnCacheAttachContext = somecard\n"
    "nnCacheDumpMinObservations = 7\n"
  );
  testAssert(overridden.dumpAdmission().describe() ==
             NNCacheDiskAdmission::minObservations(7).describe());

  // Zero is the accept-everything case, and it is reachable through this one key rather than
  // through a second key that could disagree with it.
  const AnalysisCacheLifecycle acceptAll = lifecycleFrom(
    "numSearchThreads = 1\nnnCacheDir = /tmp\nnnCacheAttachContext = somecard\n"
    "nnCacheDumpMinObservations = 0\n"
  );
  testAssert(acceptAll.dumpAdmission().admits(0));
  cout << "  the shutdown admission defaults to the wire dump's own default, overrides, and "
          "reaches accept-all at 0" << endl;
}

//-------------------------------------------------------------------------------------
// The two acts
//-------------------------------------------------------------------------------------

// EVERY HOSTED MODEL, not just the primary. A leaf that hosts two nets and attaches only the
// first would serve one of them from an empty cache, and no figure anywhere would say so.
void testTheStartupAttachCoversEveryHostedModel() {
  TestCommon::ScopedTempDir cacheDir(TMP_DIR_PREFIX);
  Engine engine(cacheDir.path(), string("nnCacheAttachContext = ") + CONTEXT + "\n");
  const vector<string> lines =
    analysisCacheStartupAttach(*engine.hosts, engine.attachments, engine.lifecycle);
  testAssert(lines.size() == engine.hosts->numSearchable());
  for(const SearchableModelIdx modelIdx: engine.hosts->searchableIdxs()) {
    testAssert(engine.attachments.isAttached(modelIdx, CONTEXT));
    testAssert(engine.hosts->searchableEval(modelIdx)->numLevelZeroSources() == 1);
  }
  for(size_t i = 0; i < lines.size(); i++)
    testAssert(lines[i].find(AnalysisCacheLifecycle::KEY_ATTACH_CONTEXT) != string::npos);
  cout << "  the startup attach attaches the configured context to every hosted model" << endl;
  cout << "    " << lines[0] << endl;
}

void testAnUnconfiguredEngineAttachesAndDumpsNothing() {
  TestCommon::ScopedTempDir cacheDir(TMP_DIR_PREFIX);
  Engine engine(cacheDir.path(), "");
  testAssert(!engine.lifecycle.isConfigured());
  testAssert(analysisCacheStartupAttach(*engine.hosts, engine.attachments, engine.lifecycle).empty());
  const AnalysisCacheDumpReport report =
    analysisCacheDumpAttachedContexts(
      *engine.hosts, engine.attachments, engine.lifecycle, AnalysisCacheDumpOccasion::Shutdown, 0
    );
  testAssert(report.lines.empty() && report.modelsDumped == 0 && report.modelsFailed == 0);
  for(const SearchableModelIdx modelIdx: engine.hosts->searchableIdxs())
    testAssert(engine.hosts->searchableEval(modelIdx)->numLevelZeroSources() == 0);
  cout << "  an engine with no lifecycle configured attaches nothing and dumps nothing" << endl;
}

// A NAME THAT CANNOT BECOME A PATH COMPONENT. The alphabet rule lives in the file types; what is
// under test here is that the startup path REFUSES on it, before the engine serves anything, and
// that the refusal names the config key rather than only the container's own complaint.
void testABadContextNameRefusesTheStartupAttachNamingTheKey() {
  TestCommon::ScopedTempDir cacheDir(TMP_DIR_PREFIX);
  Engine engine(cacheDir.path(), "nnCacheAttachContext = ../escaping\n");
  const std::optional<string> refusal = refusalOf([&]() {
    (void)analysisCacheStartupAttach(*engine.hosts, engine.attachments, engine.lifecycle);
  });
  testAssert(refusal.has_value());
  testAssert(refusal.value().find(AnalysisCacheLifecycle::KEY_ATTACH_CONTEXT) != string::npos);
  cout << "  a context name outside the path alphabet refuses startup, naming nnCacheAttachContext"
       << endl;
}

// A DIRECTORY THAT IS NOT THERE, and this leg is honest about WHOSE refusal it observes: it is
// nnCacheDir's own pre-existing existence check, which fires while the models load and would fire
// with no lifecycle configured at all. It is here because an operator reaching it has set
// nnCacheAttachContext and needs to know their leaf will not come up -- not because it witnesses
// anything this file added. The refusal that IS this file's own is the one above, on the context
// name; delete analysisCacheStartupAttach's call site and THAT one goes red while this one does
// not (ADR-0021: a leg is named for the property it actually observes).
void testAMissingCacheDirectoryRefusesBeforeAnythingAttaches() {
  const std::optional<string> refusal = refusalOf([&]() {
    Engine engine("/definitely/not/a/directory/here", string("nnCacheAttachContext = ") + CONTEXT + "\n");
  });
  testAssert(refusal.has_value());
  testAssert(refusal.value().find(NNCacheConfig::KEY_DIR) != string::npos);
  cout << "  a nnCacheDir that is not an existing directory refuses startup, naming the key" << endl;
}

// THE COMPOSED WITNESS, and the one this whole file exists for: an engine attaches from its
// config, earns work, and dumps at shutdown; a SECOND engine over the same directory attaches
// from its own config and is handed that work. No cache verb is decoded anywhere in this test.
void testWorkSurvivesAShutdownDumpAndIsReadBackByALaterStartupAttach() {
  TestCommon::ScopedTempDir cacheDir(TMP_DIR_PREFIX);
  const string cfgLine = string("nnCacheAttachContext = ") + CONTEXT + "\n" +
                         "nnCacheDumpMinObservations = 0\n";
  const SearchableModelIdx model = AnalysisModelHosts::PRIMARY_SEARCHABLE_IDX;
  int64_t written = 0;

  {
    Engine first(cacheDir.path(), cfgLine);
    const vector<string> attachLines =
      analysisCacheStartupAttach(*first.hosts, first.attachments, first.lifecycle);
    testAssert(attachLines.size() == 2);

    // The session's work: three positions presented and evaluated under the attached context.
    NNEvaluator& eval = *first.hosts->searchableEval(model);
    const NNCacheContextId contextId = first.attachments.attachmentFor(model, CONTEXT).contextId;
    const NNCacheAttribution attribution = NNCacheAttribution::toContext(contextId);
    for(int serial = 1; serial <= 3; serial++) {
      (void)eval.cacheTable().present(nthKey(serial), attribution);
      eval.cacheTable().set(makeOutput(serial), attribution);
    }

    const AnalysisCacheDumpReport report =
      analysisCacheDumpAttachedContexts(
      *first.hosts, first.attachments, first.lifecycle, AnalysisCacheDumpOccasion::Shutdown, 0
    );
    testAssert(report.modelsFailed == 0);
    // Both models were attached, so both are dumped -- the second one having nothing to write is
    // still a dump that happened, and is what makes "every hosted model" true at shutdown too.
    testAssert(report.modelsDumped == 2);
    testAssert(report.entriesWritten == 3);
    written = report.entriesWritten;
    cout << "    " << report.lines[0] << endl;
  }

  // A SECOND ENGINE, over the same directory, built from its own config. Its evaluators are new
  // objects with new empty caches, so everything it reports comes off the files the first one
  // wrote (ADR-0021 Rule 1).
  {
    Engine second(cacheDir.path(), cfgLine);
    const vector<string> lines =
      analysisCacheStartupAttach(*second.hosts, second.attachments, second.lifecycle);
    testAssert(lines.size() == 2);
    NNEvaluator& eval = *second.hosts->searchableEval(model);
    testAssert(eval.numLevelZeroSources() == 1);

    // Read off the level 0 the startup attach built: every key the first engine dumped resolves
    // without this process ever having set it.
    for(int serial = 1; serial <= (int)written; serial++) {
      shared_ptr<NNOutput> fromDisk;
      testAssert(NNCacheTableTestAccess::get(eval.cacheTable(), nthKey(serial), fromDisk));
      testAssert(fromDisk != nullptr && fromDisk->nnHash == nthKey(serial));
    }
    // And the count log the first engine's counts half wrote is there and whole.
    const NNCacheCountLogContents counts = NNCacheCountLog::forContext(cacheDir.path(), CONTEXT).load();
    testAssert(counts.tail() == NNCacheCountLogTail::Intact);
    testAssert(counts.rows().size() >= (size_t)written);
    cout << "  a second engine's STARTUP ATTACH serves what the first engine's SHUTDOWN DUMP "
            "wrote, with no cache verb sent" << endl;
  }
}

// L4's in-process half: a config-attached context is an ORDINARY attached context, so a client's
// wire cache_detach takes it, and the shutdown dump then has nothing to dump for that model and
// does not overrule the client by re-attaching it.
void testAWireDetachOfTheConfigAttachedContextLeavesNothingForShutdownToDump() {
  TestCommon::ScopedTempDir cacheDir(TMP_DIR_PREFIX);
  Engine engine(cacheDir.path(), string("nnCacheAttachContext = ") + CONTEXT + "\n");
  (void)analysisCacheStartupAttach(*engine.hosts, engine.attachments, engine.lifecycle);
  const SearchableModelIdx model = AnalysisModelHosts::PRIMARY_SEARCHABLE_IDX;

  // The wire verb, on the context the config attached, exactly as a client would send it.
  const json detached =
    cacheDetachExecute(*engine.hosts, model, engine.attachments, CacheDetachRequest{CONTEXT, false});
  testAssert(detached["sourcesDetached"].get<int64_t>() == 1);
  testAssert(!engine.attachments.isAttached(model, CONTEXT));

  const AnalysisCacheDumpReport report =
    analysisCacheDumpAttachedContexts(
      *engine.hosts, engine.attachments, engine.lifecycle, AnalysisCacheDumpOccasion::Shutdown, 0
    );
  testAssert(report.modelsFailed == 0);
  // The OTHER model is still attached and is still dumped: skipping is per model, not per engine.
  testAssert(report.modelsDumped == 1);
  cout << "  a wire cache_detach takes the config-attached context, and shutdown skips exactly "
          "that model" << endl;
}

//-------------------------------------------------------------------------------------
// The periodic dump
//-------------------------------------------------------------------------------------

// THE DEFAULT IS ON, AND OFF HAS ITS OWN SPELLING. The operator's ruling is that these engines
// are normally ended by a kill, so a lifecycle whose only persistence was the shutdown dump would
// be lossy in the documented-normal case (ledger row 1879). The number is read from its one home
// rather than restated, for the same reason the admission default is.
void testTheDumpIntervalIsOnByDefaultAndOffOnlyWhenSaidSo() {
  const string base = "numSearchThreads = 1\nnnCacheDir = /tmp\nnnCacheAttachContext = somecard\n";

  const AnalysisCacheLifecycle byDefault = lifecycleFrom(base);
  testAssert(byDefault.dumpIntervalSeconds().has_value());
  testAssert(
    byDefault.dumpIntervalSeconds().value() ==
    AnalysisCacheLifecycle::defaultDumpIntervalMinutes() * 60.0
  );

  const AnalysisCacheLifecycle off = lifecycleFrom(base + "nnCacheDumpIntervalMinutes = 0\n");
  testAssert(off.isConfigured() && !off.dumpIntervalSeconds().has_value());

  const AnalysisCacheLifecycle half = lifecycleFrom(base + "nnCacheDumpIntervalMinutes = 0.5\n");
  testAssert(half.dumpIntervalSeconds().value() == 30.0);

  // A positive interval too short to be a schedule is refused rather than honored: it would be a
  // leaf that lives holding the context's exclusive lock.
  const std::optional<string> tooShort = refusalOf([&]() {
    (void)lifecycleFrom(base + "nnCacheDumpIntervalMinutes = 0.0001\n");
  });
  testAssert(tooShort.has_value());
  testAssert(tooShort.value().find(AnalysisCacheLifecycle::KEY_DUMP_INTERVAL_MINUTES) != string::npos);

  // And the interval, like the admission, is refused with no context to dump.
  const std::optional<string> noContext = refusalOf([&]() {
    (void)lifecycleFrom("numSearchThreads = 1\nnnCacheDumpIntervalMinutes = 5\n");
  });
  testAssert(noContext.has_value());
  testAssert(noContext.value().find(AnalysisCacheLifecycle::KEY_DUMP_INTERVAL_MINUTES) != string::npos);
  testAssert(noContext.value().find(AnalysisCacheLifecycle::KEY_ATTACH_CONTEXT) != string::npos);
  cout << "  the dump interval defaults ON (" << AnalysisCacheLifecycle::defaultDumpIntervalMinutes()
       << " min), is turned off only by an explicit 0, and refuses a schedule with nothing to dump"
       << endl;
}

// A dumper over `engine`, with the mutex the engine would own. Held in one place so the two legs
// below construct it the same way the analysis command does.
struct DumperFixture {
  explicit DumperFixture(Engine& engine)
    : dumper(*engine.hosts, engine.attachments, mutex, engine.lifecycle,
             []() { return (int64_t)0; },
             [this](const string& line) {
               std::lock_guard<std::mutex> lock(linesMutex);
               lines.push_back(line);
             })
  {}
  std::mutex mutex;
  std::mutex linesMutex;
  vector<string> lines;
  AnalysisCachePeriodicDumper dumper;

  [[nodiscard]] vector<string> linesSoFar() {
    std::lock_guard<std::mutex> lock(linesMutex);
    return lines;
  }
};

// THE CLAIM THE WHOLE FEATURE RESTS ON: a leaf that is never asked for anything and never exits
// cleanly still gets its work onto disk. So this leg never calls the shutdown dump at all -- it
// starts the dumper, waits for a pass, stops it, and then reads the files with a SECOND set of
// evaluators. A shutdown dump anywhere in this function would make it unable to tell the interval
// dump from the exit dump.
void testThePeriodicDumperWritesWithNobodyAskingAndNoCleanExit() {
  TestCommon::ScopedTempDir cacheDir(TMP_DIR_PREFIX);
  const string cfgLines = string("nnCacheAttachContext = ") + CONTEXT + "\n" +
                          "nnCacheDumpMinObservations = 0\n" +
                          "nnCacheDumpIntervalMinutes = 0.002\n";  // 120 ms
  const SearchableModelIdx model = AnalysisModelHosts::PRIMARY_SEARCHABLE_IDX;
  int64_t writtenByInterval = 0;

  {
    Engine engine(cacheDir.path(), cfgLines);
    (void)analysisCacheStartupAttach(*engine.hosts, engine.attachments, engine.lifecycle);
    NNEvaluator& eval = *engine.hosts->searchableEval(model);
    const NNCacheContextId contextId = engine.attachments.attachmentFor(model, CONTEXT).contextId;
    const NNCacheAttribution attribution = NNCacheAttribution::toContext(contextId);
    for(int serial = 1; serial <= 3; serial++) {
      (void)eval.cacheTable().present(nthKey(serial), attribution);
      eval.cacheTable().set(makeOutput(serial), attribution);
    }

    DumperFixture fixture(engine);
    fixture.dumper.start();
    // WAIT ON THE PROPERTY, WITH A DEADLINE -- never a fixed sleep sized to "probably enough".
    // A fixed sleep is a leg that goes green on a fast machine and red on a loaded one, and says
    // nothing either way (ADR-0021).
    const double deadline = 60.0;
    const ClockTimer timer;
    while(fixture.dumper.entriesWritten() < 3 && timer.getSeconds() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    // Stopping is part of the claim: it must return, and it must not lose the pass in flight.
    fixture.dumper.stop();

    testAssert(fixture.dumper.passesPerformed() >= 1);
    testAssert(fixture.dumper.passesWithAFailure() == 0);
    writtenByInterval = fixture.dumper.entriesWritten();
    testAssert(writtenByInterval == 3);
    const vector<string> lines = fixture.linesSoFar();
    testAssert(!lines.empty());
    // The occasion is IN the line, so a log cannot be read as an exit dump when it was an
    // interval one.
    testAssert(lines[0].find("on the dump interval") != string::npos);
    cout << "    " << lines[0].substr(0, 150) << "..." << endl;
    // stop() is idempotent and safe to repeat -- the destructor calls it again.
    fixture.dumper.stop();
  }

  // The witness, off the files, with new evaluators: the interval dump alone put them there.
  {
    Engine second(cacheDir.path(), cfgLines);
    (void)analysisCacheStartupAttach(*second.hosts, second.attachments, second.lifecycle);
    NNEvaluator& eval = *second.hosts->searchableEval(model);
    for(int serial = 1; serial <= (int)writtenByInterval; serial++) {
      shared_ptr<NNOutput> fromDisk;
      testAssert(NNCacheTableTestAccess::get(eval.cacheTable(), nthKey(serial), fromDisk));
    }
    cout << "  the PERIODIC dump alone put " << writtenByInterval
         << " entries on disk -- no wire verb, no clean exit, and a later engine reads them back"
         << endl;
  }
}

// THE CONFIGURED ADMISSION MUST ACTUALLY GOVERN THE DUMP, and until this leg existed nothing
// checked it: every other leg either configured minObservations(0) -- which is behaviourally
// identical to all() -- or asserted on modelsDumped without looking at what was written. A
// regression that ignored the config and hardcoded all() passed the whole suite.
//
// So this one runs at the DEFAULT admission, which is minObservations(2), over a session whose
// keys are deliberately split: one key presented twice, two presented once. The dump must write
// exactly the one and refuse exactly the two.
void testTheConfiguredAdmissionGovernsWhatTheDumpWrites() {
  TestCommon::ScopedTempDir cacheDir(TMP_DIR_PREFIX);
  // No nnCacheDumpMinObservations: the default is the point.
  Engine engine(
    cacheDir.path(),
    string("nnCacheAttachContext = ") + CONTEXT + "\nnnCacheDumpIntervalMinutes = 0\n"
  );
  testAssert(
    engine.lifecycle.dumpAdmission().describe() ==
    NNCacheDiskAdmission::minObservations(cacheDumpDefaultAdmissionObservations()).describe()
  );
  (void)analysisCacheStartupAttach(*engine.hosts, engine.attachments, engine.lifecycle);

  const SearchableModelIdx model = AnalysisModelHosts::PRIMARY_SEARCHABLE_IDX;
  NNEvaluator& eval = *engine.hosts->searchableEval(model);
  const NNCacheContextId contextId = engine.attachments.attachmentFor(model, CONTEXT).contextId;
  const NNCacheAttribution attribution = NNCacheAttribution::toContext(contextId);
  for(int serial = 1; serial <= 3; serial++) {
    (void)eval.cacheTable().present(nthKey(serial), attribution);
    eval.cacheTable().set(makeOutput(serial), attribution);
  }
  // Key 1 asked for a second time: now observed twice, and the only one that clears the default.
  shared_ptr<NNOutput> got;
  (void)eval.cacheTable().present(nthKey(1), attribution);
  testAssert(NNCacheTableTestAccess::get(eval.cacheTable(), nthKey(1), got));

  const AnalysisCacheDumpReport report = analysisCacheDumpAttachedContexts(
    *engine.hosts, engine.attachments, engine.lifecycle, AnalysisCacheDumpOccasion::Shutdown, 0
  );
  testAssert(report.modelsFailed == 0);
  // THE NUMBER IS 1, NOT "SOME". A hardcoded all() would make it 3 and this assert would fail.
  testAssert(report.entriesWritten == 1);

  // And the two refused really are refused, off the FILE and not off the report: a later engine
  // attaching this context finds exactly the one admitted key.
  {
    Engine second(cacheDir.path(), string("nnCacheAttachContext = ") + CONTEXT + "\n");
    (void)analysisCacheStartupAttach(*second.hosts, second.attachments, second.lifecycle);
    NNEvaluator& secondEval = *second.hosts->searchableEval(model);
    shared_ptr<NNOutput> admitted;
    testAssert(NNCacheTableTestAccess::get(secondEval.cacheTable(), nthKey(1), admitted));
    shared_ptr<NNOutput> refused;
    testAssert(!NNCacheTableTestAccess::get(secondEval.cacheTable(), nthKey(2), refused));
    testAssert(!NNCacheTableTestAccess::get(secondEval.cacheTable(), nthKey(3), refused));
  }
  cout << "  the DEFAULT admission really governs: of 3 earned keys the dump wrote the 1 seen "
          "twice, and a later engine finds only that one on disk" << endl;
}

// THE TERMINATION-SIGNAL DUMP, exercised as the act it is. This process cannot send itself SIGTERM
// without killing the test run, so what is checked here is that the act dumps and labels itself;
// that a real engine PROCESS does it when really signalled is SC7 in the e2e witness.
void testTheTerminationSignalDumpWritesAndNamesItsOccasion() {
  TestCommon::ScopedTempDir cacheDir(TMP_DIR_PREFIX);
  Engine engine(
    cacheDir.path(),
    string("nnCacheAttachContext = ") + CONTEXT +
    "\nnnCacheDumpMinObservations = 0\nnnCacheDumpIntervalMinutes = 0\n"
  );
  (void)analysisCacheStartupAttach(*engine.hosts, engine.attachments, engine.lifecycle);
  const SearchableModelIdx model = AnalysisModelHosts::PRIMARY_SEARCHABLE_IDX;
  NNEvaluator& eval = *engine.hosts->searchableEval(model);
  const NNCacheContextId contextId = engine.attachments.attachmentFor(model, CONTEXT).contextId;
  const NNCacheAttribution attribution = NNCacheAttribution::toContext(contextId);
  for(int serial = 1; serial <= 2; serial++) {
    (void)eval.cacheTable().present(nthKey(serial), attribution);
    eval.cacheTable().set(makeOutput(serial), attribution);
  }

  std::mutex cacheActionMutex;
  const AnalysisCacheDumpReport report = analysisCacheDumpForTerminationSignal(
    *engine.hosts, engine.attachments, cacheActionMutex, engine.lifecycle, 0
  );
  testAssert(report.modelsFailed == 0 && report.entriesWritten == 2);
  testAssert(report.lines[0].find("on a termination signal") != string::npos);
  cout << "  the termination-signal dump writes, and its log line says which occasion it was"
       << endl;
}

// An interval of 0 is not "a thread that wakes up and does nothing"; there is no thread.
void testAnIntervalOfZeroRunsNoPassesAtAll() {
  TestCommon::ScopedTempDir cacheDir(TMP_DIR_PREFIX);
  Engine engine(
    cacheDir.path(),
    string("nnCacheAttachContext = ") + CONTEXT + "\nnnCacheDumpIntervalMinutes = 0\n"
  );
  (void)analysisCacheStartupAttach(*engine.hosts, engine.attachments, engine.lifecycle);
  DumperFixture fixture(engine);
  fixture.dumper.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  fixture.dumper.stop();
  testAssert(fixture.dumper.passesPerformed() == 0);
  testAssert(fixture.linesSoFar().empty());
  // And an unconfigured engine likewise starts nothing, without the caller having to ask.
  Engine none(cacheDir.path(), "");
  DumperFixture unconfigured(none);
  unconfigured.dumper.start();
  unconfigured.dumper.stop();
  testAssert(unconfigured.dumper.passesPerformed() == 0);
  cout << "  an interval of 0, and an unconfigured engine, start no dumping thread at all" << endl;
}

}  // namespace

void Tests::runAnalysisCacheLifecycleTests() {
  cout << "Running analysis engine cache lifecycle tests" << endl;
  testAnEngineWithNoLifecycleKeysHasNoLifecycle();
  testTheContextKeyWithoutACacheDirectoryIsRefusedNamingBothKeys();
  testTheShutdownAdmissionKeyWithoutAContextIsRefused();
  testAnEmptyContextNameIsRefused();
  testTheShutdownAdmissionDefaultsToTheSameNumberAnAdmissionLessDumpGets();
  testTheStartupAttachCoversEveryHostedModel();
  testAnUnconfiguredEngineAttachesAndDumpsNothing();
  testABadContextNameRefusesTheStartupAttachNamingTheKey();
  testAMissingCacheDirectoryRefusesBeforeAnythingAttaches();
  testWorkSurvivesAShutdownDumpAndIsReadBackByALaterStartupAttach();
  testAWireDetachOfTheConfigAttachedContextLeavesNothingForShutdownToDump();
  testTheDumpIntervalIsOnByDefaultAndOffOnlyWhenSaidSo();
  testThePeriodicDumperWritesWithNobodyAskingAndNoCleanExit();
  testAnIntervalOfZeroRunsNoPassesAtAll();
  testTheConfiguredAdmissionGovernsWhatTheDumpWrites();
  testTheTerminationSignalDumpWritesAndNamesItsOccasion();
  cout << "Done" << endl;
}
