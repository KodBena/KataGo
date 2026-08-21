#include "../core/global.h"
#include "../core/config_parser.h"
#include "../core/timer.h"
#include "../core/datetime.h"
#include "../core/makedir.h"
#include "../search/asyncbot.h"
#include "../search/patternbonustable.h"
#include "../program/setup.h"
#include "../program/playutils.h"
#include "../program/play.h"
#include "../command/commandline.h"
#include "../command/analysiscacheactions.h"
#include "../command/analysismodels.h"
#include "../core/test.h"
#include "../main.h"

#include <optional>

#include "../external/nlohmann_json/json.hpp"

using namespace std;
using json = nlohmann::json;

struct AnalyzeRequest {
  int64_t internalId;
  string id;
  int turnNumber;
  int64_t priority;

  //Which hosted model analyzes this request, already resolved from the request's optional "model"
  //field against the loaded name space. The request carries the resolved index and never the
  //requested name: resolution happens once, at the boundary that can refuse it, so no later
  //stage can resolve it differently or fail to resolve it at all.
  //A request that named no model carries AnalysisModelHosts::PRIMARY_SEARCHABLE_IDX, which is
  //also what a freshly-built request carries before its "model" field has been read.
  SearchableModelIdx modelIdx = AnalysisModelHosts::PRIMARY_SEARCHABLE_IDX;

  //Which of the selected model's attached cache contexts this request's new evaluations are
  //earned by, already resolved from the request's optional "cacheContext" field against that
  //model's own attached contexts. The request carries the RESOLUTION and never the requested
  //name, for the same reason it carries a resolved model index: resolution happens once, at the
  //boundary that can refuse it, so no later stage can resolve it differently or fail to.
  //A request that named no context carries NoAttributableContext unless exactly one context is
  //attached, which is also what a freshly-built request carries.
  NNCacheAttribution cacheAttribution;

  Board board;
  BoardHistory hist;
  Player nextPla;

  SearchParams params;
  Player perspective;
  int analysisPVLen;
  bool includeOwnership;
  bool includeOwnershipStdev;
  bool includeMovesOwnership;
  bool includeMovesOwnershipStdev;
  bool includePolicy;
  bool includePVVisits;
  bool includeNoResultValue;

  bool reportDuringSearch;
  double reportDuringSearchEvery;
  double firstReportDuringSearchAfter;

  vector<int> avoidMoveUntilByLocBlack;
  vector<int> avoidMoveUntilByLocWhite;

  //Starts with STATUS_IN_QUEUE.
  //Thread that grabs it from queue it changes it to STATUS_POPPED
  //Once search is fully started thread sticks in its own thread index
  //At any point it may change to STATUS_TERMINATED.
  //If it ever gets to STATUS_POPPED or later, then the analysis thread is reponsible for writing the result, else the api thread is
  static constexpr int STATUS_IN_QUEUE = -1;
  static constexpr int STATUS_POPPED = -2;
  static constexpr int STATUS_TERMINATED = -3;
  std::atomic<int> status;
};

//The bots one analysis thread owns, one per hosted searchable model.
//
//This is a type rather than a plain vector<AsyncBot*> for one reason. The engine stores bots on
//two axes -- analysis thread, then model -- and as two bare size_t subscripts both
//bots[threadIdx][modelIdx] and bots[modelIdx][threadIdx] compile; whenever the two counts happen
//to be equal, which two analysis threads hosting two models makes an ordinary shape, the
//transposed one is in bounds too and quietly serves requests from the wrong model's bot. This
//type is subscriptable only by a SearchableModelIdx, so the transposition is a compile error.
class BotsByModel {
 public:
  void add(AsyncBot* bot) { bots.push_back(bot); }
  [[nodiscard]] AsyncBot* at(SearchableModelIdx idx) const {
    //The second and last place a SearchableModelIdx is unwrapped: this is the storage it indexes.
    testAssert(idx.value() < bots.size());
    return bots[idx.value()];
  }
  //For the operations that are per-bot rather than per-model -- stopping, killing, deleting them
  //all -- where no index is involved and so none can be confused.
  [[nodiscard]] const std::vector<AsyncBot*>& all() const { return bots; }

 private:
  std::vector<AsyncBot*> bots;
};


//RENDERING a request's response and EMITTING it are separate acts, and every FINAL response goes
//through the render half first. The reason is ordering, not tidiness: a request must be gone from
//openRequests BEFORE its final response reaches the write queue. The write queue is FIFO and one
//thread drains it, so a response the client has read is a response that was pushed; if the erase
//happened before the push, it happened before anything the client could observe of this request.
//A client that waits for its query's answer and then sends cache_attach or cache_detach therefore
//reads an open-request count that no longer counts the query it just watched finish. Emitting
//first and erasing after -- which is what this code used to do -- leaves exactly that window open,
//and the refusal those two actions make on a nonzero count fires inside it. Splitting render from
//emit is what lets the erase sit between the two.
//
//These are file-scope functions rather than lambdas inside the command so that [[nodiscard]] is
//ENFORCED. In C++17 -- this project's standard (CMakeLists.txt CMAKE_CXX_STANDARD 17) -- the
//attribute cannot appertain to a lambda's call operator: GCC 15.2 answers a lambda-borne
//[[nodiscard]] with "attribute can only be applied to functions or to class or enumeration types
//[-Wattributes]" and drops it, so writing it there would decorate rather than check. Dropping any
//of these returns silently loses a client's response, which is exactly what the attribute is for
//(ADR-0012 P9 rule 5).

//The response for a request we don't actually have results for. This is used when something is
//user-terminated before being actually analyzed properly. Only used outside of search too.
[[nodiscard]] static string renderNoAnalysis(const AnalyzeRequest* request) {
  json ret;
  ret["id"] = request->id;
  ret["turnNumber"] = request->turnNumber;
  ret["isDuringSearch"] = false;
  ret["noResults"] = true;
  return ret.dump();
}

//Returns nothing if no analysis was reportable due to there being no root node or search results.
[[nodiscard]] static std::optional<string> renderAnalysis(
  const AnalyzeRequest* request, const Search* search, bool isDuringSearch, bool preventEncore
) {
  json ret;
  ret["id"] = request->id;
  ret["turnNumber"] = request->turnNumber;
  ret["isDuringSearch"] = isDuringSearch;

  bool success = search->getAnalysisJson(
    request->perspective,
    request->analysisPVLen, preventEncore, request->includePolicy,
    request->includeOwnership,request->includeOwnershipStdev,
    request->includeMovesOwnership,request->includeMovesOwnershipStdev,
    request->includePVVisits,
    request->includeNoResultValue,
    ret
  );

  if(!success)
    return std::nullopt;
  return ret.dump();
}

//Terminates `request` and returns its final response if terminating it is what closed the request
//out -- that is, if no analysis thread had claimed it, so none will ever write one. The CALLER
//emits that response, and only after erasing the request from openRequests, for the same reason
//the analysis loop does: a response the client can see must not name a request the engine still
//counts as open. Returns nothing when some analysis thread owns the response instead.
[[nodiscard]] static std::optional<string> terminateRequest(
  vector<BotsByModel>& bots, AnalyzeRequest* request
) {
  //Firstly, flag the request as terminated
  int prevStatus = request->status.exchange(AnalyzeRequest::STATUS_TERMINATED,std::memory_order_acq_rel);
  //Already terminated? Nothing to do.
  if(prevStatus == AnalyzeRequest::STATUS_TERMINATED)
  {}
  //No thread claimed it, so it's up to us to write the result
  else if(prevStatus == AnalyzeRequest::STATUS_IN_QUEUE) {
    return renderNoAnalysis(request);
  }
  //A thread popped it. That thread will notice that it's terminated once it tries to put its thread idx in, so we need not do anything.
  else if(prevStatus == AnalyzeRequest::STATUS_POPPED)
  {}
  //A thread started searching it and put its thread idx in
  else {
    testAssert(prevStatus >= 0);
    //We've already set the above status to terminated so when the thread terminates due to our killing it below, it will see this.
    //Or else the thread has already done so, in which case it's already properly written a result, also fine.
    int threadIdx = prevStatus;
    //Terminate it by thread index, and within that thread by the model the request resolved to:
    //that is the one bot of that thread's pool the request can be running on.
    bots[threadIdx].at(request->modelIdx)->stopWithoutWait();
  }
  return std::nullopt;
}

//Terminates each of the given requests and returns the responses the CALLER must emit, in order,
//once it has released openRequestsMutex.
//MUST BE CALLED WITH openRequestsMutex HELD, and the returned responses MUST NOT be emitted until
//it is released. Two obligations, one reason each:
// - held, because a request this call closes is erased from openRequests here, and that erase has
//   to be in the same critical section that found it;
// - rendered before release, because the moment the lock is dropped, the analysis thread that
//   pops a terminated request is free to erase and DELETE it, so the request is no longer there
//   to read from. The responses leave here as strings, already rendered, for exactly that reason.
//A request closed here will never be searched -- the analysis thread that pops it sees
//STATUS_TERMINATED and skips straight past the search -- so it touches no cache from this point
//on, and dropping it from the open count is truthful rather than merely convenient. Dropping it
//is also required: its response is about to be emitted, and a client that reads a response for a
//request the engine still counts as open is the very thing this ordering exists to prevent.
[[nodiscard]] static vector<string> closeTerminated(
  vector<BotsByModel>& bots,
  std::map<int64_t, AnalyzeRequest*>& openRequests,
  const vector<AnalyzeRequest*>& requests
) {
  vector<string> closedResponses;
  for(AnalyzeRequest* request: requests) {
    std::optional<string> response = terminateRequest(bots, request);
    if(response.has_value()) {
      openRequests.erase(request->internalId);
      closedResponses.push_back(std::move(response.value()));
    }
  }
  return closedResponses;
}


int MainCmds::analysis(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;

  ConfigParser cfg;
  string modelFile;
  vector<string> extraModelFiles;
  string humanModelFile;
  bool numAnalysisThreadsCmdlineSpecified;
  int numAnalysisThreadsCmdline;
  bool quitWithoutWaiting;

  KataGoCommandLine cmd("Run KataGo parallel JSON-based analysis engine.");
  try {
    cmd.addConfigFileArg("","analysis_example.cfg");
    cmd.addModelFileArg();
    cmd.addHumanModelFileArg();
    cmd.setShortUsageArgLimit();
    cmd.addOverrideConfigArg();

    TCLAP::ValueArg<int> numAnalysisThreadsArg("","analysis-threads","Analyze up to this many positions in parallel. Equivalent to numAnalysisThreads in the config.",false,0,"THREADS");
    TCLAP::SwitchArg quitWithoutWaitingArg("","quit-without-waiting","When stdin is closed, quit quickly without waiting for queued tasks");
    TCLAP::MultiArg<string> extraModelArg("","extra-model","Also host this model, selectable by its internalName via the \"model\" field of a request. May be given more than once.",false,"FILE");
    cmd.add(numAnalysisThreadsArg);
    cmd.add(quitWithoutWaitingArg);
    cmd.add(extraModelArg);
    cmd.parseArgs(args);

    modelFile = cmd.getModelFile();
    extraModelFiles = extraModelArg.getValue();
    humanModelFile = cmd.getHumanModelFile();
    numAnalysisThreadsCmdlineSpecified = numAnalysisThreadsArg.isSet();
    numAnalysisThreadsCmdline = numAnalysisThreadsArg.getValue();
    quitWithoutWaiting = quitWithoutWaitingArg.getValue();

    cmd.getConfig(cfg);
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }
  cfg.applyAlias("numSearchThreadsPerAnalysisThread", "numSearchThreads");

  if(cfg.contains("numAnalysisThreads") && numAnalysisThreadsCmdlineSpecified)
    throw StringError("When specifying numAnalysisThreads in the config (" + cfg.getFileName() + "), it is redundant and disallowed to also specify it via -analysis-threads");

  const int numAnalysisThreads = numAnalysisThreadsCmdlineSpecified ? numAnalysisThreadsCmdline : cfg.getInt("numAnalysisThreads",1,16384);
  if(numAnalysisThreads <= 0 || numAnalysisThreads > 16384)
    throw StringError("Invalid value for numAnalysisThreads: " + Global::intToString(numAnalysisThreads));

  const bool forDeterministicTesting =
    cfg.contains("forDeterministicTesting") ? cfg.getBool("forDeterministicTesting") : false;
  if(forDeterministicTesting)
    seedRand.init("forDeterministicTesting");

  const bool logToStdoutDefault = false;
  const bool logToStderrDefault = true;
  Logger logger(&cfg, logToStdoutDefault, logToStderrDefault);
  const bool logToStderr = logger.isLoggingToStderr();

  logger.write("Analysis Engine starting...");
  logger.write(Version::getKataGoVersionForHelp());
  if(!logToStderr) {
    cerr << Version::getKataGoVersionForHelp() << endl;
  }

  const bool logAllRequests = cfg.contains("logAllRequests") ? cfg.getBool("logAllRequests") : false;
  const bool logAllResponses = cfg.contains("logAllResponses") ? cfg.getBool("logAllResponses") : false;
  const bool logErrorsAndWarnings = cfg.contains("logErrorsAndWarnings") ? cfg.getBool("logErrorsAndWarnings") : true;
  const bool logSearchInfo = cfg.contains("logSearchInfo") ? cfg.getBool("logSearchInfo") : false;

  const bool warnUnusedFields = cfg.contains("warnUnusedFields") ? cfg.getBool("warnUnusedFields") : true;

  auto loadParams = [&humanModelFile](ConfigParser& config, SearchParams& params, Player& perspective, Player defaultPerspective) {
    bool hasHumanModel = humanModelFile != "";
    params = Setup::loadSingleParams(config,Setup::SETUP_FOR_ANALYSIS,hasHumanModel);
    perspective = Setup::parseReportAnalysisWinrates(config,defaultPerspective);
    //Set a default for conservativePass that differs from matches or selfplay
    if(!config.contains("conservativePass"))
      params.conservativePass = true;
  };

  SearchParams defaultParams;
  Player defaultPerspective;
  loadParams(cfg, defaultParams, defaultPerspective, C_EMPTY);

  std::unique_ptr<PatternBonusTable> patternBonusTable = nullptr;
  {
    std::vector<std::unique_ptr<PatternBonusTable>> tables = Setup::loadAvoidSgfPatternBonusTables(cfg,logger);
    testAssert(tables.size() == 1);
    patternBonusTable = std::move(tables[0]);
  }

  const int analysisPVLen = cfg.contains("analysisPVLen") ? cfg.getInt("analysisPVLen",1,100) : 15;
  const bool assumeMultipleStartingBlackMovesAreHandicap =
    cfg.contains("assumeMultipleStartingBlackMovesAreHandicap") ? cfg.getBool("assumeMultipleStartingBlackMovesAreHandicap") : true;
  const bool preventEncore = cfg.contains("preventCleanupPhase") ? cfg.getBool("preventCleanupPhase") : true;

  //Every searchable model the process hosts, primary first. Without -extra-model this is exactly
  //the one model the engine has always loaded.
  //There is deliberately no variable naming "the primary evaluator" that outlives this block: the
  //only handle on a model afterwards is modelHosts, indexed by the model a request resolved to, so
  //a later reader cannot reach for the primary net by name without meaning to.
  vector<HostedModel> searchableModels;
  NNEvaluator* humanEval = NULL;
  {
    Setup::initializeSession(cfg);
    const int expectedConcurrentEvals = numAnalysisThreads * defaultParams.numThreads;
    const bool defaultRequireExactNNLen = false;
    const int defaultMaxBatchSize = -1;
    const bool disableFP16 = false;
    const string expectedSha256 = "";
    //Loads one searchable model and admits it to the name space, refusing at once if its internal
    //name is one an already-loaded model answers to. Same rule AnalysisModelHosts::create enforces,
    //applied as each model arrives, so a process that is going to be refused for a duplicate does
    //not first pay to load every remaining net onto the device.
    auto loadSearchableModel = [&](const string& file, const string& argName) {
      NNEvaluator* eval = Setup::initializeNNEvaluator(
        file,file,expectedSha256,cfg,logger,seedRand,expectedConcurrentEvals,
        NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,defaultMaxBatchSize,defaultRequireExactNNLen,disableFP16,
        Setup::SETUP_FOR_ANALYSIS
      );
      searchableModels.push_back(
        HostedModel{ModelAddress{eval->getInternalModelName(), argName + " " + eval->getModelFileName(), ModelRole::Searchable}, eval}
      );
      const std::optional<string> collision = findInternalNameCollision(addressesOf(searchableModels));
      if(collision.has_value())
        throw StringError(collision.value());
    };
    loadSearchableModel(modelFile, "-model");
    for(const string& extraModelFile: extraModelFiles)
      loadSearchableModel(extraModelFile, "-extra-model");
    if(humanModelFile != "") {
      humanEval = Setup::initializeNNEvaluator(
        humanModelFile,humanModelFile,expectedSha256,cfg,logger,seedRand,expectedConcurrentEvals,
        NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,defaultMaxBatchSize,defaultRequireExactNNLen,disableFP16,
        Setup::SETUP_FOR_ANALYSIS
      );
      if(!humanEval->requiresSGFMetadata()) {
        string warning;
        warning += "WARNING: Human model was not trained from SGF metadata to vary by rank! Did you pass the wrong model for -human-model?\n";
        logger.write(warning);
        if(!logToStderr)
          cerr << warning << endl;
      }
    }
  }

  //The name space of the loaded models. The duplicate-name refusal above fires as each model is
  //admitted; create() enforces the same rule for the whole set, including the human companion,
  //which shares the name space because a client reads its name from query_models too.
  const AnalysisModelHosts modelHosts = [&]() {
    std::optional<HostedModel> companion;
    if(humanEval != NULL) {
      companion = HostedModel{
        ModelAddress{humanEval->getInternalModelName(), "-human-model " + humanEval->getModelFileName(), ModelRole::HumanCompanion},
        humanEval
      };
    }
    return AnalysisModelHosts::create(std::move(searchableModels), std::move(companion));
  }();

#ifndef USE_EIGEN_BACKEND
  for(const SearchableModelIdx modelIdx: modelHosts.searchableIdxs()) {
    NNEvaluator* eval = modelHosts.searchableEval(modelIdx);
    int nnMaxBatchSizeTotal = eval->getNumGpus() * eval->getMaxBatchSize();
    int numThreadsTotal = defaultParams.numThreads * numAnalysisThreads;
    if(nnMaxBatchSizeTotal * 1.5 <= numThreadsTotal) {
      logger.write(
        Global::strprintf(
          "Note: nnMaxBatchSize * number of GPUs (%d) is smaller than numSearchThreads * numAnalysisThreads (%d)",
          nnMaxBatchSizeTotal, numThreadsTotal
        )
      );
      logger.write("The number of simultaneous threads that might query the GPU could be larger than the batch size that the GPU will handle at once.");
      logger.write("It may improve performance to increase nnMaxBatchSize, unless you are constrained on GPU memory.");
    }
  }
#endif

  //Check for unused config keys
  cfg.warnUnusedKeys(cerr,&logger);
  //Per hosted model: the default params may be honorable by one net and not another, and a warning
  //that only ever consulted the primary would be silent about exactly the model a request names.
  for(const SearchableModelIdx modelIdx: modelHosts.searchableIdxs())
    Setup::maybeWarnHumanSLParams(defaultParams,modelHosts.searchableEval(modelIdx),humanEval,cerr,&logger);

  logger.write("Loaded config "+ cfg.getFileName());
  logger.write("Loaded model "+ modelFile);
  cmd.logOverrides(logger);

  if(humanModelFile != "" && !cfg.contains("humanSLProfile") && humanEval->requiresSGFMetadata()) {
    logger.write("Warning: Provided -human-model but humanSLProfile is not yet set. The human SL model will only be used on queries that provide humanSLProfile in overrideSettings.");
    if(!logger.isLoggingToStderr())
      cerr << "Warning: Provided -human-model but humanSLProfile is not yet set. The human SL model will only be used on queries that provide humanSLProfile in overrideSettings." << endl;
  }

  //Expected possible keys for queries
  const std::set<std::string> expectedKeys = {
    "id",
    "action",
    "model",
    "cacheContext",
    "terminateId",
    "turnNumbers",
    "boardXSize",
    "boardYSize",
    "initialStones",
    "moves",
    "initialPlayer",
    "analyzeTurns",
    "priorities",
    "rules",
    "komi",
    "whiteHandicapBonus",
    "overrideSettings",
    "maxVisits",
    "analysisPVLen",
    "rootFpuReductionMax",
    "rootPolicyTemperature",
    "includeMovesOwnership",
    "includeMovesOwnershipStdev",
    "includeOwnership",
    "includeOwnershipStdev",
    "includePolicy",
    "includePVVisits",
    "includeNoResultValue",
    "reportDuringSearchEvery",
    "firstReportDuringSearchAfter",
    "priority",
    "allowMoves",
    "avoidMoves"
  };

  ThreadSafeQueue<string*> toWriteQueue;
  auto writeLoop = [&toWriteQueue,&logAllResponses,&logger]() {
    while(true) {
      string* message;
      bool suc = toWriteQueue.waitPop(message);
      if(!suc)
        break;
      cout << *message << endl;
      if(logAllResponses)
        logger.write("Response: " + *message);
      delete message;
    }
  };

  auto pushToWrite = [&toWriteQueue](string* s) {
    bool suc = toWriteQueue.forcePush(s);
    if(!suc)
      delete s;
  };

  ThreadSafePriorityQueue<std::pair<int64_t,int64_t>, AnalyzeRequest*> toAnalyzeQueue;
  int64_t numRequestsSoFar = 0; // Used as tie breaker for requests with same priority
  int64_t internalIdCounter = 0; // Counter for internalId on requests.

  //Open requests, keyed by internalId, mutexed by the mutex
  std::mutex openRequestsMutex;
  std::map<int64_t, AnalyzeRequest*> openRequests;

  //What the cache actions have attached to each hosted model. Read and written ONLY on the request
  //loop below, which is a single thread, so it takes no lock; see AnalysisCacheAttachments.
  AnalysisCacheAttachments cacheAttachments(modelHosts.numSearchable());

  auto reportError = [&pushToWrite,&logger,&logErrorsAndWarnings](const string& s) {
    json ret;
    ret["error"] = s;
    pushToWrite(new string(ret.dump()));
    if(logErrorsAndWarnings)
      logger.write("Error: " + ret.dump());
  };
  auto reportErrorForId = [&pushToWrite,&logger,&logErrorsAndWarnings](const string& id, const string& field, const string& s) {
    json ret;
    ret["id"] = id;
    ret["field"] = field;
    ret["error"] = s;
    pushToWrite(new string(ret.dump()));
    if(logErrorsAndWarnings)
      logger.write("Error: " + ret.dump());
  };
  auto reportWarningForId = [&pushToWrite,&logger,&logErrorsAndWarnings](const string& id, const string& field, const string& s) {
    json ret;
    ret["id"] = id;
    ret["field"] = field;
    ret["warning"] = s;
    pushToWrite(new string(ret.dump()));
    if(logErrorsAndWarnings)
      logger.write("Warning: " + ret.dump());
  };

  //Renders and emits in one act. The ONLY emit-where-you-render caller is the during-search
  //report, which needs no ordering against the erase: the request genuinely IS open while a
  //during-search report is written, so a count of 1 read after one is exactly truthful. Every
  //FINAL response instead goes through renderAnalysis/renderNoAnalysis at file scope, so the
  //erase can be sequenced before the emit -- see the comment on those.
  auto reportAnalysis = [&pushToWrite,&preventEncore](const AnalyzeRequest* request, const Search* search, bool isDuringSearch) {
    std::optional<string> rendered = renderAnalysis(request,search,isDuringSearch,preventEncore);
    if(rendered.has_value())
      pushToWrite(new string(std::move(rendered.value())));
    return rendered.has_value();
  };

  // Common eval cache for all analysis threads
  std::shared_ptr<EvalCacheTable> evalCache = nullptr;
  if(defaultParams.useEvalCache) {
    evalCache = std::make_shared<EvalCacheTable>(defaultParams.subtreeValueBiasTableNumShards);
  }

  //One analysis thread owns one bot per hosted searchable model, indexed by model. Each bot binds
  //its evaluator for its whole life (AsyncBot and Search take it at construction), so the model a
  //request names is served by picking the bot already bound to it -- no evaluator is ever swapped
  //under a live search. The thread runs one of its bots at a time, so hosting N models multiplies
  //the idle bots, not the concurrent searches: the thread budget is unchanged.
  auto analysisLoop = [
    &logger,&toAnalyzeQueue,&pushToWrite,&reportAnalysis,&preventEncore,&logSearchInfo,&modelHosts,&openRequestsMutex,&openRequests
  ](BotsByModel* botsByModel, int threadIdx) {
    while(true) {
      std::pair<std::pair<int64_t,int64_t>,AnalyzeRequest*> analysisItem;
      bool suc = toAnalyzeQueue.waitPop(analysisItem);
      if(!suc)
        break;
      AnalyzeRequest* request = analysisItem.second;
      //Rendered below, emitted after the erase further down. Stays empty on the path where the
      //request was already terminated in the queue -- the terminate wrote its response then.
      std::optional<string> finalResponse;
      //The model was resolved when the request was parsed; this is where that resolution is spent.
      AsyncBot* bot = botsByModel->at(request->modelIdx);
      int expected = AnalyzeRequest::STATUS_IN_QUEUE;
      //If it's already terminated, then there's nothing for us to do
      if(!request->status.compare_exchange_strong(expected, AnalyzeRequest::STATUS_POPPED, std::memory_order_acq_rel)) {
        testAssert(expected == AnalyzeRequest::STATUS_TERMINATED);
      }
      //Else, the request is live and we marked it as popped
      else {
        bot->setPosition(request->nextPla,request->board,request->hist);
        bot->setAlwaysIncludeOwnerMap(request->includeOwnership || request->includeOwnershipStdev || request->includeMovesOwnership || request->includeMovesOwnershipStdev);
        bot->setParams(request->params);
        //The cache context was resolved when the request was parsed, against this very model; this
        //is where that resolution is spent.
        bot->setCacheAttribution(request->cacheAttribution);
        //Closes the previous analyze request's admission-signal-measurement window and opens
        //this one's, substituting one per-analyze-request boundary for the one-per-search
        //boundary NNCacheTable::beginAdmissionSignalMeasurementWindow's own comment names as
        //the harder-to-reach ideal. A no-op unless nnCacheAdmissionSignalMeasurement is on.
        modelHosts.searchableEval(request->modelIdx)->cacheTable().beginAdmissionSignalMeasurementWindow();
        bot->setAvoidMoveUntilByLoc(request->avoidMoveUntilByLocBlack,request->avoidMoveUntilByLocWhite);

        Player pla = request->nextPla;
        double searchFactor = 1.0;

        //Handle termination between the time we pop and the search starts
        std::function<void()> onSearchBegun = [&request,&bot,&threadIdx]() {
          //Try to record that we're handling this request and indicate that the search is started by this thread
          int expected2 = AnalyzeRequest::STATUS_POPPED;
          //If it was terminated, then stop our search
          if(!request->status.compare_exchange_strong(expected2, threadIdx, std::memory_order_acq_rel)) {
            testAssert(expected2 == AnalyzeRequest::STATUS_TERMINATED);
            bot->stopWithoutWait();
          }
        };

        if(request->reportDuringSearch) {
          std::function<void(const Search* search)> callback = [&request,&reportAnalysis](const Search* search) {
            const bool isDuringSearch = true;
            reportAnalysis(request,search,isDuringSearch);
          };
          bot->genMoveSynchronousAnalyze(
            pla, TimeControls(), searchFactor,
            request->reportDuringSearchEvery, request->firstReportDuringSearchAfter,
            callback, onSearchBegun
          );
        }
        else {
          bot->genMoveSynchronous(pla, TimeControls(), searchFactor, onSearchBegun);
        }

        if(logSearchInfo) {
          ostringstream sout;
          PlayUtils::printGenmoveLog(sout,bot->getSearch(),modelHosts.searchableEval(request->modelIdx),Board::NULL_LOC,NAN,request->perspective,false);
          logger.write(sout.str());
        }

        {
          const bool isDuringSearch = false;
          const Search* search = bot->getSearch();
          finalResponse = renderAnalysis(request,search,isDuringSearch,preventEncore);
          //If the search didn't have any root or root neural net output, it must have been interrupted and we must be quitting imminently
          if(!finalResponse.has_value()) {
            //If the reason we stopped was because we noticed a terminate, then we will write out a dummy response even if we didn't have
            //enough info to generate a real one, to fulfill a promise in the API docs that we always write something.
            if(request->status.load(std::memory_order_acquire) == AnalyzeRequest::STATUS_TERMINATED)
              finalResponse = renderNoAnalysis(request);
            //Otherwise, this case is only possible if we're just shutting down
            else
              logger.write("Note: Search quitting due to no visits - this is normal and possible when shutting down but a bug under any other situation.");
          }
        }
      }

      //Free up bot resources in case it's a while before we do more search
      bot->clearSearch();

      //This request is no longer open, and the erase says so BEFORE the response is queued for
      //output. That order is the whole point (see renderAnalysis above): a client that has read
      //this response can send a cache action and be certain this request is not in the count that
      //action reads.
      //It is also safe to erase only here and not earlier: every cache lookup and store a request
      //makes happens inside NNEvaluator::evaluate on a search thread, and genMoveSynchronous* has
      //already waited for every one of those threads to finish. So the entry's removal strictly
      //follows the last time this request could touch the list an attach or detach rewrites --
      //which is why the count can be transiently too HIGH but never too low.
      {
        std::lock_guard<std::mutex> lock(openRequestsMutex);
        openRequests.erase(request->internalId);
      }
      if(finalResponse.has_value())
        pushToWrite(new string(std::move(finalResponse.value())));
      delete request;
    }
  };
  auto analysisLoopProtected = [&logger,&analysisLoop](BotsByModel* botsByModel, int threadIdx) {
    Logger::logThreadUncaught("analysis loop", &logger, [&](){ analysisLoop(botsByModel, threadIdx); });
  };

  vector<std::thread> threads;
  std::thread write_thread = std::thread(writeLoop);
  //One pool per analysis thread, each holding one bot per model. Sized up front so that the
  //pointers handed to the threads below stay valid as later threads' pools are filled.
  vector<BotsByModel> bots(numAnalysisThreads);
  for(int threadIdx = 0; threadIdx<numAnalysisThreads; threadIdx++) {
    for(const SearchableModelIdx modelIdx: modelHosts.searchableIdxs()) {
      string searchRandSeed = Global::uint64ToHexString(seedRand.nextUInt64()) + Global::uint64ToHexString(seedRand.nextUInt64());
      AsyncBot* bot = new AsyncBot(defaultParams, modelHosts.searchableEval(modelIdx), humanEval, &logger, searchRandSeed);
      bot->setCopyOfExternalPatternBonusTable(patternBonusTable);
      bot->setExternalEvalCache(evalCache);
      bots[threadIdx].add(bot);
    }
    threads.emplace_back(analysisLoopProtected,&bots[threadIdx],threadIdx);
  }

  logger.write("Analyzing up to " + Global::intToString(numAnalysisThreads) + " positions at a time in parallel");
  logger.write("Started, ready to begin handling requests");
  if(!logToStderr) {
    cerr << "Started, ready to begin handling requests" << endl;
  }


  auto requestLoop = [&]() {
    string line;
    json input;
    while(getline(cin,line)) {
      line = Global::trim(line);
      if(line.length() == 0)
        continue;

      if(logAllRequests)
        logger.write("Request: " + line);

      try {
        input = json::parse(line);
      }
      catch(nlohmann::detail::exception& e) {
        reportError(e.what() + string(" - could not parse input line as json request: ") + line);
        continue;
      }

      if(!input.is_object()) {
        reportError("Request line was valid json but was not an object, ignoring: " + input.dump());
        continue;
      }

      if(input.find("id") == input.end() || !input["id"].is_string()) {
        reportError("Request must have a string \"id\" field");
        continue;
      }

      AnalyzeRequest rbase;
      rbase.id = input["id"].get<string>();

      //Special actions
      if(input.find("action") != input.end() && input["action"].is_string()) {
        string action = input["action"].get<string>();
        //The cache actions are the ones that DO mean something by "model": it selects whose cache
        //-- and therefore whose files -- the action addresses. Every other action is unchanged.
        const bool isCacheAction =
          action == "cache_attach" || action == "cache_detach" ||
          action == "cache_dump" || action == "cache_stats";
        //"model" selects which model analyzes a QUERY. The non-cache actions read it nowhere, and
        //an action that accepted it and then did the same thing regardless would be a field the
        //receiver cannot honor -- so it is refused here rather than ignored.
        if(!isCacheAction && input.find("model") != input.end()) {
          reportErrorForId(rbase.id, "model", "The \"model\" field selects which model analyzes a query and is not supported on this action; the cache actions take it to select whose cache they address");
          continue;
        }
        //Same disposition, same reason: "cacheContext" says which attached context a QUERY's new
        //evaluations are earned by. No action reads it -- the cache actions name their context in
        //their own "context" field, which is a different question (which body of persisted content
        //this act is ABOUT, not which one a query's earnings belong to) -- so it is refused here for
        //every action rather than silently ignored.
        if(input.find("cacheContext") != input.end()) {
          reportErrorForId(rbase.id, "cacheContext", "The \"cacheContext\" field attributes a query's new cache entries and is not supported on an action query; the cache actions name the context they act on in their own \"context\" field");
          continue;
        }
        if(action == "query_version") {
          input["version"] = Version::getKataGoVersion();
          input["git_hash"] = Version::getGitRevision();
          pushToWrite(new string(input.dump()));
        }
        else if(action == "query_models") {
          //Every hosted model, searchable ones first in the order they were configured, the human
          //companion last -- so with one model and no -human-model this is the same one entry it
          //has always been, and the "internalName" a client reads here is exactly the name the
          //"model" field of a request takes.
          input["models"] = json::array();
          for(const SearchableModelIdx modelIdx: modelHosts.searchableIdxs()) {
            NNEvaluator* eval = modelHosts.searchableEval(modelIdx);
            json modelInfo;
            modelInfo["name"] = eval->getModelName();
            modelInfo["internalName"] = eval->getInternalModelName();
            modelInfo["maxBatchSize"] = eval->getMaxBatchSize();
            modelInfo["usesHumanSLProfile"] = eval->requiresSGFMetadata();
            modelInfo["version"] = eval->getModelVersion();
            modelInfo["usingFP16"] = eval->getUsingFP16Mode().toString();
            input["models"].push_back(modelInfo);
          }
          if(humanEval != NULL) {
            json modelInfo;
            modelInfo["name"] = humanEval->getModelName();
            modelInfo["internalName"] = humanEval->getInternalModelName();
            modelInfo["maxBatchSize"] = humanEval->getMaxBatchSize();
            modelInfo["usesHumanSLProfile"] = humanEval->requiresSGFMetadata();
            modelInfo["version"] = humanEval->getModelVersion();
            modelInfo["usingFP16"] = humanEval->getUsingFP16Mode().toString();
            input["models"].push_back(modelInfo);
          }
          pushToWrite(new string(input.dump()));
        }
        else if(action == "clear_cache") {
          //This should be thread-safe.
          //Every hosted model's cache: the action has always meant "drop everything cached", and a
          //model whose cache it silently skipped would keep serving entries the client asked to be
          //rid of. With one model this is the one clearCache it has always been.
          for(const SearchableModelIdx modelIdx: modelHosts.searchableIdxs())
            modelHosts.searchableEval(modelIdx)->clearCache();
          if(humanEval != NULL)
            humanEval->clearCache();
          if(evalCache != nullptr)
            evalCache->clear();
          pushToWrite(new string(input.dump()));
        }
        else if(action == "terminate") {

          bool terminateIdFound = false;
          string terminateId;
          if(input.find("terminateId") != input.end() && input["terminateId"].is_string()) {
            terminateId = input["terminateId"].get<string>();
            terminateIdFound = true;
          }
          if(!terminateIdFound) {
            reportErrorForId(rbase.id, "terminateId", "Requests for a terminate action must have a string \"terminateId\" field");
            continue;
          }

          bool hasTurnNumbers = false;
          vector<int> turnNumbers;
          if(input.find("turnNumbers") != input.end()) {
            try {
              turnNumbers = input["turnNumbers"].get<vector<int> >();
              hasTurnNumbers = true;
            }
            catch(nlohmann::detail::exception&) {
              reportErrorForId(rbase.id, "turnNumbers", "If provided, must be an array of integers indicating turns to terminate");
              continue;
            }
          }

          vector<string> closedResponses;
          {
            std::lock_guard<std::mutex> lock(openRequestsMutex);
            std::set<int> turnNumbersSet(turnNumbers.begin(),turnNumbers.end());
            //Matched first, then terminated, because closing one erases it from the map being walked.
            vector<AnalyzeRequest*> matched;
            for(auto it = openRequests.begin(); it != openRequests.end(); ++it) {
              AnalyzeRequest* request = it->second;
              if(request->id == terminateId && (!hasTurnNumbers || (turnNumbersSet.find(request->turnNumber) != turnNumbersSet.end())))
                matched.push_back(request);
            }
            closedResponses = closeTerminated(bots, openRequests, matched);
          }
          for(string& response: closedResponses)
            pushToWrite(new string(std::move(response)));
          pushToWrite(new string(input.dump()));
        }
        else if(action == "terminate_all") {
          bool hasTurnNumbers = false;
          vector<int> turnNumbers;
          if(input.find("turnNumbers") != input.end()) {
            try {
              turnNumbers = input["turnNumbers"].get<vector<int> >();
              hasTurnNumbers = true;
            }
            catch(nlohmann::detail::exception&) {
              reportErrorForId(rbase.id, "turnNumbers", "If provided, must be an array of integers indicating turns to terminate");
              continue;
            }
          }

          vector<string> closedResponses;
          {
            std::lock_guard<std::mutex> lock(openRequestsMutex);
            std::set<int> turnNumbersSet(turnNumbers.begin(),turnNumbers.end());
            //Matched first, then terminated, because closing one erases it from the map being walked.
            vector<AnalyzeRequest*> matched;
            for(auto it = openRequests.begin(); it != openRequests.end(); ++it) {
              AnalyzeRequest* request = it->second;
              if(!hasTurnNumbers || (turnNumbersSet.find(request->turnNumber) != turnNumbersSet.end()))
                matched.push_back(request);
            }
            closedResponses = closeTerminated(bots, openRequests, matched);
          }
          for(string& response: closedResponses)
            pushToWrite(new string(std::move(response)));
          pushToWrite(new string(input.dump()));
        }
        else if(isCacheAction) {
          //Which model's cache. The same translate-and-validate Port a query's "model" field goes
          //through, and for a sharper reason: a cache action names the FILES it reads and writes, so
          //coercing an unknown name to the primary model would read or write another model's
          //container under this one's name, with nothing in the response to say it happened.
          SearchableModelIdx modelIdx = AnalysisModelHosts::PRIMARY_SEARCHABLE_IDX;
          bool modelResolved = true;
          if(input.find("model") != input.end()) {
            if(!input["model"].is_string()) {
              reportErrorForId(rbase.id, "model", "Must be a string, the \"internalName\" of a loaded model as the query_models action reports it");
              modelResolved = false;
            }
            else {
              const ModelResolution resolution = modelHosts.resolve(input["model"].get<string>());
              if(!resolution.searchableIdx().has_value()) {
                reportErrorForId(rbase.id, "model", resolution.refusal().value());
                modelResolved = false;
              }
              else
                modelIdx = resolution.searchableIdx().value();
            }
          }
          if(!modelResolved)
            continue;

          int64_t openRequestCount = 0;
          {
            std::lock_guard<std::mutex> lock(openRequestsMutex);
            openRequestCount = (int64_t)openRequests.size();
          }
          const AnalysisEngineCounters counters{openRequestCount};

          //The decode's refusal is reported before the concurrency one deliberately: a client with a
          //typo'd field learns about the typo rather than being told to try again later and hitting
          //the same wall.
          try {
            bool handled = false;
            json result;
            if(action == "cache_attach") {
              const CacheActionDecode<CacheAttachRequest> decoded = decodeCacheAttach(input);
              if(decoded.refusal().has_value())
                reportErrorForId(rbase.id, decoded.refusal().value().field, decoded.refusal().value().message);
              else if(openRequestCount > 0)
                reportErrorForId(rbase.id, "action", cacheSwapConcurrencyRefusal("cache_attach", openRequestCount));
              else {
                result = cacheAttachExecute(modelHosts, modelIdx, cacheAttachments, decoded.value().value());
                handled = true;
              }
            }
            else if(action == "cache_detach") {
              const CacheActionDecode<CacheDetachRequest> decoded = decodeCacheDetach(input);
              if(decoded.refusal().has_value())
                reportErrorForId(rbase.id, decoded.refusal().value().field, decoded.refusal().value().message);
              else if(openRequestCount > 0)
                reportErrorForId(rbase.id, "action", cacheSwapConcurrencyRefusal("cache_detach", openRequestCount));
              else {
                result = cacheDetachExecute(modelHosts, modelIdx, cacheAttachments, decoded.value().value());
                handled = true;
              }
            }
            else if(action == "cache_dump") {
              //Legal while requests are open -- every structure a dump reads is thread-safe and off
              //the get/set path -- so there is no refusal here. The response carries the open-request
              //count at dump time, so a client that dumped live can see that it did.
              const CacheActionDecode<CacheDumpRequest> decoded = decodeCacheDump(input);
              if(decoded.refusal().has_value())
                reportErrorForId(rbase.id, decoded.refusal().value().field, decoded.refusal().value().message);
              else {
                result = cacheDumpExecute(modelHosts, modelIdx, cacheAttachments, decoded.value().value(), counters);
                handled = true;
              }
            }
            else {
              testAssert(action == "cache_stats");
              const CacheActionDecode<CacheStatsRequest> decoded = decodeCacheStats(input);
              if(decoded.refusal().has_value())
                reportErrorForId(rbase.id, decoded.refusal().value().field, decoded.refusal().value().message);
              else {
                result = cacheStatsExecute(modelHosts, modelIdx, cacheAttachments);
                handled = true;
              }
            }
            if(handled) {
              for(json::const_iterator it = result.begin(); it != result.end(); ++it)
                input[it.key()] = it.value();
              pushToWrite(new string(input.dump()));
            }
          }
          catch(const StringError& e) {
            reportErrorForId(rbase.id, "action", e.what());
          }
        }
        else {
          reportError("'action' field must be 'query_version' or 'query_models' or 'clear_cache' or 'cache_attach' or 'cache_detach' or 'cache_dump' or 'cache_stats' or 'terminate' or 'terminate_all'");
        }

        continue;
      }

      //Defaults
      rbase.params = defaultParams;
      rbase.perspective = defaultPerspective;
      rbase.analysisPVLen = analysisPVLen;
      rbase.includeOwnership = false;
      rbase.includeOwnershipStdev = false;
      rbase.includeMovesOwnership = false;
      rbase.includeMovesOwnershipStdev = false;
      rbase.includePolicy = false;
      rbase.includePVVisits = false;
      rbase.includeNoResultValue = false;
      rbase.reportDuringSearch = false;
      rbase.reportDuringSearchEvery = 1e30;
      rbase.firstReportDuringSearchAfter = 1e30;
      rbase.priority = 0;
      rbase.avoidMoveUntilByLocBlack.clear();
      rbase.avoidMoveUntilByLocWhite.clear();
      //A request that names no model is served by the primary model, which without -extra-model is
      //the only model there is: the no-model path is the engine's prior behaviour, untouched.
      rbase.modelIdx = AnalysisModelHosts::PRIMARY_SEARCHABLE_IDX;
      //Reset alongside it: rbase is reused across requests, so a context resolved for the previous
      //one must not survive into a request that named none.
      rbase.cacheAttribution = NNCacheAttribution::noAttributableContext();

      //The optional "model" field. An unrecognized key elsewhere in a request is only WARNED about,
      //and that warning is switchable off with warnUnusedFields -- a disposition this field cannot
      //borrow. A "model" the engine cannot honor is refused here as an error and the request is not
      //analyzed at all, because the alternative is analyzing with a net the client did not ask for,
      //which is precisely the outcome the field exists to prevent, and which the response carries
      //no evidence of.
      if(input.find("model") != input.end()) {
        if(!input["model"].is_string()) {
          reportErrorForId(rbase.id, "model", "Must be a string, the \"internalName\" of a loaded model as the query_models action reports it");
          continue;
        }
        const ModelResolution resolution = modelHosts.resolve(input["model"].get<string>());
        if(!resolution.searchableIdx().has_value()) {
          reportErrorForId(rbase.id, "model", resolution.refusal().value());
          continue;
        }
        rbase.modelIdx = resolution.searchableIdx().value();
      }

      //The optional "cacheContext" field. It is resolved against the model this request already
      //selected, because contexts are attached per model: each model's cache has its own attach
      //order and its own name space, and a name attached to one says nothing about another.
      //Resolving it here, after "model", is what makes that true by construction.
      //An unknown name is an ERROR and the request is not analyzed, rather than being analyzed
      //with its earnings filed under some other context -- which would write this session's work
      //into the wrong card's file, with no field of the response carrying evidence it happened.
      {
        std::optional<string> requestedContext;
        if(input.find("cacheContext") != input.end()) {
          if(!input["cacheContext"].is_string()) {
            reportErrorForId(rbase.id, "cacheContext", "Must be a string naming a cache context attached to the model this query selects");
            continue;
          }
          requestedContext = input["cacheContext"].get<string>();
        }
        const NNCacheContextResolution contextResolution =
          modelHosts.searchableEval(rbase.modelIdx)->resolveCacheContext(requestedContext);
        if(!contextResolution.attribution().has_value()) {
          reportErrorForId(rbase.id, "cacheContext", contextResolution.refusal().value());
          continue;
        }
        rbase.cacheAttribution = contextResolution.attribution().value();
      }

      auto parseInteger = [&rbase,&reportErrorForId](const json& dict, const char* field, int64_t& buf, int64_t min, int64_t max, const char* errorMessage) {
        try {
          if(!dict[field].is_number_integer()) {
            reportErrorForId(rbase.id, field, errorMessage);
            return false;
          }
          int64_t x = dict[field].get<int64_t>();
          if(x < min || x > max) {
            reportErrorForId(rbase.id, field, errorMessage);
            return false;
          }
          buf = x;
          return true;
        }
        catch(nlohmann::detail::exception& e) {
          (void)e;
          reportErrorForId(rbase.id, field, errorMessage);
          return false;
        }
      };

      auto parseDouble = [&rbase,&reportErrorForId](const json& dict, const char* field, double& buf, double min, double max, const char* errorMessage) {
        try {
          if(!dict[field].is_number()) {
            reportErrorForId(rbase.id, field, errorMessage);
            return false;
          }
          double x = dict[field].get<double>();
          if(!isfinite(x) || x < min || x > max) {
            reportErrorForId(rbase.id, field, errorMessage);
            return false;
          }
          buf = x;
          return true;
        }
        catch(nlohmann::detail::exception& e) {
          (void)e;
          reportErrorForId(rbase.id, field, errorMessage);
          return false;
        }
      };

      auto parseBoolean = [&rbase,&reportErrorForId](const json& dict, const char* field, bool& buf, const char* errorMessage) {
        try {
          if(!dict[field].is_boolean()) {
            reportErrorForId(rbase.id, field, errorMessage);
            return false;
          }
          buf = dict[field].get<bool>();
          return true;
        }
        catch(nlohmann::detail::exception& e) {
          (void)e;
          reportErrorForId(rbase.id, field, errorMessage);
          return false;
        }
      };

      auto parsePlayer = [&rbase,&reportErrorForId](const json& dict, const char* field, Player& buf) {
        buf = C_EMPTY;
        try {
          string s = dict[field].get<string>();
          PlayerIO::tryParsePlayer(s,buf);
        }
        catch(nlohmann::detail::exception&) {}
        if(buf != P_BLACK && buf != P_WHITE) {
          reportErrorForId(rbase.id, field, "Must be \"b\" or \"w\"");
          return false;
        }
        return true;
      };

      int boardXSize;
      int boardYSize;
      {
        int64_t xBuf;
        int64_t yBuf;
        static const string boardSizeError = string("Must provide an integer from 2 to ") + Global::intToString(Board::MAX_LEN);
        if(input.find("boardXSize") == input.end()) {
          reportErrorForId(rbase.id, "boardXSize", boardSizeError.c_str());
          continue;
        }
        if(input.find("boardYSize") == input.end()) {
          reportErrorForId(rbase.id, "boardYSize", boardSizeError.c_str());
          continue;
        }
        if(!parseInteger(input, "boardXSize", xBuf, 2, Board::MAX_LEN, boardSizeError.c_str())) {
          continue;
        }
        if(!parseInteger(input, "boardYSize", yBuf, 2, Board::MAX_LEN, boardSizeError.c_str())) {
          continue;
        }
        boardXSize = (int)xBuf;
        boardYSize = (int)yBuf;
      }

      auto parseBoardLocs = [boardXSize,boardYSize,&rbase,&reportErrorForId](const json& dict, const char* field, vector<Loc>& buf, bool allowPass) {
        buf.clear();
        if(!dict[field].is_array()) {
          reportErrorForId(rbase.id, field, "Must be an array of GTP board vertices");
          return false;
        }
        for(auto& elt : dict[field]) {
          string s;
          try {
            s = elt.get<string>();
          }
          catch(nlohmann::detail::exception& e) {
            (void)e;
            reportErrorForId(rbase.id, field, "Must be an array of GTP board vertices");
            return false;
          }

          Loc loc;
          if(!Location::tryOfString(s, boardXSize, boardYSize, loc) ||
             (!allowPass && loc == Board::PASS_LOC) ||
             (loc == Board::NULL_LOC)) {
            reportErrorForId(rbase.id, field, "Could not parse board location: " + s);
            return false;
          }
          buf.push_back(loc);
        }
        return true;
      };

      auto parseBoardMoves = [boardXSize,boardYSize,&rbase,&reportErrorForId](const json& dict, const char* field, vector<Move>& buf, bool allowPass) {
        buf.clear();
        if(!dict[field].is_array()) {
          reportErrorForId(rbase.id, field, "Must be an array of pairs of the form: [\"b\" or \"w\", GTP board vertex]");
          return false;
        }
        for(auto& elt : dict[field]) {
          if(!elt.is_array() || elt.size() != 2) {
            reportErrorForId(rbase.id, field, "Must be an array of pairs of the form: [\"b\" or \"w\", GTP board vertex]");
            return false;
          }

          string s0;
          string s1;
          try {
            s0 = elt[0].get<string>();
            s1 = elt[1].get<string>();
          }
          catch(nlohmann::detail::exception& e) {
            (void)e;
            reportErrorForId(rbase.id, field, "Must be an array of pairs of the form: [\"b\" or \"w\", GTP board vertex]");
            return false;
          }

          Player pla;
          if(!PlayerIO::tryParsePlayer(s0,pla)) {
            reportErrorForId(rbase.id, field, "Could not parse player: " + s0);
            return false;
          }

          Loc loc;
          if(!Location::tryOfString(s1, boardXSize, boardYSize, loc) ||
             (!allowPass && loc == Board::PASS_LOC) ||
             (loc == Board::NULL_LOC)) {
            reportErrorForId(rbase.id, field, "Could not parse board location: " + s1);
            return false;
          }
          buf.emplace_back(loc,pla);
        }
        return true;
      };

      vector<Move> placements;
      if(input.find("initialStones") != input.end()) {
        if(!parseBoardMoves(input, "initialStones", placements, false))
          continue;
      }
      vector<Move> moveHistory;
      if(input.find("moves") != input.end()) {
        if(!parseBoardMoves(input, "moves", moveHistory, true))
          continue;
      }
      else {
        reportErrorForId(rbase.id, "moves", "Must specify an array of [player,location] pairs");
        continue;
      }
      Player initialPlayer = C_EMPTY;
      if(input.find("initialPlayer") != input.end()) {
        bool suc = parsePlayer(input, "initialPlayer", initialPlayer);
        if(!suc)
          continue;
      }

      vector<bool> shouldAnalyze(moveHistory.size()+1,false);
      if(input.find("analyzeTurns") != input.end()) {
        vector<int> analyzeTurns;
        try {
          analyzeTurns = input["analyzeTurns"].get<vector<int> >();
        }
        catch(nlohmann::detail::exception&) {
          reportErrorForId(rbase.id, "analyzeTurns", "Must specify an array of integers indicating turns to analyze");
          continue;
        }

        bool failed = false;
        for(int i = 0; i<analyzeTurns.size(); i++) {
          int turnNumber = analyzeTurns[i];
          if(turnNumber < 0 || turnNumber >= shouldAnalyze.size()) {
            reportErrorForId(rbase.id, "analyzeTurns", "Invalid turn number: " + Global::intToString(turnNumber));
            failed = true;
            break;
          }
          shouldAnalyze[turnNumber] = true;
        }
        if(failed)
          continue;
      }
      else {
        shouldAnalyze[shouldAnalyze.size()-1] = true;
      }

      std::map<int,int64_t> priorities;
      if(input.find("priorities") != input.end()) {
        vector<int64_t> prioritiesVec;
        try {
          prioritiesVec = input["priorities"].get<vector<int64_t> >();
        }
        catch(nlohmann::detail::exception&) {
          reportErrorForId(rbase.id, "priorities", "Must specify an array of integers indicating priorities");
          continue;
        }
        if(input.find("analyzeTurns") == input.end()) {
          reportErrorForId(rbase.id, "priorities", "Can only specify when also specifying analyzeTurns");
          continue;
        }
        vector<int> analyzeTurns = input["analyzeTurns"].get<vector<int> >();
        if(prioritiesVec.size() != analyzeTurns.size()) {
          reportErrorForId(rbase.id, "priorities", "Must be of matching length to analyzeTurns");
          continue;
        }

        bool failed = false;
        for(int i = 0; i<prioritiesVec.size(); i++) {
          int64_t priority = prioritiesVec[i];
          if(priority < -0x3FFFffffFFFFffffLL || priority > 0x3FFFffffFFFFffffLL) {
            reportErrorForId(rbase.id, "priorities", "Invalid priority: " + Global::int64ToString(priority));
            failed = true;
            break;
          }
          priorities[analyzeTurns[i]] = priority;
        }
        if(failed) {
          priorities.clear();
          continue;
        }
      }


      Rules rules;
      if(input.find("rules") != input.end()) {
        if(input["rules"].is_string()) {
          string s = input["rules"].get<string>();
          if(!Rules::tryParseRules(s,rules)) {
            reportErrorForId(rbase.id, "rules", "Could not parse rules: " + s);
            continue;
          }
        }
        else if(input["rules"].is_object()) {
          string s = input["rules"].dump();
          if(!Rules::tryParseRules(s,rules)) {
            reportErrorForId(rbase.id, "rules", "Could not parse rules: " + s);
            continue;
          }
        }
        else {
          reportErrorForId(rbase.id, "rules", "Must specify rules string, such as \"chinese\" or \"tromp-taylor\", or a JSON object with detailed rules parameters.");
          continue;
        }
      }
      else {
        reportErrorForId(rbase.id, "rules", "Must specify rules string, such as \"chinese\" or \"tromp-taylor\", or a JSON object with detailed rules parameters.");
        continue;
      }

      if(input.find("komi") != input.end()) {
        double komi;
        static_assert(Rules::MIN_USER_KOMI == -400.0f, "");
        static_assert(Rules::MAX_USER_KOMI == 400.0f, "");
        const char* msg = "Must be a integer or half-integer from -400.0 to 400.0";
        bool suc = parseDouble(input, "komi", komi, Rules::MIN_USER_KOMI, Rules::MAX_USER_KOMI, msg);
        if(!suc)
          continue;
        rules.komi = (float)komi;
        if(!Rules::komiIsIntOrHalfInt(rules.komi)) {
          reportErrorForId(rbase.id, "komi", msg);
          continue;
        }
      }

      if(input.find("whiteHandicapBonus") != input.end()) {
        if(!input["whiteHandicapBonus"].is_string()) {
          reportErrorForId(rbase.id, "whiteHandicapBonus", "Must be a string");
          continue;
        }
        string s = input["whiteHandicapBonus"].get<string>();
        try {
          int whiteHandicapBonusRule = Rules::parseWhiteHandicapBonusRule(s);
          rules.whiteHandicapBonusRule = whiteHandicapBonusRule;
        }
        catch(const StringError& err) {
          reportErrorForId(rbase.id, "whiteHandicapBonus", err.what());
          continue;
        }
      }

      if(input.find("overrideSettings") != input.end()) {
        json settings = input["overrideSettings"];
        if(!settings.is_object()) {
          reportErrorForId(rbase.id, "overrideSettings", "Must be an object");
          continue;
        }
        std::map<string,string> overrideSettings;
        for(auto it = settings.begin(); it != settings.end(); ++it) {
          overrideSettings[it.key()] = it.value().is_string() ? it.value().get<string>(): it.value().dump(); // always convert to string
        }

        // Reload settings to allow overrides
        if(!overrideSettings.empty()) {
          try {
            ConfigParser localCfg(cfg);
            //Ignore any unused keys in the ORIGINAL config
            localCfg.markAllKeysUsedWithPrefix("");
            localCfg.overrideKeys(overrideSettings);
            loadParams(localCfg, rbase.params, rbase.perspective, defaultPerspective);
            SearchParams::failIfParamsDifferOnUnchangeableParameter(defaultParams,rbase.params);
            //Soft failure on unused override keys newly present in the config
            vector<string> unusedKeys = localCfg.unusedKeys();
            if(unusedKeys.size() > 0) {
              reportWarningForId(rbase.id, "overrideSettings", string("Unknown config params: ") + Global::concat(unusedKeys,","));
            }
            ostringstream out;
            //The request's own model, not the primary one: this check decides whether THIS request
            //is honored, and asking a net the request did not name is the wrong-net service in the
            //quietest register of all -- it does not serve a wrong evaluation, it refuses or admits
            //a request on evidence from the wrong net.
            if(Setup::maybeWarnHumanSLParams(rbase.params,modelHosts.searchableEval(rbase.modelIdx),humanEval,out,NULL)) {
              throw StringError(out.str());
            }
          }
          catch(const StringError& exception) {
            reportErrorForId(rbase.id, "overrideSettings", string("Could not set settings: ") + exception.what());
            continue;
          }
        }
      }

      if(input.find("maxVisits") != input.end()) {
        bool suc = parseInteger(input, "maxVisits", rbase.params.maxVisits, 1, (int64_t)1 << 50, "Must be an integer from 1 to 2^50");
        if(!suc)
          continue;
      }

      if(input.find("analysisPVLen") != input.end()) {
        int64_t buf;
        bool suc = parseInteger(input, "analysisPVLen", buf, 1, 1000, "Must be an integer from 1 to 1000");
        if(!suc)
          continue;
        rbase.analysisPVLen = (int)buf;
      }

      if(input.find("rootFpuReductionMax") != input.end()) {
        bool suc = parseDouble(input, "rootFpuReductionMax", rbase.params.rootFpuReductionMax, 0.0, 2.0, "Must be a number from 0.0 to 2.0");
        if(!suc)
          continue;
      }
      if(input.find("rootPolicyTemperature") != input.end()) {
        bool suc = parseDouble(input, "rootPolicyTemperature", rbase.params.rootPolicyTemperature, 0.01, 100.0, "Must be a number from 0.01 to 100.0");
        if(!suc)
          continue;
        rbase.params.rootPolicyTemperatureEarly = rbase.params.rootPolicyTemperature;
      }
      if(input.find("includeMovesOwnership") != input.end()) {
        bool suc = parseBoolean(input, "includeMovesOwnership", rbase.includeMovesOwnership, "Must be a boolean");
        if(!suc)
          continue;
      }
      if(input.find("includeMovesOwnershipStdev") != input.end()) {
        bool suc = parseBoolean(input, "includeMovesOwnershipStdev", rbase.includeMovesOwnershipStdev, "Must be a boolean");
        if(!suc)
          continue;
      }
      if(input.find("includeOwnership") != input.end()) {
        bool suc = parseBoolean(input, "includeOwnership", rbase.includeOwnership, "Must be a boolean");
        if(!suc)
          continue;
      }
      if(input.find("includeOwnershipStdev") != input.end()) {
        bool suc = parseBoolean(input, "includeOwnershipStdev", rbase.includeOwnershipStdev, "Must be a boolean");
        if(!suc)
          continue;
      }
      if(input.find("includePolicy") != input.end()) {
        bool suc = parseBoolean(input, "includePolicy", rbase.includePolicy, "Must be a boolean");
        if(!suc)
          continue;
      }
      if(input.find("includePVVisits") != input.end()) {
        bool suc = parseBoolean(input, "includePVVisits", rbase.includePVVisits, "Must be a boolean");
        if(!suc)
          continue;
      }
      if(input.find("includeNoResultValue") != input.end()) {
        bool suc = parseBoolean(input, "includeNoResultValue", rbase.includeNoResultValue, "Must be a boolean");
        if(!suc)
          continue;
      }
      if(input.find("reportDuringSearchEvery") != input.end()) {
        bool suc = parseDouble(input, "reportDuringSearchEvery", rbase.reportDuringSearchEvery, 0.001, 1000000.0, "Must be number of seconds from 0.001 to 1000000.0");
        if(!suc)
          continue;
        rbase.reportDuringSearch = true;
        rbase.firstReportDuringSearchAfter = rbase.reportDuringSearchEvery;
      }
      if(input.find("firstReportDuringSearchAfter") != input.end()) {
        bool suc = parseDouble(input, "firstReportDuringSearchAfter", rbase.firstReportDuringSearchAfter, 0.001, 1000000.0, "Must be number of seconds from 0.001 to 1000000.0");
        if(!suc)
          continue;
        rbase.reportDuringSearch = true;
      }
      if(input.find("priority") != input.end()) {
        if(input.find("priorities") != input.end()) {
          reportErrorForId(rbase.id, "priority", "Cannot specify both priority and priorities");
          continue;
        }
        int64_t buf;
        bool suc = parseInteger(input, "priority", buf, -0x3FFFffffFFFFffffLL,0x3FFFffffFFFFffffLL, "Must be a number between -2^62 and 2^62");
        if(!suc)
          continue;
        rbase.priority = buf;
      }

      bool hasAllowMoves = input.find("allowMoves") != input.end();
      bool hasAvoidMoves = input.find("avoidMoves") != input.end();
      if(hasAllowMoves || hasAvoidMoves) {
        if(hasAllowMoves && hasAvoidMoves) {
          reportErrorForId(rbase.id, "allowMoves", string("Cannot specify both allowMoves and avoidMoves"));
          continue;
        }
        string field = hasAllowMoves ? "allowMoves" : "avoidMoves";
        json& avoidParamsList = input[field];
        if(!avoidParamsList.is_array()) {
          reportErrorForId(rbase.id, field, string("Must be a list of dicts with subfields 'player', 'moves', 'untilDepth'"));
          continue;
        }
        if(hasAllowMoves && avoidParamsList.size() > 2) {
          reportErrorForId(rbase.id, field, string("Currently allowMoves only allows at most one entry per player"));
          continue;
        }

        bool failed = false;
        bool gotAllowMovesBlack = false;
        bool gotAllowMovesWhite = false;
        for(size_t i = 0; i<avoidParamsList.size(); i++) {
          json& avoidParams = avoidParamsList[i];
          if(avoidParams.find("moves") == avoidParams.end() ||
             avoidParams.find("untilDepth") == avoidParams.end() ||
             avoidParams.find("player") == avoidParams.end()) {
            reportErrorForId(rbase.id, field, string("Must be a list of dicts with subfields 'player', 'moves', 'untilDepth'"));
            failed = true;
            break;
          }

          Player avoidPla;
          vector<Loc> parsedLocs;
          int64_t untilDepth;
          bool suc;
          suc = parsePlayer(avoidParams, "player", avoidPla);
          if(!suc) { failed = true; break; }
          suc = parseBoardLocs(avoidParams, "moves", parsedLocs, true);
          if(!suc) { failed = true; break; }
          suc = parseInteger(avoidParams, "untilDepth", untilDepth, 1, 1000000000, "Must be a positive integer");
          if(!suc) { failed = true; break; }

          //For allowMoves, at most one entry per player is permitted. Two entries for the same player would be
          //ambiguous/incorrect since the std::fill below for the second entry would wipe out the first entry's allowed locs.
          //Two entries for different players are fine since they write to separate per-player vectors.
          if(hasAllowMoves) {
            bool& gotAllowMoves = avoidPla == P_BLACK ? gotAllowMovesBlack : gotAllowMovesWhite;
            if(gotAllowMoves) {
              reportErrorForId(rbase.id, field, string("Cannot specify allowMoves more than once for the same player"));
              failed = true; break;
            }
            gotAllowMoves = true;
          }

          vector<int>& avoidMoveUntilByLoc = avoidPla == P_BLACK ? rbase.avoidMoveUntilByLocBlack : rbase.avoidMoveUntilByLocWhite;
          avoidMoveUntilByLoc.resize(Board::MAX_ARR_SIZE);
          if(hasAllowMoves) {
            std::fill(avoidMoveUntilByLoc.begin(),avoidMoveUntilByLoc.end(),(int)untilDepth);
            for(Loc loc: parsedLocs) {
              avoidMoveUntilByLoc[loc] = 0;
            }
          }
          else {
            for(Loc loc: parsedLocs) {
              avoidMoveUntilByLoc[loc] = (int)untilDepth;
            }
          }
        }
        if(failed)
          continue;
      }


      Board board(boardXSize,boardYSize);
      for(int i = 0; i<placements.size(); i++) {
        board.setStone(placements[i].loc,placements[i].pla);
      }

      if(initialPlayer == C_EMPTY) {
        if(moveHistory.size() > 0)
          initialPlayer = moveHistory[0].pla;
        else
          initialPlayer = BoardHistory::numHandicapStonesOnBoard(board) > 0 ? P_WHITE : P_BLACK;
      }

      //Rule support and history modes are properties of the model that will actually run the
      //search, so they are asked of the model this request resolved to.
      NNEvaluator* requestEval = modelHosts.searchableEval(rbase.modelIdx);

      bool rulesWereSupported;
      Rules supportedRules = requestEval->getSupportedRules(rules,rulesWereSupported);
      if(!rulesWereSupported) {
        ostringstream out;
        out << "Rules " << rules << " not supported by neural net, using " << supportedRules << " instead";
        reportWarningForId(rbase.id, "rules", out.str());
        rules = supportedRules;
      }

      Player nextPla = initialPlayer;
      //Keep this request's history consistent with the BoardHistoryModes that the search
      //for this request will resolve to. (The search would re-stamp its own copy anyway, but this keeps
      //any adjudication done during request setup/replay consistent with them.)
      BoardHistory hist(board,nextPla,rules,0,Search::resolveHistoryModes(rbase.params, requestEval));
      hist.setAssumeMultipleStartingBlackMovesAreHandicap(assumeMultipleStartingBlackMovesAreHandicap);

      if(warnUnusedFields) {
        for (auto it = input.begin(); it != input.end(); ++it) {
          if(expectedKeys.find(it.key()) == expectedKeys.end())
            reportWarningForId(rbase.id, it.key(), "Unexpected or unused field, do you have a typo? (set warnUnusedFields=false in the config to disable this warning)");
        }
      }

      //Build and enqueue requests
      vector<AnalyzeRequest*> newRequests;
      bool foundIllegalMove =  false;
      for(int turnNumber = 0; turnNumber <= moveHistory.size(); turnNumber++) {
        if(shouldAnalyze[turnNumber]) {
          int64_t priority = rbase.priority;
          if(priorities.size() > 0) {
            testAssert(priorities.size() > newRequests.size());
            testAssert(priorities.find(turnNumber) != priorities.end());
            priority = priorities[turnNumber];
          }

          AnalyzeRequest* newRequest = new AnalyzeRequest();
          newRequest->internalId = internalIdCounter++;
          newRequest->id = rbase.id;
          newRequest->turnNumber = turnNumber;
          newRequest->modelIdx = rbase.modelIdx;
          newRequest->cacheAttribution = rbase.cacheAttribution;
          newRequest->board = board;
          newRequest->hist = hist;
          newRequest->nextPla = nextPla;
          newRequest->params = rbase.params;
          newRequest->perspective = rbase.perspective;
          newRequest->analysisPVLen = rbase.analysisPVLen;
          newRequest->includeOwnership = rbase.includeOwnership;
          newRequest->includeOwnershipStdev = rbase.includeOwnershipStdev;
          newRequest->includeMovesOwnership = rbase.includeMovesOwnership;
          newRequest->includeMovesOwnershipStdev = rbase.includeMovesOwnershipStdev;
          newRequest->includePolicy = rbase.includePolicy;
          newRequest->includePVVisits = rbase.includePVVisits;
          newRequest->includeNoResultValue = rbase.includeNoResultValue;
          newRequest->reportDuringSearch = rbase.reportDuringSearch;
          newRequest->reportDuringSearchEvery = rbase.reportDuringSearchEvery;
          newRequest->firstReportDuringSearchAfter = rbase.firstReportDuringSearchAfter;
          newRequest->priority = priority;
          newRequest->avoidMoveUntilByLocBlack = rbase.avoidMoveUntilByLocBlack;
          newRequest->avoidMoveUntilByLocWhite = rbase.avoidMoveUntilByLocWhite;
          newRequest->status.store(AnalyzeRequest::STATUS_IN_QUEUE,std::memory_order_release);
          newRequests.push_back(newRequest);
        }
        if(turnNumber >= moveHistory.size())
          break;

        Player movePla = moveHistory[turnNumber].pla;
        Loc moveLoc = moveHistory[turnNumber].loc;
        if(movePla != nextPla) {
          board.clearSimpleKoLoc();
          hist.clear(board,movePla,rules,hist.encorePhase);
          hist.setAssumeMultipleStartingBlackMovesAreHandicap(assumeMultipleStartingBlackMovesAreHandicap);
        }

        bool suc = hist.makeBoardMoveTolerant(board,moveLoc,movePla,preventEncore);
        if(!suc) {
          reportErrorForId(rbase.id, "moves", "Illegal move " + Global::intToString(turnNumber) + ": " + Location::toString(moveLoc,board));
          foundIllegalMove = true;
          break;
        }
        nextPla = getOpp(movePla);
      }

      if(foundIllegalMove) {
        for(int i = 0; i<newRequests.size(); i++)
          delete newRequests[i];
        newRequests.clear();
        continue;
      }

      //Add all requests to open requests
      {
        std::lock_guard<std::mutex> lock(openRequestsMutex);
        for(int i = 0; i<newRequests.size(); i++) {
          openRequests[newRequests[i]->internalId] = newRequests[i];
        }
      }
      //Push into queue for processing
      for(int i = 0; i<newRequests.size(); i++) {
        //Compare first by user-provided priority, and next breaks ties by preferring earlier requests.
        std::pair<int64_t,int64_t> priorityKey = std::make_pair(newRequests[i]->priority, -numRequestsSoFar);
        bool suc = toAnalyzeQueue.forcePush( std::make_pair(priorityKey, newRequests[i]) );
        testAssert(suc);
        numRequestsSoFar++;
      }
      newRequests.clear();
    }
  };

  //If request loop raises an exception, we need to log here BEFORE destructing main context, because in some cases
  //gameThreads[i].join() will abort without useful exception due to thread not being joinable,
  //hiding the real exception.
  Logger::logThreadUncaught("request loop", &logger, requestLoop);

  if(quitWithoutWaiting) {
    //Making this readOnly will halt futher output that isn't already queued and signal the write loop thread to terminate.
    toWriteQueue.setReadOnly();
    //Making this readOnly should signal the analysis loop threads to terminate once they have nothing left.
    toAnalyzeQueue.setReadOnly();
    //Interrupt any searches going on to help the analysis threads realize to terminate faster.
    for(size_t i = 0; i<bots.size(); i++)
      for(AsyncBot* bot: bots[i].all())
        bot->stopWithoutWait();
    for(size_t i = 0; i<bots.size(); i++)
      for(AsyncBot* bot: bots[i].all())
        bot->setKilled();
    for(int i = 0; i<threads.size(); i++)
      threads[i].join();
    write_thread.join();
  }
  else {
    //Making this readOnly should signal the analysis loop threads to terminate once they have nothing left.
    toAnalyzeQueue.setReadOnly();
    //Wait patiently for everything to finish
    for(int i = 0; i<threads.size(); i++)
      threads[i].join();
    //Signal the write loop thread to terminate
    toWriteQueue.setReadOnly();
    write_thread.join();
  }

  for(size_t i = 0; i<bots.size(); i++)
    for(AsyncBot* bot: bots[i].all())
      delete bot;

  //Per hosted model, so a session that ran two nets can be read for how much work each one did.
  for(const SearchableModelIdx modelIdx: modelHosts.searchableIdxs()) {
    NNEvaluator* eval = modelHosts.searchableEval(modelIdx);
    logger.write(eval->getModelFileName());
    logger.write("NN rows: " + Global::int64ToString(eval->numRowsProcessed()));
    logger.write("NN batches: " + Global::int64ToString(eval->numBatchesProcessed()));
    logger.write("NN avg batch size: " + Global::doubleToString(eval->averageProcessedBatchSize()));
  }
  if(humanEval != NULL) {
    logger.write(humanEval->getModelFileName());
    logger.write("NN rows: " + Global::int64ToString(humanEval->numRowsProcessed()));
    logger.write("NN batches: " + Global::int64ToString(humanEval->numBatchesProcessed()));
    logger.write("NN avg batch size: " + Global::doubleToString(humanEval->averageProcessedBatchSize()));
  }
  //main owns the evaluators; modelHosts only refers to them.
  for(const SearchableModelIdx modelIdx: modelHosts.searchableIdxs())
    delete modelHosts.searchableEval(modelIdx);
  delete humanEval;
  NeuralNet::globalCleanup();
  ScoreValue::freeTables();
  logger.write("All cleaned up, quitting");
  return 0;
}
