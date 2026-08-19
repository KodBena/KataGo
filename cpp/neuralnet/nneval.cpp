#include "../neuralnet/nneval.h"
#include "../neuralnet/modelversion.h"
#include "../core/test.h"

using namespace std;

//-------------------------------------------------------------------------------------

NNResultBuf::NNResultBuf()
  : clientWaitingForResult(),
    resultMutex(),
    hasResult(false),
    includeOwnerMap(false),
    boardXSizeForServer(0),
    boardYSizeForServer(0),
    rowSpatialBuf(),
    rowGlobalBuf(),
    rowMetaBuf(),
    hasRowMeta(false),
    result(nullptr),
    errorLogLockout(false),
    // If no symmetry is specified, it will use default or random based on config.
    symmetry(NNInputs::SYMMETRY_NOTSPECIFIED),
    policyOptimism(0.0),
    cacheAttribution()
{}

NNResultBuf::~NNResultBuf() {
}

//-------------------------------------------------------------------------------------

NNServerBuf::NNServerBuf(const NNEvaluator& nnEval, const LoadedModel* model)
  :inputBuffers(NULL)
{
  int maxBatchSize = nnEval.getMaxBatchSize();
  if(model != NULL)
    inputBuffers = NeuralNet::createInputBuffers(model,maxBatchSize,nnEval.getNNXLen(),nnEval.getNNYLen());
}

NNServerBuf::~NNServerBuf() {
  if(inputBuffers != NULL)
    NeuralNet::freeInputBuffers(inputBuffers);
  inputBuffers = NULL;
}

//-------------------------------------------------------------------------------------

NNEvaluator::NNEvaluator(
  const string& mName,
  const string& mFileName,
  const string& expectedSha256,
  Logger* lg,
  int maxBatchSz,
  int xLen,
  int yLen,
  bool rExactNNLen,
  bool iUseNHWC,
  const NNCacheConfig& nnCacheConfig,
  bool skipNeuralNet,
  const string& homeDataDirOverride,
  enabled_t useFP16Mode,
  int numThr,
  const vector<int>& gpuIdxByServerThr,
  const string& rSeed,
  bool doRandomize,
  int defaultSymmetry,
  bool disableWarmup_,
  ConfigParser& cfg
)
  :modelName(mName),
   modelFileName(mFileName),
   nnXLen(xLen),
   nnYLen(yLen),
   requireExactNNLen(rExactNNLen),
   policySize(NNPos::getPolicySize(xLen,yLen)),
   inputsUseNHWC(iUseNHWC),
   usingFP16Mode(useFP16Mode),
   numThreads(numThr),
   gpuIdxByServerThread(gpuIdxByServerThr),
   randSeed(rSeed),
   debugSkipNeuralNet(skipNeuralNet),
   disableWarmup(disableWarmup_),
   computeContext(NULL),
   loadedModel(NULL),
   nnCacheTable(nullptr),
   nnCacheLevelZeroTable(nullptr),
   nnCacheDirectory(nnCacheConfig.cacheDirectory),
#ifndef NDEBUG
   nnCacheEvaluationsInFlight(0),
#endif
   logger(lg),
   internalModelName(),
   modelVersion(-1),
   inputsVersion(-1),
   numInputMetaChannels(0),
   postProcessParams(),
   numServerThreadsEverSpawned(0),
   serverThreads(),
   maxBatchSize(maxBatchSz),
   m_numRowsProcessed(0),
   m_numBatchesProcessed(0),
   m_numCacheHits(0),
   bufferMutex(),
   isKilled(false),
   numServerThreadsStartingUp(0),
   mainThreadWaitingForSpawn(),
   numOngoingEvals(0),
   numWaitingEvals(0),
   numEvalsToAwaken(0),
   waitingForFinish(),
   currentDoRandomize(doRandomize),
   currentDefaultSymmetry(defaultSymmetry),
   currentBatchSize(maxBatchSz),
   queryQueue()
{
  if(nnXLen > NNPos::MAX_BOARD_LEN)
    throw StringError("Maximum supported nnEval board size is " + Global::intToString(NNPos::MAX_BOARD_LEN));
  if(nnYLen > NNPos::MAX_BOARD_LEN)
    throw StringError("Maximum supported nnEval board size is " + Global::intToString(NNPos::MAX_BOARD_LEN));
  if(maxBatchSize <= 0)
    throw StringError("maxBatchSize is negative: " + Global::intToString(maxBatchSize));
  if(gpuIdxByServerThread.size() != numThreads)
    throw StringError("gpuIdxByServerThread.size() != numThreads");

  if(logger != NULL) {
    logger->write(
      "Initializing neural net buffer to be size " +
      Global::intToString(nnXLen) + " * " + Global::intToString(nnYLen) +
      (requireExactNNLen ? " exactly" : " allowing smaller boards")
    );
  }

  // A configured cache directory IS the decision to carry a level-0 resolution list, so the
  // two branches here are the whole of that decision and nothing downstream re-decides it.
  // Without one, this is byte for byte the table this evaluator always built.
  if(nnCacheConfig.sizePowerOfTwo >= 0) {
    if(nnCacheConfig.cacheDirectory.has_value()) {
      std::unique_ptr<NNCacheTwoLevelTable> twoLevel = NNCacheTable::createWithLevelZeroList(nnCacheConfig);
      nnCacheLevelZeroTable = twoLevel.get();
      nnCacheTable = std::move(twoLevel);
    }
    else {
      nnCacheTable = NNCacheTable::create(nnCacheConfig);
    }
  }

  if(!debugSkipNeuralNet) {
    vector<int> gpuIdxs = gpuIdxByServerThread;
    std::sort(gpuIdxs.begin(), gpuIdxs.end());
    auto last = std::unique(gpuIdxs.begin(), gpuIdxs.end());
    gpuIdxs.erase(last,gpuIdxs.end());
    loadedModel = NeuralNet::loadModelFile(modelFileName,expectedSha256);
    const ModelDesc& desc = NeuralNet::getModelDesc(loadedModel);
    internalModelName = desc.name;
    modelVersion = desc.modelVersion;
    inputsVersion = NNModelVersion::getInputsVersion(modelVersion);
    numInputMetaChannels = desc.numInputMetaChannels;
    computeContext = NeuralNet::createComputeContext(
      gpuIdxs,logger,nnXLen,nnYLen,
      homeDataDirOverride,
      usingFP16Mode,loadedModel,cfg
    );
    // Snapshot postProcessParams only after createComputeContext: backends may apply
    // config-dependent transforms to the model desc there (e.g. the ONNX backend's
    // scale8 workaround multiplies outputScaleMultiplier by 8).
    postProcessParams = desc.postProcessParams;
  }
  else {
    internalModelName = "random";
    modelVersion = NNModelVersion::defaultModelVersion;
    inputsVersion = NNModelVersion::getInputsVersion(modelVersion);
  }

  // Reserve a decent amount above the batch size so that allocation is unlikely.
  queryQueue.reserve(maxBatchSize * 4 * gpuIdxByServerThread.size());
  // Starts readonly. Becomes writable once we spawn server threads
  queryQueue.setReadOnly();
}

NNEvaluator::~NNEvaluator() {
  killServerThreads();

  if(computeContext != NULL)
    NeuralNet::freeComputeContext(computeContext);
  computeContext = NULL;

  if(loadedModel != NULL)
    NeuralNet::freeLoadedModel(loadedModel);
  loadedModel = NULL;

  nnCacheLevelZeroTable = nullptr;
  nnCacheTable.reset();
}

string NNEvaluator::getModelName() const {
  return modelName;
}
string NNEvaluator::getModelFileName() const {
  return modelFileName;
}
string NNEvaluator::getInternalModelName() const {
  return internalModelName;
}

static bool tryAbbreviateStepString(const string& input, string& buf) {
  size_t i = 0;
  while(i < input.length() && !Global::isDigit(input[i]))
    i++;
  if(i > 1)
    return false;

  string prefix = input.substr(0, i);
  int64_t number;
  bool suc = Global::tryStringToInt64(input.substr(i),number);
  if(!suc)
    return false;

  if(number >= 10000000000LL)
    buf = prefix + std::to_string(number / 1000000000LL) + "G";
  if(number >= 10000000)
    buf = prefix + std::to_string(number / 1000000) + "M";
  else if(number >= 10000)
    buf = prefix + std::to_string(number / 1000) + "K";
  else
    buf = input;
  return true;
}

string NNEvaluator::getAbbrevInternalModelName() const {
  string name = getInternalModelName();
  std::vector<string> pieces = Global::split(name,'-');
  std::vector<string> newPieces;
  for(const string& piece: pieces) {
    string buf;
    if(piece == "kata1") {
      // skip
    }
    else if(piece.size() > 1 && piece[0] == 's' && tryAbbreviateStepString(piece,buf)) {
      newPieces.push_back(buf);
    }
    else if(piece.size() > 1 && piece[0] == 'd' && tryAbbreviateStepString(piece,buf)) {
      // skip
    }
    else {
      newPieces.push_back(piece);
    }
  }
  return Global::concat(newPieces,"-");
}

Logger* NNEvaluator::getLogger() {
  return logger;
}
bool NNEvaluator::isNeuralNetLess() const {
  return debugSkipNeuralNet;
}
int NNEvaluator::getMaxBatchSize() const {
  return maxBatchSize;
}
int NNEvaluator::getCurrentBatchSize() const {
  return currentBatchSize.load(std::memory_order_acquire);
}
void NNEvaluator::setCurrentBatchSize(int batchSize) {
  if(batchSize <= 0 || batchSize > maxBatchSize)
    throw StringError("Invalid setting for batch size");
  currentBatchSize.store(batchSize,std::memory_order_release);
}
bool NNEvaluator::requiresSGFMetadata() const {
  return numInputMetaChannels > 0;
}

int NNEvaluator::getNumGpus() const {
#ifdef USE_EIGEN_BACKEND
  return 1;
#else
  std::set<int> gpuIdxs;
  for(int i = 0; i<gpuIdxByServerThread.size(); i++) {
    gpuIdxs.insert(gpuIdxByServerThread[i]);
  }
  return (int)gpuIdxs.size();
#endif
}
int NNEvaluator::getNumServerThreads() const {
  return (int)gpuIdxByServerThread.size();
}
std::set<int> NNEvaluator::getGpuIdxs() const {
  std::set<int> gpuIdxs;
#ifdef USE_EIGEN_BACKEND
  gpuIdxs.insert(0);
#else
  for(int i = 0; i<gpuIdxByServerThread.size(); i++) {
    gpuIdxs.insert(gpuIdxByServerThread[i]);
  }
#endif
  return gpuIdxs;
}

int NNEvaluator::getNNXLen() const {
  return nnXLen;
}
int NNEvaluator::getNNYLen() const {
  return nnYLen;
}
bool NNEvaluator::getRequireExactNNLen() const {
  return requireExactNNLen;
}
int NNEvaluator::getModelVersion() const {
  return modelVersion;
}
double NNEvaluator::getTrunkSpatialConvDepth() const {
  return NeuralNet::getModelDesc(loadedModel).getTrunkSpatialConvDepth();
}

int64_t NNEvaluator::getNumModelParameters() const {
  return NeuralNet::getModelDesc(loadedModel).getNumParameters();
}

bool NNEvaluator::modelHasAnyTransformerBlocks() const {
  return NeuralNet::getModelDesc(loadedModel).hasAnyTransformerBlocks();
}

bool NNEvaluator::modelHasAnyNestedBottleneckBlocks() const {
  return NeuralNet::getModelDesc(loadedModel).hasAnyNestedBottleneckBlocks();
}

enabled_t NNEvaluator::getUsingFP16Mode() const {
  return usingFP16Mode;
}

bool NNEvaluator::supportsShorttermError() const {
  return modelVersion >= 9;
}

bool NNEvaluator::modelPreferPassAliveUnderSuicideRules() const {
  if(loadedModel == NULL)
    return false;
  return NeuralNet::getModelDesc(loadedModel).preferPassAliveUnderSuicideRules;
}

bool NNEvaluator::modelPreferExcludeTerritoryAdjacentToAtari() const {
  if(loadedModel == NULL)
    return false;
  return NeuralNet::getModelDesc(loadedModel).preferExcludeTerritoryAdjacentToAtari;
}

bool NNEvaluator::getDoRandomize() const {
  return currentDoRandomize.load(std::memory_order_acquire);
}
int NNEvaluator::getDefaultSymmetry() const {
  return currentDefaultSymmetry.load(std::memory_order_acquire);
}
void NNEvaluator::setDoRandomize(bool b) {
  currentDoRandomize.store(b, std::memory_order_release);
}
void NNEvaluator::setDefaultSymmetry(int s) {
  currentDefaultSymmetry.store(s, std::memory_order_release);
}

Rules NNEvaluator::getSupportedRules(const Rules& desiredRules, bool& supported) const {
  if(loadedModel == NULL) {
    supported = true;
    return desiredRules;
  }
  return NeuralNet::getModelDesc(loadedModel).getSupportedRules(desiredRules, supported);
}

uint64_t NNEvaluator::numRowsProcessed() const {
  return m_numRowsProcessed.load(std::memory_order_relaxed);
}
uint64_t NNEvaluator::numBatchesProcessed() const {
  return m_numBatchesProcessed.load(std::memory_order_relaxed);
}
double NNEvaluator::averageProcessedBatchSize() const {
  return (double)numRowsProcessed() / (double)numBatchesProcessed();
}
uint64_t NNEvaluator::numCacheHits() const {
  return m_numCacheHits.load(std::memory_order_relaxed);
}

void NNEvaluator::clearStats() {
  m_numRowsProcessed.store(0);
  m_numBatchesProcessed.store(0);
  m_numCacheHits.store(0);
}

void NNEvaluator::clearCache() {
  if(nnCacheTable != nullptr)
    nnCacheTable->clear();
}

NNCacheContextId NNEvaluator::attachCacheContext(const string& name) {
  if(nnCacheTable == nullptr)
    throw StringError(
      "NNEvaluator: model '" + modelName + "' has no NN cache configured, so there is nothing "
      "for context '" + name + "' to be attached to and nothing it could ever be attributed. "
      "Attaching it would report success for a registration that can never be spent."
    );
  return nnCacheTable->attachCacheContext(name);
}

NNCacheContextResolution NNEvaluator::resolveCacheContext(const std::optional<string>& requested) const {
  if(nnCacheTable == nullptr) {
    if(!requested.has_value())
      return NNCacheContextResolution::resolved(NNCacheAttribution::noAttributableContext());
    return NNCacheContextResolution::refused(
      "Unknown cacheContext '" + requested.value() + "'. Model '" + modelName +
      "' has no NN cache configured, so no context is attached to it."
    );
  }
  return nnCacheTable->cacheContexts().resolveForRequest(requested);
}

NNCacheAttributionLedger NNEvaluator::harvestCacheAttribution() const {
  if(nnCacheTable == nullptr)
    return NNCacheAttributionLedger::notAttributed();
  return nnCacheTable->harvestAttribution();
}

NNCacheHitLedger NNEvaluator::harvestCacheHitCountsFor(const NNCacheContextId& context) const {
  if(nnCacheTable == nullptr)
    throw StringError(
      "NNEvaluator: model '" + modelName + "' has no NN cache configured, so no context is "
      "attached to it and none has earned anything here."
    );
  return nnCacheTable->harvestHitCountsFor(context);
}

const std::optional<std::string>& NNEvaluator::getCacheDirectory() const {
  return nnCacheDirectory;
}

NNCacheTable& NNEvaluator::cacheTable() const {
  if(nnCacheTable == nullptr)
    throw StringError(
      "NNEvaluator: model '" + modelName + "' has no NN cache configured (nnCacheSizePowerOfTwo "
      "is negative), so there is no cache table to read or persist."
    );
  return *nnCacheTable;
}

// The one home of "this evaluator was built with a level-0 resolution list", so the three
// surfaces below refuse in the same words and a fourth cannot drift from them.
NNCacheTwoLevelTable& NNEvaluator::levelZeroTableOrThrow() const {
  if(nnCacheLevelZeroTable == nullptr)
    throw StringError(
      "NNEvaluator: model '" + modelName + "' has no persisted cache: '" +
      string(NNCacheConfig::KEY_DIR) + "' is not set in its config, so its cache was built "
      "without a level-0 resolution list and there is nothing to attach a context to."
    );
  return *nnCacheLevelZeroTable;
}

// The debug tripwire both acts carry. It observes the property the contract is about -- an
// evaluation is inside evaluate(), and therefore may be inside the lock-free walk of the very
// vector these acts mutate -- at the moment of the act, rather than a symptom of it later
// (ADR-0021 Rule 1). Compiled out of a release build entirely.
void NNEvaluator::assertNoEvaluationInFlightForLevelZeroSwap(const char* act) const {
#ifndef NDEBUG
  const int inFlight = nnCacheEvaluationsInFlight.load(std::memory_order_acquire);
  if(inFlight != 0)
    Global::fatalError(
      "NNEvaluator: " + string(act) + " on model '" + modelName + "' while " +
      Global::intToString(inFlight) + " evaluation(s) are in flight. A get walks the level-0 "
      "resolution list lock-free and this mutates the vector it walks, so this is a "
      "use-after-free, not a torn counter. The analysis engine's protocol layer refuses "
      "cache_attach and cache_detach while any request is open; this caller reached past it."
    );
#else
  (void)act;
#endif
}

NNCacheLevelZeroAttachment NNEvaluator::attachLevelZeroSource(
  std::unique_ptr<NNCacheFrozen> source, const NNCacheContextId& servesContext
) {
  NNCacheTwoLevelTable& table = levelZeroTableOrThrow();
  assertNoEvaluationInFlightForLevelZeroSwap("attachLevelZeroSource");
  // REQUIRED HERE, THOUGH THE TABLE ACCEPTS A SOURCE WITHOUT ONE. This is the protocol's door,
  // and every source that comes through it was loaded from some context's container on that
  // context's behalf -- so an attach through here that named no context would be a source whose
  // retrievals no per-context dump could ever write, which is a silent loss with a client on the
  // other end of it. The table's own door stays permissive because a table can legitimately be
  // handed a source before any context exists: its own construction does exactly that.
  return table.attachLevelZero(std::move(source), std::optional<NNCacheContextId>(servesContext));
}

std::unique_ptr<NNCacheFrozen> NNEvaluator::detachLevelZeroSource(const NNCacheLevelZeroSourceId& id) {
  NNCacheTwoLevelTable& table = levelZeroTableOrThrow();
  assertNoEvaluationInFlightForLevelZeroSwap("detachLevelZeroSource");
  return table.detachLevelZero(id);
}

size_t NNEvaluator::numLevelZeroSources() const {
  return levelZeroTableOrThrow().numLevelZeroSources();
}


bool NNEvaluator::isAnyThreadUsingFP16() const {
  lock_guard<std::mutex> lock(bufferMutex);
  for(const int& isUsingFP16: serverThreadsIsUsingFP16) {
    if(isUsingFP16)
      return true;
  }
  return false;
}

static void serveEvals(
  string randSeedThisThread,
  NNEvaluator* nnEval, const LoadedModel* loadedModel,
  int gpuIdxForThisThread,
  int serverThreadIdx
) {
  NNServerBuf* buf = new NNServerBuf(*nnEval,loadedModel);
  Rand rand(randSeedThisThread);

  // Used to have a try catch around this but actually we're in big trouble if this raises an exception
  // and causes possibly the only nnEval thread to die, so actually go ahead and let the exception escape to
  // toplevel for easier debugging
  nnEval->serve(*buf,rand,gpuIdxForThisThread,serverThreadIdx);
  delete buf;
}

void NNEvaluator::setNumThreads(const vector<int>& gpuIdxByServerThr) {
  if(serverThreads.size() != 0)
    throw StringError("NNEvaluator::setNumThreads called when threads were already running!");
  numThreads = (int)gpuIdxByServerThr.size();
  gpuIdxByServerThread = gpuIdxByServerThr;
}

void NNEvaluator::spawnServerThreads() {
  if(serverThreads.size() != 0)
    throw StringError("NNEvaluator::spawnServerThreads called when threads were already running!");

  {
    lock_guard<std::mutex> lock(bufferMutex);
    serverThreadsIsUsingFP16.resize(numThreads,0);
  }

  queryQueue.unsetReadOnly();

  numServerThreadsStartingUp = numThreads;
  for(int i = 0; i<numThreads; i++) {
    int gpuIdxForThisThread = gpuIdxByServerThread[i];
    string randSeedThisThread = randSeed + ":NNEvalServerThread:" + Global::intToString(numServerThreadsEverSpawned);
    numServerThreadsEverSpawned++;
    std::thread* thread = new std::thread(
      &serveEvals,randSeedThisThread,this,loadedModel,gpuIdxForThisThread,i
    );
    serverThreads.push_back(thread);
  }

  unique_lock<std::mutex> lock(bufferMutex);
  while(numServerThreadsStartingUp > 0)
    mainThreadWaitingForSpawn.wait(lock);
}

void NNEvaluator::killServerThreads() {
  unique_lock<std::mutex> lock(bufferMutex);
  isKilled = true;
  lock.unlock();
  queryQueue.setReadOnly();

  waitingForFinish.notify_all();

  for(size_t i = 0; i<serverThreads.size(); i++)
    serverThreads[i]->join();
  for(size_t i = 0; i<serverThreads.size(); i++)
    delete serverThreads[i];
  serverThreads.clear();
  serverThreadsIsUsingFP16.clear();

  // Can unset now that threads are dead
  isKilled = false;

  testAssert(numOngoingEvals == 0);
  testAssert(numWaitingEvals == 0);
  testAssert(numEvalsToAwaken == 0);
}

void NNEvaluator::fillRowBufs(
  const Board& board,
  const BoardHistory& history,
  Player nextPlayer,
  const SGFMetadata* sgfMeta,
  const MiscNNInputParams& nnInputParams,
  NNResultBuf& buf
) const {
  const int rowSpatialLen = NNModelVersion::getNumSpatialFeatures(modelVersion) * nnXLen * nnYLen;
  if(buf.rowSpatialBuf.size() < rowSpatialLen)
    buf.rowSpatialBuf.resize(rowSpatialLen);
  const int rowGlobalLen = NNModelVersion::getNumGlobalFeatures(modelVersion);
  if(buf.rowGlobalBuf.size() < rowGlobalLen)
    buf.rowGlobalBuf.resize(rowGlobalLen);
  const int rowMetaLen = numInputMetaChannels;
  if(buf.rowMetaBuf.size() < rowMetaLen)
    buf.rowMetaBuf.resize(rowMetaLen);

  static_assert(NNModelVersion::latestInputsVersionImplemented == 7, "");
  if(inputsVersion == 3)
    NNInputs::fillRowV3(board, history, nextPlayer, nnInputParams, nnXLen, nnYLen, inputsUseNHWC, buf.rowSpatialBuf.data(), buf.rowGlobalBuf.data());
  else if(inputsVersion == 4)
    NNInputs::fillRowV4(board, history, nextPlayer, nnInputParams, nnXLen, nnYLen, inputsUseNHWC, buf.rowSpatialBuf.data(), buf.rowGlobalBuf.data());
  else if(inputsVersion == 5)
    NNInputs::fillRowV5(board, history, nextPlayer, nnInputParams, nnXLen, nnYLen, inputsUseNHWC, buf.rowSpatialBuf.data(), buf.rowGlobalBuf.data());
  else if(inputsVersion == 6)
    NNInputs::fillRowV6(board, history, nextPlayer, nnInputParams, nnXLen, nnYLen, inputsUseNHWC, buf.rowSpatialBuf.data(), buf.rowGlobalBuf.data());
  else if(inputsVersion == 7)
    NNInputs::fillRowV7(board, history, nextPlayer, nnInputParams, nnXLen, nnYLen, inputsUseNHWC, buf.rowSpatialBuf.data(), buf.rowGlobalBuf.data());
  else
    ASSERT_UNREACHABLE;

  if(rowMetaLen > 0) {
    if(sgfMeta == NULL)
      Global::fatalError("SGFMetadata is required for " + modelName + " but was not provided");
    if(!sgfMeta->initialized)
      Global::fatalError("SGFMetadata is required for " + modelName + " but was not initialized. Did you specify humanSLProfile=... in katago's config or via overrides?");
    SGFMetadata::fillMetadataRow(
      sgfMeta,
      buf.rowMetaBuf.data(),
      nextPlayer,
      board.x_size*board.y_size
    );
    buf.hasRowMeta = true;
  }
  else {
    buf.hasRowMeta = false;
  }
}

void NNEvaluator::maybeWarmupComputeHandle(ComputeHandle* gpuHandle, int serverThreadIdx) {
  if(disableWarmup || gpuHandle == NULL || debugSkipNeuralNet || loadedModel == NULL)
    return;
  // Warmup currently only matters on CUDA, where cuDNN lazily compiles an SDPA execution plan per
  // batch size on first use. Other backends: nothing to warm up for now.
#if !defined(USE_CUDA_BACKEND)
  (void)serverThreadIdx;
  return;
#else
  // Only transformer models build the lazy SDPA graphs; skip the (otherwise harmless but wasteful)
  // warmup passes for plain convnets.
  if(!NeuralNet::getModelDesc(loadedModel).hasAnyTransformerBlocks())
    return;

  if(logger != NULL) {
    logger->write(
      "Cuda backend thread " + Global::intToString(serverThreadIdx) +
      ": warming up transformer graphs for batch sizes 1.." + Global::intToString(maxBatchSize)
    );
  }

  // Empty board of the configured size, default rules/params. Outputs are discarded; we only want
  // the forward passes to trigger graph compilation for every batch size that will be seen.
  Board board(nnXLen, nnYLen);
  //Featurize the way this model expects (a no-op under Tromp-Taylorish rules, but robust if the
  //warmup rules ever change).
  BoardHistory history(
    board, P_BLACK, Rules::getTrompTaylorish(), 0,
    BoardHistoryModes(modelPreferPassAliveUnderSuicideRules(), modelPreferExcludeTerritoryAdjacentToAtari())
  );
  MiscNNInputParams nnInputParams;
  SGFMetadata sgfMeta;
  const SGFMetadata* sgfMetaPtr = NULL;
  if(numInputMetaChannels > 0) {
    sgfMeta = SGFMetadata::makeDummyWarmupProfile();
    sgfMetaPtr = &sgfMeta;
  }

  // Mark the handle as warming up so the backend treats lazy-graph-compilation failures (e.g. cudnn
  // SDPA) leniently, falling back to a custom kernel instead of failing hard. Restored when done.
  bool prevIsWarmup = NeuralNet::setIsWarmup(gpuHandle, true);

  InputBuffers* inputBuffers = NeuralNet::createInputBuffers(loadedModel, maxBatchSize, nnXLen, nnYLen);

  // Reusable per-row input; identical for every row since it's an empty board.
  std::vector<std::unique_ptr<NNResultBuf>> ownedBufs;
  std::vector<NNResultBuf*> resultBufs;
  ownedBufs.reserve(maxBatchSize);
  resultBufs.reserve(maxBatchSize);
  for(int i = 0; i < maxBatchSize; i++) {
    ownedBufs.push_back(std::make_unique<NNResultBuf>());
    NNResultBuf* buf = ownedBufs.back().get();
    fillRowBufs(board, history, P_BLACK, sgfMetaPtr, nnInputParams, *buf);
    buf->symmetry = 0;
    buf->policyOptimism = nnInputParams.policyOptimism;
    resultBufs.push_back(buf);
  }

  for(int batchSize = 1; batchSize <= maxBatchSize; batchSize++) {
    std::vector<NNOutput*> outputs;
    outputs.reserve(batchSize);
    for(int row = 0; row < batchSize; row++) {
      NNOutput* out = new NNOutput();
      out->nnXLen = nnXLen;
      out->nnYLen = nnYLen;
      out->whiteOwnerMap = NULL;
      outputs.push_back(out);
    }
    NeuralNet::getOutput(gpuHandle, inputBuffers, batchSize, resultBufs.data(), outputs);
    for(NNOutput* out : outputs)
      delete out;
  }

  NeuralNet::freeInputBuffers(inputBuffers);
  NeuralNet::setIsWarmup(gpuHandle, prevIsWarmup);
#endif
}

void NNEvaluator::serve(
  NNServerBuf& buf, Rand& rand,
  int gpuIdxForThisThread,
  int serverThreadIdx
) {
  int64_t numBatchesHandledThisThread = 0;
  int64_t numRowsHandledThisThread = 0;

  ComputeHandle* gpuHandle = NULL;
  if(loadedModel != NULL) {
    gpuHandle = NeuralNet::createComputeHandle(
      computeContext,
      loadedModel,
      logger,
      maxBatchSize,
      requireExactNNLen,
      inputsUseNHWC,
      gpuIdxForThisThread,
      serverThreadIdx
    );

    // Warm up lazily-compiled backend graphs before reporting this thread as started.
    maybeWarmupComputeHandle(gpuHandle, serverThreadIdx);
  }

  {
    lock_guard<std::mutex> lock(bufferMutex);
    testAssert(serverThreadIdx < serverThreadsIsUsingFP16.size());
    serverThreadsIsUsingFP16[serverThreadIdx] = gpuHandle == NULL ? 0 : NeuralNet::isUsingFP16(gpuHandle) ? 1 : 0;
    numServerThreadsStartingUp--;
    if(numServerThreadsStartingUp <= 0)
      mainThreadWaitingForSpawn.notify_all();
  }

  vector<NNResultBuf*> resultBufs;
  resultBufs.reserve(maxBatchSize);

  vector<NNOutput*> outputBuf;

  unique_lock<std::mutex> lock(bufferMutex,std::defer_lock);
  while(true) {
    resultBufs.clear();
    int desiredBatchSize = std::min(maxBatchSize, currentBatchSize.load(std::memory_order_acquire));
    bool gotAnything = queryQueue.waitPopUpToN(resultBufs,desiredBatchSize);
    // Queue being closed is a signal that we're done.
    if(!gotAnything)
      break;

    int numRows = (int)resultBufs.size();
    testAssert(numRows > 0);

    bool doRandomize = currentDoRandomize.load(std::memory_order_acquire);
    int defaultSymmetry = currentDefaultSymmetry.load(std::memory_order_acquire);

    if(debugSkipNeuralNet) {
      for(int row = 0; row < numRows; row++) {
        testAssert(resultBufs[row] != NULL);
        NNResultBuf* resultBuf = resultBufs[row];
        resultBufs[row] = NULL;

        int boardXSize = resultBuf->boardXSizeForServer;
        int boardYSize = resultBuf->boardYSizeForServer;

        unique_lock<std::mutex> resultLock(resultBuf->resultMutex);
        testAssert(resultBuf->hasResult == false);
        resultBuf->result = std::make_shared<NNOutput>();

        float* policyProbs = resultBuf->result->policyProbs;
        for(int i = 0; i<NNPos::MAX_NN_POLICY_SIZE; i++)
          policyProbs[i] = 0;

        // At this point, these aren't probabilities, since this is before the postprocessing
        // that happens for each result. These just need to be unnormalized log probabilities.
        // Illegal move filtering happens later.
        for(int y = 0; y<boardYSize; y++) {
          for(int x = 0; x<boardXSize; x++) {
            int pos = NNPos::xyToPos(x,y,nnXLen);
            policyProbs[pos] = (float)rand.nextGaussian();
          }
        }
        policyProbs[NNPos::locToPos(Board::PASS_LOC,boardXSize,nnXLen,nnYLen)] = (float)rand.nextGaussian();

        resultBuf->result->nnXLen = nnXLen;
        resultBuf->result->nnYLen = nnYLen;
        if(resultBuf->includeOwnerMap) {
          float* whiteOwnerMap = new float[nnXLen*nnYLen];
          for(int i = 0; i<nnXLen*nnYLen; i++)
            whiteOwnerMap[i] = 0.0;
          for(int y = 0; y<boardYSize; y++) {
            for(int x = 0; x<boardXSize; x++) {
              int pos = NNPos::xyToPos(x,y,nnXLen);
              whiteOwnerMap[pos] = (float)rand.nextGaussian() * 0.20f;
            }
          }
          resultBuf->result->whiteOwnerMap = whiteOwnerMap;
        }
        else {
          resultBuf->result->whiteOwnerMap = NULL;
        }

        // These aren't really probabilities. Win/Loss/NoResult will get softmaxed later
        double whiteWinProb = 0.0 + rand.nextGaussian() * 0.20;
        double whiteLossProb = 0.0 + rand.nextGaussian() * 0.20;
        double whiteScoreMean = 0.0 + rand.nextGaussian() * 0.20;
        double whiteScoreMeanSq = 0.0 + rand.nextGaussian() * 0.20;
        double whiteNoResultProb = 0.0 + rand.nextGaussian() * 0.20;
        double varTimeLeft = 0.5 * boardXSize * boardYSize;
        resultBuf->result->whiteWinProb = (float)whiteWinProb;
        resultBuf->result->whiteLossProb = (float)whiteLossProb;
        resultBuf->result->whiteNoResultProb = (float)whiteNoResultProb;
        resultBuf->result->whiteScoreMean = (float)whiteScoreMean;
        resultBuf->result->whiteScoreMeanSq = (float)whiteScoreMeanSq;
        resultBuf->result->whiteLead = (float)whiteScoreMean;
        resultBuf->result->varTimeLeft = (float)varTimeLeft;
        resultBuf->result->shorttermWinlossError = 0.0f;
        resultBuf->result->shorttermScoreError = 0.0f;
        resultBuf->result->policyOptimismUsed = (float)resultBuf->policyOptimism;
        resultBuf->hasResult = true;
        resultBuf->clientWaitingForResult.notify_all();
        resultLock.unlock();
      }
    }
    else {
      outputBuf.clear();
      for(int row = 0; row<numRows; row++) {
        NNOutput* emptyOutput = new NNOutput();
        testAssert(resultBufs[row] != NULL);
        emptyOutput->nnXLen = nnXLen;
        emptyOutput->nnYLen = nnYLen;
        if(resultBufs[row]->includeOwnerMap)
          emptyOutput->whiteOwnerMap = new float[nnXLen*nnYLen];
        else
          emptyOutput->whiteOwnerMap = NULL;
        outputBuf.push_back(emptyOutput);
      }

      for(int row = 0; row<numRows; row++) {
        if(resultBufs[row]->symmetry == NNInputs::SYMMETRY_NOTSPECIFIED) {
          if(doRandomize)
            resultBufs[row]->symmetry = rand.nextUInt(SymmetryHelpers::NUM_SYMMETRIES);
          else {
            testAssert(defaultSymmetry >= 0 && defaultSymmetry <= SymmetryHelpers::NUM_SYMMETRIES-1);
            resultBufs[row]->symmetry = defaultSymmetry;
          }
        }
      }

      NeuralNet::getOutput(gpuHandle, buf.inputBuffers, numRows, resultBufs.data(), outputBuf);
      testAssert(outputBuf.size() == numRows);

      m_numRowsProcessed.fetch_add(numRows, std::memory_order_relaxed);
      m_numBatchesProcessed.fetch_add(1, std::memory_order_relaxed);
      numRowsHandledThisThread += numRows;
      numBatchesHandledThisThread += 1;

      for(int row = 0; row < numRows; row++) {
        testAssert(resultBufs[row] != NULL);
        NNResultBuf* resultBuf = resultBufs[row];
        resultBufs[row] = NULL;

        unique_lock<std::mutex> resultLock(resultBuf->resultMutex);
        testAssert(resultBuf->hasResult == false);
        resultBuf->result = std::shared_ptr<NNOutput>(outputBuf[row]);
        resultBuf->hasResult = true;
        resultBuf->clientWaitingForResult.notify_all();
        resultLock.unlock();
      }
    }

    // Lock and update stats before looping again
    lock.lock();
    numOngoingEvals -= numRows;

    if(numWaitingEvals > 0) {
      numEvalsToAwaken += numWaitingEvals;
      numWaitingEvals = 0;
      waitingForFinish.notify_all();
    }
    lock.unlock();
    continue;
  }

  NeuralNet::freeComputeHandle(gpuHandle);
  if(logger != NULL) {
    logger->write(
      "GPU " + Global::intToString(gpuIdxForThisThread) + " finishing, processed " +
      Global::int64ToString(numRowsHandledThisThread) + " rows " +
      Global::int64ToString(numBatchesHandledThisThread) + " batches"
    );
  }
}

void NNEvaluator::waitForNextNNEvalIfAny() {
  unique_lock<std::mutex> lock(bufferMutex);
  if(numOngoingEvals <= 0)
    return;

  numWaitingEvals++;
  while(numEvalsToAwaken <= 0 && !isKilled)
    waitingForFinish.wait(lock);
  numEvalsToAwaken--;
}


static double softPlus(double x) {
  // Avoid blowup
  if(x > 40.0)
    return x;
  else
    return log(1.0 + exp(x));
}

static const int daggerPattern[9][8] = {
  {0,0,0,0,0,0,0,0},
  {0,0,0,0,0,0,0,0},
  {0,0,2,1,0,0,0,0},
  {0,0,2,1,0,0,0,0},
  {0,0,0,0,0,0,0,0},
  {0,2,1,0,0,0,0,0},
  {0,3,0,0,0,0,0,0},
  {0,0,0,0,0,0,0,0},
  {0,0,0,0,0,0,0,0},
};
static bool daggerMatch(const Board& board, Player nextPla, Loc& banned, int symmetry) {
  for(int yi = 0; yi < 9; yi++) {
    for(int xi = 0; xi < 8; xi++) {
      int y = yi;
      int x = xi;
      if((symmetry & 0x1) != 0)
        std::swap(x,y);
      if((symmetry & 0x2) != 0)
        x = board.x_size-1-x;
      if((symmetry & 0x4) != 0)
        y = board.y_size-1-y;
      Loc loc = Location::getLoc(x,y,board.x_size);
      int m = daggerPattern[yi][xi];
      if(m == 0 && board.colors[loc] != C_EMPTY)
        return false;
      if(m == 1 && board.colors[loc] != nextPla)
        return false;
      if(m == 2 && board.colors[loc] != getOpp(nextPla))
        return false;
      if(m == 3)
        banned = loc;
    }
  }
  return true;
}

std::shared_ptr<NNOutput>* NNEvaluator::averageMultipleSymmetries(
  const Board& board,
  const BoardHistory& history,
  Player nextPlayer,
  const SGFMetadata* sgfMeta,
  const MiscNNInputParams& baseNNInputParams,
  NNResultBuf& buf,
  bool includeOwnerMap,
  Rand& rand,
  int numSymmetriesToSample
) {
  MiscNNInputParams nnInputParams = baseNNInputParams;
  vector<std::shared_ptr<NNOutput>> ptrs;
  std::array<int, SymmetryHelpers::NUM_SYMMETRIES> symmetryIndexes;
  std::iota(symmetryIndexes.begin(), symmetryIndexes.end(), 0);
  // evaluate() CONSUMES the buffer's cache-context tag, so it has to be re-supplied for each
  // symmetry rather than set once by the caller. All of these evaluations are the same query's,
  // earned by the same context, and every one of them sets an entry (skipCache is on).
  const NNCacheAttribution attributionForEverySymmetry = buf.cacheAttribution;
  for(int i = 0; i<numSymmetriesToSample; i++) {
    std::swap(symmetryIndexes[i], symmetryIndexes[rand.nextInt(i,SymmetryHelpers::NUM_SYMMETRIES-1)]);
    nnInputParams.symmetry = symmetryIndexes[i];
    bool skipCacheThisIteration = true; // Skip cache since there's no guarantee which symmetry is in the cache
    buf.cacheAttribution = attributionForEverySymmetry;
    evaluate(
      board, history, nextPlayer, sgfMeta,
      nnInputParams,
      buf, skipCacheThisIteration, includeOwnerMap
    );
    ptrs.push_back(std::move(buf.result));
  }
  return new std::shared_ptr<NNOutput>(new NNOutput(ptrs));
}

void NNEvaluator::evaluate(
  const Board& board,
  const BoardHistory& history,
  Player nextPlayer,
  const MiscNNInputParams& nnInputParams,
  NNResultBuf& buf,
  bool skipCache,
  bool includeOwnerMap
) {
  evaluate(
    board,
    history,
    nextPlayer,
    NULL,
    nnInputParams,
    buf,
    skipCache,
    includeOwnerMap
  );
}

void NNEvaluator::evaluate(
  const Board& board,
  const BoardHistory& history,
  Player nextPlayer,
  const SGFMetadata* sgfMeta,
  const MiscNNInputParams& nnInputParamsArg,
  NNResultBuf& buf,
  bool skipCache,
  bool includeOwnerMap
) {
  testAssert(!isKilled);
#ifndef NDEBUG
  // See assertNoEvaluationInFlightForLevelZeroSwap. Debug builds only; the release build's
  // evaluation path is unchanged.
  const InFlightEvaluationMark inFlightMark(nnCacheEvaluationsInFlight);
#endif
  buf.hasResult = false;

  // THE CACHE-CONTEXT TAG IS CONSUMED HERE, not read at the end.
  //
  // NNResultBuf is allocated once per thread and reused for every evaluation that thread ever
  // makes, including this evaluator's and a companion evaluator's. A tag left standing on it
  // would be spent by whichever evaluation came next -- filing one query's earnings under the
  // context of the query before it, silently, since both are legal values. Taking it out of the
  // buffer at entry makes a stale tag unrepresentable rather than something every call site has
  // to remember not to leave behind: a caller that supplies one gets it spent exactly once, and
  // a caller that supplies none is unattributed, which is counted and reported.
  const NNCacheAttribution cacheAttribution = buf.cacheAttribution;
  buf.cacheAttribution = NNCacheAttribution::noAttributableContext();

  if(board.x_size > nnXLen || board.y_size > nnYLen)
    throw StringError("NNEvaluator was configured with nnXLen = " + Global::intToString(nnXLen) +
                      " nnYLen = " + Global::intToString(nnYLen) +
                      " but was asked to evaluate board with larger x or y size");
  if(requireExactNNLen) {
    if(board.x_size != nnXLen || board.y_size != nnYLen)
      throw StringError("NNEvaluator was configured with nnXLen = " + Global::intToString(nnXLen) +
                        " nnYLen = " + Global::intToString(nnYLen) +
                        " and requireExactNNLen, but was asked to evaluate board with different x or y size");
  }

  // Avoid using policy optimism for humanSL
  MiscNNInputParams nnInputParams = nnInputParamsArg;
  if(numInputMetaChannels > 0)
    nnInputParams.policyOptimism = 0.0;

  Hash128 nnHash = NNInputs::getHash(board, history, nextPlayer, nnInputParams);
  if(numInputMetaChannels > 0) {
    if(sgfMeta == NULL)
      Global::fatalError("SGFMetadata is required for " + modelName + " but was not provided");
    if(!sgfMeta->initialized)
      Global::fatalError("SGFMetadata is required for " + modelName + " but was not initialized. Did you specify humanSLProfile=... in katago's config or via overrides?");
    nnHash ^= sgfMeta->getHash(nextPlayer);
  }

  bool hadResultWithoutOwnerMap = false;
  shared_ptr<NNOutput> resultWithoutOwnerMap;
  if(nnCacheTable != nullptr && !skipCache && nnCacheTable->get(nnHash,buf.result)) {
    if(!(includeOwnerMap && buf.result->whiteOwnerMap == NULL))
    {
      m_numCacheHits.fetch_add(1, std::memory_order_relaxed);
      buf.hasResult = true;
      return;
    }
    else {
      hadResultWithoutOwnerMap = true;
      resultWithoutOwnerMap = std::move(buf.result);
      buf.result = nullptr;
    }
  }
  buf.includeOwnerMap = includeOwnerMap;

  buf.boardXSizeForServer = board.x_size;
  buf.boardYSizeForServer = board.y_size;

  if(!debugSkipNeuralNet) {
    fillRowBufs(board, history, nextPlayer, sgfMeta, nnInputParams, buf);
  }

  buf.symmetry = nnInputParams.symmetry;
  buf.policyOptimism = nnInputParams.policyOptimism;

  unique_lock<std::mutex> lock(bufferMutex);
  numOngoingEvals += 1;
  lock.unlock();

  bool suc = queryQueue.forcePush(&buf);
  testAssert(suc);

  unique_lock<std::mutex> resultLock(buf.resultMutex);
  while(!buf.hasResult)
    buf.clientWaitingForResult.wait(resultLock);
  resultLock.unlock();

  // Perform postprocessing on the result - turn the nn output into probabilities
  // As a hack though, if the only thing we were missing was the ownermap, just grab the old policy and values
  // and use those. This avoids recomputing in a randomly different orientation when we just need the ownermap
  // and causing policy weights to be different, which would reduce performance of successive searches in a game
  // by making the successive searches distribute their playouts less coherently and using the cache more poorly.
  if(hadResultWithoutOwnerMap) {
    buf.result->whiteWinProb = resultWithoutOwnerMap->whiteWinProb;
    buf.result->whiteLossProb = resultWithoutOwnerMap->whiteLossProb;
    buf.result->whiteNoResultProb = resultWithoutOwnerMap->whiteNoResultProb;
    buf.result->whiteScoreMean = resultWithoutOwnerMap->whiteScoreMean;
    buf.result->whiteScoreMeanSq = resultWithoutOwnerMap->whiteScoreMeanSq;
    buf.result->whiteLead = resultWithoutOwnerMap->whiteLead;
    buf.result->varTimeLeft = resultWithoutOwnerMap->varTimeLeft;
    buf.result->shorttermWinlossError = resultWithoutOwnerMap->shorttermWinlossError;
    buf.result->shorttermScoreError = resultWithoutOwnerMap->shorttermScoreError;
    std::copy(resultWithoutOwnerMap->policyProbs, resultWithoutOwnerMap->policyProbs + NNPos::MAX_NN_POLICY_SIZE, buf.result->policyProbs);
    buf.result->policyOptimismUsed = (float)resultWithoutOwnerMap->policyOptimismUsed;
    buf.result->nnXLen = resultWithoutOwnerMap->nnXLen;
    buf.result->nnYLen = resultWithoutOwnerMap->nnYLen;
    testAssert(buf.result->whiteOwnerMap != NULL);
  }
  else {
    float* policy = buf.result->policyProbs;

    float policyOutputScaling = postProcessParams.outputScaleMultiplier / nnInputParams.nnPolicyTemperature;

    int xSize = board.x_size;
    int ySize = board.y_size;

    float maxPolicy = -1e25f;
    bool isLegal[NNPos::MAX_NN_POLICY_SIZE];
    int legalCount = 0;
    testAssert(nextPlayer == history.presumedNextMovePla);
    for(int i = 0; i<policySize; i++) {
      Loc loc = NNPos::posToLoc(i,xSize,ySize,nnXLen,nnYLen);
      isLegal[i] = history.isLegal(board,loc,nextPlayer);
    }

    if(nnInputParams.avoidMYTDaggerHack && xSize >= 13 && ySize >= 13) {
      for(int symmetry = 0; symmetry < 8; symmetry++) {
        Loc banned = Board::NULL_LOC;
        if(daggerMatch(board, nextPlayer, banned, symmetry)) {
          if(banned != Board::NULL_LOC) {
            isLegal[NNPos::locToPos(banned,xSize,nnXLen,nnYLen)] = false;
          }
        }
      }
    }

    for(int i = 0; i<policySize; i++) {
      float policyValue;
      if(isLegal[i]) {
        legalCount += 1;
        policyValue = policy[i] * policyOutputScaling;
      }
      else
        policyValue = -1e30f;

      policy[i] = policyValue;
      if(policyValue > maxPolicy)
        maxPolicy = policyValue;
    }

    testAssert(legalCount > 0);

    float policySum = 0.0f;

    if(nnInputParams.enablePassingHacks) {
      //Cap passing prior policy at 95% (19x other moves)
      float maxPassPolicySumFactor = 19.0f;

      for(int i = 0; i<policySize-1; i++) {
        policy[i] = exp(policy[i] - maxPolicy);
        policySum += policy[i];
      }
      int passPos = NNPos::locToPos(Board::PASS_LOC, xSize, nnXLen, nnYLen);
      testAssert(passPos == policySize-1);
      int i = passPos;
      policy[i] = std::max(1e-20f, std::min(exp(policy[i] - maxPolicy), policySum * maxPassPolicySumFactor));
      policySum += policy[i];
    }
    else {
      for(int i = 0; i<policySize; i++) {
        policy[i] = exp(policy[i] - maxPolicy);
        policySum += policy[i];
      }
    }

    if(!isfinite(policySum)) {
      cout << "Got nonfinite for policy sum" << endl;
      history.printDebugInfo(cout,board);
      throw StringError("Got nonfinite for policy sum");
    }

    // Somehow all legal moves rounded to 0 probability
    if(policySum <= 0.0) {
      if(!buf.errorLogLockout && logger != NULL) {
        buf.errorLogLockout = true;
        logger->write("Warning: all legal moves rounded to 0 probability for " + string(modelFileName));
      }
      float uniform = 1.0f / legalCount;
      for(int i = 0; i<policySize; i++) {
        policy[i] = isLegal[i] ? uniform : -1.0f;
      }
    }
    // Normal case
    else {
      for(int i = 0; i<policySize; i++)
        policy[i] = isLegal[i] ? (policy[i] / policySum) : -1.0f;
    }

    // Fill everything out-of-bounds too, for robustness.
    for(int i = policySize; i<NNPos::MAX_NN_POLICY_SIZE; i++)
      policy[i] = -1.0f;

    buf.result->policyOptimismUsed = (float)nnInputParams.policyOptimism;

    // Fix up the value as well. Note that the neural net gives us back the value from the perspective
    // of the player so we need to negate that to make it the white value.
    if(modelVersion == 3) {
      const double twoOverPi = 0.63661977236758134308;

      double winProb;
      double lossProb;
      double noResultProb;
      // Version 3 neural nets just pack the pre-arctanned scoreValue into the whiteScoreMean field
      double scoreValue = atan(buf.result->whiteScoreMean * postProcessParams.outputScaleMultiplier) * twoOverPi;
      {
        double winLogits = buf.result->whiteWinProb * postProcessParams.outputScaleMultiplier;
        double lossLogits = buf.result->whiteLossProb * postProcessParams.outputScaleMultiplier;
        double noResultLogits = buf.result->whiteNoResultProb * postProcessParams.outputScaleMultiplier;

        // Softmax
        double maxLogits = std::max(std::max(winLogits,lossLogits),noResultLogits);
        winProb = exp(winLogits - maxLogits);
        lossProb = exp(lossLogits - maxLogits);
        noResultProb = exp(noResultLogits - maxLogits);

        double probSum = winProb + lossProb + noResultProb;
        winProb /= probSum;
        lossProb /= probSum;
        noResultProb /= probSum;

        if(!isfinite(probSum) || !isfinite(scoreValue)) {
          cout << "Got nonfinite for nneval value" << endl;
          cout << winLogits << " " << lossLogits << " " << noResultLogits << " " << scoreValue << endl;
          throw StringError("Got nonfinite for nneval value");
        }
      }

      if(nextPlayer == P_WHITE) {
        buf.result->whiteWinProb = (float)winProb;
        buf.result->whiteLossProb = (float)lossProb;
        buf.result->whiteNoResultProb = (float)noResultProb;
        buf.result->whiteScoreMean = (float)ScoreValue::approxWhiteScoreOfScoreValueSmooth(scoreValue,0.0,2.0,board.sqrtBoardArea());
        buf.result->whiteScoreMeanSq = buf.result->whiteScoreMean * buf.result->whiteScoreMean;
        buf.result->whiteLead = buf.result->whiteScoreMean;
        buf.result->varTimeLeft = -1;
        buf.result->shorttermWinlossError = -1;
        buf.result->shorttermScoreError = -1;
      }
      else {
        buf.result->whiteWinProb = (float)lossProb;
        buf.result->whiteLossProb = (float)winProb;
        buf.result->whiteNoResultProb = (float)noResultProb;
        buf.result->whiteScoreMean = -(float)ScoreValue::approxWhiteScoreOfScoreValueSmooth(scoreValue,0.0,2.0,board.sqrtBoardArea());
        buf.result->whiteScoreMeanSq = buf.result->whiteScoreMean * buf.result->whiteScoreMean;
        buf.result->whiteLead = buf.result->whiteScoreMean;
        buf.result->varTimeLeft = -1;
        buf.result->shorttermWinlossError = -1;
        buf.result->shorttermScoreError = -1;
      }

    }
    else if(modelVersion >= 4) {
      double winProb;
      double lossProb;
      double noResultProb;
      double scoreMean;
      double scoreMeanSq;
      double lead;
      double varTimeLeft;
      double shorttermWinlossError;
      double shorttermScoreError;
      {
        double winLogits = buf.result->whiteWinProb * postProcessParams.outputScaleMultiplier;
        double lossLogits = buf.result->whiteLossProb * postProcessParams.outputScaleMultiplier;
        double noResultLogits = buf.result->whiteNoResultProb * postProcessParams.outputScaleMultiplier;
        double scoreMeanPreScaled = buf.result->whiteScoreMean * postProcessParams.outputScaleMultiplier;
        double scoreStdevPreSoftplus = buf.result->whiteScoreMeanSq * postProcessParams.outputScaleMultiplier;
        double leadPreScaled = buf.result->whiteLead * postProcessParams.outputScaleMultiplier;
        double varTimeLeftPreSoftplus = buf.result->varTimeLeft * postProcessParams.outputScaleMultiplier;
        double shorttermWinlossErrorPreSoftplus = buf.result->shorttermWinlossError * postProcessParams.outputScaleMultiplier;
        double shorttermScoreErrorPreSoftplus = buf.result->shorttermScoreError * postProcessParams.outputScaleMultiplier;

        if(history.rules.koRule != Rules::KO_SIMPLE && history.rules.scoringRule != Rules::SCORING_TERRITORY)
          noResultLogits -= 100000.0;

        // Softmax
        double maxLogits = std::max(std::max(winLogits,lossLogits),noResultLogits);
        winProb = exp(winLogits - maxLogits);
        lossProb = exp(lossLogits - maxLogits);
        noResultProb = exp(noResultLogits - maxLogits);

        if(history.rules.koRule != Rules::KO_SIMPLE && history.rules.scoringRule != Rules::SCORING_TERRITORY)
          noResultProb = 0.0;

        double probSum = winProb + lossProb + noResultProb;
        winProb /= probSum;
        lossProb /= probSum;
        noResultProb /= probSum;

        scoreMean = scoreMeanPreScaled * postProcessParams.scoreMeanMultiplier;
        double scoreStdev = softPlus(scoreStdevPreSoftplus) * postProcessParams.scoreStdevMultiplier;
        scoreMeanSq = scoreMean * scoreMean + scoreStdev * scoreStdev;
        lead = leadPreScaled * postProcessParams.leadMultiplier;
        varTimeLeft = softPlus(varTimeLeftPreSoftplus) * postProcessParams.varianceTimeMultiplier;

        // scoreMean and scoreMeanSq are still conditional on having a result, we need to make them unconditional now
        // noResult counts as 0 score for scorevalue purposes.
        scoreMean = scoreMean * (1.0-noResultProb);
        scoreMeanSq = scoreMeanSq * (1.0-noResultProb);
        lead = lead * (1.0-noResultProb);

        if(modelVersion >= 14) {
          {
            double s = softPlus(shorttermWinlossErrorPreSoftplus * 0.5);
            shorttermWinlossError = sqrt(s * s * postProcessParams.shorttermValueErrorMultiplier);
          }
          {
            double s = softPlus(shorttermScoreErrorPreSoftplus * 0.5);
            shorttermScoreError = sqrt(s * s * postProcessParams.shorttermScoreErrorMultiplier);
          }
        }
        else if(modelVersion >= 10) {
          shorttermWinlossError = sqrt(softPlus(shorttermWinlossErrorPreSoftplus) * postProcessParams.shorttermValueErrorMultiplier);
          shorttermScoreError = sqrt(softPlus(shorttermScoreErrorPreSoftplus) * postProcessParams.shorttermScoreErrorMultiplier);
        }
        else {
          shorttermWinlossError = softPlus(shorttermWinlossErrorPreSoftplus);
          shorttermScoreError = softPlus(shorttermScoreErrorPreSoftplus) * 10.0;
        }

        if(
          !isfinite(probSum) ||
          !isfinite(scoreMean) ||
          !isfinite(scoreMeanSq) ||
          !isfinite(lead) ||
          !isfinite(varTimeLeft) ||
          !isfinite(shorttermWinlossError) ||
          !isfinite(shorttermScoreError)
        ) {
          cout << "Got nonfinite for nneval value" << endl;
          cout << winLogits << " " << lossLogits << " " << noResultLogits
               << " " << scoreMean << " " << scoreMeanSq
               << " " << lead << " " << varTimeLeft
               << " " << shorttermWinlossError << " " << shorttermScoreError
               << endl;
          throw StringError("Got nonfinite for nneval value");
        }
      }

      if(nextPlayer == P_WHITE) {
        buf.result->whiteWinProb = (float)winProb;
        buf.result->whiteLossProb = (float)lossProb;
        buf.result->whiteNoResultProb = (float)noResultProb;
        buf.result->whiteScoreMean = (float)scoreMean;
        buf.result->whiteScoreMeanSq = (float)scoreMeanSq;
        buf.result->whiteLead = (float)lead;
      }
      else {
        buf.result->whiteWinProb = (float)lossProb;
        buf.result->whiteLossProb = (float)winProb;
        buf.result->whiteNoResultProb = (float)noResultProb;
        buf.result->whiteScoreMean = -(float)scoreMean;
        buf.result->whiteScoreMeanSq = (float)scoreMeanSq;
        buf.result->whiteLead = -(float)lead;
      }

      if(modelVersion >= 9) {
        buf.result->varTimeLeft = (float)varTimeLeft;
        buf.result->shorttermWinlossError = (float)shorttermWinlossError;
        buf.result->shorttermScoreError = (float)shorttermScoreError;
      }
      else {
        buf.result->varTimeLeft = -1;
        buf.result->shorttermWinlossError = -1;
        buf.result->shorttermScoreError = -1;
      }
    }
    else {
      throw StringError("NNEval value postprocessing not implemented for model version");
    }
  }

  // Postprocess ownermap
  if(buf.result->whiteOwnerMap != NULL) {
    if(modelVersion >= 3) {
      for(int pos = 0; pos<nnXLen*nnYLen; pos++) {
        int y = pos / nnXLen;
        int x = pos % nnXLen;
        if(y >= board.y_size || x >= board.x_size)
          buf.result->whiteOwnerMap[pos] = 0.0f;
        else {
          // Similarly as mentioned above, the result we get back from the net is actually not from white's perspective,
          // but from the player to move, so we need to flip it to make it white at the same time as we tanh it.
          if(nextPlayer == P_WHITE)
            buf.result->whiteOwnerMap[pos] = tanh(buf.result->whiteOwnerMap[pos] * postProcessParams.outputScaleMultiplier);
          else
            buf.result->whiteOwnerMap[pos] = -tanh(buf.result->whiteOwnerMap[pos] * postProcessParams.outputScaleMultiplier);
        }
      }
    }
    else {
      throw StringError("NNEval value postprocessing not implemented for model version");
    }
  }


  // And record the nnHash in the result and put it into the table, filed under the context
  // that earned it -- which is the context the request named, carried here on the buffer.
  // With no context attached anywhere this is the same store it always was.
  buf.result->nnHash = nnHash;
  if(nnCacheTable != nullptr)
    nnCacheTable->set(buf.result, cacheAttribution);

}
