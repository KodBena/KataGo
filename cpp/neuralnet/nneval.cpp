#include "../neuralnet/nneval.h"
#include "../neuralnet/modelversion.h"
#include "../core/test.h"

#include <algorithm>
#include <chrono>
#include <memory>
#include <optional>

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
    cacheAttribution(),
    cachePresentationRole(NNCachePresentationRole::ThePresentation)
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
   computeContext(NULL),
   loadedModel(NULL),
   nnCacheTable(nullptr),
   nnCacheLevelZeroTable(nullptr),
   nnCacheDirectory(nnCacheConfig.cacheDirectory),
   logger(lg),
   internalModelName(),
   modelVersion(-1),
   inputsVersion(-1),
   numInputMetaChannels(0),
   postProcessParams(),
   numServerThreadsEverSpawned(0),
   serverThreads(),
   maxBatchSize(maxBatchSz),
   numEffectiveDevices(NeuralNet::getNumEffectiveDevices(cfg, gpuIdxByServerThr)),
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
   maxRowsToSendPerBatch(maxBatchSz),
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
#ifdef KATAGO_NNCACHE_VERIFY_HITS
    // BUILT BESIDE THE TABLE AND ONLY BESIDE IT, so there is no configuration in which hits
    // exist and nothing is verifying them. Verify builds only -- see nncacheverifyhits.h.
    nnCacheHitVerifier = std::make_unique<NNCacheHitVerifier>(
      NNCacheHitVerifyTolerances::fromConfig(cfg),
      cfg.contains("nnCacheVerifyHitsIncludeResident") ? cfg.getBool("nnCacheVerifyHitsIncludeResident") : false,
      logger
    );
#endif
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
int NNEvaluator::getMaxRowsToSendPerBatch() const {
  return maxRowsToSendPerBatch.load(std::memory_order_acquire);
}
void NNEvaluator::setMaxRowsToSendPerBatch(int maxRows) {
  if(maxRows <= 0 || maxRows > maxBatchSize)
    throw StringError("Invalid setting for max rows to send per batch");
  maxRowsToSendPerBatch.store(maxRows,std::memory_order_release);
}
bool NNEvaluator::requiresSGFMetadata() const {
  return numInputMetaChannels > 0;
}

int NNEvaluator::getNumGpus() const {
  return numEffectiveDevices;
}
int NNEvaluator::getNumServerThreads() const {
  return (int)gpuIdxByServerThread.size();
}
const std::vector<int>& NNEvaluator::getGpuIdxByServerThread() const {
  return gpuIdxByServerThread;
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

NNCacheObservationLedger NNEvaluator::harvestCacheObservationCountsFor(const NNCacheContextId& context) const {
  if(nnCacheTable == nullptr)
    throw StringError(
      "NNEvaluator: model '" + modelName + "' has no NN cache configured, so no context is "
      "attached to it and none has presented anything here."
    );
  return nnCacheTable->harvestObservationCountsFor(context);
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

// THE TRIPWIRE THAT USED TO STAND HERE IS GONE, and its removal is the point rather than a
// casualty. assertNoEvaluationInFlightForLevelZeroSwap existed for exactly one caller: one that
// reached past the protocol layer to these two acts directly. That caller can no longer be
// written -- both acts now require an NNCacheLevelZeroSwapPermit, which nothing outside the three
// mints named on that type can construct -- so the assertion policed a class the type system no
// longer lets anyone express. Keeping it would give one rule two homes, the weaker of which was
// compiled out of every release build this project ships and therefore never protected a shipped
// binary at all (ADR-0012 P1; ADR-0000 Rule 2a -- the class is foreclosed at construction, which
// is the top of ADR-0002's loudness hierarchy, so a run-time check below it adds nothing). The
// axis the permit does NOT cover, named rather than left silent: a bug INSIDE the protocol layer
// that called these while a request was open. That axis is the request loop's own refusal
// (cacheSwapConcurrencyRefusal), which is always on, is exercised by the analysis engine cache
// action suite, and is not compiled out of anything.

NNCacheLevelZeroAttachment NNEvaluator::attachLevelZeroSource(
  NNCacheLevelZeroSwapPermit permit,
  std::unique_ptr<NNCacheFrozen> source,
  const NNCacheContextId& servesContext
) {
  NNCacheTwoLevelTable& table = levelZeroTableOrThrow();
  // REQUIRED HERE, THOUGH THE TABLE ACCEPTS A SOURCE WITHOUT ONE. This is the protocol's door,
  // and every source that comes through it was loaded from some context's container on that
  // context's behalf -- so an attach through here that named no context would be a source whose
  // retrievals no per-context dump could ever write, which is a silent loss with a client on the
  // other end of it. The table's own door stays permissive because a table can legitimately be
  // handed a source before any context exists: its own construction does exactly that.
  return table.attachLevelZero(permit, std::move(source), std::optional<NNCacheContextId>(servesContext));
}

std::unique_ptr<NNCacheFrozen> NNEvaluator::detachLevelZeroSource(
  NNCacheLevelZeroSwapPermit permit,
  const NNCacheLevelZeroSourceId& id
) {
  NNCacheTwoLevelTable& table = levelZeroTableOrThrow();
  return table.detachLevelZero(permit, id);
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
  NNEvaluator* nnEval,
  int gpuIdxForThisThread,
  int serverThreadIdx
) {
  Rand rand(randSeedThisThread);

  // We're in big trouble if this raises an exception (e.g. out of GPU memory) and causes possibly
  // the only nnEval thread to die, so let the exception escape and terminate the process. But log
  // and print the error first, since on some platforms an exception escaping a thread aborts
  // without printing anything. cerr is used unconditionally rather than only when there is no
  // logger, since the logger may be temporarily disabled (e.g. the benchmark suppresses logging
  // while respawning server threads).
  try {
    nnEval->serve(rand,gpuIdxForThisThread,serverThreadIdx);
  }
  catch(const std::exception& e) {
    Logger* logger = nnEval->getLogger();
    if(logger != NULL)
      logger->write(string("ERROR: NN server thread failed: ") + e.what());
    cerr << (string("ERROR: NN server thread failed: ") + e.what()) << endl;
    throw;
  }
  catch(...) {
    Logger* logger = nnEval->getLogger();
    if(logger != NULL)
      logger->write("ERROR: NN server thread failed with an exception that is not a std::exception");
    cerr << "ERROR: NN server thread failed with an exception that is not a std::exception" << endl;
    throw;
  }
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
      &serveEvals,randSeedThisThread,this,gpuIdxForThisThread,i
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
  if(gpuHandle == NULL || debugSkipNeuralNet || loadedModel == NULL)
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

NNEvalBenchmarkResult NNEvaluator::benchmarkPureForward(
  int numWarmups,
  int numIterations,
  const std::vector<int>& boardSizes
) {
  if(numIterations <= 0)
    throw StringError("benchmarkPureForward: numIterations must be positive");
  if(numWarmups < 0)
    throw StringError("benchmarkPureForward: numWarmups must be nonnegative");
  if(boardSizes.size() <= 0)
    throw StringError("benchmarkPureForward: no board sizes specified");
  if(debugSkipNeuralNet || loadedModel == NULL || computeContext == NULL)
    throw StringError("benchmarkPureForward requires a real neural net model");
  if(serverThreads.size() > 0)
    throw StringError("benchmarkPureForward: server threads must not be spawned");
  for(int bSize: boardSizes) {
    if(bSize < 2 || bSize > nnXLen || bSize > nnYLen)
      throw StringError("benchmarkPureForward: board size " + Global::intToString(bSize) + " out of range for NN buffer size");
    if(requireExactNNLen && (bSize != nnXLen || bSize != nnYLen))
      throw StringError("benchmarkPureForward: requireExactNNLen is set but board size " + Global::intToString(bSize) + " does not fill the NN buffer");
  }

  const int numThreadsToUse = (int)gpuIdxByServerThread.size();
  const int batchSize = maxBatchSize;
  testAssert(numThreadsToUse > 0);

  // Left populated after the benchmark returns, deliberately: callers query
  // isAnyThreadUsingFP16() afterward to report what precision was measured.
  {
    std::lock_guard<std::mutex> lock(bufferMutex);
    if(serverThreadsIsUsingFP16.size() < (size_t)numThreadsToUse)
      serverThreadsIsUsingFP16.resize(numThreadsToUse,0);
  }

  NNEvalBenchmarkResult result;
  result.batchSize = batchSize;
  result.numThreads = numThreadsToUse;
  result.numIterations = numIterations;
  result.perThreadIterationSeconds.resize(numThreadsToUse);
  result.perThreadMedianSeconds.assign(numThreadsToUse, 0.0);
  result.perThreadNNEvalsPerSec.assign(numThreadsToUse, 0.0);
  result.sumMedianNNEvalsPerSec = 0.0;
  result.actualWallSeconds = 0.0;
  result.actualWallNNEvalsPerSec = 0.0;

  std::atomic<int> readyCount(0);
  std::atomic<bool> startFlag(false);
  std::atomic<bool> anyError(false);
  std::mutex errorMutex;
  std::exception_ptr firstError;

  // Per-thread end-of-timed-loop timestamps, so that wall time excludes compute handle
  // teardown (freeing the model is slow and an early finisher's frees can also perturb
  // other threads' tail iterations, but that perturbation is at least visible in samples).
  std::vector<std::chrono::steady_clock::time_point> threadEndTimes(numThreadsToUse);

  std::vector<std::thread> threads;
  threads.reserve(numThreadsToUse);
  for(int threadIdx = 0; threadIdx < numThreadsToUse; threadIdx++) {
    threads.emplace_back([&,threadIdx]() {
      try {
        ComputeHandle* gpuHandle = NeuralNet::createComputeHandle(
          computeContext,
          loadedModel,
          logger,
          maxBatchSize,
          requireExactNNLen,
          inputsUseNHWC,
          gpuIdxByServerThread[threadIdx],
          threadIdx
        );
        // Declared before the try so that on an exception it is destroyed after the catch frees
        // the compute handle, matching the normal teardown order below.
        std::unique_ptr<NNServerBuf> serverBuf;
        try {
          // Mirror serve(): one compute handle and one set of input buffers per server thread,
          // with the buffers allocated only after createComputeHandle has bound this thread to
          // its GPU (see serve() for why the order matters).
          serverBuf = std::make_unique<NNServerBuf>(*this, loadedModel);
          maybeWarmupComputeHandle(gpuHandle, threadIdx);
          {
            std::lock_guard<std::mutex> lock(bufferMutex);
            if((size_t)threadIdx < serverThreadsIsUsingFP16.size())
              serverThreadsIsUsingFP16[threadIdx] = NeuralNet::isUsingFP16(gpuHandle) ? 1 : 0;
          }

          MiscNNInputParams nnInputParams;
          SGFMetadata sgfMeta;
          const SGFMetadata* sgfMetaPtr = NULL;
          if(numInputMetaChannels > 0) {
            sgfMeta = SGFMetadata::makeDummyWarmupProfile();
            sgfMetaPtr = &sgfMeta;
          }

          // Empty boards, cycling through the requested sizes across the rows of the batch.
          // Position content doesn't affect speed, but board size does when masking is active.
          std::vector<std::unique_ptr<NNResultBuf>> ownedBufs;
          std::vector<NNResultBuf*> resultBufs;
          ownedBufs.reserve(batchSize);
          resultBufs.reserve(batchSize);
          for(int row = 0; row < batchSize; row++) {
            int bSize = boardSizes[row % boardSizes.size()];
            Board board(bSize, bSize);
            BoardHistory history(
              board, P_BLACK, Rules::getTrompTaylorish(), 0,
              BoardHistoryModes(modelPreferPassAliveUnderSuicideRules(), modelPreferExcludeTerritoryAdjacentToAtari())
            );
            ownedBufs.push_back(std::make_unique<NNResultBuf>());
            NNResultBuf* buf = ownedBufs.back().get();
            fillRowBufs(board, history, P_BLACK, sgfMetaPtr, nnInputParams, *buf);
            buf->symmetry = 0;
            buf->policyOptimism = nnInputParams.policyOptimism;
            resultBufs.push_back(buf);
          }

          std::vector<std::unique_ptr<NNOutput>> ownedOutputs;
          std::vector<NNOutput*> outputs;
          ownedOutputs.reserve(batchSize);
          outputs.reserve(batchSize);
          for(int row = 0; row < batchSize; row++) {
            ownedOutputs.push_back(std::make_unique<NNOutput>());
            NNOutput* out = ownedOutputs.back().get();
            out->nnXLen = nnXLen;
            out->nnYLen = nnYLen;
            out->whiteOwnerMap = NULL;
            outputs.push_back(out);
          }

          for(int i = 0; i < numWarmups; i++)
            NeuralNet::getOutput(gpuHandle, serverBuf->inputBuffers, batchSize, resultBufs.data(), outputs);

          readyCount.fetch_add(1);
          while(!startFlag.load(std::memory_order_acquire))
            std::this_thread::yield();

          std::vector<double>& times = result.perThreadIterationSeconds[threadIdx];
          times.reserve(numIterations);
          for(int i = 0; i < numIterations; i++) {
            std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
            NeuralNet::getOutput(gpuHandle, serverBuf->inputBuffers, batchSize, resultBufs.data(), outputs);
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
            times.push_back(std::chrono::duration<double>(t1-t0).count());
          }
          threadEndTimes[threadIdx] = std::chrono::steady_clock::now();
        }
        catch(...) {
          NeuralNet::freeComputeHandle(gpuHandle);
          throw;
        }
        NeuralNet::freeComputeHandle(gpuHandle);
      }
      catch(...) {
        anyError.store(true);
        std::lock_guard<std::mutex> lock(errorMutex);
        if(firstError == nullptr)
          firstError = std::current_exception();
      }
    });
  }

  while(readyCount.load() < numThreadsToUse && !anyError.load())
    std::this_thread::yield();
  std::chrono::steady_clock::time_point wallStart = std::chrono::steady_clock::now();
  startFlag.store(true, std::memory_order_release);
  for(std::thread& t: threads)
    t.join();

  if(firstError != nullptr)
    std::rethrow_exception(firstError);

  std::chrono::steady_clock::time_point wallEnd = wallStart;
  for(int threadIdx = 0; threadIdx < numThreadsToUse; threadIdx++)
    wallEnd = std::max(wallEnd, threadEndTimes[threadIdx]);

  for(int threadIdx = 0; threadIdx < numThreadsToUse; threadIdx++) {
    std::vector<double> sorted = result.perThreadIterationSeconds[threadIdx];
    std::sort(sorted.begin(), sorted.end());
    double median =
      sorted.size() % 2 == 1 ? sorted[sorted.size()/2] :
      0.5 * (sorted[sorted.size()/2-1] + sorted[sorted.size()/2]);
    result.perThreadMedianSeconds[threadIdx] = median;
    result.perThreadNNEvalsPerSec[threadIdx] = median > 0.0 ? (double)batchSize / median : 0.0;
    result.sumMedianNNEvalsPerSec += result.perThreadNNEvalsPerSec[threadIdx];
  }
  result.actualWallSeconds = std::chrono::duration<double>(wallEnd - wallStart).count();
  if(result.actualWallSeconds > 0.0)
    result.actualWallNNEvalsPerSec =
      (double)numThreadsToUse * (double)batchSize * (double)numIterations / result.actualWallSeconds;
  return result;
}

void NNEvaluator::serve(
  Rand& rand,
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

  // Allocate input buffers only after createComputeHandle, which binds this thread to its GPU.
  // On the CUDA backend the pinned host allocation in the buffers initializes a context on the
  // thread's current device, so doing it before the GPU binding would leave a stray context
  // (and some wasted VRAM) on GPU 0 even when only other GPUs are configured.
  NNServerBuf buf(*this, loadedModel);

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
    int desiredBatchSize = std::min(maxBatchSize, maxRowsToSendPerBatch.load(std::memory_order_acquire));
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
  // AND THEY ARE ONE PRESENTATION, NOT numSymmetriesToSample OF THEM. Every iteration below
  // computes the same cache key -- NNInputs::getHash does not fold in symmetry, which is
  // exactly why skipCache is set -- so the position has COME UP ONCE and the count log must say
  // once. The first iteration is the presentation and the rest serve it; without this the
  // default seen-twice admission cleared itself in-session at numSymmetriesToSample >= 2. See
  // NNCachePresentationRole (nncacheobservations.h) for the whole account.
  for(int i = 0; i<numSymmetriesToSample; i++) {
    std::swap(symmetryIndexes[i], symmetryIndexes[rand.nextInt(i,SymmetryHelpers::NUM_SYMMETRIES-1)]);
    nnInputParams.symmetry = symmetryIndexes[i];
    bool skipCacheThisIteration = true; // Skip cache since there's no guarantee which symmetry is in the cache
    buf.cacheAttribution = attributionForEverySymmetry;
    buf.cachePresentationRole =
      i == 0 ? NNCachePresentationRole::ThePresentation : NNCachePresentationRole::ServesACountedPresentation;
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
  // THE PRESENTATION ROLE IS CONSUMED IN THE SAME BREATH, AND FOR A SHARPER VERSION OF THE SAME
  // REASON. A stale tag files one query's earnings under the previous query's card; a stale
  // role SILENTLY STOPS COUNTING a later, unrelated request, because its unsafe value is the
  // sticky one. Taking it out of the buffer at entry makes that unrepresentable rather than
  // something every fan-out caller has to remember to clear on its way out.
  const NNCachePresentationRole presentationRole = buf.cachePresentationRole;
  buf.cachePresentationRole = NNCachePresentationRole::ThePresentation;

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

  // THE OBSERVATION, COUNTED HERE AND NOWHERE ELSE, and this is the exact point because this
  // is the moment the position has been named and nothing has been decided about it yet.
  //
  // ONE evaluate() IS ONE PRESENTATION, whatever it goes on to do: a hit that returns below, a
  // miss that runs the net and sets, a skipCache caller that consults no level at all, or the
  // ownership-map fall-through that does a get, rejects the hit and sets a fuller result. All
  // four are one request for one position, and the currency the count log records is exactly
  // "how often does this position come up" -- a would-have-been-computed forward pass. Counted
  // inside get() and set() instead, the fall-through and the ordinary miss would each read as
  // two, which would put a freshly evaluated key over a seen-twice threshold in the session
  // that evaluated it and destroy the cross-session bootstrap the currency exists for. See
  // nncacheobservations.h.
  //
  // BEFORE THE EARLY RETURN, deliberately: a hit is a presentation. This is the whole defect
  // the old retrieval currency had in the other direction, where a MISS was not one.
  //
  // THE PRESENTATION IS MINTED HERE, and every cache interaction below is driven from it
  // rather than from the bare hash -- which is what makes "this position was presented and not
  // counted" unwritable on this path rather than merely unwritten.
  //
  // AND ONE evaluate() IS NOT ALWAYS ONE DEMAND, which is why WHICH mint is a decision and not
  // an assumption. averageMultipleSymmetries re-enters this function once per symmetry for a
  // single root query, and every one of those iterations computes the SAME nnHash --
  // NNInputs::getHash does not fold in symmetry, which is precisely why that path passes
  // skipCache. Counting each of them wrote N observations of one position for one query and
  // cleared a seen-twice threshold inside the session that evaluated it. The operator has ruled
  // that this is ONE demand (ledger row 1814); NNCachePresentationRole carries the whole
  // account.
  //
  // COST WHEN NO CONTEXT IS ATTACHED: one predictable branch, inlined -- see
  // NNCacheTable::present. The role test is a second predictable branch on a value that is
  // ThePresentation for every ordinary path.
  //
  // A TABLELESS EVALUATOR (nnCacheSizePowerOfTwo negative) has no table to mint from, so the
  // presentation is an optional here and every cache interaction below is already guarded by
  // the same null test it always was. Dereferenced with * and not .value(): the optional is
  // engaged under exactly the condition every use of it is already guarded by, and .value()
  // would add a second check and a throw path to the loop MCTS hammers for a state that cannot
  // occur.
  std::optional<NNCachePresentation> presentation;
  if(nnCacheTable != nullptr) {
    presentation =
      presentationRole == NNCachePresentationRole::ThePresentation
        ? nnCacheTable->present(nnHash, cacheAttribution)
        : nnCacheTable->presentAgainForSameRequest(nnHash);
  }

  bool hadResultWithoutOwnerMap = false;
  shared_ptr<NNOutput> resultWithoutOwnerMap;
#ifdef KATAGO_NNCACHE_VERIFY_HITS
  // VERIFY BUILDS ONLY: the same lookup, saying which level answered, because the hit worth a
  // forward pass is the one whose bytes came off disk. See nncacheverifyhits.h.
  NNCacheHitOrigin hitOrigin = NNCacheHitOrigin::LevelOneResident;
  const bool servedFromCache =
    nnCacheTable != nullptr && !skipCache && nnCacheTable->get(*presentation, buf.result, hitOrigin);
#else
  const bool servedFromCache =
    nnCacheTable != nullptr && !skipCache && nnCacheTable->get(*presentation, buf.result);
#endif
  if(servedFromCache) {
    if(!(includeOwnerMap && buf.result->whiteOwnerMap == NULL))
    {
      m_numCacheHits.fetch_add(1, std::memory_order_relaxed);
#ifdef KATAGO_NNCACHE_VERIFY_HITS
      // AFTER the hit is counted and BEFORE the return. This is ONE OF TWO arms that serve
      // deserialized bytes; the other is the ownermap fall-through immediately below, verified
      // further down where its forward pass lands. It cannot change buf.result: see the header.
      verifyCacheHitAgainstForwardPass(
        board, history, nextPlayer, sgfMeta, nnInputParams, hitOrigin, buf.result,
        /*viaOwnerMapFallThrough*/ false
      );
#endif
      buf.hasResult = true;
      return;
    }
    else {
      hadResultWithoutOwnerMap = true;
      resultWithoutOwnerMap = std::move(buf.result);
      buf.result = nullptr;
#ifdef KATAGO_NNCACHE_VERIFY_HITS
      // THE SECOND ARM THAT SERVES DESERIALIZED BYTES, AND IT IS VERIFIED TOO. This entry does
      // not answer the request as it stands -- it lacks the ownership map -- but its scalars and
      // its whole policy array are copied back over the fresh result below and handed to the
      // caller, and then STORED into level 1. So a corrupt on-disk value reaches the client here
      // exactly as it would through the early return, and is additionally promoted into resident
      // memory. Verified against its own recompute rather than against the forward pass this
      // call is about to make: that one is deliberately NOT postprocessed on this path (see the
      // copy below), so it is raw logits and comparing it to a postprocessed cached entry would
      // measure the postprocessing, not the deserialization.
      verifyCacheHitAgainstForwardPass(
        board, history, nextPlayer, sgfMeta, nnInputParams, hitOrigin, resultWithoutOwnerMap,
        /*viaOwnerMapFallThrough*/ true
      );
#endif
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
  // THROUGH THE PRESENTATION, so the store is of the position this call presented and the two
  // cannot come apart: NNCacheTable::set refuses a payload whose hash is not the presented key.
  if(nnCacheTable != nullptr
#ifdef KATAGO_NNCACHE_VERIFY_HITS
     // A VERIFICATION RECOMPUTE MUST NOT STORE ITS ANSWER. This call is the nested forward pass
     // a level-0 hit is being checked against; storing it would shadow the level-0 entry, move
     // the key to level 1, and silently end verification of that key after its first hit --
     // the instrument consuming the thing it is there to observe. The default build compiles
     // no part of this condition (nncacheverifyhits.h).
     && !NNCacheHitVerifier::inRecompute()
#endif
  )
    nnCacheTable->set(*presentation, buf.result, cacheAttribution);

}

#ifdef KATAGO_NNCACHE_VERIFY_HITS

std::optional<NNCacheHitVerifyStats> NNEvaluator::getHitVerifyStats() const {
  if(nnCacheHitVerifier == nullptr)
    return std::optional<NNCacheHitVerifyStats>();
  return std::optional<NNCacheHitVerifyStats>(nnCacheHitVerifier->stats());
}

void NNEvaluator::verifyCacheHitAgainstForwardPass(
  const Board& board,
  const BoardHistory& history,
  Player nextPlayer,
  const SGFMetadata* sgfMeta,
  const MiscNNInputParams& nnInputParams,
  NNCacheHitOrigin origin,
  const shared_ptr<NNOutput>& served,
  bool viaOwnerMapFallThrough
) {
  if(nnCacheHitVerifier == nullptr)
    return;
  // A RECOMPUTE'S OWN HITS ARE NOT VERIFIED. It passes skipCache, so it takes no hits today;
  // this is here so that stays true if that ever changes, rather than being a property a
  // reader has to go check.
  if(NNCacheHitVerifier::inRecompute())
    return;
  // A NEURAL-NET-LESS EVALUATOR HAS NO FORWARD PASS TO COMPARE AGAINST. Recomputing here would
  // compare a cached entry against the fixed dummy output and report every hit as a mismatch.
  // Counted rather than returned from silently: a build configured this way verifies nothing,
  // and that must be readable off cache_stats rather than inferred.
  if(debugSkipNeuralNet) {
    nnCacheHitVerifier->countSkippedNoRecompute();
    return;
  }
  if(!nnCacheHitVerifier->shouldVerify(origin))
    return;

  const int symmetryToUse = pinnedVerifySymmetryOrRefuse(nnInputParams);
  if(symmetryToUse < 0)
    return;

  MiscNNInputParams recomputeParams = nnInputParams;
  recomputeParams.symmetry = symmetryToUse;

  // ITS OWN BUFFER, not the caller's: the caller's buffer holds the result that is about to be
  // returned, and evaluate() writes through buf.result.
  NNResultBuf verifyBuf;
  // ONE PRESENTATION FOR THE WHOLE REQUEST, NOT TWO. The same channel averageMultipleSymmetries
  // uses to fan one demand out over several forward passes: the hit that brought us here was
  // already counted as the presentation, and this evaluation SERVES it.
  verifyBuf.cachePresentationRole = NNCachePresentationRole::ServesACountedPresentation;
  // Unattributed on purpose: this evaluation earns no card anything, and it will not be stored.
  verifyBuf.cacheAttribution = NNCacheAttribution::noAttributableContext();

  try {
    // THE SCOPE IS WHAT KEEPS THE INSTRUMENT OUTSIDE THE INSTRUMENT -- it suppresses the
    // terminal store at the end of the nested evaluate(). It must cover the whole call.
    NNCacheHitVerifier::RecomputeScope scope;
    // includeOwnerMap MIRRORS WHAT WAS SERVED rather than what the caller asked for, so the two
    // outputs have the same shape and the ownermap-present flag is compared as the persisted
    // field it is rather than being papered over by asking for one either way.
    evaluate(
      board, history, nextPlayer, sgfMeta, recomputeParams, verifyBuf,
      /*skipCache*/ true, /*includeOwnerMap*/ served->whiteOwnerMap != NULL
    );
  }
  catch(const std::exception&) {
    // COUNTED AND SWALLOWED. The caller of the outer evaluate() is about to be handed the
    // evaluation the cache served; letting this throw past that would replace their answer with
    // an exception, which is the instrument taking down the request it exists to observe. The
    // count is what keeps it from vanishing instead (audit finding, 2026-08-21).
    nnCacheHitVerifier->countRecomputeThrew();
    return;
  }

  if(verifyBuf.result == nullptr) {
    nnCacheHitVerifier->countSkippedNoRecompute();
    return;
  }
  nnCacheHitVerifier->compare(
    served->nnHash, symmetryToUse, *served, *verifyBuf.result, policySize, viaOwnerMapFallThrough
  );
}

// PINNING THE SYMMETRY, OR REFUSING -- written once because both verify paths ask exactly the
// same question and two copies of it could disagree (ADR-0012 P1). Returns the symmetry to
// recompute under, or -1 having COUNTED the refusal.
//
// See nncacheverifyhits.h for the whole account. In short: a cached evaluation does not record
// the symmetry it was computed under and the cache key does not fold symmetry, so the only
// configuration in which a recompute is comparable to a served entry is one where every
// unspecified evaluation takes the same currentDefaultSymmetry. A request that NAMES a symmetry
// is refused, not honoured: pinning the recompute to the request's symmetry says nothing about
// the entry's, and comparing across two symmetries measures the net's non-equivariance.
int NNEvaluator::pinnedVerifySymmetryOrRefuse(const MiscNNInputParams& nnInputParams) {
  if(nnInputParams.symmetry != NNInputs::SYMMETRY_NOTSPECIFIED) {
    nnCacheHitVerifier->countSkippedNondeterministicSymmetry();
    return -1;
  }
  if(currentDoRandomize.load(std::memory_order_acquire)) {
    nnCacheHitVerifier->countSkippedNondeterministicSymmetry();
    return -1;
  }
  const int defaultSymmetry = currentDefaultSymmetry.load(std::memory_order_acquire);
  if(defaultSymmetry < 0 || defaultSymmetry >= SymmetryHelpers::NUM_SYMMETRIES) {
    nnCacheHitVerifier->countSkippedNondeterministicSymmetry();
    return -1;
  }
  return defaultSymmetry;
}

#endif  // KATAGO_NNCACHE_VERIFY_HITS
