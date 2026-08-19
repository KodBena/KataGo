#ifndef NEURALNET_NNEVAL_H_
#define NEURALNET_NNEVAL_H_

#include <memory>

#include "../core/global.h"
#include "../core/commontypes.h"
#include "../core/logger.h"
#include "../core/multithread.h"
#include "../core/threadsafequeue.h"
#include "../game/board.h"
#include "../game/boardhistory.h"
#include "../neuralnet/nncache.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/sgfmetadata.h"
#include "../neuralnet/nninterface.h"

class NNEvaluator;

// Each thread should allocate and re-use one of these
struct NNResultBuf {
  std::condition_variable clientWaitingForResult;
  std::mutex resultMutex;
  bool hasResult;
  bool includeOwnerMap;
  int boardXSizeForServer;
  int boardYSizeForServer;
  std::vector<float> rowSpatialBuf;
  std::vector<float> rowGlobalBuf;
  std::vector<float> rowMetaBuf;
  bool hasRowMeta;
  std::shared_ptr<NNOutput> result;
  bool errorLogLockout; // error flag to restrict log to 1 error to prevent spam
  int symmetry; // The symmetry to use for this eval
  double policyOptimism; // The policy optimism to use for this eval

  // Which attached cache context this evaluation is earned by, if this evaluator's cache has
  // any attached. The last leg of the plumb an analysis request starts: the request names a
  // context, the boundary resolves it against the model the request selected, the search
  // carries the resolution, and it arrives HERE -- on the buffer the evaluation rides on --
  // because the set that files the entry happens at the end of the very call this buffer is
  // the argument to. Anything shorter-lived than the buffer would need a second channel to
  // reach that set.
  //
  // Defaults to NoAttributableContext, which is what a request naming no context carries and
  // what every caller of evaluate() that never heard of contexts leaves it as.
  //
  // DEFERRED ALTERNATIVE, FILED HERE RATHER THAN NARRATED ELSEWHERE (ADR-0000 Exceptions).
  // The type-safe carrier is a REQUIRED PARAMETER of NNEvaluator::evaluate rather than a field
  // on a buffer, because a parameter is checked by the compiler at every call site and a field
  // is not. It was not taken, and the cost is concrete rather than a preference: outside
  // nneval.cpp and searchnnhelpers.cpp there are 15 files holding live evaluate() calls -- 7
  // production (evalsgf, gtp, writetrainingdata, genbook, startposes, play, playutils) and 8
  // test -- every one of which would have to thread an explicit
  // NNCacheAttribution::noAttributableContext() argument for no behavioural reason, and none
  // of which has anything to do with cache contexts. The accepted design
  // (cache-protocol-consult.wiki section 7) also names NNResultBuf as the carrier by name.
  //
  // WHAT THAT COSTS, EXACTLY, so a later reader can weigh it rather than re-derive it: a
  // MISSING assignment at some future evaluate() call site inside Search is not caught by the
  // compiler. It degrades to NoAttributableContext, which is counted and reported -- loud, not
  // silently wrong -- so the residual is under-attribution, never mis-attribution. The other
  // half of the class, a STALE tag surviving into the next evaluation, is already foreclosed:
  // evaluate() consumes this field at entry (see nneval.cpp), and testnncachecontext.cpp's
  // buffer-reuse tripwire holds it there.
  //
  // REVISIT WHEN a further evaluate() call site is added inside Search, or when a second
  // per-evaluation tag of this kind wants the same carrier -- either is the point at which
  // one parameter pays for the fifteen files, and the second would make the buffer a place
  // where two independent facts are remembered by convention.
  NNCacheAttribution cacheAttribution;

  NNResultBuf();
  ~NNResultBuf();
  NNResultBuf(const NNResultBuf& other) = delete;
  NNResultBuf& operator=(const NNResultBuf& other) = delete;
};

// Each server thread should allocate and re-use one of these
struct NNServerBuf {
  InputBuffers* inputBuffers;

  NNServerBuf(const NNEvaluator& nneval, const LoadedModel* model);
  ~NNServerBuf();
  NNServerBuf(const NNServerBuf& other) = delete;
  NNServerBuf& operator=(const NNServerBuf& other) = delete;
};

class NNEvaluator {
 public:
  NNEvaluator(
    const std::string& modelName,
    const std::string& modelFileName,
    const std::string& expectedSha256,
    Logger* logger,
    int maxBatchSize,
    int nnXLen,
    int nnYLen,
    bool requireExactNNLen,
    bool inputsUseNHWC,
    const NNCacheConfig& nnCacheConfig,
    bool debugSkipNeuralNet,
    const std::string& homeDataDirOverride,
    enabled_t useFP16Mode,
    int numThreads,
    const std::vector<int>& gpuIdxByServerThread,
    const std::string& randSeed,
    bool doRandomize,
    int defaultSymmetry,
    bool disableWarmup,
    // Consulted by the compute backend for its own custom options; not stored.
    ConfigParser& cfg
  );
  ~NNEvaluator();

  NNEvaluator(const NNEvaluator& other) = delete;
  NNEvaluator& operator=(const NNEvaluator& other) = delete;

  std::string getModelName() const;
  std::string getModelFileName() const;
  std::string getInternalModelName() const;
  std::string getAbbrevInternalModelName() const;
  Logger* getLogger();
  bool isNeuralNetLess() const;
  int getMaxBatchSize() const;
  int getCurrentBatchSize() const;
  void setCurrentBatchSize(int batchSize);
  bool requiresSGFMetadata() const;

  int getNumGpus() const;
  int getNumServerThreads() const;
  std::set<int> getGpuIdxs() const;
  int getNNXLen() const;
  int getNNYLen() const;
  bool getRequireExactNNLen() const;
  int getModelVersion() const;
  double getTrunkSpatialConvDepth() const;
  int64_t getNumModelParameters() const;
  bool modelHasAnyTransformerBlocks() const;
  bool modelHasAnyNestedBottleneckBlocks() const;
  enabled_t getUsingFP16Mode() const;

  // Check if the loaded neural net supports shorttermError fields
  bool supportsShorttermError() const;

  // Whether the loaded model declares that it expects pass-alive area input features to be
  // computed as if multi-stone suicide were always legal, regardless of the actual suicide rule.
  // False if there is no loaded model (e.g. debugSkipNeuralNet).
  bool modelPreferPassAliveUnderSuicideRules() const;

  // Whether the loaded model declares that it expects territory scoring with TaxRule NONE to
  // exclude empty points adjacent to chains in atari (rules version 3), both for adjudication and
  // for its territory input features. False if there is no loaded model (e.g. debugSkipNeuralNet).
  bool modelPreferExcludeTerritoryAdjacentToAtari() const;

  // Return the "nearest" supported ruleset to desiredRules by this model.
  // Fills supported with true if desiredRules itself was exactly supported, false if some modifications had to be made.
  Rules getSupportedRules(const Rules& desiredRules, bool& supported) const;

  // Clear all entires cached in the table
  void clearCache();

  //-----------------------------------------------------------------------------------
  // Cache contexts (see nncachecontext.h)
  //-----------------------------------------------------------------------------------

  // Attaches a context that this evaluator's cached earnings may be attributed to, and
  // returns the only kind of value that can address it.
  //
  // THIS IS THE SEAM the cache_attach action plugs into. That action does not exist yet: it
  // is the increment that reads a context's evaluation container, builds its frozen level-0
  // structure and joins its count log for build order, and it will call this to register the
  // name that content arrived under. Registering the name is separable from loading the
  // content, and separating them is what lets the request tag be resolved, refused and spent
  // before any loader exists -- rather than the tag waiting on the loader and the loader
  // landing with no tested consumer.
  //
  // Throws StringError, naming what failed, if this evaluator has no cache at all, if the
  // name is outside the closed alphabet a context name must fit, or if it is already
  // attached.
  NNCacheContextId attachCacheContext(const std::string& name);

  // What a request's optional "cacheContext" field selects on THIS model, or the refusal to
  // hand the client. See NNCacheContextSet::resolveForRequest for the rule; an evaluator
  // with no cache configured at all answers a named context with a refusal saying so, and no
  // name with NoAttributableContext, which is exactly today's behaviour.
  [[nodiscard]] NNCacheContextResolution resolveCacheContext(const std::optional<std::string>& requested) const;

  // The key -> context ledger of what this session earned in this evaluator's cache.
  // NotAttributed if there is no cache or no context was attached.
  [[nodiscard]] NNCacheAttributionLedger harvestCacheAttribution() const;

  // The hit-count rows a dump of exactly this context would write. Throws StringError if
  // there is no cache, or for a context this evaluator did not attach.
  [[nodiscard]] NNCacheHitLedger harvestCacheHitCountsFor(const NNCacheContextId& context) const;

  // Queue a position for the next neural net batch evaluation and wait for it. Upon evaluation, result
  // will be supplied in NNResultBuf& buf, the shared_ptr there can grabbed via std::move if desired.
  // logStream is for some error logging, can be NULL.
  // This function is threadsafe.
  void evaluate(
    const Board& board,
    const BoardHistory& history,
    Player nextPlayer,
    const MiscNNInputParams& nnInputParams,
    NNResultBuf& buf,
    bool skipCache,
    bool includeOwnerMap
  );
  void evaluate(
    const Board& board,
    const BoardHistory& history,
    Player nextPlayer,
    const SGFMetadata* sgfMeta,
    const MiscNNInputParams& nnInputParams,
    NNResultBuf& buf,
    bool skipCache,
    bool includeOwnerMap
  );
  std::shared_ptr<NNOutput>* averageMultipleSymmetries(
    const Board& board,
    const BoardHistory& history,
    Player nextPlayer,
    const SGFMetadata* sgfMeta,
    const MiscNNInputParams& baseNNInputParams,
    NNResultBuf& buf,
    bool includeOwnerMap,
    Rand& rand,
    int numSymmetriesToSample
  );

  // If there is at least one evaluate ongoing, wait until at least one finishes.
  // Returns immediately if there isn't one ongoing right now.
  void waitForNextNNEvalIfAny();

  // Actually spawn threads to handle evaluations.
  // If doRandomize, uses randSeed as a seed, further randomized per-thread
  // If not doRandomize, uses defaultSymmetry for all nn evaluations, unless a symmetry is requested in MiscNNInputParams.
  // This function itself is not threadsafe.
  void spawnServerThreads();

  // Kill spawned server threads and join and free them. This function is not threadsafe, and along with spawnServerThreads
  // should have calls to it and spawnServerThreads singlethreaded.
  void killServerThreads();

  // Set the number of threads and what gpus they use. Only call this if threads are not spawned yet, or have been killed.
  void setNumThreads(const std::vector<int>& gpuIdxByServerThr);

  // After spawnServerThreads has returned, check if is was using FP16.
  bool isAnyThreadUsingFP16() const;

  // These are thread-safe. Setting them in the middle of operation might only affect future
  // neural net evals, rather than any in-flight.
  bool getDoRandomize() const;
  int getDefaultSymmetry() const;
  void setDoRandomize(bool b);
  void setDefaultSymmetry(int s);

  // Some stats
  uint64_t numRowsProcessed() const;
  uint64_t numBatchesProcessed() const;
  double averageProcessedBatchSize() const;
  uint64_t numCacheHits() const;

  void clearStats();

 private:
  const std::string modelName;
  const std::string modelFileName;
  const int nnXLen;
  const int nnYLen;
  const bool requireExactNNLen;
  const int policySize;
  const bool inputsUseNHWC;
  const enabled_t usingFP16Mode;
  int numThreads;
  std::vector<int> gpuIdxByServerThread;
  const std::string randSeed;
  const bool debugSkipNeuralNet;
  const bool disableWarmup;

  ComputeContext* computeContext;
  LoadedModel* loadedModel;
  std::unique_ptr<NNCacheTable> nnCacheTable;
  Logger* logger;

  std::string internalModelName;
  int modelVersion;
  int inputsVersion;
  int numInputMetaChannels;

  ModelPostProcessParams postProcessParams;

  int numServerThreadsEverSpawned;
  std::vector<std::thread*> serverThreads;

  const int maxBatchSize;

  // Counters for statistics
  std::atomic<uint64_t> m_numRowsProcessed;
  std::atomic<uint64_t> m_numBatchesProcessed;
  std::atomic<uint64_t> m_numCacheHits;

  mutable std::mutex bufferMutex;

  // Everything in this section is protected under bufferMutex--------------------------------------------

  bool isKilled; // Flag used for killing server threads
  int numServerThreadsStartingUp; // Counter for waiting until server threads are spawned
  std::condition_variable mainThreadWaitingForSpawn; // Condvar for waiting until server threads are spawned

  std::vector<int> serverThreadsIsUsingFP16;

  int numOngoingEvals; // Current number of ongoing evals.
  int numWaitingEvals; // Current number of things waiting for finish.
  int numEvalsToAwaken; // Current number of things waitingForFinish that should be woken up. Used to avoid spurious wakeups.
  std::condition_variable waitingForFinish; // Condvar for waiting for at least one ongoing eval to finish.

  //-------------------------------------------------------------------------------------------------

  // Randomization settings for symmetries
  std::atomic<bool> currentDoRandomize;
  std::atomic<int> currentDefaultSymmetry;
  // Modifiable batch size smaller than maxBatchSize
  std::atomic<int> currentBatchSize;

  // Queued up requests
  ThreadSafeQueue<NNResultBuf*> queryQueue;

  // Fill buf.row{Spatial,Global,Meta}Buf from a position. Shared by evaluate() and warmup.
  void fillRowBufs(
    const Board& board,
    const BoardHistory& history,
    Player nextPlayer,
    const SGFMetadata* sgfMeta,
    const MiscNNInputParams& nnInputParams,
    NNResultBuf& buf
  ) const;

  // Run a forward pass on this freshly-created handle for each batch size 1 to maxBatchSize, with an
  // empty board. This pre-compiles any lazily-built backend graphs (e.g. cuDNN SDPA execution plans)
  // so the first real searches aren't stalled. No-op unless this is a transformer model on a backend
  // where warmup matters, and unless warmup is enabled.
  // gpuHandle may be NULL (neural-net-less), in which case this is a no-op.
  void maybeWarmupComputeHandle(ComputeHandle* gpuHandle, int serverThreadIdx);

 public:
  // Helper, for internal use only
  void serve(NNServerBuf& buf, Rand& rand, int gpuIdxForThisThread, int serverThreadIdx);
};

#endif  // NEURALNET_NNEVAL_H_
