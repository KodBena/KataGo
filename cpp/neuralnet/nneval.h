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
#include "../neuralnet/nncachefrozen.h"
#include "../neuralnet/nncachetwolevel.h"
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

  // WHETHER THIS EVALUATION IS THE REQUEST, OR ONE OF SEVERAL SERVING ONE REQUEST.
  //
  // It rides this buffer for exactly the reason the tag above it does, and it is CONSUMED at
  // the top of evaluate() in the same statement and against the same hazard: the buffer is
  // reused for every evaluation a thread ever makes, so a role left standing would be spent by
  // whichever evaluation came next -- and this one's unsafe value is the STICKY direction, so
  // a stale ServesACountedPresentation would silently stop counting a later, unrelated request.
  //
  // Defaults to ThePresentation, which is what every caller that never heard of this leaves it
  // as and what every ordinary path is. See NNCachePresentationRole for the defect that made it
  // necessary and for what it deliberately does not foreclose.
  NNCachePresentationRole cachePresentationRole;

  NNResultBuf();
  ~NNResultBuf();
  NNResultBuf(const NNResultBuf& other) = delete;
  NNResultBuf& operator=(const NNResultBuf& other) = delete;
};

// Result of NNEvaluator::benchmarkPureForward
struct NNEvalBenchmarkResult {
  int batchSize;
  int numThreads;
  int numIterations;
  // Per thread, the wall time of each timed getOutput call, in order.
  std::vector<std::vector<double>> perThreadIterationSeconds;
  std::vector<double> perThreadMedianSeconds;
  std::vector<double> perThreadNNEvalsPerSec;
  // Sum over threads of batchSize / medianSeconds.
  double sumMedianNNEvalsPerSec;
  // Wall time of the timed region: all threads started together, ending at the last thread's
  // final iteration (warmups and compute handle teardown excluded), and
  // numThreads * batchSize * numIterations / that wall time. This is the primary
  // throughput metric since it accounts for real concurrency between threads.
  double actualWallSeconds;
  double actualWallNNEvalsPerSec;
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

  // This session's observation count for every key of exactly this context, reported without
  // taking. Throws StringError if there is no cache, or for a context this evaluator did not
  // attach.
  [[nodiscard]] NNCacheObservationLedger harvestCacheObservationCountsFor(const NNCacheContextId& context) const;

  //-----------------------------------------------------------------------------------
  // The persisted cache (see nncachelevelzero.h, nnevalcontainer.h, nncachedump.h)
  //-----------------------------------------------------------------------------------

  // WHERE THIS MODEL'S PERSISTED CACHE LIVES: the directory holding <context>.nncounts and
  // <context>.<model>.nnevals, present exactly when the operator set nnCacheDir -- and
  // therefore exactly when this evaluator's cache can attach a level-0 source at all. The
  // two are one configuration decision, so they are one field here as they are one field in
  // NNCacheConfig, and there is no state in which an engine has a directory it cannot attach
  // from or an attach surface with nowhere to read.
  [[nodiscard]] const std::optional<std::string>& getCacheDirectory() const;

  // THIS EVALUATOR'S CACHE TABLE, for the protocol layer that persists and reports on it --
  // the harvest, attribution, dump-planning and stats surfaces NNCacheTable already carries.
  //
  // A reference rather than a pointer, and a throw rather than a null, so a caller cannot
  // hold "maybe there is a cache" as a value it might forget to test (ADR-0012 P9 rule 1).
  // Throws StringError, saying so, when this evaluator has no cache configured.
  [[nodiscard]] NNCacheTable& cacheTable() const;

  // ATTACHES ONE PRE-WARMED LEVEL-0 SOURCE at the END of this cache's resolution order, and
  // returns the only kind of value that can address it. Last attached is last tried, which
  // is the whole of the cross-model priority mechanism: a client that wants a source to win
  // a key attaches it earlier.
  //
  // NOT SAFE AGAINST A CONCURRENT EVALUATION, and that is a property of the structure rather
  // than a caution: a get walks the resolution list lock-free, and this mutates the vector it
  // walks, so a concurrent get can read freed memory. The protocol layer forecloses it by
  // refusing attach and detach while any request is open (docs/Analysis_Engine.md,
  // "cache_attach"), AND THAT REFUSAL IS THE ONLY DOOR: this act takes an
  // NNCacheLevelZeroSwapPermit, which cannot be constructed outside the three places that type
  // names, so a caller holding an NNEvaluator& and no permit cannot write this call at all. It
  // is not refused at run time; it does not compile. See NNCacheLevelZeroSwapPermit for why a
  // type replaced the release-compiled-out assertion that used to sit here.
  //
  // Throws StringError if this evaluator has no level-0 resolution list -- no nnCacheDir --
  // or for a null source.
  // `servesContext` is the context this source was loaded for, and it is REQUIRED at this door:
  // it is what lets a dump of that context write the retrievals this source serves, which no
  // key-shaped rule could recover afterwards. It must be a context THIS evaluator's cache
  // attached, and is refused by name otherwise.
  //
  // The returned value carries the handle AND what reconciling the source against level 1 did --
  // attach shadows every arriving key level 1 already owns, so a source detached across a set
  // and re-attached cannot serve the evaluation that set superseded
  // (NNCacheLevelZeroSources::attach).
  [[nodiscard]] NNCacheLevelZeroAttachment attachLevelZeroSource(
    NNCacheLevelZeroSwapPermit permit,
    std::unique_ptr<NNCacheFrozen> source,
    const NNCacheContextId& servesContext
  );

  // Removes the source `id` names and HANDS IT BACK, leaving every other source's relative
  // order unchanged. Returning it is what lets the caller release its storage through
  // nnCacheReleaseLevelZero, which OBSERVES whether the arena actually went rather than
  // assuming the destructor ran.
  //
  // Carries the same concurrency contract, and the same permit, as attachLevelZeroSource.
  // Throws StringError for an id this cache did not mint, for one already detached, and when
  // there is no level-0 resolution list at all.
  [[nodiscard]] std::unique_ptr<NNCacheFrozen> detachLevelZeroSource(
    NNCacheLevelZeroSwapPermit permit,
    const NNCacheLevelZeroSourceId& id
  );

  // How many level-0 sources are attached right now. Zero for a freshly started engine.
  // Throws StringError when there is no level-0 resolution list at all.
  [[nodiscard]] size_t numLevelZeroSources() const;

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

  // Benchmark raw NN forward throughput, bypassing the query queue and search.
  // Spins up one thread per configured NN server thread, each with its own compute handle,
  // fills a full batch once (cycling through boardSizes across the rows of the batch), then
  // times numIterations calls of NeuralNet::getOutput per thread after numWarmups untimed
  // calls, with all threads released simultaneously. Includes H2D/D2H transfer and host-side
  // output postprocessing, excludes feature generation and search.
  // Requires that server threads are NOT spawned (or have been killed).
  // This function is not threadsafe.
  NNEvalBenchmarkResult benchmarkPureForward(
    int numWarmups,
    int numIterations,
    const std::vector<int>& boardSizes
  );

  // Some stats
  uint64_t numRowsProcessed() const;
  uint64_t numBatchesProcessed() const;
  double averageProcessedBatchSize() const;
  uint64_t numCacheHits() const;

  void clearStats();

#ifdef KATAGO_NNCACHE_VERIFY_HITS
  // WHAT THE HIT VERIFIER HAS SEEN, for cache_stats to report. Verify builds only, so a client
  // that finds these fields in a response is looking at a debug build by construction and
  // cannot mistake one for the other (nncacheverifyhits.h). Absent (a disengaged optional)
  // when this evaluator has no cache table at all.
  [[nodiscard]] std::optional<NNCacheHitVerifyStats> getHitVerifyStats() const;
#endif

 private:
  // The one home of "this evaluator was built with a level-0 resolution list", so the three
  // public level-0 surfaces refuse in the same words.
  [[nodiscard]] NNCacheTwoLevelTable& levelZeroTableOrThrow() const;

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

  ComputeContext* computeContext;
  LoadedModel* loadedModel;
  std::unique_ptr<NNCacheTable> nnCacheTable;
  // The same table as nnCacheTable when this evaluator was built with a level-0 resolution
  // list, and null otherwise. NOT OWNED -- nnCacheTable owns it -- and set once at
  // construction beside it, so "has a directory" and "has an attach surface" are decided in
  // one place and cannot come apart.
  NNCacheTwoLevelTable* nnCacheLevelZeroTable;
  std::optional<std::string> nnCacheDirectory;
  Logger* logger;

#ifdef KATAGO_NNCACHE_VERIFY_HITS
  // VERIFY BUILDS ONLY. Built beside nnCacheTable and null exactly when that is null, so
  // "there is a cache" and "there is something verifying its hits" are one decision.
  std::unique_ptr<NNCacheHitVerifier> nnCacheHitVerifier;
#endif

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

#ifdef KATAGO_NNCACHE_VERIFY_HITS
  // VERIFY BUILDS ONLY. Called from the cache-hit branch of evaluate(), with the evaluation
  // the cache just served. Runs a fresh forward pass for the same position under a PINNED
  // symmetry and hands both to the verifier; refuses (and counts the refusal) when the
  // symmetry cannot be pinned. Never touches `served`, never stores, never changes what the
  // caller receives. See nncacheverifyhits.h for the whole account.
  void verifyCacheHitAgainstForwardPass(
    const Board& board,
    const BoardHistory& history,
    Player nextPlayer,
    const SGFMetadata* sgfMeta,
    const MiscNNInputParams& nnInputParams,
    NNCacheHitOrigin origin,
    const std::shared_ptr<NNOutput>& served
  );
#endif

 public:
  // Helper, for internal use only
  void serve(NNServerBuf& buf, Rand& rand, int gpuIdxForThisThread, int serverThreadIdx);
};

#endif  // NEURALNET_NNEVAL_H_
