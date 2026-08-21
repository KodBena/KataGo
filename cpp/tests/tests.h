#ifndef TESTS_H
#define TESTS_H

#include <sstream>

#include "../core/global.h"
#include "../core/logger.h"
#include "../core/rand.h"
#include "../core/test.h"
#include "../game/board.h"
#include "../game/rules.h"
#include "../game/boardhistory.h"

class NNEvaluator;

namespace Tests {
  //testboardbasic.cpp
  void runBoardIOTests();
  void runBoardBasicTests();
  void runBoardUndoTest();
  void runBoardHandicapTest();
  void runBoardStressTest();
  void runBoardReplayTest();


  //testboardarea.cpp
  void runBoardAreaTests();

  //testrules.cpp
  void runRulesTests();

  //testsuperkohash.cpp
  void runSuperKoBannedHashTests();

  //testpassalivesuicide.cpp
  void runPassAliveSuicideModeTests();
  void runExcludeTerritoryAtariModeTests();

  //testscore.cpp
  void runScoreTests();

  //testsgf.cpp
  void runSgfTests();
  void runSgfFileTests();

  //testnnposgeometry.cpp
  void runNNPosGeometryTests();

  //testpolicymaskedargmax.cpp
  void runPolicyMaskedArgmaxTests();

  //testnninputs.cpp
  void runNNInputsV3V4Tests();
  void runExcludeTerritoryAtariNNInputsTests();

  //testsymmetries.cpp
  void runBasicSymmetryTests();
  void runBoardSymmetryTests();
  void runSymmetryDifferenceTests();

  //testsearchnonn.cpp
  void runNNLessSearchTests();
  //testsearch.cpp
  void runSearchTests(const std::string& modelFile, bool inputsNHWC, bool cudaNHWC, int symmetry, bool useFP16);
  //testsearchv3.cpp
  void runSearchTestsV3(const std::string& modelFile, bool inputsNHWC, bool cudaNHWC, int symmetry, bool useFP16);
  //testsearchv8.cpp
  void runSearchTestsV8(const std::string& modelFile, bool inputsNHWC, bool cudaNHWC, bool useFP16);
  //testsearchv9.cpp
  void runSearchTestsV9(const std::string& modelFile, bool inputsNHWC, bool cudaNHWC, bool useFP16);

  //testsearchmisc.cpp
  void runNNOnTinyBoard(const std::string& modelFile, bool inputsNHWC, bool cudaNHWC, int symmetry, bool useFP16);
  void runNNSymmetries(const std::string& modelFile, bool inputsNHWC, bool cudaNHWC, bool useFP16);
  void runNNOnManyPoses(const std::string& modelFile, bool inputsNHWC, bool cudaNHWC, int symmetry, bool useFP16, const std::string& comparisonFile);
  void runNNBatchingTest(const std::string& modelFile, bool inputsNHWC, bool cudaNHWC, bool useFP16);

  //testtime.cpp
  void runTimeControlsTests();

  //testtrainingwrite.cpp
  void runTrainingWriteTests();
  void runPassAliveSuicideGameTests();
  void runSelfplayInitTestsWithNN(const std::string& modelFile);
  void runSekiTrainWriteTests(const std::string& modelFile);
  void runMoreSelfplayTestsWithNN(const std::string& modelFile);
  void runSelfplayStatTestsWithNN(const std::string& modelFile);

  //testnn.cpp
  void runNNLayerTests();
  void runNNSymmetryTests();

  //testonnxmodelfile.cpp
  void runOnnxModelFileTests(const std::string& scratchDir, const std::string& modelFile);

  //testownership.cpp
  void runOwnershipTests(const std::string& configFile, const std::string& modelFile);

  //testnnevalcanary.cpp
  void runCanaryTests(NNEvaluator* nnEval, int symmetry, bool print);
  bool runBackendErrorTest(
    NNEvaluator* nnEval,
    NNEvaluator* nnEval32,
    Logger& logger,
    const std::string& boardSizeDataset,
    int maxBatchSizeCap,
    bool verbose,
    bool quickTest,
    double policyOptimismForTest,
    double pdaForTest,
    double nnPolicyTemperatureForTest,
    bool& fp32BatchSuccessBuf,
    //Values on disk to compare correctness. We consider the pure-cpu float32 Eigen implementation of the neural network
    //to be the source of truth, since it is more likely to be stable and doesn't depend special hardware or drivers like
    //GPUs or other accelerators.
    //When running with Eigen backend, will overwrite this file with Eigen's results.
    const std::string& referenceFileName
  );

  //testbackendreference.cpp
  //Absolute-output check against compiled-in reference data blended across a sampling of nets
  //from the training run (see backendreferencedata.cpp). Only nets from that run are expected
  //to pass. Returns true regardless for models too small for the thresholds to be meaningful.
  bool runBackendReferenceTest(
    NNEvaluator* nnEval,
    Logger& logger,
    bool verbose,
    //Defaults for positions whose reference data does not specify its own policyOptimism/pda.
    //The temperature applies to all positions.
    double policyOptimismForTest,
    double pdaForTest,
    double nnPolicyTemperatureForTest,
    //Scales all limits in the lenient direction as it grows. 1.0 for normal checking. Composed
    //with the automatic model-size-based lenience.
    double lenienceFactor,
    //If nonempty, load reference data from this file instead of the compiled-in data.
    const std::string& referenceDataFileOverride,
    //If nonempty, dump this net's outputs on the reference positions, for calibration.
    const std::string& dumpCandidateFileName
  );

  //testconfig.cpp
  void runInlineConfigTests();
  void runConfigTests(const std::vector<std::string>& args);
  void runParseAllConfigsTest();
  void runTaskParsingTests();

  //testmisc.cpp
  void runCollectFilesTests();
  void runLoadModelTests();

  //testbook.cpp
  void runBookTests();

  //testnncache.cpp
  void runNNCacheConfigTests();
  void runNNCachePolicyTests();

  //testnncachefrozen.cpp
  void runNNCacheFrozenTests();

  //testnncachecountlog.cpp
  void runNNCacheCountLogTests();

  //testnnevalcontainer.cpp
  void runNNEvalContainerTests();

  //testnncachelevelzero.cpp
  void runNNCacheLevelZeroTests();

  //testnncachetwolevel.cpp
  void runNNCacheTwoLevelTests();

  //testnncachecontext.cpp
  void runNNCacheContextTests();

  //testnncacheobservations.cpp
  void runNNCacheObservationTests();

  //testnncachedump.cpp
  void runNNCacheDumpTests();

  //testnncachefilelock.cpp
  void runNNCacheFileLockTests();

  //testnncountsdump.cpp
  void runNNCountsDumpTests();

#ifdef KATAGO_NNCACHE_VERIFY_HITS
  //testnncacheverifyhits.cpp -- VERIFY BUILDS ONLY (cpp/neuralnet/nncacheverifyhits.h).
  //Neither of these exists in a default build, because the thing they cover does not.
  void runNNCacheVerifyHitsTests();
  // NOT a test: the seen-red instrument. Flips one payload scalar of the first evaluation in a
  // real container file and RE-MINTS both block checksums through the production functions, so
  // the file still loads and only a forward pass can tell it is wrong. Reached by the
  // nncachecorruptpayload subcommand. Prints the key it corrupted and the before/after value.
  void corruptFirstPersistedEvaluation(const std::string& containerPath);
#endif

  //testanalysismodels.cpp
  void runAnalysisModelNameSpaceTests();
  void runAnalysisCacheActionTests();
  void runAnalysisCacheLifecycleTests();
  // Not part of runtests: it is a measurement, not an assertion, and it writes real files.
  // Reached by the runnncachecountlogbench subcommand, which names the directory.
  void runNNCacheCountLogBench(const std::string& directory);

  //testnnevalcontainerbench.cpp
  // Not part of runtests, for the two reasons the count log's bench is not: it is a
  // measurement rather than an assertion, and it writes real files -- of GiB scale here --
  // so the directory is named on the command line rather than defaulted. `targetMiB` is the
  // container size to synthesise and `entriesPerDump` the block size to synthesise it in.
  // `mode` is one of:
  //   "index"  -- synthesise, time the cold key-set scan, then the torn-tail repair.
  //   "full"   -- the same, plus the payload-decoding load, which holds the whole live set
  //               in memory and so is not performable on a machine smaller than the container.
  //   "synth"  -- synthesise and STOP, leaving the container on disk.
  //   "append" -- append one dump to the container already on disk, and stop.
  // The last two exist so the torn-tail witness can be taken from OUTSIDE the process:
  // synthesise, hash the intact prefix, tear the tail with truncate(1), append, hash the
  // same prefix again. Within one process there is no moment between synthesis and the tear
  // at which an external hash could be taken, so a single-process bench can report the
  // repair's byte counts but can never witness the surviving bytes themselves.
  void runNNEvalContainerLoadIOBench(
    const std::string& directory, int64_t targetMiB, int64_t entriesPerDump,
    const std::string& mode);

  // Not part of runtests: it is a measurement, not an assertion. Reached by the
  // runnncachefrozenbench subcommand.
  void runNNCacheFrozenBench();
  // Not part of runtests: it is a measurement, not an assertion. Reached by the
  // runnncachebench subcommand.
  void runNNCacheBench();
  // Not part of runtests: it is a measurement, not an assertion. Reached by the
  // runnncachetwolevelbench subcommand.
  void runNNCacheTwoLevelBench();
}

namespace TestCommon {
  bool boardsSeemEqual(const Board& b1, const Board& b2);

  // A uniquely-named directory under the working directory that removes itself when the test
  // that made it returns or throws. For the tests that genuinely need a file on disk -- a count
  // log, a model file the loader will open -- not for tests in general.
  //
  // One thing it does NOT survive, stated because the name suggests otherwise: a failed
  // testAssert calls Global::fatalError, which calls quick_exit, which runs no destructors. A
  // red leg therefore does leave its directory behind (witnessed), which is why the names are
  // in .gitignore. The unique suffix is what keeps that from tripping up the next run.
  class ScopedTempDir {
   public:
    explicit ScopedTempDir(const std::string& namePrefix);
    ~ScopedTempDir();
    ScopedTempDir(const ScopedTempDir&) = delete;
    ScopedTempDir& operator=(const ScopedTempDir&) = delete;
    const std::string& path() const;

   private:
    std::string path_;
  };

  constexpr int MIN_BENCHMARK_SGF_DATA_SIZE = 7;
  constexpr int MAX_BENCHMARK_SGF_DATA_SIZE = 19;
  constexpr int DEFAULT_BENCHMARK_SGF_DATA_SIZE = std::min(Board::DEFAULT_LEN,MAX_BENCHMARK_SGF_DATA_SIZE);
  std::string getBenchmarkSGFData(int boardSize);

  std::vector<std::string> getMultiGameSize9Data();
  std::vector<std::string> getMultiGameSize13Data();
  std::vector<std::string> getMultiGameSize19Data();
  std::vector<std::string> getMultiGameSize10x14Data();
  std::vector<std::string> getMultiGameRectangleData();

  //backendreferencedata.cpp (machine-generated, see that file for the JSON schema)
  std::vector<std::string> getBackendReferenceJsonData();

  void overrideForBackends(bool& inputsNHWC, bool& useNHWC);
}

#endif
