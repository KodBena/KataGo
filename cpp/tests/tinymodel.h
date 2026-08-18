#ifndef TESTS_TINYMODEL_H
#define TESTS_TINYMODEL_H

#include "../core/global.h"
#include "../core/config_parser.h"
#include "../core/logger.h"

class NNEvaluator;

namespace TinyModelTest {
  extern const char* tinyModelBase64Part0;
  extern const char* tinyModelBase64Part1;
  extern const char* tinyModelBase64Part2;
  extern const char* tinyModelBase64Part3;
  extern const char* tinyModelBase64Part4;
  extern const char* tinyModelBase64Part5;
  extern const char* tinyModelBase64Part6;

  extern const char* tinyMishModelBase64;

  // Which of the two neural nets embedded above to decode. They declare different internal
  // model names -- "rect15-b2c16-s13679744-d94886722" and "b1c6nbt" -- which is what makes them
  // usable by a test that needs more than one distinctly-named REAL model in one process, where
  // the debug stub cannot help because every stub calls itself "random".
  enum class EmbeddedModel {
    Rect15B2C16,
    B1C6Nbt,
  };

  // One embedded model, decoded to a file under baseDir and loaded through the same
  // Setup::initializeNNEvaluator path the engine itself uses -- a real net, not the debug stub.
  // Both the evaluator and the file it was loaded from belong to the caller: the file is named
  // in the result because the caller decides when it may be removed.
  struct LoadedTinyModel {
    NNEvaluator* eval;
    std::string modelFile;
  };
  LoadedTinyModel loadEmbeddedModel(
    EmbeddedModel which, const std::string& baseDir, Logger& logger, ConfigParser& cfg, bool randFileName
  );

  NNEvaluator* runTinyModelTest(const std::string& baseDir, Logger& logger, ConfigParser& cfg, bool randFileName, double errorTolFactor);
}



#endif
