#include "../core/fileutils.h"
#include "../core/global.h"
#include "../command/commandline.h"
#include "../main.h"
#include "../neuralnet/desc.h"
#include "../neuralnet/nncachecountlog.h"
#include "../neuralnet/nnevalcontainer.h"

#include <ghc/filesystem.hpp>

using namespace std;
namespace gfs = ghc::filesystem;

// The `nncachecompact` subcommand: compact a persisted NN-cache context's two on-disk files
// in place -- the count log (<context>.nncounts) and the evaluation container
// (<context>.<model>.nnevals) -- without reimplementing anything about their formats.
//
// WHY A SUBCOMMAND AND NOT A tools/ SCRIPT. A script that rewrote these files would need
// its own copy of the on-disk format: file header size, block header size, record size,
// format version. NNCacheCountLog and NNEvalContainer already expose exactly those
// constants (formatVersion(), fileHeaderBytes(), blockHeaderBytes(), recordBytes()) "named
// here so a test asserts against the implementation rather than against a second copy of
// the numbers" (see nncachecountlog.h). A second, script-side copy of that format is
// precisely what that comment guards against. This subcommand calls the engine's own
// compactIfNeeded on both stores; it holds no format knowledge of its own.
//
// WHAT COMPACTION IS. Both formats are append-only logs of increments; compactIfNeeded
// rewrites a file down to one block holding the merged live set once its physical size
// (records applied / entries applied) exceeds a multiple of that live set, or repairs a
// torn tail either way. See the format comments atop NNCacheCountLog and NNEvalContainer.
//
// EXCLUSIVE ACCESS IS REQUIRED AND NOT ENFORCED. Compaction rewrites a file IN PLACE via a
// temp file and an atomic rename. Both formats say plainly that they assume "ONE WRITER":
// there is no lock file and no reader/writer protocol (nncachecountlog.h, "ONE WRITER";
// nnevalcontainer.h's analogous header). An engine process that has this context attached
// establishes no exclusivity beyond that same assumption -- AnalysisCacheAttachments only
// stops one process from double-attaching a context to itself, and nothing here or
// upstream stops a second process from writing the same files at the same time. So this is
// a FINDING, not a mechanism this subcommand can lean on: running nncachecompact against a
// context an engine process has attached is the silent-corruption case a torn/interleaved
// rewrite would produce, and this subcommand cannot detect that condition and refuses to
// pretend otherwise. It says so below, at startup, every run.
//
// WHAT IS REPORTED. Before and after file sizes for both stores, and whether each store's
// compactIfNeeded call actually rewrote its file. "Nothing needed doing" (compacted=false)
// is reported as its own clean outcome, not folded into either "succeeded" or "failed" --
// see the printed report below.
//
// WHAT COUNTS AS A MISSING CONTEXT. Neither store's load treats a missing FILE as an
// error -- "no dump has happened here yet" is a normal answer for a context an engine has
// attached but never dumped. But this subcommand is not attaching anything; it is asked to
// compact a context's PERSISTED files, so a directory holding NEITHER the count log NOR the
// evaluation container for the (context, model) pair asked for is refused by name, loudly,
// rather than treated as an empty cache: there is nothing here that a dump created, so
// there is nothing for compaction to act on (ADR-0002).

namespace {

int64_t fileSizeOrZero(const string& path) {
  std::error_code ec;
  const gfs::path p(path);
  if(!gfs::exists(p, ec) || ec)
    return 0;
  const uintmax_t sz = gfs::file_size(p, ec);
  if(ec)
    return 0;
  return (int64_t)sz;
}

string compactedLine(const char* label, const string& path, int64_t before, int64_t after, bool compacted) {
  ostringstream out;
  out << label << " (" << path << "):\n"
      << "  before: " << before << " bytes\n"
      << "  after:  " << after << " bytes\n"
      << "  compacted: " << (compacted ? "true" : "false")
      << (compacted ? "" : "  (nothing needed doing -- not a failure)") << "\n";
  return out.str();
}

}  // namespace

int MainCmds::nncachecompact(const vector<string>& args) {
  string cacheDir;
  string context;
  string modelFile;
  int multiple;

  try {
    KataGoCommandLine cmd(
      "Compact a persisted NN-cache context's on-disk files (its count log and its "
      "evaluation container) in place. REQUIRES EXCLUSIVE ACCESS: the on-disk formats "
      "assume one writer and neither has a lock file or a reader/writer protocol, so this "
      "must not be run while any engine process has the context attached."
    );
    cmd.addModelFileArg();

    TCLAP::ValueArg<string> cacheDirArg(
      "", "cache-dir",
      "The persisted-cache directory: where <context>.nncounts and <context>.<model>.nnevals "
      "are read and rewritten. The same directory an analysis config's nnCacheDir would name.",
      true, string(), "DIR"
    );
    TCLAP::ValueArg<string> contextArg(
      "", "context", "The context name to compact (the same 'context' an analysis cache_attach/"
      "cache_dump request would use).",
      true, string(), "CONTEXT"
    );
    TCLAP::ValueArg<int> multipleArg(
      "", "multiple",
      "Compact a store only if it holds more than this many times its live set of distinct "
      "keys (must be >= 1). Default 1: compact now, whenever there is anything to gain -- "
      "compactIfNeeded(1) is already \"compact now\", not \"compact unconditionally\": a "
      "file already down to one record per live key is left alone. Pass the engine's own "
      "amortised default (4, NNCacheCountLog::defaultCompactionMultiple()) to run this as a "
      "scheduled maintenance step that only rewrites when an attached engine would have "
      "triggered a compaction on its own.",
      false, 1, "N"
    );
    cmd.add(cacheDirArg);
    cmd.add(contextArg);
    cmd.add(multipleArg);
    cmd.parseArgs(args);

    modelFile = cmd.getModelFile();
    cacheDir = cacheDirArg.getValue();
    context = contextArg.getValue();
    multiple = multipleArg.getValue();
  }
  catch(TCLAP::ArgException& e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  cout << "nncachecompact requires EXCLUSIVE ACCESS to this context's files. The count-log "
       << "and evaluation-container formats assume a single writer -- there is no lock file "
       << "and no reader/writer protocol -- so rewriting these files while an engine process "
       << "has this context attached is the silent-corruption case: a rewrite underneath a "
       << "live writer. Make sure no engine holds context '" << context << "' attached in '"
       << cacheDir << "' before proceeding." << endl;

  // A model file is loaded only far enough to read its name and version -- ModelDesc parsing
  // needs no compute backend -- because the evaluation container's file identity is
  // (context, modelInternalName), same as cacheDumpExecute's own resolution
  // (command/analysiscacheactions.cpp), and its header additionally pins the exact model
  // version that wrote it.
  ModelDesc modelDesc;
  ModelDesc::loadFromFileMaybeGZipped(modelFile, modelDesc, "");

  const NNCacheCountLog log = NNCacheCountLog::forContext(cacheDir, context);
  const NNEvalContainer container =
    NNEvalContainer::forContextAndModel(cacheDir, context, modelDesc.name, modelDesc.modelVersion);

  const bool countsFileExists = FileUtils::exists(log.path());
  const bool evalsFileExists = FileUtils::exists(container.path());
  if(!countsFileExists && !evalsFileExists) {
    throw StringError(
      "nncachecompact: neither " + log.path() + " nor " + container.path() + " exists. There is "
      "no persisted cache on disk for context '" + context + "' and model '" + modelDesc.name +
      "' (version " + Global::intToString(modelDesc.modelVersion) + ") in '" + cacheDir + "'. "
      "This refuses rather than treating an absent cache as an empty one: compaction acts on "
      "files a dump already created, and creates neither file itself."
    );
  }

  const int64_t countsBefore = fileSizeOrZero(log.path());
  const int64_t evalsBefore = fileSizeOrZero(container.path());

  const bool countsCompacted = log.compactIfNeeded(multiple);
  const bool evalsCompacted = container.compactIfNeeded(multiple);

  const int64_t countsAfter = fileSizeOrZero(log.path());
  const int64_t evalsAfter = fileSizeOrZero(container.path());

  cout << "Context: " << context << "  Model: " << modelDesc.name
       << " (version " << modelDesc.modelVersion << ")" << endl;
  cout << compactedLine("Count log", log.path(), countsBefore, countsAfter, countsCompacted);
  cout << compactedLine("Evaluation container", container.path(), evalsBefore, evalsAfter, evalsCompacted);

  return 0;
}
