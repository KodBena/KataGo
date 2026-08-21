#include "../command/analysiscachelifecycle.h"

#include "../core/global.h"
#include "../neuralnet/nncache.h"

using namespace std;
using json = nlohmann::json;

const char* const AnalysisCacheLifecycle::KEY_ATTACH_CONTEXT = "nnCacheAttachContext";
const char* const AnalysisCacheLifecycle::KEY_SHUTDOWN_MIN_OBSERVATIONS = "nnCacheShutdownDumpMinObservations";

//-------------------------------------------------------------------------------------
// What the config asked for
//-------------------------------------------------------------------------------------

AnalysisCacheLifecycle::AnalysisCacheLifecycle(optional<Configured> configured)
  :configured_(std::move(configured))
{}

AnalysisCacheLifecycle AnalysisCacheLifecycle::none() {
  return AnalysisCacheLifecycle(optional<Configured>());
}

AnalysisCacheLifecycle AnalysisCacheLifecycle::fromCfg(ConfigParser& cfg) {
  const bool hasContext = cfg.contains(KEY_ATTACH_CONTEXT);
  const bool hasMinObservations = cfg.contains(KEY_SHUTDOWN_MIN_OBSERVATIONS);

  // REFUSED RATHER THAN IGNORED. An admission is a policy for a dump, and without a context
  // there is no dump for it to govern -- so an operator who set only this key believes the
  // engine is persisting something, and it is not. Ignoring it would be the silent-failure
  // shape (ADR-0002).
  if(hasMinObservations && !hasContext)
    throw StringError(
      "Key '" + string(KEY_SHUTDOWN_MIN_OBSERVATIONS) + "' governs the dump this engine performs "
      "at shutdown, and this engine performs none: '" + string(KEY_ATTACH_CONTEXT) + "' is not set, "
      "so no persisted-cache context is attached and nothing is ever written. Set '" +
      string(KEY_ATTACH_CONTEXT) + " = <name>' to give the engine a context to attach and dump, or "
      "remove '" + string(KEY_SHUTDOWN_MIN_OBSERVATIONS) + "'."
    );

  if(!hasContext)
    return none();

  const string context = cfg.getString(KEY_ATTACH_CONTEXT);
  if(Global::trim(context) == "")
    throw StringError(
      "Key '" + string(KEY_ATTACH_CONTEXT) + "' is empty. It names the persisted-cache context "
      "every hosted model attaches at startup, and it becomes a file name under nnCacheDir, so it "
      "has to be a name. Remove the key to run without a persisted-cache lifecycle."
    );

  // NOWHERE TO ATTACH FROM. nnCacheDir is what decides that a model's cache carries a level-0
  // resolution list at all, so without it there is no attach surface and this key could not be
  // honored under any spelling. The refusal names BOTH keys because the fix is to set the other
  // one, and a message naming only the key at fault would leave the operator to find that out.
  if(!cfg.contains(NNCacheConfig::KEY_DIR))
    throw StringError(
      "Key '" + string(KEY_ATTACH_CONTEXT) + "' = " + context + " cannot be honored without '" +
      string(NNCacheConfig::KEY_DIR) + "'. The context names a body of persisted evaluations; '" +
      string(NNCacheConfig::KEY_DIR) + "' names the directory <context>.nncounts and "
      "<context>.<model>.nnevals are read from and written to, and it is also what decides that a "
      "model's cache carries a level-0 resolution list to attach onto. Set '" +
      string(NNCacheConfig::KEY_DIR) + " = /some/existing/directory' -- KataGo will not create it -- "
      "or remove '" + string(KEY_ATTACH_CONTEXT) + "'."
    );

  // The upper bound is the one the wire decoder already puts on "minObservations", so a number
  // the config accepts is a number a cache_dump would accept (ADR-0012 P1).
  const int64_t minObservations =
    hasMinObservations ? cfg.getInt64(KEY_SHUTDOWN_MIN_OBSERVATIONS, 0, (int64_t)1 << 40)
                       : (int64_t)cacheDumpDefaultAdmissionObservations();

  return AnalysisCacheLifecycle(
    optional<Configured>(Configured{context, NNCacheDiskAdmission::minObservations((uint64_t)minObservations)})
  );
}

bool AnalysisCacheLifecycle::isConfigured() const {
  return configured_.has_value();
}

const string& AnalysisCacheLifecycle::context() const {
  if(!configured_.has_value())
    throw StringError(
      "AnalysisCacheLifecycle::context: this engine has no persisted-cache lifecycle ('" +
      string(KEY_ATTACH_CONTEXT) + "' is not set). Ask isConfigured() first."
    );
  return configured_.value().context;
}

const NNCacheDiskAdmission& AnalysisCacheLifecycle::shutdownAdmission() const {
  if(!configured_.has_value())
    throw StringError(
      "AnalysisCacheLifecycle::shutdownAdmission: this engine has no persisted-cache lifecycle ('" +
      string(KEY_ATTACH_CONTEXT) + "' is not set). Ask isConfigured() first."
    );
  return configured_.value().shutdownAdmission;
}

//-------------------------------------------------------------------------------------
// The two acts
//-------------------------------------------------------------------------------------

namespace {

// A NUMBER THE DUMP'S OWN RESPONSE CARRIES, REQUIRED RATHER THAN DEFAULTED. This shutdown dump
// always asks for what=both, so cacheDumpExecute always returns both halves; a missing field is
// therefore a broken contract between these two files and not a zero. Defaulting it would make
// "the dump wrote nothing" and "this build can no longer read what the dump wrote" the same line
// in the operator's log, which is the one distinction an operator reading that log needs
// (ADR-0002). It throws, and the caller's catch then reports the model as one whose dump cannot
// be vouched for -- which is what an unreadable report means.
int64_t requiredInt(const json& obj, const string& outer, const string& inner) {
  const json::const_iterator outerIt = obj.find(outer);
  if(outerIt != obj.end() && outerIt->is_object()) {
    const json& sub = *outerIt;
    const json::const_iterator innerIt = sub.find(inner);
    if(innerIt != sub.end() && innerIt->is_number_integer())
      return innerIt->get<int64_t>();
  }
  throw StringError(
    "The shutdown dump's own response carries no integer \"" + outer + "\".\"" + inner +
    "\", which a what=both dump always reports. Nothing about the context changed; what cannot "
    "be done is say what the dump wrote."
  );
}

}  // namespace

vector<string> analysisCacheStartupAttach(
  const AnalysisModelHosts& hosts,
  AnalysisCacheAttachments& attachments,
  const AnalysisCacheLifecycle& lifecycle
) {
  vector<string> lines;
  if(!lifecycle.isConfigured())
    return lines;

  const string& context = lifecycle.context();
  for(const SearchableModelIdx modelIdx: hosts.searchableIdxs()) {
    const string modelName = hosts.searchableEval(modelIdx)->getInternalModelName();
    const CacheAttachRequest request{
      context, NNCacheLevelZeroBound::all(), optional<int64_t>(), vector<string>()
    };
    json result;
    try {
      result = cacheAttachExecute(hosts, modelIdx, attachments, request);
    }
    catch(const StringError& e) {
      // NAMED AT THE KEY THE OPERATOR CAN ACT ON. The underlying refusal is about a container,
      // a lock or a name; what the operator has in front of them is a line in a config file.
      throw StringError(
        "Config key '" + string(AnalysisCacheLifecycle::KEY_ATTACH_CONTEXT) + " = " + context +
        "' could not be honored for hosted model \"" + modelName + "\": " + e.what() +
        " The engine is refusing to start rather than serving from an empty cache while the "
        "operator believes it is attached."
      );
    }
    lines.push_back(
      Global::strprintf(
        "%s: model \"%s\" attached context \"%s\": %s entries in level 0, %s payload bytes, "
        "container tail %s, count log tail %s, %.1f ms",
        AnalysisCacheLifecycle::KEY_ATTACH_CONTEXT, modelName.c_str(), context.c_str(),
        Global::int64ToString(result["entriesInLevelZero"].get<int64_t>()).c_str(),
        Global::int64ToString(result["levelZeroPayloadBytes"].get<int64_t>()).c_str(),
        result["containerTail"].get<string>().c_str(),
        result["countLogTail"].get<string>().c_str(),
        result["buildMilliseconds"].get<double>()
      )
    );
  }
  return lines;
}

AnalysisCacheShutdownDumpReport analysisCacheShutdownDump(
  const AnalysisModelHosts& hosts,
  AnalysisCacheAttachments& attachments,
  const AnalysisCacheLifecycle& lifecycle
) {
  AnalysisCacheShutdownDumpReport report{vector<string>(), 0, 0, 0};
  if(!lifecycle.isConfigured())
    return report;

  const string& context = lifecycle.context();
  // Every analysis thread has been joined by the time this runs, so this is the truth and not a
  // sample: no request can be open in a process with one thread left in it.
  const AnalysisEngineCounters counters{0};

  for(const SearchableModelIdx modelIdx: hosts.searchableIdxs()) {
    const string modelName = hosts.searchableEval(modelIdx)->getInternalModelName();
    if(!attachments.isAttached(modelIdx, context))
      continue;
    const CacheDumpRequest request{context, CacheDumpWhat::Both, lifecycle.shutdownAdmission()};
    // WHICH OF THE TWO FAILURES HAPPENED, tracked rather than guessed at in the handler. A dump
    // that never ran left this session's work off the disk; a dump that ran and whose report
    // could not be read left the work ON the disk and the operator unable to say how much. Both
    // are failures and neither is the other, so the line says which (ADR-0008).
    bool dumpPerformed = false;
    try {
      const json result = cacheDumpExecute(hosts, modelIdx, attachments, request, counters);
      dumpPerformed = true;
      const int64_t written = requiredInt(result, "evaluations", "entriesWritten");
      report.modelsDumped += 1;
      report.entriesWritten += written;
      // THE DUMP'S OWN RESPONSE, VERBATIM. Every figure this act produced -- what was written,
      // what was already persisted, what fell below the admission, what the counts append cost,
      // the honesty counters -- is already named and rendered by cacheDumpExecute, for a wire
      // client. Re-selecting a handful of them into a prose sentence would be a second, shorter
      // account of one act, authored here, drifting from the first the moment a field is added
      // (ADR-0012 P1). So the operator's log line carries exactly what a client that had sent
      // cache_dump by hand would have read.
      report.lines.push_back(
        string(AnalysisCacheLifecycle::KEY_ATTACH_CONTEXT) + ": model \"" + modelName +
        "\" dumped context \"" + context + "\" at shutdown: " + result.dump()
      );
    }
    catch(const StringError& e) {
      // COUNTED AND SAID, NEVER RAISED. See this function's header comment: the process is on
      // its way out and the remaining models still have work owed to disk. cacheDumpExecute's
      // own text already says what a failed counts append cost in rows and observations, so it
      // is carried through verbatim rather than summarized away.
      report.modelsFailed += 1;
      report.lines.push_back(
        dumpPerformed
        ? Global::strprintf(
            "%s: ERROR -- model \"%s\" DID dump context \"%s\" at shutdown, but this build cannot "
            "read its own dump's report, so what reached disk cannot be stated: %s",
            AnalysisCacheLifecycle::KEY_ATTACH_CONTEXT, modelName.c_str(), context.c_str(), e.what()
          )
        : Global::strprintf(
            "%s: ERROR -- model \"%s\" could NOT dump context \"%s\" at shutdown, and this "
            "session's work for that model is not on disk: %s",
            AnalysisCacheLifecycle::KEY_ATTACH_CONTEXT, modelName.c_str(), context.c_str(), e.what()
          )
      );
    }
  }
  return report;
}
