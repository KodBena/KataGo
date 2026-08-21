#include "../command/analysiscachelifecycle.h"

#include <chrono>
#include <csignal>
#include <typeinfo>

#include "../core/global.h"
#include "../core/test.h"
#include "../neuralnet/nncache.h"

using namespace std;
using json = nlohmann::json;

const char* const AnalysisCacheLifecycle::KEY_ATTACH_CONTEXT = "nnCacheAttachContext";
const char* const AnalysisCacheLifecycle::KEY_DUMP_MIN_OBSERVATIONS = "nnCacheDumpMinObservations";
const char* const AnalysisCacheLifecycle::KEY_DUMP_INTERVAL_MINUTES = "nnCacheDumpIntervalMinutes";

// FIFTEEN MINUTES, and the number is a trade rather than a round figure picked for looking like
// one. It bounds what one kill costs at a quarter hour of a leaf's work. Shorter would buy a
// tighter bound and spend it on the one resource several leaves genuinely contend for -- a
// context's exclusive file lock, which a dump holds while it appends and which a sibling's attach
// or dump waits behind. Longer would make the operator's stated normal failure -- a kill for
// memory -- expensive again. It is a default, and the key exists precisely because a deployment
// that measures its own dump duration can do better than a default.
double AnalysisCacheLifecycle::defaultDumpIntervalMinutes() {
  return 15.0;
}

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
  const bool hasMinObservations = cfg.contains(KEY_DUMP_MIN_OBSERVATIONS);
  const bool hasInterval = cfg.contains(KEY_DUMP_INTERVAL_MINUTES);

  // REFUSED RATHER THAN IGNORED. Both of these govern a dump, and without a context there is no
  // dump for them to govern -- so an operator who set one of them believes the engine is
  // persisting something, and it is not. Ignoring them would be the silent-failure shape
  // (ADR-0002). One refusal for both keys because it is one mistake and one fix.
  if((hasMinObservations || hasInterval) && !hasContext) {
    // EVERY OFFENDING KEY IS NAMED, not just the first one found: an operator who set both and
    // was told about one would fix that one and be refused again by the other.
    const string offenders =
      string(hasMinObservations ? KEY_DUMP_MIN_OBSERVATIONS : "") +
      (hasMinObservations && hasInterval ? "' and '" : "") +
      string(hasInterval ? KEY_DUMP_INTERVAL_MINUTES : "");
    throw StringError(
      "Key '" + offenders + "' governs the dumps this engine performs on its own, and this engine "
      "performs none: '" + string(KEY_ATTACH_CONTEXT) + "' is not set, so no persisted-cache context "
      "is attached and nothing is ever written. Set '" + string(KEY_ATTACH_CONTEXT) + " = <name>' to "
      "give the engine a context to attach and dump, or remove '" + offenders + "'."
    );
  }

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
    hasMinObservations ? cfg.getInt64(KEY_DUMP_MIN_OBSERVATIONS, 0, (int64_t)1 << 40)
                       : (int64_t)cacheDumpDefaultAdmissionObservations();

  // MINUTES AT THE BOUNDARY BECAUSE THAT IS WHAT AN OPERATOR THINKS IN, seconds inside because
  // that is what a wait takes. A week is the upper bound -- past that the key is indistinguishable
  // from "off", which has its own honest spelling. The lower bound is a thousandth of a minute
  // (60 ms) rather than something rounder: it exists so a witness can observe several passes
  // without sleeping for minutes, and a bound that made the feature untestable would be a bound
  // that made the feature unwitnessed (ADR-0021).
  const double intervalMinutes =
    hasInterval ? cfg.getDouble(KEY_DUMP_INTERVAL_MINUTES, 0.0, 10080.0) : defaultDumpIntervalMinutes();
  if(hasInterval && intervalMinutes > 0.0 && intervalMinutes < 0.001)
    throw StringError(
      "Key '" + string(KEY_DUMP_INTERVAL_MINUTES) + "' = " + Global::doubleToString(intervalMinutes) +
      " is a positive interval shorter than 0.001 minutes (60 ms), which is not a schedule -- it is "
      "a leaf that spends its life holding this context's exclusive file lock and starving every "
      "sibling that shares the directory. Set 0 to turn periodic dumping off, or a real interval."
    );
  const optional<double> intervalSeconds =
    intervalMinutes > 0.0 ? optional<double>(intervalMinutes * 60.0) : optional<double>();

  return AnalysisCacheLifecycle(
    optional<Configured>(
      Configured{context, NNCacheDiskAdmission::minObservations((uint64_t)minObservations), intervalSeconds}
    )
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

const NNCacheDiskAdmission& AnalysisCacheLifecycle::dumpAdmission() const {
  if(!configured_.has_value())
    throw StringError(
      "AnalysisCacheLifecycle::dumpAdmission: this engine has no persisted-cache lifecycle ('" +
      string(KEY_ATTACH_CONTEXT) + "' is not set). Ask isConfigured() first."
    );
  return configured_.value().dumpAdmission;
}

const optional<double>& AnalysisCacheLifecycle::dumpIntervalSeconds() const {
  if(!configured_.has_value())
    throw StringError(
      "AnalysisCacheLifecycle::dumpIntervalSeconds: this engine has no persisted-cache lifecycle ('" +
      string(KEY_ATTACH_CONTEXT) + "' is not set). Ask isConfigured() first."
    );
  return configured_.value().dumpIntervalSeconds;
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

// THE ERROR LINE, in one home, because there are now three handlers producing it and the
// distinction it draws is the load-bearing part: whether the bytes reached disk. Three separate
// spellings of that distinction would be three chances to get it backwards (ADR-0012 P1).
string dumpFailureLine(
  const string& modelName, const string& context, const char* occasionWords,
  bool dumpPerformed, const string& cause
) {
  if(dumpPerformed)
    return Global::strprintf(
      "%s: ERROR -- model \"%s\" DID dump context \"%s\" %s, but this build cannot read its own "
      "dump's report, so what reached disk cannot be stated: %s",
      AnalysisCacheLifecycle::KEY_ATTACH_CONTEXT, modelName.c_str(), context.c_str(), occasionWords,
      cause.c_str()
    );
  return Global::strprintf(
    "%s: ERROR -- model \"%s\" could NOT dump context \"%s\" %s, and the work it holds is not on "
    "disk: %s",
    AnalysisCacheLifecycle::KEY_ATTACH_CONTEXT, modelName.c_str(), context.c_str(), occasionWords,
    cause.c_str()
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
    // THE RESPONSE IS READ INSIDE THE try, AND THAT IS NOT TIDINESS. `result` is a non-const json,
    // so operator[] on a key the response did not carry INSERTS a null and .get<int64_t>() then
    // throws nlohmann::json::type_error -- which is not a StringError, is caught by nothing here,
    // and (main.cpp installs no top-level handler outside Windows) reaches the operator as an
    // abort instead of the key-naming refusal every other line of this function exists to produce.
    // Same contract, same two files, same treatment as the dump side's requiredInt (ADR-0002).
    try {
      const json result = cacheAttachExecute(hosts, modelIdx, attachments, request);
      lines.push_back(
        Global::strprintf(
          "%s: model \"%s\" attached context \"%s\": %s entries in level 0, %s payload bytes, "
          "container tail %s, count log tail %s, %.1f ms",
          AnalysisCacheLifecycle::KEY_ATTACH_CONTEXT, modelName.c_str(), context.c_str(),
          Global::int64ToString(result.at("entriesInLevelZero").get<int64_t>()).c_str(),
          Global::int64ToString(result.at("levelZeroPayloadBytes").get<int64_t>()).c_str(),
          result.at("containerTail").get<string>().c_str(),
          result.at("countLogTail").get<string>().c_str(),
          result.at("buildMilliseconds").get<double>()
        )
      );
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
    catch(const std::exception& e) {
      throw StringError(
        "Config key '" + string(AnalysisCacheLifecycle::KEY_ATTACH_CONTEXT) + " = " + context +
        "' could not be honored for hosted model \"" + modelName +
        "\", and the cause is one this layer does not model (" + typeid(e).name() + "): " + e.what()
      );
    }
  }
  return lines;
}

namespace {

// The occasion, in the words the log line uses. One home, so the success line and the two error
// lines cannot say it three different ways.
const char* occasionPhrase(AnalysisCacheDumpOccasion occasion) {
  if(occasion == AnalysisCacheDumpOccasion::Interval) return "on the dump interval";
  if(occasion == AnalysisCacheDumpOccasion::Shutdown) return "at shutdown";
  if(occasion == AnalysisCacheDumpOccasion::TerminationSignal) return "on a termination signal";
  // Not a switch: the build turns on -Wswitch-default, so a switch over a closed enum would need
  // a default case that can never run and that would then hide a missing case from -Wswitch.
  throw StringError("AnalysisCacheDumpOccasion: unhandled occasion in a dump report line.");
}

}  // namespace

AnalysisCacheDumpReport analysisCacheDumpAttachedContexts(
  const AnalysisModelHosts& hosts,
  AnalysisCacheAttachments& attachments,
  const AnalysisCacheLifecycle& lifecycle,
  AnalysisCacheDumpOccasion occasion,
  int64_t openRequestCount
) {
  AnalysisCacheDumpReport report{vector<string>(), 0, 0, 0};
  if(!lifecycle.isConfigured())
    return report;

  const string& context = lifecycle.context();
  const char* const occasionWords = occasionPhrase(occasion);
  const AnalysisEngineCounters counters{openRequestCount};

  for(const SearchableModelIdx modelIdx: hosts.searchableIdxs()) {
    const string modelName = hosts.searchableEval(modelIdx)->getInternalModelName();
    if(!attachments.isAttached(modelIdx, context))
      continue;
    const CacheDumpRequest request{context, CacheDumpWhat::Both, lifecycle.dumpAdmission()};
    // WHICH OF THE TWO FAILURES HAPPENED, tracked rather than guessed at in the handler. A dump
    // that never ran left this session's work off the disk; a dump that ran and whose report
    // could not be read left the work ON the disk and the operator unable to say how much. Both
    // are failures and neither is the other, so the line says which (ADR-0008).
    bool dumpPerformed = false;
    // THE HANDLERS ARE THREE AND THE THIRD ONE IS NOT DECORATION. StringError (and its IOError /
    // ConfigParsingError subclasses) is what this stack refuses with, and it is what the first
    // handler is for. But "never throws" is a claim about EVERY exception, not about the ones
    // this code chose to think about: a std::bad_alloc out of a large append, or a json throw out
    // of rendering a report, would escape a StringError-only handler -- and on the interval
    // thread that is std::terminate, the whole engine killed by a failed write it was doing on
    // its own initiative. So the second handler takes std::exception and the third takes
    // everything, and both say plainly that the cause could not be named (ADR-0002: an unnameable
    // failure is still reported, it is not reported as a success).
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
        "\" dumped context \"" + context + "\" " + occasionWords + ": " + result.dump()
      );
    }
    catch(const StringError& e) {
      report.modelsFailed += 1;
      report.lines.push_back(dumpFailureLine(modelName, context, occasionWords, dumpPerformed, e.what()));
    }
    catch(const std::exception& e) {
      report.modelsFailed += 1;
      report.lines.push_back(
        dumpFailureLine(
          modelName, context, occasionWords, dumpPerformed,
          string("an exception this layer does not model (") + typeid(e).name() + "): " + e.what()
        )
      );
    }
    catch(...) {
      report.modelsFailed += 1;
      report.lines.push_back(
        dumpFailureLine(
          modelName, context, occasionWords, dumpPerformed,
          "a throw of a type that is not std::exception, so nothing can be said about it beyond "
          "that it happened."
        )
      );
    }
  }
  return report;
}

//-------------------------------------------------------------------------------------
// The periodic dump
//-------------------------------------------------------------------------------------

AnalysisCachePeriodicDumper::AnalysisCachePeriodicDumper(
  const AnalysisModelHosts& hosts_,
  AnalysisCacheAttachments& attachments_,
  std::mutex& cacheActionMutex_,
  const AnalysisCacheLifecycle& lifecycle_,
  std::function<int64_t()> openRequestCount_,
  std::function<void(const string&)> report_
)
  :hosts(hosts_),
   attachments(attachments_),
   cacheActionMutex(cacheActionMutex_),
   lifecycle(lifecycle_),
   openRequestCount(std::move(openRequestCount_)),
   report(std::move(report_))
{}

AnalysisCachePeriodicDumper::~AnalysisCachePeriodicDumper() {
  stop();
}

void AnalysisCachePeriodicDumper::start() {
  // NO THREAD AT ALL when there is nothing for it to do, rather than one that wakes up and
  // decides not to act: a thread that exists is a thread an operator can see, and one that exists
  // only to do nothing invites the reader to wonder what it is waiting for (ADR-0012 P11).
  if(!lifecycle.isConfigured() || !lifecycle.dumpIntervalSeconds().has_value())
    return;
  testAssert(!started);
  started = true;
  thread = std::thread([this]() { this->loop(); });
}

void AnalysisCachePeriodicDumper::stop() {
  if(!started)
    return;
  {
    std::lock_guard<std::mutex> lock(wakeMutex);
    stopping = true;
  }
  wake.notify_all();
  // JOINS RATHER THAN DETACHES, and waits out a pass in flight. An append that is abandoned
  // halfway is not made safe by the file format's torn-tail repair -- that repair exists for a
  // process that DIED, and choosing it deliberately to save a second at shutdown would be
  // choosing to leave a repair for the next reader to do.
  if(thread.joinable())
    thread.join();
  started = false;
}

void AnalysisCachePeriodicDumper::loop() {
  const double intervalSeconds = lifecycle.dumpIntervalSeconds().value();
  while(true) {
    {
      // THE WAIT IS ON A CONDITION, NOT A SLEEP. A sleeping thread would make every shutdown wait
      // out the remainder of an interval that may be a quarter of an hour long; this one is woken
      // by stop() and returns at once. The predicate is what makes the wakeup spurious-proof.
      std::unique_lock<std::mutex> lock(wakeMutex);
      wake.wait_for(lock, std::chrono::duration<double>(intervalSeconds), [this]() { return stopping; });
      if(stopping)
        return;
    }
    // THE FULL INTERVAL IS WAITED AGAIN AFTER THIS PASS RETURNS, which is what makes an overlap
    // impossible: there is one thread, and it is either waiting or dumping, never both.
    AnalysisCacheDumpReport pass;
    {
      std::lock_guard<std::mutex> lock(cacheActionMutex);
      pass = analysisCacheDumpAttachedContexts(
        hosts, attachments, lifecycle, AnalysisCacheDumpOccasion::Interval, openRequestCount()
      );
    }
    passesPerformed_.fetch_add(1);
    entriesWritten_.fetch_add(pass.entriesWritten);
    if(pass.modelsFailed > 0)
      passesWithAFailure_.fetch_add(1);
    // REPORTED OUTSIDE THE CACHE-ACTION MUTEX. The report goes to a Logger, which takes a lock of
    // its own and may write to a file; holding the cache-action mutex across that would make a
    // client's cache verb wait on disk I/O that has nothing to do with the cache.
    for(size_t i = 0; i < pass.lines.size(); i++)
      report(pass.lines[i]);
  }
}

int64_t AnalysisCachePeriodicDumper::passesPerformed() const {
  return passesPerformed_.load();
}

int64_t AnalysisCachePeriodicDumper::passesWithAFailure() const {
  return passesWithAFailure_.load();
}

int64_t AnalysisCachePeriodicDumper::entriesWritten() const {
  return entriesWritten_.load();
}

//-------------------------------------------------------------------------------------
// The dump a termination signal triggers
//-------------------------------------------------------------------------------------

AnalysisCacheDumpReport analysisCacheDumpForTerminationSignal(
  const AnalysisModelHosts& hosts,
  AnalysisCacheAttachments& attachments,
  std::mutex& cacheActionMutex,
  const AnalysisCacheLifecycle& lifecycle,
  int64_t openRequestCount
) {
  std::lock_guard<std::mutex> lock(cacheActionMutex);
  return analysisCacheDumpAttachedContexts(
    hosts, attachments, lifecycle, AnalysisCacheDumpOccasion::TerminationSignal, openRequestCount
  );
}

namespace {

// THE ONLY THING A SIGNAL HANDLER MAY TOUCH, and it is process-global because signal dispositions
// are. Zero means nothing has arrived. `sig_atomic_t` would be the C answer; a lock-free
// std::atomic<int> is the C++ one and start() REFUSES if this platform's is not lock-free rather
// than installing a handler that would be undefined behaviour (the same check contribute.cpp makes
// for the same reason).
std::atomic<int> terminationSignalReceived(0);
// Whether an AnalysisCacheTerminationSignalDumper currently owns the dispositions. Two would each
// believe they had them.
std::atomic<bool> terminationSignalHandlerInstalled(false);

extern "C" void analysisCacheTerminationSignalHandler(int sig) {
  // One store, to a lock-free atomic, and nothing else. No allocation, no lock, no I/O.
  terminationSignalReceived.store(sig);
}

// How often the watching thread looks at that atomic. Small enough that a supervisor's SIGTERM is
// acted on immediately by any human measure, large enough that the thread is not measurable.
const int WATCH_POLL_MILLISECONDS = 25;

}  // namespace

AnalysisCacheTerminationSignalDumper::AnalysisCacheTerminationSignalDumper(
  const AnalysisModelHosts& hosts_,
  AnalysisCacheAttachments& attachments_,
  std::mutex& cacheActionMutex_,
  const AnalysisCacheLifecycle& lifecycle_,
  std::function<int64_t()> openRequestCount_,
  std::function<void(const string&)> report_
)
  :hosts(hosts_),
   attachments(attachments_),
   cacheActionMutex(cacheActionMutex_),
   lifecycle(lifecycle_),
   openRequestCount(std::move(openRequestCount_)),
   report(std::move(report_))
{}

AnalysisCacheTerminationSignalDumper::~AnalysisCacheTerminationSignalDumper() {
  stop();
}

void AnalysisCacheTerminationSignalDumper::start() {
  if(!lifecycle.isConfigured())
    return;
  if(!std::atomic_is_lock_free(&terminationSignalReceived))
    throw StringError(
      "This platform's std::atomic<int> is not lock-free, so a signal handler cannot store the "
      "signal number safely, so the persisted cache CANNOT be dumped on SIGTERM here. Refusing to "
      "install a handler that would only appear to work. Set '" +
      string(AnalysisCacheLifecycle::KEY_DUMP_INTERVAL_MINUTES) + "' low enough that the periodic "
      "dump alone bounds the loss you are willing to take, and stop this engine by closing its "
      "stdin rather than by signalling it."
    );
  if(terminationSignalHandlerInstalled.exchange(true))
    throw StringError(
      "A persisted-cache termination-signal dumper is already installed in this process. Signal "
      "dispositions are process-global, so a second one would silently take the first one's "
      "handlers and the first one would never fire again."
    );
  terminationSignalReceived.store(0);
  stopping.store(false);
  std::signal(SIGINT, analysisCacheTerminationSignalHandler);
  std::signal(SIGTERM, analysisCacheTerminationSignalHandler);
  started = true;
  thread = std::thread([this]() { this->watch(); });
}

void AnalysisCacheTerminationSignalDumper::stop() {
  if(!started)
    return;
  stopping.store(true);
  if(thread.joinable())
    thread.join();
  // RESTORED, so that a signal arriving after this point kills the process exactly as it would
  // have before this class existed. Leaving the handler installed past the point where anything
  // is left to dump would mean a signal was swallowed by a thread that is no longer watching.
  std::signal(SIGINT, SIG_DFL);
  std::signal(SIGTERM, SIG_DFL);
  terminationSignalHandlerInstalled.store(false);
  started = false;
}

bool AnalysisCacheTerminationSignalDumper::isWatching() const {
  return started;
}

void AnalysisCacheTerminationSignalDumper::watch() {
  while(true) {
    // THE SIGNAL IS CHECKED BEFORE THE STOP FLAG, and the order is a bug fix rather than a
    // preference. Checked the other way round, a SIGTERM that arrived a few milliseconds before a
    // normal shutdown began was SWALLOWED: stop() set stopping, the loop saw it first and returned
    // without dumping, and the signal the operator sent did nothing. The e2e witness caught
    // exactly that (SC7b), which is what a witness is for.
    //
    // A WINDOW REMAINS AND IS NOT PAPERED OVER: a signal delivered after this load and before
    // stop() joins is not acted on by this thread. It is harmless in every case that matters --
    // the shutdown dump immediately following writes the same state, so the work still reaches
    // disk -- and closing it properly would mean holding a lock across a signal delivery, which is
    // the thing a signal handler may not do. The mechanism is "the work gets written", not "this
    // particular thread writes it" (ADR-0002: the residual is stated, not hidden).
    const int sig = terminationSignalReceived.load();
    if(sig == 0) {
      if(stopping.load())
        return;
      std::this_thread::sleep_for(std::chrono::milliseconds(WATCH_POLL_MILLISECONDS));
      continue;
    }

    report(
      Global::strprintf(
        "%s: signal %d received -- dumping the persisted cache before this process dies",
        AnalysisCacheLifecycle::KEY_ATTACH_CONTEXT, sig
      )
    );
    // MAY WAIT on a periodic pass already in flight, by design: the mutex is what makes one writer
    // at a time true, and a signal is not a reason to append into a file another thread is
    // appending into. The wait is bounded by one pass.
    const AnalysisCacheDumpReport dumped = analysisCacheDumpForTerminationSignal(
      hosts, attachments, cacheActionMutex, lifecycle, openRequestCount()
    );
    for(size_t i = 0; i < dumped.lines.size(); i++)
      report(dumped.lines[i]);
    report(
      Global::strprintf(
        "%s: %s model(s) dumped, %s failed, %s entries written; re-raising signal %d",
        AnalysisCacheLifecycle::KEY_ATTACH_CONTEXT,
        Global::int64ToString(dumped.modelsDumped).c_str(),
        Global::int64ToString(dumped.modelsFailed).c_str(),
        Global::int64ToString(dumped.entriesWritten).c_str(), sig
      )
    );

    // THE PROCESS DIES OF THE SIGNAL IT WAS SENT, with the exit status a supervisor expects, and
    // this call does not return. Restoring the default disposition first is what makes the raise
    // fatal rather than re-entering the handler above.
    std::signal(sig, SIG_DFL);
    std::raise(sig);
    // Unreachable for SIGINT and SIGTERM under SIG_DFL. If some platform ever returned here, the
    // honest thing is to stop watching rather than to loop dumping forever.
    return;
  }
}
