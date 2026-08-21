#ifndef COMMAND_ANALYSISCACHELIFECYCLE_H_
#define COMMAND_ANALYSISCACHELIFECYCLE_H_

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "../command/analysiscacheactions.h"
#include "../command/analysismodels.h"
#include "../core/config_parser.h"
#include "../neuralnet/nncachedump.h"

// THE PERSISTED CACHE'S LIFECYCLE, DRIVEN BY THE CONFIG FILE AND BY NOTHING ELSE: attach at
// startup, dump at shutdown, no wire verb sent by anybody.
//
// WHAT THIS IS FOR. analysiscacheactions.h gives a CLIENT four verbs for driving one model's
// persisted cache across a session boundary. That is the right surface for a client that knows
// which card it is studying. It is the wrong surface for the deployment this feature was
// actually built for: a proxy fanning queries across several same-model KataGo leaves on one
// host, all pointed at one shared directory. There, every leaf attaches the SAME context at
// startup and dumps it at exit, forever, and nothing about that is a per-request decision. A
// deployment that had to send cache_attach down the wire to each leaf before it could be
// useful, and cache_dump to each leaf before it could be stopped, would put a KataGo lifecycle
// fact into every client that ever talks to the pool -- and would silently lose a leaf's whole
// session whenever a client forgot, crashed, or was replaced by one that did not know.
//
// So the operator states the context ONCE, in the config file the leaf is already started with,
// and the engine does the rest. This file is that.
//
// WHAT IT IS NOT. It is not a second implementation of attach and dump: it CALLS
// cacheAttachExecute and cacheDumpExecute, records into the same AnalysisCacheAttachments the
// wire verbs record into, and therefore produces attachments that are attached in exactly the
// sense the wire verbs mean. A client may cache_stats a config-attached context, may cache_dump
// it early, and may cache_detach it -- after which this engine has nothing left to dump at exit
// and does not pretend otherwise. Two lifecycles over one state would be two authors of one
// truth (ADR-0012 P1); there is one.
//
// WHY IT IS ITS OWN FILE. Same reason analysiscacheactions.cpp is: the interesting behaviour
// here is a set of REFUSALS and a set of REPORTS, and a refusal that can only be exercised by
// starting a whole engine process is a refusal that gets tested once. Everything below is a
// function of values -- a ConfigParser, a model registry, an attachment registry -- so
// tests/testanalysiscachelifecycle.cpp exercises it directly and the end-to-end witness then
// confirms the same behaviour over real processes rather than being the only thing that sees it.

//-------------------------------------------------------------------------------------
// What the config asked for
//-------------------------------------------------------------------------------------

// THE LIFECYCLE AN ENGINE WAS CONFIGURED WITH: either a context name TOGETHER WITH the admission
// and the interval its engine-driven dumps will use, or nothing at all.
//
// They are one value rather than three fields because "an admission with no context" is not a
// state an operator should be able to reach: it is a policy for a write that will never happen,
// and an engine holding one would have to explain itself (ADR-0012 P11, ADR-0000). Setting either
// dump key without the context key is therefore a startup REFUSAL, not a field left dangling, and
// it is refused in fromCfg where all three keys are in hand.
class AnalysisCacheLifecycle {
 public:
  // The cfg keys this Port reads. Named once here so the engine, the refusals and the tests
  // agree on their spelling (ADR-0012 P1).

  // The context every hosted searchable model attaches at startup and dumps at shutdown.
  // Becomes a path component, so it is validated to the closed alphabet by the count log and
  // the container -- the same validation a wire cache_attach gets, at startup instead.
  static const char* const KEY_ATTACH_CONTEXT;      // nnCacheAttachContext
  // How many recorded observations an entry needs before an ENGINE-DRIVEN dump will write it --
  // the periodic one and the shutdown one alike. ONE key rather than one per occasion: it is one
  // policy ("what this leaf is willing to put on the shared disk"), and two keys would let an
  // operator set a leaf that persists different things depending on how it happened to stop.
  // Defaults to the same number an "admission"-less cache_dump gets, which is the one home of
  // that policy (cacheDumpDefaultAdmissionObservations). Zero means "write everything", which
  // is exactly what NNCacheDiskAdmission::minObservations(0) is documented to be -- so the
  // accept-all case needs no second key and cannot disagree with this one.
  static const char* const KEY_DUMP_MIN_OBSERVATIONS;  // nnCacheDumpMinObservations
  // How long the engine waits between one engine-driven dump finishing and the next one
  // starting. 0 turns periodic dumping off; ABSENT does NOT -- see defaultDumpIntervalMinutes()
  // for why the default is on.
  static const char* const KEY_DUMP_INTERVAL_MINUTES;  // nnCacheDumpIntervalMinutes

  // WHY PERIODIC DUMPING IS ON BY DEFAULT once a context is configured, rather than off.
  //
  // The shutdown dump is BEST EFFORT and the deployment says so plainly: these leaves are
  // long-lived and are stopped by machine shutdown or by an arbitrary kill when memory is wanted
  // elsewhere (operator ruling, ledger row 1879). A process that is killed never reaches its
  // shutdown dump, so an engine whose ONLY persistence was that dump would lose every hour of
  // work it had done, silently, in the failure mode the operator says is normal. Defaulting the
  // interval to off would make the documented-normal case the lossy one.
  //
  // The operator has already asked for persistence by setting nnCacheAttachContext -- that key
  // means "this leaf's cache lives on disk" -- so honoring it only at a clean exit would be
  // honoring half of it. Off remains reachable, explicitly, by setting the interval to 0.
  //
  // THE NUMBER is a bound on how much work one kill can cost, traded against how often several
  // leaves contend for one context's EXCLUSIVE lock: a dump holds that lock, and while it does,
  // a sibling leaf attaching or dumping the same context waits.
  [[nodiscard]] static double defaultDumpIntervalMinutes();

  // An engine with no persisted-cache lifecycle: the behaviour every engine had before this
  // file existed.
  [[nodiscard]] static AnalysisCacheLifecycle none();

  // THE ONE PLACE .cfg TEXT BECOMES A LIFECYCLE. Throws StringError, naming the key at fault
  // and what to do about it, for:
  //   - KEY_DUMP_MIN_OBSERVATIONS or KEY_DUMP_INTERVAL_MINUTES set without KEY_ATTACH_CONTEXT
  //     (a write policy, or a schedule, for a write that will never happen);
  //   - KEY_ATTACH_CONTEXT set while the engine has no nnCacheDir (nowhere to attach FROM);
  //   - an empty KEY_ATTACH_CONTEXT.
  // It does NOT validate the context name's alphabet: that rule lives in the file types that
  // turn a name into a path, and restating it here would be a second author of it. A bad name
  // is refused by the startup attach below, which is still before the engine serves anything.
  //
  // Marks every key it reads as used on `cfg`, like every other ConfigParser get* call, so a
  // config that sets them is not flagged by warnUnusedKeys.
  [[nodiscard]] static AnalysisCacheLifecycle fromCfg(ConfigParser& cfg);

  // Present exactly when this engine attaches a context at startup and dumps it at shutdown.
  [[nodiscard]] bool isConfigured() const;
  // All three throw StringError if !isConfigured(), rather than handing back a default that
  // would read as a configured one.
  [[nodiscard]] const std::string& context() const;
  // Governs the periodic dump and the shutdown dump alike.
  [[nodiscard]] const NNCacheDiskAdmission& dumpAdmission() const;
  // The gap between engine-driven dumps, or nothing when periodic dumping was turned off with
  // an explicit 0. Seconds rather than the key's minutes, because seconds is what a wait takes
  // and converting once here beats converting at every use (ADR-0012 P1).
  [[nodiscard]] const std::optional<double>& dumpIntervalSeconds() const;

 private:
  struct Configured {
    std::string context;
    NNCacheDiskAdmission dumpAdmission;
    std::optional<double> dumpIntervalSeconds;
  };
  explicit AnalysisCacheLifecycle(std::optional<Configured> configured);
  std::optional<Configured> configured_;
};

//-------------------------------------------------------------------------------------
// The two acts
//-------------------------------------------------------------------------------------

// STARTUP. Attaches the configured context to EVERY hosted searchable model, in the registry's
// own order, recording each attachment in `attachments` exactly as a wire cache_attach does.
// Does nothing and reports nothing when the lifecycle is not configured.
//
// ONE CONTEXT NAME FOR EVERY HOSTED MODEL, and no per-model override. A context is a CARD -- a
// body of positions someone is studying -- and each model already gets its own file for it
// (<context>.<model>.nnevals), so one name across the hosted models is not a collision, it is
// the same card evaluated by each net. That is the deployment: a pool of leaves hosting the same
// nets, sharing one directory. A per-model key would need a model-name-to-key mapping and a
// refusal for a key naming a model this process does not host, for no witnessed need; it is
// filed, not built.
//
// THE BOUNDS ARE THE WIRE DEFAULTS: every key into level 0, no level-1 fill, no foreign sources.
// Exposing them as further config keys is filed for the same reason -- the leaf deployment wants
// the whole card, and a bound nobody has asked for is a knob to get wrong.
//
// THROWS StringError, naming KEY_ATTACH_CONTEXT and the model, on the first model that cannot
// attach -- a bad context name, an unreadable or unlockable directory, a store torn beyond
// recovery. The engine must then NOT start: a leaf that silently served an empty cache because
// its shared directory was unreachable is the failure this whole feature exists to notice
// (ADR-0002). Nothing has been written to disk at that point, and the partially-attached
// in-memory state dies with the refusing process.
//
// Returns one human-readable line per model, for the caller to log. It returns them rather than
// logging them so that this file needs no Logger and a test can read what an operator would.
[[nodiscard]] std::vector<std::string> analysisCacheStartupAttach(
  const AnalysisModelHosts& hosts,
  AnalysisCacheAttachments& attachments,
  const AnalysisCacheLifecycle& lifecycle
);

// WHY AN ENGINE-DRIVEN DUMP IS HAPPENING. It changes nothing about what the dump does; it changes
// the words in the log line, and an operator reading "this leaf wrote 4,000 entries" needs to know
// whether that was the interval or the exit (ADR-0008: the two are different events, so the record
// distinguishes them rather than making the reader infer it from a timestamp).
enum class AnalysisCacheDumpOccasion {
  Interval,
  Shutdown,
};

// What one engine-driven dumping pass did, in the figures an operator has to be able to read off
// the log: whether anything went wrong, and how much was written.
struct AnalysisCacheDumpReport {
  // One line per model dumped or failed, in hosting order.
  std::vector<std::string> lines;
  int64_t modelsDumped;
  // Nonzero means this session's work is short. The lines say which model and why.
  int64_t modelsFailed;
  int64_t entriesWritten;
};

// ONE ENGINE-DRIVEN DUMPING PASS. Dumps what=both, under the configured admission, for every
// hosted model that has the configured context attached. Does nothing when the lifecycle is not
// configured.
//
// ONE FUNCTION FOR BOTH OCCASIONS, not two. The periodic dump and the shutdown dump are the same
// act -- same verb, same admission, same skip rule -- and writing them twice would be two authors
// of one behaviour, drifting the first time one of them gained a field (ADR-0012 P1).
//
// SKIPS A MODEL WHOSE CONTEXT IS NOT ATTACHED, silently and correctly: a client that sent
// cache_detach by wire already decided what happened to that work, and cache_detach refuses to
// discard undumped state unless the client said discardUndumped. Re-attaching it here to dump it
// anyway would overrule that decision.
//
// NEVER THROWS, and that is load-bearing at both call sites: at shutdown the process is on its way
// out and an exception would replace an orderly exit with an abort that says less; on the interval
// it runs on a bare thread, where an escaping exception is std::terminate and takes the whole
// engine down over a failed write. Every failure is CAUGHT -- including the ones that are not
// StringError -- COUNTED, and put in `lines` for the caller to log, and the next model is still
// dumped (ADR-0002: the loss is said, never silent). The observation-loss disposition on a failed
// counts append is the one cacheDumpExecute already owns -- taken counts are not re-armed, and its
// refusal text says how many rows and observations went with them -- so that text is carried into
// the line verbatim rather than restated here.
//
// THE CALLER MUST HOLD THE CACHE-ACTION MUTEX (see AnalysisCachePeriodicDumper). This function
// reads `attachments`, which the request loop's cache verbs write.
//
// `openRequestCount` is reported in each dump's response, verbatim, as the engine's own record of
// whether this dump was taken at rest. At shutdown it is zero and that is a fact rather than a
// sample -- every analysis thread has been joined. On the interval it is whatever was open, which
// is exactly what a client reading the log wants to know.
[[nodiscard]] AnalysisCacheDumpReport analysisCacheDumpAttachedContexts(
  const AnalysisModelHosts& hosts,
  AnalysisCacheAttachments& attachments,
  const AnalysisCacheLifecycle& lifecycle,
  AnalysisCacheDumpOccasion occasion,
  int64_t openRequestCount
);

// THE PERIODIC DUMP, ON ITS OWN THREAD, because the engine it protects can be killed at any moment.
//
// WHY THIS EXISTS AT ALL. The shutdown dump only fires on a clean exit, and in the deployment this
// is for a clean exit is not the normal way a leaf stops: leaves are long-lived and are ended by
// machine shutdown or by an arbitrary kill when the box needs the memory (operator ruling, ledger
// row 1879). So the shutdown dump is BEST EFFORT and this is the primary persistence mechanism.
// The interval is the bound on how much one kill can cost.
//
// WHY A THREAD AND NOT THE REQUEST LOOP. A dump takes real time and holds the context's EXCLUSIVE
// file lock while it does. Run on the request loop it would stop the engine reading stdin -- and,
// worse, an idle leaf's request loop is blocked in a read that may not return for hours, so a
// timer that only fired when the loop next woke would not fire at all on exactly the leaf that has
// most to lose. It touches no structure an analysis thread touches on the get/set path, which is
// what makes running it beside live searches legal (see cacheDumpExecute's own note).
//
// WHAT IT IS SERIALIZED AGAINST, and how. `cacheActionMutex` is supplied by the engine and is held
// by the request loop around every cache_attach / cache_detach / cache_dump / cache_stats, and by
// this thread around each dumping pass. That is the whole of the concurrency design: the shared
// mutable state is AnalysisCacheAttachments and the level-0 resolution list, both of which the
// wire verbs write, and one mutex around both writers and this reader is a rule that fits in a
// sentence. Contention is nil in practice -- a client's cache verbs happen at session boundaries,
// which is when this thread is asleep -- and where it is not nil the cost falls on the client's
// verb, never on a search.
//
// OVERLAP IS NOT POSSIBLE, BY CONSTRUCTION RATHER THAN BY A GUARD. One thread does the dumps, and
// it waits the full interval AFTER a pass finishes rather than on a fixed schedule. So a pass that
// takes longer than the interval cannot be joined by a second one; it simply pushes the next start
// later, and the effective period becomes interval + duration. An operator whose dumps take longer
// than their interval sees that in the log's timestamps, and the honest fix is a longer interval,
// not a queue of overlapping writers contending for one exclusive lock.
class AnalysisCachePeriodicDumper {
 public:
  // `report` is called from the dumping thread, once per line, and must be safe to call from a
  // thread other than the one that constructed this (Logger::write is).
  AnalysisCachePeriodicDumper(
    const AnalysisModelHosts& hosts,
    AnalysisCacheAttachments& attachments,
    std::mutex& cacheActionMutex,
    const AnalysisCacheLifecycle& lifecycle,
    std::function<int64_t()> openRequestCount,
    std::function<void(const std::string&)> report
  );
  // Stops and joins, so the thread cannot outlive the evaluators it dumps from.
  ~AnalysisCachePeriodicDumper();
  AnalysisCachePeriodicDumper(const AnalysisCachePeriodicDumper&) = delete;
  AnalysisCachePeriodicDumper& operator=(const AnalysisCachePeriodicDumper&) = delete;

  // Starts the thread. A no-op when the lifecycle is unconfigured or its interval is off, so the
  // caller does not have to ask -- an engine that was told not to dump periodically simply has no
  // thread, rather than one that wakes up and decides to do nothing.
  void start();
  // Wakes the thread and joins it. Idempotent, and safe to call when start() never did anything.
  // RETURNS ONLY ONCE ANY DUMP IN FLIGHT HAS FINISHED: a half-written append is not something to
  // abandon to make a shutdown faster, and the append is the part that holds the file lock.
  void stop();

  // Read after stop() for the session's own accounting. Both are also visible while running.
  [[nodiscard]] int64_t passesPerformed() const;
  // Passes in which at least one model's dump failed. Nonzero means the log holds error lines.
  [[nodiscard]] int64_t passesWithAFailure() const;
  [[nodiscard]] int64_t entriesWritten() const;

 private:
  void loop();

  const AnalysisModelHosts& hosts;
  AnalysisCacheAttachments& attachments;
  std::mutex& cacheActionMutex;
  const AnalysisCacheLifecycle& lifecycle;
  std::function<int64_t()> openRequestCount;
  std::function<void(const std::string&)> report;

  std::thread thread;
  std::mutex wakeMutex;
  std::condition_variable wake;
  bool stopping = false;
  bool started = false;

  std::atomic<int64_t> passesPerformed_{0};
  std::atomic<int64_t> passesWithAFailure_{0};
  std::atomic<int64_t> entriesWritten_{0};
};

#endif  // COMMAND_ANALYSISCACHELIFECYCLE_H_
