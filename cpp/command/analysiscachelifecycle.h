#ifndef COMMAND_ANALYSISCACHELIFECYCLE_H_
#define COMMAND_ANALYSISCACHELIFECYCLE_H_

#include <optional>
#include <string>
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
// its shutdown dump will use, or nothing at all.
//
// The two are one value rather than two fields because "an admission with no context" is not a
// state an operator should be able to reach: it is a policy for a write that will never happen,
// and an engine holding one would have to explain itself (ADR-0012 P11, ADR-0000). Setting the
// admission key without the context key is therefore a startup REFUSAL, not a field left
// dangling, and it is refused in fromCfg where both keys are in hand.
class AnalysisCacheLifecycle {
 public:
  // The cfg keys this Port reads. Named once here so the engine, the refusals and the tests
  // agree on their spelling (ADR-0012 P1).

  // The context every hosted searchable model attaches at startup and dumps at shutdown.
  // Becomes a path component, so it is validated to the closed alphabet by the count log and
  // the container -- the same validation a wire cache_attach gets, at startup instead.
  static const char* const KEY_ATTACH_CONTEXT;      // nnCacheAttachContext
  // How many recorded observations an entry needs before the SHUTDOWN dump will write it.
  // Defaults to the same number an "admission"-less cache_dump gets, which is the one home of
  // that policy (cacheDumpDefaultAdmissionObservations). Zero means "write everything", which
  // is exactly what NNCacheDiskAdmission::minObservations(0) is documented to be -- so the
  // accept-all case needs no second key and cannot disagree with this one.
  static const char* const KEY_SHUTDOWN_MIN_OBSERVATIONS;  // nnCacheShutdownDumpMinObservations

  // An engine with no persisted-cache lifecycle: the behaviour every engine had before this
  // file existed.
  [[nodiscard]] static AnalysisCacheLifecycle none();

  // THE ONE PLACE .cfg TEXT BECOMES A LIFECYCLE. Throws StringError, naming the key at fault
  // and what to do about it, for:
  //   - KEY_SHUTDOWN_MIN_OBSERVATIONS set without KEY_ATTACH_CONTEXT (a write policy for a
  //     write that will never happen);
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
  // Both throw StringError if !isConfigured(), rather than handing back a default that would
  // read as a configured one.
  [[nodiscard]] const std::string& context() const;
  [[nodiscard]] const NNCacheDiskAdmission& shutdownAdmission() const;

 private:
  struct Configured {
    std::string context;
    NNCacheDiskAdmission shutdownAdmission;
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

// What one clean shutdown's dumping did, in the two figures an operator has to be able to read
// off the log: whether anything went wrong, and how much was written.
struct AnalysisCacheShutdownDumpReport {
  // One line per model dumped or failed, in hosting order.
  std::vector<std::string> lines;
  int64_t modelsDumped;
  // Nonzero means this session's work is short. The lines say which model and why.
  int64_t modelsFailed;
  int64_t entriesWritten;
};

// SHUTDOWN. Dumps what=both, under the configured admission, for every hosted model that still
// has the configured context attached. Does nothing when the lifecycle is not configured.
//
// SKIPS A MODEL WHOSE CONTEXT IS NO LONGER ATTACHED, silently and correctly: a client that sent
// cache_detach by wire already decided what happened to that work, and cache_detach refuses to
// discard undumped state unless the client said discardUndumped. Re-attaching it here to dump it
// anyway would overrule that decision.
//
// NEVER THROWS. A shutdown is not a place to raise: the process is on its way out, the client is
// gone, and an exception here would replace an orderly exit with an abort that says less. Every
// failure is CAUGHT, COUNTED and put in `lines` for the caller to print to stderr, and the next
// model is still dumped (ADR-0002: the loss is said, never silent). The observation-loss
// disposition on a failed counts append is the one cacheDumpExecute already owns -- taken counts
// are not re-armed, and its refusal text says how many rows and observations went with them --
// so that text is carried into the line verbatim rather than restated here.
//
// CALLED WITH EVERY ANALYSIS THREAD ALREADY JOINED, which is why it passes an open-request count
// of zero: at that point there is exactly one thread left in the process and no request can be
// open. The count is reported in a dump's response; here it is a fact, not an estimate.
[[nodiscard]] AnalysisCacheShutdownDumpReport analysisCacheShutdownDump(
  const AnalysisModelHosts& hosts,
  AnalysisCacheAttachments& attachments,
  const AnalysisCacheLifecycle& lifecycle
);

#endif  // COMMAND_ANALYSISCACHELIFECYCLE_H_
