#ifndef COMMAND_ANALYSISCACHEACTIONS_H_
#define COMMAND_ANALYSISCACHEACTIONS_H_

#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "../command/analysismodels.h"
#include "../external/nlohmann_json/json.hpp"
#include "../neuralnet/nncachedump.h"
#include "../neuralnet/nncachelevelzero.h"
#include "../neuralnet/nneval.h"

// THE PERSISTED-CACHE ACTIONS OF THE ANALYSIS ENGINE'S JSON PROTOCOL: the boundary at which a
// client's request object becomes a decided act on one model's neural-net cache.
//
// WHAT THE ACTIONS ARE FOR. The analysis engine's neural-net cache normally lives and dies
// with the process. A client that studies the same positions across many sessions -- a
// spaced-repetition trainer is the case this was built for -- wants the evaluations it paid
// for last week back, without paying for them again. So a session ATTACHES a body of
// previously persisted evaluations (a "context") to a model, analyzes against it, DUMPS what
// it newly earned back to disk, and DETACHES. Four verbs, one per client-visible state
// change, plus cache_stats to see what is resident. docs/Analysis_Engine.md is the
// client-facing statement of all five; this header is the engine-side one.
//
// WHERE THE FILES ARE. Under the one directory the operator configures as nnCacheDir, and
// nowhere else: <context>.nncounts holds a context's per-key retrieval counts and
// <context>.<model>.nnevals holds its evaluations, one file per (context, model). KataGo owns
// every byte under that directory. A context name and a model name are validated to a closed
// path-component alphabet before either reaches a path, by the file types themselves.
//
// THIS FILE IS SPLIT FROM analysis.cpp FOR THE REASON analysismodels.h ALREADY WAS: the
// decode boundary is where the interesting refusals live, and a refusal that can only be
// exercised by starting an engine and writing a line of JSON at it is a refusal that gets
// tested once. Everything here is a function of values -- a json object, an evaluator, an
// attachment registry -- so tests/testanalysiscacheactions.cpp exercises it directly, and the
// end-to-end witness then confirms the same refusals over a real socket-shaped surface rather
// than being the only thing that ever sees them.
//
// -----------------------------------------------------------------------------------------
// AN UNKNOWN FIELD IS AN ERROR HERE, NOT A WARNING. This is the one place these actions
// deliberately do NOT follow the engine's existing habit, and the reason is specific.
//
// An ANALYSIS query with an unrecognized top-level key gets a warning and is then analyzed
// anyway, and that warning is switchable off with warnUnusedFields=false. For a misspelled
// analysis knob that is a defensible trade: the client gets an analysis, slightly not the one
// it asked for. For these actions it is not. Every field below changes WHICH BYTES ON DISK
// ARE READ OR WRITTEN -- which context, which model, how much of it, whether counts or
// evaluations go out -- so a misspelled "levelOneFill" silently means "do not fill level 1",
// a misspelled "discardUndumped" silently means "refuse", and a client that configured the
// warning off is told nothing at all. The response would carry no evidence either way.
//
// So each action declares the closed set of keys it accepts, a key outside it is refused by
// name with an error response, and the act is NOT performed (ADR-0002: a boundary validates
// and does not guess; the refusal is not switchable, because a switchable refusal is a
// warning wearing a different word).
// -----------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------
// The decode boundary
//-------------------------------------------------------------------------------------

// A refusal to hand the client: which field was wrong, and what is wrong with it. The two
// halves the engine's existing error response already carries as "field" and "error".
struct CacheActionRefusal {
  std::string field;
  std::string message;
};

// The outcome of decoding one action's request object: the decoded request, or the refusal.
//
// Constructed by the two named factories and by nothing else, so decoded-and-refused cannot
// both be true and neither can both be false -- the shape ModelResolution and
// NNCacheContextResolution already carry, here as one template because the four actions
// decode to four different values while the outcome discipline is one fact (ADR-0012 P1).
template<class T>
class CacheActionDecode {
 public:
  [[nodiscard]] static CacheActionDecode decoded(T value) {
    return CacheActionDecode(std::optional<T>(std::move(value)), std::optional<CacheActionRefusal>());
  }
  [[nodiscard]] static CacheActionDecode refused(std::string field, std::string message) {
    return CacheActionDecode(
      std::optional<T>(), std::optional<CacheActionRefusal>(CacheActionRefusal{std::move(field), std::move(message)})
    );
  }

  // Present exactly when the action may proceed.
  [[nodiscard]] const std::optional<T>& value() const { return value_; }
  // Present exactly when it may not, and then it says why, for the client to read.
  [[nodiscard]] const std::optional<CacheActionRefusal>& refusal() const { return refusal_; }

 private:
  CacheActionDecode(std::optional<T> value, std::optional<CacheActionRefusal> refusal)
    :value_(std::move(value)), refusal_(std::move(refusal)) {}
  std::optional<T> value_;
  std::optional<CacheActionRefusal> refusal_;
};

// The keys each action's request object may carry, "id", "action" and "model" included. One
// home per action, so the decoder and the test read the same list (ADR-0012 P1).
[[nodiscard]] const std::set<std::string>& cacheAttachRequestKeys();
[[nodiscard]] const std::set<std::string>& cacheDetachRequestKeys();
[[nodiscard]] const std::set<std::string>& cacheDumpRequestKeys();
[[nodiscard]] const std::set<std::string>& cacheStatsRequestKeys();

// What one cache_attach was asked for.
struct CacheAttachRequest {
  // The context to attach. Becomes a path component, so it is validated to the closed
  // alphabet by the count log and the container before it reaches a path.
  std::string context;
  // What the client lets into the frozen level-0 structure, out of the container's key set
  // ordered by recorded lookups. Defaults to every key.
  NNCacheLevelZeroBound levelZeroBound;
  // A byte budget for admitting the level-0 remainder into level 1, or nothing for no fill,
  // which is the default. Denominated in resident bytes, the resource that actually exhausts.
  std::optional<int64_t> levelOneFillMaxBytes;
  // Other loaded models whose containers for this same context are attached AFTER this
  // model's own, in this order. The order IS the priority: the first source in resolution
  // order that holds a key serves it. Empty by default, which is "never overlap".
  std::vector<std::string> foreignModelSources;
};

// What one cache_detach was asked for.
struct CacheDetachRequest {
  std::string context;
  // Whether the client accepts losing what this attachment earned and has not dumped. False
  // by default, and then a detach with undumped state is REFUSED rather than silently
  // discarding it or silently writing it -- see cacheDetachExecute.
  bool discardUndumped;
};

// Which of the two files a cache_dump writes. There is no default: a dump is the one act that
// writes, and "which of my two files did that touch" is not a question a client should have to
// infer from a missing field.
enum class CacheDumpWhat {
  Counts,
  Evaluations,
  Both,
};

// What one cache_dump was asked for.
struct CacheDumpRequest {
  std::string context;
  CacheDumpWhat what;
  // Which earned entries are let onto disk, in recorded lookups. Defaults to all of them.
  NNCacheDiskAdmission admission;
};

// cache_stats takes no field of its own beyond "model". It still has a decode, because the
// closed-key-set refusal above is the point and a query with a typo'd field must not read as
// a bare stats request.
struct CacheStatsRequest {};

// The first key of `request` that `allowed` does not hold, in the object's own order, or
// nothing when every key is allowed.
[[nodiscard]] std::optional<std::string> firstUnexpectedKey(
  const nlohmann::json& request,
  const std::set<std::string>& allowed
);

// THE REFUSAL THAT KEEPS attach AND detach OFF THE HOT PATH, in one home so the two acts say
// the same thing and a test can read the same words the client does.
//
// A get walks the level-0 resolution list LOCK-FREE and takes no snapshot, while attach and
// detach mutate the vector it walks. That is a use-after-free, not a torn counter, so it is
// not a hazard to be documented and lived with: attach and detach are REFUSED while any
// request is open, and the message names the count so a client knows what it is waiting for.
// The cost is nothing, because attaching and detaching is what a client does at a session
// boundary -- exactly when no request is open. A lock-free swap that would make this
// unnecessary is deliberately not built; it is complexity with no witnessed need.
[[nodiscard]] std::string cacheSwapConcurrencyRefusal(const std::string& action, int64_t openRequestCount);

// The four decoders. Each applies its closed key set first, then each field in turn, and
// refuses on the first thing it cannot honor.
[[nodiscard]] CacheActionDecode<CacheAttachRequest> decodeCacheAttach(const nlohmann::json& request);
[[nodiscard]] CacheActionDecode<CacheDetachRequest> decodeCacheDetach(const nlohmann::json& request);
[[nodiscard]] CacheActionDecode<CacheDumpRequest> decodeCacheDump(const nlohmann::json& request);
[[nodiscard]] CacheActionDecode<CacheStatsRequest> decodeCacheStats(const nlohmann::json& request);

//-------------------------------------------------------------------------------------
// What is attached
//-------------------------------------------------------------------------------------

// ONE ATTACHED LEVEL-0 SOURCE OF ONE CONTEXT: which model's container it came from, and what
// it contributed. A cache_attach that lists foreign model sources produces several of these
// for one context, in resolution order.
struct CacheAttachedSource {
  std::string modelInternalName;
  int64_t entriesInLevelZero;
  int64_t entriesLeftOver;
  int64_t arenaTotalBytes;
  int64_t structureBytes;
  NNCacheLevelZeroSourceId sourceId;
};

// WHAT ONE cache_attach PUT ON ONE MODEL, so the matching cache_detach takes back exactly
// that and cache_stats reports it without re-reading the files.
struct CacheAttachmentRecord {
  std::string context;
  NNCacheContextId contextId;
  // In the order they were attached, which is the order they resolve in.
  std::vector<CacheAttachedSource> sources;
  int64_t levelOneFilled;
  int64_t levelOneFilledBytes;
};

// THE ATTACHED CONTEXTS OF EVERY HOSTED MODEL, plus the two facts a detach and a dump have to
// remember between actions.
//
// SINGLE-THREADED BY CONSTRUCTION, and it is worth saying which construction: every action in
// this file is executed on the analysis engine's request loop, which is one thread reading one
// input stream. Nothing here takes a lock, and nothing may be called from an analysis thread.
//
// A CONTEXT'S NAME IS REGISTERED ONCE FOR THE LIFE OF THE PROCESS, its content is not. The
// name space a request's "cacheContext" field resolves against (NNCacheContextSet) has no
// detach -- an attributed entry's context id is a position in that set, so removing one would
// make every later id name a different context. So a detach frees a context's level-0 sources
// and leaves its NAME registered, and a later attach of the same context reuses the same id
// rather than being refused as a duplicate. That is what registeredContextId is for.
class AnalysisCacheAttachments {
 public:
  explicit AnalysisCacheAttachments(size_t numSearchableModels);

  // The contexts attached to this model right now, SORTED BY NAME -- not in attach order, which
  // this does not record. It is a set for reporting and for refusal messages; where order is
  // load-bearing, which is the resolution order of one context's sources, it is
  // CacheAttachmentRecord::sources that carries it.
  [[nodiscard]] std::vector<std::string> attachedContexts(SearchableModelIdx modelIdx) const;
  [[nodiscard]] bool isAttached(SearchableModelIdx modelIdx, const std::string& context) const;
  // Throws StringError naming the context if it is not attached, rather than handing back a
  // null the caller must remember to test.
  [[nodiscard]] const CacheAttachmentRecord& attachmentFor(SearchableModelIdx modelIdx, const std::string& context) const;

  // The context id this model registered for `context` on its FIRST attach, if it ever did.
  [[nodiscard]] std::optional<NNCacheContextId> registeredContextId(SearchableModelIdx modelIdx, const std::string& context) const;

  void recordAttach(SearchableModelIdx modelIdx, CacheAttachmentRecord record);
  void recordDetach(SearchableModelIdx modelIdx, const std::string& context);

  // HOW MANY REQUESTS THE ENGINE HAD ACCEPTED WHEN THIS MODEL LAST DUMPED ITS COUNTS, against
  // the engine's own running request count. It is half of what cache_detach's undumped-work
  // refusal reads, and it exists because there is no way to ask a cache table how many
  // retrievals are unpersisted WITHOUT CONSUMING THEM: the delta surface is consuming by
  // design, which is exactly what makes it safe to append. So the question this can ask is
  // "was any request ACCEPTED since the last counts dump", and a yes arms the refusal.
  //
  // WHAT IT IS NOT, STATED FIRST BECAUSE AN EARLIER VERSION OF THIS COMMENT CLAIMED IT WAS.
  // This is NOT independently sound in the safe direction. The counter it reads is bumped when
  // a query is ACCEPTED onto the queue (analysis.cpp, beside the openRequests insert), not when
  // that query finishes; a counts dump is legal while requests are open, and the marks it
  // advances are taken at the moment of its harvest. So a request accepted BEFORE a dump and
  // still running AFTER it keeps recording retrievals that the dump did not write, while its
  // acceptance already predates the dump -- no new acceptance is ever observed, and this reads
  // false. The class of workload where that is the ONLY thing standing between a detach and a
  // silent loss is a session with no new evaluations at all, so that the other half of the
  // refusal cannot fire: a fully pre-warmed context, re-studied with every position already in
  // level 0. That is not an exotic shape -- it is the mature spaced-repetition card this whole
  // feature exists for.
  //
  // SO THE HONEST STATEMENT IS ABOUT THE PAIR, NOT ABOUT THIS. cacheDetachExecute refuses on
  // this OR on unpersistedKeysFor(context) being nonempty, and it is the CONJUNCTION of the two
  // that has held under every adversarial workload anyone has yet built: a request substantial
  // enough to still be open across a dump has, in every constructed case, also evaluated
  // positions that were not already on disk, and that is the check that fired. Isolating this
  // one -- an open request that touches nothing new -- is UNEXERCISED, by this increment's
  // author and by its reviewer; neither could build the workload, and neither is claiming the
  // gap is unreachable.
  //
  // The fix is not a better proxy. It is a NON-CONSUMING "are there unpersisted counts" query,
  // which can only be written where the counters live (nncachetwolevel.cpp) and is filed as its
  // own work. Until then this is a stopgap that is coarse in two further named ways -- the
  // engine's request counter is not per model, so another model's request arms the refusal, and
  // a request that hit nothing arms it too -- each costing a client one extra cache_dump, which
  // writes nothing when there is nothing to write.
  void noteCountsDumped(SearchableModelIdx modelIdx, int64_t requestsAcceptedSoFar);
  [[nodiscard]] bool anyRequestAcceptedSinceCountsDump(SearchableModelIdx modelIdx, int64_t requestsAcceptedSoFar) const;

 private:
  struct PerModel {
    std::map<std::string, CacheAttachmentRecord> attached;
    std::map<std::string, NNCacheContextId> registered;
    int64_t requestsAcceptedAtLastCountsDump;
  };
  [[nodiscard]] PerModel& at(SearchableModelIdx modelIdx);
  [[nodiscard]] const PerModel& at(SearchableModelIdx modelIdx) const;

  std::vector<PerModel> perModel;
};

//-------------------------------------------------------------------------------------
// The acts
//-------------------------------------------------------------------------------------

// Each of these returns the RESULT FIELDS the response carries beside the echoed request, and
// throws StringError -- naming the cause -- for everything the loader, the container, the
// count log or the cache table refuses. The caller turns a StringError into the engine's
// existing error response; nothing here writes to the output stream.

// The two engine-wide figures these acts need and cannot see for themselves. Passed in rather
// than reached for, because both are the request loop's own bookkeeping and this file is not
// the place a second copy of them would live (ADR-0012 P1).
struct AnalysisEngineCounters {
  // Requests the engine has accepted since it started, as the request loop counts them. Read
  // by AnalysisCacheAttachments::anyRequestAcceptedSinceCountsDump; see that method for what
  // it is conservative about.
  int64_t requestsAcceptedSoFar;
  // Requests open right now. A dump is legal while requests are open -- every structure it
  // reads is thread-safe and off the hot path -- but the documented posture is dump-at-rest,
  // so the count is reported back in the response and a client that dumped live can see that
  // it did. attach and detach are refused outright while it is nonzero, by the caller, which
  // is the layer that can hold it still.
  int64_t openRequestCount;
};

// Reads the context's container and count log, builds one frozen level 0 per source, attaches
// them in the order the request named, and optionally fills level 1 with the remainder.
[[nodiscard]] nlohmann::json cacheAttachExecute(
  const AnalysisModelHosts& hosts,
  SearchableModelIdx modelIdx,
  AnalysisCacheAttachments& attachments,
  const CacheAttachRequest& request
);

// Frees the context's level-0 sources and their arenas, and asks the allocator for the pages
// back. REFUSES, naming what would be lost, when the attachment has state newer than its last
// dump and the request did not say discardUndumped.
[[nodiscard]] nlohmann::json cacheDetachExecute(
  const AnalysisModelHosts& hosts,
  SearchableModelIdx modelIdx,
  AnalysisCacheAttachments& attachments,
  const CacheDetachRequest& request,
  const AnalysisEngineCounters& counters
);

// The one write verb. Appends this model's unpersisted retrieval counts to <context>.nncounts,
// and/or the context's not-yet-persisted level-1 entries to <context>.<model>.nnevals.
[[nodiscard]] nlohmann::json cacheDumpExecute(
  const AnalysisModelHosts& hosts,
  SearchableModelIdx modelIdx,
  AnalysisCacheAttachments& attachments,
  const CacheDumpRequest& request,
  const AnalysisEngineCounters& counters
);

// What one model's cache is holding right now, and what each of its attached contexts
// contributed. A reporting call: it walks the table under its region locks, so it is taken
// between searches and not inside one.
[[nodiscard]] nlohmann::json cacheStatsExecute(
  const AnalysisModelHosts& hosts,
  SearchableModelIdx modelIdx,
  const AnalysisCacheAttachments& attachments
);

#endif  // COMMAND_ANALYSISCACHEACTIONS_H_
