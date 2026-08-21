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
  // ordered by recorded observations. Defaults to every key.
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
  // Which earned entries are let onto disk. Required at decode with no default (ledger row
  // 1652): a dump writes to disk, so which entries it admits is not something to infer from a
  // field the client did not send. "Every entry" is still reachable via NNCacheDiskAdmission::
  // all(), just never silently.
  NNCacheDiskAdmission admission;
};

// cache_stats takes no field of its own beyond "model". It still has a decode, because the
// closed-key-set refusal above is the point and a query with a typo'd field must not read as
// a bare stats request.
struct CacheStatsRequest {};

// WHAT AN ABSENT cache_dump "admission" MEANS, as a number with ONE home. Two: the operator's
// standing policy, "store only what has been seen at least twice", which under observation
// currency is reachable across sessions -- a position observed once in one session and once in
// the next clears it (ledger rows 1717/1722).
//
// It is a function on this header rather than a constant in the decoder's translation unit
// because the decoder is no longer its only reader: an engine configured to dump at SHUTDOWN
// with no admission key of its own gets the same default, and two homes for one policy is two
// numbers waiting to disagree (ADR-0012 P1).
[[nodiscard]] uint64_t cacheDumpDefaultAdmissionObservations();

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
  // What the attach's reconcile against level 1 did to this source: how many of its entries
  // level 1 already owned -- and were therefore shadowed on the way in, so the superseded
  // evaluation cannot be served -- and the unpersisted retrievals those entries handed over.
  int64_t entriesLevelOneAlreadyOwned;
  int64_t hitsTransferredToLevelOne;
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

  // WHAT USED TO LIVE HERE, and why it does not any more (recorded rather than silently
  // deleted). This class carried a pair -- noteCountsDumped / anyRequestAcceptedSinceCountsDump
  // -- that stood in for the question cache_detach actually has to ask: does this context hold
  // retrieval counts that are not on disk? It stood in because no surface could answer that
  // without CONSUMING the counts, the delta surface being consuming by design, which is exactly
  // what makes it safe to append.
  //
  // The proxy was documented as over-refusing and never under-refusing, AND THAT WAS FALSE AS
  // STATED. It read the engine's accepted-request counter, which rises when a query is pushed
  // onto the queue and not when it finishes; a counts dump is legal while requests are open and
  // advances its marks at its own harvest; so a request accepted BEFORE a dump and still running
  // after it kept recording retrievals the dump had not written, while no new acceptance was
  // ever observed. The other half of the refusal did not cover the gap either, because a
  // retrieval served entirely out of level 0 never calls set() and so earns no key -- which is
  // the fully pre-warmed card this whole feature is for.
  //
  // The replacement is not a better proxy: it is NNCacheTable::hasUnpersistedHitCountsFor, a
  // non-consuming query that reads the same counters against the same marks the per-context take
  // reads. cacheDetachExecute asks it directly, so nothing in this class stands between the
  // refusal and the fact any more.

 private:
  struct PerModel {
    std::map<std::string, CacheAttachmentRecord> attached;
    std::map<std::string, NNCacheContextId> registered;
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
  // Requests open right now. A dump is legal while requests are open -- every structure it
  // reads is thread-safe and off the hot path -- but the documented posture is dump-at-rest,
  // so the count is reported back in the response and a client that dumped live can see that
  // it did. attach and detach are refused outright while it is nonzero, by the caller, which
  // is the layer that can hold it still.
  int64_t openRequestCount;
};

// THIS LAYER'S MINT FOR THE LEVEL-0 SWAP PERMIT, and it mints for exactly two functions.
//
// NNCacheLevelZeroSwapPermit is the key to NNEvaluator::attachLevelZeroSource and
// detachLevelZeroSource; without one those calls do not compile. This class is the reason the
// protocol layer can still make them, and it grants that power to cacheAttachExecute and
// cacheDetachExecute BY NAME rather than to whoever includes this header: permit() is private and
// those two functions are its only friends. The precondition the permit stands for is established
// one layer up, in the request loop, which refuses cache_attach and cache_detach while any request
// is open (cacheSwapConcurrencyRefusal) -- these two verbs are what it calls once it has.
class AnalysisCacheSwapAuthority {
 private:
  friend nlohmann::json cacheAttachExecute(
    const AnalysisModelHosts& hosts,
    SearchableModelIdx modelIdx,
    AnalysisCacheAttachments& attachments,
    const CacheAttachRequest& request
  );
  friend nlohmann::json cacheDetachExecute(
    const AnalysisModelHosts& hosts,
    SearchableModelIdx modelIdx,
    AnalysisCacheAttachments& attachments,
    const CacheDetachRequest& request
  );
  [[nodiscard]] static NNCacheLevelZeroSwapPermit permit() { return NNCacheLevelZeroSwapPermit(); }
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
// TAKES NO ENGINE COUNTERS, and used to. It read one to decide whether this context might hold
// undumped retrieval counts; it now asks the cache that question directly, so the argument became a
// parameter the body did not honor -- which is a lying signature whether or not anything noticed
// (ADR-0012 P2). The open-request count it also carried was never this function's: the concurrency
// refusal belongs to the caller, which is the layer that can hold the engine still.
[[nodiscard]] nlohmann::json cacheDetachExecute(
  const AnalysisModelHosts& hosts,
  SearchableModelIdx modelIdx,
  AnalysisCacheAttachments& attachments,
  const CacheDetachRequest& request
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
