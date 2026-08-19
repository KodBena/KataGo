#include "../command/analysiscacheactions.h"

#include <algorithm>

#include "../core/global.h"
#include "../core/test.h"
#include "../neuralnet/nncachecountlog.h"
#include "../neuralnet/nncachefileformat.h"
#include "../neuralnet/nnevalcontainer.h"

using namespace std;
using json = nlohmann::json;

//-------------------------------------------------------------------------------------
// The closed key sets
//-------------------------------------------------------------------------------------

const std::set<std::string>& cacheAttachRequestKeys() {
  static const std::set<std::string> keys = {
    "id", "action", "model", "context", "level0", "level1Fill", "foreignModelSources"
  };
  return keys;
}

const std::set<std::string>& cacheDetachRequestKeys() {
  static const std::set<std::string> keys = {"id", "action", "model", "context", "discardUndumped"};
  return keys;
}

const std::set<std::string>& cacheDumpRequestKeys() {
  static const std::set<std::string> keys = {"id", "action", "model", "context", "what", "admission"};
  return keys;
}

const std::set<std::string>& cacheStatsRequestKeys() {
  static const std::set<std::string> keys = {"id", "action", "model"};
  return keys;
}

std::optional<std::string> firstUnexpectedKey(const json& request, const std::set<std::string>& allowed) {
  for(json::const_iterator it = request.begin(); it != request.end(); ++it) {
    if(allowed.find(it.key()) == allowed.end())
      return std::optional<std::string>(it.key());
  }
  return std::optional<std::string>();
}

//-------------------------------------------------------------------------------------
// Decoding
//-------------------------------------------------------------------------------------

namespace {

// The refusal every action's key check produces. One home, so the four decoders say the same
// thing about the same mistake, and a client that hits it on one action recognizes it on the
// next (ADR-0012 P1).
string unexpectedKeyMessage(const string& key, const string& action, const std::set<std::string>& allowed) {
  string message =
    "Unknown field \"" + key + "\" on a " + action + " action. Every field of a cache action decides "
    "which bytes on disk are read or written, so an unrecognized one is refused and the action is NOT "
    "performed -- it is not warned about and analyzed anyway, the way an unknown field of an analysis "
    "query is, and there is no config key that turns this into a warning. The fields " + action +
    " accepts are: ";
  bool first = true;
  for(const string& k: allowed) {
    message += (first ? "" : ", ");
    message += k;
    first = false;
  }
  return message + ".";
}

// "context", present, a string, and a name that can be a path component.
//
// THE ALPHABET IS CHECKED HERE, THROUGH THE ONE FUNCTION THAT OWNS IT. NNCacheFileName::verify is
// the single home of "what may be a path component" and the count log, the container and the
// context set all already call it -- this calls the same function rather than re-authoring the
// rule, so there is no second copy to drift (ADR-0012 P1).
//
// It is checked at THIS boundary, and not left to the deeper layers that also check it, because of
// what the client is told. A refusal from down there arrives through the generic error handler and
// is reported under field "action", which the documented error contract reserves for "the request
// was well-formed and the engine refused to carry it out". An illegal context name is not that: it
// is a request that cannot be read, and a client author following the contract would look for
// field "context". Reporting it from here makes the contract true instead of nearly true.
bool decodeContextField(const json& request, string& context, string& refusalMessage) {
  if(request.find("context") == request.end()) {
    refusalMessage =
      "A cache action must name the context it acts on, as a string \"context\" field. A context is "
      "the client's own opaque name for one body of persisted cache content; the engine reads no "
      "meaning into it and only ever compares it for equality and uses it as a file name.";
    return false;
  }
  if(!request["context"].is_string() || request["context"].get<string>().empty()) {
    refusalMessage = "Must be a nonempty string naming the context this action acts on.";
    return false;
  }
  const string name = request["context"].get<string>();
  try {
    NNCacheFileName::verify(name, "a cache action", "context name");
  }
  catch(const StringError& e) {
    refusalMessage = string(e.what());
    return false;
  }
  context = name;
  return true;
}

// A nonnegative integer field of a nested object.
bool decodeInt64(const json& object, const char* key, int64_t min, int64_t max, int64_t& out, string& refusalMessage) {
  if(!object[key].is_number_integer()) {
    refusalMessage = string("\"") + key + "\" must be an integer.";
    return false;
  }
  const int64_t value = object[key].get<int64_t>();
  if(value < min || value > max) {
    refusalMessage =
      string("\"") + key + "\" must be between " + Global::int64ToString(min) + " and " +
      Global::int64ToString(max) + ", got " + Global::int64ToString(value) + ".";
    return false;
  }
  out = value;
  return true;
}

}  // namespace

std::string cacheSwapConcurrencyRefusal(const std::string& action, int64_t openRequestCount) {
  return
    "Refusing " + action + " while " + Global::int64ToString(openRequestCount) +
    " request(s) are open. Attaching and detaching pre-warmed cache content changes the list that "
    "every cache lookup walks without taking a lock, so doing it under a live request is a "
    "use-after-free rather than a stale read. Wait for the open requests to finish -- or terminate "
    "them -- and send this again. A session boundary, which is where a client attaches and detaches, "
    "is exactly when nothing is open.";
}

CacheActionDecode<CacheAttachRequest> decodeCacheAttach(const json& request) {
  const std::optional<string> unexpected = firstUnexpectedKey(request, cacheAttachRequestKeys());
  if(unexpected.has_value())
    return CacheActionDecode<CacheAttachRequest>::refused(
      unexpected.value(), unexpectedKeyMessage(unexpected.value(), "cache_attach", cacheAttachRequestKeys())
    );

  string context;
  string refusalMessage;
  if(!decodeContextField(request, context, refusalMessage))
    return CacheActionDecode<CacheAttachRequest>::refused("context", refusalMessage);

  // "level0": exactly one of three bounds, or absent for every key. EXACTLY ONE, because the
  // bound type is a closed set of kinds with one per attach: "minLookups and maxBytes
  // together" is not a request this engine can honor, so it is refused here rather than
  // silently interpreted as whichever one the decoder happened to read last.
  NNCacheLevelZeroBound bound = NNCacheLevelZeroBound::all();
  if(request.find("level0") != request.end()) {
    const json& level0 = request["level0"];
    if(!level0.is_object())
      return CacheActionDecode<CacheAttachRequest>::refused(
        "level0",
        "Must be an object holding exactly one of \"minLookups\", \"maxEntries\" or \"maxBytes\", or be "
        "omitted for every persisted key."
      );
    const std::set<std::string> level0Keys = {"minLookups", "maxEntries", "maxBytes"};
    const std::optional<string> unexpectedInner = firstUnexpectedKey(level0, level0Keys);
    if(unexpectedInner.has_value())
      return CacheActionDecode<CacheAttachRequest>::refused(
        "level0", unexpectedKeyMessage(unexpectedInner.value(), "cache_attach \"level0\"", level0Keys)
      );
    if(level0.size() != 1)
      return CacheActionDecode<CacheAttachRequest>::refused(
        "level0",
        "Must hold exactly one of \"minLookups\", \"maxEntries\" or \"maxBytes\", and holds " +
        Global::int64ToString((int64_t)level0.size()) +
        ". A level-0 selection is one bound applied to one ranked order; two bounds would be two "
        "different prefixes and the engine will not choose between them. A client wanting both "
        "composes two attaches."
      );
    int64_t amount = 0;
    if(level0.find("minLookups") != level0.end()) {
      if(!decodeInt64(level0, "minLookups", 0, (int64_t)1 << 40, amount, refusalMessage))
        return CacheActionDecode<CacheAttachRequest>::refused("level0", refusalMessage);
      bound = NNCacheLevelZeroBound::minLookups((uint64_t)amount);
    }
    else if(level0.find("maxEntries") != level0.end()) {
      if(!decodeInt64(level0, "maxEntries", 0, (int64_t)1 << 40, amount, refusalMessage))
        return CacheActionDecode<CacheAttachRequest>::refused("level0", refusalMessage);
      bound = NNCacheLevelZeroBound::maxEntries(amount);
    }
    else {
      if(!decodeInt64(level0, "maxBytes", 0, (int64_t)1 << 46, amount, refusalMessage))
        return CacheActionDecode<CacheAttachRequest>::refused("level0", refusalMessage);
      bound = NNCacheLevelZeroBound::maxBytes(amount);
    }
  }

  // "level1Fill": false, or an object with a byte budget. `false` and absent mean the same
  // thing and both are accepted, because "no fill" is the default and a client that says so
  // explicitly is not making a mistake.
  std::optional<int64_t> levelOneFillMaxBytes;
  if(request.find("level1Fill") != request.end()) {
    const json& fill = request["level1Fill"];
    if(fill.is_boolean()) {
      if(fill.get<bool>())
        return CacheActionDecode<CacheAttachRequest>::refused(
          "level1Fill",
          "true is not a value here: a level-1 fill is bounded in BYTES, and there is no default "
          "budget the engine could invent for you. Pass {\"maxBytes\":N}, or false for no fill."
        );
    }
    else if(fill.is_object()) {
      const std::set<std::string> fillKeys = {"maxBytes"};
      const std::optional<string> unexpectedInner = firstUnexpectedKey(fill, fillKeys);
      if(unexpectedInner.has_value())
        return CacheActionDecode<CacheAttachRequest>::refused(
          "level1Fill", unexpectedKeyMessage(unexpectedInner.value(), "cache_attach \"level1Fill\"", fillKeys)
        );
      if(fill.find("maxBytes") == fill.end())
        return CacheActionDecode<CacheAttachRequest>::refused(
          "level1Fill", "Must hold \"maxBytes\", the byte budget the fill stays under."
        );
      int64_t maxBytes = 0;
      if(!decodeInt64(fill, "maxBytes", 0, (int64_t)1 << 46, maxBytes, refusalMessage))
        return CacheActionDecode<CacheAttachRequest>::refused("level1Fill", refusalMessage);
      levelOneFillMaxBytes = maxBytes;
    }
    else {
      return CacheActionDecode<CacheAttachRequest>::refused(
        "level1Fill", "Must be false, or an object {\"maxBytes\":N}."
      );
    }
  }

  // "foreignModelSources": other loaded models' containers for this same context, attached
  // after this model's own, in this order.
  vector<string> foreignModelSources;
  if(request.find("foreignModelSources") != request.end()) {
    const json& sources = request["foreignModelSources"];
    if(!sources.is_array())
      return CacheActionDecode<CacheAttachRequest>::refused(
        "foreignModelSources",
        "Must be an array of \"internalName\" strings, as the query_models action reports them, in "
        "priority order."
      );
    for(size_t i = 0; i < sources.size(); i++) {
      if(!sources[i].is_string() || sources[i].get<string>().empty())
        return CacheActionDecode<CacheAttachRequest>::refused(
          "foreignModelSources", "Every entry must be a nonempty \"internalName\" string."
        );
      const string name = sources[i].get<string>();
      if(std::find(foreignModelSources.begin(), foreignModelSources.end(), name) != foreignModelSources.end())
        return CacheActionDecode<CacheAttachRequest>::refused(
          "foreignModelSources",
          "\"" + name + "\" is listed twice. The list IS the priority order, so one model cannot hold "
          "two positions in it."
        );
      foreignModelSources.push_back(name);
    }
  }

  return CacheActionDecode<CacheAttachRequest>::decoded(
    CacheAttachRequest{context, bound, levelOneFillMaxBytes, foreignModelSources}
  );
}

CacheActionDecode<CacheDetachRequest> decodeCacheDetach(const json& request) {
  const std::optional<string> unexpected = firstUnexpectedKey(request, cacheDetachRequestKeys());
  if(unexpected.has_value())
    return CacheActionDecode<CacheDetachRequest>::refused(
      unexpected.value(), unexpectedKeyMessage(unexpected.value(), "cache_detach", cacheDetachRequestKeys())
    );

  string context;
  string refusalMessage;
  if(!decodeContextField(request, context, refusalMessage))
    return CacheActionDecode<CacheDetachRequest>::refused("context", refusalMessage);

  bool discardUndumped = false;
  if(request.find("discardUndumped") != request.end()) {
    if(!request["discardUndumped"].is_boolean())
      return CacheActionDecode<CacheDetachRequest>::refused(
        "discardUndumped",
        "Must be true or false. true says this session's undumped work may be thrown away, which is "
        "the only way a detach will throw it away."
      );
    discardUndumped = request["discardUndumped"].get<bool>();
  }

  return CacheActionDecode<CacheDetachRequest>::decoded(CacheDetachRequest{context, discardUndumped});
}

CacheActionDecode<CacheDumpRequest> decodeCacheDump(const json& request) {
  const std::optional<string> unexpected = firstUnexpectedKey(request, cacheDumpRequestKeys());
  if(unexpected.has_value())
    return CacheActionDecode<CacheDumpRequest>::refused(
      unexpected.value(), unexpectedKeyMessage(unexpected.value(), "cache_dump", cacheDumpRequestKeys())
    );

  string context;
  string refusalMessage;
  if(!decodeContextField(request, context, refusalMessage))
    return CacheActionDecode<CacheDumpRequest>::refused("context", refusalMessage);

  // "what" is REQUIRED, and has no default. A dump is the one verb that writes; which of the
  // two files it touched is not something a client should have to infer from a field it did
  // not send.
  if(request.find("what") == request.end() || !request["what"].is_string())
    return CacheActionDecode<CacheDumpRequest>::refused(
      "what",
      "Required, and must be one of \"counts\", \"evaluations\" or \"both\". There is no default: a "
      "dump is the only action that writes to disk, and which of the context's two files it wrote is "
      "not something to infer from an absent field."
    );
  const string whatStr = request["what"].get<string>();
  CacheDumpWhat what;
  if(whatStr == "counts")
    what = CacheDumpWhat::Counts;
  else if(whatStr == "evaluations")
    what = CacheDumpWhat::Evaluations;
  else if(whatStr == "both")
    what = CacheDumpWhat::Both;
  else
    return CacheActionDecode<CacheDumpRequest>::refused(
      "what", "Must be one of \"counts\", \"evaluations\" or \"both\", got \"" + whatStr + "\"."
    );

  NNCacheDiskAdmission admission = NNCacheDiskAdmission::all();
  if(request.find("admission") != request.end()) {
    const json& admissionJson = request["admission"];
    if(!admissionJson.is_object())
      return CacheActionDecode<CacheDumpRequest>::refused(
        "admission",
        "Must be an object holding \"minLookups\", or be omitted to write every entry this context "
        "earned and does not already have on disk."
      );
    const std::set<std::string> admissionKeys = {"minLookups"};
    const std::optional<string> unexpectedInner = firstUnexpectedKey(admissionJson, admissionKeys);
    if(unexpectedInner.has_value())
      return CacheActionDecode<CacheDumpRequest>::refused(
        "admission", unexpectedKeyMessage(unexpectedInner.value(), "cache_dump \"admission\"", admissionKeys)
      );
    if(admissionJson.find("minLookups") == admissionJson.end())
      return CacheActionDecode<CacheDumpRequest>::refused(
        "admission", "Must hold \"minLookups\", the recorded retrievals an entry needs to reach disk."
      );
    int64_t minLookups = 0;
    if(!decodeInt64(admissionJson, "minLookups", 0, (int64_t)1 << 40, minLookups, refusalMessage))
      return CacheActionDecode<CacheDumpRequest>::refused("admission", refusalMessage);
    admission = NNCacheDiskAdmission::minLookups((uint64_t)minLookups);
  }

  return CacheActionDecode<CacheDumpRequest>::decoded(CacheDumpRequest{context, what, admission});
}

CacheActionDecode<CacheStatsRequest> decodeCacheStats(const json& request) {
  const std::optional<string> unexpected = firstUnexpectedKey(request, cacheStatsRequestKeys());
  if(unexpected.has_value())
    return CacheActionDecode<CacheStatsRequest>::refused(
      unexpected.value(), unexpectedKeyMessage(unexpected.value(), "cache_stats", cacheStatsRequestKeys())
    );
  return CacheActionDecode<CacheStatsRequest>::decoded(CacheStatsRequest{});
}

//-------------------------------------------------------------------------------------
// The attachment registry
//-------------------------------------------------------------------------------------

AnalysisCacheAttachments::AnalysisCacheAttachments(size_t numSearchableModels)
  :perModel(numSearchableModels)
{
  for(size_t i = 0; i < perModel.size(); i++)
    perModel[i].requestsAcceptedAtLastCountsDump = 0;
}

AnalysisCacheAttachments::PerModel& AnalysisCacheAttachments::at(SearchableModelIdx modelIdx) {
  testAssert(modelIdx.value() < perModel.size());
  return perModel[modelIdx.value()];
}

const AnalysisCacheAttachments::PerModel& AnalysisCacheAttachments::at(SearchableModelIdx modelIdx) const {
  testAssert(modelIdx.value() < perModel.size());
  return perModel[modelIdx.value()];
}

vector<string> AnalysisCacheAttachments::attachedContexts(SearchableModelIdx modelIdx) const {
  const PerModel& model = at(modelIdx);
  vector<string> out;
  for(map<string, CacheAttachmentRecord>::const_iterator it = model.attached.begin(); it != model.attached.end(); ++it)
    out.push_back(it->first);
  return out;
}

bool AnalysisCacheAttachments::isAttached(SearchableModelIdx modelIdx, const string& context) const {
  const PerModel& model = at(modelIdx);
  return model.attached.find(context) != model.attached.end();
}

const CacheAttachmentRecord& AnalysisCacheAttachments::attachmentFor(
  SearchableModelIdx modelIdx, const string& context
) const {
  const PerModel& model = at(modelIdx);
  const map<string, CacheAttachmentRecord>::const_iterator it = model.attached.find(context);
  if(it == model.attached.end()) {
    string message = "Context \"" + context + "\" is not attached to this model. Attached: ";
    if(model.attached.empty())
      message += "(none)";
    else {
      bool first = true;
      for(map<string, CacheAttachmentRecord>::const_iterator a = model.attached.begin(); a != model.attached.end(); ++a) {
        message += (first ? "" : ", ");
        message += a->first;
        first = false;
      }
    }
    throw StringError(message + ".");
  }
  return it->second;
}

std::optional<NNCacheContextId> AnalysisCacheAttachments::registeredContextId(
  SearchableModelIdx modelIdx, const string& context
) const {
  const PerModel& model = at(modelIdx);
  const map<string, NNCacheContextId>::const_iterator it = model.registered.find(context);
  if(it == model.registered.end())
    return std::optional<NNCacheContextId>();
  return std::optional<NNCacheContextId>(it->second);
}

void AnalysisCacheAttachments::recordAttach(SearchableModelIdx modelIdx, CacheAttachmentRecord record) {
  PerModel& model = at(modelIdx);
  model.registered.insert(std::make_pair(record.context, record.contextId));
  const string context = record.context;
  model.attached.erase(context);
  model.attached.insert(std::make_pair(context, std::move(record)));
}

void AnalysisCacheAttachments::recordDetach(SearchableModelIdx modelIdx, const string& context) {
  at(modelIdx).attached.erase(context);
}

void AnalysisCacheAttachments::noteCountsDumped(SearchableModelIdx modelIdx, int64_t requestsAcceptedSoFar) {
  at(modelIdx).requestsAcceptedAtLastCountsDump = requestsAcceptedSoFar;
}

bool AnalysisCacheAttachments::anyRequestAcceptedSinceCountsDump(
  SearchableModelIdx modelIdx, int64_t requestsAcceptedSoFar
) const {
  return requestsAcceptedSoFar > at(modelIdx).requestsAcceptedAtLastCountsDump;
}

//-------------------------------------------------------------------------------------
// The acts
//-------------------------------------------------------------------------------------

namespace {

// The directory this model's persisted cache lives in, or the refusal to say why there is
// none. One home for that refusal, so every action says the same thing about an engine that
// was started without nnCacheDir.
const string& cacheDirectoryOrThrow(const NNEvaluator& eval) {
  const std::optional<string>& dir = eval.getCacheDirectory();
  if(!dir.has_value())
    throw StringError(
      "The cache actions need a persisted-cache directory, and this engine was started without one. "
      "Set '" + string(NNCacheConfig::KEY_DIR) + " = /some/existing/directory' in the analysis config: "
      "it is where <context>.nncounts and <context>.<model>.nnevals are read and written, and it is "
      "also what decides that a model's cache carries a level-0 resolution list at all."
    );
  return dir.value();
}

// The evaluator a foreignModelSources entry names, refused by name against the loaded models
// when it names nothing this engine hosts. A foreign source is another model's container, and
// reading one needs that model's internal name and model version -- facts that live in the
// model file -- so v1 can only offer containers of models this process actually loaded, and
// says so rather than guessing a version.
NNEvaluator& foreignSourceEvalOrThrow(const AnalysisModelHosts& hosts, const string& name, const NNEvaluator& self) {
  const ModelResolution resolution = hosts.resolve(name);
  if(!resolution.searchableIdx().has_value())
    throw StringError(
      "\"foreignModelSources\" names \"" + name + "\", which this engine cannot read a container for. " +
      resolution.refusal().value() +
      " (A foreign source is another model's own .nnevals file, and reading one needs that model's "
      "internal name and model version, which are facts in the model file -- so only models this "
      "process has loaded can be listed.)"
    );
  NNEvaluator* eval = hosts.searchableEval(resolution.searchableIdx().value());
  if(eval == &self)
    throw StringError(
      "\"foreignModelSources\" names \"" + name + "\", which is the model this attach is FOR. Its own "
      "container is always the first source; listing it again would attach the same file twice, where "
      "the second copy is resident memory no lookup can ever reach."
    );
  return *eval;
}

// WHERE A LEVEL-1 FILL'S ENTRIES ARE DECODED: ordinary heap NNOutputs, each owning its own
// ownership map, because these entries go into the LIVE table and are owned by it entry by
// entry -- ~NNOutput will delete[] the map. That is exactly the opposite of the level-0
// arena's rule, where the map is an interior pointer into a block the arena owns and
// ~NNOutput must never see it, and the two are different sinks for that reason rather than
// one sink with a flag.
class HeapEntrySink final : public NNEvalContainerEntrySink {
 public:
  explicit HeapEntrySink(size_t numEntries) {
    outputs.reserve(numEntries);
    for(size_t i = 0; i < numEntries; i++)
      outputs.push_back(std::make_shared<NNOutput>());
  }

  NNOutput& outputFor(size_t i) override {
    if(i >= outputs.size())
      throw StringError("HeapEntrySink: entry index out of range while filling level 1.");
    return *outputs[i];
  }

  float* ownerMapFor(size_t i, size_t numFloats) override {
    (void)i;
    return new float[numFloats];
  }

  vector<shared_ptr<NNOutput>> outputs;
};

json tailToJson(NNEvalContainerTail tail) {
  return tail == NNEvalContainerTail::Intact ? "intact" : "truncated";
}

json tailToJson(NNCacheCountLogTail tail) {
  return tail == NNCacheCountLogTail::Intact ? "intact" : "truncated";
}

json reclaimToJson(NNCacheHeapReclaim reclaim) {
  // Not a switch: the build turns on -Wswitch-default, so a switch over a closed enum has to
  // carry a default case that can never run and that would then hide a missing case from
  // -Wswitch. Three comparisons and a throw keep both properties -- a new enumerator is a loud
  // failure here rather than a silent "unavailable" (ADR-0002).
  if(reclaim == NNCacheHeapReclaim::Trimmed) return "trimmed";
  if(reclaim == NNCacheHeapReclaim::NothingToTrim) return "nothingToTrim";
  if(reclaim == NNCacheHeapReclaim::Unavailable) return "unavailable";
  throw StringError("NNCacheHeapReclaim: unhandled disposition in cacheDetachExecute's response.");
}

json sourceToJson(const CacheAttachedSource& source) {
  json out;
  out["model"] = source.modelInternalName;
  out["entriesInLevelZero"] = source.entriesInLevelZero;
  out["entriesLeftOver"] = source.entriesLeftOver;
  out["payloadBytes"] = source.arenaTotalBytes;
  out["structureBytes"] = source.structureBytes;
  return out;
}

}  // namespace

json cacheAttachExecute(
  const AnalysisModelHosts& hosts,
  SearchableModelIdx modelIdx,
  AnalysisCacheAttachments& attachments,
  const CacheAttachRequest& request
) {
  NNEvaluator& eval = *hosts.searchableEval(modelIdx);
  const string& directory = cacheDirectoryOrThrow(eval);

  if(attachments.isAttached(modelIdx, request.context))
    throw StringError(
      "Context \"" + request.context + "\" is already attached to model \"" + eval.getInternalModelName() +
      "\". Attaching it twice would put the same content on the resolution list twice, where the second "
      "copy is resident memory no lookup can ever reach. Detach it first if you want to attach it under "
      "different bounds."
    );

  // Every foreign source is resolved BEFORE anything is attached, so a request naming an
  // unknown model refuses without having half-attached the ones before it.
  vector<NNEvaluator*> foreignEvals;
  for(size_t i = 0; i < request.foreignModelSources.size(); i++)
    foreignEvals.push_back(&foreignSourceEvalOrThrow(hosts, request.foreignModelSources[i], eval));

  // The context's NAME is registered once for the life of the process; see
  // AnalysisCacheAttachments. A re-attach after a detach reuses the id the first attach got.
  const std::optional<NNCacheContextId> alreadyRegistered =
    attachments.registeredContextId(modelIdx, request.context);
  const NNCacheContextId contextId =
    alreadyRegistered.has_value() ? alreadyRegistered.value() : eval.attachCacheContext(request.context);

  CacheAttachmentRecord record{request.context, contextId, vector<CacheAttachedSource>(), 0, 0};
  vector<NNCacheLevelZeroCandidate> ownRemainder;
  NNEvalContainerTail containerTail = NNEvalContainerTail::Intact;
  int64_t containerDiscardedTailBytes = 0;
  NNCacheCountLogTail countLogTail = NNCacheCountLogTail::Intact;
  int64_t countLogDiscardedTailBytes = 0;
  double totalMilliseconds = 0.0;

  // ONE SOURCE PER MODEL, this model's own first and each foreign one after it in the order
  // the request listed them. The list order IS the priority: the first source in resolution
  // order that holds a key serves it, so the attaching model's own evaluations always win over
  // a foreign model's for a key both hold.
  vector<NNEvaluator*> sourceEvals;
  sourceEvals.push_back(&eval);
  for(size_t i = 0; i < foreignEvals.size(); i++)
    sourceEvals.push_back(foreignEvals[i]);

  try {
    for(size_t i = 0; i < sourceEvals.size(); i++) {
      const NNEvaluator& sourceEval = *sourceEvals[i];
      NNCacheLevelZeroLoadRequest loadRequest{
        directory, request.context, sourceEval.getInternalModelName(), sourceEval.getModelVersion(),
        request.levelZeroBound
      };
      NNCacheLevelZeroLoad load = nnCacheLoadLevelZero(loadRequest);
      const int64_t structureBytes = load.report.levelZeroStructureBytes;
      const NNCacheLevelZeroSourceId sourceId = eval.attachLevelZeroSource(std::move(load.levelZero));
      record.sources.push_back(
        CacheAttachedSource{
          sourceEval.getInternalModelName(), load.report.entriesInLevelZero, load.report.entriesLeftOver,
          load.report.arenaTotalBytes, structureBytes, sourceId
        }
      );
      totalMilliseconds += load.report.totalMilliseconds;
      if(i == 0) {
        ownRemainder = std::move(load.remainder);
        containerTail = load.report.containerTail;
        containerDiscardedTailBytes = load.report.containerDiscardedTailBytes;
        countLogTail = load.report.countLogTail;
        countLogDiscardedTailBytes = load.report.countLogDiscardedTailBytes;
      }
    }

    // The level-1 fill, out of THIS MODEL'S OWN remainder only. A foreign model's leftovers
    // are not this model's to hold in a live table: an entry in level 1 is one this model
    // would dump into its own container, and dumping another net's evaluation there would put
    // it on disk under this model's name.
    if(request.levelOneFillMaxBytes.has_value() && !ownRemainder.empty()) {
      const NNEvalContainer container = NNEvalContainer::forContextAndModel(
        directory, request.context, eval.getInternalModelName(), eval.getModelVersion()
      );
      // A second read of the file's headers, and only when a fill was asked for: the load
      // above returned the remainder as keys and container positions, having deliberately
      // read no payload for them, so the locations those positions index have to be re-read.
      const NNEvalContainerIndex index = container.loadIndex();
      const vector<NNEvalContainerEntryLocation>& locations = index.entries();

      vector<NNEvalContainerEntryLocation> selected;
      int64_t bytes = 0;
      for(size_t i = 0; i < ownRemainder.size(); i++) {
        if(bytes + ownRemainder[i].residentBytes > request.levelOneFillMaxBytes.value())
          break;
        if(ownRemainder[i].containerIndex >= locations.size())
          throw StringError(
            "The evaluation container changed between the level-0 load and the level-1 fill of context \"" +
            request.context + "\". Nothing was filled."
          );
        selected.push_back(locations[ownRemainder[i].containerIndex]);
        bytes += ownRemainder[i].residentBytes;
      }

      if(!selected.empty()) {
        HeapEntrySink sink(selected.size());
        container.readEntriesInto(selected, sink);
        const NNCacheAttribution attribution = NNCacheAttribution::toContext(contextId);
        for(size_t i = 0; i < sink.outputs.size(); i++) {
          // LoadedFromContainer, which is the whole point of the provenance argument: these
          // bytes are already in the very file a dump of this context appends to, so a dump
          // must not offer them again.
          eval.cacheTable().set(sink.outputs[i], attribution, NNCacheEntryProvenance::LoadedFromContainer);
        }
        record.levelOneFilled = (int64_t)sink.outputs.size();
        record.levelOneFilledBytes = bytes;
      }
    }
  }
  catch(const StringError&) {
    // A partial attach is not a state a client can be handed: the sources already on the list
    // would serve keys the client never asked to have attached, and no response field would
    // say so. Take back exactly what this call put on, then let the refusal through.
    for(size_t i = record.sources.size(); i > 0; i--) {
      const unique_ptr<NNCacheFrozen> taken = eval.detachLevelZeroSource(record.sources[i - 1].sourceId);
      (void)taken;
    }
    throw;
  }

  int64_t entriesInLevelZero = 0;
  int64_t payloadBytes = 0;
  int64_t structureBytes = 0;
  json sources = json::array();
  for(size_t i = 0; i < record.sources.size(); i++) {
    entriesInLevelZero += record.sources[i].entriesInLevelZero;
    payloadBytes += record.sources[i].arenaTotalBytes;
    structureBytes += record.sources[i].structureBytes;
    sources.push_back(sourceToJson(record.sources[i]));
  }
  const int64_t levelOneFilled = record.levelOneFilled;
  const int64_t levelOneFilledBytes = record.levelOneFilledBytes;
  attachments.recordAttach(modelIdx, std::move(record));

  json out;
  out["context"] = request.context;
  out["model"] = eval.getInternalModelName();
  out["entriesInLevelZero"] = entriesInLevelZero;
  out["levelZeroPayloadBytes"] = payloadBytes;
  out["levelZeroStructureBytes"] = structureBytes;
  out["levelOneFilled"] = levelOneFilled;
  out["levelOneFilledBytes"] = levelOneFilledBytes;
  out["sources"] = sources;
  out["containerTail"] = tailToJson(containerTail);
  out["containerDiscardedTailBytes"] = containerDiscardedTailBytes;
  out["countLogTail"] = tailToJson(countLogTail);
  out["countLogDiscardedTailBytes"] = countLogDiscardedTailBytes;
  out["buildMilliseconds"] = totalMilliseconds;
  return out;
}

json cacheDetachExecute(
  const AnalysisModelHosts& hosts,
  SearchableModelIdx modelIdx,
  AnalysisCacheAttachments& attachments,
  const CacheDetachRequest& request,
  const AnalysisEngineCounters& counters
) {
  NNEvaluator& eval = *hosts.searchableEval(modelIdx);
  (void)cacheDirectoryOrThrow(eval);
  const CacheAttachmentRecord& record = attachments.attachmentFor(modelIdx, request.context);

  // WHAT THIS SESSION WOULD LOSE, in the two currencies it can be lost in. Silently discarding
  // it is the silent-failure shape; silently dumping it would make cache_dump no longer the one
  // verb that writes. So the refusal wins, and the client that means to throw a session away
  // says so in the request, where the decision is visible in the log (ADR-0002).
  //
  // THE TWO CHECKS ARE NOT INDEPENDENTLY SOUND AND ARE NOT CLAIMED TO BE. The evaluations half
  // is exact: unpersistedKeysFor is the recorded truth about which of this context's earned keys
  // are not on disk. The counts half is a PROXY, and its gap is stated in full at
  // AnalysisCacheAttachments::anyRequestAcceptedSinceCountsDump -- read it before relying on
  // this refusal for counts. What has held under every adversarial workload built so far is the
  // OR of the two, not either alone.
  const int64_t undumpedEntries = (int64_t)eval.cacheTable().unpersistedKeysFor(record.contextId).size();
  const bool maybeUndumpedCounts =
    attachments.anyRequestAcceptedSinceCountsDump(modelIdx, counters.requestsAcceptedSoFar);
  if(!request.discardUndumped && (undumpedEntries > 0 || maybeUndumpedCounts))
    throw StringError(
      "Refusing to detach context \"" + request.context + "\" from model \"" + eval.getInternalModelName() +
      "\": it holds " + Global::int64ToString(undumpedEntries) + " earned entries that are not on disk" +
      (maybeUndumpedCounts
         ? ", and requests have been served since its counts were last dumped, so it may hold "
           "retrieval counts that are not on disk either"
         : "") +
      ". Send cache_dump first, or send this detach again with \"discardUndumped\":true to throw that "
      "work away deliberately."
    );

  // Detached in reverse attach order, so the resolution list shrinks from the end and no
  // surviving source's position moves under a concurrent reader. (There is no concurrent
  // reader: the protocol refuses this while any request is open, and a debug build asserts it.)
  int64_t sourcesDetached = 0;
  int64_t sourcesWhoseStorageWentBack = 0;
  NNCacheHeapReclaim reclaim = NNCacheHeapReclaim::NothingToTrim;
  for(size_t i = record.sources.size(); i > 0; i--) {
    unique_ptr<NNCacheFrozen> taken = eval.detachLevelZeroSource(record.sources[i - 1].sourceId);
    const NNCacheLevelZeroRelease release = nnCacheReleaseLevelZero(std::move(taken));
    sourcesDetached += 1;
    if(release.storageReleased)
      sourcesWhoseStorageWentBack += 1;
    reclaim = release.reclaim;
  }

  json out;
  out["context"] = request.context;
  out["model"] = eval.getInternalModelName();
  out["sourcesDetached"] = sourcesDetached;
  // OBSERVED, not assumed: a level-0 get hands out its evaluation through an aliasing
  // shared_ptr against the whole arena, so a caller still holding one returned NNOutput keeps
  // the entire arena alive. This says whether the memory actually went.
  out["storageReleased"] = sourcesWhoseStorageWentBack == sourcesDetached;
  out["heapReclaim"] = reclaimToJson(reclaim);
  out["discardedUndumpedEntries"] = request.discardUndumped ? undumpedEntries : (int64_t)0;
  attachments.recordDetach(modelIdx, request.context);
  return out;
}

json cacheDumpExecute(
  const AnalysisModelHosts& hosts,
  SearchableModelIdx modelIdx,
  AnalysisCacheAttachments& attachments,
  const CacheDumpRequest& request,
  const AnalysisEngineCounters& counters
) {
  NNEvaluator& eval = *hosts.searchableEval(modelIdx);
  const string& directory = cacheDirectoryOrThrow(eval);
  const CacheAttachmentRecord& record = attachments.attachmentFor(modelIdx, request.context);
  const bool wantsCounts = request.what != CacheDumpWhat::Evaluations;
  const bool wantsEvaluations = request.what != CacheDumpWhat::Counts;

  const NNCacheCountLog log = NNCacheCountLog::forContext(directory, request.context);

  json out;
  out["context"] = request.context;
  out["model"] = eval.getInternalModelName();
  out["openRequestsAtDump"] = counters.openRequestCount;

  if(wantsCounts) {
    // THE ONE CASE THIS VERSION REFUSES, and it is a limit of the surface underneath rather
    // than a policy. The counts a dump may append are the table's UNPERSISTED DELTA, and the
    // delta surface is whole-table: there is no per-context delta. With one context attached
    // that is exact, because everything the table earned belongs to that context. With two,
    // writing the whole table's delta into one context's file would file the other context's
    // retrievals under this card, and no field of the response would say so. So it is refused
    // by name, and the evaluations leg -- which IS per context -- still works.
    const vector<string> attached = attachments.attachedContexts(modelIdx);
    if(attached.size() > 1) {
      string message =
        "Refusing to dump COUNTS for context \"" + request.context + "\" while " +
        Global::int64ToString((int64_t)attached.size()) + " contexts are attached to model \"" +
        eval.getInternalModelName() + "\" (";
      for(size_t i = 0; i < attached.size(); i++)
        message += (i == 0 ? "" : ", ") + attached[i];
      throw StringError(
        message +
        "). A dump appends the retrievals that have not reached the log yet, and that figure is kept "
        "per TABLE, not per context -- so with more than one context attached it cannot be divided "
        "between them, and writing it whole into one context's file would file another context's "
        "retrievals under this one. Detach the others first, or dump \"evaluations\" only, which is "
        "per context and unaffected."
      );
    }

    const NNCacheCountLogAppendResult appended = log.appendDump(NNCacheHitCountDelta::take(eval.cacheTable()));
    const bool compacted = log.compactIfNeeded(NNCacheCountLog::defaultCompactionMultiple());
    attachments.noteCountsDumped(modelIdx, counters.requestsAcceptedSoFar);
    json counts;
    counts["bytesAppended"] = appended.bytesAppended;
    counts["tornTailBytesDiscarded"] = appended.tornTailBytesDiscarded;
    counts["rewroteTheFile"] = appended.rewroteTheFile;
    counts["compacted"] = compacted;
    const NNCacheCountLogContents contents = log.load();
    counts["rowsInLog"] = (int64_t)contents.rows().size();
    counts["unattributedLookups"] = contents.unattributedLookups();
    counts["tail"] = tailToJson(contents.tail());
    out["counts"] = counts;
  }

  if(wantsEvaluations) {
    // THE ORDER IS THE CONTRACT, and it is why the count log is re-read here rather than
    // reused from above: the admission predicate reads the count log AFTER this dump's counts
    // have been appended, so an entry earned and retrieved twice inside this very session is
    // admitted on the strength of this session rather than having to wait for the next one.
    const NNCacheCountLogContents observations = log.load();
    const NNEvalContainer container = NNEvalContainer::forContextAndModel(
      directory, request.context, eval.getInternalModelName(), eval.getModelVersion()
    );
    const NNCacheEvaluationDumpResult result = nnCacheDumpEvaluations(
      container, eval.cacheTable(), record.contextId, request.admission, observations.rows()
    );
    const bool compacted = container.compactIfNeeded(NNEvalContainer::defaultCompactionMultiple());
    json evaluations;
    evaluations["entriesWritten"] = (int64_t)result.plan.keys.size();
    evaluations["bytesAppended"] = result.append.bytesAppended;
    evaluations["tornTailBytesDiscarded"] = result.append.tornTailBytesDiscarded;
    evaluations["rewroteTheFile"] = result.append.rewroteTheFile;
    evaluations["compacted"] = compacted;
    evaluations["markedPersisted"] = result.marked;
    // EVERY EXCLUSION, COUNTED AND NAMED. A dump that wrote 12 of 40,000 earned keys and
    // reported only "12 written" would read the same as a dump that lost 39,988 of them.
    evaluations["alreadyPersisted"] = result.plan.alreadyPersisted;
    evaluations["belowThreshold"] = result.plan.belowThreshold;
    evaluations["notResident"] = result.plan.notResident;
    evaluations["admission"] = request.admission.describe();
    out["evaluations"] = evaluations;
  }

  // The honesty counters, reported verbatim rather than summed away: nonzero means this
  // session's record is short and says by how much.
  const NNCacheAttributionLedger attribution = eval.harvestCacheAttribution();
  if(attribution.isAttributed()) {
    out["noAttributableContextEntries"] = attribution.noAttributableContextEntries();
    out["unrecordedAttributions"] = attribution.unrecordedAttributions();
  }
  return out;
}

json cacheStatsExecute(
  const AnalysisModelHosts& hosts,
  SearchableModelIdx modelIdx,
  const AnalysisCacheAttachments& attachments
) {
  NNEvaluator& eval = *hosts.searchableEval(modelIdx);
  const NNCacheStats stats = eval.cacheTable().stats();

  json out;
  out["model"] = eval.getInternalModelName();
  out["residentEntries"] = stats.residentEntries;
  out["residentPayloadBytes"] = stats.residentPayloadBytes;
  out["fixedStructureBytes"] = stats.fixedStructureBytes;
  // 0 for a chained table, which is bounded by bytes and has no slot capacity -- a fabricated
  // ratio would be worse than an absent one.
  out["capacitySlots"] = stats.capacitySlots;

  if(eval.getCacheDirectory().has_value()) {
    out["cacheDirectory"] = eval.getCacheDirectory().value();
    out["levelZeroSourcesAttached"] = (int64_t)eval.numLevelZeroSources();
  }

  // ONE-SHOT REPORTING, so this is the surface that reports RUNNING TOTALS -- the absolute
  // harvest -- and not the delta a dump appends. Nothing here is written anywhere, which is
  // what makes the absolute the right answer to ask for.
  const NNCacheHitLedger hits = eval.cacheTable().harvestHitCounts();
  if(hits.isCounted()) {
    out["countedKeys"] = (int64_t)hits.entries().size();
    int64_t totalHits = 0;
    for(size_t i = 0; i < hits.entries().size(); i++)
      totalHits += (int64_t)hits.entries()[i].hits;
    out["retrievalsThisSession"] = totalHits;
    out["unrecordedHits"] = hits.unrecordedHits();
  }

  const NNCacheAttributionLedger attribution = eval.harvestCacheAttribution();
  if(attribution.isAttributed()) {
    out["attributedKeys"] = (int64_t)attribution.rows().size();
    out["noAttributableContextEntries"] = attribution.noAttributableContextEntries();
    out["unrecordedAttributions"] = attribution.unrecordedAttributions();
  }

  json contexts = json::array();
  const vector<string> attached = attachments.attachedContexts(modelIdx);
  for(size_t i = 0; i < attached.size(); i++) {
    const CacheAttachmentRecord& record = attachments.attachmentFor(modelIdx, attached[i]);
    json context;
    context["context"] = record.context;
    context["levelOneFilled"] = record.levelOneFilled;
    context["levelOneFilledBytes"] = record.levelOneFilledBytes;
    context["unpersistedEntries"] = (int64_t)eval.cacheTable().unpersistedKeysFor(record.contextId).size();
    json sources = json::array();
    for(size_t s = 0; s < record.sources.size(); s++)
      sources.push_back(sourceToJson(record.sources[s]));
    context["sources"] = sources;
    if(eval.getCacheDirectory().has_value()) {
      const NNCacheCountLogContents contents =
        NNCacheCountLog::forContext(eval.getCacheDirectory().value(), record.context).load();
      context["countLogRows"] = (int64_t)contents.rows().size();
      context["countLogBlocks"] = contents.blocksApplied();
      context["countLogTail"] = tailToJson(contents.tail());
    }
    contexts.push_back(context);
  }
  out["contexts"] = contexts;
  return out;
}
