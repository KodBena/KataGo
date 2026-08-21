#include "../command/analysismodels.h"

#include <algorithm>
#include <cctype>

#include "../core/fileutils.h"
#include "../core/global.h"
#include "../core/test.h"

using namespace std;

//-------------------------------------------------------------------------------------
// ModelResolution
//-------------------------------------------------------------------------------------

ModelResolution::ModelResolution(std::optional<SearchableModelIdx> idx_, std::optional<std::string> refusalMessage_)
  : idx(idx_), refusalMessage(std::move(refusalMessage_))
{}

ModelResolution ModelResolution::resolved(SearchableModelIdx searchableIdx_) {
  return ModelResolution(searchableIdx_, std::nullopt);
}

ModelResolution ModelResolution::companionRefusal(const string& requestedName) {
  return ModelResolution(
    std::nullopt,
    "Model \"" + requestedName + "\" is the human SL companion model. It participates in searches of the "
    "searchable models as configured, but it cannot itself be the model of an analysis request."
  );
}

ModelResolution ModelResolution::unknownRefusal(const string& requestedName, const vector<ModelAddress>& addresses) {
  string loaded;
  for(const ModelAddress& address: addresses) {
    if(address.role != ModelRole::Searchable)
      continue;
    if(loaded.size() > 0)
      loaded += ", ";
    loaded += "\"" + address.internalName + "\"";
  }
  return ModelResolution(
    std::nullopt,
    "Unknown model \"" + requestedName + "\". Selectable models are: " + loaded +
    ". Names are the \"internalName\" values that the query_models action reports."
  );
}

std::optional<SearchableModelIdx> ModelResolution::searchableIdx() const {
  return idx;
}

std::optional<string> ModelResolution::refusal() const {
  return refusalMessage;
}

//-------------------------------------------------------------------------------------
// The two rules, over the name space alone
//-------------------------------------------------------------------------------------

std::optional<string> findInternalNameCollision(const vector<ModelAddress>& addresses) {
  for(size_t i = 0; i < addresses.size(); i++) {
    for(size_t j = i + 1; j < addresses.size(); j++) {
      if(addresses[i].internalName != addresses[j].internalName)
        continue;
      return
        "Two hosted models share the internal model name \"" + addresses[i].internalName + "\": " +
        addresses[i].sourceLabel + " and " + addresses[j].sourceLabel +
        ". Analysis requests select a model by that name, so two models under one name have no "
        "unambiguous answer and the engine refuses to start rather than serve one of them under the "
        "other's name. Load distinct models, or drop the duplicate.";
    }
  }
  return std::nullopt;
}

vector<ModelAddress> addressesOf(const vector<HostedModel>& models) {
  vector<ModelAddress> addresses;
  for(const HostedModel& model: models)
    addresses.push_back(model.address);
  return addresses;
}

ModelResolution resolveModelName(const vector<ModelAddress>& addresses, const string& requestedName) {
  //Searchable-first, or a searchable model's index is not its position and resolution hands
  //back the wrong model. See the header.
  bool seenCompanion = false;
  for(const ModelAddress& address: addresses) {
    if(address.role == ModelRole::HumanCompanion)
      seenCompanion = true;
    else
      testAssert(!seenCompanion);
  }

  for(size_t i = 0; i < addresses.size(); i++) {
    if(addresses[i].internalName != requestedName)
      continue;
    if(addresses[i].role == ModelRole::HumanCompanion)
      return ModelResolution::companionRefusal(requestedName);
    return ModelResolution::resolved(SearchableModelIdx(i));
  }
  return ModelResolution::unknownRefusal(requestedName, addresses);
}

//-------------------------------------------------------------------------------------
// AnalysisModelHosts
//-------------------------------------------------------------------------------------

AnalysisModelHosts::AnalysisModelHosts(vector<ModelAddress> addrs_, vector<NNEvaluator*> evals_, size_t numSearchableModels_)
  : addrs(std::move(addrs_)), evals(std::move(evals_)), numSearchableModels(numSearchableModels_)
{}

AnalysisModelHosts AnalysisModelHosts::create(vector<HostedModel> searchable, std::optional<HostedModel> companion) {
  if(searchable.size() <= 0)
    throw StringError("AnalysisModelHosts::create - the analysis engine must host at least one searchable model");

  vector<HostedModel> models = std::move(searchable);
  for(const HostedModel& model: models) {
    testAssert(model.address.role == ModelRole::Searchable);
    testAssert(model.eval != NULL);
  }
  if(companion.has_value()) {
    testAssert(companion.value().address.role == ModelRole::HumanCompanion);
    testAssert(companion.value().eval != NULL);
    models.push_back(companion.value());
  }

  const size_t numSearchableModels = models.size() - (companion.has_value() ? 1 : 0);
  vector<ModelAddress> addrs = addressesOf(models);
  vector<NNEvaluator*> evals;
  for(const HostedModel& model: models)
    evals.push_back(model.eval);

  std::optional<string> collision = findInternalNameCollision(addrs);
  if(collision.has_value())
    throw StringError(collision.value());

  return AnalysisModelHosts(std::move(addrs), std::move(evals), numSearchableModels);
}

size_t AnalysisModelHosts::numSearchable() const {
  return numSearchableModels;
}

NNEvaluator* AnalysisModelHosts::searchableEval(SearchableModelIdx idx) const {
  //One of the two places a SearchableModelIdx is unwrapped: this is the storage it indexes.
  testAssert(idx.value() < numSearchableModels);
  return evals[idx.value()];
}

vector<SearchableModelIdx> AnalysisModelHosts::searchableIdxs() const {
  vector<SearchableModelIdx> idxs;
  for(size_t i = 0; i < numSearchableModels; i++)
    idxs.push_back(SearchableModelIdx(i));
  return idxs;
}

ModelResolution AnalysisModelHosts::resolve(const string& requestedName) const {
  return resolveModelName(addrs, requestedName);
}

//-------------------------------------------------------------------------------------
// collectExtraModelFiles
//-------------------------------------------------------------------------------------

vector<ExtraModelFile> collectExtraModelFiles(ConfigParser& cfg, const vector<string>& extraModelFilesFromCommandLine) {
  vector<ExtraModelFile> fromConfig;
  int firstMissingIdx = 0;
  for(;; firstMissingIdx++) {
    const string key = "extraModelFile" + Global::intToString(firstMissingIdx);
    if(!cfg.contains(key))
      break;
    const string file = cfg.getString(key);
    if(!FileUtils::exists(file))
      throw StringError("Config key " + key + " names a file that does not exist: " + file);
    fromConfig.push_back(ExtraModelFile{file, key});
  }

  //A numbered key past the first gap means the numbering is non-contiguous -- e.g. extraModelFile0
  //and extraModelFile2 with no extraModelFile1. Rather than silently host only the contiguous
  //prefix (which would host zero of the intended extra models here, and quietly stop reading
  //keys the operator plainly meant to be read), this is refused, matching the fail-loud style the
  //rest of the codebase applies to holes in numbered config keys. cfg.unusedKeys() is exactly the
  //keys not yet read by any get* call in this process, so it still names extraModelFile2 here even
  //though the loop above already consumed extraModelFile0.
  for(const string& unusedKey: cfg.unusedKeys()) {
    if(!Global::isPrefix(unusedKey, "extraModelFile"))
      continue;
    const string suffix = unusedKey.substr(string("extraModelFile").size());
    if(suffix.size() == 0 || !std::all_of(suffix.begin(), suffix.end(), ::isdigit))
      continue;
    throw StringError(
      "Config key " + unusedKey + " is set, but extraModelFile" + Global::intToString(firstMissingIdx) +
      " is missing. Numbered extraModelFile keys must be contiguous starting from extraModelFile0."
    );
  }

  //UNION WITH DEDUPE BY PATH, not either source overriding the other. The config and the
  //-extra-model flag name the same kind of thing -- another model to host -- and an operator
  //whose deployment tooling manages the config file while a wrapper script still passes
  //-extra-model (the shape this feature exists for) expects models named by BOTH to end up
  //hosted, not one list silently discarding the other. Deduping by exact file-path string (the
  //same comparison match.cpp's nnModelFile dedup uses) means naming the same file both ways hosts
  //it once rather than refusing it as a duplicate-name collision or loading it twice.
  vector<ExtraModelFile> combined = fromConfig;
  for(const string& file: extraModelFilesFromCommandLine) {
    bool alreadyPresent = false;
    for(const ExtraModelFile& existing: combined) {
      if(existing.file == file) {
        alreadyPresent = true;
        break;
      }
    }
    if(!alreadyPresent)
      combined.push_back(ExtraModelFile{file, "-extra-model"});
  }
  return combined;
}
