#include "../command/analysismodels.h"

#include "../core/global.h"
#include "../core/test.h"

using namespace std;

//-------------------------------------------------------------------------------------
// ModelResolution
//-------------------------------------------------------------------------------------

ModelResolution::ModelResolution(std::optional<size_t> idx_, std::optional<std::string> refusalMessage_)
  : idx(idx_), refusalMessage(std::move(refusalMessage_))
{}

ModelResolution ModelResolution::resolved(size_t searchableIdx_) {
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

std::optional<size_t> ModelResolution::searchableIdx() const {
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

ModelResolution resolveModelName(const vector<ModelAddress>& addresses, const string& requestedName) {
  for(size_t i = 0; i < addresses.size(); i++) {
    if(addresses[i].internalName != requestedName)
      continue;
    if(addresses[i].role == ModelRole::HumanCompanion)
      return ModelResolution::companionRefusal(requestedName);
    return ModelResolution::resolved(i);
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

  vector<ModelAddress> addrs;
  vector<NNEvaluator*> evals;
  for(const HostedModel& model: searchable) {
    testAssert(model.address.role == ModelRole::Searchable);
    testAssert(model.eval != NULL);
    addrs.push_back(model.address);
    evals.push_back(model.eval);
  }
  if(companion.has_value()) {
    testAssert(companion.value().address.role == ModelRole::HumanCompanion);
    testAssert(companion.value().eval != NULL);
    addrs.push_back(companion.value().address);
    evals.push_back(companion.value().eval);
  }

  std::optional<string> collision = findInternalNameCollision(addrs);
  if(collision.has_value())
    throw StringError(collision.value());

  return AnalysisModelHosts(std::move(addrs), std::move(evals), searchable.size());
}

size_t AnalysisModelHosts::numSearchable() const {
  return numSearchableModels;
}

NNEvaluator* AnalysisModelHosts::searchableEval(size_t idx) const {
  testAssert(idx < numSearchableModels);
  return evals[idx];
}

ModelResolution AnalysisModelHosts::resolve(const string& requestedName) const {
  return resolveModelName(addrs, requestedName);
}
