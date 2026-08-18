#ifndef COMMAND_ANALYSISMODELS_H_
#define COMMAND_ANALYSISMODELS_H_

#include <optional>
#include <string>
#include <vector>

#include "../neuralnet/nneval.h"

// THE NAME SPACE OF THE MODELS ONE ANALYSIS-ENGINE PROCESS HOSTS.
//
// The analysis engine can host more than one neural net at a time. A request selects one
// by its INTERNAL MODEL NAME -- the name the model file declares for itself, which
// NNEvaluator::getInternalModelName returns and which the query_models action already
// reports as "internalName". That existing name is the only address a model has here.
// Nothing in this header invents an alias, an index the client can see, or a second
// identifier: a model's name has one home, the model file, and the engine reads it.
//
// Two facts make that address usable, and both are enforced here rather than assumed:
//
//  1. NAMES ARE UNIQUE, REFUSED AT LOAD TIME. The internal name is self-declared, so two
//     configured models can present the same one (the same file loaded twice; the debug
//     stub net, which always calls itself "random"). Two models under one name means a
//     request naming it has no single answer, and picking either of them serves the
//     client an evaluation from a net it did not ask for -- silently, since the response
//     carries no evidence of which net ran. So a collision is refused at load, by name,
//     naming both sources; it is never resolved last-one-wins or first-one-wins.
//     The refusal is construction-time: the process does not start.
//
//  2. AN UNKNOWN NAME IS REFUSED, NEVER COERCED TO THE DEFAULT. Resolution is a
//     translate-and-validate boundary. A name that is not loaded is an error response
//     listing the names that are, because the alternative -- falling back to the primary
//     model -- is the same silent wrong-net service that (1) forbids, reached by a typo
//     instead of by a config.
//
// The companion (human SL) model is part of this name space even though it is not
// independently searchable: it appears in query_models, so a client can read its name
// there, and a name that resolves to nothing at all would be a worse answer than one that
// says what the model is. It resolves to its own refusal, and it participates in the
// uniqueness check, since a companion sharing a searchable model's name is the same
// ambiguity as any other collision.

// What a model may be asked to do. The companion model is named but not searchable; the
// distinction is a property of the model, not of the request, so it lives here.
enum class ModelRole {
  Searchable,
  HumanCompanion,
};

// How one hosted model is addressed and where it came from. This is the whole of what
// name resolution and the uniqueness check read -- neither needs an evaluator -- so it is
// separated from the evaluator it belongs to and both are exercisable on their own.
struct ModelAddress {
  // As NNEvaluator::getInternalModelName reports it. The address.
  std::string internalName;
  // Where this model was configured from, in the operator's own vocabulary
  // (e.g. "-model /path/to/net.bin.gz"). Appears in refusal messages so a collision names
  // the two things that collided, not just the name they collided under.
  std::string sourceLabel;
  ModelRole role;
};

// The answer to "which model does this name select", carrying its own reason when the
// answer is not a model. The three outcomes are constructed by the three named factories
// and by nothing else, so "resolved to an index" and "refused with a message" cannot both
// be true, and neither can both be false.
class ModelResolution {
 public:
  // Selects the searchable model at this index.
  static ModelResolution resolved(size_t searchableIdx);
  // The name is loaded, but names the companion model, which is not independently searchable.
  static ModelResolution companionRefusal(const std::string& requestedName);
  // The name is not loaded at all. The message lists what is.
  static ModelResolution unknownRefusal(const std::string& requestedName, const std::vector<ModelAddress>& addresses);

  // Present exactly when the name selected a searchable model.
  [[nodiscard]] std::optional<size_t> searchableIdx() const;
  // Present exactly when it did not, and then it says why, for the client to read.
  [[nodiscard]] std::optional<std::string> refusal() const;

 private:
  ModelResolution(std::optional<size_t> idx, std::optional<std::string> refusalMessage);
  std::optional<size_t> idx;
  std::optional<std::string> refusalMessage;
};

// The two rules above, as functions of the name space alone.

// The collision refusal for the first pair of addresses sharing an internal name, or
// nothing when every name is distinct. The message names the name and both sources.
[[nodiscard]] std::optional<std::string> findInternalNameCollision(const std::vector<ModelAddress>& addresses);

// Resolves a requested internal name against the name space. Searchable models resolve to
// their index among the searchable models, which is their position in `addresses` -- so the
// vector must list every searchable model before any companion. That is not left to the
// caller's good manners: it is asserted here, because a companion ordered first would shift
// every searchable index by one and hand back a model the request did not name, which is the
// exact failure this whole header exists to make impossible.
[[nodiscard]] ModelResolution resolveModelName(const std::vector<ModelAddress>& addresses, const std::string& requestedName);

// One hosted model: its address and the evaluator that answers for it.
struct HostedModel {
  ModelAddress address;
  NNEvaluator* eval;  // Not owned. Never null.
};

// The addresses of these models, in the same order. The one way to get an address vector out
// of a model vector, so the uniqueness check reads the same list whoever asks it.
[[nodiscard]] std::vector<ModelAddress> addressesOf(const std::vector<HostedModel>& models);

// The models a running analysis engine hosts. Constructed once at load, read-only
// afterwards, shared by every analysis thread.
class AnalysisModelHosts {
 public:
  // The model a request that names none is served by: the -model the engine was started
  // with, whose behaviour this class leaves exactly as it was before more than one model
  // could be hosted.
  static constexpr size_t PRIMARY_SEARCHABLE_IDX = 0;

  // Refuses with a StringError naming the collision if two of these models share an
  // internal name. `searchable` must be non-empty and its first element is the primary.
  // Taking the companion as its own parameter is what makes the searchable-first ordering
  // the resolution indices depend on hold by construction rather than by convention.
  static AnalysisModelHosts create(std::vector<HostedModel> searchable, std::optional<HostedModel> companion);

  [[nodiscard]] size_t numSearchable() const;
  // idx must be < numSearchable().
  [[nodiscard]] NNEvaluator* searchableEval(size_t idx) const;
  [[nodiscard]] ModelResolution resolve(const std::string& requestedName) const;

 private:
  AnalysisModelHosts(std::vector<ModelAddress> addrs, std::vector<NNEvaluator*> evals, size_t numSearchableModels);
  std::vector<ModelAddress> addrs;
  std::vector<NNEvaluator*> evals;  // Parallel to addrs. Not owned.
  size_t numSearchableModels;
};

#endif  // COMMAND_ANALYSISMODELS_H_
