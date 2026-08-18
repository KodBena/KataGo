#include "../tests/tests.h"

#include "../command/analysismodels.h"

// The analysis engine's model NAME SPACE, exercised without loading a neural net.
//
// Everything asserted here is a property of the names alone -- which name selects which model,
// which name is refused and with what said to the client -- so it is written against
// ModelAddress, the addressing half of a hosted model, and needs no NNEvaluator. The other half,
// that the selected model is the one that actually runs the search, is not a property of names
// and is not witnessed here: it is witnessed end to end against a running engine with two nets
// loaded, which is the only place the served evaluation exists to be observed.
//
// The refusal MESSAGES are asserted on, deliberately. They are not incidental prose: a collision
// message that does not name what collided, and an unknown-name message that does not say what
// the valid names are, leave an operator with a refusal they cannot act on, which is the whole
// difference between failing loudly and merely failing.

using namespace std;

namespace {

ModelAddress searchable(const string& internalName, const string& sourceLabel) {
  return ModelAddress{internalName, sourceLabel, ModelRole::Searchable};
}

ModelAddress companion(const string& internalName, const string& sourceLabel) {
  return ModelAddress{internalName, sourceLabel, ModelRole::HumanCompanion};
}

void testDistinctNamesCollideWithNothing() {
  const vector<ModelAddress> addresses = {
    searchable("kata1-b18-alpha", "-model /nets/alpha.bin.gz"),
    searchable("kata1-b18-beta", "-extra-model /nets/beta.bin.gz"),
    companion("kata1-human", "-human-model /nets/human.bin.gz"),
  };
  testAssert(!findInternalNameCollision(addresses).has_value());
}

void testDuplicateNameIsRefusedAndTheRefusalNamesTheCollision() {
  // The same file configured twice: one internal name, two models, no way to tell them apart.
  const vector<ModelAddress> addresses = {
    searchable("kata1-b18-alpha", "-model /nets/alpha.bin.gz"),
    searchable("kata1-b18-alpha", "-extra-model /nets/copy-of-alpha.bin.gz"),
  };
  const std::optional<string> refusal = findInternalNameCollision(addresses);
  testAssert(refusal.has_value());
  testAssert(refusal.value().find("kata1-b18-alpha") != string::npos);
  testAssert(refusal.value().find("/nets/alpha.bin.gz") != string::npos);
  testAssert(refusal.value().find("/nets/copy-of-alpha.bin.gz") != string::npos);

  // The companion is in the same name space: a companion sharing a searchable model's name is the
  // same ambiguity, and is refused for the same reason rather than tolerated as a different kind.
  const vector<ModelAddress> withCompanionCollision = {
    searchable("kata1-b18-alpha", "-model /nets/alpha.bin.gz"),
    companion("kata1-b18-alpha", "-human-model /nets/alpha.bin.gz"),
  };
  testAssert(findInternalNameCollision(withCompanionCollision).has_value());

  // The stub net calls itself "random" whatever file it came from, so two stubs collide by
  // construction -- the case that is easiest to hit by accident and least visible once served.
  const vector<ModelAddress> twoStubs = {
    searchable("random", "-model /dev/null"),
    searchable("random", "-extra-model /dev/null"),
  };
  testAssert(findInternalNameCollision(twoStubs).has_value());
}

void testEachNameResolvesToItsOwnModel() {
  const vector<ModelAddress> addresses = {
    searchable("kata1-b18-alpha", "-model /nets/alpha.bin.gz"),
    searchable("kata1-b18-beta", "-extra-model /nets/beta.bin.gz"),
    searchable("kata1-b18-gamma", "-extra-model /nets/gamma.bin.gz"),
  };
  for(size_t i = 0; i<addresses.size(); i++) {
    const ModelResolution resolution = resolveModelName(addresses, addresses[i].internalName);
    testAssert(resolution.searchableIdx().has_value());
    testAssert(resolution.searchableIdx().value() == i);
    testAssert(!resolution.refusal().has_value());
  }
}

void testUnknownNameIsRefusedRatherThanFallingBackToTheDefault() {
  const vector<ModelAddress> addresses = {
    searchable("kata1-b18-alpha", "-model /nets/alpha.bin.gz"),
    searchable("kata1-b18-beta", "-extra-model /nets/beta.bin.gz"),
  };
  // A near miss, which is what a typo actually looks like, and which the primary model would
  // happily answer if the name were coerced instead of resolved.
  const ModelResolution resolution = resolveModelName(addresses, "kata1-b18-alpah");
  testAssert(!resolution.searchableIdx().has_value());
  testAssert(resolution.refusal().has_value());
  testAssert(resolution.refusal().value().find("kata1-b18-alpah") != string::npos);
  // The refusal is actionable: it says what the client could have asked for.
  testAssert(resolution.refusal().value().find("kata1-b18-alpha") != string::npos);
  testAssert(resolution.refusal().value().find("kata1-b18-beta") != string::npos);

  // The empty string is a name like any other, and is not a way of asking for the default: a
  // request that wants the default omits the field.
  testAssert(!resolveModelName(addresses, "").searchableIdx().has_value());
}

void testTheCompanionModelIsNamedButNotSearchable() {
  const vector<ModelAddress> addresses = {
    searchable("kata1-b18-alpha", "-model /nets/alpha.bin.gz"),
    companion("kata1-human", "-human-model /nets/human.bin.gz"),
  };
  const ModelResolution resolution = resolveModelName(addresses, "kata1-human");
  testAssert(!resolution.searchableIdx().has_value());
  testAssert(resolution.refusal().has_value());
  // The client read this name from query_models, so the refusal says what the model IS rather
  // than claiming it does not exist.
  testAssert(resolution.refusal().value().find("kata1-human") != string::npos);
  testAssert(resolution.refusal().value().find("companion") != string::npos);
}

void testAResolutionIsEitherAnIndexOrARefusalAndNeverBoth() {
  const ModelResolution resolved = ModelResolution::resolved(3);
  testAssert(resolved.searchableIdx().has_value() && resolved.searchableIdx().value() == 3);
  testAssert(!resolved.refusal().has_value());

  const ModelResolution refusedAsCompanion = ModelResolution::companionRefusal("kata1-human");
  testAssert(!refusedAsCompanion.searchableIdx().has_value());
  testAssert(refusedAsCompanion.refusal().has_value());

  const ModelResolution refusedAsUnknown = ModelResolution::unknownRefusal("nope", {});
  testAssert(!refusedAsUnknown.searchableIdx().has_value());
  testAssert(refusedAsUnknown.refusal().has_value());
}

}  // namespace

void Tests::runAnalysisModelNameSpaceTests() {
  cout << "Running analysis engine model name space tests" << endl;
  testDistinctNamesCollideWithNothing();
  testDuplicateNameIsRefusedAndTheRefusalNamesTheCollision();
  testEachNameResolvesToItsOwnModel();
  testUnknownNameIsRefusedRatherThanFallingBackToTheDefault();
  testTheCompanionModelIsNamedButNotSearchable();
  testAResolutionIsEitherAnIndexOrARefusalAndNeverBoth();
  cout << "Done" << endl;
}
