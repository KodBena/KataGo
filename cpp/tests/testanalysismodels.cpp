#include "../tests/tests.h"

#include <sstream>

#include "../command/analysismodels.h"
#include "../core/config_parser.h"
#include "../neuralnet/nneval.h"
#include "../tests/tinymodel.h"

// The analysis engine's model NAME SPACE, in two halves.
//
// The FIRST half is a property of the names alone -- which name selects which model, which name
// is refused and with what said to the client -- so it is written against ModelAddress, the
// addressing half of a hosted model, and needs no NNEvaluator.
//
// The SECOND half is AnalysisModelHosts itself, whose create() takes real evaluators and refuses
// two models that declare the same internal name. Exercising that needs two real, distinctly-named
// neural nets in this process, which the debug stub cannot provide because every stub calls itself
// "random" (nneval.cpp:153). They come instead from the two nets embedded as base64 in
// tinymodel.cpp -- "rect15-b2c16-s13679744-d94886722" and "b1c6nbt" -- loaded through the same
// Setup::initializeNNEvaluator path the engine uses. An earlier version of this file recorded that
// refusal as unexercisable here; that was wrong, and the seam it missed is this one.
//
// What is still NOT witnessed here, because it is not a property of names or of a registry: that
// the model a request selects is the one that actually runs its search. That is observable only in
// a served analysis response, and is witnessed end to end against a running engine with two nets
// loaded (audit-reports/impl-multimodel-hosting-witness.py).
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
    testAssert(resolution.searchableIdx().value() == SearchableModelIdx(i));
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
  const ModelResolution resolved = ModelResolution::resolved(SearchableModelIdx(3));
  testAssert(resolved.searchableIdx().has_value() && resolved.searchableIdx().value() == SearchableModelIdx(3));
  testAssert(!resolved.refusal().has_value());

  const ModelResolution refusedAsCompanion = ModelResolution::companionRefusal("kata1-human");
  testAssert(!refusedAsCompanion.searchableIdx().has_value());
  testAssert(refusedAsCompanion.refusal().has_value());

  const ModelResolution refusedAsUnknown = ModelResolution::unknownRefusal("nope", {});
  testAssert(!refusedAsUnknown.searchableIdx().has_value());
  testAssert(refusedAsUnknown.refusal().has_value());
}

//-------------------------------------------------------------------------------------
// AnalysisModelHosts itself, against real neural nets
//-------------------------------------------------------------------------------------

// Three real evaluators loaded from the two nets embedded in tinymodel.cpp: one of each, plus a
// SECOND load of the first, which is the "same model file configured twice" case -- two distinct
// evaluators that declare one name, exactly what the engine faces when an operator passes the same
// net to -model and -extra-model.
class RealModels {
 public:
  RealModels()
    : dir("tmpanalysismodels"),
      logger(nullptr, false, false, false, false)
  {
    // Only what the loader needs, and small: this test loads models, it does not search with them.
    istringstream cfgIn(
      "nnCacheSizePowerOfTwo = 12\n"
      "nnMutexPoolSizePowerOfTwo = 10\n"
      "numSearchThreads = 1\n"
    );
    cfg.initialize(cfgIn);
    const bool randFileName = true;
    rect15 = TinyModelTest::loadEmbeddedModel(TinyModelTest::EmbeddedModel::Rect15B2C16, dir.path(), logger, cfg, randFileName);
    nbt = TinyModelTest::loadEmbeddedModel(TinyModelTest::EmbeddedModel::B1C6Nbt, dir.path(), logger, cfg, randFileName);
    rect15Again = TinyModelTest::loadEmbeddedModel(TinyModelTest::EmbeddedModel::Rect15B2C16, dir.path(), logger, cfg, randFileName);
  }
  ~RealModels() {
    delete rect15.eval;
    delete nbt.eval;
    delete rect15Again.eval;
  }
  RealModels(const RealModels&) = delete;
  RealModels& operator=(const RealModels&) = delete;

  // A hosted model addressed by the name its own file declares -- the same read analysis.cpp does,
  // rather than a name this test invents.
  HostedModel hosted(const TinyModelTest::LoadedTinyModel& loaded, const string& sourceLabel, ModelRole role) const {
    return HostedModel{ModelAddress{loaded.eval->getInternalModelName(), sourceLabel, role}, loaded.eval};
  }

  TestCommon::ScopedTempDir dir;
  ConfigParser cfg;
  Logger logger;
  TinyModelTest::LoadedTinyModel rect15;
  TinyModelTest::LoadedTinyModel nbt;
  TinyModelTest::LoadedTinyModel rect15Again;
};

// The refusal AnalysisModelHosts::create raised, or nothing if it accepted the models. Named so
// that the throw is observed as a value at each site rather than each site growing its own try.
std::optional<string> createRefusal(vector<HostedModel> searchableModels, std::optional<HostedModel> companionModel) {
  try {
    const AnalysisModelHosts hosts = AnalysisModelHosts::create(std::move(searchableModels), std::move(companionModel));
    (void)hosts;
    return std::nullopt;
  }
  catch(const StringError& e) {
    return string(e.what());
  }
}

// The precondition every test below rests on. Asserted rather than assumed, so that if the embedded
// models are ever replaced by two that share a name, this says so instead of the collision tests
// quietly passing for the wrong reason and the acceptance test quietly failing.
void testTheEmbeddedModelsSupplyTheNamesTheseTestsNeed(const RealModels& models) {
  const string nameA = models.rect15.eval->getInternalModelName();
  const string nameB = models.nbt.eval->getInternalModelName();
  const string nameAAgain = models.rect15Again.eval->getInternalModelName();
  cout << "Real models loaded in-process: \"" << nameA << "\", \"" << nameB << "\"" << endl;
  testAssert(nameA.size() > 0 && nameB.size() > 0);
  testAssert(nameA != nameB);
  // Two distinct evaluators, one declared name: the collision the engine must refuse.
  testAssert(models.rect15.eval != models.rect15Again.eval);
  testAssert(nameA == nameAAgain);
  // Not the debug stub, which would make every name "random" and every test below vacuous.
  testAssert(nameA != "random" && nameB != "random");
}

void testHostsAcceptTwoDistinctlyNamedRealModels(const RealModels& models) {
  const AnalysisModelHosts hosts = AnalysisModelHosts::create(
    {models.hosted(models.rect15, "-model rect15.bin.gz", ModelRole::Searchable),
     models.hosted(models.nbt, "-extra-model nbt.bin.gz", ModelRole::Searchable)},
    std::nullopt
  );
  testAssert(hosts.numSearchable() == 2);
  testAssert(hosts.searchableIdxs().size() == 2);
  testAssert(hosts.searchableEval(hosts.searchableIdxs()[0]) == models.rect15.eval);
  testAssert(hosts.searchableEval(hosts.searchableIdxs()[1]) == models.nbt.eval);
  testAssert(hosts.searchableEval(AnalysisModelHosts::PRIMARY_SEARCHABLE_IDX) == models.rect15.eval);

  // Each name selects its own model, read back through the registry the engine uses.
  const ModelResolution a = hosts.resolve(models.rect15.eval->getInternalModelName());
  const ModelResolution b = hosts.resolve(models.nbt.eval->getInternalModelName());
  testAssert(a.searchableIdx().has_value() && hosts.searchableEval(a.searchableIdx().value()) == models.rect15.eval);
  testAssert(b.searchableIdx().has_value() && hosts.searchableEval(b.searchableIdx().value()) == models.nbt.eval);
  testAssert(!hosts.resolve("not-a-loaded-model").searchableIdx().has_value());
}

void testHostsRefuseTwoRealModelsSharingAnInternalName(const RealModels& models) {
  const string name = models.rect15.eval->getInternalModelName();
  const std::optional<string> refusal = createRefusal(
    {models.hosted(models.rect15, "-model tiny.bin.gz", ModelRole::Searchable),
     models.hosted(models.rect15Again, "-extra-model tiny-copy.bin.gz", ModelRole::Searchable)},
    std::nullopt
  );
  testAssert(refusal.has_value());
  testAssert(refusal.value().find(name) != string::npos);
  testAssert(refusal.value().find("-model tiny.bin.gz") != string::npos);
  testAssert(refusal.value().find("-extra-model tiny-copy.bin.gz") != string::npos);
}

void testHostsRefuseACompanionSharingASearchableModelsName(const RealModels& models) {
  // The companion is addressed by its own declared name too; it collides because it is the same
  // net the primary was loaded from, which is what passing one file to -model and -human-model does.
  const std::optional<string> refusal = createRefusal(
    {models.hosted(models.rect15, "-model tiny.bin.gz", ModelRole::Searchable)},
    models.hosted(models.rect15Again, "-human-model tiny.bin.gz", ModelRole::HumanCompanion)
  );
  testAssert(refusal.has_value());
  testAssert(refusal.value().find(models.rect15.eval->getInternalModelName()) != string::npos);
  testAssert(refusal.value().find("-human-model tiny.bin.gz") != string::npos);
}

void testHostsRefuseHostingNoSearchableModelAtAll() {
  const std::optional<string> refusal = createRefusal({}, std::nullopt);
  testAssert(refusal.has_value());
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
  {
    const RealModels models;
    testTheEmbeddedModelsSupplyTheNamesTheseTestsNeed(models);
    testHostsAcceptTwoDistinctlyNamedRealModels(models);
    testHostsRefuseTwoRealModelsSharingAnInternalName(models);
    testHostsRefuseACompanionSharingASearchableModelsName(models);
    testHostsRefuseHostingNoSearchableModelAtAll();
  }
  cout << "Done" << endl;
}
