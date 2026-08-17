#include "../tests/tests.h"

#include "../core/config_parser.h"
#include "../neuralnet/nncache.h"

using namespace std;
using namespace TestCommon;

//-------------------------------------------------------------------------------------
// Helpers
//-------------------------------------------------------------------------------------

static NNCacheConfig parseCacheCfg(const map<string,string>& kvs) {
  ConfigParser cfg(kvs);
  return NNCacheConfig::fromCfg(cfg, 12, 10);
}

// Asserts that the given keys are refused, and that the refusal message mentions
// every one of `mustMention`. The second half is the point: a refusal that does not
// name the valid vocabulary sends the operator back to the source (ADR-0002).
static void expectRefused(const map<string,string>& kvs, const vector<string>& mustMention) {
  bool threw = false;
  string what;
  try {
    parseCacheCfg(kvs);
  }
  // IOError derives from StringError, so this one handler covers both the
  // ConfigParser's own vocabulary refusals and this Port's coherence refusals.
  catch(const StringError& e) { threw = true; what = e.what(); }
  if(!threw)
    throw StringError("NNCacheConfig test: expected a refusal but the config was accepted");
  for(const string& fragment: mustMention) {
    if(what.find(fragment) == string::npos)
      throw StringError(
        "NNCacheConfig test: refusal message did not mention '" + fragment + "'. Message was: " + what
      );
  }
}

static void expectAccepted(const map<string,string>& kvs) {
  parseCacheCfg(kvs);
}

//-------------------------------------------------------------------------------------

void Tests::runNNCacheConfigTests() {
  cout << "Running nn cache config tests" << endl;

  //---- Defaults reproduce today's behaviour exactly ----
  {
    NNCacheConfig config = parseCacheCfg({});
    testAssert(config.isStatusQuo());
    testAssert(config.shape.isStatusQuo());
    testAssert(config.shape.scheme() == NNCacheCollisionScheme::Direct);
    testAssert(config.shape.ways() == 1);
    testAssert(config.shape.maxBytes() == 0);
    testAssert(config.admission == NNCacheAdmissionPolicy::Always);
    testAssert(config.sizePowerOfTwo == 12);
    testAssert(config.mutexPoolSizePowerOfTwo == 10);

    NNCacheConfig statusQuo = NNCacheConfig::statusQuo(12,10);
    testAssert(statusQuo.isStatusQuo());
    testAssert(statusQuo.shape.scheme() == config.shape.scheme());
  }

  //---- nnCacheCollision: accepted and refused ----
  {
    NNCacheConfig config = parseCacheCfg({{"nnCacheCollision","direct"}});
    testAssert(config.isStatusQuo());
  }
  expectRefused({{"nnCacheCollision","lru"}}, {"direct","linearprobe","quadraticprobe","chain"});
  expectRefused({{"nnCacheCollision","Direct"}}, {"direct","chain"});
  expectRefused({{"nnCacheCollision",""}}, {"direct","chain"});

  //---- The coherence rule: eviction is meaningless under 1-way direct mapping ----
  // A collision under direct mapping presents exactly one candidate victim, so there
  // is no policy to choose. Every eviction value is refused here, not just the exotic
  // ones, and the message says why and names what to set instead.
  expectRefused(
    {{"nnCacheCollision","direct"},{"nnCacheEviction","lru"}},
    {"nnCacheEviction","direct","one candidate victim","linearprobe","quadraticprobe","chain"}
  );
  expectRefused(
    {{"nnCacheEviction","lru"}},  // direct is the default, so the same refusal must fire with no collision key
    {"nnCacheEviction","one candidate victim"}
  );
  expectRefused({{"nnCacheEviction","random"}}, {"nnCacheEviction","one candidate victim"});
  expectRefused({{"nnCacheEviction","lfu"}}, {"nnCacheEviction","one candidate victim"});
  expectRefused({{"nnCacheEviction","none"}}, {"nnCacheEviction","one candidate victim"});
  // And the same for the other two keys a direct-mapped table cannot honor.
  expectRefused({{"nnCacheWays","4"}}, {"nnCacheWays","direct","1-way"});
  expectRefused(
    {{"nnCacheMaxBytes","1000000"}},
    {"nnCacheMaxBytes","direct","no capacity-driven","chain","SLOTS"}
  );

  //---- Probed schemes ----
  {
    NNCacheConfig config = parseCacheCfg({{"nnCacheCollision","linearprobe"},{"nnCacheEviction","lru"}});
    testAssert(!config.isStatusQuo());
    testAssert(config.shape.scheme() == NNCacheCollisionScheme::LinearProbe);
    testAssert(config.shape.eviction() == NNCacheEvictionPolicy::Lru);
    testAssert(config.shape.ways() == 2);
    testAssert(config.shape.maxBytes() == 0);
  }
  {
    NNCacheConfig config = parseCacheCfg(
      {{"nnCacheCollision","quadraticprobe"},{"nnCacheEviction","lfu"},{"nnCacheWays","8"}}
    );
    testAssert(config.shape.scheme() == NNCacheCollisionScheme::QuadraticProbe);
    testAssert(config.shape.eviction() == NNCacheEvictionPolicy::Lfu);
    testAssert(config.shape.ways() == 8);
  }
  expectAccepted({{"nnCacheCollision","linearprobe"},{"nnCacheEviction","random"}});
  expectRefused(
    {{"nnCacheCollision","linearprobe"}},
    {"nnCacheEviction","required","random","lru","lfu"}
  );
  expectRefused(
    {{"nnCacheCollision","linearprobe"},{"nnCacheEviction","none"}},
    {"nnCacheEviction","none","chain","random","lru","lfu"}
  );
  expectRefused(
    // ConfigParser's own range check fires first here and already names the valid
    // range, so this is the message the operator actually sees.
    {{"nnCacheCollision","linearprobe"},{"nnCacheEviction","lru"},{"nnCacheWays","1"}},
    {"nnCacheWays","range 2 to 1024"}
  );
  expectRefused(
    {{"nnCacheCollision","linearprobe"},{"nnCacheEviction","lru"},{"nnCacheMaxBytes","1000"}},
    {"nnCacheMaxBytes","linearprobe","chain"}
  );
  expectRefused(
    {{"nnCacheCollision","linearprobe"},{"nnCacheEviction","mru"}},
    {"random","lru","lfu","none"}
  );

  //---- Chaining ----
  // The chained table's CAPACITY eviction order used to be recency, in force but
  // unnameable: this key was required to say `none` here, which asserted that no policy
  // was chosen while one was. It now names the capacity victim, and defaults to the
  // order that was already running, so a config that says nothing behaves identically.
  {
    NNCacheConfig config = parseCacheCfg({{"nnCacheCollision","chain"},{"nnCacheMaxBytes","4000000000"}});
    testAssert(!config.isStatusQuo());
    testAssert(config.shape.scheme() == NNCacheCollisionScheme::Chain);
    testAssert(config.shape.eviction() == NNCacheEvictionPolicy::Lru);
    testAssert(config.shape.maxBytes() == (int64_t)4000000000LL);
  }
  // All three orders are now sayable, which is the whole point of the change.
  {
    NNCacheConfig config = parseCacheCfg(
      {{"nnCacheCollision","chain"},{"nnCacheMaxBytes","4000000000"},{"nnCacheEviction","lru"}}
    );
    testAssert(config.shape.eviction() == NNCacheEvictionPolicy::Lru);
  }
  {
    NNCacheConfig config = parseCacheCfg(
      {{"nnCacheCollision","chain"},{"nnCacheMaxBytes","4000000000"},{"nnCacheEviction","random"}}
    );
    testAssert(config.shape.eviction() == NNCacheEvictionPolicy::Random);
  }
  {
    NNCacheConfig config = parseCacheCfg(
      {{"nnCacheCollision","chain"},{"nnCacheMaxBytes","4000000000"},{"nnCacheEviction","lfu"}}
    );
    testAssert(config.shape.eviction() == NNCacheEvictionPolicy::Lfu);
  }
  // The one acceptance this change REMOVES, and it is removed rather than quietly
  // reinterpreted: `none` used to be the required word here and is now refused, because
  // a chained table's byte budget is always enforced, so a victim is always chosen and
  // "no eviction" is not a shape this table can be. The message must say what the word
  // used to mean and which value reproduces the old behaviour, or the operator has to
  // read the source to migrate a config that used to be correct.
  expectRefused(
    {{"nnCacheCollision","chain"},{"nnCacheEviction","none"},{"nnCacheMaxBytes","1000"}},
    {"nnCacheEviction","none","chain","nnCacheMaxBytes","lru","random","lfu"}
  );
  expectRefused(
    {{"nnCacheCollision","chain"}},
    {"nnCacheMaxBytes","required","chain","byte budget"}
  );
  expectRefused(
    {{"nnCacheCollision","chain"},{"nnCacheMaxBytes","1000"},{"nnCacheWays","4"}},
    {"nnCacheWays","chain"}
  );
  expectRefused({{"nnCacheCollision","chain"},{"nnCacheMaxBytes","0"}}, {"nnCacheMaxBytes"});
  expectRefused({{"nnCacheCollision","chain"},{"nnCacheMaxBytes","-1"}}, {"nnCacheMaxBytes"});

  //---- nnCacheAdmission ----
  {
    NNCacheConfig config = parseCacheCfg({{"nnCacheAdmission","always"}});
    testAssert(config.isStatusQuo());
    testAssert(config.admission == NNCacheAdmissionPolicy::Always);
  }
  {
    NNCacheConfig config = parseCacheCfg({{"nnCacheAdmission","secondsighting"}});
    testAssert(!config.isStatusQuo());
    testAssert(config.admission == NNCacheAdmissionPolicy::SecondSighting);
  }
  expectRefused({{"nnCacheAdmission","never"}}, {"always","secondsighting"});

  //---- The coherence rule again, at the type layer rather than the cfg layer ----
  // directMapped() carries no eviction argument, so "direct + lru" has no
  // representation to construct; asking a direct-mapped shape for its eviction
  // policy is itself a category error and says so.
  {
    NNCacheShape shape = NNCacheShape::directMapped();
    bool threw = false;
    try { (void)shape.eviction(); }
    catch(const StringError&) { threw = true; }
    testAssert(threw);
  }
  // And the same for the chained coherence rule, at the factory rather than at the cfg
  // Port: the type refuses `none` too, so a caller that bypasses the .cfg text cannot
  // construct a chained shape with no capacity policy either.
  {
    bool threw = false;
    try { (void)NNCacheShape::chained(4000000000LL, NNCacheEvictionPolicy::None); }
    catch(const StringError&) { threw = true; }
    testAssert(threw);
  }

  //---- Every shape the config accepts now builds ----
  // These two used to assert a not-implemented-yet refusal. They now assert the
  // opposite, which is what the policy work delivered; what each shape then DOES is
  // asserted in testnncachepolicy.cpp, not here. The refusals above are untouched,
  // and that is the point: the config Port's contract did not move.
  {
    NNCacheConfig config = parseCacheCfg({{"nnCacheCollision","linearprobe"},{"nnCacheEviction","lru"}});
    testAssert(NNCacheTable::create(config) != nullptr);
  }
  {
    NNCacheConfig config = parseCacheCfg({{"nnCacheAdmission","secondsighting"}});
    testAssert(NNCacheTable::create(config) != nullptr);
  }
  {
    NNCacheConfig config = parseCacheCfg(
      {{"nnCacheCollision","quadraticprobe"},{"nnCacheEviction","lfu"},{"nnCacheWays","4"}}
    );
    testAssert(NNCacheTable::create(config) != nullptr);
  }
  {
    NNCacheConfig config = parseCacheCfg({{"nnCacheCollision","chain"},{"nnCacheMaxBytes","100000000"}});
    testAssert(NNCacheTable::create(config) != nullptr);
  }

  //---- Byte accounting: per-entry footprint is not a constant ----
  // sizeof(NNOutput) alone under-counts by roughly a factor of two in the regime
  // that actually matters -- an analysis client asking for ownership turns the
  // ownermap on for every node in the search, not just the root.
  {
    NNOutput bare;
    bare.nnXLen = 19;
    bare.nnYLen = 19;
    testAssert(nnOutputFootprintBytes(bare) == sizeof(NNOutput));

    NNOutput withOwnerMap;
    withOwnerMap.nnXLen = 19;
    withOwnerMap.nnYLen = 19;
    withOwnerMap.whiteOwnerMap = new float[19*19];
    testAssert(nnOutputFootprintBytes(withOwnerMap) == sizeof(NNOutput) + 19*19*sizeof(float));

    NNOutput withBoth;
    withBoth.nnXLen = 19;
    withBoth.nnYLen = 19;
    withBoth.whiteOwnerMap = new float[19*19];
    withBoth.noisedPolicyProbs = new float[NNPos::MAX_NN_POLICY_SIZE];
    testAssert(
      nnOutputFootprintBytes(withBoth) ==
      sizeof(NNOutput) + 19*19*sizeof(float) + NNPos::MAX_NN_POLICY_SIZE*sizeof(float)
    );
    // The two modes really are different sizes; a budget multiplying a count by any
    // one constant would be wrong for the other mode.
    testAssert(nnOutputFootprintBytes(withOwnerMap) > nnOutputFootprintBytes(bare));
  }

  //---- The default table still behaves like a cache ----
  {
    std::unique_ptr<NNCacheTable> table = NNCacheTable::create(NNCacheConfig::statusQuo(10,6));
    testAssert(table != nullptr);

    Hash128 hash(0x1234567812345678ULL, 0x8765432187654321ULL);
    shared_ptr<NNOutput> got;
    testAssert(!table->get(hash,got));
    testAssert(got == nullptr);

    shared_ptr<NNOutput> entry = std::make_shared<NNOutput>();
    entry->nnHash = hash;
    entry->nnXLen = 19;
    entry->nnYLen = 19;
    table->set(entry);

    testAssert(table->get(hash,got));
    testAssert(got != nullptr);
    testAssert(got->nnHash == hash);

    Hash128 otherHash(0xdeadbeefdeadbeefULL, 0xfeedfacefeedfaceULL);
    testAssert(!table->get(otherHash,got));
    testAssert(got == nullptr);

    testAssert(table->get(hash,got));
    table->clear();
    testAssert(!table->get(hash,got));
  }

  cout << "nn cache config tests passed" << endl;
}
