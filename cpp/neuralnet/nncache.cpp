#include "../neuralnet/nncache.h"

#include "../core/config_parser.h"
#include "../neuralnet/nncacheimpl.h"
#include "../search/mutexpool.h"

using namespace std;

//-------------------------------------------------------------------------------------
// Policy vocabulary
//-------------------------------------------------------------------------------------

const char* const NNCacheConfig::KEY_COLLISION = "nnCacheCollision";
const char* const NNCacheConfig::KEY_WAYS = "nnCacheWays";
const char* const NNCacheConfig::KEY_EVICTION = "nnCacheEviction";
const char* const NNCacheConfig::KEY_ADMISSION = "nnCacheAdmission";
const char* const NNCacheConfig::KEY_MAX_BYTES = "nnCacheMaxBytes";

static const string COLLISION_DIRECT = "direct";
static const string COLLISION_LINEAR = "linearprobe";
static const string COLLISION_QUADRATIC = "quadraticprobe";
static const string COLLISION_CHAIN = "chain";

static const string EVICTION_RANDOM = "random";
static const string EVICTION_LRU = "lru";
static const string EVICTION_LFU = "lfu";
static const string EVICTION_NONE = "none";

static const string ADMISSION_ALWAYS = "always";
static const string ADMISSION_SECOND_SIGHTING = "secondsighting";

const set<string>& NNCacheConfig::collisionVocabulary() {
  static const set<string> vocab = {COLLISION_DIRECT, COLLISION_LINEAR, COLLISION_QUADRATIC, COLLISION_CHAIN};
  return vocab;
}
const set<string>& NNCacheConfig::evictionVocabulary() {
  static const set<string> vocab = {EVICTION_RANDOM, EVICTION_LRU, EVICTION_LFU, EVICTION_NONE};
  return vocab;
}
const set<string>& NNCacheConfig::admissionVocabulary() {
  static const set<string> vocab = {ADMISSION_ALWAYS, ADMISSION_SECOND_SIGHTING};
  return vocab;
}

static string collisionSchemeToString(NNCacheCollisionScheme scheme) {
  switch(scheme) {
  case NNCacheCollisionScheme::Direct: return COLLISION_DIRECT;
  case NNCacheCollisionScheme::LinearProbe: return COLLISION_LINEAR;
  case NNCacheCollisionScheme::QuadraticProbe: return COLLISION_QUADRATIC;
  case NNCacheCollisionScheme::Chain: return COLLISION_CHAIN;
  }
  throw StringError("NNCacheShape: unhandled collision scheme");
}

static string evictionPolicyToString(NNCacheEvictionPolicy eviction) {
  switch(eviction) {
  case NNCacheEvictionPolicy::Random: return EVICTION_RANDOM;
  case NNCacheEvictionPolicy::Lru: return EVICTION_LRU;
  case NNCacheEvictionPolicy::Lfu: return EVICTION_LFU;
  case NNCacheEvictionPolicy::None: return EVICTION_NONE;
  }
  throw StringError("NNCacheShape: unhandled eviction policy");
}

//-------------------------------------------------------------------------------------
// NNCacheShape
//-------------------------------------------------------------------------------------

NNCacheShape::NNCacheShape(NNCacheCollisionScheme scheme, int ways, NNCacheEvictionPolicy eviction, int64_t maxBytes)
  :scheme_(scheme), ways_(ways), eviction_(eviction), maxBytes_(maxBytes)
{}

NNCacheShape NNCacheShape::directMapped() {
  // Eviction is not a parameter here, and that is the whole point: under 1-way
  // direct mapping a collision presents exactly one candidate victim, so there is
  // no policy to state. The stored eviction_ is a placeholder eviction() refuses
  // to hand out.
  return NNCacheShape(NNCacheCollisionScheme::Direct, 1, NNCacheEvictionPolicy::None, 0);
}

NNCacheShape NNCacheShape::probed(NNCacheCollisionScheme scheme, int ways, NNCacheEvictionPolicy eviction) {
  if(scheme != NNCacheCollisionScheme::LinearProbe && scheme != NNCacheCollisionScheme::QuadraticProbe)
    throw StringError(
      "NNCacheShape::probed: scheme must be one of (" + COLLISION_LINEAR + "|" + COLLISION_QUADRATIC +
      "), got " + collisionSchemeToString(scheme)
    );
  if(ways < 2)
    throw StringError(
      "NNCacheShape::probed: " + string(NNCacheConfig::KEY_WAYS) + " must be at least 2, got " +
      Global::intToString(ways) + ". A 1-way probed table is a direct-mapped table; say " +
      string(NNCacheConfig::KEY_COLLISION) + " = " + COLLISION_DIRECT + " for that."
    );
  if(eviction == NNCacheEvictionPolicy::None)
    throw StringError(
      "NNCacheShape::probed: " + string(NNCacheConfig::KEY_EVICTION) + " = " + EVICTION_NONE +
      " is only valid under " + string(NNCacheConfig::KEY_COLLISION) + " = " + COLLISION_CHAIN +
      ". A probed table has finite ways and must choose a victim; valid here are (" +
      EVICTION_RANDOM + "|" + EVICTION_LRU + "|" + EVICTION_LFU + ")."
    );
  return NNCacheShape(scheme, ways, eviction, 0);
}

NNCacheShape NNCacheShape::chained(int64_t maxBytes) {
  if(maxBytes <= 0)
    throw StringError(
      "NNCacheShape::chained: " + string(NNCacheConfig::KEY_COLLISION) + " = " + COLLISION_CHAIN +
      " requires a positive " + string(NNCacheConfig::KEY_MAX_BYTES) +
      ", since a chained table has no fixed capacity and is bounded only by its byte budget."
    );
  return NNCacheShape(NNCacheCollisionScheme::Chain, 0, NNCacheEvictionPolicy::None, maxBytes);
}

NNCacheEvictionPolicy NNCacheShape::eviction() const {
  if(scheme_ == NNCacheCollisionScheme::Direct)
    throw StringError(
      "NNCacheShape::eviction: a 1-way direct-mapped table has no eviction policy -- a collision "
      "presents exactly one candidate victim, so there is no choice to report."
    );
  return eviction_;
}

string NNCacheShape::toString() const {
  if(scheme_ == NNCacheCollisionScheme::Direct)
    return COLLISION_DIRECT;
  if(scheme_ == NNCacheCollisionScheme::Chain)
    return COLLISION_CHAIN + ", maxBytes " + Global::int64ToString(maxBytes_);
  return collisionSchemeToString(scheme_) + ", " + Global::intToString(ways_) + "-way, " +
    evictionPolicyToString(eviction_);
}

//-------------------------------------------------------------------------------------
// NNCacheConfig -- the .cfg Port
//-------------------------------------------------------------------------------------

NNCacheConfig NNCacheConfig::statusQuo(int sizePowerOfTwo, int mutexPoolSizePowerOfTwo) {
  NNCacheConfig config = {
    sizePowerOfTwo,
    mutexPoolSizePowerOfTwo,
    NNCacheShape::directMapped(),
    NNCacheAdmissionPolicy::Always
  };
  return config;
}

bool NNCacheConfig::isStatusQuo() const {
  return shape.isStatusQuo() && admission == NNCacheAdmissionPolicy::Always;
}

NNCacheConfig NNCacheConfig::fromCfg(ConfigParser& cfg, int sizePowerOfTwo, int mutexPoolSizePowerOfTwo) {
  NNCacheConfig config = NNCacheConfig::statusQuo(sizePowerOfTwo, mutexPoolSizePowerOfTwo);

  // getString(key,possibles) already refuses an unrecognized value naming the whole
  // vocabulary; reuse it rather than re-authoring a second such check (ADR-0012 P1).
  const string collision =
    cfg.contains(KEY_COLLISION) ? cfg.getString(KEY_COLLISION, collisionVocabulary()) : COLLISION_DIRECT;

  if(cfg.contains(KEY_ADMISSION)) {
    const string admission = cfg.getString(KEY_ADMISSION, admissionVocabulary());
    if(admission == ADMISSION_SECOND_SIGHTING)
      config.admission = NNCacheAdmissionPolicy::SecondSighting;
  }

  if(collision == COLLISION_DIRECT) {
    // The coherence rule, refused at the boundary because it cannot be represented past it.
    if(cfg.contains(KEY_EVICTION))
      throw StringError(
        "Key '" + string(KEY_EVICTION) + "' is meaningless under " + string(KEY_COLLISION) + " = " +
        COLLISION_DIRECT + ": a 1-way direct-mapped table gives a collision exactly one candidate "
        "victim, so there is no policy to choose. Either remove '" + string(KEY_EVICTION) +
        "', or set " + string(KEY_COLLISION) + " to one of (" + COLLISION_LINEAR + "|" +
        COLLISION_QUADRATIC + "|" + COLLISION_CHAIN + ")."
      );
    if(cfg.contains(KEY_WAYS))
      throw StringError(
        "Key '" + string(KEY_WAYS) + "' is meaningless under " + string(KEY_COLLISION) + " = " +
        COLLISION_DIRECT + ", which is 1-way by definition. Set " + string(KEY_COLLISION) +
        " to one of (" + COLLISION_LINEAR + "|" + COLLISION_QUADRATIC + ") to choose an associativity."
      );
    if(cfg.contains(KEY_MAX_BYTES))
      throw StringError(
        "Key '" + string(KEY_MAX_BYTES) + "' cannot be honored under " + string(KEY_COLLISION) + " = " +
        COLLISION_DIRECT + ": a direct-mapped table evicts only on collision and has no capacity-driven "
        "eviction, so it cannot be held to a byte budget. Set " + string(KEY_COLLISION) + " = " +
        COLLISION_CHAIN + " to use " + string(KEY_MAX_BYTES) + ", or bound memory with "
        "nnCacheSizePowerOfTwo instead -- noting that that key counts SLOTS, not bytes."
      );
    config.shape = NNCacheShape::directMapped();
    return config;
  }

  if(collision == COLLISION_CHAIN) {
    if(cfg.contains(KEY_WAYS))
      throw StringError(
        "Key '" + string(KEY_WAYS) + "' is meaningless under " + string(KEY_COLLISION) + " = " +
        COLLISION_CHAIN + ": a chain has no fixed associativity."
      );
    if(cfg.contains(KEY_EVICTION) && cfg.getString(KEY_EVICTION, evictionVocabulary()) != EVICTION_NONE)
      throw StringError(
        "Key '" + string(KEY_EVICTION) + "' under " + string(KEY_COLLISION) + " = " + COLLISION_CHAIN +
        " must be " + EVICTION_NONE + " or absent: chaining never evicts on collision, so eviction is "
        "driven by '" + string(KEY_MAX_BYTES) + "' instead."
      );
    if(!cfg.contains(KEY_MAX_BYTES))
      throw StringError(
        "Key '" + string(KEY_MAX_BYTES) + "' is required under " + string(KEY_COLLISION) + " = " +
        COLLISION_CHAIN + ": a chained table has no fixed capacity and is bounded only by its byte budget."
      );
    config.shape = NNCacheShape::chained(cfg.getInt64(KEY_MAX_BYTES, 1, (int64_t)1 << 62));
    return config;
  }

  // Probed schemes.
  const NNCacheCollisionScheme scheme =
    collision == COLLISION_LINEAR ? NNCacheCollisionScheme::LinearProbe : NNCacheCollisionScheme::QuadraticProbe;
  if(cfg.contains(KEY_MAX_BYTES))
    throw StringError(
      "Key '" + string(KEY_MAX_BYTES) + "' cannot be honored under " + string(KEY_COLLISION) + " = " +
      collision + ": a probed table has fixed capacity and evicts on collision, not against a byte budget. "
      "Set " + string(KEY_COLLISION) + " = " + COLLISION_CHAIN + " to use " + string(KEY_MAX_BYTES) + "."
    );
  if(!cfg.contains(KEY_EVICTION))
    throw StringError(
      "Key '" + string(KEY_EVICTION) + "' is required under " + string(KEY_COLLISION) + " = " + collision +
      ": a probed table has more than one candidate victim per bucket, so the choice must be stated. "
      "Valid values are (" + EVICTION_RANDOM + "|" + EVICTION_LRU + "|" + EVICTION_LFU + ")."
    );
  const string evictionStr = cfg.getString(KEY_EVICTION, evictionVocabulary());
  const NNCacheEvictionPolicy eviction =
    evictionStr == EVICTION_RANDOM ? NNCacheEvictionPolicy::Random :
    evictionStr == EVICTION_LRU ? NNCacheEvictionPolicy::Lru :
    evictionStr == EVICTION_LFU ? NNCacheEvictionPolicy::Lfu :
    NNCacheEvictionPolicy::None;
  const int ways = cfg.contains(KEY_WAYS) ? cfg.getInt(KEY_WAYS, 2, 1024) : 2;
  config.shape = NNCacheShape::probed(scheme, ways, eviction);
  return config;
}

//-------------------------------------------------------------------------------------
// Footprint accounting
//-------------------------------------------------------------------------------------

size_t nnOutputFootprintBytes(const NNOutput& out) {
  size_t bytes = sizeof(NNOutput);
  if(out.whiteOwnerMap != NULL)
    bytes += (size_t)out.nnXLen * (size_t)out.nnYLen * sizeof(float);
  if(out.noisedPolicyProbs != NULL)
    bytes += (size_t)NNPos::MAX_NN_POLICY_SIZE * sizeof(float);
  return bytes;
}

//-------------------------------------------------------------------------------------
// The direct-mapped table
//-------------------------------------------------------------------------------------

// Uncomment this to lower the effective hash size down to one where we get true collisions
// #define SIMULATE_TRUE_HASH_COLLISIONS

namespace {

// 1-way direct-mapped, overwrite-on-collision, admit-always. KataGo's cache as it has
// always been; the bodies below are the pre-extraction ones, relocated unchanged.
class NNCacheTableDirect final : public NNCacheTable {
  struct Entry {
    std::shared_ptr<NNOutput> ptr;
    Entry();
    ~Entry();
  };

  Entry* entries;
  MutexPool* mutexPool;
  uint64_t tableSize;
  uint64_t tableMask;
  uint32_t mutexPoolMask;

 public:
  NNCacheTableDirect(int sizePowerOfTwo, int mutexPoolSizePowerOfTwo);
  ~NNCacheTableDirect() override;

  bool get(Hash128 nnHash, std::shared_ptr<NNOutput>& ret) override;
  void set(const std::shared_ptr<NNOutput>& p) override;
  void clear() override;
};

NNCacheTableDirect::Entry::Entry()
  :ptr(nullptr)
{}
NNCacheTableDirect::Entry::~Entry()
{}

NNCacheTableDirect::NNCacheTableDirect(int sizePowerOfTwo, int mutexPoolSizePowerOfTwo) {
  if(sizePowerOfTwo < 0 || sizePowerOfTwo > 63)
    throw StringError("NNCacheTable: Invalid sizePowerOfTwo: " + Global::intToString(sizePowerOfTwo));
  if(mutexPoolSizePowerOfTwo < 0 || mutexPoolSizePowerOfTwo > 31)
    throw StringError("NNCacheTable: Invalid mutexPoolSizePowerOfTwo: " + Global::intToString(mutexPoolSizePowerOfTwo));
#if defined(SIMULATE_TRUE_HASH_COLLISIONS)
  sizePowerOfTwo = sizePowerOfTwo > 12 ? 12 : sizePowerOfTwo;
#endif
  if(mutexPoolSizePowerOfTwo > sizePowerOfTwo)
    mutexPoolSizePowerOfTwo = sizePowerOfTwo;

  tableSize = ((uint64_t)1) << sizePowerOfTwo;
  tableMask = tableSize-1;
  entries = new Entry[tableSize];
  uint32_t mutexPoolSize = ((uint32_t)1) << mutexPoolSizePowerOfTwo;
  mutexPoolMask = mutexPoolSize-1;
  mutexPool = new MutexPool(mutexPoolSize);
}
NNCacheTableDirect::~NNCacheTableDirect() {
  delete[] entries;
  delete mutexPool;
}

bool NNCacheTableDirect::get(Hash128 nnHash, shared_ptr<NNOutput>& ret) {
  // Free ret BEFORE locking, to avoid any expensive operations while locked.
  if(ret != nullptr)
    ret.reset();

  uint64_t idx = nnHash.hash0 & tableMask;
  uint32_t mutexIdx = (uint32_t)idx & mutexPoolMask;
  Entry& entry = entries[idx];
  std::mutex& mutex = mutexPool->getMutex(mutexIdx);

  std::lock_guard<std::mutex> lock(mutex);

  bool found = false;
#if defined(SIMULATE_TRUE_HASH_COLLISIONS)
  if(entry.ptr != nullptr && ((entry.ptr->nnHash.hash0 ^ nnHash.hash0) & 0xFFF) == 0) {
    ret = entry.ptr;
    found = true;
  }
#else
  if(entry.ptr != nullptr && entry.ptr->nnHash == nnHash) {
    ret = entry.ptr;
    found = true;
  }
#endif
  return found;
}

void NNCacheTableDirect::set(const shared_ptr<NNOutput>& p) {
  // Immediately copy p right now, before locking, to avoid any expensive operations while locked.
  shared_ptr<NNOutput> buf(p);

  uint64_t idx = p->nnHash.hash0 & tableMask;
  uint32_t mutexIdx = (uint32_t)idx & mutexPoolMask;
  Entry& entry = entries[idx];
  std::mutex& mutex = mutexPool->getMutex(mutexIdx);

  {
    std::lock_guard<std::mutex> lock(mutex);
    // Perform a swap, to avoid any expensive free under the mutex.
    entry.ptr.swap(buf);
  }

  // No longer locked, allow buf to fall out of scope now, will free whatever used to be present in the table.
}

void NNCacheTableDirect::clear() {
  shared_ptr<NNOutput> buf;
  for(size_t idx = 0; idx<tableSize; idx++) {
    Entry& entry = entries[idx];
    uint32_t mutexIdx = (uint32_t)idx & mutexPoolMask;
    std::mutex& mutex = mutexPool->getMutex(mutexIdx);
    {
      std::lock_guard<std::mutex> lock(mutex);
      entry.ptr.swap(buf);
    }
    buf.reset();
  }
}

}  // namespace

//-------------------------------------------------------------------------------------
// The seam
//-------------------------------------------------------------------------------------

NNCacheTable::NNCacheTable() {}
NNCacheTable::~NNCacheTable() {}

// Builds the collision-resolution layer the shape asks for, then wraps it in the
// admission layer if one is asked for. Admission is orthogonal to collision
// resolution, so it composes over all four shapes rather than being repeated in each.
//
// Each implementation refuses what it specifically cannot honor at its own
// constructor, where the real constraint is known -- ways against the size of a lock
// region, a byte budget against the smallest entry that could ever fit. This seam
// therefore no longer carries a not-implemented-yet refusal for any shape.
unique_ptr<NNCacheTable> NNCacheTable::create(const NNCacheConfig& config) {
  unique_ptr<NNCacheTable> table;
  switch(config.shape.scheme()) {
  case NNCacheCollisionScheme::Direct:
    table = unique_ptr<NNCacheTable>(
      new NNCacheTableDirect(config.sizePowerOfTwo, config.mutexPoolSizePowerOfTwo)
    );
    break;
  case NNCacheCollisionScheme::LinearProbe:
  case NNCacheCollisionScheme::QuadraticProbe:
    table = makeProbedNNCacheTable(config);
    break;
  case NNCacheCollisionScheme::Chain:
    table = makeChainedNNCacheTable(config);
    break;
  }
  if(table == nullptr)
    throw StringError(
      "Key '" + string(NNCacheConfig::KEY_COLLISION) + "' = " + config.shape.toString() +
      " has no implementation; the valid values are (" + COLLISION_DIRECT + "|" + COLLISION_LINEAR +
      "|" + COLLISION_QUADRATIC + "|" + COLLISION_CHAIN + ")."
    );

  switch(config.admission) {
  case NNCacheAdmissionPolicy::Always:
    break;
  case NNCacheAdmissionPolicy::SecondSighting:
    table = makeSecondSightingNNCacheTable(std::move(table), config.sizePowerOfTwo);
    break;
  }
  return table;
}
