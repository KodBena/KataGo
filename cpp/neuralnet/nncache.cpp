#include "../neuralnet/nncache.h"

#include <cstdlib>

#include "../core/config_parser.h"
#include "../core/fileutils.h"
#include "../core/test.h"
#include "../neuralnet/nncacheimpl.h"
#include "../neuralnet/nncachefrozen.h"
#include "../neuralnet/nncachetrace.h"
#include "../neuralnet/nncachetwolevel.h"
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
const char* const NNCacheConfig::KEY_REPLACEMENT = "nnCacheReplacement";
const char* const NNCacheConfig::KEY_SIGHTING_GHOST_POW = "nnCacheSightingGhostPowerOfTwo";
const char* const NNCacheConfig::KEY_DIR = "nnCacheDir";
const char* const NNCacheConfig::KEY_ADMISSION_SIGNAL_MEASUREMENT = "nnCacheAdmissionSignalMeasurement";

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

// The replacement vocabulary names the SURVIVOR of a direct-mapped collision, never the
// victim. "Replace the more-seen" is ambiguous in English about which key ends up in the
// slot, and the two readings are opposite policies; naming the survivor makes the two
// impossible to confuse and impossible to select by accident.
static const string REPLACEMENT_ALWAYS = "always";
static const string REPLACEMENT_KEEP_LESS_SEEN = "keeplessseen";
static const string REPLACEMENT_KEEP_MORE_SEEN = "keepmoreseen";
static const string REPLACEMENT_KEEP_SIGHTED = "keepsighted";

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
const set<string>& NNCacheConfig::replacementVocabulary() {
  static const set<string> vocab = {
    REPLACEMENT_ALWAYS, REPLACEMENT_KEEP_LESS_SEEN, REPLACEMENT_KEEP_MORE_SEEN, REPLACEMENT_KEEP_SIGHTED
  };
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

// The one decode of an eviction word, shared by the probed and chained arms of the Port
// so the two cannot drift (ADR-0012 P1). Its input is always a word getString() has
// already checked against evictionVocabulary(), so the fallthrough is unreachable text
// rather than a policy default.
static NNCacheEvictionPolicy evictionPolicyFromString(const string& s) {
  if(s == EVICTION_RANDOM) return NNCacheEvictionPolicy::Random;
  if(s == EVICTION_LRU) return NNCacheEvictionPolicy::Lru;
  if(s == EVICTION_LFU) return NNCacheEvictionPolicy::Lfu;
  if(s == EVICTION_NONE) return NNCacheEvictionPolicy::None;
  throw StringError("NNCacheConfig: unrecognized eviction policy '" + s + "'");
}

// The one decode of a replacement word, so the Port and any future caller cannot drift
// (ADR-0012 P1). Its input is always a word getString() has already checked against
// replacementVocabulary(), so the fallthrough is unreachable text, not a default.
static NNCacheReplacementPolicy replacementPolicyFromString(const string& s) {
  if(s == REPLACEMENT_ALWAYS) return NNCacheReplacementPolicy::Always;
  if(s == REPLACEMENT_KEEP_LESS_SEEN) return NNCacheReplacementPolicy::KeepLessSeen;
  if(s == REPLACEMENT_KEEP_MORE_SEEN) return NNCacheReplacementPolicy::KeepMoreSeen;
  if(s == REPLACEMENT_KEEP_SIGHTED) return NNCacheReplacementPolicy::KeepSighted;
  throw StringError("NNCacheConfig: unrecognized replacement policy '" + s + "'");
}

static string replacementPolicyToString(NNCacheReplacementPolicy replacement) {
  switch(replacement) {
  case NNCacheReplacementPolicy::Always: return REPLACEMENT_ALWAYS;
  case NNCacheReplacementPolicy::KeepLessSeen: return REPLACEMENT_KEEP_LESS_SEEN;
  case NNCacheReplacementPolicy::KeepMoreSeen: return REPLACEMENT_KEEP_MORE_SEEN;
  case NNCacheReplacementPolicy::KeepSighted: return REPLACEMENT_KEEP_SIGHTED;
  // A value added to the enum and not to this switch reaches the throw below rather than
  // an unspecified return. The four sibling switches in this file warn identically under
  // -Wswitch-default and are pre-existing on the parent branch; they are left alone as
  // outside this change (ADR-0004) rather than swept in silently.
  default: break;
  }
  throw StringError("NNCacheShape: unhandled replacement policy");
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

NNCacheShape::NNCacheShape(
  NNCacheCollisionScheme scheme, int ways, NNCacheEvictionPolicy eviction, int64_t maxBytes,
  NNCacheReplacementPolicy replacement, std::optional<int> sightingGhostPowerOfTwo
)
  :scheme_(scheme), ways_(ways), eviction_(eviction), maxBytes_(maxBytes), replacement_(replacement),
   sightingGhostPowerOfTwo_(sightingGhostPowerOfTwo)
{}

// True exactly for the rules that allocate a sighting-count ghost. Named once, because
// three different refusals and one accessor all ask this same question and a second copy
// of the list would drift (ADR-0012 P1).
static bool replacementNeedsGhost(NNCacheReplacementPolicy replacement) {
  return replacement == NNCacheReplacementPolicy::KeepLessSeen ||
         replacement == NNCacheReplacementPolicy::KeepMoreSeen;
}

NNCacheShape NNCacheShape::directMapped() {
  return NNCacheShape::directMapped(NNCacheReplacementPolicy::Always);
}

NNCacheShape NNCacheShape::directMapped(NNCacheReplacementPolicy replacement) {
  // EVICTION is not a parameter here, and that is still the whole point: under 1-way
  // direct mapping there is exactly one RESIDENT candidate, so there is no victim to
  // choose among. The stored eviction_ is a placeholder eviction() refuses to hand out.
  //
  // REPLACEMENT is a parameter here, and only here, because the other half of the same
  // collision is a real choice that the eviction axis never covered: the newcomer may
  // lose. `Always` -- the newcomer wins -- is the rule that has always been in force and
  // stays the default.
  return NNCacheShape(
    NNCacheCollisionScheme::Direct, 1, NNCacheEvictionPolicy::None, 0, replacement, std::nullopt
  );
}

NNCacheShape NNCacheShape::directMapped(NNCacheReplacementPolicy replacement, int ghostPowerOfTwo) {
  if(!replacementNeedsGhost(replacement))
    throw StringError(
      "NNCacheShape::directMapped: " + string(NNCacheConfig::KEY_SIGHTING_GHOST_POW) +
      " cannot be honored under " + string(NNCacheConfig::KEY_REPLACEMENT) + " = " +
      replacementPolicyToString(replacement) + ", which allocates no sighting-count ghost at all. "
      "Only (" + REPLACEMENT_KEEP_LESS_SEEN + "|" + REPLACEMENT_KEEP_MORE_SEEN + ") keep counts; " +
      REPLACEMENT_KEEP_SIGHTED + " keeps one bit per slot in the table and " + REPLACEMENT_ALWAYS +
      " keeps nothing. Sizing a structure that does not exist is not a setting to correct, it is a "
      "shape with no meaning."
    );
  if(ghostPowerOfTwo < 0 || ghostPowerOfTwo > 40)
    throw StringError(
      "NNCacheShape::directMapped: " + string(NNCacheConfig::KEY_SIGHTING_GHOST_POW) +
      " must be in 0..40, got " + Global::intToString(ghostPowerOfTwo) +
      ". It is a POWER OF TWO of ghost slots at 4 bytes each, so 40 is already a terabyte."
    );
  return NNCacheShape(
    NNCacheCollisionScheme::Direct, 1, NNCacheEvictionPolicy::None, 0, replacement, ghostPowerOfTwo
  );
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
  // No replacement argument: a probed table's newcomer always wins its contest, taking
  // the eviction policy's victim's slot, so there is no second candidate to name.
  return NNCacheShape(scheme, ways, eviction, 0, NNCacheReplacementPolicy::Always, std::nullopt);
}

NNCacheShape NNCacheShape::chained(int64_t maxBytes, NNCacheEvictionPolicy eviction) {
  if(maxBytes <= 0)
    throw StringError(
      "NNCacheShape::chained: " + string(NNCacheConfig::KEY_COLLISION) + " = " + COLLISION_CHAIN +
      " requires a positive " + string(NNCacheConfig::KEY_MAX_BYTES) +
      ", since a chained table has no fixed capacity and is bounded only by its byte budget."
    );
  if(eviction == NNCacheEvictionPolicy::None)
    throw StringError(
      "NNCacheShape::chained: " + string(NNCacheConfig::KEY_EVICTION) + " = " + EVICTION_NONE +
      " cannot be honored under " + string(NNCacheConfig::KEY_COLLISION) + " = " + COLLISION_CHAIN +
      ". Chaining never evicts on COLLISION, which is what this value used to mean here, but its "
      "byte budget is always enforced, so a resident entry is always given up when the budget is "
      "exceeded and something must choose which. Valid here are (" + EVICTION_RANDOM + "|" +
      EVICTION_LRU + "|" + EVICTION_LFU + "); " + EVICTION_LRU + " is the order that was in force "
      "before this axis was expressible."
    );
  // No replacement argument, for the same reason probed() has none.
  return NNCacheShape(
    NNCacheCollisionScheme::Chain, 0, eviction, maxBytes, NNCacheReplacementPolicy::Always, std::nullopt
  );
}

NNCacheEvictionPolicy NNCacheShape::eviction() const {
  if(scheme_ == NNCacheCollisionScheme::Direct)
    throw StringError(
      "NNCacheShape::eviction: a 1-way direct-mapped table has no eviction policy -- a collision "
      "presents exactly one candidate victim, so there is no choice to report."
    );
  return eviction_;
}

NNCacheReplacementPolicy NNCacheShape::replacement() const {
  if(scheme_ != NNCacheCollisionScheme::Direct)
    throw StringError(
      "NNCacheShape::replacement: only a 1-way direct-mapped table has a replacement rule -- under " +
      collisionSchemeToString(scheme_) + " the newcomer always wins its contest and takes the "
      "eviction policy's victim's slot, so there is no second candidate to keep or give up. "
      "Ask this shape for eviction() instead."
    );
  return replacement_;
}

std::optional<int> NNCacheShape::sightingGhostPowerOfTwo() const {
  if(scheme_ != NNCacheCollisionScheme::Direct || !replacementNeedsGhost(replacement_))
    throw StringError(
      "NNCacheShape::sightingGhostPowerOfTwo: this shape keeps no sighting-count ghost, so it has "
      "no ghost size to report. Only " + string(NNCacheConfig::KEY_COLLISION) + " = " +
      COLLISION_DIRECT + " with " + string(NNCacheConfig::KEY_REPLACEMENT) + " in (" +
      REPLACEMENT_KEEP_LESS_SEEN + "|" + REPLACEMENT_KEEP_MORE_SEEN + ") does; this one is " +
      toString() + "."
    );
  return sightingGhostPowerOfTwo_;
}

string NNCacheShape::toString() const {
  if(scheme_ == NNCacheCollisionScheme::Direct) {
    if(replacement_ == NNCacheReplacementPolicy::Always)
      return COLLISION_DIRECT;
    string out = COLLISION_DIRECT + ", " + replacementPolicyToString(replacement_);
    if(sightingGhostPowerOfTwo_.has_value())
      out += ", ghost 2^" + Global::intToString(*sightingGhostPowerOfTwo_);
    return out;
  }
  if(scheme_ == NNCacheCollisionScheme::Chain)
    return COLLISION_CHAIN + ", maxBytes " + Global::int64ToString(maxBytes_) + ", " +
      evictionPolicyToString(eviction_);
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

  // The persisted-cache directory, and with it the decision to build a level-0 resolution
  // list at all. Both refusals below are construction-time -- the process does not start --
  // because both describe a configuration the engine cannot honor, and an engine that
  // started anyway would answer every cache action with a surprise instead of the config
  // being wrong where it is wrong (ADR-0002 rung 1).
  if(cfg.contains(KEY_DIR)) {
    if(sizePowerOfTwo < 0)
      throw StringError(
        "Key '" + string(KEY_DIR) + "' cannot be honored with the neural net cache disabled "
        "(nnCacheSizePowerOfTwo is negative). It names where a cache's persisted contexts are "
        "read and written, and there is no cache here to attach one to. Either remove '" +
        string(KEY_DIR) + "' or enable the cache."
      );
    const string dir = cfg.getString(KEY_DIR);
    if(!FileUtils::isDirectory(dir))
      throw StringError(
        "Key '" + string(KEY_DIR) + "' = " + dir + " is not an existing directory. KataGo owns "
        "every byte under it and will not create it: a mistyped path that the engine silently "
        "created would put a session's persisted cache somewhere nobody looks."
      );
    config.cacheDirectory = dir;
  }

  // GATED OFF BY DEFAULT (ledger rows 1652/1655/1660): measures three candidate admission
  // currencies side by side without changing any admission decision. See
  // NNCachePresentationLedger's own comment for what it counts and why picking a currency
  // is deliberately left open.
  if(cfg.contains(KEY_ADMISSION_SIGNAL_MEASUREMENT))
    config.admissionSignalMeasurement = cfg.getBool(KEY_ADMISSION_SIGNAL_MEASUREMENT);

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
    // The one key this scheme, and only this scheme, accepts beyond the default. Absent
    // means `always`, which is the rule that has always been in force, so a config that
    // says nothing builds exactly the table it built before this key existed.
    const string replacementStr =
      cfg.contains(KEY_REPLACEMENT) ? cfg.getString(KEY_REPLACEMENT, replacementVocabulary()) : REPLACEMENT_ALWAYS;
    const NNCacheReplacementPolicy replacement = replacementPolicyFromString(replacementStr);
    // The ghost's size, when the operator states one. Left unstated it is derived from
    // nnCacheSizePowerOfTwo, which is what it did before this key existed -- so upgrading
    // moves nobody's memory. That derived default is stated in gtp_example.cfg as being
    // WRONG for the regime the counting rules exist for: the ghost's natural size is the
    // WORKING SET, an absolute quantity, and a table deliberately smaller than the working
    // set gives a ghost that cannot hold the counts either.
    if(cfg.contains(KEY_SIGHTING_GHOST_POW)) {
      if(!replacementNeedsGhost(replacement))
        throw StringError(
          "Key '" + string(KEY_SIGHTING_GHOST_POW) + "' cannot be honored under " +
          string(KEY_REPLACEMENT) + " = " + replacementStr + ", which keeps no sighting-count ghost. "
          "Only (" + REPLACEMENT_KEEP_LESS_SEEN + "|" + REPLACEMENT_KEEP_MORE_SEEN + ") keep counts; " +
          REPLACEMENT_KEEP_SIGHTED + " keeps one bit per slot inside the table and " + REPLACEMENT_ALWAYS +
          " keeps nothing at all. Either remove '" + string(KEY_SIGHTING_GHOST_POW) + "', or set " +
          string(KEY_REPLACEMENT) + " to one of (" + REPLACEMENT_KEEP_LESS_SEEN + "|" +
          REPLACEMENT_KEEP_MORE_SEEN + ")."
        );
      config.shape = NNCacheShape::directMapped(replacement, cfg.getInt(KEY_SIGHTING_GHOST_POW, 0, 40));
    }
    else {
      config.shape = NNCacheShape::directMapped(replacement);
    }
    return config;
  }

  // Same refusal one scheme out: a ghost size under a scheme that has no replacement rule
  // to keep counts for.
  if(cfg.contains(KEY_SIGHTING_GHOST_POW))
    throw StringError(
      "Key '" + string(KEY_SIGHTING_GHOST_POW) + "' is meaningless under " + string(KEY_COLLISION) +
      " = " + collision + ". It sizes the sighting-count ghost that the " + string(KEY_REPLACEMENT) +
      " rules keep, and " + string(KEY_REPLACEMENT) + " exists only under " + string(KEY_COLLISION) +
      " = " + COLLISION_DIRECT + "."
    );

  // Every non-direct scheme refuses the replacement key, because past this point it has
  // no representation: probed() and chained() take no such argument.
  if(cfg.contains(KEY_REPLACEMENT))
    throw StringError(
      "Key '" + string(KEY_REPLACEMENT) + "' is meaningless under " + string(KEY_COLLISION) + " = " +
      collision + ". It names which of the TWO candidates a 1-WAY DIRECT-MAPPED collision keeps, the "
      "resident incumbent or the offered newcomer. Under " + collision + " the newcomer never loses: it "
      "takes the slot of whichever resident '" + string(KEY_EVICTION) + "' names. Either remove '" +
      string(KEY_REPLACEMENT) + "', or set " + string(KEY_COLLISION) + " = " + COLLISION_DIRECT + ". "
      "(A replacement rule for the probed and chained shapes -- one where the newcomer may also be "
      "refused -- is a real and deliberately UNIMPLEMENTED extension, not an oversight; it is named "
      "here rather than left silent.)"
    );

  if(collision == COLLISION_CHAIN) {
    if(cfg.contains(KEY_WAYS))
      throw StringError(
        "Key '" + string(KEY_WAYS) + "' is meaningless under " + string(KEY_COLLISION) + " = " +
        COLLISION_CHAIN + ": a chain has no fixed associativity."
      );
    if(!cfg.contains(KEY_MAX_BYTES))
      throw StringError(
        "Key '" + string(KEY_MAX_BYTES) + "' is required under " + string(KEY_COLLISION) + " = " +
        COLLISION_CHAIN + ": a chained table has no fixed capacity and is bounded only by its byte budget."
      );
    // Chaining never evicts on COLLISION, but its byte budget is always enforced, so a
    // resident entry IS given up when a region goes over and something chooses which.
    // That order used to be unnameable -- the key was required to say `none`, which
    // denied a policy was in force while one was. It is now this key's meaning here,
    // defaulting to the recency order that was in force before it was expressible.
    const string evictionStr =
      cfg.contains(KEY_EVICTION) ? cfg.getString(KEY_EVICTION, evictionVocabulary()) : EVICTION_LRU;
    if(evictionStr == EVICTION_NONE)
      throw StringError(
        "Key '" + string(KEY_EVICTION) + "' = " + EVICTION_NONE + " cannot be honored under " +
        string(KEY_COLLISION) + " = " + COLLISION_CHAIN + ". It used to be required here, and it "
        "meant 'no COLLISION eviction' -- but a chained table's '" + string(KEY_MAX_BYTES) +
        "' budget is always enforced, so a resident entry is always given up when a region exceeds "
        "it, and this key now names WHICH one. Valid here are (" + EVICTION_RANDOM + "|" +
        EVICTION_LRU + "|" + EVICTION_LFU + "). " + EVICTION_LRU + " is the order that was already "
        "in force, so that is the value that changes nothing."
      );
    config.shape = NNCacheShape::chained(
      cfg.getInt64(KEY_MAX_BYTES, 1, (int64_t)1 << 62),
      evictionPolicyFromString(evictionStr)
    );
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
  const NNCacheEvictionPolicy eviction =
    evictionPolicyFromString(cfg.getString(KEY_EVICTION, evictionVocabulary()));
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
  bool contains(Hash128 nnHash) const override;
  NNCacheStats stats() const override;
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

// This shape's get happens to touch nothing either -- a 1-way direct-mapped table under
// `always` has no recency, no frequency and no sighting count to move -- so the two bodies look
// alike. They are still two methods: the equality is a property of THIS shape, and writing
// contains() as a call to get() here would make the no-trace guarantee depend on that staying
// true through every future edit of the shape beside it (ADR-0012 P1: the guarantee has one
// home, which is this method).
bool NNCacheTableDirect::contains(Hash128 nnHash) const {
  const uint64_t idx = nnHash.hash0 & tableMask;
  const uint32_t mutexIdx = (uint32_t)idx & mutexPoolMask;
  const Entry& entry = entries[idx];
  std::lock_guard<std::mutex> lock(mutexPool->getMutex(mutexIdx));
  // The collision-simulation build's own notion of membership, mirrored rather than ignored: a
  // key this build's get() would answer is a key this build's level 1 owns, and the two must
  // not disagree about that under any flag.
#if defined(SIMULATE_TRUE_HASH_COLLISIONS)
  return entry.ptr != nullptr && ((entry.ptr->nnHash.hash0 ^ nnHash.hash0) & 0xFFF) == 0;
#else
  return entry.ptr != nullptr && entry.ptr->nnHash == nnHash;
#endif
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

// A snapshot, taken one lock region at a time rather than under a single global lock:
// the table has no global lock and inventing one for a reporting call would be a new
// serialisation point on a path that never had one. The consequence is stated rather
// than hidden -- concurrent writers can move entries between regions the walk has
// already passed and regions it has not, so under live traffic this is an approximation
// whose error is bounded by the number of writes that land during the walk. The harness
// that uses it takes it between measured windows, with no other thread running, where
// it is exact.
NNCacheStats NNCacheTableDirect::stats() const {
  NNCacheStats s = {0,0,0,(int64_t)tableSize};
  for(uint64_t idx = 0; idx<tableSize; idx++) {
    std::mutex& mutex = mutexPool->getMutex((uint32_t)idx & mutexPoolMask);
    std::lock_guard<std::mutex> lock(mutex);
    const Entry& entry = entries[idx];
    if(entry.ptr != nullptr) {
      s.residentEntries += 1;
      s.residentPayloadBytes += (int64_t)nnOutputFootprintBytes(*entry.ptr);
    }
  }
  s.fixedStructureBytes =
    (int64_t)(tableSize * sizeof(Entry)) +
    (int64_t)((uint64_t)mutexPoolMask + 1) * (int64_t)sizeof(std::mutex);
  return s;
}

}  // namespace

//-------------------------------------------------------------------------------------
// The seam
//-------------------------------------------------------------------------------------

NNCacheTable::NNCacheTable() : attribution_(nullptr), contexts_() {}
NNCacheTable::~NNCacheTable() {}

//-------------------------------------------------------------------------------------
// Context attribution
//-------------------------------------------------------------------------------------

NNCacheContextId NNCacheTable::attachCacheContext(const string& name) {
  // attach() validates and refuses first, so a refused name never causes an allocation.
  const NNCacheContextId id = contexts_.attach(name);
  if(attribution_ == nullptr) {
    const int pow = NNCacheAttributionRecorder::defaultPowerOfTwo();
    // Enough to keep concurrent earnings off one lock, never more mutexes than rows. The
    // same rule the two-level table's hit ledger uses, read from the same shape rather than
    // chosen twice.
    const int mutexPow = pow < 10 ? pow : 10;
    attribution_.reset(new NNCacheAttributionRecorder(pow, mutexPow));
  }
  return id;
}

const NNCacheContextSet& NNCacheTable::cacheContexts() const {
  return contexts_;
}

bool NNCacheTable::peek(Hash128 nnHash, shared_ptr<NNOutput>& ret) {
  // See the header: correct for every shape that counts nothing, which is every shape but
  // the two-level one, and that one overrides.
  return get(nnHash, ret);
}

void NNCacheTable::set(const shared_ptr<NNOutput>& p, const NNCacheAttribution& attribution) {
  set(p, attribution, NNCacheEntryProvenance::LiveEvaluation);
}

void NNCacheTable::set(
  const shared_ptr<NNOutput>& p, const NNCacheAttribution& attribution, NNCacheEntryProvenance provenance
) {
  if(attribution.isToContext() && !contexts_.owns(attribution.contextId()))
    throw StringError(
      "NNCacheTable::set: this attribution names a context attached to a different cache. "
      "Spending it here would file the entry under whichever context sits at the same "
      "position in this table's own name space, which is a different card."
    );
  // No context attached means nothing to attribute to and nothing an attribution could have
  // been: the only representable attribution is NoAttributableContext, and counting a
  // universe of one outcome would be a number with no reading. This is also what keeps the
  // default configuration's set path exactly what it was.
  if(attribution_ != nullptr)
    attribution_->record(p->nnHash, attribution, provenance);
  set(p);
}

vector<Hash128> NNCacheTable::unpersistedKeysFor(const NNCacheContextId& context) const {
  // The same ownership refusal attributedKeysFor makes, from the same one home, so asking
  // against the wrong cache's context is refused rather than answered with an empty list.
  if(!contexts_.owns(context))
    throw StringError(
      "NNCacheTable: this context is not attached to this cache, so it owes nothing here. "
      "Asking against the wrong cache's context is refused rather than answered with an "
      "empty list a caller would read as 'this context has nothing left to dump'."
    );
  if(attribution_ == nullptr)
    return vector<Hash128>();
  return attribution_->unpersistedKeysFor(context);
}

int64_t NNCacheTable::markPersisted(const NNCacheContextId& context, const vector<Hash128>& keys) {
  if(!contexts_.owns(context))
    throw StringError(
      "NNCacheTable: this context is not attached to this cache, so nothing here can be "
      "marked persisted under it."
    );
  if(attribution_ == nullptr) {
    // No context has been attached, so there is no ledger and no key could have been earned
    // under one -- but contexts_.owns() just said this context IS attached, so the two
    // disagree and that is a defect in this class, not a caller's doing (ADR-0002).
    throw StringError(
      "NNCacheTable::markPersisted: a context is attached but no attribution ledger exists."
    );
  }
  return attribution_->markPersisted(context, keys);
}

NNCacheAttributionLedger NNCacheTable::harvestAttribution() const {
  if(attribution_ == nullptr)
    return NNCacheAttributionLedger::notAttributed();
  return NNCacheAttributionLedger::attributed(
    attribution_->harvest(contexts_),
    attribution_->noAttributableContextEntries(),
    attribution_->unrecordedAttributions()
  );
}

vector<Hash128> NNCacheTable::attributedKeysFor(const NNCacheContextId& context) const {
  if(!contexts_.owns(context))
    throw StringError(
      "NNCacheTable: this context is not attached to this cache, so it has earned nothing "
      "here. Asking against the wrong cache's context is refused rather than answered with "
      "an empty list a caller would read as 'this context earned nothing'."
    );
  if(attribution_ == nullptr)
    return vector<Hash128>();
  return attribution_->keysFor(context);
}

// Every table that keeps no per-key hit counts answers the per-context question the same way
// it answers the whole-table one, and for the same reason: there are no counts to partition.
// The context is still checked, so asking against the wrong cache is refused rather than
// answered with a disposition that reads as an ordinary "this table does not count".
NNCacheHitLedger NNCacheTable::harvestHitCountsFor(const NNCacheContextId& context) const {
  const vector<Hash128> keys = attributedKeysFor(context);
  (void)keys;
  return NNCacheHitLedger::notCounted();
}

// The delta twin's default, through the same ownership check and for the same reason: a table
// with no counters has no unpersisted counts to divide, and "counted, zero rows" would be a
// claim it cannot make.
NNCacheHitLedger NNCacheTable::takeUnpersistedHitCountsFor(const NNCacheContextId& context) {
  const vector<Hash128> keys = attributedKeysFor(context);
  (void)keys;
  return NNCacheHitLedger::notCounted();
}

// And the non-consuming question's default. FALSE IS THE HONEST ANSWER HERE rather than a hedge:
// a table that keeps no per-key hit counts has none that are unpersisted, so a detach refusal
// reading this is told the truth about this table and not "probably nothing".
bool NNCacheTable::hasUnpersistedHitCountsFor(const NNCacheContextId& context) const {
  const vector<Hash128> keys = attributedKeysFor(context);
  (void)keys;
  return false;
}

//-------------------------------------------------------------------------------------
// The unified hit-count surface
//-------------------------------------------------------------------------------------

NNCacheHitLedger::NNCacheHitLedger(
  NNCacheHitLedgerDisposition disposition, vector<NNCacheHitCount> entries, int64_t unrecordedHits
)
  :disposition_(disposition), entries_(std::move(entries)), unrecordedHits_(unrecordedHits)
{}

NNCacheHitLedger NNCacheHitLedger::notCounted() {
  return NNCacheHitLedger(NNCacheHitLedgerDisposition::NotCounted, vector<NNCacheHitCount>(), 0);
}

NNCacheHitLedger NNCacheHitLedger::counted(vector<NNCacheHitCount> entries, int64_t unrecordedHits) {
  return NNCacheHitLedger(NNCacheHitLedgerDisposition::Counted, std::move(entries), unrecordedHits);
}

// Refuses rather than handing back an empty vector, which a caller could read as "this
// session hit nothing" when the truth is "this table never counted" (ADR-0012 P11).
const vector<NNCacheHitCount>& NNCacheHitLedger::entries() const {
  if(disposition_ != NNCacheHitLedgerDisposition::Counted)
    throw StringError(
      "NNCacheHitLedger: this table keeps no per-key hit counts, so it has no rows to hand "
      "out. Check disposition() before asking; an empty row list would be indistinguishable "
      "from a session in which nothing was hit."
    );
  return entries_;
}

int64_t NNCacheHitLedger::unrecordedHits() const {
  if(disposition_ != NNCacheHitLedgerDisposition::Counted)
    throw StringError(
      "NNCacheHitLedger: this table keeps no per-key hit counts, so it has no unrecorded "
      "count to report either."
    );
  return unrecordedHits_;
}

// Every single-level table's answer, inherited rather than implemented four times: hit
// counting is a property of the TWO-LEVEL strategy, which exists exactly when a frozen
// level 0 does. So the default configuration -- no level 0, one ordinary table -- gains no
// field, no branch and no allocation from this surface existing.
NNCacheHitLedger NNCacheTable::harvestHitCounts() const {
  return NNCacheHitLedger::notCounted();
}

// And the same for the delta surface, for the same reason: a table with no counters has no
// unpersisted counts either, and saying "counted, zero rows" would be a claim it cannot make.
NNCacheHitLedger NNCacheTable::takeUnpersistedHitCounts() {
  return NNCacheHitLedger::notCounted();
}

//-------------------------------------------------------------------------------------
// The admission-signal-candidate measurement's base defaults
//-------------------------------------------------------------------------------------
//
// Every table's inherited answer unless NNCacheTableTwoLevel overrides it: gated off is
// the byte-identical-default this whole axis promises, and a shape that has never heard of
// this measurement is exactly as "gated off" as one that has but was configured false.

NNCachePresentationLedger NNCacheTable::harvestPresentationCounts() const {
  return NNCachePresentationLedger::notCounted();
}

NNCachePresentationLedger NNCacheTable::harvestPresentationCountsFor(const NNCacheContextId& context) const {
  const vector<Hash128> keys = attributedKeysFor(context);
  (void)keys;
  return NNCachePresentationLedger::notCounted();
}

void NNCacheTable::beginAdmissionSignalMeasurementWindow() {
  // Deliberately empty. See the declaration: this must be a single predictable no-op for
  // every table not measuring, which is every table shape except the two-level one, and
  // that one only when NNCacheConfig::admissionSignalMeasurement was set.
}

NNCachePresentationLedger::NNCachePresentationLedger(
  NNCachePresentationLedgerDisposition disposition,
  vector<NNCachePresentationCount> entries,
  int64_t unrecordedRaw,
  int64_t unrecordedDeduped
)
  :disposition_(disposition),
   entries_(std::move(entries)),
   unrecordedRaw_(unrecordedRaw),
   unrecordedDeduped_(unrecordedDeduped)
{}

NNCachePresentationLedger NNCachePresentationLedger::notCounted() {
  return NNCachePresentationLedger(NNCachePresentationLedgerDisposition::NotCounted, vector<NNCachePresentationCount>(), 0, 0);
}

NNCachePresentationLedger NNCachePresentationLedger::counted(
  vector<NNCachePresentationCount> entries, int64_t unrecordedRaw, int64_t unrecordedDeduped
) {
  return NNCachePresentationLedger(
    NNCachePresentationLedgerDisposition::Counted, std::move(entries), unrecordedRaw, unrecordedDeduped
  );
}

const vector<NNCachePresentationCount>& NNCachePresentationLedger::entries() const {
  if(disposition_ != NNCachePresentationLedgerDisposition::Counted)
    throw StringError(
      "NNCachePresentationLedger: this table is not measuring the admission-signal candidates "
      "(NNCacheConfig::admissionSignalMeasurement is false, or this shape never carries the "
      "axis), so it has no rows to hand out. Check disposition() before asking."
    );
  return entries_;
}

int64_t NNCachePresentationLedger::unrecordedRaw() const {
  if(disposition_ != NNCachePresentationLedgerDisposition::Counted)
    throw StringError("NNCachePresentationLedger: not measuring, so no unrecorded-raw count either.");
  return unrecordedRaw_;
}

int64_t NNCachePresentationLedger::unrecordedDeduped() const {
  if(disposition_ != NNCachePresentationLedgerDisposition::Counted)
    throw StringError("NNCachePresentationLedger: not measuring, so no unrecorded-deduped count either.");
  return unrecordedDeduped_;
}

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
    // The shipped 1-way table is reached only under `always`, and it is not the class
    // that carries the replacement rule: the sighting-count table lives in its own
    // translation unit and is constructed only when a rule was asked for. So the DEFAULT
    // configuration runs exactly the code it ran before this axis existed -- byte for
    // byte, no branch added to its get or set -- rather than running a generalised table
    // that happens to be configured to behave the same (ADR-0009: the instrument stays
    // outside the instrument).
    if(config.shape.replacement() == NNCacheReplacementPolicy::Always)
      table = unique_ptr<NNCacheTable>(
        new NNCacheTableDirect(config.sizePowerOfTwo, config.mutexPoolSizePowerOfTwo)
      );
    else
      table = makeSightingDirectNNCacheTable(config);
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

  // Trace recording, outermost so that what it records is what the ENGINE asked of the
  // cache -- every get and every set as offered, before admission decides whether a set
  // is stored. That is the stream a replay of another policy needs; recording inside the
  // admission filter would bake this run's admission policy into the trace and make the
  // admission axis unsweepable.
  //
  // Reached only through the environment, and only once, at construction: the shipped
  // get/set path contains no test for this at all, because when it is not asked for the
  // decorator is not there (ADR-0009 -- the instrument stays outside the instrument).
  const char* tracePath = getenv(NNCacheTrace::TRACE_ENV);
  if(tracePath != NULL && tracePath[0] != '\0') {
    cerr << "NNCache: " << NNCacheTrace::TRACE_ENV << " is set, recording every cache operation to "
         << tracePath << endl;
    cerr << "NNCache: this serialises every cache operation through one lock. It is a CAPTURE run. "
         << "Do NOT read timings from it." << endl;
    table = NNCacheTrace::wrapWithTrace(std::move(table), string(tracePath));
  }
  return table;
}

unique_ptr<NNCacheTwoLevelTable> NNCacheTable::createWithLevelZeroList(const NNCacheConfig& config) {
  if(!config.cacheDirectory.has_value())
    throw StringError(
      "NNCacheTable::createWithLevelZeroList: no cache directory is configured, so a level-0 "
      "resolution list would have nothing it could ever load from. Set '" +
      string(NNCacheConfig::KEY_DIR) + "', or build the table with create()."
    );

  // BUILT WITH ONE SOURCE AND THEN LEFT WITH NONE, which is the two-level table's own
  // documented shape rather than a workaround: makeTwoLevelNNCacheTable refuses a null level
  // 0 on purpose -- "absent is represented by not having one of these at all" -- while
  // detaching the last source is explicitly legal and leaves a table that serves everything
  // from level 1 until the next attach. An engine starts in exactly that state: it has a
  // directory and an attach surface, and no session has attached anything yet. So it is
  // handed an EMPTY frozen structure, which is a legal structure that answers absent to
  // every key, and that structure is taken straight back off the list.
  //
  // The level-1 hit ledger is sized at NNCacheAttributionRecorder::defaultPowerOfTwo(),
  // deliberately the same constant and for the same reason: both are sized against the
  // number of DISTINCT KEYS ONE SESSION EARNS, the largest real card in the operator's
  // corpus at the load factor that structure's bounded probe window needs. One quantity, one
  // home for its size (ADR-0012 P1).
  unique_ptr<NNCacheTwoLevelTable> table = makeTwoLevelNNCacheTable(
    NNCacheFrozen::build(vector<unique_ptr<NNOutput>>()),
    NNCacheTable::create(config),
    NNCacheAttributionRecorder::defaultPowerOfTwo(),
    config.admissionSignalMeasurement
  );
  const vector<NNCacheLevelZeroSourceId> order = table->levelZeroResolutionOrder();
  testAssert(order.size() == 1);
  const unique_ptr<NNCacheFrozen> placeholder = table->detachLevelZero(NNCacheLevelZeroSwapPermit(), order[0]);
  testAssert(placeholder != nullptr);
  testAssert(table->numLevelZeroSources() == 0);
  return table;
}
