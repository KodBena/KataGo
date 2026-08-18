#ifndef NEURALNET_NNCACHE_H_
#define NEURALNET_NNCACHE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>

#include "../core/global.h"
#include "../core/hash.h"
#include "../neuralnet/nninputs.h"

class ConfigParser;

// How the table resolves two keys landing on the same bucket.
enum class NNCacheCollisionScheme {
  Direct,          // 1-way direct-mapped: exactly one slot per key. Today's behaviour.
  LinearProbe,     // open addressing, candidate slots at offsets 0,1,2,... from home
  QuadraticProbe,  // open addressing, candidate slots at triangular offsets 0,1,3,6,...
  Chain,           // separate chaining; no collision eviction, bounded by a byte budget
};

// Which victim an eviction gives up.
//
// Under a probed scheme this chooses among the `ways` candidates a full bucket offers.
// Under Chain it chooses among a lock region's resident entries when the byte budget is
// exceeded -- a different trigger, the same question. There is no member for the
// direct-mapped case on purpose: see NNCacheShape.
enum class NNCacheEvictionPolicy {
  Random,
  Lru,
  Lfu,
  // No longer reachable through any factory or through the .cfg Port. It survives as
  // the placeholder directMapped() stores in a field eviction() refuses to hand out,
  // and as a vocabulary word the Port names in its refusals. Before chained eviction
  // became selectable this was what `chain` was required to say, which meant a recency
  // policy was in force under a name that denied any policy was -- the defect
  // NNCacheShape::chained(maxBytes,eviction) exists to close.
  None,
};

// Whether an entry offered to the table is actually stored.
enum class NNCacheAdmissionPolicy {
  Always,          // today's behaviour
  SecondSighting,  // store a key only from its second sighting onward
};

// Which of the TWO candidates a 1-way direct-mapped collision keeps.
//
// This is a third axis, and it is neither of the two above. Eviction answers "which of
// the `ways` residents is given up", and is meaningless at one way -- that is the whole
// reason NNCacheShape::directMapped() takes no eviction argument. But a direct-mapped
// collision still presents exactly TWO candidates, the resident INCUMBENT and the
// offered NEWCOMER, and something has to choose between them. Until this enum existed
// the choice was made and unnameable: the table unconditionally took the newcomer, which
// is `Always` below. That is the same defect the chained table's capacity order had
// before NNCacheShape::chained gained an eviction argument -- a policy in force under no
// name -- and it is fixed the same way.
//
// It is also not ADMISSION. NNCacheAdmissionPolicy is a decorator that never sees the
// resident entry: it decides whether an offer is stored at all, from the offered key
// alone. A rule that compares the newcomer against the incumbent cannot be decided
// there.
//
// SIGHTINGS. A key's sighting count is the number of times it has been PRESENTED to the
// table -- every get and every set, hit or miss -- saturating at 255. Counting stores
// alone would be useless in KataGo, which stores a position once (the perfmatrix sweep
// measured 442,735 sets over 442,192 distinct keys), so every comparison would be a tie.
// Counts are kept for NON-RESIDENT keys too, in a ghost table; without that a newcomer's
// count is structurally zero and any comparison collapses back to `Always`. The ghost's
// size is nnCacheSightingGhostPowerOfTwo, and it is NOT the table's size: see
// NNCacheShape::directMapped(replacement, ghostPowerOfTwo).
//
// WHY THESE ARE NAMED FOR THE SURVIVOR. "Replace the one that has been seen more often"
// is ambiguous in English about which key ends up in the slot -- the sentence names the
// VICTIM, and the two readings are opposite policies. This vocabulary names the key that
// KEEPS the slot, so a config file cannot be the place that ambiguity is discovered. It
// has already cost one round of measurement labelled against the wrong arm.
enum class NNCacheReplacementPolicy {
  // The newcomer always wins. Today's behaviour, the default, and the only value under
  // which the shipped 1-way table is used unchanged.
  Always,
  // The candidate seen FEWER times survives; the more-seen one is replaced. The INVERSE
  // arm, carried so the mechanism can be measured in both directions -- it is one
  // comparison flip away from KeepMoreSeen and measuring both is strictly more
  // informative than measuring either. Not a recommendation, and not anybody's proposal.
  KeepLessSeen,
  // The candidate seen MORE times survives; the less-seen one is replaced. THIS is the
  // "most-seen direct" the operator proposed, and it is the conventional, LFU-shaped
  // direction.
  KeepMoreSeen,
  // Keep the incumbent if it has been sighted since it was stored, and give it exactly
  // ONE reprieve -- the second-chance rule. Needs no ghost table at all: one bool per
  // slot, in the cache line the lookup already touched, so it costs no extra memory
  // access per operation. It is NOT cheaper in bytes: alignment inflates that bool to
  // 8 bytes per slot against the two counting rules' 4-byte ghost entry, which the test
  // suite measures and prints rather than asserting from a comment. The alternative
  // trade, not the cheap one. Its inverse ("keep the incumbent only if it has NOT been
  // sighted") is deliberately not offered: a never-sighted incumbent would then own its
  // slot forever, which is not a policy but a leak.
  KeepSighted,
};

// The coherent composition of the collision-resolution and eviction axes.
//
// Why this is a class with a private constructor rather than two independent enum
// fields: an eviction policy is MEANINGLESS under 1-way direct mapping, because a
// collision presents exactly one candidate victim and there is no choice to make.
// "direct + lru" is therefore not a configuration that gets constructed and then
// rejected by a validator -- it is a configuration that has no representation here
// at all, because directMapped() takes no eviction argument to carry one in
// (ADR-0000 Rule 2a: the type forecloses the class, rather than a guard catching
// each instance of it). The .cfg boundary in NNCacheConfig::fromCfg is then a
// translate-and-validate Port (ADR-0012 P2) that refuses the text it cannot decode
// into this type, naming the valid vocabulary.
class NNCacheShape {
 public:
  // 1-way direct-mapped, overwrite-on-collision. Today's behaviour, and the default.
  static NNCacheShape directMapped();
  // 1-way direct-mapped, with `replacement` deciding which of the two candidates a
  // collision keeps. directMapped(Always) is exactly directMapped().
  //
  // `replacement` is an argument of THIS factory and of no other, which is what keeps
  // the incoherent shapes unrepresentable in the direction this axis opens: probed() and
  // chained() take no replacement argument, so "linearprobe + keepmoreseen" has no
  // representation here at all, exactly as "direct + lru" has none. Under a probed or
  // chained shape the newcomer never loses the contest -- it displaces the eviction
  // policy's victim -- so there is no second candidate for a replacement rule to name.
  static NNCacheShape directMapped(NNCacheReplacementPolicy replacement);
  // As above, with the sighting-count ghost sized EXPLICITLY as 2^ghostPowerOfTwo slots
  // rather than derived from nnCacheSizePowerOfTwo.
  //
  // Why this is a knob at all. The ghost is a lossy sketch: two keys landing on one ghost
  // slot overwrite each other's counts. Sized from the table, its load factor IS the
  // table's load factor -- so in exactly the regime a replacement rule exists for, where
  // the table cannot hold the working set, the ghost cannot hold the counts either.
  //
  // And a saturated ghost does not degrade these rules, it REPLACES them with degenerate
  // ones. A newcomer's own sighting rewrites its ghost slot immediately before the
  // comparison, so the newcomer reads 1 whatever was there before; the incumbent's count
  // is only read, so an overwritten incumbent reads 0. The comparison becomes 1 against 0
  // almost every time, which makes KeepMoreSeen admit unconditionally -- it stops
  // choosing and becomes Always -- and KeepLessSeen refuse unconditionally. A sweep
  // through a table-sized ghost therefore measures neither rule. (An earlier version of
  // this comment claimed the opposite asymmetry, that an overwritten incumbent's 0 makes
  // it WIN under KeepLessSeen; that was wrong in sign and the re-measurement refuted it.
  // The conclusion is unchanged.) See nncachesighting.cpp for the walk through set().
  //
  // The ghost's natural size is the WORKING SET, which is an absolute quantity and not a
  // function of how big a table someone chose.
  //
  // REFUSED for Always and KeepSighted: neither allocates a ghost, so a ghost size under
  // them is not a setting to be validated away but a shape with no representation here.
  static NNCacheShape directMapped(NNCacheReplacementPolicy replacement, int ghostPowerOfTwo);
  // An open-addressed table of `ways` slots per bucket, choosing its victim by `eviction`.
  static NNCacheShape probed(NNCacheCollisionScheme scheme, int ways, NNCacheEvictionPolicy eviction);
  // Separate chaining: a collision costs nothing, and eviction is driven by `maxBytes`.
  //
  // `eviction` names which resident entry a region gives up when it is over budget. It
  // is a required argument for the same reason probed() has one: a capacity sweep makes
  // a choice, and a choice the operator cannot see is a policy in force under no name.
  // Random, Lru and Lfu are accepted; None is refused, because a chained table's budget
  // is always enforced and "evict nothing" is not a shape this table can be.
  static NNCacheShape chained(int64_t maxBytes, NNCacheEvictionPolicy eviction);

  NNCacheCollisionScheme scheme() const { return scheme_; }
  // Meaningful only when scheme() != Direct. Under Direct there is no choice to report.
  // Under a probed scheme it is the collision victim; under Chain it is the capacity
  // victim. Both are "which resident entry is given up", so they share the axis.
  NNCacheEvictionPolicy eviction() const;
  // Meaningful only when scheme() == Direct, and refused otherwise for the same reason
  // eviction() is refused under Direct: a probed or chained table's newcomer is never a
  // candidate for removal, so there is no two-way choice to report.
  NNCacheReplacementPolicy replacement() const;
  // Slots in the sighting-count ghost, as a power of two. std::nullopt means "not stated,
  // so derive it from nnCacheSizePowerOfTwo", which is what the ghost did before this key
  // existed -- an optional rather than a negative sentinel, because that absence carries a
  // meaning and a bare sentinel would leave the meaning in a comment (ADR-0012 P11).
  // Refused for any shape or rule that has no ghost at all.
  std::optional<int> sightingGhostPowerOfTwo() const;
  // Slots per bucket. Always 1 under Direct.
  int ways() const { return ways_; }
  // Byte budget the table must stay under. 0 means unbounded. Nonzero only under Chain.
  int64_t maxBytes() const { return maxBytes_; }

  // True exactly for the shape that reproduces today's cache behaviour. Direct mapping
  // alone is no longer sufficient: a direct-mapped table with a replacement rule is a
  // different table.
  bool isStatusQuo() const {
    return scheme_ == NNCacheCollisionScheme::Direct && replacement_ == NNCacheReplacementPolicy::Always;
  }

  std::string toString() const;

 private:
  NNCacheShape(
    NNCacheCollisionScheme scheme, int ways, NNCacheEvictionPolicy eviction, int64_t maxBytes,
    NNCacheReplacementPolicy replacement, std::optional<int> sightingGhostPowerOfTwo
  );

  NNCacheCollisionScheme scheme_;
  int ways_;
  NNCacheEvictionPolicy eviction_;
  int64_t maxBytes_;
  NNCacheReplacementPolicy replacement_;
  std::optional<int> sightingGhostPowerOfTwo_;
};

// Everything needed to build a cache table.
struct NNCacheConfig {
  // Slot count is 2^sizePowerOfTwo. A negative value disables the cache entirely,
  // which is the meaning NNEvaluator has always given it.
  int sizePowerOfTwo;
  int mutexPoolSizePowerOfTwo;
  NNCacheShape shape;
  NNCacheAdmissionPolicy admission;

  // The status-quo configuration: exactly what NNEvaluator built before the policy
  // axes existed.
  static NNCacheConfig statusQuo(int sizePowerOfTwo, int mutexPoolSizePowerOfTwo);

  // The one place .cfg text becomes a cache configuration. sizePowerOfTwo and
  // mutexPoolSizePowerOfTwo are supplied by the caller because their per-command
  // defaults already live in Setup; this reads only the policy keys. Refuses an
  // unknown value or an incoherent combination by throwing, naming the valid
  // vocabulary (ADR-0002 rung 1).
  static NNCacheConfig fromCfg(ConfigParser& cfg, int sizePowerOfTwo, int mutexPoolSizePowerOfTwo);

  // The cfg keys this Port reads. Named once here so callers and tests agree (ADR-0012 P1).
  static const char* const KEY_COLLISION;   // nnCacheCollision
  static const char* const KEY_WAYS;        // nnCacheWays
  static const char* const KEY_EVICTION;    // nnCacheEviction
  static const char* const KEY_ADMISSION;   // nnCacheAdmission
  static const char* const KEY_MAX_BYTES;   // nnCacheMaxBytes
  static const char* const KEY_REPLACEMENT; // nnCacheReplacement
  static const char* const KEY_SIGHTING_GHOST_POW; // nnCacheSightingGhostPowerOfTwo

  static const std::set<std::string>& collisionVocabulary();
  static const std::set<std::string>& evictionVocabulary();
  static const std::set<std::string>& admissionVocabulary();
  static const std::set<std::string>& replacementVocabulary();

  // True exactly when this configuration reproduces today's behaviour.
  bool isStatusQuo() const;
};

// The true resident footprint of one cached entry's payload, in bytes.
//
// This is deliberately NOT sizeof(NNOutput) times a count. whiteOwnerMap and
// noisedPolicyProbs are separate heap blocks reached through pointers and are
// present on only some entries, so per-entry footprint is not a constant and a
// byte budget denominated in slots would be a bound in the wrong currency
// (ADR-0012, 2026-07-02 proxy-bound amendment). The slot array itself is a
// separate, fixed cost: 2^sizePowerOfTwo * sizeof(std::shared_ptr<NNOutput>).
size_t nnOutputFootprintBytes(const NNOutput& out);

// What a table is actually holding right now, in the currency a capacity-management
// question is asked in.
//
// This is a SNAPSHOT COMPUTED ON DEMAND, not a set of running counters: residentBytes
// for a direct-mapped or probed table means walking the slot array, which is O(slots)
// and takes every region lock in turn. It is therefore a reporting call for a harness
// to make between measured windows, never something to call inside a search. Nothing
// on the get/set path maintains any of these, so asking for them costs the hot path
// exactly nothing (ADR-0009: the instrument must not be inside the instrument).
//
// Why a struct rather than four accessors: the four numbers are only meaningful
// together -- residentEntries against capacitySlots is the occupancy that decides which
// regime a hit rate belongs to, and residentPayloadBytes against fixedStructureBytes is
// what a memory budget is actually spent on. Returning them one at a time would invite
// a reader to take two of them from two different moments.
struct NNCacheStats {
  // Entries the table would return on a get right now.
  int64_t residentEntries;
  // Sum of nnOutputFootprintBytes over exactly those entries: the heap the payloads
  // hold, including the separately-allocated ownership maps.
  int64_t residentPayloadBytes;
  // What the table costs whether it holds anything or not: the slot or bucket array,
  // per-slot metadata, the chained nodes' own bytes, the mutex pool, a ghost table.
  int64_t fixedStructureBytes;
  // The denominator occupancy is measured against: 2^sizePowerOfTwo for the direct and
  // probed tables. A chained table has no slot capacity -- it is bounded by bytes, not
  // by count -- and reports 0 here, so a reader is never handed a fabricated ratio.
  int64_t capacitySlots;
};

// A concurrent, hash-sharded table mapping an NN input hash to its NNOutput.
//
// The same position must land on the same slot whichever thread asks, so the
// sharding is over the hash and never over the calling thread.
class NNCacheTable {
 public:
  virtual ~NNCacheTable();

  NNCacheTable(const NNCacheTable& other) = delete;
  NNCacheTable& operator=(const NNCacheTable& other) = delete;

  // These are thread-safe. For get, ret will be set to nullptr upon a failure to find.
  virtual bool get(Hash128 nnHash, std::shared_ptr<NNOutput>& ret) = 0;
  virtual void set(const std::shared_ptr<NNOutput>& p) = 0;
  virtual void clear() = 0;

  // Thread-safe, and O(table). See NNCacheStats: a reporting call, not a hot-path one.
  virtual NNCacheStats stats() const = 0;

  // Builds the table a config asks for. Throws, naming what is missing, for a
  // shape that is coherent but not implemented yet -- never silently substituting
  // the default one (ADR-0002).
  static std::unique_ptr<NNCacheTable> create(const NNCacheConfig& config);

 protected:
  NNCacheTable();
};

#endif  // NEURALNET_NNCACHE_H_
