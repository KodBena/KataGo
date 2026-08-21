#ifndef NEURALNET_NNCACHE_H_
#define NEURALNET_NNCACHE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "../core/global.h"
#include "../core/hash.h"
#include "../neuralnet/nncachecontext.h"
#include "../neuralnet/nncacheobservations.h"
#include "../neuralnet/nninputs.h"

class ConfigParser;
class NNCacheFrozen;
class NNCacheTwoLevelTable;

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

  // THE DIRECTORY THIS CACHE'S PERSISTED CONTENT LIVES IN -- the one home of
  // <context>.nncounts and <context>.<model>.nnevals -- present exactly when the operator
  // set nnCacheDir.
  //
  // IT IS THE SAME DECISION AS "does this table carry a level-0 resolution list", and it is
  // one field rather than two because they are one fact. A directory with no list is a
  // configured path nothing can ever read or write; a list with no directory is an attach
  // surface with nowhere to attach from. Either as a separate key would be a state an
  // operator could reach and an engine would then have to explain (ADR-0012 P11); as one
  // field neither is representable. NNCacheTable::createWithLevelZeroList is what reads it.
  //
  // DEFAULTED IN THE DECLARATION, so that the several brace-initializations of this struct that
  // predate it -- statusQuo's and the policy tests' -- keep meaning exactly what they meant, and
  // so that the next field added here does not silently become "whatever the caller forgot".
  std::optional<std::string> cacheDirectory = std::nullopt;

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
  static const char* const KEY_DIR;         // nnCacheDir

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

// One row of the unified hit-count surface: a key, and how many times an evaluation was
// retrieved for it this session -- whichever level served the retrieval.
struct NNCacheHitCount {
  Hash128 key;
  uint32_t hits;
};

// Whether a table counts hits at all.
//
// This is a typed disposition rather than an empty row vector standing in for two quite
// different facts (ADR-0012 P11): "this table counts and nothing was hit" and "this table
// does not count" are not the same answer, and a caller that read an empty vector as the
// first when it was the second would silently persist nothing.
enum class NNCacheHitLedgerDisposition {
  // This table keeps no per-key hit counts. Every single-level table is here: counting is
  // a property of the two-level strategy, which exists exactly when a level 0 does, so the
  // default configuration pays nothing for a surface it does not use.
  NotCounted,
  // The rows are this session's counts, one row per key, from every level.
  Counted,
};

// The unified per-key hit-count surface. See NNCacheTable::harvestHitCounts.
//
// ONE ROW PER KEY, ALWAYS, because at most one level ever owns a key. Level 0 is frozen,
// so a key it holds is answered there and no evaluation follows; the two engine paths that
// can still offer a set for a level-0 key (a caller passing skipCache, and the
// ownership-map fall-through in NNEvaluator::evaluate) transfer that key's ownership --
// and its accrued count -- to level 1 rather than duplicating it. So a persistence layer
// reading this surface never merges, never double-counts, and never has to ask which
// structure answered.
class NNCacheHitLedger {
 public:
  static NNCacheHitLedger notCounted();
  static NNCacheHitLedger counted(std::vector<NNCacheHitCount> entries, int64_t unrecordedHits);

  NNCacheHitLedgerDisposition disposition() const { return disposition_; }
  bool isCounted() const { return disposition_ == NNCacheHitLedgerDisposition::Counted; }

  // The rows. Throws under NotCounted rather than handing back an empty vector a caller
  // could read as "nothing was hit".
  const std::vector<NNCacheHitCount>& entries() const;

  // Hits that occurred and could NOT be attributed to a row, because the level-1 ledger
  // had no room for the key. Zero in every ordinary run; nonzero means this harvest is
  // incomplete and says by how much, rather than being silently short (ADR-0002).
  int64_t unrecordedHits() const;

 private:
  NNCacheHitLedger(NNCacheHitLedgerDisposition disposition, std::vector<NNCacheHitCount> entries, int64_t unrecordedHits);

  NNCacheHitLedgerDisposition disposition_;
  std::vector<NNCacheHitCount> entries_;
  int64_t unrecordedHits_;
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
  virtual void clear() = 0;

  // DOES THIS TABLE HOLD AN ENTRY FOR `nnHash` RIGHT NOW, TOUCHING NOTHING AT ALL?
  //
  // WHY IT IS NOT get(), AND WHY IT IS NOT peek() EITHER. A get is a RETRIEVAL, and several
  // shapes record that one happened: the chained table moves the entry to the front of its
  // recency order, the probed table's eviction policy marks a hit, and the sighting table
  // counts a sighting for EVERY key it is handed -- present or absent, in a ghost that decides
  // which of two candidates keeps a slot. Those are correct responses to a query. They are
  // wrong responses to an OWNERSHIP QUESTION asked by machinery that is not querying at all:
  // NNCacheLevelZeroSources::attach asks it once per entry of an arriving level-0 source, so a
  // get-shaped probe would count an entire card's worth of sightings that no client ever made
  // and would rewrite a replacement policy's own data at a session boundary (ADR-0009 -- the
  // instrument must not be inside the instrument).
  //
  // PURE VIRTUAL, DELIBERATELY, rather than defaulted to get(). A default is exactly the shape
  // that lets a table which never thought about this question perturb its own replacement state
  // silently, at a call site with no way to notice. The compiler refusing to build a table that
  // has not answered is the strongest available surface for that (ADR-0002).
  //
  // Thread-safe, and it takes whatever lock a lookup takes: it is not lock-free, it just leaves
  // no trace.
  [[nodiscard]] virtual bool contains(Hash128 nnHash) const = 0;

  // Retrieves an entry WITHOUT counting a retrieval against it, and without consulting any
  // pre-warmed level 0.
  //
  // WHY IT EXISTS. A dump has to read the entries it is about to write. Reading them through
  // get() would count each one as a retrieval, so the act of persisting a session's counts
  // would inflate the very counts being persisted -- the instrument inside the instrument
  // (ADR-0009). And a dump wants LEVEL 1's entries specifically: a level-0 entry's bytes are
  // already in the file by construction, so serving one here would hand a dump exactly the
  // duplicate it exists to avoid.
  //
  // The base answer is get(), which is correct for every single-level table: counting is a
  // property of the two-level strategy and no other shape counts anything. Only the
  // two-level table overrides it, and it overrides it to skip level 0.
  virtual bool peek(Hash128 nnHash, std::shared_ptr<NNOutput>& ret);

  //-----------------------------------------------------------------------------------
  // Context attribution of what this session earns
  //-----------------------------------------------------------------------------------

  // See nncachecontext.h. In one line: with more than one body of pre-warmed content
  // attached to a model, the engine cannot tell which of them a query served, because the
  // cache key names a position and nothing else -- so the query carries a tag, and this is
  // where the tag is spent.
  //
  // ATTRIBUTION LIVES ON THE BASE, not on the two-level table, and the reason is that "what
  // this session earned" is exactly the set of keys that reach set() -- which is true of
  // every table shape, single-level and two-level alike. Under a two-level table those keys
  // are precisely level 1's, since level 0 is frozen and never takes a set. Putting the
  // ledger here means one owner for the fact rather than one copy per table implementation
  // (ADR-0012 P1/P3), and it is what lets a context be attached to an ordinary cache at all.

  // Registers a context entries may be attributed to, and returns the only kind of value
  // that can address it. Throws StringError, naming the name, for a name outside the closed
  // alphabet or one already attached.
  //
  // The FIRST attach allocates the ledger. A table with no attached context has none, so the
  // default configuration -- which attaches nothing, because no protocol action attaches yet
  // -- pays one predictable null test per set and not one byte of memory.
  NNCacheContextId attachCacheContext(const std::string& name);

  // What this table has attached, for a resolution boundary to refuse an unknown name
  // against. Const: attaching goes through attachCacheContext, which owns the ledger's
  // lifetime too.
  [[nodiscard]] const NNCacheContextSet& cacheContexts() const;

  // Stores `p` exactly as set(p) does, having first recorded which context earned it.
  //
  // Thread-safe, and NOT virtual: recording is the same act for every table shape, and the
  // storing half is the virtual one it delegates to. Throws StringError for an attribution
  // naming a context this table did not attach -- an id minted by another model's cache
  // cannot be spent here, and is refused rather than indexing into this table's name space
  // at the same position.
  void set(const std::shared_ptr<NNOutput>& p, const NNCacheAttribution& attribution);

  // The same, saying where the entry came from. The two-argument form above is exactly this
  // one at LiveEvaluation -- which is what every evaluation path is, and what the default
  // keeps costing nothing -- and the LoadedFromContainer form is for the ONE caller that has
  // a different answer: an attach filling level 1 from the context's own container.
  //
  // See NNCacheEntryProvenance for why this is recorded here, at admission, rather than
  // reconstructed at dump time.
  void set(
    const std::shared_ptr<NNOutput>& p,
    const NNCacheAttribution& attribution,
    NNCacheEntryProvenance provenance
  );

  // Exactly the keys `context` earned, for an implementation that can put hit counts beside
  // them and for a dump that needs the denominator its own selection is a numerator of.
  // Total: a context this table did not attach is refused by attachCacheContext's own
  // ownership rule before it can be asked about here.
  [[nodiscard]] std::vector<Hash128> attributedKeysFor(const NNCacheContextId& context) const;

  // Exactly the keys `context` earned whose bytes are not on disk: what a dump of this
  // context owes. Empty for a table with no attached context, and refused for a context this
  // table did not attach, exactly as the attribution surface is.
  [[nodiscard]] std::vector<Hash128> unpersistedKeysFor(const NNCacheContextId& context) const;

  // Records that these keys of `context` are now on disk, and returns how many rows were
  // marked. Called after a container append has succeeded and never before: a mark set ahead
  // of a write that then throws would drop the entry from every future dump.
  int64_t markPersisted(const NNCacheContextId& context, const std::vector<Hash128>& keys);

  // The key -> context ledger of what this session earned, for whoever persists it.
  // Thread-safe and O(ledger); a reporting call taken between sessions, never in a search.
  //
  // NotAttributed when no context was ever attached -- the same typed disposition
  // harvestHitCounts uses, and for the same reason.
  //
  // EARNED MEANS EVALUATED AND OFFERED, NOT NECESSARILY RESIDENT, and the difference is named
  // rather than left to be discovered. Recording happens where the attribution arrives -- at
  // this entry point -- and the storing half below it may still decline the entry: a
  // SecondSighting admission policy drops a key on its first sighting, and a capacity sweep
  // can drop it later. Under the default policy the two sets coincide exactly. Under any other
  // the ledger is a SUPERSET of what the table holds, which is the honest reading for a record
  // of what a study session worked on, and which a dump of evaluations intersects with the
  // table anyway. Reading it as "what is resident" would be wrong under those policies.
  [[nodiscard]] NNCacheAttributionLedger harvestAttribution() const;

  //-----------------------------------------------------------------------------------
  // Observation counting: the currency the count log records
  //-----------------------------------------------------------------------------------
  //
  // See nncacheobservations.h for what an observation IS and why the currency is this and
  // not retrievals. What belongs here is only the door and its cost.

  // MINTS THE PRESENTATION OF THIS POSITION, COUNTING IT. Thread-safe, and NOT virtual:
  // presenting is the same act for every table shape, and no table shape has any say in it.
  //
  // THE COUNTED ACT IS THE VALUE THIS RETURNS, and the request-path get() and set() below take
  // one and nothing else -- so on that surface a position cannot be looked up or stored without
  // having been counted, rather than the counting being a call a caller must remember to pair
  // with them. See NNCachePresentation for what that replaced and why.
  //
  // CALLED ONCE PER DEMAND -- by NNEvaluator::evaluate, immediately after the position's hash
  // is final and before any level is consulted. Hit or miss, get or skipCache-set, one demand
  // is one observation.
  //
  // THE COST WHEN NOTHING IS ATTACHED IS ONE PREDICTABLE BRANCH. The ledger is allocated by
  // the first attachCacheContext and by nothing else, so plain play -- no cache directory, no
  // context, which is the overwhelmingly common configuration -- pays one null test against a
  // pointer that is null for the life of the process, and not one byte of memory (ADR-0009).
  // A NoAttributableContext attribution costs one more test and no work: an observation
  // belongs to a card or to no file at all, and there is no third place to put it.
  //
  // Throws StringError for an attribution naming a context this table did not attach, exactly
  // as set() does and from the same rule: an id minted by another model's cache cannot be
  // spent here.
  //
  // DEFINED HERE RATHER THAN IN THE .cpp, and that is the whole of the "zero-to-one branch"
  // claim above: inlined, the no-context configuration compiles to one test of a pointer that
  // is null for the life of the process and a call that is never made. Out of line it would be
  // an unconditional call on the path MCTS hammers, which is a cost the default configuration
  // has no reason to pay for a feature it does not use (ADR-0009).
  [[nodiscard]] NNCachePresentation present(Hash128 nnHash, const NNCacheAttribution& attribution) {
    if(observations_ != nullptr)
      recordObservation(nnHash, attribution);
    return NNCachePresentation(nnHash);
  }

  // THE SAME VALUE WITHOUT COUNTING, for an evaluation that SERVES a demand some earlier call
  // already counted rather than being one.
  //
  // ITS ONLY CALLER IS A FAN-OUT, and there is exactly one today:
  // NNEvaluator::averageMultipleSymmetries runs N forward passes of ONE key for one root query,
  // and the operator has ruled that this is one demand and not N (ledger row 1814). The first
  // iteration calls present(); every later one calls this. That choice is a line at the call
  // site, which is the whole reason there are two named mints rather than one call with a bool:
  // a future fan-out must WRITE which its evaluations are, where a reviewer can see it.
  //
  // It is not a hole in the counting. Reaching it requires already believing the demand was
  // counted, and saying so by name, in a function that is fanning one request out.
  [[nodiscard]] NNCachePresentation presentAgainForSameRequest(Hash128 nnHash) {
    return NNCachePresentation(nnHash);
  }

  // THE REQUEST PATH'S LOOKUP AND STORE: they take a presentation and no other key.
  //
  // Non-virtual and inline. They forward to the virtuals below, which is where every table
  // shape's real work is and which is unchanged -- so this costs the hot path nothing and
  // changes no dispatch. What it buys is that the request path cannot express a lookup for a
  // position it never presented.
  bool get(const NNCachePresentation& presentation, std::shared_ptr<NNOutput>& ret) {
    return get(presentation.key(), ret);
  }
  void set(
    const NNCachePresentation& presentation,
    const std::shared_ptr<NNOutput>& p,
    const NNCacheAttribution& attribution
  ) {
    // The presentation and the payload must be of the same position, or the entry is filed
    // under a key nobody asked about. Cheap, always-taken-the-same-way, and it catches the one
    // way these two arguments can disagree (ADR-0002).
    if(presentation.key() != p->nnHash)
      throw StringError(
        "NNCacheTable::set: the presentation and the evaluation are of different positions. "
        "Storing this would file the evaluation under a key no request ever asked about."
      );
    set(p, attribution);
  }

  // Exactly the rows a dump of `context` appends: one per key this context has presented since
  // the last dump, carrying that many observations. CONSUMING -- it advances each row's
  // persisted mark -- and OMITTING A KEY WITH NOTHING TO SAY, because a count-log record is an
  // INCREMENT and its presence raises that key's `sessions`.
  //
  // A ROW FOR EVERY KEY OBSERVED, INCLUDING ONE OBSERVED EXACTLY ONCE whose evaluation no
  // admission threshold will let onto disk. That is the whole mechanism of the cross-session
  // bootstrap: the count is what a later session adds to, so it must be written in the session
  // that could not yet use it (ratified spec, ledger rows 1717/1722).
  //
  // NotObserved from a table with no attached context. Throws StringError for a context this
  // table did not attach.
  [[nodiscard]] NNCacheObservationLedger takeUnpersistedObservationCountsFor(const NNCacheContextId& context);

  // The same population reported WITHOUT taking: this session's running observation count for
  // every key of `context`, every mark left where it was. For a one-shot report, never for a
  // dump -- appendDump takes a delta, and this is a level.
  [[nodiscard]] NNCacheObservationLedger harvestObservationCountsFor(const NNCacheContextId& context) const;

  // ARE THERE UNPERSISTED OBSERVATIONS FOR `context`, WITHOUT TAKING THEM?
  //
  // THE NON-CONSUMING QUESTION, which is the only question a REFUSAL can ask. cache_detach
  // refuses to drop a context holding work that has not reached disk; taking the delta is
  // exactly what makes it safe to append, so a refusal that asked the consuming question would
  // destroy the thing it refused to lose. TRUE MEANS EXACTLY
  // "takeUnpersistedObservationCountsFor(context) WOULD YIELD AT LEAST ONE ROW".
  //
  // False from a table with no attached context, which is not a hedge: a table that observes
  // nothing has nothing unpersisted. Throws StringError for a context this table did not attach.
  [[nodiscard]] bool hasUnpersistedObservationCountsFor(const NNCacheContextId& context) const;

  // Every observed row of every context, this session, for a whole-table one-shot report.
  // NotObserved from a table with no attached context.
  [[nodiscard]] NNCacheObservationLedger harvestObservationCounts() const;

  // What the observation ledger costs in resident memory, or ZERO for a table with no attached
  // context -- which is a real figure and not an absence, because such a table allocates
  // nothing. Reported by cache_stats: it is tens of megabytes on a table that has attached
  // anything, and a memory bill this feature adds is a bill the operator gets to see
  // (ADR-0002).
  [[nodiscard]] int64_t observationStructureBytes() const;

  // Thread-safe, and O(table). See NNCacheStats: a reporting call, not a hot-path one.
  virtual NNCacheStats stats() const = 0;

  // The unified per-key RETRIEVAL counts of this session: how many times each key was
  // actually SERVED out of a level, whichever level served it. Thread-safe and O(table); a
  // reporting call taken between sessions, never inside a search.
  //
  // THIS IS NOT WHAT THE COUNT LOG PERSISTS, and the two are not two homes for one fact.
  // Retrievals answer "how well is the cache working" -- a diagnostic, reported by
  // cache_stats and by nothing else. What a dump writes is OBSERVATIONS, which count every
  // presentation whether a level answered it or a forward pass had to, and which live in a
  // different structure with a different door (see observe() above and
  // nncacheobservations.h). Nothing appends this surface to a count log; there is no delta
  // twin of it, deliberately, so there is no expression that could.
  //
  // A single-level table returns NotCounted, which is the default configuration's answer
  // and is why the default get/set path is untouched by this surface existing. The
  // two-level table returns Counted with one row per key.
  virtual NNCacheHitLedger harvestHitCounts() const;

  // Builds the table a config asks for. Throws, naming what is missing, for a
  // shape that is coherent but not implemented yet -- never silently substituting
  // the default one (ADR-0002).
  static std::unique_ptr<NNCacheTable> create(const NNCacheConfig& config);

  // THE SAME TABLE, CARRYING A LEVEL-0 RESOLUTION LIST: what an evaluator gets when the
  // operator configured a cache directory, and what the cache_attach and cache_detach
  // actions of the analysis engine drive.
  //
  // It returns NNCacheTwoLevelTable -- the type that carries attach and detach -- rather
  // than the NNCacheTable base, so a caller that needs that surface HOLDS it in its type
  // and no downcast from a base pointer is written anywhere. `config` describes LEVEL 1
  // exactly as create() would build it; level 0 is not a configuration but content a
  // session attaches, so the returned table starts with NO attached source and resolves
  // every get from level 1, which is what create() would have done.
  //
  // Throws StringError if config.cacheDirectory is empty -- a level-0 list with nowhere to
  // load from is the half-configured state NNCacheConfig::cacheDirectory exists to make
  // unreachable, and it is refused here rather than built and left useless.
  static std::unique_ptr<NNCacheTwoLevelTable> createWithLevelZeroList(const NNCacheConfig& config);

 protected:
  NNCacheTable();

  // THE RAW, HASH-KEYED SURFACE. PROTECTED, NOT PUBLIC: the presentation-minted
  // get(NNCachePresentation,...)/set(NNCachePresentation,...,attribution) pair above, and the
  // named accessors elsewhere in this class (peek, the attributed-set overloads, the
  // observation and attribution surfaces), are what production code outside this class
  // hierarchy is entitled to call. A caller holding a bare Hash128 and an NNOutput has no
  // presentation, no attribution, and no ledger entry -- exactly the state the counting and
  // attribution machinery above exists to make unreachable from the request path -- so the
  // raw form is not exposed for a caller to reach for instead. Every table shape still
  // implements it: it is what the presentation-minted overloads forward to, and it is the
  // only door left to concrete table implementations, decorators, the one bench tool that
  // replays a byte-level trace with no presentation to mint, and one test-tree shim (see the
  // two friends below).
  //
  // These are thread-safe. For get, ret will be set to nullptr upon a failure to find.
  virtual bool get(Hash128 nnHash, std::shared_ptr<NNOutput>& ret) = 0;
  virtual void set(const std::shared_ptr<NNOutput>& p) = 0;

  // A DECORATOR OVER NNCacheTable (tracing, second-sighting admission, the two-level table's
  // level 1) holds its wrapped table as a plain `unique_ptr<NNCacheTable>` and must call that
  // wrapped table's own raw get/set. Ordinary protected inheritance does not reach it: the
  // wrapped object's static type is NNCacheTable itself, not the decorator's own derived type,
  // and the additional access check on a protected NON-STATIC member ([class.protected])
  // refuses access through an object typed as the base class even from a member of a derived
  // class. A protected STATIC member carries no such restriction, so these two forwarders are
  // the bounded fix: any NNCacheTable subclass may call them on any other table, and nothing
  // outside this class hierarchy can, because they are themselves protected.
  static bool getRaw(NNCacheTable& table, Hash128 nnHash, std::shared_ptr<NNOutput>& ret) {
    return table.get(nnHash, ret);
  }
  static void setRaw(NNCacheTable& table, const std::shared_ptr<NNOutput>& p) {
    table.set(p);
  }

 private:
  // TWO NARROW, NAMED EXCEPTIONS to "production code outside the presentation-minted path
  // cannot call raw get/set" -- neither is a general-purpose backdoor.
  //
  //   * NNCacheTableBenchAccess (cpp/command/benchnncachepolicy.cpp) replays a captured
  //     byte-level trace of (hash, bytes) tuples straight against a freshly-built table, to
  //     measure the table's own hash-tag and eviction-policy cost. There is no board position
  //     and no NNEvaluator in that tool, so there is nothing to mint a presentation from --
  //     the raw form is what the measurement is OF.
  //   * NNCacheTableTestAccess (cpp/tests/nncachetabletestaccess.h) is the one door the
  //     existing get/set-driven unit tests reach the raw surface through, so a correctness
  //     test can still exercise get/set directly without every test file becoming a friend of
  //     its own.
  friend class NNCacheTableBenchAccess;
  friend class NNCacheTableTestAccess;
  // What observe() does once the null test above it has passed. Out of line so the hot path
  // holds one branch and a call that is never taken, rather than the ownership check and the
  // attribution switch inlined into every evaluation.
  void recordObservation(Hash128 nnHash, const NNCacheAttribution& attribution);

  // The ownership refusal every per-context surface makes, written once (ADR-0012 P1).
  void refuseForeignContext(const NNCacheContextId& context) const;

  // BOTH ALLOCATED BY THE FIRST attachCacheContext, IN ONE STATEMENT, AND NEVER OTHERWISE.
  // They are two different facts about one session -- which context earned a key, and how
  // often each key was presented -- so they are two structures; but "one exists and the other
  // does not" is a state no caller could act on, so there is no path that creates one alone.
  std::unique_ptr<NNCacheAttributionRecorder> attribution_;
  std::unique_ptr<NNCacheObservationRecorder> observations_;
  NNCacheContextSet contexts_;
};

#endif  // NEURALNET_NNCACHE_H_
