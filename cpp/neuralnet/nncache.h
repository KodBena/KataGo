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

  // These are thread-safe. For get, ret will be set to nullptr upon a failure to find.
  virtual bool get(Hash128 nnHash, std::shared_ptr<NNOutput>& ret) = 0;
  virtual void set(const std::shared_ptr<NNOutput>& p) = 0;
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

  // The hit-count rows a dump of exactly `context` writes: one per key that context earned,
  // carrying the hits that key took this session. A key earned and never retrieved again
  // appears with zero hits, because "this context earned this position and nothing came back
  // for it" is precisely the fact that says to stop carrying it.
  //
  // NotCounted from a table that keeps no per-key hit counts, exactly as harvestHitCounts is.
  // Throws StringError for a context this table did not attach.
  //
  // unrecordedHits() on the returned ledger is zero, and that is a statement rather than an
  // omission: the unrecorded-hit and unattributed-entry residues are hits and entries whose
  // CONTEXT is the thing that was lost, so they cannot be divided among contexts. They are
  // reported once, whole, by harvestHitCounts() and harvestAttribution(), and are never
  // smeared across per-context dumps where each context would appear to carry all of them.
  [[nodiscard]] virtual NNCacheHitLedger harvestHitCountsFor(const NNCacheContextId& context) const;

  // THE DELTA TWIN OF harvestHitCountsFor: exactly the rows a dump of `context` appends, taking
  // them as it reports them.
  //
  // It stands to harvestHitCountsFor exactly as takeUnpersistedHitCounts stands to
  // harvestHitCounts -- CONSUMING, and OMITTING A KEY WITH NOTHING TO SAY -- and it exists for
  // the same reason that one does: a count log record is an INCREMENT, so what a dump may
  // append is the delta, never the running total.
  //
  // WHY A PER-CONTEXT DELTA IS A SURFACE AND NOT A FILTER THE CALLER APPLIES. The whole-table
  // delta is exact only while one context is attached, because then everything the table earned
  // and everything its level 0 served belongs to that one context. With two attached, a caller
  // holding the whole-table delta cannot divide it: the rows carry keys, and a key names a
  // position, never a card. The division is knowable only where the two facts live -- which
  // level-0 source serves which context, and which context earned which level-1 key -- and both
  // of those live in the table. So the table divides, and the marks it advances are exactly the
  // ones whose rows it handed over (ADR-0012 P1).
  //
  // unrecordedHits() on the result is zero, for the reason harvestHitCountsFor states: the
  // residues are hits whose CONTEXT is what was lost, so they cannot be divided among contexts
  // and are reported once, whole, by the whole-table surfaces.
  //
  // WHAT NO PER-CONTEXT DUMP WRITES, named rather than left to be discovered. Retrievals of a key
  // that no context can be attributed to -- a key earned while several contexts were attached and
  // the request named none, or earned before any context was -- belong to no context's file, so
  // they are written by no per-context dump and their marks never advance. That population is not
  // silent: it is exactly what NNCacheAttributionLedger::noAttributableContextEntries counts, and
  // the dump action reports that figure in every response. The same is true of a level-0 source
  // attached under no context, which only the whole-table delta above ever reaches.
  //
  // NotCounted from a table that keeps no per-key hit counts. Throws StringError for a context
  // this table did not attach.
  [[nodiscard]] virtual NNCacheHitLedger takeUnpersistedHitCountsFor(const NNCacheContextId& context);

  // ARE THERE UNPERSISTED HIT COUNTS FOR `context`, WITHOUT TAKING THEM?
  //
  // THE NON-CONSUMING QUESTION, which is the only question a REFUSAL can ask. cache_detach
  // refuses to drop a context holding work that has not reached disk; to ask "is there any" it
  // may not use the delta surface, because taking the delta advances the marks -- which is
  // exactly what makes the delta safe to append, and exactly what would make a refusal destroy
  // the thing it refused to lose. Before this existed the refusal read a protocol-layer PROXY
  // (whether any request had been accepted since the last counts dump), which under-refuses:
  // a retrieval served entirely out of level 0 calls no set(), records no attributed key, and
  // accepts no new request, so a session that only re-studied pre-warmed positions could be
  // detached with every one of its retrievals unwritten.
  //
  // TRUE MEANS EXACTLY "takeUnpersistedHitCountsFor(context) WOULD YIELD AT LEAST ONE ROW", so
  // the two cannot disagree about a state. False from a table that keeps no per-key hit counts
  // -- which is not a hedge: a table that counts nothing has nothing unpersisted.
  //
  // Const, and it advances no mark anywhere. Throws StringError for a context this table did
  // not attach.
  [[nodiscard]] virtual bool hasUnpersistedHitCountsFor(const NNCacheContextId& context) const;

  // Thread-safe, and O(table). See NNCacheStats: a reporting call, not a hot-path one.
  virtual NNCacheStats stats() const = 0;

  // The unified per-key hit counts of this session, for whoever persists them. Thread-safe
  // and O(table); a reporting call taken between sessions, never inside a search.
  //
  // A single-level table returns NotCounted, which is the default configuration's answer
  // and is why the default get/set path is untouched by this surface existing. The
  // two-level table returns Counted with one row per key.
  virtual NNCacheHitLedger harvestHitCounts() const;

  // THE COUNTS THAT HAVE NOT REACHED THE COUNT LOG YET, and taking them advances the mark.
  //
  // This is the numeric twin of the persisted bit, and it exists for the same defect in the
  // same shape. A count log record is an INCREMENT: appendDump adds each row's lookups to
  // that key's running total and adds one to its sessions. So a dump must hand it the DELTA
  // this attachment earned, and a second dump with nothing in between must hand it NOTHING
  // -- otherwise an attach, a dump, a detach and a re-attach re-append what the attach
  // loaded, and the record inflates by a whole session's worth of retrievals that never
  // happened.
  //
  // TWO PROPERTIES, both of which harvestHitCounts() deliberately does NOT have, which is
  // why this is a second surface rather than a change to that one:
  //
  //   IT IS CONSUMING. Taking the rows advances each key's persisted mark to its current
  //   count, so the next take reports only what accrued after this one. The accumulated
  //   total's one home is the count log file; what the in-memory counters hold is exactly
  //   what has not reached it (ADR-0012 P1).
  //
  //   IT OMITS A KEY WITH NOTHING TO SAY. harvestHitCounts() reports a pre-warmed entry that
  //   earned nothing with a row of zero hits, deliberately -- that is the fact that says to
  //   stop carrying it. A DUMP must not write that row, because appendDump would raise its
  //   sessions, so an attach-dump-detach-attach-dump cycle over an untouched card would
  //   climb the sessions of every key it loaded while nothing was ever looked up.
  //
  // NotCounted from a table that keeps no per-key hit counts, exactly as harvestHitCounts is.
  [[nodiscard]] virtual NNCacheHitLedger takeUnpersistedHitCounts();

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

 private:
  // Allocated by the first attachCacheContext and never before. See attachCacheContext.
  std::unique_ptr<NNCacheAttributionRecorder> attribution_;
  NNCacheContextSet contexts_;
};

#endif  // NEURALNET_NNCACHE_H_
