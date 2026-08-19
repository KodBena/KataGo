#ifndef NEURALNET_NNCACHECONTEXT_H_
#define NEURALNET_NNCACHECONTEXT_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "../core/hash.h"

// THE CACHE CONTEXT: WHAT EARNED AN ENTRY, AS A NAME THIS ENGINE DOES NOT INTERPRET.
//
// A client that has attached more than one body of pre-warmed cache content to a model needs
// to know which of them a session's NEW entries belong to, because that is what decides which
// file those entries and their counts are dumped into. The engine cannot infer it: the cache
// key names a position, never a card, a deck or a session (NNInputs::getHash folds in board,
// rules, komi and search-shape flags and nothing else), so with two contexts attached the
// engine has no way to tell which one a query served -- unless the query says.
//
// SO THE QUERY SAYS, AND THE ENGINE UNDERSTANDS NOTHING BY IT. A context is an OPAQUE STRING
// the client supplies and the engine only ever compares for equality, validates as a path
// component, and stores. There is no relation between contexts, no ordering, no heredity, no
// containment, and no vocabulary of what a context "is" -- deliberately, and this is the
// operator's own standing constraint on the design: the caller supplies context ids only, and
// KataGo interprets no semantics. Nothing in this header knows what a card, a deck or a
// lineage is, and nothing downstream of it may learn.
//
// WHAT IS TYPED HERE, AND WHY EACH TYPE EXISTS.
//
//   NNCacheContextId is an ATTACHED context, and it can only be minted by the
//   NNCacheContextSet that attached it. There is no public constructor and no conversion
//   from a string or an integer. An unvalidated, unattached, or misspelled context id is
//   therefore not a value this program can hold at the point where an entry is attributed --
//   it is not "checked and rejected", it cannot be constructed (ADR-0000 Rule 2a). Every id
//   also carries the identity of the set that minted it, so an id from one model's cache
//   used against another model's is refused by name rather than silently indexing into a
//   different name space -- the wrong-axis lookup class SearchableModelIdx forecloses at
//   compile time on the axis where that was available, foreclosed here at the only surface
//   two independently-constructed sets leave open.
//
//   NNCacheAttribution is what an entry was earned by, as a CLOSED two-case disposition:
//   a context, or no attributable context. It is not an optional id with a comment saying
//   what the absence means (ADR-0012 P11): "several contexts are attached and this request
//   named none" is a fact a dump must report, not a null to be read past.
//
//   NNCacheContextResolution is what a request's optional "cacheContext" field resolved to,
//   carrying its own refusal when it resolved to nothing. Resolved-and-refused cannot both
//   be true and neither can both be false, exactly as ModelResolution does for the "model"
//   field the same requests carry.

class NNCacheContextSet;

//-------------------------------------------------------------------------------------
// The id
//-------------------------------------------------------------------------------------

// One attached context, as a value that can only mean that.
//
// Minted by NNCacheContextSet::attach and by nothing else: the constructor is private and
// the set is its only friend. So there is no path by which a raw string a client typed, or
// an index a loop produced, becomes an attribution -- the resolution boundary is the only
// door, and it is the door that can refuse.
class NNCacheContextId {
 public:
  // Which set minted this, so a set can refuse an id that is not its own.
  [[nodiscard]] uint64_t setId() const { return setId_; }
  // Its position in that set's attach order. Unwrapped in exactly one place: the set's own
  // name and row lookups, which is what keeps this from being a size_t wearing a hat.
  [[nodiscard]] size_t index() const { return idx_; }

  [[nodiscard]] bool operator==(const NNCacheContextId& other) const {
    return setId_ == other.setId_ && idx_ == other.idx_;
  }
  [[nodiscard]] bool operator!=(const NNCacheContextId& other) const { return !(*this == other); }

 private:
  friend class NNCacheContextSet;
  NNCacheContextId(uint64_t setId, size_t idx) : setId_(setId), idx_(idx) {}

  uint64_t setId_;
  size_t idx_;
};

//-------------------------------------------------------------------------------------
// The attribution
//-------------------------------------------------------------------------------------

enum class NNCacheAttributionKind {
  // This entry was earned by a named, attached context.
  ToContext,
  // Nothing in the request said which context earned it, and more than one was attached --
  // or none was. This is a REPORTED outcome, not a missing value: an entry earned with no
  // attributable context is counted and surfaced as such, never assigned to whichever
  // context happened to be first.
  NoAttributableContext,
};

class NNCacheAttribution {
 public:
  // The default is NoAttributableContext, which is what every request that says nothing
  // carries and is the pre-existing behaviour of every path that never mentions a context.
  NNCacheAttribution();

  static NNCacheAttribution toContext(NNCacheContextId id);
  static NNCacheAttribution noAttributableContext();

  [[nodiscard]] NNCacheAttributionKind kind() const { return id_.has_value() ? NNCacheAttributionKind::ToContext : NNCacheAttributionKind::NoAttributableContext; }
  [[nodiscard]] bool isToContext() const { return id_.has_value(); }

  // Throws under NoAttributableContext rather than handing back a fabricated id.
  [[nodiscard]] NNCacheContextId contextId() const;

 private:
  explicit NNCacheAttribution(std::optional<NNCacheContextId> id);
  std::optional<NNCacheContextId> id_;
};

//-------------------------------------------------------------------------------------
// The resolution of a request's field
//-------------------------------------------------------------------------------------

// What a request's optional "cacheContext" field selected, carrying its own reason when it
// selected nothing. Constructed by the two named factories and by nothing else.
class NNCacheContextResolution {
 public:
  // The field named an attached context, or named nothing and the default applied.
  static NNCacheContextResolution resolved(NNCacheAttribution attribution);
  // The field named something this model has not attached. The message says what is
  // attached, so the client has something to act on.
  static NNCacheContextResolution refused(std::string message);

  // Present exactly when the request may proceed.
  [[nodiscard]] std::optional<NNCacheAttribution> attribution() const { return attribution_; }
  // Present exactly when it may not, and then it says why, for the client to read.
  [[nodiscard]] std::optional<std::string> refusal() const { return refusal_; }

 private:
  NNCacheContextResolution(std::optional<NNCacheAttribution> attribution, std::optional<std::string> refusal);
  std::optional<NNCacheAttribution> attribution_;
  std::optional<std::string> refusal_;
};

//-------------------------------------------------------------------------------------
// The name space
//-------------------------------------------------------------------------------------

// The contexts one cache table has attached, in attach order.
//
// This is the NAME SPACE half of attaching a context, and it is deliberately only that. It
// holds no cache content: binding a context to its own frozen level-0 structure -- reading
// the container, building the CHD, joining the count log for order -- is the loader's work,
// and this set is the name space that work attaches into. Splitting them is what lets the
// tag a request carries be plumbed, refused and attributed before any loader exists, and it
// is what a later cache_attach action calls when it has content to attach.
class NNCacheContextSet {
 public:
  NNCacheContextSet();

  // Registers a context that entries may be attributed to, and returns the only kind of
  // value that can address it.
  //
  // Throws StringError, naming the name, if it is not a legal context name or if it is
  // already attached. The alphabet is the shared one from NNCacheFileName: a context name
  // becomes a component of the path of <context>.nncounts and <context>.<model>.nnevals, so
  // it is validated to a closed alphabet at this boundary and REFUSED when it does not fit,
  // never escaped or rewritten into something acceptable (ADR-0012, the 2026-07-18
  // interpreter-boundary amendment). Validating it here rather than at the file layer is
  // what makes an attached context a name that is already safe to be a path everywhere
  // downstream; the file layer validates too, from the same one home, because it is also
  // reachable by callers who never attached.
  NNCacheContextId attach(const std::string& name);

  [[nodiscard]] bool empty() const { return names_.empty(); }
  [[nodiscard]] size_t size() const { return names_.size(); }
  // The attached names, in attach order.
  [[nodiscard]] const std::vector<std::string>& names() const { return names_; }

  // Whether this id was minted by this set. False for an id from any other set.
  [[nodiscard]] bool owns(const NNCacheContextId& id) const;
  // Throws StringError for an id this set did not mint, rather than indexing a name space
  // it does not belong to.
  [[nodiscard]] const std::string& nameOf(const NNCacheContextId& id) const;

  // THE PROTOCOL RULE, IN ONE PLACE.
  //
  //  * A name that is attached resolves to it.
  //  * A name that is not attached is REFUSED, and the refusal names it and lists what is
  //    attached. It is never coerced to the default: attributing this session's earnings to
  //    a context the client did not name writes them into the wrong file, and no response
  //    field would carry evidence that it happened (ADR-0002 Rule 2 -- a boundary validates
  //    and does not guess).
  //  * No name, with exactly ONE context attached, resolves to that context: with one
  //    attached, everything the session earns belongs to it, and there is nothing to infer.
  //  * No name, with any other number attached, resolves to NoAttributableContext -- which
  //    is counted and reported, not guessed.
  [[nodiscard]] NNCacheContextResolution resolveForRequest(const std::optional<std::string>& requested) const;

  // The identity ids carry. Distinct for every set constructed in this process.
  [[nodiscard]] uint64_t id() const { return setId_; }

 private:
  uint64_t setId_;
  std::vector<std::string> names_;
};

//-------------------------------------------------------------------------------------
// The harvested attribution surface
//-------------------------------------------------------------------------------------

// One key this session earned, and the context it was earned by.
struct NNCacheAttributionRow {
  Hash128 key;
  std::string context;
};

// Whether a table attributes its earnings at all.
//
// The same typed disposition NNCacheHitLedger carries, for the same reason: "no context was
// ever attached here, so attribution is not a question this table answers" and "contexts are
// attached and nothing was earned" are two different facts, and an empty row vector standing
// for both would let a dump persist nothing while reporting success.
enum class NNCacheAttributionDisposition {
  NotAttributed,
  Attributed,
};

class NNCacheAttributionLedger {
 public:
  static NNCacheAttributionLedger notAttributed();
  static NNCacheAttributionLedger attributed(
    std::vector<NNCacheAttributionRow> rows,
    int64_t noAttributableContextEntries,
    int64_t unrecordedAttributions
  );

  [[nodiscard]] NNCacheAttributionDisposition disposition() const { return disposition_; }
  [[nodiscard]] bool isAttributed() const { return disposition_ == NNCacheAttributionDisposition::Attributed; }

  // One row per key attributed to a context. Throws under NotAttributed.
  [[nodiscard]] const std::vector<NNCacheAttributionRow>& rows() const;

  // Entries this session earned that NO context could be attributed to -- several contexts
  // attached and the request naming none. Reported so a dump can say how much of the session
  // it is not writing into any context's file, rather than that content vanishing or being
  // guessed into a card (ADR-0002). Throws under NotAttributed.
  [[nodiscard]] int64_t noAttributableContextEntries() const;

  // Attributions that happened and could NOT be given a row, because the recorder's probe
  // window was full. Zero in every ordinary run; nonzero means this harvest is incomplete and
  // says by how much -- the same honesty NNCacheHitLedger::unrecordedHits carries for hits.
  // Throws under NotAttributed.
  [[nodiscard]] int64_t unrecordedAttributions() const;

 private:
  NNCacheAttributionLedger(
    NNCacheAttributionDisposition disposition,
    std::vector<NNCacheAttributionRow> rows,
    int64_t noAttributableContextEntries,
    int64_t unrecordedAttributions
  );

  NNCacheAttributionDisposition disposition_;
  std::vector<NNCacheAttributionRow> rows_;
  int64_t noAttributableContextEntries_;
  int64_t unrecordedAttributions_;
};

//-------------------------------------------------------------------------------------
// The recorder
//-------------------------------------------------------------------------------------

// The per-key key -> context ledger a cache table writes on its set path.
//
// Shaped exactly like the two-level table's hit ledger, for the same reasons: a fixed row
// array with a bounded probe window and a mutex pool, so a set costs one lock and O(1) work
// and never a rehash; the FULL 128-bit key beside every row, so an attribution is never given
// to the wrong key; and an overflow that is REFUSED and counted rather than absorbed by
// overwriting somebody else's row.
//
// LAST WRITER WINS for a key attributed twice. The alternative considered and rejected was
// first-wins: a key is re-set within a session by the ownership-map fall-through and by any
// skipCache caller, and under first-wins a context that paid for the fuller evaluation would
// see it filed under a context that has not been active since. The rejected reading is not
// absurd -- "who earned it first" is a defensible sense of earned -- but the entry that is
// resident is the one the later evaluation produced, and it is that entry a dump writes.
//
// IT IS ALLOCATED ONLY WHEN A CONTEXT IS ATTACHED. A table with no attached context never
// constructs one, so the default configuration -- which attaches nothing -- gains no memory,
// no lock and no branch beyond a null test on its set path.
class NNCacheAttributionRecorder {
 public:
  // 2^18 rows, 24 bytes each: about 6.3 MB.
  //
  // Denominated in the resource that actually runs out -- rows, one per distinct key a
  // session EARNS -- and sized from the corpus rather than from a round number: the largest
  // real card in the operator's corpus holds 291,129 keys, and a session cannot earn more
  // new entries than that without having earned an entire largest-card's worth of positions
  // it did not already have. Overflow past it is reported, not absorbed.
  static int defaultPowerOfTwo();

  NNCacheAttributionRecorder(int powerOfTwo, int mutexPoolSizePowerOfTwo);
  ~NNCacheAttributionRecorder();
  NNCacheAttributionRecorder(const NNCacheAttributionRecorder&) = delete;
  NNCacheAttributionRecorder& operator=(const NNCacheAttributionRecorder&) = delete;

  // Records that `key` was earned by `attribution`. Thread-safe; on the set path.
  void record(Hash128 key, const NNCacheAttribution& attribution);

  // The rows, with each id resolved to its name against the set that minted it. Thread-safe
  // and O(rows): a reporting call taken between sessions, never inside a search.
  [[nodiscard]] std::vector<NNCacheAttributionRow> harvest(const NNCacheContextSet& contexts) const;

  // Exactly the keys attributed to `context`. This is what a per-context dump writes, so it
  // is a first-class query rather than a filter every caller re-derives.
  [[nodiscard]] std::vector<Hash128> keysFor(const NNCacheContextId& context) const;

  [[nodiscard]] int64_t noAttributableContextEntries() const;
  [[nodiscard]] int64_t unrecordedAttributions() const;
  [[nodiscard]] size_t structureBytes() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

#endif  // NEURALNET_NNCACHECONTEXT_H_
