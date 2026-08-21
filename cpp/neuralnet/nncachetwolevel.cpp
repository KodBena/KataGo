#include "../neuralnet/nncachetwolevel.h"

#include <atomic>
#include <map>
#include <mutex>
#include <string>

#include "../core/global.h"
#include "../search/mutexpool.h"

// The two-level resolution strategy: an ORDERED LIST of frozen level-0 sources over one
// ordinary level 1, presented as one NNCacheTable with one hit-count surface across all of
// them.
//
// See nncachetwolevel.h for the semantics this file implements. The two things worth
// restating here, because they are what the file is shaped around:
//
//   LEVEL 0 IS FROZEN, so the only way a key can move is from a level-0 source to level 1,
//   once, irreversibly, and it takes its hit count with it. There is no path by which a
//   key comes to be resolvable in both.
//
//   THE LIST IS ORDERED AND THE ORDER IS THE PRIORITY. A get returns the first source that
//   resolves the key. A set, symmetrically, shadows the key in EVERY source that holds it
//   -- because "at most one level owns any key" is a statement about the whole list, and a
//   lower-priority source left resolving a superseded key would answer before level 1 ever
//   got the chance to.

//-------------------------------------------------------------------------------------
// The ordered resolution list
//-------------------------------------------------------------------------------------

namespace {
// Distinct per constructed list, so an id minted by one cache's list is refused by
// another's rather than silently addressing the source at the same serial.
std::atomic<uint64_t> nextLevelZeroListId(1);
}  // namespace

NNCacheLevelZeroSources::NNCacheLevelZeroSources()
  :listId_(nextLevelZeroListId.fetch_add(1, std::memory_order_relaxed)),
   nextSerial_(0),
   entries_()
{}

NNCacheLevelZeroSources::~NNCacheLevelZeroSources() = default;

NNCacheLevelOneOwner::NNCacheLevelOneOwner() {}
NNCacheLevelOneOwner::~NNCacheLevelOneOwner() {}

NNCacheLevelZeroAttachment NNCacheLevelZeroSources::attach(
  std::unique_ptr<NNCacheFrozen> source,
  const std::optional<NNCacheContextId>& servesContext,
  NNCacheLevelOneOwner& levelOne
) {
  if(source == nullptr)
    throw StringError(
      "NNCacheLevelZeroSources::attach: no source was supplied. A list with nothing "
      "attached is a list of length zero, never a list holding a null."
    );

  // THE RECONCILE, BEFORE THE SOURCE IS REACHABLE. It runs while the source is still this
  // function's own -- not yet on the list -- so there is no instant at which a get could resolve
  // a key level 1 owns out of it. The ordering is the guarantee, not an optimization: a
  // reconcile after the push would leave exactly that window open, and the whole point of doing
  // this at attach rather than at the next get is that no window exists at all.
  //
  // AND IT IS ALL-OR-NOTHING, WHICH IS WHY IT IS TWO PHASES AND NOT ONE LOOP. The walk asks
  // level 1 about every arriving key, and asking is the only step here that can fail: it is a
  // virtual call onto whatever table shape the operator configured. A single loop that asked and
  // shadowed in the same pass would, on a throw at entry k, leave the first k keys shadowed and
  // the rest not -- the operation whose whole purpose is restoring "at most one level owns any
  // key" leaving the invariant HALF-restored, with the arriving source's evaluations retired for
  // an attach that never happened and the retrievals they carried already moved into level 1's
  // ledger, where nothing records that they came from a source that was thrown away.
  //
  // So: SURVEY, which may throw and mutates nothing, then COMMIT, which mutates and cannot
  // throw. The plan the survey returns is a value -- the arriving keys level 1 owns -- and until
  // that value exists in full, not one entry has been touched. This is the "do not mutate until
  // the walk has succeeded" answer rather than the undo-path answer, and deliberately: an undo
  // would have to un-shadow entries and claw hit counts back out of level 1's ledger, and both
  // of those steps can themselves fail, which leaves the atomicity guarantee resting on a
  // recovery path that is never exercised (ADR-0012 P5 -- the band-aid stacked on the band-aid).
  // Not mutating has no recovery path to get wrong.
  //
  // ASKED OF EVERY ENTRY, in index order, through contains() -- the surface that answers
  // membership without recording a retrieval. What level 1 OWNS is what level 1 HOLDS: an entry
  // it can still serve is one that would be served after this source, wrongly, if this source
  // kept resolving the key. A key whose level-1 entry a capacity sweep has already dropped is
  // deliberately NOT shadowed here -- level 1 cannot serve it, this source can, and shadowing it
  // would throw away pre-warmed content to no one's benefit. That leaves such a key able to
  // appear on both halves of the composed count surface, which is closed at the other seam, by
  // the fold in NNCacheTableTwoLevel::harvestHitCounts (ADR-0000's closure statement: the axis
  // this reconcile does not cover is named, and covered elsewhere, rather than left silent).
  std::vector<Hash128> ownedByLevelOne;
  const uint32_t numEntries = source->index().numEntries();
  for(uint32_t i = 0; i < numEntries; i++) {
    const Hash128 key = source->index().keyAt(i);
    if(levelOne.ownsKeyForResolution(key))
      ownedByLevelOne.push_back(key);
  }

  // THE LAST ALLOCATION THE COMMIT COULD HAVE NEEDED, made here where a throw is still free.
  // With the slot reserved, the push_back below moves a nothrow-movable Entry into space that
  // already exists, so the commit phase has nothing left in it that can fail.
  entries_.reserve(entries_.size() + 1);

  Entry entry;
  entry.serial = nextSerial_;
  entry.servesContext = servesContext;
  entry.source = std::move(source);

  return commitAttach(std::move(entry), ownedByLevelOne, levelOne);
}

// THE COMMIT, AND IT IS noexcept BY DECLARATION RATHER THAN BY COMMENT. Every step below is one
// the survey has already made infallible: shadow() is an atomic exchange over an index probe,
// absorbTransferredHits() is a bounded probe under a lock, and the push_back moves into capacity
// reserved by the caller. Marking the function noexcept is what makes that claim CHECKED rather
// than asserted -- if any of those three ever grows a throw, this terminates at the point of the
// violation instead of silently resurrecting the half-applied state the two phases exist to
// foreclose. That is the sanctioned disposition for a genuine invariant violation, which is what
// a throw from here would be (ADR-0012 P9 rule 5), and it is louder than any recovery.
NNCacheLevelZeroAttachment NNCacheLevelZeroSources::commitAttach(
  Entry entry,
  const std::vector<Hash128>& ownedByLevelOne,
  NNCacheLevelOneOwner& levelOne
) noexcept {
  NNCacheLevelZeroAttachment attachment{NNCacheLevelZeroSourceId(listId_, nextSerial_), 0, 0};
  for(size_t i = 0; i < ownedByLevelOne.size(); i++) {
    const Hash128 key = ownedByLevelOne[i];
    const std::optional<uint32_t> transferred = entry.source->shadow(key);
    // nullopt only if this source's own entry was already shadowed, which a freshly loaded
    // source's is not and a re-attached one's may well be; either way there is nothing to move.
    if(!transferred.has_value())
      continue;
    attachment.entriesLevelOneAlreadyOwned += 1;
    if(transferred.value() > 0) {
      // The retrievals this entry accrued and never wrote. They are real, and level 1's counter
      // is the one that owns the key now, so they go there rather than being dropped with the
      // entry they were shadowed out of.
      levelOne.absorbTransferredHits(key, transferred.value());
      attachment.hitsTransferredToLevelOne += (int64_t)transferred.value();
    }
  }

  nextSerial_ += 1;
  entries_.push_back(std::move(entry));
  // The end of the vector IS the end of the resolution order. Appending is what makes
  // "attach the strongest first" the whole of the priority mechanism.
  return attachment;
}

std::unique_ptr<NNCacheFrozen> NNCacheLevelZeroSources::detach(const NNCacheLevelZeroSourceId& id) {
  if(id.listId() != listId_)
    throw StringError(
      "NNCacheLevelZeroSources::detach: this id names a source attached to a different "
      "cache. Spending it here would detach whichever source sits at the same serial in "
      "this list, which is a different card."
    );
  for(size_t i = 0; i < entries_.size(); i++) {
    if(entries_[i].serial != id.serial())
      continue;
    std::unique_ptr<NNCacheFrozen> detached = std::move(entries_[i].source);
    // erase, not swap-and-pop: the RELATIVE ORDER of everything still attached is the
    // priority the client asked for, and a detach is not a reordering.
    entries_.erase(entries_.begin() + (ptrdiff_t)i);
    return detached;
  }
  throw StringError(
    "NNCacheLevelZeroSources::detach: this source is not attached; serial " +
    Global::uint64ToString(id.serial()) + " has already been detached, and a serial is "
    "never reissued, so no other source can ever answer to this id."
  );
}

size_t NNCacheLevelZeroSources::size() const { return entries_.size(); }

std::vector<NNCacheLevelZeroSourceId> NNCacheLevelZeroSources::resolutionOrder() const {
  std::vector<NNCacheLevelZeroSourceId> out;
  out.reserve(entries_.size());
  for(size_t i = 0; i < entries_.size(); i++)
    out.push_back(NNCacheLevelZeroSourceId(listId_, entries_[i].serial));
  return out;
}

bool NNCacheLevelZeroSources::get(Hash128 key, std::shared_ptr<NNOutput>& ret) {
  // FIRST MATCH IN ATTACH ORDER WINS. The loop stops at the first source that resolves the
  // key, so exactly one source's counter is touched and every later source's entry for the
  // same key stays untouched and unreached.
  for(size_t i = 0; i < entries_.size(); i++) {
    if(entries_[i].source->get(key, ret))
      return true;
  }
  return false;
}

bool NNCacheLevelZeroSources::containsUnshadowed(Hash128 key) const {
  for(size_t i = 0; i < entries_.size(); i++) {
    if(entries_[i].source->contains(key))
      return true;
  }
  return false;
}

uint64_t NNCacheLevelZeroSources::shadowAllHolders(Hash128 key) {
  // EVERY holder, and there is no early exit. See the header: stopping at the first would
  // leave a lower-priority source resolving a key level 1 now owns.
  uint64_t transferred = 0;
  for(size_t i = 0; i < entries_.size(); i++) {
    const std::optional<uint32_t> fromThis = entries_[i].source->shadow(key);
    if(fromThis.has_value())
      transferred += (uint64_t)fromThis.value();
  }
  return transferred;
}

namespace {
// HOW PER-SOURCE ROWS BECOME ONE ROW PER KEY, in the one place both aggregating surfaces
// read it from. Two callers folding rows by two rules is the divergence this file already
// paid for once, and the rule is one fact, so it has one home (ADR-0012 P1).
//
// The row's POSITION is set by the first source in resolution order to say anything about
// the key; a later holder's count is SUMMED into it. Summing double-counts nothing: a get
// increments exactly one source's counter, every counter starts at zero when its source is
// built or loaded, and a set takes the key out of level 0 entirely (shadowAllHolders), so
// two holders' counts for one key are disjoint sets of real retrievals.
//
// The one refusal: a sum that will not fit the 32-bit row a count log record carries. That
// is a limit of the OUTPUT TYPE, not an invariant a legitimate state could violate, and it
// is refused rather than wrapped -- a wrapped count is four billion retrievals lost with
// nothing to show for it (ADR-0002). `consequence` is what the caller must know about what
// has already happened when it fires.
void foldByKey(
  const NNCacheHitCount& row,
  std::map<Hash128,size_t>& rowFor,
  std::vector<NNCacheHitCount>& out,
  const char* caller,
  const char* consequence
) {
  const std::map<Hash128,size_t>::const_iterator found = rowFor.find(row.key);
  if(found == rowFor.end()) {
    rowFor[row.key] = out.size();
    out.push_back(row);
    return;
  }
  NNCacheHitCount& into = out[found->second];
  const uint64_t summed = (uint64_t)into.hits + (uint64_t)row.hits;
  if(summed > 0xFFFFFFFFull)
    throw StringError(
      std::string(caller) + ": the hit counts for key " + row.key.toString() + " sum to " +
      Global::uint64ToString(summed) + " across attached sources, which does not fit the "
      "32-bit row a count log dump writes. " + std::string(consequence)
    );
  into.hits = (uint32_t)summed;
}

// THE SECOND SEAM THE ONE-ROW-PER-KEY RULE HAS TO HOLD AT: level 1's rows folded into the
// list's, rather than appended after them.
//
// It is the same rule and the same function as the list's own fold -- a key already on the
// surface keeps its position and takes the sum; a key that is new is appended -- applied where
// the OTHER pair of owners meets. The list folds its sources against each other; this folds
// their result against the level-1 ledger, so "one row per key" is a property of what a caller
// actually receives and not of one of its two halves.
//
// WHAT MAKES THE TWO HALVES OVERLAP AT ALL, since a set shadows level 0 as it hands level 1 the
// key: a ledger row outlives the level-1 ENTRY, deliberately ("this key was hot this session" is
// the fact a persistence layer wants, and a capacity sweep does not make it untrue), while
// attach's reconcile shadows only what level 1 still HOLDS. A key evicted from level 1 and then
// carried in by a re-attached source therefore sits in both, honestly, with two disjoint sets of
// real retrievals -- which sum.
//
// COST, stated because it is not free: one std::map insert per row on a call that is already
// O(table) and is taken between sessions, never in a search. A dump with no level-1 rows at all
// pays nothing, which is the freshly-attached case; the multi-source list harvest above has
// always paid the same map for the same reason.
std::vector<NNCacheHitCount> foldLevelOneInto(
  std::vector<NNCacheHitCount> levelZeroRows,
  const std::vector<NNCacheHitCount>& levelOneRows,
  const char* caller,
  const char* consequence
) {
  if(levelOneRows.empty())
    return levelZeroRows;
  std::vector<NNCacheHitCount> out = std::move(levelZeroRows);
  std::map<Hash128,size_t> rowFor;
  for(size_t i = 0; i < out.size(); i++)
    rowFor.insert(std::make_pair(out[i].key, i));
  for(size_t i = 0; i < levelOneRows.size(); i++)
    foldByKey(levelOneRows[i], rowFor, out, caller, consequence);
  return out;
}
}  // namespace

std::vector<NNCacheHitCount> NNCacheLevelZeroSources::harvest() const {
  if(entries_.empty())
    return std::vector<NNCacheHitCount>();
  // ONE SOURCE CANNOT COLLIDE WITH ITSELF: a frozen index refuses duplicate keys at build
  // (SPEC.md 4.1), so the single-source harvest needs no cross-source bookkeeping and does
  // not pay for any. This is the shape this call had before the list existed and it is
  // byte-for-byte the same answer.
  if(entries_.size() == 1)
    return entries_[0].source->harvest();

  std::vector<NNCacheHitCount> out;
  // ONE ROW PER KEY, by folding every holder's count into the first holder's row, through
  // the one function that owns that rule (foldByKey above). The map is built and
  // thrown away inside this call: harvest is an O(table) between-sessions report and is never
  // on the get or set path, so the allocation costs nothing that matters, and the alternative
  // -- a permanent cross-source key index -- would be a second home for a fact the sources
  // already hold (ADR-0012 P1).
  std::map<Hash128,size_t> rowFor;
  for(size_t i = 0; i < entries_.size(); i++) {
    const std::vector<NNCacheHitCount> rows = entries_[i].source->harvest();
    for(size_t j = 0; j < rows.size(); j++)
      foldByKey(
        rows[j], rowFor, out,
        "NNCacheLevelZeroSources::harvest",
        "This call reports without taking, so nothing has been consumed and no mark has moved."
      );
  }
  return out;
}

int64_t NNCacheLevelZeroSources::numReachableEntries() const {
  int64_t total = 0;
  for(size_t i = 0; i < entries_.size(); i++)
    total += entries_[i].source->numReachableEntries();
  return total;
}

int64_t NNCacheLevelZeroSources::reachablePayloadBytes() const {
  int64_t total = 0;
  for(size_t i = 0; i < entries_.size(); i++)
    total += entries_[i].source->reachablePayloadBytes();
  return total;
}

int64_t NNCacheLevelZeroSources::shadowedPayloadBytes() const {
  int64_t total = 0;
  for(size_t i = 0; i < entries_.size(); i++)
    total += entries_[i].source->shadowedPayloadBytes();
  return total;
}

int64_t NNCacheLevelZeroSources::structureBytes() const {
  int64_t total = 0;
  for(size_t i = 0; i < entries_.size(); i++)
    total += (int64_t)entries_[i].source->structureBytes();
  return total;
}

int64_t NNCacheLevelZeroSources::numEntries() const {
  int64_t total = 0;
  for(size_t i = 0; i < entries_.size(); i++)
    total += (int64_t)entries_[i].source->numEntries();
  return total;
}

NNCacheTwoLevelTable::NNCacheTwoLevelTable() {}
NNCacheTwoLevelTable::~NNCacheTwoLevelTable() {}

namespace {

//-------------------------------------------------------------------------------------
// The level-1 hit ledger
//-------------------------------------------------------------------------------------

// Per-key hit counts for the keys level 1 owns.
//
// Level 1 is whatever shape the operator configured -- direct, probed or chained -- and
// none of those carries a per-entry counter. Giving them one would mean touching four
// table implementations and changing the default table's per-slot memory, which the
// byte-identical-default bar forbids. So the counts live here instead, in one structure
// the two-level table owns, and level 1's own code is untouched.
//
// IT IS EXACT, NOT A SKETCH. Every row holds the FULL 128-bit key beside its count, so a
// count is never attributed to the wrong key, and a key that cannot be given a row is
// REFUSED and counted in unrecorded_ rather than silently overwriting somebody else's row.
// A structure that forgets counts, or that quietly hands a count to a different key, is
// exactly what a persistence layer must not be handed.
//
// A ROW PERSISTS AFTER LEVEL 1 EVICTS ITS KEY, deliberately: "this key was hot this
// session" is the fact a persistence layer wants, and it does not stop being true when a
// capacity sweep drops the entry. The consequence is that the ledger's occupancy is
// bounded by the number of DISTINCT keys hit in a session rather than by level 1's
// capacity, so a long enough session can fill it -- at which point unrecorded_ rises and
// the harvest says by how much, rather than being silently short (ADR-0002).
class HitLedger {
 public:
  HitLedger(int powerOfTwo, int mutexPoolSizePowerOfTwo)
    :rows_(((size_t)1) << powerOfTwo),
     mask_((((uint64_t)1) << powerOfTwo) - 1),
     mutexPool_(((uint32_t)1) << mutexPoolSizePowerOfTwo),
     mutexMask_((((uint32_t)1) << mutexPoolSizePowerOfTwo) - 1),
     unrecorded_(0)
  {}

  // How many slots forward of home a key may be placed. A bounded window keeps the write
  // O(1) and keeps a nearly-full ledger from degrading into a scan; overflowing it is
  // reported, not absorbed.
  static const uint32_t PROBE_WINDOW = 16;

  void add(Hash128 key, uint32_t amount) {
    if(amount == 0)
      return;
    const uint64_t home = key.hash0 & mask_;
    std::mutex& mutex = mutexPool_.getMutex((uint32_t)home & mutexMask_);
    std::lock_guard<std::mutex> lock(mutex);
    for(uint32_t step = 0; step < PROBE_WINDOW; step++) {
      Row& row = rows_[(home + step) & mask_];
      // count == 0 marks a free row: a row is only ever written with a count of at least
      // one, so no occupied row can be mistaken for a free one.
      if(row.count == 0) {
        row.key = key;
        row.count = amount;
        return;
      }
      if(row.key == key) {
        const uint64_t sum = (uint64_t)row.count + (uint64_t)amount;
        row.count = sum > 0xFFFFFFFFull ? 0xFFFFFFFFu : (uint32_t)sum;
        return;
      }
    }
    unrecorded_.fetch_add((int64_t)amount, std::memory_order_relaxed);
  }

  // This key's count, or zero if it has no row. The zero is not a sentinel: a key with no row
  // was never hit, and "never hit" and "hit zero times" are one fact for a counter that only
  // ever counts up. A row is never written with a count below one, so the two cases cannot be
  // told apart by any observer either.
  uint32_t hitsFor(Hash128 key) const {
    const uint64_t home = key.hash0 & mask_;
    std::mutex& mutex = mutexPool_.getMutex((uint32_t)home & mutexMask_);
    std::lock_guard<std::mutex> lock(mutex);
    for(uint32_t step = 0; step < PROBE_WINDOW; step++) {
      const Row& row = rows_[(home + step) & mask_];
      if(row.count == 0)
        return 0;
      if(row.key == key)
        return row.count;
    }
    return 0;
  }

  std::vector<NNCacheHitCount> harvest() const {
    std::vector<NNCacheHitCount> out;
    for(size_t i = 0; i < rows_.size(); i++) {
      std::mutex& mutex = mutexPool_.getMutex((uint32_t)i & mutexMask_);
      std::lock_guard<std::mutex> lock(mutex);
      if(rows_[i].count == 0)
        continue;
      NNCacheHitCount row;
      row.key = rows_[i].key;
      row.hits = rows_[i].count;
      out.push_back(row);
    }
    return out;
  }

  int64_t unrecordedHits() const { return unrecorded_.load(std::memory_order_relaxed); }
  size_t structureBytes() const {
    return rows_.size() * sizeof(Row) + ((size_t)mutexMask_ + 1) * sizeof(std::mutex);
  }

 private:
  // 24 bytes: 16 of key, 4 of count, and 4 that alignment spends on this row whether they
  // hold anything or not -- twoLevelHitLedgerBytes counts them explicitly. They held a
  // PERSISTED MARK while a dump appended this ledger's retrievals to a count log; nothing
  // appends retrievals anywhere now (the count log's currency is observations, and its rows
  // come from nncacheobservations.h), so the mark has no reader and is gone. The row's size is
  // unchanged, which is what twoLevelHitLedgerBytes reads from the type rather than restating.
  struct Row {
    Hash128 key;
    uint32_t count;
    Row() : key(), count(0) {}
  };

  std::vector<Row> rows_;
  uint64_t mask_;
  mutable MutexPool mutexPool_;
  uint32_t mutexMask_;
  std::atomic<int64_t> unrecorded_;
};

//-------------------------------------------------------------------------------------
// The table
//-------------------------------------------------------------------------------------

// The table is its own level-1 ownership oracle, and privately so: it is the only object that
// holds both the arriving source and the level 1 the reconcile must ask about, and nothing
// outside this file has any business asking it those two questions in isolation.
class NNCacheTableTwoLevel final : public NNCacheTwoLevelTable, private NNCacheLevelOneOwner {
 public:
  NNCacheTableTwoLevel(
    std::unique_ptr<NNCacheFrozen> levelZero,
    std::unique_ptr<NNCacheTable> levelOne,
    int hitLedgerPowerOfTwo,
    int hitLedgerMutexPowerOfTwo
  )
    :levelZero_(),
     levelOne_(std::move(levelOne)),
     ledger_(hitLedgerPowerOfTwo, hitLedgerMutexPowerOfTwo)
  {
    // The factory has already refused a null, so this cannot throw; the first source is
    // attached through the same door every later one goes through -- reconcile included, against
    // a level 1 that is empty at this instant and therefore owns nothing -- so there is no
    // second way for a source to enter the list. It names no context because none can exist
    // yet: contexts are attached to this table, and this table is what is being built.
    (void)levelZero_.attach(std::move(levelZero), std::optional<NNCacheContextId>(), *this);
  }

  // ATTACH RECONCILES; see NNCacheLevelZeroSources::attach, which is where the walk lives and
  // where the two rejected alternatives are recorded. This method's own job is the two things
  // the list cannot do for itself: refuse a context that is not this table's, and be the oracle.
  //
  // MEASURED COST OF THE RECONCILE, on this box, at the corpus sizes the design is dimensioned
  // against: see runnncachetwolevelbench's attach arm. It is one NNCacheTable::contains per
  // entry of the arriving source, which for the default direct-mapped level 1 is a mask, a lock
  // and a 128-bit comparison.
  NNCacheLevelZeroAttachment attachLevelZero(
    NNCacheLevelZeroSwapPermit permit,
    std::unique_ptr<NNCacheFrozen> source,
    const std::optional<NNCacheContextId>& servesContext
  ) override {
    // The permit's whole work is done by the time control reaches here: it was spent at the call
    // site, by a caller that could name a mint. See NNCacheLevelZeroSwapPermit.
    (void)permit;
    if(servesContext.has_value() && !cacheContexts().owns(servesContext.value()))
      throw StringError(
        "NNCacheTwoLevelTable::attachLevelZero: this context is attached to a different cache. "
        "Spending it here would file this source's retrievals under whichever context sits at "
        "the same position in this table's own name space, which is a different card."
      );
    return levelZero_.attach(std::move(source), servesContext, *this);
  }
  std::unique_ptr<NNCacheFrozen> detachLevelZero(
    NNCacheLevelZeroSwapPermit permit,
    const NNCacheLevelZeroSourceId& id
  ) override {
    (void)permit;
    return levelZero_.detach(id);
  }
  size_t numLevelZeroSources() const override { return levelZero_.size(); }
  std::vector<NNCacheLevelZeroSourceId> levelZeroResolutionOrder() const override {
    return levelZero_.resolutionOrder();
  }

  bool get(Hash128 nnHash, std::shared_ptr<NNOutput>& ret) override {
    // The attached sources first, in attach order, first match winning. A hit counts
    // itself in the same 32-bit word the lookup's own entry read already brought into
    // cache, so counting costs the level-0 path nothing beyond the increment.
    if(levelZero_.get(nnHash, ret))
      return true;
    // Fall through. A level-1 hit is counted in the ledger; that is one extra random
    // access on the level-1 hit path, and it is the only place in this design where
    // counting costs a memory access it would not otherwise make.
    // getRaw, not levelOne_->get: levelOne_'s static type is NNCacheTable, the base, and the
    // raw get/set are protected -- see nncache.h's own comment on why this needs the static
    // forwarder rather than plain protected access.
    if(NNCacheTable::getRaw(*levelOne_, nnHash, ret)) {
      ledger_.add(nnHash, 1);
      return true;
    }
    return false;
  }

  void set(const std::shared_ptr<NNOutput>& p) override {
    // Level 0 is frozen, so every set is level 1's. If level 0 holds this key, retire its
    // entry FIRST and fold the count it accrued into the ledger, so the key and its count
    // change hands together and are never resolvable in two places at once.
    //
    // This is reached for real: a caller passing skipCache=true consults no level and sets
    // unconditionally, and NNEvaluator::evaluate's ownership-map fall-through deliberately
    // re-evaluates a hit that lacked a requested ownership map and then sets the fuller
    // result. Dropping those sets would leave level 0 answering with an entry the caller
    // has already rejected, forcing a full re-evaluation of that position on every
    // subsequent such query.
    // EVERY attached source, not the first that holds the key: see
    // NNCacheLevelZeroSources::shadowAllHolders. The counts they gave up are summed,
    // because they are all counts of retrievals of THIS key and level 1 is now the one
    // counter for it.
    const uint64_t transferred = levelZero_.shadowAllHolders(p->nnHash);
    if(transferred > 0)
      ledger_.add(p->nnHash, transferred > 0xFFFFFFFFull ? 0xFFFFFFFFu : (uint32_t)transferred);
    NNCacheTable::setRaw(*levelOne_, p);
  }

  // LEVEL 1 ONLY, AND WITHOUT COUNTING. Both halves are the point. Not counting is what
  // keeps a dump from appearing in the counts it is dumping (ADR-0009); skipping level 0 is
  // what keeps a dump from being handed an entry whose bytes are already in the file it is
  // appending to, which is the duplicate NNCacheEntryProvenance exists to foreclose.
  bool peek(Hash128 nnHash, std::shared_ptr<NNOutput>& ret) override {
    return NNCacheTable::getRaw(*levelOne_, nnHash, ret);
  }

  // Either level holds it, and nothing is touched in finding out -- NNCacheFrozen::contains
  // reads the entry's state word without incrementing it, and level 1's own contains is the
  // no-trace surface by construction.
  bool contains(Hash128 nnHash) const override {
    return levelZero_.containsUnshadowed(nnHash) || levelOne_->contains(nnHash);
  }

  // Level 1 only. Level 0 is the pre-warmed content this session was handed; it cannot be
  // rebuilt in process, and clearing it would discard the whole point of having it. The
  // hit counts survive too: clear() is not the end of a session.
  void clear() override {
    levelOne_->clear();
  }

  NNCacheStats stats() const override {
    // SUMMED OVER EVERY ATTACHED SOURCE, and an entry held by two sources is counted
    // twice on purpose: these are memory figures, and two sources holding one key really
    // do hold two evaluations and pay for both. What is unreachable is not thereby free.
    NNCacheStats s = levelOne_->stats();
    s.residentEntries += levelZero_.numReachableEntries();
    s.residentPayloadBytes += levelZero_.reachablePayloadBytes();
    // The sources' own structures, the ledger, and -- because it is real memory this table
    // holds and can no longer hand out -- the evaluations of shadowed entries.
    s.fixedStructureBytes +=
      levelZero_.structureBytes() +
      (int64_t)ledger_.structureBytes() +
      levelZero_.shadowedPayloadBytes();
    // A chained level 1 reports no slot capacity because it is bounded by bytes; adding
    // level 0's fixed count to a zero would fabricate a ratio, so the zero is preserved.
    if(s.capacitySlots != 0)
      s.capacitySlots += levelZero_.numEntries();
    return s;
  }

  //---- The level-1 ownership oracle the reconcile asks ----------------------------
  //
  // Private, and implemented right beside the table's own get/set, because the two answers ARE
  // that state: what level 1 holds, and where a transferred count goes.

  bool ownsKeyForResolution(Hash128 key) const override {
    // contains(), never get(): the reconcile asks this once per entry of an arriving source, and
    // a retrieval-shaped probe would move a whole card's worth of keys in level 1's replacement
    // order and count a sighting for each. See NNCacheTable::contains.
    return levelOne_->contains(key);
  }

  void absorbTransferredHits(Hash128 key, uint32_t hits) override {
    // The same ledger, through the same door, that set() folds a shadowed entry's count into.
    // A transfer at attach and a transfer at set are the same act -- level 1 taking ownership of
    // a key that level 0 was holding -- so they land in the same place by construction.
    ledger_.add(key, hits);
  }

  NNCacheHitLedger harvestHitCounts() const override {
    // The attached sources' unshadowed entries first -- each source in resolution order,
    // each source's entries in index order 0..n-1, which is the harvest order SPEC.md 3.4
    // requires and -- because the input arrived in descending reference count --
    // descending-popularity order within a source. Then level 1's keys. A key held by
    // several sources is emitted once, with their counts summed, by the list (see
    // NNCacheLevelZeroSources::harvest).
    //
    // AND LEVEL 1'S ROWS ARE FOLDED IN, NOT CONCATENATED. A set shadows the key in every
    // ATTACHED source at the moment its count arrives in the ledger, and an attach shadows what
    // level 1 already holds, so the two halves are disjoint in the ordinary case and the fold
    // costs a map lookup per row and changes nothing. The case it exists for is the one neither
    // of those covers: a ledger row OUTLIVES the level-1 entry a capacity sweep dropped, and a
    // source attached after that sweep holds the key legitimately unshadowed -- attach asks what
    // level 1 HOLDS, and by then it holds nothing. Concatenated, that key appears twice here and
    // a dump raises its sessions twice for one dump. Summed, it appears once carrying every
    // retrieval, which is what it is: two disjoint sets of real retrievals of one position, at
    // different times, kept in two places.
    std::vector<NNCacheHitCount> rows = levelZero_.harvest();
    const std::vector<NNCacheHitCount> levelOneRows = ledger_.harvest();
    return NNCacheHitLedger::counted(
      foldLevelOneInto(
        std::move(rows), levelOneRows,
        "NNCacheTableTwoLevel::harvestHitCounts",
        "This call reports without taking, so nothing has been consumed and no mark has moved."
      ),
      ledger_.unrecordedHits()
    );
  }

 private:
  NNCacheLevelZeroSources levelZero_;
  std::unique_ptr<NNCacheTable> levelOne_;
  mutable HitLedger ledger_;
};

}  // namespace

size_t twoLevelHitLedgerBytes(int hitLedgerPowerOfTwo) {
  // Kept in step with HitLedger::Row by construction rather than by a second copy of the
  // arithmetic (ADR-0012 P1). The mutex pool is not included: it is sized independently
  // and is reported through stats(), where the whole figure is what matters.
  // The second uint32_t is the alignment padding the row has always carried; it briefly held
  // a persisted mark and no longer does, and the figure is unchanged either way, which is the
  // point (ADR-0012 P1: the row type decides, this reads it).
  return (((size_t)1) << hitLedgerPowerOfTwo) * (sizeof(Hash128) + sizeof(uint32_t) + sizeof(uint32_t));
}

std::unique_ptr<NNCacheTwoLevelTable> makeTwoLevelNNCacheTable(
  std::unique_ptr<NNCacheFrozen> levelZero,
  std::unique_ptr<NNCacheTable> levelOne,
  int hitLedgerPowerOfTwo
) {
  // "No level 0" is represented by not building one of these, never by one of these
  // holding a null. Refusing here keeps that the only representation there is.
  if(levelZero == nullptr)
    throw StringError(
      "makeTwoLevelNNCacheTable: no level 0 was supplied. A configuration with no frozen "
      "level 0 is served by NNCacheTable::create alone; it is not this table holding a null."
    );
  if(levelOne == nullptr)
    throw StringError("makeTwoLevelNNCacheTable: no level 1 was supplied.");
  if(hitLedgerPowerOfTwo < 0 || hitLedgerPowerOfTwo > 40)
    throw StringError(
      "makeTwoLevelNNCacheTable: hitLedgerPowerOfTwo must be between 0 and 40; got " +
      Global::intToString(hitLedgerPowerOfTwo) + "."
    );
  // The ledger's mutex pool: enough to keep concurrent level-1 hits off one lock, never
  // more mutexes than rows.
  const int mutexPow = hitLedgerPowerOfTwo < 10 ? hitLedgerPowerOfTwo : 10;
  return std::unique_ptr<NNCacheTwoLevelTable>(
    new NNCacheTableTwoLevel(
      std::move(levelZero), std::move(levelOne), hitLedgerPowerOfTwo, mutexPow
    )
  );
}
