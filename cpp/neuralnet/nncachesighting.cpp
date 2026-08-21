#include "../neuralnet/nncacheimpl.h"

#include <atomic>
#include <vector>

#include "../search/mutexpool.h"

using namespace std;

// The direct-mapped table with a collision REPLACEMENT rule.
//
// ---------------------------------------------------------------------------
// What this axis is, and what it is not
// ---------------------------------------------------------------------------
// A 1-way direct-mapped collision presents two candidates: the resident INCUMBENT and the
// offered NEWCOMER. The eviction axis does not cover this and never did -- it answers
// "which of the `ways` residents is given up", and at one way there is nothing to choose
// among. But the OTHER half of the same collision is a real binary decision, and until
// this file existed it was made and unnameable: the table unconditionally took the
// newcomer. That is the same shape of defect the chained table's capacity order had
// before NNCacheShape::chained gained an eviction argument -- a policy in force under no
// name -- and it is repaired the same way, by making it a factory argument.
//
// The admission axis does not cover it either. NNCacheTableSecondSighting is a decorator:
// it decides whether an offer is stored at all, from the offered key alone, and never
// sees the resident entry. A rule comparing newcomer against incumbent cannot live there.
//
// ---------------------------------------------------------------------------
// Why this is a separate class rather than a branch in NNCacheTableDirect
// ---------------------------------------------------------------------------
// Because the default configuration must keep running the code it has always run. A
// branch inside NNCacheTableDirect would put a test for this axis on the hot path of
// every KataGo search that never asked for it, on the one path this whole programme is
// measuring (ADR-0009: the instrument stays outside the instrument). NNCacheTable::create
// reaches this file only when the rule is not Always, so when it is not asked for it is
// not there -- the same disposition the probed table, the chained table, the admission
// filter and the trace decorator all already use (ADR-0012 P3, one concern per file).
//
// ---------------------------------------------------------------------------
// Sightings, and the ghost table
// ---------------------------------------------------------------------------
// A key's SIGHTING COUNT is the number of times it has been PRESENTED to the table -- on
// every get and every set, hit or miss.
//
// Counting STORES instead would produce a policy that cannot fire. KataGo stores a
// position once, on the first miss, and re-stores it only on nneval's ownermap-upgrade
// path: the perfmatrix sweep measured 442,735 sets over 442,192 distinct keys, i.e. 0.12%
// of keys ever offered twice. Every count would be 1, every comparison a tie, and the
// mechanism would be a no-op wearing a policy's name. Counting presentations makes it the
// same quantity the operator's own persisted-cache prototype counts as `num_refs`.
//
// The counts are kept for NON-RESIDENT keys, in a fixed-size ghost array. Without that a
// newcomer's count is structurally zero -- it has no slot to have carried one -- and any
// comparison against a resident incumbent collapses straight back to "the newcomer always
// wins", which is the rule this axis exists to be an alternative to.
//
// THE BOUND, exact and fixed at construction:
//
//     4 bytes * 2^nnCacheSightingGhostPowerOfTwo
//
// which is 8 MiB at 2^21, 32 MiB at 2^23 and 128 MiB at 2^25 -- an INDEPENDENT budget
// line, not a rider on the table's size. Selecting BOTH second-sighting admission and a
// count-comparing replacement rule allocates BOTH ghosts; they are not shared, because
// they hold different facts and merging them would silently change second-sighting's
// semantics, which is a question the perfmatrix report put to the operator and which is
// his to answer.
//
// WHY THE GHOST HAS ITS OWN SIZE, AND WHY DERIVING IT FROM THE TABLE WAS A MEASUREMENT
// BUG AS WELL AS A DESIGN ONE. The ghost is a lossy sketch (below). Sized from
// nnCacheSizePowerOfTwo -- which is what it did in the first version of this file, and
// what it still does when the key is left unset -- its load factor IS the table's load
// factor. So in exactly the regime a replacement rule exists for, where the table cannot
// hold the working set, the ghost cannot hold the counts either.
//
// And the consequence is worse than a loss of precision: A SATURATED GHOST DOES NOT
// DEGRADE THESE RULES, IT REPLACES THEM WITH DEGENERATE ONES. Follow set() below.
// sight() records the newcomer's presentation and returns the count INCLUDING it, so the
// newcomer's own write has just landed in its ghost slot when the comparison is made --
// even if that slot belonged to some other key a moment ago, the newcomer reads 1. The
// incumbent's count is read by peek(), which does not write, so an incumbent whose slot
// was taken by another key reads 0. Under saturation the comparison is therefore 1
// against 0, almost every time, for almost every contest:
//
//   KeepMoreSeen  admits iff newcomer >= incumbent  ->  1 >= 0  ->  ADMIT EVERYTHING.
//                 It stops choosing and becomes `always`.
//   KeepLessSeen  admits iff newcomer <= incumbent  ->  1 <= 0  ->  REFUSE EVERYTHING.
//
// So a sweep taken through a table-sized ghost measures neither policy. It measures
// `always` under one name and refuse-everything under the other, and reports the pair as
// if they were the two arms of a comparison. That is not hypothetical: it happened once
// on this branch, before this key existed, and the numbers it produced inverted when the
// ghost was given a size of its own.
//
// HISTORY, because the next reader should know the question was tested rather than
// assumed: an earlier version of this comment gave a DIFFERENT mechanism -- that a
// clobbered incumbent reading 0 makes it win under KeepLessSeen and lose under
// KeepMoreSeen, i.e. an asymmetric handicap. That is wrong in sign, and it is wrong
// because it reasons about the incumbent's read and forgets that the newcomer's own
// sighting rewrites its slot first. The re-measurement refuted it; the conclusion it was
// offered in support of -- that the ghost needs a size of its own -- is unchanged and is
// better supported by the mechanism above.
//
// The ghost's natural size is the WORKING SET -- an absolute quantity, and not a function
// of how large a table someone chose.
//
// Each ghost word packs a 24-bit tag and an 8-bit saturating count. The slot is indexed by
// hash1 and the tag is drawn from the top 24 bits of hash0, so neither quantity collides
// with the low bits of hash0 the table itself indexes on. Two deliberate imprecisions,
// both hit-rate effects and neither a correctness effect, stated rather than left implicit:
//
//   * IT IS A LOSSY SKETCH, not a map. Two keys landing on one ghost slot overwrite each
//     other's counts, and a key whose count was overwritten reads as unseen. The fraction
//     of keys clobbered at least once over a run of N distinct keys into M ghost slots is
//     about 1 - exp(-(N-1)/M), so M >= 16N keeps it near 6% and M >= 64N near 1.6%.
//     The exact structure (a count-min sketch, i.e. W-TinyLFU's admission half) is the
//     named next thing to try if these rules measure well enough to be worth sharpening;
//     it is a much larger structure and building it before any measurement said the cheap
//     one is insufficient is the order this branch has declined twice already.
//   * RELAXED ATOMICS, NO LOCK, exactly as the second-sighting ghost. Two threads racing
//     on one word can lose an update, which costs at most one wrong replacement decision.
//     A 32-bit aligned relaxed load/store cannot tear, so no interleaving produces
//     anything but a stale or a fresh word.
//
// The count saturates at 255. Two very hot keys therefore tie, and a tie is resolved in
// the newcomer's favour (below), so a saturated pair behaves like `always`. At the
// measured order of a thousand cache operations per second that is reachable only for keys
// presented hundreds of times, which is exactly the regime where both are "hot" and the
// distinction has stopped carrying information.

namespace {

//-------------------------------------------------------------------------------------
// The ghost sighting-count array
//-------------------------------------------------------------------------------------

static const uint32_t GHOST_COUNT_MAX = 255u;
static const uint32_t GHOST_COUNT_MASK = 0xFFu;

// Forced nonzero so that a never-written word (0) stays distinguishable from a real key
// whose tag happens to be zero. That costs one bit: a 2^-23 chance that a different key
// reads as this one and hands back its count -- a wrong count, never a wrong entry, since
// the table itself still compares the full 128-bit hash before returning anything.
static uint32_t ghostTagOf(Hash128 hash) {
  return ((uint32_t)(hash.hash0 >> 40) & 0x00FFFFFFu) | 1u;
}

class GhostSightingCounts {
  std::atomic<uint32_t>* words;
  uint64_t mask;

 public:
  explicit GhostSightingCounts(int sizePowerOfTwo)
    :words(NULL), mask(0)
  {
    if(sizePowerOfTwo < 0 || sizePowerOfTwo > 63)
      throw StringError("NNCacheTable: Invalid sizePowerOfTwo: " + Global::intToString(sizePowerOfTwo));
    const uint64_t numSlots = ((uint64_t)1) << sizePowerOfTwo;
    mask = numSlots - 1;
    words = new std::atomic<uint32_t>[numSlots]();
  }
  ~GhostSightingCounts() { delete[] words; }

  GhostSightingCounts(const GhostSightingCounts&) = delete;
  GhostSightingCounts& operator=(const GhostSightingCounts&) = delete;

  // Records one presentation of `hash` and returns the key's count INCLUDING it.
  uint32_t sight(Hash128 hash) {
    const uint64_t idx = hash.hash1 & mask;
    const uint32_t tag = ghostTagOf(hash);
    const uint32_t word = words[idx].load(std::memory_order_relaxed);
    // A slot holding a different key (or never written) starts this key from zero rather
    // than inheriting a stranger's count.
    uint32_t count = (word >> 8) == tag ? (word & GHOST_COUNT_MASK) : 0u;
    if(count < GHOST_COUNT_MAX)
      count += 1;
    words[idx].store((tag << 8) | count, std::memory_order_relaxed);
    return count;
  }

  // Reads `hash`'s count without recording a presentation. 0 means "no count on record",
  // which is not the same claim as "never presented" -- the slot may have been taken by
  // another key. Stated here because the difference is exactly the sketch's imprecision.
  uint32_t peek(Hash128 hash) const {
    const uint64_t idx = hash.hash1 & mask;
    const uint32_t word = words[idx].load(std::memory_order_relaxed);
    return (word >> 8) == ghostTagOf(hash) ? (word & GHOST_COUNT_MASK) : 0u;
  }

  void clear() {
    for(uint64_t i = 0; i <= mask; i++)
      words[i].store(0u, std::memory_order_relaxed);
  }

  int64_t bytes() const { return (int64_t)(sizeof(std::atomic<uint32_t>) * (mask + 1)); }
};

//-------------------------------------------------------------------------------------
// The rules
//-------------------------------------------------------------------------------------
// A rule owns two things as TYPES: the per-slot metadata it needs (Meta, which the slot
// INHERITS, so a rule needing none costs zero bytes by empty-base optimization -- C++17
// has no [[no_unique_address]]) and whether it needs the ghost array at all. It never sees
// the table.
//
// admitNewcomer(incumbentCount, newcomerCount, incumbentMeta) answers the one question
// the axis exists to ask: does the offered key take the slot? Returning false means the
// incumbent stays and the offered payload is dropped -- freed after the lock is released,
// like every other payload this table gives up.
//
// EVERY rule below gives a TIE to the newcomer. That is load-bearing rather than
// arbitrary: on a stream with no reuse, every key is presented the same number of times
// before it contends, so every comparison is a tie, and a tie going to the newcomer makes
// such a stream behave EXACTLY as `always` does. The policies then deviate only where
// there is real sighting-count information to deviate on, which is the property
// testSightingRulesMatchAlwaysWhenNoKeyIsEverReseen asserts.

class KeepLessSeenRule {
 public:
  static const char* name() { return "keeplessseen"; }
  struct Meta {};
  static const bool NEEDS_GHOST = true;
  static void onHit(Meta&) {}
  static void onInsert(Meta&) {}
  // The INVERSE arm: the candidate seen MORE often is the one replaced, so the SURVIVOR
  // is the less-seen one. Carried so the mechanism can be measured in both directions --
  // it is one comparison flip from KeepMoreSeenRule. Not a recommendation and not
  // anybody's proposal; the operator's proposal is KeepMoreSeenRule below.
  static bool admitNewcomer(uint32_t incumbentCount, uint32_t newcomerCount, Meta&) {
    return newcomerCount <= incumbentCount;
  }
};

class KeepMoreSeenRule {
 public:
  static const char* name() { return "keepmoreseen"; }
  struct Meta {};
  static const bool NEEDS_GHOST = true;
  static void onHit(Meta&) {}
  static void onInsert(Meta&) {}
  // THE OPERATOR'S OWN PROPOSAL, "most-seen direct": the more-seen candidate survives.
  // Also the conventional, LFU-shaped direction.
  static bool admitNewcomer(uint32_t incumbentCount, uint32_t newcomerCount, Meta&) {
    return newcomerCount >= incumbentCount;
  }
};

class KeepSightedRule {
 public:
  static const char* name() { return "keepsighted"; }
  // One bool, in the cache line the lookup already touched. No ghost array and no second
  // memory access per operation -- but not cheaper in BYTES: alignment beside a 16-byte
  // shared_ptr inflates that bool to 8 bytes per slot, against the counting rules' 4, so
  // this rule costs 16 MiB at k=21 where they cost 8. WITNESSED by the test suite, which
  // prints both figures from the tables' own stats(). Cheaper in accesses, dearer in
  // bytes; which trade wins is a measurement, not an argument.
  struct Meta { bool sighted; };
  static const bool NEEDS_GHOST = false;
  static void onHit(Meta& m) { m.sighted = true; }
  static void onInsert(Meta& m) { m.sighted = false; }
  // Second chance, and exactly one: an incumbent that has been re-read since it was stored
  // survives this collision and gives up its reprieve. Without the clearing, a single hit
  // would make an entry immortal -- the naive-LFU pollution failure that LFUDA exists to
  // avoid on the probed table, in this register.
  static bool admitNewcomer(uint32_t, uint32_t, Meta& m) {
    if(m.sighted) {
      m.sighted = false;
      return false;
    }
    return true;
  }
};

//-------------------------------------------------------------------------------------
// The table
//-------------------------------------------------------------------------------------

template<class Rule>
class NNCacheTableDirectSighting final : public NNCacheTable {
  struct Entry : Rule::Meta {
    std::shared_ptr<NNOutput> ptr;
  };

  Entry* entries;
  GhostSightingCounts* ghost;   // NULL exactly when Rule::NEEDS_GHOST is false
  MutexPool* mutexPool;
  uint64_t tableSize;
  uint64_t tableMask;
  uint32_t mutexPoolMask;

 public:
  // ghostPowerOfTwo is EXPLICIT and is not assumed equal to sizePowerOfTwo; see the
  // load-factor argument in this file's header.
  NNCacheTableDirectSighting(int sizePowerOfTwo, int mutexPoolSizePowerOfTwo, int ghostPowerOfTwo)
    :entries(NULL), ghost(NULL), mutexPool(NULL), tableSize(0), tableMask(0), mutexPoolMask(0)
  {
    if(sizePowerOfTwo < 0 || sizePowerOfTwo > 63)
      throw StringError("NNCacheTable: Invalid sizePowerOfTwo: " + Global::intToString(sizePowerOfTwo));
    if(mutexPoolSizePowerOfTwo < 0 || mutexPoolSizePowerOfTwo > 31)
      throw StringError("NNCacheTable: Invalid mutexPoolSizePowerOfTwo: " + Global::intToString(mutexPoolSizePowerOfTwo));
    if(mutexPoolSizePowerOfTwo > sizePowerOfTwo)
      mutexPoolSizePowerOfTwo = sizePowerOfTwo;

    tableSize = ((uint64_t)1) << sizePowerOfTwo;
    tableMask = tableSize - 1;
    // () is load-bearing: it value-initializes, so the rule's Meta starts at a defined
    // state rather than whatever the allocator left behind.
    entries = new Entry[tableSize]();
    const uint32_t mutexPoolSize = ((uint32_t)1) << mutexPoolSizePowerOfTwo;
    mutexPoolMask = mutexPoolSize - 1;
    mutexPool = new MutexPool(mutexPoolSize);
    if(Rule::NEEDS_GHOST)
      ghost = new GhostSightingCounts(ghostPowerOfTwo);
  }

  ~NNCacheTableDirectSighting() override {
    delete[] entries;
    delete ghost;
    delete mutexPool;
  }

  // NOT A SIGHTING, AND THAT IS THE WHOLE REASON THIS METHOD EXISTS. A get on this shape counts
  // a sighting for EVERY key it is handed, present or absent, because the ghost is what the
  // replacement rule reads -- so a get-shaped ownership probe over an arriving level-0 source
  // would write a whole card's worth of sightings nobody ever made and change which of two
  // candidates keeps a slot from then on. This asks the membership question and writes nothing:
  // no ghost sighting, no Rule::onHit. See NNCacheTable::contains.
  bool contains(Hash128 nnHash) const override {
    const uint64_t idx = nnHash.hash0 & tableMask;
    const Entry& entry = entries[idx];
    std::lock_guard<std::mutex> lock(mutexPool->getMutex((uint32_t)idx & mutexPoolMask));
    return entry.ptr != nullptr && entry.ptr->nnHash == nnHash;
  }

  // PROTECTED, matching the base: this class has no header today, so nothing outside this
  // translation unit can name the concrete type and reach these publicly regardless -- but a
  // header could be added later without anyone thinking to re-check access, exactly the defect
  // an out-of-frame audit found in NNCacheTableProbed (which DOES have a header). Fixed here in
  // the same pass, defense in depth.
 protected:
  // A get IS a sighting -- that is the whole point of the axis, and it is what makes the
  // count something other than a constant. The ghost update is deliberately OUTSIDE the
  // region lock: it is a lock-free hint, and putting it under the lock would lengthen the
  // one critical section this table has for no guarantee the structure needs.
  bool get(Hash128 nnHash, std::shared_ptr<NNOutput>& ret) override {
    // Free ret BEFORE locking, to avoid any expensive operations while locked.
    if(ret != nullptr)
      ret.reset();

    if(Rule::NEEDS_GHOST)
      (void)ghost->sight(nnHash);

    const uint64_t idx = nnHash.hash0 & tableMask;
    Entry& entry = entries[idx];
    std::lock_guard<std::mutex> lock(mutexPool->getMutex((uint32_t)idx & mutexPoolMask));
    if(entry.ptr != nullptr && entry.ptr->nnHash == nnHash) {
      Rule::onHit(static_cast<typename Rule::Meta&>(entry));
      ret = entry.ptr;
      return true;
    }
    return false;
  }

  void set(const std::shared_ptr<NNOutput>& p) override {
    // Immediately copy p right now, before locking, to avoid any expensive operations
    // while locked. Whatever this ends up displacing -- and, when the newcomer LOSES,
    // the newcomer itself -- is freed after the unlock.
    std::shared_ptr<NNOutput> buf(p);

    const Hash128 nnHash = p->nnHash;
    const uint32_t newcomerCount = Rule::NEEDS_GHOST ? ghost->sight(nnHash) : 0u;

    const uint64_t idx = nnHash.hash0 & tableMask;
    Entry& entry = entries[idx];
    {
      std::lock_guard<std::mutex> lock(mutexPool->getMutex((uint32_t)idx & mutexPoolMask));
      typename Rule::Meta& meta = static_cast<typename Rule::Meta&>(entry);
      if(entry.ptr == nullptr) {
        // An empty slot is not a collision and presents no choice.
        entry.ptr.swap(buf);
        Rule::onInsert(meta);
      }
      else if(entry.ptr->nnHash == nnHash) {
        // The same key offered again -- nneval's ownermap-upgrade path is exactly this.
        // It is a sighting of a live entry, not a contest between two keys, and it must
        // never be refused: refusing it would silently discard an upgrade the engine
        // computed. The probed table draws the same distinction for the same reason.
        entry.ptr.swap(buf);
        Rule::onHit(meta);
      }
      else {
        // A genuine collision. Recovering the incumbent's count costs one dereference of
        // its payload to read its hash; the direct-mapped slot carries no inline tag to
        // read it from. That cost is paid only on collisions, and only under this axis.
        const uint32_t incumbentCount = Rule::NEEDS_GHOST ? ghost->peek(entry.ptr->nnHash) : 0u;
        if(Rule::admitNewcomer(incumbentCount, newcomerCount, meta)) {
          entry.ptr.swap(buf);
          Rule::onInsert(meta);
        }
        // else: the incumbent stays and `buf` still holds the newcomer, which is freed
        // below, outside the lock, exactly as a displaced incumbent would have been.
      }
    }
  }

 public:
  void clear() override {
    std::shared_ptr<NNOutput> buf;
    for(uint64_t idx = 0; idx < tableSize; idx++) {
      Entry& entry = entries[idx];
      {
        std::lock_guard<std::mutex> lock(mutexPool->getMutex((uint32_t)idx & mutexPoolMask));
        entry.ptr.swap(buf);
        static_cast<typename Rule::Meta&>(entry) = typename Rule::Meta();
      }
      buf.reset();
    }
    // The counts go too. Keeping them would let the table make replacement decisions on
    // the strength of sightings of positions it no longer holds -- the same reason
    // second-sighting admission clears its own ghost on clear().
    if(Rule::NEEDS_GHOST)
      ghost->clear();
  }

  // A snapshot taken one lock region at a time; see NNCacheTableDirect::stats for why
  // there is no global lock and what that costs under live traffic.
  NNCacheStats stats() const override {
    NNCacheStats s = {0,0,0,(int64_t)tableSize};
    for(uint64_t idx = 0; idx < tableSize; idx++) {
      std::lock_guard<std::mutex> lock(mutexPool->getMutex((uint32_t)idx & mutexPoolMask));
      const Entry& entry = entries[idx];
      if(entry.ptr != nullptr) {
        s.residentEntries += 1;
        s.residentPayloadBytes += (int64_t)nnOutputFootprintBytes(*entry.ptr);
      }
    }
    // The ghost is real resident memory the operator pays for and belongs to no other
    // layer, so leaving it out would make this axis look free.
    s.fixedStructureBytes =
      (int64_t)(tableSize * sizeof(Entry)) +
      (int64_t)((uint64_t)mutexPoolMask + 1) * (int64_t)sizeof(std::mutex) +
      (ghost != NULL ? ghost->bytes() : (int64_t)0);
    return s;
  }
};

}  // namespace

size_t sightingCountGhostBytes(int sizePowerOfTwo) {
  if(sizePowerOfTwo < 0 || sizePowerOfTwo > 63)
    throw StringError("sightingCountGhostBytes: Invalid sizePowerOfTwo: " + Global::intToString(sizePowerOfTwo));
  return sizeof(std::atomic<uint32_t>) * (((size_t)1) << sizePowerOfTwo);
}

unique_ptr<NNCacheTable> makeSightingDirectNNCacheTable(const NNCacheConfig& config) {
  if(config.shape.scheme() != NNCacheCollisionScheme::Direct)
    throw StringError(
      "makeSightingDirectNNCacheTable: a collision replacement rule exists only under " +
      string(NNCacheConfig::KEY_COLLISION) + " = direct; got " + config.shape.toString() + ". "
      "NNCacheShape's factories already make that unconstructable, so reaching here means a caller "
      "bypassed the shape type -- refused rather than defaulted, so the bypass fails loudly."
    );
  switch(config.shape.replacement()) {
  case NNCacheReplacementPolicy::Always:
    throw StringError(
      "makeSightingDirectNNCacheTable: the 'always' rule is NNCacheTableDirect's, not this table's. "
      "Building it here would mean the DEFAULT configuration ran a generalised table configured to "
      "behave like the shipped one, instead of the shipped one -- refused so that cannot happen "
      "silently."
    );
  case NNCacheReplacementPolicy::KeepLessSeen:
    // value_or is the one place the "not stated" case is resolved, and it resolves to the
    // table's own size -- exactly what the ghost was before this key existed, so a config
    // that says nothing builds the structure it built before.
    return unique_ptr<NNCacheTable>(new NNCacheTableDirectSighting<KeepLessSeenRule>(
      config.sizePowerOfTwo, config.mutexPoolSizePowerOfTwo,
      config.shape.sightingGhostPowerOfTwo().value_or(config.sizePowerOfTwo)));
  case NNCacheReplacementPolicy::KeepMoreSeen:
    return unique_ptr<NNCacheTable>(new NNCacheTableDirectSighting<KeepMoreSeenRule>(
      config.sizePowerOfTwo, config.mutexPoolSizePowerOfTwo,
      config.shape.sightingGhostPowerOfTwo().value_or(config.sizePowerOfTwo)));
  case NNCacheReplacementPolicy::KeepSighted:
    // No ghost at all, so this argument is never read; the table's own size is passed
    // rather than a literal, so nothing here looks like a size that means something.
    return unique_ptr<NNCacheTable>(new NNCacheTableDirectSighting<KeepSightedRule>(
      config.sizePowerOfTwo, config.mutexPoolSizePowerOfTwo, config.sizePowerOfTwo));
  default:
    break;
  }
  // A value added to the enum and not to the switch above lands here rather than silently
  // getting one of the rules (ADR-0002).
  throw StringError("makeSightingDirectNNCacheTable: unhandled replacement policy");
}
