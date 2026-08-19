#include "../neuralnet/nncachelevelzero.h"

#include <algorithm>
#include <map>
#include <utility>

#include "../core/global.h"
#include "../core/timer.h"

#if defined(__GLIBC__) && !defined(__UCLIBC__)
#include <malloc.h>
#define NNCACHE_HAVE_MALLOC_TRIM 1
#endif

// The level-0 loader. See nncachelevelzero.h for the contract and for the memory posture,
// which is the reason this file exists rather than a convenience of it.

//-------------------------------------------------------------------------------------
// Giving the pages back
//-------------------------------------------------------------------------------------

NNCacheHeapReclaim nnCacheReclaimFreedHeap() {
#ifdef NNCACHE_HAVE_MALLOC_TRIM
  // The argument is the padding to keep at the top of the heap. Zero, because this is called
  // at a session boundary where nothing is about to be re-allocated and holding pages back
  // for a caller that is not coming would be the whole complaint in miniature.
  return malloc_trim(0) != 0 ? NNCacheHeapReclaim::Trimmed : NNCacheHeapReclaim::NothingToTrim;
#else
  // NOT A SILENT NO-OP. This build's allocator has no trim call, the caller is told so by
  // name, and a report that says "Unavailable" is a different statement from one that says
  // the allocator was asked and had nothing (ADR-0002).
  return NNCacheHeapReclaim::Unavailable;
#endif
}

//-------------------------------------------------------------------------------------
// The arena
//-------------------------------------------------------------------------------------

NNCacheLevelZeroArena::NNCacheLevelZeroArena(size_t numEvaluations, size_t numOwnerMapFloats)
  :outputs_(numEvaluations),
   ownerMaps_(numOwnerMapFloats, 0.0f),
   ownerMapFloatsUsed_(0)
{}

NNCacheLevelZeroArena::~NNCacheLevelZeroArena() {
  // THE ONE PLACE THE ARENA HAZARD IS ANSWERED. ~NNOutput calls delete[] whiteOwnerMap, and
  // every ownership map here points into ownerMaps_ -- an interior pointer the allocator
  // never issued. This runs BEFORE outputs_ is destroyed, because a destructor body runs
  // before its members', so no ~NNOutput ever sees an arena pointer. There is no other exit
  // from this object: it cannot be copied, it hands out no owning handle, and every entry it
  // ever filled is in this loop.
  for(size_t i = 0; i < outputs_.size(); i++)
    outputs_[i].whiteOwnerMap = NULL;
}

std::shared_ptr<NNCacheLevelZeroArena> NNCacheLevelZeroArena::reserve(
  size_t numEvaluations,
  size_t numOwnerMapFloats
) {
  // The two blocks are sized here and never again. A figure this build cannot allocate for
  // is refused by name rather than reaching the allocator as a wrapped or truncated size.
  const size_t maxOutputs = (size_t)1 << 31;
  const size_t maxFloats = (size_t)1 << 34;
  if(numEvaluations > maxOutputs)
    throw StringError(
      "NNCacheLevelZeroArena: " + Global::uint64ToString((uint64_t)numEvaluations) +
      " evaluations is past what this arena reserves in one block (max " +
      Global::uint64ToString((uint64_t)maxOutputs) + ")."
    );
  if(numOwnerMapFloats > maxFloats)
    throw StringError(
      "NNCacheLevelZeroArena: " + Global::uint64ToString((uint64_t)numOwnerMapFloats) +
      " ownership-map floats is past what this arena reserves in one block (max " +
      Global::uint64ToString((uint64_t)maxFloats) + ")."
    );
  return std::shared_ptr<NNCacheLevelZeroArena>(
    new NNCacheLevelZeroArena(numEvaluations, numOwnerMapFloats));
}

size_t NNCacheLevelZeroArena::numEvaluations() const {
  return outputs_.size();
}

NNOutput* NNCacheLevelZeroArena::evaluationAt(size_t i) const {
  // The store's contract is that it hands back the caller's own mutable evaluation; the
  // const on this method is about the STORE's bookkeeping, which is untouched, exactly as
  // the heap store's unique_ptr::get() const hands back a mutable NNOutput*.
  return const_cast<NNOutput*>(&outputs_[i]);
}

size_t NNCacheLevelZeroArena::handleBytes() const {
  // Zero, and deliberately so: evaluation i is at index i of a contiguous block, so there is
  // no per-entry pointer to keep. This is what puts an arena-backed level 0's structure at
  // the index's own footprint rather than eight bytes per entry above it.
  return 0;
}

NNOutput& NNCacheLevelZeroArena::outputFor(size_t i) {
  if(i >= outputs_.size())
    throw StringError(
      "NNCacheLevelZeroArena: entry " + Global::uint64ToString((uint64_t)i) +
      " was asked for and this arena reserved " + Global::uint64ToString((uint64_t)outputs_.size()) + "."
    );
  return outputs_[i];
}

float* NNCacheLevelZeroArena::ownerMapFor(size_t i, size_t numFloats) {
  (void)i;
  if(numFloats > ownerMaps_.size() - ownerMapFloatsUsed_)
    throw StringError(
      "NNCacheLevelZeroArena: an ownership map of " + Global::uint64ToString((uint64_t)numFloats) +
      " floats was asked for and " +
      Global::uint64ToString((uint64_t)(ownerMaps_.size() - ownerMapFloatsUsed_)) +
      " remain of the " + Global::uint64ToString((uint64_t)ownerMaps_.size()) + " reserved. "
      "The reservation is computed from the selected entries' own headers, so this means the "
      "container changed between the two reads."
    );
  float* out = ownerMaps_.data() + ownerMapFloatsUsed_;
  ownerMapFloatsUsed_ += numFloats;
  return out;
}

int64_t NNCacheLevelZeroArena::evaluationBlockBytes() const {
  return (int64_t)(outputs_.size() * sizeof(NNOutput));
}

int64_t NNCacheLevelZeroArena::ownerMapBlockBytes() const {
  return (int64_t)(ownerMaps_.size() * sizeof(float));
}

int64_t NNCacheLevelZeroArena::totalBytes() const {
  return evaluationBlockBytes() + ownerMapBlockBytes();
}

//-------------------------------------------------------------------------------------
// The order
//-------------------------------------------------------------------------------------

std::vector<NNCacheLevelZeroCandidate> nnCacheLevelZeroOrder(
  const std::vector<NNEvalContainerEntryLocation>& containerEntries,
  const std::vector<NNCacheCountRow>& countRows
) {
  std::map<std::pair<uint64_t,uint64_t>, uint64_t> lookupsOf;
  for(size_t i = 0; i < countRows.size(); i++)
    lookupsOf[std::make_pair(countRows[i].key.hash0, countRows[i].key.hash1)] = countRows[i].lookups;

  std::vector<NNCacheLevelZeroCandidate> out;
  out.reserve(containerEntries.size());
  for(size_t i = 0; i < containerEntries.size(); i++) {
    const NNEvalContainerEntryLocation& loc = containerEntries[i];
    NNCacheLevelZeroCandidate c;
    c.key = loc.key;
    c.containerIndex = i;
    const std::map<std::pair<uint64_t,uint64_t>, uint64_t>::const_iterator it =
      lookupsOf.find(std::make_pair(loc.key.hash0, loc.key.hash1));
    c.counted = it != lookupsOf.end();
    c.lookups = c.counted ? it->second : 0;
    // The RESIDENT cost, which is what the byte bound and the arena reservation are both
    // denominated in: the NNOutput itself plus its ownership map. It is deliberately not the
    // entry's size on disk, which omits the policy slots past the board and the struct
    // padding and is therefore a different number.
    c.residentBytes =
      (int64_t)sizeof(NNOutput) +
      (loc.hasOwnerMap ? (int64_t)loc.nnXLen * (int64_t)loc.nnYLen * (int64_t)sizeof(float) : 0);
    out.push_back(c);
  }

  std::stable_sort(out.begin(), out.end(),
    [](const NNCacheLevelZeroCandidate& x, const NNCacheLevelZeroCandidate& y) {
      // Descending lookups first.
      if(x.lookups != y.lookups)
        return x.lookups > y.lookups;
      // Then a key the log mentioned before a key it did not. This is the whole content of
      // "a container key absent from the count log orders as zero lookups, AFTER every
      // counted key": at equal lookups -- which for an uncounted key is always zero -- the
      // counted one wins, so a key counted at zero still outranks a key never seen.
      if(x.counted != y.counted)
        return x.counted;
      // Then by key, so the order is total and a test can assert it exactly rather than
      // asserting a set and hoping.
      return x.key < y.key;
    }
  );
  return out;
}

//-------------------------------------------------------------------------------------
// The bound
//-------------------------------------------------------------------------------------

NNCacheLevelZeroBound::NNCacheLevelZeroBound(Kind kind, uint64_t lookups, int64_t amount)
  :kind_(kind), lookups_(lookups), amount_(amount)
{}

NNCacheLevelZeroBound NNCacheLevelZeroBound::all() {
  return NNCacheLevelZeroBound(Kind::All, 0, 0);
}

NNCacheLevelZeroBound NNCacheLevelZeroBound::minLookups(uint64_t lookups) {
  return NNCacheLevelZeroBound(Kind::MinLookups, lookups, 0);
}

NNCacheLevelZeroBound NNCacheLevelZeroBound::maxEntries(int64_t entries) {
  if(entries < 0)
    throw StringError(
      "NNCacheLevelZeroBound: a maximum of " + Global::int64ToString(entries) + " entries has no reading."
    );
  return NNCacheLevelZeroBound(Kind::MaxEntries, 0, entries);
}

NNCacheLevelZeroBound NNCacheLevelZeroBound::maxBytes(int64_t bytes) {
  if(bytes < 0)
    throw StringError(
      "NNCacheLevelZeroBound: a maximum of " + Global::int64ToString(bytes) + " bytes has no reading."
    );
  return NNCacheLevelZeroBound(Kind::MaxBytes, 0, bytes);
}

size_t NNCacheLevelZeroBound::select(const std::vector<NNCacheLevelZeroCandidate>& ordered) const {
  switch(kind_) {
  case Kind::All:
    return ordered.size();
  case Kind::MaxEntries:
    return std::min((size_t)amount_, ordered.size());
  case Kind::MinLookups: {
    // A prefix, because the order is descending lookups: the first candidate below the
    // threshold is followed only by candidates at or below it.
    //
    // AN UNCOUNTED KEY IS ADMITTED ONLY BY A THRESHOLD OF ZERO, and there is NO SEPARATE
    // CHECK FOR IT because there is nothing for one to catch: an uncounted candidate carries
    // a lookups of 0, so any threshold above zero already excludes it by the comparison
    // below. A guard here would read as though it decided something and would in fact be
    // unreachable -- which is worse than its absence, because a later reader would trust it
    // (ADR-0013 Rule 4: the honest disposition of a check that catches nothing is to remove
    // it and say why, not to keep it as reassurance).
    //
    // THE COMPARISON ITSELF IS NOT WRITTEN HERE. It is NNCacheLookupThreshold::admits, the
    // one home of "this key has been seen often enough", shared with the write side's
    // NNCacheDiskAdmission so the two cannot drift on exactly the boundary case the
    // paragraph above turns on (ADR-0012 P1).
    const NNCacheLookupThreshold threshold = NNCacheLookupThreshold::of(lookups_);
    size_t taken = 0;
    while(taken < ordered.size() && threshold.admits(ordered[taken].lookups))
      taken += 1;
    return taken;
  }
  case Kind::MaxBytes: {
    int64_t used = 0;
    size_t taken = 0;
    while(taken < ordered.size()) {
      if(used + ordered[taken].residentBytes > amount_)
        break;
      used += ordered[taken].residentBytes;
      taken += 1;
    }
    return taken;
  }
  default:
    break;
  }
  // Every kind the enum names is above, so reaching here means a kind was added to the enum
  // and not to this switch. Refuse rather than silently take everything or nothing -- the
  // house idiom for a closed-vocabulary switch in this cache (nncachechained.cpp).
  throw StringError("NNCacheLevelZeroBound: a bound kind this build does not implement.");
}

std::string NNCacheLevelZeroBound::describe() const {
  switch(kind_) {
  case Kind::All: return "every persisted key";
  case Kind::MinLookups: return NNCacheLookupThreshold::of(lookups_).describe();
  case Kind::MaxEntries: return "at most " + Global::int64ToString(amount_) + " entries";
  case Kind::MaxBytes: return "at most " + Global::int64ToString(amount_) + " resident payload bytes";
  default:
    break;
  }
  throw StringError("NNCacheLevelZeroBound: a bound kind this build does not implement.");
}

//-------------------------------------------------------------------------------------
// The attach
//-------------------------------------------------------------------------------------

NNCacheLevelZeroLoad nnCacheLoadLevelZero(const NNCacheLevelZeroLoadRequest& request) {
  ClockTimer totalTimer;
  NNCacheLevelZeroLoad out;

  // Both names reach a path, and both file types validate them to the closed
  // path-component alphabet at their own boundary before doing so. Binding here is what
  // performs that validation; neither call touches a file.
  const NNEvalContainer container = NNEvalContainer::forContextAndModel(
    request.directory, request.context, request.modelInternalName, request.modelVersion);
  const NNCacheCountLog countLog = NNCacheCountLog::forContext(request.directory, request.context);

  ClockTimer keySetTimer;
  // THE KEY SETS ONLY: 32 bytes per container entry and 24 per count-log row. The container's
  // ~69 MB of payload is not decoded here and most of it may never be.
  const NNEvalContainerIndex containerIndex = container.loadIndex();
  const NNCacheCountLogContents counts = countLog.load();
  const double keySetMs = keySetTimer.getSeconds() * 1000.0;

  const std::vector<NNCacheLevelZeroCandidate> ordered =
    nnCacheLevelZeroOrder(containerIndex.entries(), counts.rows());
  const size_t taken = request.bound.select(ordered);

  // The arena is sized from the SELECTED entries' own headers, exactly, before a payload
  // byte is read. There is no growth path and no estimate.
  size_t ownerMapFloats = 0;
  std::vector<NNEvalContainerEntryLocation> selected;
  selected.reserve(taken);
  for(size_t i = 0; i < taken; i++) {
    const NNEvalContainerEntryLocation& loc = containerIndex.entries()[ordered[i].containerIndex];
    if(loc.hasOwnerMap)
      ownerMapFloats += (size_t)loc.nnXLen * (size_t)loc.nnYLen;
    selected.push_back(loc);
  }

  const std::shared_ptr<NNCacheLevelZeroArena> arena =
    NNCacheLevelZeroArena::reserve(taken, ownerMapFloats);

  ClockTimer payloadTimer;
  container.readEntriesInto(selected, *arena);
  const double payloadMs = payloadTimer.getSeconds() * 1000.0;

  ClockTimer buildTimer;
  out.levelZero = NNCacheFrozen::build(arena);
  const double buildMs = buildTimer.getSeconds() * 1000.0;

  out.remainder.assign(ordered.begin() + (ptrdiff_t)taken, ordered.end());

  NNCacheLevelZeroLoadReport& r = out.report;
  r.entriesInContainer = (int64_t)containerIndex.entries().size();
  r.entriesCounted = 0;
  for(size_t i = 0; i < ordered.size(); i++)
    r.entriesCounted += ordered[i].counted ? 1 : 0;
  r.entriesUncounted = r.entriesInContainer - r.entriesCounted;
  r.entriesInLevelZero = (int64_t)taken;
  r.entriesLeftOver = (int64_t)out.remainder.size();
  r.arenaEvaluationBytes = arena->evaluationBlockBytes();
  r.arenaOwnerMapBytes = arena->ownerMapBlockBytes();
  r.arenaTotalBytes = arena->totalBytes();
  r.levelZeroStructureBytes = (int64_t)out.levelZero->structureBytes();
  r.containerTail = containerIndex.tail();
  r.containerDiscardedTailBytes = containerIndex.discardedTailBytes();
  r.countLogTail = counts.tail();
  r.countLogDiscardedTailBytes = counts.discardedTailBytes();
  r.keySetMilliseconds = keySetMs;
  r.payloadMilliseconds = payloadMs;
  r.buildMilliseconds = buildMs;
  r.totalMilliseconds = totalTimer.getSeconds() * 1000.0;
  return out;
}

//-------------------------------------------------------------------------------------
// The detach
//-------------------------------------------------------------------------------------

NNCacheLevelZeroRelease nnCacheReleaseLevelZero(std::unique_ptr<NNCacheFrozen> levelZero) {
  NNCacheLevelZeroRelease result;
  result.storageReleased = true;
  if(levelZero != nullptr) {
    // A WEAK REFERENCE, taken before the release and read after it. This observes whether the
    // storage object actually went, which is the claim being made, rather than observing that
    // this function dropped its own handle, which is not the same claim: an aliased
    // evaluation handed out by a get holds the whole store alive, so the handle going and the
    // storage going are two different events (ADR-0021 Rule 1).
    const std::weak_ptr<NNCacheEvaluationStore> watch = levelZero->evaluationStore();
    levelZero.reset();
    result.storageReleased = watch.expired();
  }
  // After the release, once, at the boundary. See nnCacheReclaimFreedHeap for what it is and
  // is not for.
  result.reclaim = nnCacheReclaimFreedHeap();
  return result;
}
