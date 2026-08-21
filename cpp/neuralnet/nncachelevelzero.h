#ifndef NEURALNET_NNCACHELEVELZERO_H_
#define NEURALNET_NNCACHELEVELZERO_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "../core/hash.h"
#include "../neuralnet/nncachecountlog.h"
#include "../neuralnet/nncachefrozen.h"
#include "../neuralnet/nnevalcontainer.h"

// THE LEVEL-0 LOADER: an attach reads a context's evaluation container and its count log,
// orders the container's key set by the count log, applies the caller's selection bound,
// and builds one frozen CHD level 0 whose evaluations live in an ARENA that the matching
// detach releases whole.
//
// WHAT THIS FILE IS FOR, in one sentence: nncachefrozen.h builds a level 0 from a vector of
// evaluations someone else already has; this file is where that vector comes from, and
// where the memory it occupies is answered for.
//
// THE MEMORY POSTURE IS HALF THE POINT AND IS STATED FIRST. The operator's complaint is
// that his KataGo instances grow to ten gigabytes and hold it. So:
//
//   NO LONG-LIVED ARENA. The arena is created by an attach and destroyed by the matching
//   detach. There is no pool, no free list, no reuse across sessions, and nothing here is
//   allocated at process start. A session that attaches three cards and detaches them has
//   the memory back; a process that attaches nothing allocates nothing.
//
//   TWO CONTIGUOUS BLOCKS PER ATTACH, NOT ONE ALLOCATION PER ENTRY. The evaluations of one
//   level 0 are one block of NNOutputs and one block of ownership-map floats, each sized
//   exactly once from the container's own entry headers before a payload byte is read. The
//   shape being replaced is ~45,664 separate `new NNOutput` plus ~45,664 separate
//   `new float[361]` per card -- ninety thousand allocations whose freeing returns almost
//   nothing to the operating system because they are interleaved with everything else the
//   process allocated in between.
//
//   THE ALLOCATOR IS ASKED TO GIVE THE PAGES BACK, once, at the detach boundary, where a
//   few hundred milliseconds cost nothing. See NNCacheHeapReclaim: it is glibc-only, the
//   non-glibc answer is a named disposition rather than a silent no-op, and the caller is
//   handed what happened rather than a promise about it.
//
// THE HAZARD THE ARENA EXISTS AROUND, named because it is the whole reason the evaluation
// store in nncachefrozen.h is an interface rather than a vector: ~NNOutput calls
// `delete[] whiteOwnerMap`. An ownership map carved out of a contiguous block is an
// interior pointer the allocator never issued, so ~NNOutput must never see one. The arena
// nulls every whiteOwnerMap in its own destructor, before its blocks are released, and it
// hands out no owning handle to an NNOutput at all -- so a `std::unique_ptr<NNOutput>`
// pointing into arena memory is not a thing to be careful about, it is a thing that cannot
// be constructed from anything this file exposes (ADR-0000 Rule 2a).

//-------------------------------------------------------------------------------------
// Giving the pages back
//-------------------------------------------------------------------------------------

// What the detach-time reclaim did.
//
// A TYPED DISPOSITION AND NOT A BOOL, because three different things can happen and two of
// them look identical from a bool: the allocator released pages, the allocator had nothing
// to release, and this build has no such allocator call at all. The third is the one that
// matters: `malloc_trim` is a glibc extension, and a build against musl, against Apple's
// libc or against MSVC's has no equivalent. Compiling it out silently would leave the
// non-glibc operator with a detach that quietly does less than the documentation says
// (ADR-0002); naming the case makes the platform difference a fact the caller reports
// rather than a fact nobody can see.
enum class NNCacheHeapReclaim {
  // The allocator was asked to return free heap pages and reported that it released some.
  Trimmed,
  // The allocator was asked and reported it had nothing to release. This is the ordinary
  // answer when the freed blocks were large enough that the allocator had already returned
  // them at free() -- which is exactly what an arena of a few tens of megabytes is.
  NothingToTrim,
  // This build's allocator exposes no trim call. NOTHING WAS ATTEMPTED, and that is why the
  // caller is being told.
  Unavailable,
};

// Asks the allocator to return free heap pages to the operating system, now.
//
// WHAT IT IS FOR AND WHAT IT IS NOT FOR. Under glibc, freeing memory that sits below other
// still-live allocations does not shrink the process: the pages stay in the allocator's
// arena, available for reuse but charged to the process. malloc_trim walks the free lists
// and releases what it can. It is worth calling exactly at a boundary where a large,
// interleaved set of allocations has just gone -- a detach -- and it is worth calling
// nowhere else, because it walks every arena and is not free.
//
// It is NOT what returns a large arena block. A block of a few tens of megabytes is served
// by mmap and handed straight back at free(); this call is about the rest -- the container's
// key set, the count log's rows, the CHD build's transient tables, the decode buffers --
// which are ordinary heap and are exactly the interleaved shape trimming is for.
NNCacheHeapReclaim nnCacheReclaimFreedHeap();

//-------------------------------------------------------------------------------------
// The arena
//-------------------------------------------------------------------------------------

// The storage one attached level 0's evaluations live in: one contiguous block of NNOutputs
// and one contiguous block of ownership-map floats, both sized exactly at construction and
// neither ever grown.
//
// WHY IT NEVER GROWS, stated as a rule rather than left as a habit: the ownership-map
// pointers inside the NNOutput block point into the float block, so a reallocation of either
// would leave every one of them dangling. The sizes are therefore taken from the container's
// own entry headers -- which state every selected entry's shape and whether it carries a map
// -- before a single payload byte is read, and the arena exposes no way to add an entry
// afterwards. "Do not grow this" is not a comment here; there is no growth operation.
//
// IT IS BOTH the evaluation store a frozen level 0 owns AND the sink the container decodes
// into, because those are two views of one fact: this object is where these evaluations
// live. An adapter between them would be a second object with the same lifetime and no
// second concern.
class NNCacheLevelZeroArena final : public NNCacheEvaluationStore, public NNEvalContainerEntrySink {
 public:
  // Reserves room for exactly `numEvaluations` evaluations and exactly `numOwnerMapFloats`
  // ownership-map floats between them.
  //
  // Throws StringError if either figure is one this build cannot allocate for.
  static std::shared_ptr<NNCacheLevelZeroArena> reserve(size_t numEvaluations, size_t numOwnerMapFloats);

  ~NNCacheLevelZeroArena() override;

  // The evaluation store (see nncachefrozen.h).
  size_t numEvaluations() const override;
  NNOutput* evaluationAt(size_t i) const override;
  // ZERO. The arena keeps no per-entry bookkeeping at all: evaluation i is at index i of a
  // contiguous block, so there is no pointer to store. That is why an arena-backed level 0
  // is 8 bytes per entry lighter than a heap-backed one.
  size_t handleBytes() const override;

  // The container sink (see nnevalcontainer.h).
  NNOutput& outputFor(size_t i) override;
  float* ownerMapFor(size_t i, size_t numFloats) override;

  // The two blocks, and their sum -- the memory an attach took and a detach gives back.
  int64_t evaluationBlockBytes() const;
  int64_t ownerMapBlockBytes() const;
  int64_t totalBytes() const;
  // Ownership-map floats handed out so far. Equal to the reserved figure once every selected
  // entry has been read.
  size_t ownerMapFloatsUsed() const { return ownerMapFloatsUsed_; }

 private:
  NNCacheLevelZeroArena(size_t numEvaluations, size_t numOwnerMapFloats);

  std::vector<NNOutput> outputs_;
  std::vector<float> ownerMaps_;
  size_t ownerMapFloatsUsed_;
};

//-------------------------------------------------------------------------------------
// The order, and the bound
//-------------------------------------------------------------------------------------

// One container key considered for level 0, with everything the ordering and the bound need.
struct NNCacheLevelZeroCandidate {
  Hash128 key;
  // Where this key sits in the container's own key set, so the caller can recover its
  // location without a second lookup.
  size_t containerIndex;
  // Observations the count log has recorded for this key across every dump it holds -- how
  // many times this position has come up under this context, across sessions. Zero for a key
  // the log does not mention -- see `counted`.
  uint64_t observations;
  // WHETHER THE COUNT LOG MENTIONED THIS KEY AT ALL. It is a separate fact from a count of
  // zero and is kept separate rather than both reading as "0" (ADR-0012 P11: an absence
  // carries a typed reason, never a value standing in for one). A container can legitimately
  // hold keys the log has never heard of -- a client that dumped "evaluations" without
  // "counts" produces exactly that -- and the ordering rule below treats the two
  // differently, so a reader that could not tell them apart could not check the ordering.
  bool counted;
  // What this entry will occupy in the arena: its NNOutput and its ownership map, if any.
  // The bound's byte arithmetic is denominated in THIS -- the resident bytes that actually
  // exhaust memory -- and not in the entry's size on disk, which is a different number.
  int64_t residentBytes;
};

// THE BUILD ORDER: the container's key set, ordered by the count log.
//
// DESCENDING OBSERVATIONS, and then every key the count log does not mention, ties broken by key
// throughout so the order is total and a test can assert it exactly.
//
// The absent-from-the-log case is the one worth stating precisely, because it is the one a
// loader could plausibly get wrong in three different ways -- drop the key, guess a count
// for it, or let it sort among the counted keys. It does none of them: an unmentioned key is
// KEPT, is ordered as zero observations, and sorts AFTER every key the log did mention, including
// a key the log mentioned with a count of zero. A key the log has counted at zero is a key
// the log has something to say about; a key it has never seen is not, and the second is the
// weaker claim on level 0.
//
// Ordering by observations rather than by anything else is the operator's own ruling, and
// byDescendingObservations is the count log's only ordering helper for that reason.
[[nodiscard]] std::vector<NNCacheLevelZeroCandidate> nnCacheLevelZeroOrder(
  const std::vector<NNEvalContainerEntryLocation>& containerEntries,
  const std::vector<NNCacheCountRow>& countRows
);

// WHAT A CLIENT LETS INTO LEVEL 0, in the caller's own currency.
//
// The knobs are mechanism and not policy: the client decides what belongs in a frozen level
// 0 and what should be left for level 1, and says so as a bound. There is one bound per
// attach and it is a closed set of four kinds, so "minObservations and maxBytes together" is not
// a request that can be made and then interpreted -- a caller wanting both composes two
// attaches or picks the one that expresses its intent.
class NNCacheLevelZeroBound {
 public:
  // Every candidate.
  static NNCacheLevelZeroBound all();
  // Every candidate the count log recorded at least `observations` observations for. A key
  // the log does not mention is never admitted by this bound at any threshold above zero,
  // because its count is not zero -- it is unknown, and admitting it would be the guess
  // NNCacheLevelZeroCandidate::counted exists to prevent.
  static NNCacheLevelZeroBound minObservations(uint64_t observations);
  // The first `entries` candidates in order.
  static NNCacheLevelZeroBound maxEntries(int64_t entries);
  // The longest prefix whose RESIDENT bytes do not exceed `bytes`. The bound is denominated
  // in the resource that actually exhausts -- the arena's own footprint -- rather than in a
  // slot count or a file size standing proxy for it.
  static NNCacheLevelZeroBound maxBytes(int64_t bytes);

  // How many of `ordered` this bound takes: always a PREFIX, because the order is the
  // ranking and taking a non-prefix would mean admitting a less popular key over a more
  // popular one.
  //
  // The bound applies itself rather than exposing its kind and its number for a caller to
  // switch over, so each kind's arithmetic has exactly one home and a caller cannot read a
  // byte bound as an entry count (ADR-0012 P1/P8).
  [[nodiscard]] size_t select(const std::vector<NNCacheLevelZeroCandidate>& ordered) const;

  // For a report: what was asked for, in words, without the caller re-deriving it.
  [[nodiscard]] std::string describe() const;

 private:
  enum class Kind { All, MinObservations, MaxEntries, MaxBytes };
  NNCacheLevelZeroBound(Kind kind, uint64_t observations, int64_t amount);

  Kind kind_;
  uint64_t observations_;
  int64_t amount_;
};

//-------------------------------------------------------------------------------------
// The attach
//-------------------------------------------------------------------------------------

// What an attach was asked to do.
struct NNCacheLevelZeroLoadRequest {
  // The single directory the engine owns, holding <context>.nncounts and
  // <context>.<model>.nnevals. Both names are validated to the closed path-component
  // alphabet by the two file types themselves before either touches a path.
  std::string directory;
  std::string context;
  std::string modelInternalName;
  int modelVersion;
  NNCacheLevelZeroBound bound;
};

// What an attach did. Every figure here is a fact the loader observed, not one it assumed:
// the counts come from the files, the bytes from the arena, the milliseconds from a clock.
struct NNCacheLevelZeroLoadReport {
  // The container's live key set, after its merge.
  int64_t entriesInContainer;
  // Of those, how many the count log mentioned, and how many it did not. They sum to
  // entriesInContainer, and the second being large is the honest signal that a client has
  // been dumping evaluations without counts.
  int64_t entriesCounted;
  int64_t entriesUncounted;
  // How many the bound admitted, and how many it left for level 1.
  int64_t entriesInLevelZero;
  int64_t entriesLeftOver;
  // The arena's two blocks and their sum: what this attach is holding.
  int64_t arenaEvaluationBytes;
  int64_t arenaOwnerMapBytes;
  int64_t arenaTotalBytes;
  // The frozen structure's own footprint -- the CHD machinery and the key array -- which is
  // a separate figure from the payloads above and is the one the 44.9 B/entry ceiling is
  // about.
  int64_t levelZeroStructureBytes;
  // How the two files ended. An attach on a torn file is not an error: the intact prefix is
  // what a crash left behind and is exactly what a session should be given back.
  NNEvalContainerTail containerTail;
  int64_t containerDiscardedTailBytes;
  NNCacheCountLogTail countLogTail;
  int64_t countLogDiscardedTailBytes;
  // Wall time, split so a slow attach can be attributed rather than guessed at.
  double keySetMilliseconds;   // reading both files' key sets
  double payloadMilliseconds;  // reading the selected payloads into the arena
  double buildMilliseconds;    // the CHD build
  double totalMilliseconds;
};

// The result of an attach.
struct NNCacheLevelZeroLoad {
  // The built structure. Never null: a refusal throws and yields nothing (SPEC.md 5).
  std::unique_ptr<NNCacheFrozen> levelZero;
  // The candidates the bound did NOT take, still in build order -- what a level-1 fill would
  // admit next, and in what order. Their payloads were never read, so this is keys and
  // locations only.
  std::vector<NNCacheLevelZeroCandidate> remainder;
  NNCacheLevelZeroLoadReport report;
};

// Reads <context>.<model>.nnevals and <context>.nncounts under the request's directory,
// orders, selects, and builds.
//
// A MISSING CONTAINER IS A NORMAL ANSWER, not an error: it loads as an empty level 0, which
// is a legal structure that answers absent to everything (SPEC.md 4.3). A missing count log
// is likewise normal and means every container key is uncounted. The two absences are
// distinguishable in the report, through entriesInContainer and entriesCounted, rather than
// both presenting as a level 0 of zero entries.
//
// Throws StringError, naming the cause and yielding nothing, for everything the container,
// the count log or the CHD build refuses -- a container that names another model, a file
// this build cannot read, a key set that will not construct.
[[nodiscard]] NNCacheLevelZeroLoad nnCacheLoadLevelZero(const NNCacheLevelZeroLoadRequest& request);

//-------------------------------------------------------------------------------------
// The detach
//-------------------------------------------------------------------------------------

// What a detach did.
struct NNCacheLevelZeroRelease {
  // WHETHER THE EVALUATIONS' STORAGE ACTUALLY WENT. This is observed, through a weak
  // reference to the store that outlives the release by one statement, and not inferred from
  // the caller having dropped its handle: a level-0 get hands out its evaluation through
  // shared_ptr's aliasing constructor against the store, so a caller still holding one
  // returned NNOutput keeps the ENTIRE arena alive. False here means exactly that happened,
  // and it is a fact a detach must be able to report rather than one it may assume away
  // (ADR-0021 Rule 1: the witness observes the property, not the destructor).
  //
  // The protocol layer forecloses it upstream by refusing a detach while any request is
  // open; this figure is what makes that refusal checkable rather than trusted.
  bool storageReleased;
  // What the allocator was asked for afterwards, and what it said.
  NNCacheHeapReclaim reclaim;
};

// Releases an attached level 0 and asks the allocator to return what it can.
//
// IT TAKES THE STRUCTURE BY VALUE, so a caller cannot detach and still hold it: "released at
// detach" is a property of the signature rather than a rule the caller has to follow
// (ADR-0000 Rule 1). The reclaim runs after the release, at the boundary, once.
[[nodiscard]] NNCacheLevelZeroRelease nnCacheReleaseLevelZero(std::unique_ptr<NNCacheFrozen> levelZero);

#endif  // NEURALNET_NNCACHELEVELZERO_H_
