#ifndef NEURALNET_NNEVALCONTAINER_H_
#define NEURALNET_NNEVALCONTAINER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "../core/hash.h"
#include "../neuralnet/nncachefileformat.h"
#include "../neuralnet/nninputs.h"

// An APPEND-ONLY CONTAINER of persisted NN evaluations for one (context, model) pair --
// the ".nnevals" file the count log deliberately declined to invent.
//
// WHAT IT IS FOR. NNCacheCountLog persists COUNTS: how often a key was asked for. This
// persists the EVALUATIONS themselves, so a session can be handed back the NNOutputs a
// previous session earned instead of recomputing them at 2.4-2.8 ms each. The two files are
// separate on purpose -- counts and evaluations are different facts with different write
// rhythms -- and are joined by their shared context identity (see nncachefileformat.h).
//
// ONE CONTAINER, ONE MODEL, AND THE MODEL IS NAMED IN THE HEADER. The NN cache key
// (NNInputs::getHash) folds in board, rules, komi and search-shaping parameters and names
// NO net, so the same position produces the same 128-bit key under every model. A store
// that mixed models under bare keys would therefore serve one net's evaluation as another's,
// silently. The type answer is that model identity is a property of the CONTAINER: one file
// per (context, model), the model's internalName in the file header, and NO per-entry model
// field -- so an entry cannot carry the wrong model, because an entry carries no model at
// all. A container whose header names a different model than the one a caller bound to is
// REFUSED, by name. Reading another model's evaluations deliberately (the "strongest net
// replaces the weaker" case) is done by binding a container to THAT model and merging the
// result upstream, never by relaxing this check.
//
// WRITES HAPPEN ONLY ON AN EXPLICIT DUMP, exactly as for the count log. Nothing here is
// reachable from NNCacheTable::get or ::set, and appendBlock writes the changed set once.
//
// ONE WRITER. As with the count log, a given (context, model) file is owned by one engine
// process for the duration of a dump. There is no lock file and no reader/writer protocol;
// two concurrent appenders would interleave partial blocks and the loser's would be
// discarded as a torn tail. Stated, not defended against.
//
// THE SIZES THIS IS DESIGNED FOR. The operator's median card holds ~45,664 entries at
// ~1,520 bytes each without ownership maps and ~2,964 with, so a median container is
// ~69-135 MB and the largest is ~6x that. Two consequences are built into the code rather
// than hoped about: a block is CHECKSUMMED BY STREAMING it through a bounded buffer, so
// verifying a block never requires holding the block; and a block is WRITTEN by encoding
// each entry into a reused per-entry buffer, so appending N entries never builds an N-entry
// image in memory. The only memory proportional to the file is the decoded entries
// themselves, which are what the caller asked for.

//-------------------------------------------------------------------------------------
// The values
//-------------------------------------------------------------------------------------

// Whether the file ended on a block boundary, or on bytes a crash left behind.
//
// A typed disposition rather than a byte count whose zero has to be read as a meaning
// (ADR-0012 P11); the coupling to the byte count is checked at construction.
enum class NNEvalContainerTail {
  // Every byte of the file was accounted for by the header and by whole, checksum-verified
  // blocks.
  Intact,
  // The file ended part-way through a block, or a block failed its checksum. Everything up
  // to the last intact block was applied; the remainder is reported in
  // discardedTailBytes() and is what the next append will drop.
  Truncated,
};

// The result of reading a container.
class NNEvalContainerContents {
 public:
  // Throws StringError if tail and discardedTailBytes disagree -- Truncated with zero
  // discarded bytes, or Intact with some, are both unrepresentable rather than states a
  // reader has to be careful about.
  static NNEvalContainerContents of(
    std::vector<std::unique_ptr<NNOutput>> entries,
    NNEvalContainerTail tail,
    int64_t discardedTailBytes,
    int64_t blocksApplied,
    int64_t entriesApplied
  );

  NNEvalContainerContents(NNEvalContainerContents&&) = default;
  NNEvalContainerContents& operator=(NNEvalContainerContents&&) = default;
  NNEvalContainerContents(const NNEvalContainerContents&) = delete;
  NNEvalContainerContents& operator=(const NNEvalContainerContents&) = delete;

  // One entry per distinct key, after merging, in first-appearance order of the key.
  const std::vector<std::unique_ptr<NNOutput>>& entries() const { return entries_; }

  // Moves the entries out, so a loader can hand them straight to NNCacheFrozen::build
  // without copying ~69 MB of payload. The contents keep their accounting and report zero
  // entries afterwards.
  [[nodiscard]] std::vector<std::unique_ptr<NNOutput>> takeEntries();

  NNEvalContainerTail tail() const { return tail_; }
  // Bytes after the last intact block. Positive exactly when tail() is Truncated.
  int64_t discardedTailBytes() const { return discardedTailBytes_; }
  // How many blocks were applied. Zero means the file was absent or held only its header.
  int64_t blocksApplied() const { return blocksApplied_; }
  // How many entries were applied, counting a key once per block it appeared in. This is
  // the file's physical size in entries; entries().size() is its live set, and their ratio
  // is what the compaction trigger reads.
  int64_t entriesApplied() const { return entriesApplied_; }

 private:
  NNEvalContainerContents(
    std::vector<std::unique_ptr<NNOutput>> entries,
    NNEvalContainerTail tail,
    int64_t discardedTailBytes,
    int64_t blocksApplied,
    int64_t entriesApplied
  );

  std::vector<std::unique_ptr<NNOutput>> entries_;
  NNEvalContainerTail tail_;
  int64_t discardedTailBytes_;
  int64_t blocksApplied_;
  int64_t entriesApplied_;
};

// WHERE ONE ENTRY SITS IN A CONTAINER, and what shape it decodes to -- everything about an
// entry except its numbers.
//
// This is what the gathered entry-header array is FOR. A caller that wants a container's key
// set -- to order it, to select part of it, to cost it, or to answer "is this key already on
// disk" -- reads 32 bytes per entry and no payload at all: 1.46 MB at the operator's median
// card against the ~69 MB of payload those headers index. The level-0 loader is that caller:
// it orders the whole key set against the count log and then reads only the payloads its
// selection bound kept, so an attach that takes the top 10,000 keys of a 45,664-key card
// reads a tenth of the payload rather than all of it and then throwing nine tenths away.
struct NNEvalContainerEntryLocation {
  Hash128 key;
  int nnXLen;
  int nnYLen;
  bool hasOwnerMap;
  // Absolute offsets in this container's file, of the entry's 32-byte header and of its
  // payload. Absolute rather than block-relative because a caller reads entries in an order
  // of its own choosing across blocks and has no business knowing where blocks begin.
  int64_t headerFileOffset;
  int64_t payloadFileOffset;
  int64_t payloadBytes;
};

// A container's merged key set with each key's location: the same merge load() applies, over
// the same blocks, without a payload byte read.
class NNEvalContainerIndex {
 public:
  // Throws StringError if tail and discardedTailBytes disagree, exactly as
  // NNEvalContainerContents does and for the same reason.
  static NNEvalContainerIndex of(
    std::vector<NNEvalContainerEntryLocation> entries,
    NNEvalContainerTail tail,
    int64_t discardedTailBytes,
    int64_t blocksApplied,
    int64_t entriesApplied
  );

  // One location per distinct key, after merging, in first-appearance order of the key --
  // the same order and the same live set load() would produce.
  const std::vector<NNEvalContainerEntryLocation>& entries() const { return entries_; }

  NNEvalContainerTail tail() const { return tail_; }
  int64_t discardedTailBytes() const { return discardedTailBytes_; }
  int64_t blocksApplied() const { return blocksApplied_; }
  int64_t entriesApplied() const { return entriesApplied_; }

  // The payload bytes of the whole live set, and the ownership-map floats within them --
  // what an arena has to reserve to hold a selection, summed from the headers rather than
  // estimated from a per-entry average.
  int64_t totalPayloadBytes() const;

 private:
  NNEvalContainerIndex(
    std::vector<NNEvalContainerEntryLocation> entries,
    NNEvalContainerTail tail,
    int64_t discardedTailBytes,
    int64_t blocksApplied,
    int64_t entriesApplied
  );

  std::vector<NNEvalContainerEntryLocation> entries_;
  NNEvalContainerTail tail_;
  int64_t discardedTailBytes_;
  int64_t blocksApplied_;
  int64_t entriesApplied_;
};

// WHERE A READ ENTRY IS PUT. readEntriesInto decodes into storage the caller supplies rather
// than allocating and handing back, because the caller is the only one who knows how the
// storage must be destroyed: the ordinary path wants a new[] ownership map that ~NNOutput
// will delete[], and the level-0 loader wants one carved out of an arena that ~NNOutput must
// never touch. A reader that allocated on the caller's behalf would have to guess.
class NNEvalContainerEntrySink {
 public:
  virtual ~NNEvalContainerEntrySink();

  // The NNOutput the i'th requested entry is decoded into, where i indexes the locations
  // vector as the caller passed it -- NOT the order the file is read in.
  virtual NNOutput& outputFor(size_t i) = 0;
  // Storage for the i'th requested entry's ownership map: exactly `numFloats` floats, whose
  // lifetime the sink owns. Called only for an entry whose header says it carries one.
  virtual float* ownerMapFor(size_t i, size_t numFloats) = 0;
};

// What one appendBlock did.
struct NNEvalContainerAppendResult {
  // Bytes this call added to the file, framing included.
  int64_t bytesAppended;
  // Bytes of torn tail this call had to discard before it could append. Zero on every
  // ordinary append; positive exactly when the previous writer died mid-dump.
  int64_t tornTailBytesDiscarded;
  // WHAT THIS CALL DID ABOUT THE TAIL IT FOUND. Coupled to the count above: NotNeeded
  // exactly when it is zero, Truncated exactly when it is positive.
  //
  // It replaces a `bool rewroteTheFile`, which said whether the repair had rewritten the
  // whole file. Once the repair became a truncation no code path could set that flag, so it
  // was a field that documented an outcome it could no longer report and left a reader to
  // infer that a repair had happened at all from a byte count beside a flag that was always
  // false. See NNCacheFileTailRepair.
  NNCacheFileTailRepair tailRepair;
};

//-------------------------------------------------------------------------------------
// The container
//-------------------------------------------------------------------------------------

// One (context, model) evaluation container file.
//
// THE FILE FORMAT, in full, because the framing is the correctness core.
//
//   file header, 48 fixed bytes plus the model name, once:
//     [ 0.. 8)  magic, the 8 ASCII bytes "KGNNEVAL"
//     [ 8..12)  format version, u32                   -- 1 today
//     [12..16)  file header bytes, u32                -- 48 + model name length
//     [16..20)  entry header bytes, u32               -- 32
//     [20..24)  block header bytes, u32               -- 32
//     [24..32)  context name hash, u64                -- the count log's function, shared
//     [32..36)  model version, u32                    -- NNEvaluator::getModelVersion()
//     [36..40)  model internalName length, u32
//     [40..48)  reserved, zero
//     [48..  )  model internalName, ASCII, closed alphabet
//   then zero or more blocks, one per dump:
//     block header, 32 bytes:
//       [ 0.. 4)  block magic, u32
//       [ 4.. 8)  entry count, u32
//       [ 8..16)  total PAYLOAD bytes of this block's entries, u64 (entry headers excluded)
//       [16..24)  checksum over the block's entry bytes, headers included, u64
//       [24..32)  checksum over bytes [0..24) of this header, u64, seeded with the context
//                 hash
//     then the block's KEY INDEX: entry count entry headers, CONTIGUOUS, 32 bytes each:
//       [ 0..16)  key: hash0 u64, hash1 u64 -- NNOutput::nnHash, restored verbatim
//       [16..18)  flags, u16 -- bit0: ownership map present; bits 1..15 reserved, zero
//       [18..19)  nnXLen, u8
//       [19..20)  nnYLen, u8
//       [20..24)  payload bytes, u32 -- of this entry, its header excluded
//       [24..32)  payload offset within this block's payload region, u64
//     then the block's PAYLOAD REGION: the same entry count payloads, concatenated in the
//     same order, every value an IEEE-754 f32 little-endian:
//       10 scalars: whiteWinProb, whiteLossProb, whiteNoResultProb, whiteScoreMean,
//                   whiteScoreMeanSq, whiteLead, varTimeLeft, shorttermWinlossError,
//                   shorttermScoreError, policyOptimismUsed
//       policy:     nnXLen*nnYLen + 1 values, BOARD-SIZED, illegal moves negative as in
//                   memory
//       ownermap:   nnXLen*nnYLen values, present exactly when flags bit0 is set
//
// WHY THE ENTRY HEADERS ARE GATHERED RATHER THAN INTERLEAVED WITH THEIR PAYLOADS. A caller
// that wants a block's KEY SET -- to decide what to re-dump, to merge, to compact, or to
// answer "is this key already on disk" without attaching anything -- would otherwise have to
// walk every payload to find the next key, reading ~69 MB at the operator's median card to
// recover ~1.5 MB of keys. Gathered, the key set with its ownermap flags is ONE SEQUENTIAL
// READ of 32 bytes per entry: 1.46 MB at the median card, 9.3 MB at the largest.
//
// AND WHY IT IS A GATHERING OF THE HEADERS RATHER THAN A SEPARATE ARRAY OF KEYS. A key array
// beside the entry headers would store each key and each ownermap flag TWICE, in two places
// that can disagree -- and a reader meeting a disagreement would have to invent which one is
// authoritative (ADR-0012 P1: every fact has exactly one home). The gathering buys the same
// sequential key scan with no second copy of anything: there is still exactly one entry
// header per entry, holding exactly the fields it always held.
//
// THE PAYLOAD OFFSET occupies the 8 bytes the entry header formerly reserved, so reaching
// one entry's payload is O(1) from its header rather than a prefix sum over every preceding
// entry's payload-bytes field. It is a derived quantity with one authority -- the running
// sum of the preceding payload sizes -- so the reader RECOMPUTES that sum and refuses a file
// whose stored offset disagrees with it, rather than trusting a second statement of a fact
// the payload sizes already determine.
//
// Every multi-byte field is little-endian and is packed and unpacked byte by byte, so the
// format is a fact about these bytes and not about any compiler's struct layout (ADR-0012
// P7), and a file written by one build is readable by another.
//
// WHY THE POLICY IS BOARD-SIZED AND NOT MAX_NN_POLICY_SIZE. The compile-time maximum is a
// property of a BUILD (COMPILE_MAX_BOARD_LEN), not of an evaluation; storing 362 slots for
// a 9x9 position would bake one build's constant into every file, and a build with a larger
// maximum could not read a smaller build's files without knowing which constant they were
// written under. The loader re-expands into whatever the running build's array is, ZEROING
// the slots beyond the board -- those slots are not a fact about the evaluation -- and
// REFUSES an entry whose board exceeds what this build can represent, naming both numbers.
//
// WHY noisedPolicyProbs CANNOT BE STORED. There is no flag for it, so a file cannot carry
// it, and an entry that has one is REFUSED at write time rather than silently written
// without it. It is search-time randomization attached to a tree node, not a fact about the
// position; persisting it would replay one session's noise as another session's evaluation.
// A future need for it is a format version bump, visible by construction.
//
// WHY THE BLOCK HEADER CARRIES TWO CHECKSUMS. The entry count and the payload total are
// lengths read from a file a crash may have left half-written. Trusting them before they
// are verified is how a loader ends up allocating on a number a crash chose. So the header
// checksums itself first, over its own preceding 24 bytes, seeded with the context hash;
// only then are the lengths read, and they are bounded a second time against the bytes
// actually remaining in the file before a single entry byte is read.
//
// HOW A TORN TAIL IS DETECTED AND WHAT HAPPENS TO IT. Reading stops at the first block that
// is short, whose header checksum fails, whose declared extent exceeds the bytes remaining,
// or whose entry checksum fails; everything before it has already been applied and is kept.
// The gathered header array is a NEW PLACE FOR A LENGTH TO LIE and is bounded like every
// other: a block claiming N entries is refused before its header array is allocated unless
// the file actually holds 32*N bytes for it, and the payload total is bounded against what
// remains after that.
// A block is all-or-nothing: an entry that sits whole inside a torn block is discarded with
// it, because the block, not the entry, is the unit a dump wrote.
//
// A REFUSAL AND A TORN TAIL ARE DIFFERENT ANSWERS TO DIFFERENT QUESTIONS, and the order the
// checks run in is what keeps them apart. Anything a crash can produce -- a short file, a
// bad checksum, an impossible length -- is a TORN TAIL, reported as a disposition. Anything
// a crash cannot produce -- a wrong magic, an unknown version, a stride this build does not
// write, another context's hash, another model's name, an entry whose declared shape does
// not match its declared size, an unknown flag bit -- is a REFUSAL, thrown by name, and is
// only ever judged AFTER the block's checksum has already proven the bytes are the bytes
// the writer wrote. An operator pointing at the wrong file and a process that died mid-dump
// are different events and this format never conflates them.
//
// THE READER NEVER REPAIRS -- THE WRITER DOES. load() is a pure read and reports the tail as
// a disposition. appendBlock scans first and, if the tail is torn, rewrites the intact
// prefix atomically before appending. That is not tidiness: appending after a torn tail
// would put the new block at an offset no loader ever reaches, so every subsequent dump
// would be silently lost while every call reported success.
//
// MERGE SEMANTICS ACROSS BLOCKS: LAST-WINS PER KEY, EXCEPT THAT AN ENTRY WITHOUT AN
// OWNERSHIP MAP NEVER SUPERSEDES ONE WITH. Last-wins makes a dump an increment, with no
// absolute-versus-delta flag to get wrong -- the count log's additive blocks in the
// evaluation register. The exception makes the live supersession rule hold at the store: a
// cached entry lacking a requested ownership map costs a full re-evaluation (42% of hits in
// a measured mixed workload), so a later ownermap-less re-evaluation must not be allowed to
// erase ownership knowledge a card already earned. Compaction applies the same two rules.
//
// EVERY PUBLIC OPERATION BELOW TAKES THE CONTEXT'S CROSS-PROCESS LOCK, and which one it takes
// is decided by whether it writes: appendBlock, compact and compactIfNeeded take the EXCLUSIVE
// lock, load, loadIndex and readEntriesInto take the SHARED one. See NNCacheFileLock
// (nncachefileformat.h) for why the lock lives on a file of its own and why a reader needs one
// at all. The lock is acquired and released WITHIN each call, so this class holds no lock
// between calls and a caller cannot forget to release one.
//
// THE LOCK IS TAKEN HERE, NOT AT THE ACTION LAYER, and the choice is load-bearing rather than
// aesthetic. Locking inside the primitive covers every caller -- the dump path, the level-0
// attach loader, and the reporting path (cacheStatsExecute reads a context's count log to
// answer cache_stats, which no lock at a dump or attach entry point would ever cover) -- and it
// is the only placement under which a caller cannot reach these bytes unlocked by taking a
// different route to them. It also means locks nest nowhere: an outer exclusive lock at an
// action entry point around an inner exclusive lock here would self-conflict, because flock
// excludes two open file descriptions whether or not they belong to the same process.
class NNEvalContainer {
 public:
  // Binds to the container for `context` and `modelInternalName` under `directory`.
  //
  // Throws StringError, naming what failed, if `directory` is not an existing directory, if
  // either name is not in the closed alphabet a path component may hold (see
  // nncachefileformat.h), or if modelVersion is negative. A real internalName such as
  // kata1-b18c384nbt-s9732312320-d4245566942 is inside that alphabet.
  //
  // Touches no file: it neither creates nor reads the container.
  static NNEvalContainer forContextAndModel(
    const std::string& directory,
    const std::string& context,
    const std::string& modelInternalName,
    int modelVersion
  );

  const std::string& path() const { return path_; }
  const std::string& context() const { return context_; }
  const std::string& modelInternalName() const { return modelInternalName_; }
  int modelVersion() const { return modelVersion_; }

  // Appends one block holding these evaluations, then fsyncs, so that when this returns the
  // bytes are on the device rather than in a page cache.
  //
  // Repairs a torn tail first, if there is one, and reports it in the result. Creates the
  // file, with its header, if it does not exist.
  //
  // Throws StringError if an entry is null, if an entry carries noisedPolicyProbs, if an
  // entry's nnXLen/nnYLen are outside what this format can frame, if the block would hold
  // more entries or bytes than the framing can express, or if any file operation fails.
  NNEvalContainerAppendResult appendBlock(const std::vector<std::shared_ptr<const NNOutput>>& entries) const;

  // Reads the whole container, merged. A missing file is not an error: it reads as zero
  // entries, Intact -- "no dump has happened here yet" is a normal answer.
  //
  // Throws StringError only for a file that exists and is not this container: see the
  // refusal-versus-torn-tail paragraph above.
  [[nodiscard]] NNEvalContainerContents load() const;

  // Reads the container's KEY SET AND NOTHING ELSE: the file header, every block header,
  // and every block's gathered entry-header array, with the same merge and the same
  // torn-tail contract load() applies. No payload byte is decoded and none is held.
  //
  // A block's bytes are still STREAMED through the checksum before its headers are believed,
  // exactly as in load(), so this reads the whole file even though it retains 32 bytes per
  // entry of it. Verifying less would mean trusting lengths a crash may have chosen.
  [[nodiscard]] NNEvalContainerIndex loadIndex() const;

  // Decodes exactly the entries at `locations` into `sink`, filling sink slot i from
  // locations[i]. The file is read in ASCENDING OFFSET ORDER whatever order the locations
  // arrive in, so a selection scattered across the file is still one forward pass.
  //
  // Throws StringError if the file is not this container (the same boundary load() applies:
  // wrong magic, wrong version, wrong context, WRONG MODEL), or if an entry's header on disk
  // no longer says what its location says -- which means the file changed under the caller
  // between loadIndex and here, and is refused rather than read as if it had not.
  void readEntriesInto(
    const std::vector<NNEvalContainerEntryLocation>& locations,
    NNEvalContainerEntrySink& sink
  ) const;

  // Rewrites the container as its header plus one block holding the merged live set, via a
  // temp file and an atomic rename. Repairs a torn tail as a side effect, since it writes
  // only what load() applied. Returns what it wrote.
  NNEvalContainerContents compact() const;

  // Compacts if the file holds more than `liveSetMultiple` times as many entries as it has
  // distinct keys; repairs a torn tail either way. Returns WHICH OF THE THREE THINGS it did,
  // because a bool could not tell two of them apart and the caller reports the answer to an
  // operator:
  //
  //   OVER THE MULTIPLE: a compaction, the whole file rewritten as its header plus one block
  //   holding the merged live set.
  //
  //   TORN BUT UNDER THE MULTIPLE: a TRUNCATION back to the end of the last intact block,
  //   and nothing else. The size trigger did not fire, so nothing here authorises rewriting
  //   a card the operator did not ask to have rewritten -- and at 10-20 GB per card that
  //   distinction is the difference between a metadata operation and a full rewrite. A
  //   compaction, when it does fire, subsumes the repair: it writes a fresh file from the
  //   intact part and the torn tail goes with the old inode.
  //
  //   BOTH: the compaction, which subsumes the repair.
  //
  // Throws StringError if liveSetMultiple is below 1: a multiple of zero would compact on
  // every call and a negative one has no reading.
  NNCacheFileMaintenance compactIfNeeded(int liveSetMultiple) const;

  // The default multiple, the count log's, for the same amortisation reason.
  static int defaultCompactionMultiple();

  // The format's own constants, named here so a test asserts against the implementation
  // rather than against a second copy of the numbers (ADR-0012 P1).
  static uint32_t formatVersion();
  static size_t fixedFileHeaderBytes();
  static size_t blockHeaderBytes();
  static size_t entryHeaderBytes();
  // 48 + the model name's length. Throws if the name is not in the closed alphabet.
  static int64_t fileHeaderBytesFor(const std::string& modelInternalName);
  // The payload bytes of one entry of this shape, its header excluded. Throws if the shape
  // is not one this format can frame.
  static int64_t payloadBytesFor(int nnXLen, int nnYLen, bool hasOwnerMap);
  // The exact bytes one entry of this shape occupies in a block, its header included. The
  // header and the payload are not adjacent -- the headers are gathered -- so this is a
  // total, not a span.
  static int64_t bytesForEntry(int nnXLen, int nnYLen, bool hasOwnerMap);

 private:
  NNEvalContainer(
    std::string path,
    std::string directory,
    std::string context,
    std::string modelInternalName,
    int modelVersion,
    uint64_t contextHash
  );

  // compact() WITHOUT taking the context lock, for the one caller that is already holding it.
  // compactIfNeeded decides on a scan it took under its own exclusive lock and then rewrites;
  // routing that rewrite through the public compact() would ask for a SECOND exclusive lock on
  // the same context from the same process. An advisory lock has no reason to grant that to its
  // own holder -- flock conflicts between two open file descriptions whoever owns them -- so the
  // call would sit out the whole deadline and then throw. The rewrite still has exactly one home
  // (this function); the public entry point is a lock plus a call to it.
  NNEvalContainerContents compactUnlocked() const;

  std::string path_;
  // The directory the two data files and the lock file share. Kept rather than recovered from
  // path_ by stripping a suffix: the lock file is a THIRD name in this directory, not a
  // rewriting of the container's own name, and a parser that recovered the directory from a
  // path would be a second author of the naming convention (ADR-0012 P1).
  std::string directory_;
  std::string context_;
  std::string modelInternalName_;
  int modelVersion_;
  uint64_t contextHash_;
};

#endif  // NEURALNET_NNEVALCONTAINER_H_
