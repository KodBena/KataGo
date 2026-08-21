#ifndef NEURALNET_NNCACHECOUNTLOG_H_
#define NEURALNET_NNCACHECOUNTLOG_H_

#include <cstdint>
#include <string>
#include <vector>

#include "../core/hash.h"
#include "../neuralnet/nncache.h"
#include "../neuralnet/nncacheobservations.h"

// An APPEND-ONLY LOG of per-(key, context) OBSERVATION counts, with no dependency on
// anything outside the C and C++ standard libraries.
//
// WHAT IT IS FOR, AND WHAT IT IS NOT FOR. It persists COUNTS: for each 128-bit position
// hash, how many times that position was PRESENTED to the cache under this context, and in
// how many dumps it appeared. An observation is a would-have-been-computed forward pass --
// one evaluation request for one position, hit or miss -- and NOT a cache hit; see
// nncacheobservations.h, which owns that definition and the reasoning behind it. It does not
// persist evaluations. There is no NNOutput in this header, no payload container, and no
// schema for one -- persisting evaluations is a separate piece of work, and a container
// format invented here in passing would be that work done badly.
//
// WRITES HAPPEN ONLY ON AN EXPLICIT DUMP. Nothing in this file is reachable from the MCTS
// hot path. appendDump reads the finished per-key surface a table already keeps
// (NNCacheTable::takeUnpersistedObservationCountsFor) and writes it once.
//
// ONE WRITER. The file format assumes a single engine process owns a given context's log
// for the duration of a dump. There is no lock file, no advisory locking, and no
// reader/writer protocol; two processes appending to one context concurrently would
// interleave partial blocks and the loser's dump would be discarded as a torn tail on the
// next load. This is stated rather than defended against, because the deployment is one
// engine process.
//
// THE OPTIMISATION TARGET IS BYTES WRITTEN PER DUMP, not throughput. The keys are uniform
// 128-bit hashes, so there is no clustering to exploit and no ordering worth preserving,
// and a random-key update against any indexed store rewrites far more than the changed
// data. An append-only log writes exactly the changed set: 24 bytes per key that this dump
// has something to say about, plus 32 bytes of framing for a dump that has anything to say
// at all. At the working figure of ~40,000 entries per context that is ~960 KB. A dump with
// NO key to report and no unattributed observation either appends NOTHING -- not even the
// framing: see appendDump's own statement of this. A context an idle leaf keeps re-dumping
// on an interval therefore leaves this file byte-identical, rather than growing by one
// empty block forever.
//
// CRASH SAFETY IS APPEND-PLUS-FSYNC AND NOTHING MORE, so the framing has to make a torn
// tail detectable. See the format description above NNCacheCountLog.

//-------------------------------------------------------------------------------------
// The values
//-------------------------------------------------------------------------------------

// One key's accumulated totals across every dump the log holds.
//
// The totals are 64-bit here and 32-bit on the wire, deliberately: a single dump's count
// comes from NNCacheObservationCount::observations, which is a uint32_t, but a sum across dumps has no such
// bound and silently wrapping it would be the worst kind of quiet failure. Compaction,
// which is the only operation that has to fit an accumulated total back into a record,
// refuses loudly and names the key if one will not fit (ADR-0002).
struct NNCacheCountRow {
  Hash128 key;
  // Total OBSERVATIONS recorded for this key, summed over every dump in the log: how many
  // times this position has been presented to the cache under this context across every
  // session that has dumped here. This is the quantity a level-0 attach orders by and a disk
  // admission threshold compares against.
  uint64_t observations;
  // HOW MANY DUMPS THIS KEY WAS OBSERVED IN. Named "sessions" in the operator's own schema
  // and kept under that name; the definition is the one stated here.
  //
  // A dump appends the DELTA since the last one (NNCacheObservationDelta below), and the
  // delta surface omits a key with nothing to say -- so this rises by one per dump in which
  // the position was actually presented at least once, and a pre-warmed key a whole session
  // never asked for does not rise at all. That is what an additive log of deltas necessarily
  // produces, and it is the reading a client wants: in how many study sessions did this
  // position come up.
  //
  // WHAT THE CURRENCY CHANGE DID TO THIS FIELD, stated rather than left to drift (the K8
  // reconciliation). Its NAME, its wire slot, its arithmetic and the sentence above are all
  // unchanged. What changed is that the sentence is now TRUE. Under the old retrieval
  // currency a position that came up and MISSED -- freshly evaluated, or evaluated again
  // after an eviction -- earned no row, so its sessions did not rise even though the session
  // demonstrably worked on it; "in how many study sessions did this come up" was a claim the
  // field could not actually support, and only a position that came up AND was already
  // cached was ever credited. An observation is counted whether a level answered it or a
  // forward pass had to, so every session in which the position came up now credits it,
  // which is exactly what the field always said.
  //
  // NOT "how many dumps carried this key through". Whether an entry is still being carried
  // is a fact about CACHE RETENTION -- answered by the entry still being in the container --
  // and not a fact about how often the position was needed.
  //
  // The log still cannot tell two dumps in one engine run from two runs, and does not
  // pretend to: it counts dumps, which is what a dump can observe.
  uint64_t sessions;
};

// A MINIMUM RECORDED-OBSERVATION COUNT: the one home of the rule "this key has been seen
// often enough", for every side that asks it.
//
// WHY THIS IS A TYPE AND NOT A uint64_t PASSED AROUND. Two independent surfaces apply the
// same rule to the same fact. The READ side is NNCacheLevelZeroBound::minObservations, which
// decides what an attach admits into a frozen level 0. The WRITE side is
// NNCacheDiskAdmission::minObservations, which decides what a dump lets onto disk. They are
// different decisions -- one is a prefix of a ranked order, the other a per-entry
// predicate -- but the QUESTION they ask of a key is one question, and a key's recorded
// observations are one fact with one home, this log. Written out twice, the two comparisons
// could drift in the one place drifting is silent: the boundary case below (ADR-0012 P1).
//
// THE BOUNDARY CASE, stated once here so neither side restates it. A key the log has never
// mentioned carries a recorded count of zero, so every threshold ABOVE zero excludes it and
// a threshold OF zero admits it -- which is right in both directions: "admit everything" must
// not turn into "admit everything the log happens to know about". An uncounted key is not
// given a separate check, because there is nothing for one to catch; see
// NNCacheLevelZeroBound::select, where the same reasoning is already recorded against the
// read side's own comparison.
class NNCacheObservationThreshold {
 public:
  // Admits a key recorded at `observations` observations or more. of(0) admits every key,
  // counted or not.
  static NNCacheObservationThreshold of(uint64_t observations) { return NNCacheObservationThreshold(observations); }

  [[nodiscard]] uint64_t observations() const { return observations_; }

  // Whether a key the log records `recordedObservations` observations for clears this
  // threshold. A key the log does not mention is passed as zero, by the rule above.
  [[nodiscard]] bool admits(uint64_t recordedObservations) const { return recordedObservations >= observations_; }

  // For a report or a refusal: what was asked for, in words, without a caller re-deriving
  // it from the number.
  [[nodiscard]] std::string describe() const;

 private:
  explicit NNCacheObservationThreshold(uint64_t observations) : observations_(observations) {}
  uint64_t observations_;
};

// THE ONLY THING A DUMP MAY APPEND: a table's UNPERSISTED OBSERVATION DELTA, as a type that
// only the delta surface can produce.
//
// WHY THIS IS A TYPE AND NOT A SENTENCE IN appendDump's COMMENT. A block is a set of
// INCREMENTS -- the reader ADDS each record's observations to that key's running total and its
// sessions to that key's dump count (see the format description above NNCacheCountLog). So
// the only sound thing to append is what has accrued since the last dump, which is
// NNCacheTable::takeUnpersistedObservationCountsFor. Beside it sits
// NNCacheTable::harvestObservationCountsFor, which reports this session's RUNNING TOTAL for
// every key: a legitimate answer to a different question -- a one-shot report such as the
// cache_stats action -- and a double count the instant it reaches an additive reader.
//
// Both surfaces return NNCacheObservationLedger. So while appendDump took a ledger, the
// unsound composition read exactly as well as the sound one, and the only thing standing
// between them was a reviewer noticing. Choosing correctly once does not stop the next caller
// choosing wrong, so the fix is the one ADR-0000 Rule 2a asks for: appendDump takes a type
// the absolute surface cannot produce, and `log.appendDump(table.harvestObservationCounts())`
// stops compiling instead of stopping at review. This is the same move SearchableModelIdx
// (command/analysismodels.h) makes for a wrong-axis subscript, in the delta-versus-absolute
// register.
//
// AND THE WRONG CURRENCY IS FORECLOSED IN THE SAME MOVE, which is new. This type is
// constructible only from an OBSERVATION ledger, so `log.appendDump(table.harvestHitCounts())`
// -- retrievals, a different quantity that happens to have the same shape -- does not name a
// viable overload either. The two currencies were confused once already (ledger rows
// 1651/1654); the type is now what keeps them apart, not a comment.
//
// THE TWO DOORS, and why neither of them lets an absolute through. takeFor consumes a live
// table's per-context delta and is the production path. ofDeltaRows takes ROWS and never a
// ledger, for a caller that assembled the delta itself -- a test's synthetic dump, a fixture
// -- so there is no expression that converts a harvest into one of these; unwrapping a
// harvest's rows and re-wrapping them under a name that says "delta" is a deliberate act with
// the wrong word written at the call site, not a plausible-looking one-liner.
//
// THERE IS NO WHOLE-TABLE take(), and its absence is deliberate rather than an omission. A
// count log file belongs to ONE context, and an observation row belongs to exactly one context
// by construction -- the ledger is keyed on (key, context) -- so a whole-table delta could
// only be written into some context's file by attributing another context's presentations to
// it. The division is made where the fact lives and there is no door that skips it.
class NNCacheObservationDelta {
 public:
  // Takes the observations of exactly `context` that have not reached a count log yet,
  // advancing the marks of the rows it hands over AND OF NO OTHERS. CONSUMING: the same
  // presentations are never reported twice, which is exactly the property that makes the
  // result appendable. With two cards attached, each one's dump still finds its own delta
  // whole.
  //
  // Throws StringError for a context the table did not attach.
  [[nodiscard]] static NNCacheObservationDelta takeFor(NNCacheTable& table, const NNCacheContextId& context);

  // A delta the caller already holds as rows. `unrecordedObservations` is the honesty residue
  // -- presentations that happened and could not be given a row -- carried whole, exactly as
  // NNCacheObservationLedger carries it.
  [[nodiscard]] static NNCacheObservationDelta ofDeltaRows(
    std::vector<NNCacheObservationCount> rows, int64_t unrecordedObservations
  );

  // A delta from a table that observes nothing, because no context is attached to it.
  // Distinct from a delta of zero rows, and appendDump refuses it by name: see appendDump.
  [[nodiscard]] static NNCacheObservationDelta notObserved();

  // The rows and the residue, as the ledger type the rest of the cache already speaks.
  [[nodiscard]] const NNCacheObservationLedger& ledger() const { return ledger_; }

 private:
  explicit NNCacheObservationDelta(NNCacheObservationLedger ledger);
  NNCacheObservationLedger ledger_;
};

// Whether the file ended on a block boundary, or on bytes a crash left behind.
//
// A typed disposition rather than a byte count whose zero has to be read as a meaning
// (ADR-0012 P11). "Intact" and "Truncated with nothing discarded" are not the same claim,
// and the second is not a state this type can be in -- the coupling is checked at
// construction.
enum class NNCacheCountLogTail {
  // Every byte of the file was accounted for by the header and by whole, checksum-verified
  // blocks.
  Intact,
  // The file ended part-way through a block, or a block failed its checksum. Everything up
  // to the last intact block was applied; the remainder is reported in
  // discardedTailBytes() and is what the next append will drop.
  Truncated,
};

// The result of reading a context's log.
class NNCacheCountLogContents {
 public:
  // Throws StringError if tail and discardedTailBytes disagree -- Truncated with zero
  // discarded bytes, or Intact with some, are both unrepresentable states rather than
  // states a reader has to be careful about.
  static NNCacheCountLogContents of(
    std::vector<NNCacheCountRow> rows,
    int64_t unattributedObservations,
    NNCacheCountLogTail tail,
    int64_t discardedTailBytes,
    int64_t blocksApplied,
    int64_t recordsApplied
  );

  // One row per distinct key, in first-appearance order. Not sorted: see
  // byDescendingObservations.
  const std::vector<NNCacheCountRow>& rows() const { return rows_; }

  // Observations that happened and could not be attributed to any key, summed over every
  // applied block. This is NNCacheObservationLedger::unrecordedObservations carried through
  // the log rather than dropped at the door: a harvest that was short says by how much, and so
  // does the log that stored it (ADR-0002). Zero in every ordinary run.
  int64_t unattributedObservations() const { return unattributedObservations_; }

  NNCacheCountLogTail tail() const { return tail_; }
  // Bytes after the last intact block. Positive exactly when tail() is Truncated.
  int64_t discardedTailBytes() const { return discardedTailBytes_; }
  // How many blocks were applied. Zero means the file was absent or held only its header.
  int64_t blocksApplied() const { return blocksApplied_; }
  // How many records were applied, counting a key once per block it appeared in. This is
  // the log's physical size in records; rows().size() is its live set. Their ratio is what
  // the compaction trigger reads.
  int64_t recordsApplied() const { return recordsApplied_; }

  // The rows in descending observations, ties broken by key so the order is total and a test
  // can assert it.
  //
  // ORDERING IS BY OBSERVATIONS. That is the operator's ruling for what a frozen level 0
  // should be built in the order of -- in the currency he ratified, which is how often the
  // position comes up -- and it is why this is the only ordering helper here. Nothing in this
  // file ranks, weights or prefers sessions; sessions is stored because it is cheap to store
  // and is a fact he asked for, not because anything here treats it as the better predictor.
  [[nodiscard]] std::vector<NNCacheCountRow> byDescendingObservations() const;

 private:
  NNCacheCountLogContents(
    std::vector<NNCacheCountRow> rows,
    int64_t unattributedObservations,
    NNCacheCountLogTail tail,
    int64_t discardedTailBytes,
    int64_t blocksApplied,
    int64_t recordsApplied
  );

  std::vector<NNCacheCountRow> rows_;
  int64_t unattributedObservations_;
  NNCacheCountLogTail tail_;
  int64_t discardedTailBytes_;
  int64_t blocksApplied_;
  int64_t recordsApplied_;
};

// One block's own rows, as recorded -- never accumulated. A block is one dump's delta (see
// NNCacheCountLog's class comment below, "A BLOCK IS ONE DUMP AND ITS RECORDS ARE
// INCREMENTS"), and this is that unit kept separate rather than merged, for a caller that
// wants to show dump-by-dump history rather than only the totals NNCacheCountLogContents
// gives.
struct NNCacheCountLogBlockRow {
  Hash128 key;
  uint64_t observations;
  uint64_t sessions;
};

// One applied block, in file order.
struct NNCacheCountLogBlock {
  std::vector<NNCacheCountLogBlockRow> rows;
  // Observations this one dump could not attribute to any key. See
  // NNCacheCountLogContents::unattributedObservations for the sum of these over every block.
  int64_t unattributedObservations;
};

// load()'s answer, but with every applied block's own rows kept separate as well as merged
// into the usual totals.
//
// A SEPARATE TYPE RATHER THAN A FIELD ADDED TO NNCacheCountLogContents, because every
// existing caller of load() wants only the merged totals and gains nothing from carrying a
// whole file's raw blocks along for the ride; the two are different questions asked of the
// same scan, and a caller states which one it wants by which method it calls.
class NNCacheCountLogDetailedContents {
 public:
  static NNCacheCountLogDetailedContents of(
    NNCacheCountLogContents aggregate, std::vector<NNCacheCountLogBlock> blocks
  );

  const NNCacheCountLogContents& aggregate() const { return aggregate_; }
  const std::vector<NNCacheCountLogBlock>& blocks() const { return blocks_; }

 private:
  NNCacheCountLogDetailedContents(NNCacheCountLogContents aggregate, std::vector<NNCacheCountLogBlock> blocks);
  NNCacheCountLogContents aggregate_;
  std::vector<NNCacheCountLogBlock> blocks_;
};

// What one appendDump did.
//
// AN EMPTY DELTA APPENDS NOTHING AT ALL -- not a zero-record block. Mirrors the sibling rule
// nncachedump.h states for the evaluation container ("An empty plan appends NOTHING AT ALL
// -- not a zero-entry block"): a delta with no key to report and no unattributed observation
// is nothing owed, and a block written for it would be pure framing bytes recording no fact.
// A second dump with no intervening observation therefore leaves the file byte-identical.
struct NNCacheCountLogAppendResult {
  // Bytes this call added to the file, framing included.
  int64_t bytesAppended;
  // Bytes of torn tail this call had to discard before it could append. Zero on every
  // ordinary append; positive exactly when the previous writer died mid-dump.
  int64_t tornTailBytesDiscarded;
  // Whether this call rewrote the file (a torn-tail repair, or a triggered compaction)
  // rather than appending to it in place. Rewrites are what write volume per dump is
  // bounded by the changed set EXCEPT for.
  bool rewroteTheFile;
};

//-------------------------------------------------------------------------------------
// The log
//-------------------------------------------------------------------------------------

// One context's log file.
//
// THE FILE FORMAT, in full, because the framing is the correctness core.
//
//   file header, 32 bytes, once:
//     [ 0.. 8)  magic, the 8 ASCII bytes "KGCNTLOG"
//     [ 8..12)  format version, u32                  -- 1 today
//     [12..16)  file header bytes, u32               -- 32
//     [16..20)  record bytes, u32                    -- 24
//     [20..24)  block header bytes, u32              -- 32
//     [24..32)  context name hash, u64
//   then zero or more blocks, one per dump:
//     block header, 32 bytes:
//       [ 0.. 4)  block magic, u32
//       [ 4.. 8)  record count, u32
//       [ 8..16)  unattributed observations this dump, u64
//       [16..24)  checksum over the block's records, u64
//       [24..32)  checksum over bytes [0..24) of this header, u64
//     then record count records, 24 bytes each:
//       [ 0.. 8)  key.hash0, u64
//       [ 8..16)  key.hash1, u64
//       [16..20)  observations, u32
//       [20..24)  sessions, u32
//
// Every multi-byte field is little-endian and is packed and unpacked byte by byte. The
// format is therefore a fact about these bytes and not about any compiler's struct layout
// (ADR-0012 P7: a cross-boundary fact has one authoritative definition), and the file
// written by one build is readable by another.
//
// WHY A MAGIC AND A VERSION. The trace format on this programme already needed a v2 once.
// A file whose magic, version, record stride or block header stride is not the one this
// build writes is REFUSED by name, never read on the assumption that the fields happen to
// line up.
//
// WHY THE BLOCK HEADER CARRIES TWO CHECKSUMS. The record count is a length read from a file
// that may have been left half-written by a crash. Trusting it before it is verified is how
// a loader ends up allocating on a number a crash chose. So the header checksums itself
// first, over its own preceding 24 bytes, seeded with the context hash; only then is the
// record count read, and it is bounded a second time against the bytes actually remaining
// in the file before a single record is read.
//
// HOW A TORN TAIL IS DETECTED AND WHAT HAPPENS TO IT. Reading stops at the first block that
// is short, whose header checksum fails, whose record count exceeds the bytes remaining, or
// whose payload checksum fails; everything before it has already been applied and is kept.
// The checksum is an FNV-1a variant with an avalanche finalizer, which absorbs a run of
// zero bytes (the shape a lost page leaves behind) rather than ignoring it the way a plain
// XOR would. It detects corruption; it is not a MAC and defends against nothing adversarial.
//
// THE READER NEVER REPAIRS -- THE WRITER DOES. load() is a pure read and reports the tail as
// a disposition. appendDump scans first and, if the tail is torn, rewrites the intact prefix
// atomically before appending. That is not tidiness: appending after a torn tail would put
// the new block at an offset no loader ever reaches, so every subsequent dump would be
// silently lost while every call reported success.
//
// A BLOCK IS ONE DUMP AND ITS RECORDS ARE INCREMENTS. Merging is addition: a key's observations
// accumulate and its sessions rise by whatever the block says, which is 1 per dump. A
// compacted file holding one block of accumulated sums is therefore observationally
// identical to the log it replaced, and there is no absolute-versus-delta flag to get wrong.
//
// EVERY PUBLIC OPERATION TAKES THE CONTEXT'S CROSS-PROCESS LOCK, shared to read and exclusive
// to write: load takes the SHARED lock, appendDump, compact and compactIfNeeded take the
// EXCLUSIVE one. THE LOCK IS THE CONTEXT'S AND NOT THIS FILE'S -- one <context>.nnlock,
// shared with the evaluation containers of the same context -- because a dump of a context
// writes this log AND its container, and two locks would let one process hold each. See
// NNCacheFileLock (nncachefileformat.h). The lock is acquired and released within each call.
class NNCacheCountLog {
 public:
  // Binds to the log for `context` under `directory`.
  //
  // Throws StringError, naming what failed, if `directory` is not an existing directory or
  // if `context` is not a legal context name. A CONTEXT NAME IS VALIDATED TO A CLOSED
  // ALPHABET -- ASCII letters, digits, '.', '_' and '-', nonempty, at most 128 characters,
  // and neither "." nor ".." -- because it becomes a component of a path, and a path
  // expression is an interpreter with no typed value-carrier to hand a value to. The
  // sanctioned move for that case is a strict validation at the boundary that REFUSES what
  // it cannot honor rather than escaping or rewriting it (ADR-0012, the 2026-07-18
  // interpreter-boundary amendment). Without it a context of ".." writes outside the
  // directory the caller named.
  //
  // Touches no file: it neither creates nor reads the log.
  static NNCacheCountLog forContext(const std::string& directory, const std::string& context);

  const std::string& path() const { return path_; }
  const std::string& context() const { return context_; }
  // The identity this context's header carries and verifies against on every read. Exposed
  // for a report that names it (e.g. nncountsdump) rather than recomputing it a second way.
  uint64_t contextHash() const { return contextHash_; }

  // Appends one dump, then fsyncs, so that when this returns the bytes are on the device
  // rather than in a page cache.
  //
  // TAKES A DELTA, AND THE TYPE IS WHAT SAYS SO. A record is an increment, so what this
  // appends must be what has accrued since the last dump; NNCacheObservationDelta is the type
  // only the delta surface can produce, and handing this an absolute harvest -- or a ledger of
  // retrievals, which is a different currency of the same shape -- is a compile error rather
  // than a silent double count. The reasoning is above that type.
  //
  // Every row's observations is added to that key's running total and every row present adds
  // 1 to that key's sessions. The delta surface omits a key with nothing to say, so a
  // pre-warmed entry a session never asked for contributes no record and its sessions does
  // not rise -- see NNCacheCountRow::sessions, which states that reading in full.
  //
  // EVERY OBSERVED KEY GETS A ROW, INCLUDING ONE OBSERVED EXACTLY ONCE whose evaluation no
  // admission threshold will admit to disk. That is not an oversight in the caller: the count
  // is what a LATER session adds to, so the session that could not yet use it is the session
  // that has to write it. It is the whole mechanism of a seen-twice policy bootstrapping
  // across sessions (ratified spec, ledger rows 1717/1722).
  //
  // A delta whose disposition is NotObserved is REFUSED here, naming that a table with no
  // attached context observes nothing, rather than being written as a dump of zero rows that a
  // later reader would take for "this session asked for nothing".
  //
  // Repairs a torn tail first, if there is one, and reports it in the result. Creates the
  // file, with its header, if it does not exist.
  //
  // Throws StringError if the delta is NotCounted, if any file operation fails, or if a
  // repair would produce a total that does not fit a record.
  NNCacheCountLogAppendResult appendDump(const NNCacheObservationDelta& delta) const;

  // Reads the whole log. A missing file is not an error: it reads as zero rows, Intact --
  // "no dump has happened here yet" is a normal answer.
  //
  // Throws StringError only for a file that exists and is not this format: a bad magic, an
  // unknown version, a stride this build does not write, or another context's hash. Those
  // are an operator pointing at the wrong file, not a crash artifact, and merging them
  // would attribute one context's counts to another.
  [[nodiscard]] NNCacheCountLogContents load() const;

  // load(), but keeping each applied block's own rows separate as well as merged -- for a
  // reader that wants to show dump-by-dump history rather than only the merged totals. See
  // NNCacheCountLogDetailedContents.
  //
  // NO LOCK IS TAKEN, unlike every other public operation here (see the class comment
  // above). Its one caller today is nncountsdump, a storage-side inspection tool that has
  // to remain usable on a mount where the engine-side lock does not bind (an NFS mount with
  // no working flock, say) -- proceeding unlocked with a caveat is the whole point of that
  // tool, not an oversight here. A caller that CAN take the lock is expected to hold its own
  // NNCacheFileLock::overContext(..., Shared) around this call rather than have this call
  // take a second, nested one. Reading against a log an appender is mid-write to still gets
  // the right answer for the reason load() already states about a torn tail: this applies
  // every whole, checksum-verified block and reports the remainder as the tail, never a
  // wrong number for a block it only partially observed.
  //
  // Throws StringError under the same conditions as load().
  [[nodiscard]] NNCacheCountLogDetailedContents loadDetailedUnlocked() const;

  // Rewrites the log as its header plus one block holding the accumulated totals, via a
  // temp file and an atomic rename. Repairs a torn tail as a side effect, since it writes
  // only what load() applied. Returns what it wrote.
  //
  // Throws StringError if an accumulated total exceeds what a record can hold, naming the
  // key -- the record's fields are 32-bit and the largest lifetime count in the operator's
  // own corpus is 11,997, so this is a refusal about an unreachable state rather than a
  // policy about a likely one.
  NNCacheCountLogContents compact() const;

  // Compacts if the file holds more than `liveSetMultiple` times as many records as it has
  // distinct keys. Returns whether it compacted.
  //
  // Throws StringError if liveSetMultiple is below 1: a multiple of zero would compact on
  // every call and a negative one has no reading.
  bool compactIfNeeded(int liveSetMultiple) const;

  // The default multiple. At 4 the file is bounded at about four times the live set and one
  // compaction, which writes one live set, buys three live sets of appends -- about 1.33
  // bytes written per useful byte, amortised.
  static int defaultCompactionMultiple();

  // The format's own constants, named here so a test asserts against the implementation
  // rather than against a second copy of the numbers (ADR-0012 P1).
  static uint32_t formatVersion();
  static size_t fileHeaderBytes();
  static size_t blockHeaderBytes();
  static size_t recordBytes();
  // The exact bytes a dump of `numRows` rows adds to an existing file.
  static int64_t bytesForDumpOf(int64_t numRows);

  // WHAT EVERY COUNT IN THIS LOG MEASURES, as one sentence for a reader to print rather than
  // invent its own wording of. TODAY that is an OBSERVATION: one presentation of the position
  // to the cache under this context -- a would-have-been-computed forward pass -- counted
  // whether a cached evaluation answered it or one had to be computed. This is the one home
  // for that sentence so that a re-denomination of what the log counts changes it once, here,
  // and every reader (nncountsdump among them) picks up the new wording rather than one copy
  // drifting from another -- see the currency confusion this already caused once (ledger rows
  // 1651/1654) and the ratification that settled it (ledger rows 1717/1722).
  static const std::string& countsCurrencyDescription();

 private:
  NNCacheCountLog(std::string path, std::string directory, std::string context, uint64_t contextHash);

  // compact() WITHOUT taking the context lock, for the one caller that already holds it. See
  // NNEvalContainer::compactUnlocked, which exists for the same reason and in the same shape:
  // asking for a second exclusive lock on a context this process already locked would wait out
  // the whole deadline against itself and then throw.
  NNCacheCountLogContents compactUnlocked() const;

  std::string path_;
  // The directory this log, its context's evaluation containers, and their shared lock file all
  // live in. Kept rather than recovered from path_ by stripping ".nncounts": the lock file is a
  // third name in this directory, not a rewriting of this one (ADR-0012 P1).
  std::string directory_;
  std::string context_;
  uint64_t contextHash_;
};

#endif  // NEURALNET_NNCACHECOUNTLOG_H_
