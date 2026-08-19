#ifndef NEURALNET_NNCACHECOUNTLOG_H_
#define NEURALNET_NNCACHECOUNTLOG_H_

#include <cstdint>
#include <string>
#include <vector>

#include "../core/hash.h"
#include "../neuralnet/nncache.h"

// An APPEND-ONLY LOG of per-(key, context) hit counts, with no dependency on anything
// outside the C and C++ standard libraries.
//
// WHAT IT IS FOR, AND WHAT IT IS NOT FOR. It persists COUNTS: for each 128-bit position
// hash, how many times an evaluation was retrieved for it, and in how many dumps it
// appeared. It does not persist evaluations. There is no NNOutput in this header, no
// payload container, and no schema for one -- persisting evaluations is a separate piece
// of work, and a container format invented here in passing would be that work done badly.
//
// WRITES HAPPEN ONLY ON AN EXPLICIT DUMP. Nothing in this file is reachable from the MCTS
// hot path, from NNCacheTable::get or from NNCacheTable::set. appendDump reads the finished
// per-key surface a table already keeps (NNCacheTable::harvestHitCounts) and writes it once.
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
// has something to say about, plus 32 bytes of framing for the whole dump. At the working
// figure of ~40,000 entries per context that is ~960 KB.
//
// CRASH SAFETY IS APPEND-PLUS-FSYNC AND NOTHING MORE, so the framing has to make a torn
// tail detectable. See the format description above NNCacheCountLog.

//-------------------------------------------------------------------------------------
// The values
//-------------------------------------------------------------------------------------

// One key's accumulated totals across every dump the log holds.
//
// The totals are 64-bit here and 32-bit on the wire, deliberately: a single dump's count
// comes from NNCacheHitCount::hits, which is a uint32_t, but a sum across dumps has no such
// bound and silently wrapping it would be the worst kind of quiet failure. Compaction,
// which is the only operation that has to fit an accumulated total back into a record,
// refuses loudly and names the key if one will not fit (ADR-0002).
struct NNCacheCountRow {
  Hash128 key;
  // Total retrievals recorded for this key, summed over every dump in the log.
  uint64_t lookups;
  // How many DUMPS this key appeared in. Named "sessions" in the operator's own schema and
  // kept under that name, but the honest definition is the one stated here: if an operator
  // dumps twice inside one engine run, a key present in both rises by two. The log has no
  // way to tell two dumps in one run from two runs and does not pretend to.
  uint64_t sessions;
};

// A MINIMUM RECORDED-LOOKUP COUNT: the one home of the rule "this key has been seen often
// enough", for every side that asks it.
//
// WHY THIS IS A TYPE AND NOT A uint64_t PASSED AROUND. Two independent surfaces apply the
// same rule to the same fact. The READ side is NNCacheLevelZeroBound::minLookups, which
// decides what an attach admits into a frozen level 0. The WRITE side is
// NNCacheDiskAdmission::minLookups, which decides what a dump lets onto disk. They are
// different decisions -- one is a prefix of a ranked order, the other a per-entry
// predicate -- but the QUESTION they ask of a key is one question, and a key's recorded
// lookups are one fact with one home, this log. Written out twice, the two comparisons
// could drift in the one place drifting is silent: the boundary case below (ADR-0012 P1).
//
// THE BOUNDARY CASE, stated once here so neither side restates it. A key the log has never
// mentioned carries a recorded count of zero, so every threshold ABOVE zero excludes it and
// a threshold OF zero admits it -- which is right in both directions: "admit everything" must
// not turn into "admit everything the log happens to know about". An uncounted key is not
// given a separate check, because there is nothing for one to catch; see
// NNCacheLevelZeroBound::select, where the same reasoning is already recorded against the
// read side's own comparison.
class NNCacheLookupThreshold {
 public:
  // Admits a key recorded at `lookups` retrievals or more. of(0) admits every key, counted
  // or not.
  static NNCacheLookupThreshold of(uint64_t lookups) { return NNCacheLookupThreshold(lookups); }

  [[nodiscard]] uint64_t lookups() const { return lookups_; }

  // Whether a key the log records `recordedLookups` retrievals for clears this threshold.
  // A key the log does not mention is passed as zero, by the rule above.
  [[nodiscard]] bool admits(uint64_t recordedLookups) const { return recordedLookups >= lookups_; }

  // For a report or a refusal: what was asked for, in words, without a caller re-deriving
  // it from the number.
  [[nodiscard]] std::string describe() const;

 private:
  explicit NNCacheLookupThreshold(uint64_t lookups) : lookups_(lookups) {}
  uint64_t lookups_;
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
    int64_t unattributedLookups,
    NNCacheCountLogTail tail,
    int64_t discardedTailBytes,
    int64_t blocksApplied,
    int64_t recordsApplied
  );

  // One row per distinct key, in first-appearance order. Not sorted: see
  // byDescendingLookups.
  const std::vector<NNCacheCountRow>& rows() const { return rows_; }

  // Lookups that happened and could not be attributed to any key, summed over every applied
  // block. This is NNCacheHitLedger::unrecordedHits carried through the log rather than
  // dropped at the door: a harvest that was short says by how much, and so does the log
  // that stored it (ADR-0002). Zero in every ordinary run.
  int64_t unattributedLookups() const { return unattributedLookups_; }

  NNCacheCountLogTail tail() const { return tail_; }
  // Bytes after the last intact block. Positive exactly when tail() is Truncated.
  int64_t discardedTailBytes() const { return discardedTailBytes_; }
  // How many blocks were applied. Zero means the file was absent or held only its header.
  int64_t blocksApplied() const { return blocksApplied_; }
  // How many records were applied, counting a key once per block it appeared in. This is
  // the log's physical size in records; rows().size() is its live set. Their ratio is what
  // the compaction trigger reads.
  int64_t recordsApplied() const { return recordsApplied_; }

  // The rows in descending lookups, ties broken by key so the order is total and a test can
  // assert it.
  //
  // ORDERING IS BY LOOKUPS. That is the operator's ruling for what a frozen level 0 should
  // be built in the order of, and it is why this is the only ordering helper here. Nothing
  // in this file ranks, weights or prefers sessions; sessions is stored because it is cheap
  // to store and is a fact he asked for, not because anything here treats it as the better
  // predictor.
  [[nodiscard]] std::vector<NNCacheCountRow> byDescendingLookups() const;

 private:
  NNCacheCountLogContents(
    std::vector<NNCacheCountRow> rows,
    int64_t unattributedLookups,
    NNCacheCountLogTail tail,
    int64_t discardedTailBytes,
    int64_t blocksApplied,
    int64_t recordsApplied
  );

  std::vector<NNCacheCountRow> rows_;
  int64_t unattributedLookups_;
  NNCacheCountLogTail tail_;
  int64_t discardedTailBytes_;
  int64_t blocksApplied_;
  int64_t recordsApplied_;
};

// What one appendDump did.
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
//       [ 8..16)  unattributed lookups this dump, u64
//       [16..24)  checksum over the block's records, u64
//       [24..32)  checksum over bytes [0..24) of this header, u64
//     then record count records, 24 bytes each:
//       [ 0.. 8)  key.hash0, u64
//       [ 8..16)  key.hash1, u64
//       [16..20)  lookups, u32
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
// A BLOCK IS ONE DUMP AND ITS RECORDS ARE INCREMENTS. Merging is addition: a key's lookups
// accumulate and its sessions rise by whatever the block says, which is 1 per dump. A
// compacted file holding one block of accumulated sums is therefore observationally
// identical to the log it replaced, and there is no absolute-versus-delta flag to get wrong.
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

  // Appends one dump, then fsyncs, so that when this returns the bytes are on the device
  // rather than in a page cache.
  //
  // Takes the LEDGER rather than a row vector, because the ledger's disposition is the
  // whole point of that type: a NotCounted ledger is REFUSED here, naming that a
  // single-level table keeps no per-key counts, rather than being written as a dump of zero
  // rows that a later reader would take for "this session hit nothing". Every row's lookups
  // is added to that key's total and every row present contributes 1 to its sessions,
  // including rows whose hits are zero -- a pre-warmed entry that earned nothing this
  // session is exactly the fact that says to stop carrying it.
  //
  // Repairs a torn tail first, if there is one, and reports it in the result. Creates the
  // file, with its header, if it does not exist.
  //
  // Throws StringError if the ledger is NotCounted, if any file operation fails, or if a
  // repair would produce a total that does not fit a record.
  NNCacheCountLogAppendResult appendDump(const NNCacheHitLedger& ledger) const;

  // Reads the whole log. A missing file is not an error: it reads as zero rows, Intact --
  // "no dump has happened here yet" is a normal answer.
  //
  // Throws StringError only for a file that exists and is not this format: a bad magic, an
  // unknown version, a stride this build does not write, or another context's hash. Those
  // are an operator pointing at the wrong file, not a crash artifact, and merging them
  // would attribute one context's counts to another.
  [[nodiscard]] NNCacheCountLogContents load() const;

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

 private:
  NNCacheCountLog(std::string path, std::string context, uint64_t contextHash);

  std::string path_;
  std::string context_;
  uint64_t contextHash_;
};

#endif  // NEURALNET_NNCACHECOUNTLOG_H_
