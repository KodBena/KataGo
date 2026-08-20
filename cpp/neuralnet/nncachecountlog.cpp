#include "../neuralnet/nncachecountlog.h"

#include <cstdio>
#include <cstring>
#include <algorithm>
#include <map>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#include <fcntl.h>
#endif

#include "../core/fileutils.h"
#include "../core/global.h"
#include "../neuralnet/nncachefileformat.h"

// The append-only per-(key, context) count log. See nncachecountlog.h for the format and
// for why it is the shape it is; this file is the mechanism.
//
// WHY C STDIO AND NOT std::ofstream, since ADR-0012 P9 asks for a measured reason or a real
// named constraint rather than a habit. The constraint is real and is named: every
// durability claim this format makes rests on fsync, fsync needs the underlying file
// descriptor, and std::ofstream exposes no portable way to obtain one. std::filebuf's
// descriptor is an implementation extension where it exists at all. So the write path uses
// FILE*, held by an RAII wrapper below so that a throw between fopen and fclose cannot leak
// it, and the read path uses FILE* too rather than mixing two I/O stacks over one format.

namespace {

//-------------------------------------------------------------------------------------
// The format's constants, in one place
//-------------------------------------------------------------------------------------

const char FILE_MAGIC[8] = {'K','G','C','N','T','L','O','G'};
const uint32_t FORMAT_VERSION = 1;
const size_t FILE_HEADER_BYTES = 32;
const size_t BLOCK_HEADER_BYTES = 32;
const size_t RECORD_BYTES = 24;
// The bytes of a block header that the header's own checksum covers: everything before the
// checksum field itself.
const size_t BLOCK_HEADER_CHECKED_BYTES = 24;
const uint32_t BLOCK_MAGIC = 0x4247434Bu;  // 'K','G','C','B' little-endian
const int DEFAULT_COMPACTION_MULTIPLE = 4;

//-------------------------------------------------------------------------------------
// The byte layer
//-------------------------------------------------------------------------------------

// The little-endian packing, the checksum and the context-name boundary live in
// nncachefileformat.h and are shared with the evaluation container, because they are one
// fact each and not two (ADR-0012 P1/P7). In particular the CONTEXT HASH must be the same
// function of the same validated name in both formats: an attach joins a context's
// <context>.nncounts and <context>.<model>.nnevals by exactly that identity, and two
// hand-copied hashes would make the join a coincidence rather than a contract.

using NNCacheFileBytes::put32;
using NNCacheFileBytes::put64;
using NNCacheFileBytes::get32;
using NNCacheFileBytes::get64;

uint64_t checksumOf(const uint8_t* data, size_t len, uint64_t seed) {
  return NNCacheFileChecksum::of(data, len, seed);
}

uint64_t contextHashOf(const std::string& context) {
  return NNCacheFileName::hashOf(context);
}

//-------------------------------------------------------------------------------------
// File handling
//-------------------------------------------------------------------------------------

// The RAII file handle and the directory fsync are the evaluation container's too, so they
// live in nncachefileformat.h. The names below are this file's original ones, kept so the
// call sites read as they always did.
using ScopedFile = NNCacheFileHandle;

void syncDirectoryOf(const std::string& path) {
  NNCacheFileSync::directoryOf(path);
}

//-------------------------------------------------------------------------------------
// Header and record encoding
//-------------------------------------------------------------------------------------

void encodeFileHeader(uint8_t* out, uint64_t contextHash) {
  std::memcpy(out, FILE_MAGIC, 8);
  put32(out + 8, FORMAT_VERSION);
  put32(out + 12, (uint32_t)FILE_HEADER_BYTES);
  put32(out + 16, (uint32_t)RECORD_BYTES);
  put32(out + 20, (uint32_t)BLOCK_HEADER_BYTES);
  put64(out + 24, contextHash);
}

// Refuses, naming the field, anything this build did not write. This is the boundary that
// keeps another format, another version, or another context's counts from being merged in
// as if they were this context's (ADR-0012 P2: translate and validate, never coerce).
void verifyFileHeader(const uint8_t* hdr, const std::string& path, const std::string& context, uint64_t contextHash) {
  if(std::memcmp(hdr, FILE_MAGIC, 8) != 0)
    throw StringError("NNCacheCountLog: " + path + " does not begin with the count-log magic; it is not a count log.");
  const uint32_t version = get32(hdr + 8);
  if(version != FORMAT_VERSION)
    throw StringError(
      "NNCacheCountLog: " + path + " is format version " + Global::uint64ToString(version) +
      " and this build reads version " + Global::uint64ToString(FORMAT_VERSION) + " only."
    );
  const uint32_t headerBytes = get32(hdr + 12);
  if(headerBytes != (uint32_t)FILE_HEADER_BYTES)
    throw StringError(
      "NNCacheCountLog: " + path + " declares a " + Global::uint64ToString(headerBytes) +
      "-byte file header; this build writes " + Global::uint64ToString(FILE_HEADER_BYTES) + "."
    );
  const uint32_t recordBytes = get32(hdr + 16);
  if(recordBytes != (uint32_t)RECORD_BYTES)
    throw StringError(
      "NNCacheCountLog: " + path + " declares a " + Global::uint64ToString(recordBytes) +
      "-byte record; this build writes " + Global::uint64ToString(RECORD_BYTES) + "."
    );
  const uint32_t blockHeaderBytes = get32(hdr + 20);
  if(blockHeaderBytes != (uint32_t)BLOCK_HEADER_BYTES)
    throw StringError(
      "NNCacheCountLog: " + path + " declares a " + Global::uint64ToString(blockHeaderBytes) +
      "-byte block header; this build writes " + Global::uint64ToString(BLOCK_HEADER_BYTES) + "."
    );
  const uint64_t storedContext = get64(hdr + 24);
  if(storedContext != contextHash)
    throw StringError(
      "NNCacheCountLog: " + path + " was written for a different context than '" + context +
      "'. Counts from two contexts are not merged."
    );
}

void encodeBlockHeader(uint8_t* out, uint32_t recordCount, uint64_t unattributed, uint64_t payloadChecksum, uint64_t contextHash) {
  put32(out + 0, BLOCK_MAGIC);
  put32(out + 4, recordCount);
  put64(out + 8, unattributed);
  put64(out + 16, payloadChecksum);
  put64(out + 24, checksumOf(out, BLOCK_HEADER_CHECKED_BYTES, contextHash));
}

void encodeRecord(uint8_t* out, Hash128 key, uint32_t lookups, uint32_t sessions) {
  put64(out + 0, key.hash0);
  put64(out + 8, key.hash1);
  put32(out + 16, lookups);
  put32(out + 20, sessions);
}

//-------------------------------------------------------------------------------------
// The scan
//-------------------------------------------------------------------------------------

// One pass over a context's log. Both load() and appendDump() go through this, so there is
// exactly one implementation of "where does the intact part of this file end" (ADR-0012 P1).
struct ScanResult {
  std::vector<NNCacheCountRow> rows;
  int64_t unattributedLookups = 0;
  int64_t blocksApplied = 0;
  int64_t recordsApplied = 0;
  // Byte offset one past the last intact block. Equal to the file size when the tail is
  // intact.
  int64_t intactEndOffset = 0;
  int64_t tornTailBytes = 0;
  bool fileExists = false;
};

ScanResult scanLog(const std::string& path, const std::string& context, uint64_t contextHash) {
  ScanResult result;

  ScopedFile f(path, "rb");
  if(!f.isOpen())
    return result;  // Absent is a normal answer: no dump has happened here yet.
  result.fileExists = true;

  if(std::fseek(f.get(), 0, SEEK_END) != 0)
    throw StringError("NNCacheCountLog: could not seek " + path + ".");
  const long endPos = std::ftell(f.get());
  if(endPos < 0)
    throw StringError("NNCacheCountLog: could not size " + path + ".");
  const int64_t fileSize = (int64_t)endPos;
  std::rewind(f.get());

  // A file too short to hold a header is the torn tail at offset zero: a crash during the
  // very first dump. There is nothing to verify and nothing to keep, so it is reported as
  // discardable rather than refused -- refusing would make a crash indistinguishable from
  // an operator naming the wrong file, which is the one distinction this boundary exists to
  // draw.
  if(fileSize < (int64_t)FILE_HEADER_BYTES) {
    result.tornTailBytes = fileSize;
    result.intactEndOffset = 0;
    return result;
  }

  uint8_t fileHeader[FILE_HEADER_BYTES];
  if(std::fread(fileHeader, 1, FILE_HEADER_BYTES, f.get()) != FILE_HEADER_BYTES)
    throw StringError("NNCacheCountLog: could not read the header of " + path + ".");
  verifyFileHeader(fileHeader, path, context, contextHash);

  int64_t offset = (int64_t)FILE_HEADER_BYTES;
  result.intactEndOffset = offset;

  // Accumulate into a map keyed by the 128-bit key, and keep first-appearance order beside
  // it so the returned rows are stable and a test can assert on them.
  std::map<std::pair<uint64_t,uint64_t>, size_t> indexOfKey;
  std::vector<uint8_t> payload;

  while(true) {
    const int64_t remaining = fileSize - offset;
    if(remaining <= 0)
      break;
    if(remaining < (int64_t)BLOCK_HEADER_BYTES)
      break;  // torn: not even a whole block header left

    uint8_t blockHeader[BLOCK_HEADER_BYTES];
    if(std::fread(blockHeader, 1, BLOCK_HEADER_BYTES, f.get()) != BLOCK_HEADER_BYTES)
      break;  // torn: the file shrank under us, or the read failed

    // The header checksums ITSELF before its record count is believed. Everything after
    // this point trusts a length that came out of a file a crash may have half-written, so
    // this is the check that has to come first.
    if(get64(blockHeader + 24) != checksumOf(blockHeader, BLOCK_HEADER_CHECKED_BYTES, contextHash))
      break;
    if(get32(blockHeader + 0) != BLOCK_MAGIC)
      break;

    const uint32_t recordCount = get32(blockHeader + 4);
    const int64_t payloadBytes = (int64_t)recordCount * (int64_t)RECORD_BYTES;
    // The second bound on the length, against the bytes that actually exist. A header can
    // checksum correctly and still describe a block whose payload never reached the device.
    if(payloadBytes > remaining - (int64_t)BLOCK_HEADER_BYTES)
      break;

    payload.resize((size_t)payloadBytes);
    if(payloadBytes > 0 && std::fread(payload.data(), 1, (size_t)payloadBytes, f.get()) != (size_t)payloadBytes)
      break;
    if(get64(blockHeader + 16) != checksumOf(payload.data(), (size_t)payloadBytes, contextHash))
      break;

    // The block is whole. Apply it -- and only now advance intactEndOffset, so a block that
    // fails any check above leaves the offset pointing at its own first byte.
    for(uint32_t i = 0; i < recordCount; i++) {
      const uint8_t* r = payload.data() + (size_t)i * RECORD_BYTES;
      const Hash128 key(get64(r + 0), get64(r + 8));
      const uint32_t lookups = get32(r + 16);
      const uint32_t sessions = get32(r + 20);
      const std::pair<uint64_t,uint64_t> mapKey(key.hash0, key.hash1);
      const std::map<std::pair<uint64_t,uint64_t>, size_t>::iterator it = indexOfKey.find(mapKey);
      if(it == indexOfKey.end()) {
        indexOfKey[mapKey] = result.rows.size();
        NNCacheCountRow row;
        row.key = key;
        row.lookups = (uint64_t)lookups;
        row.sessions = (uint64_t)sessions;
        result.rows.push_back(row);
      }
      else {
        result.rows[it->second].lookups += (uint64_t)lookups;
        result.rows[it->second].sessions += (uint64_t)sessions;
      }
    }
    result.unattributedLookups += (int64_t)get64(blockHeader + 8);
    result.blocksApplied += 1;
    result.recordsApplied += (int64_t)recordCount;
    offset += (int64_t)BLOCK_HEADER_BYTES + payloadBytes;
    result.intactEndOffset = offset;
  }

  result.tornTailBytes = fileSize - result.intactEndOffset;
  return result;
}

//-------------------------------------------------------------------------------------
// Writing
//-------------------------------------------------------------------------------------

// Encodes one block. Kept separate from the writing so both the append path and the rewrite
// path emit bytes through one function.
std::vector<uint8_t> encodeBlock(
  const std::vector<NNCacheCountRow>& rows,
  int64_t unattributed,
  uint64_t contextHash,
  const std::string& path
) {
  if(rows.size() > 0xFFFFFFFFull)
    throw StringError("NNCacheCountLog: " + path + ": a dump of more than 2^32-1 rows cannot be framed.");
  std::vector<uint8_t> out(BLOCK_HEADER_BYTES + rows.size() * RECORD_BYTES);
  for(size_t i = 0; i < rows.size(); i++) {
    // The record's fields are 32-bit; an accumulated total that will not fit is refused by
    // name rather than wrapped or clamped. The largest lifetime reference count in the
    // operator's own corpus is 11,997, so this refuses an unreachable state -- which is
    // exactly why it can afford to be a refusal rather than a policy.
    if(rows[i].lookups > 0xFFFFFFFFull || rows[i].sessions > 0xFFFFFFFFull)
      throw StringError(
        "NNCacheCountLog: " + path + ": key " + rows[i].key.toString() +
        " has accumulated totals (" + Global::uint64ToString(rows[i].lookups) + " lookups, " +
        Global::uint64ToString(rows[i].sessions) + " sessions) that do not fit a 32-bit record field."
      );
    encodeRecord(out.data() + BLOCK_HEADER_BYTES + i * RECORD_BYTES, rows[i].key,
                 (uint32_t)rows[i].lookups, (uint32_t)rows[i].sessions);
  }
  const uint64_t payloadChecksum = checksumOf(out.data() + BLOCK_HEADER_BYTES, rows.size() * RECORD_BYTES, contextHash);
  encodeBlockHeader(out.data(), (uint32_t)rows.size(), (uint64_t)unattributed, payloadChecksum, contextHash);
  return out;
}

// Writes header + one block to a temp sibling, fsyncs it, renames it over `path`, and
// fsyncs the directory.
//
// THIS IS THE CRASH STORY FOR COMPACTION AND FOR TORN-TAIL REPAIR, and it is why neither
// ever writes in place. rename(2) is atomic: at every instant `path` names either the whole
// old file or the whole new one. A crash before the rename leaves the old file untouched
// and a stale temp that the next rewrite opens with "wb" and overwrites; nothing ever reads
// the temp. A crash after the rename but before the directory fsync is the case the
// directory fsync exists for -- both files' data are durable but the directory entry may
// not be, so without it the rename can be lost.
void rewriteAsOneBlock(
  const std::string& path,
  const std::vector<NNCacheCountRow>& rows,
  int64_t unattributed,
  uint64_t contextHash
) {
  const std::string tempPath = path + ".compacting";
  const std::vector<uint8_t> block = encodeBlock(rows, unattributed, contextHash, path);

  {
    ScopedFile f(tempPath, "wb");
    if(!f.isOpen())
      throw StringError("NNCacheCountLog: could not open " + tempPath + " for writing.");
    uint8_t fileHeader[FILE_HEADER_BYTES];
    encodeFileHeader(fileHeader, contextHash);
    if(std::fwrite(fileHeader, 1, FILE_HEADER_BYTES, f.get()) != FILE_HEADER_BYTES)
      throw StringError("NNCacheCountLog: could not write the header of " + tempPath + ".");
    if(block.size() > 0 && std::fwrite(block.data(), 1, block.size(), f.get()) != block.size())
      throw StringError("NNCacheCountLog: could not write the compacted block of " + tempPath + ".");
    if(!f.flushAndSync())
      throw StringError("NNCacheCountLog: could not fsync " + tempPath + ".");
    if(!f.closeNow())
      throw StringError("NNCacheCountLog: could not close " + tempPath + ".");
  }

  FileUtils::rename(tempPath, path);
  syncDirectoryOf(path);
}

//-------------------------------------------------------------------------------------
// The context name boundary
//-------------------------------------------------------------------------------------

// A context name becomes a path component, so it is validated to a closed alphabet that
// REFUSES what it cannot honor -- never an escape, never a rewrite into something
// acceptable (ADR-0012, the 2026-07-18 interpreter-boundary amendment). The alphabet and
// the refusal live in nncachefileformat.h, shared with the evaluation container, because
// the same context name becomes a component of that file's path too.
void verifyContextName(const std::string& context) {
  NNCacheFileName::verify(context, "NNCacheCountLog", "context name");
}

}  // namespace

//-------------------------------------------------------------------------------------
// NNCacheCountLogContents
//-------------------------------------------------------------------------------------

NNCacheCountLogContents::NNCacheCountLogContents(
  std::vector<NNCacheCountRow> rows,
  int64_t unattributedLookups,
  NNCacheCountLogTail tail,
  int64_t discardedTailBytes,
  int64_t blocksApplied,
  int64_t recordsApplied
)
  :rows_(std::move(rows)),
   unattributedLookups_(unattributedLookups),
   tail_(tail),
   discardedTailBytes_(discardedTailBytes),
   blocksApplied_(blocksApplied),
   recordsApplied_(recordsApplied)
{}

std::string NNCacheLookupThreshold::describe() const {
  if(lookups_ == 0)
    return "every key, counted or not";
  return "keys with at least " + Global::uint64ToString(lookups_) + " recorded lookups";
}

NNCacheCountLogContents NNCacheCountLogContents::of(
  std::vector<NNCacheCountRow> rows,
  int64_t unattributedLookups,
  NNCacheCountLogTail tail,
  int64_t discardedTailBytes,
  int64_t blocksApplied,
  int64_t recordsApplied
) {
  // The disposition and the byte count are coupled at construction, so "Truncated with
  // nothing discarded" is not a value a reader has to defend against (ADR-0012 P11: the
  // typed reason is TIED to the absence by a check, not merely written down beside it).
  const bool truncated = tail == NNCacheCountLogTail::Truncated;
  if(truncated != (discardedTailBytes > 0))
    throw StringError(
      "NNCacheCountLogContents: tail disposition and discarded byte count disagree -- " +
      std::string(truncated ? "Truncated" : "Intact") + " with " +
      Global::int64ToString(discardedTailBytes) + " discarded bytes."
    );
  if(discardedTailBytes < 0 || blocksApplied < 0 || recordsApplied < 0 || unattributedLookups < 0)
    throw StringError("NNCacheCountLogContents: a count was negative.");
  return NNCacheCountLogContents(
    std::move(rows), unattributedLookups, tail, discardedTailBytes, blocksApplied, recordsApplied);
}

std::vector<NNCacheCountRow> NNCacheCountLogContents::byDescendingLookups() const {
  std::vector<NNCacheCountRow> sorted = rows_;
  std::sort(sorted.begin(), sorted.end(), [](const NNCacheCountRow& a, const NNCacheCountRow& b) {
    if(a.lookups != b.lookups)
      return a.lookups > b.lookups;
    // Ties broken by the key so the order is total and a test can assert it. Nothing here
    // consults sessions: ordering is by lookups, which is the operator's ruling.
    if(a.key.hash0 != b.key.hash0)
      return a.key.hash0 < b.key.hash0;
    return a.key.hash1 < b.key.hash1;
  });
  return sorted;
}

//-------------------------------------------------------------------------------------
// NNCacheCountLog
//-------------------------------------------------------------------------------------

NNCacheCountLog::NNCacheCountLog(std::string path, std::string context, uint64_t contextHash)
  :path_(std::move(path)), context_(std::move(context)), contextHash_(contextHash)
{}

NNCacheCountLog NNCacheCountLog::forContext(const std::string& directory, const std::string& context) {
  verifyContextName(context);
  if(!FileUtils::isDirectory(directory))
    throw StringError("NNCacheCountLog: '" + directory + "' is not an existing directory.");
  const std::string path = directory + "/" + context + ".nncounts";
  return NNCacheCountLog(path, context, contextHashOf(context));
}

uint32_t NNCacheCountLog::formatVersion() { return FORMAT_VERSION; }
size_t NNCacheCountLog::fileHeaderBytes() { return FILE_HEADER_BYTES; }
size_t NNCacheCountLog::blockHeaderBytes() { return BLOCK_HEADER_BYTES; }
size_t NNCacheCountLog::recordBytes() { return RECORD_BYTES; }
int NNCacheCountLog::defaultCompactionMultiple() { return DEFAULT_COMPACTION_MULTIPLE; }

int64_t NNCacheCountLog::bytesForDumpOf(int64_t numRows) {
  return (int64_t)BLOCK_HEADER_BYTES + numRows * (int64_t)RECORD_BYTES;
}

NNCacheCountLogContents NNCacheCountLog::load() const {
  const ScanResult scan = scanLog(path_, context_, contextHash_);
  return NNCacheCountLogContents::of(
    scan.rows,
    scan.unattributedLookups,
    scan.tornTailBytes > 0 ? NNCacheCountLogTail::Truncated : NNCacheCountLogTail::Intact,
    scan.tornTailBytes,
    scan.blocksApplied,
    scan.recordsApplied
  );
}

NNCacheCountLogContents NNCacheCountLog::compact() const {
  const ScanResult scan = scanLog(path_, context_, contextHash_);
  rewriteAsOneBlock(path_, scan.rows, scan.unattributedLookups, contextHash_);
  return NNCacheCountLogContents::of(
    scan.rows,
    scan.unattributedLookups,
    NNCacheCountLogTail::Intact,
    0,
    1,
    (int64_t)scan.rows.size()
  );
}

bool NNCacheCountLog::compactIfNeeded(int liveSetMultiple) const {
  if(liveSetMultiple < 1)
    throw StringError(
      "NNCacheCountLog: a compaction multiple of " + Global::intToString(liveSetMultiple) +
      " has no reading; it must be at least 1."
    );
  const ScanResult scan = scanLog(path_, context_, contextHash_);
  const int64_t liveSet = (int64_t)scan.rows.size();
  // A torn tail is repaired whether or not the size trigger fires: leaving it would make
  // the next append land at an offset no loader reaches.
  const bool torn = scan.tornTailBytes > 0;
  const bool overMultiple = liveSet > 0 && scan.recordsApplied > (int64_t)liveSetMultiple * liveSet;
  if(!torn && !overMultiple)
    return false;
  // DELEGATES rather than rewriting here. "Rewrite this log" has exactly one home; a
  // second call to rewriteAsOneBlock at this site would be a second statement of the same
  // act, and a change to either one would silently leave the other saying the old thing.
  // That is not hypothetical: a seen-red leg that mutated compact() left this path green
  // while it still carried its own copy. The cost is one extra scan on the compacting
  // path -- a read of at most a few megabytes, on an explicitly-invoked dump.
  const NNCacheCountLogContents written = compact();
  (void)written;
  return true;
}

NNCacheHitCountDelta::NNCacheHitCountDelta(NNCacheHitLedger ledger)
  :ledger_(std::move(ledger))
{}

NNCacheHitCountDelta NNCacheHitCountDelta::take(NNCacheTable& table) {
  return NNCacheHitCountDelta(table.takeUnpersistedHitCounts());
}

// The same act restricted to one context, through the one door that can perform it. It is a
// second FACTORY and not a second KIND: what it produces is the same delta type, so a
// per-context dump reaches appendDump by the same barrier a whole-table one does and there is
// still no expression that converts an absolute harvest into either.
NNCacheHitCountDelta NNCacheHitCountDelta::takeFor(NNCacheTable& table, const NNCacheContextId& context) {
  return NNCacheHitCountDelta(table.takeUnpersistedHitCountsFor(context));
}

NNCacheHitCountDelta NNCacheHitCountDelta::ofDeltaRows(std::vector<NNCacheHitCount> rows, int64_t unrecordedHits) {
  return NNCacheHitCountDelta(NNCacheHitLedger::counted(std::move(rows), unrecordedHits));
}

NNCacheHitCountDelta NNCacheHitCountDelta::notCounted() {
  return NNCacheHitCountDelta(NNCacheHitLedger::notCounted());
}

NNCacheCountLogAppendResult NNCacheCountLog::appendDump(const NNCacheHitCountDelta& delta) const {
  const NNCacheHitLedger& ledger = delta.ledger();
  // A NotCounted delta is refused here rather than written as a dump of zero rows. That is
  // the entire reason the disposition is typed: "this table keeps no counts" and "this
  // session hit nothing" are different facts, and persisting the first as the second would
  // quietly record that a whole session found nothing worth caching (ADR-0002).
  if(!ledger.isCounted())
    throw StringError(
      "NNCacheCountLog: " + path_ + ": the table reported NotCounted, so there are no per-key "
      "counts to persist. A single-level table keeps none; counting is a property of the "
      "two-level strategy."
    );

  NNCacheCountLogAppendResult result;
  result.bytesAppended = 0;
  result.tornTailBytesDiscarded = 0;
  result.rewroteTheFile = false;

  // Scan first. A torn tail must be repaired BEFORE anything is appended: an append past a
  // torn tail lands at an offset no loader ever reaches, so every subsequent dump would be
  // silently lost while every call reported success.
  const ScanResult scan = scanLog(path_, context_, contextHash_);
  if(scan.tornTailBytes > 0) {
    rewriteAsOneBlock(path_, scan.rows, scan.unattributedLookups, contextHash_);
    result.tornTailBytesDiscarded = scan.tornTailBytes;
    result.rewroteTheFile = true;
  }

  // Every row the delta carries, including a row whose hits are zero. The delta surface
  // already omits a key with nothing to say, so a zero row here is a key the caller
  // deliberately has something to say about, and dropping it would make "never asked for"
  // indistinguishable from "not present".
  const std::vector<NNCacheHitCount>& entries = ledger.entries();
  std::vector<NNCacheCountRow> rows;
  rows.reserve(entries.size());
  for(size_t i = 0; i < entries.size(); i++) {
    NNCacheCountRow row;
    row.key = entries[i].key;
    row.lookups = (uint64_t)entries[i].hits;
    row.sessions = 1;  // one dump, one session credited per key present in it
    rows.push_back(row);
  }
  const std::vector<uint8_t> block = encodeBlock(rows, ledger.unrecordedHits(), contextHash_, path_);

  const bool fileExists = FileUtils::exists(path_);
  ScopedFile f(path_, fileExists ? "ab" : "wb");
  if(!f.isOpen())
    throw StringError("NNCacheCountLog: could not open " + path_ + " for appending.");
  if(!fileExists) {
    uint8_t fileHeader[FILE_HEADER_BYTES];
    encodeFileHeader(fileHeader, contextHash_);
    if(std::fwrite(fileHeader, 1, FILE_HEADER_BYTES, f.get()) != FILE_HEADER_BYTES)
      throw StringError("NNCacheCountLog: could not write the header of " + path_ + ".");
    result.bytesAppended += (int64_t)FILE_HEADER_BYTES;
  }
  if(std::fwrite(block.data(), 1, block.size(), f.get()) != block.size())
    throw StringError("NNCacheCountLog: could not append a block to " + path_ + ".");
  result.bytesAppended += (int64_t)block.size();
  if(!f.flushAndSync())
    throw StringError("NNCacheCountLog: could not fsync " + path_ + ".");
  // A newly created file needs its directory entry forced too; an append to an existing one
  // does not, because the entry is already durable.
  if(!fileExists)
    syncDirectoryOf(path_);
  return result;
}
