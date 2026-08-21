#include "../tests/tests.h"

#include <cstdio>
#include <iostream>
#include <vector>

#include <ghc/filesystem.hpp>

#include "../core/fileutils.h"
#include "../core/rand.h"
#include "../neuralnet/nncachefileformat.h"
#include "../neuralnet/nncache.h"
#include "../neuralnet/nncachecountlog.h"

using namespace std;
using namespace TestCommon;

namespace gfs = ghc::filesystem;

// Correctness tests for the append-only per-(key, context) count log.
//
// Everything asserted here is a LOGIC INVARIANT and is asserted exactly, with no tolerance
// (ADR-0009, Calibration): a count written is the count read back, a key is present or it
// is not, a byte count is the byte count. Nothing here is a measurement -- the write-volume
// measurement is a separate subcommand, because a number that means nothing unless the box
// is idle does not belong in a pass/fail suite.
//
// The load-bearing test is testCountLogTornTailIsDiscardedAndThePrefixSurvives. A test that
// only round-trips a clean file does not test the thing this format exists for.

namespace {

//-------------------------------------------------------------------------------------
// Fixtures
//-------------------------------------------------------------------------------------

// A distinct key per serial, spread over both halves.
Hash128 nthKey(int serial) {
  return Hash128(
    ((uint64_t)(serial + 1)) * 0x9E3779B97F4A7C15ULL,
    ((uint64_t)(serial + 1)) * 0xD6E8FEB86659FD93ULL + 0x1234567ULL
  );
}

// The rows of one DUMP, as the delta type appendDump takes. These are hand-built deltas --
// what a session accrued since its last dump -- which is what ofDeltaRows is named for.
NNCacheObservationDelta deltaOf(const vector<pair<int,uint32_t>>& serialAndObservations, int64_t unrecorded) {
  vector<NNCacheObservationCount> rows;
  for(size_t i = 0; i < serialAndObservations.size(); i++) {
    NNCacheObservationCount row;
    row.key = nthKey(serialAndObservations[i].first);
    row.observations = serialAndObservations[i].second;
    rows.push_back(row);
  }
  return NNCacheObservationDelta::ofDeltaRows(std::move(rows), unrecorded);
}

// The one row for `serial`. Asserts there is exactly one -- so "one row per key" is
// checked on every read rather than once -- and asserts it is present, so a caller reads
// the count off a reference and never dereferences a maybe-null pointer.
const NNCacheCountRow& rowFor(const NNCacheCountLogContents& contents, int serial) {
  const Hash128 key = nthKey(serial);
  const NNCacheCountRow* found = NULL;
  for(size_t i = 0; i < contents.rows().size(); i++) {
    if(contents.rows()[i].key == key) {
      testAssert(found == NULL);
      found = &contents.rows()[i];
    }
  }
  testAssert(found != NULL);
  return *found;
}

// Whether the log holds this key at all. Separate from rowFor because the two are
// different questions and an absence assertion should not go through an accessor that
// asserts presence.
bool hasRowFor(const NNCacheCountLogContents& contents, int serial) {
  const Hash128 key = nthKey(serial);
  for(size_t i = 0; i < contents.rows().size(); i++) {
    if(contents.rows()[i].key == key)
      return true;
  }
  return false;
}

// The self-removing directory these tests write their logs into is TestCommon::ScopedTempDir,
// which moved there when a second test in this suite needed one. This is the one home of the
// name its directories carry.
const char* const TMP_DIR_PREFIX = "tmpnncachecountlog";

// True if load() refused this file. A named helper rather than a bare call in a try, so
// the [[nodiscard]] on load() stays honest instead of being waived at each site.
bool loadIsRefused(const NNCacheCountLog& log) {
  try {
    const NNCacheCountLogContents contents = log.load();
    (void)contents;
    return false;
  }
  catch(const StringError&) {
    return true;
  }
}

int64_t sizeOf(const string& path) {
  return (int64_t)gfs::file_size(gfs::u8path(path));
}

void truncateTo(const string& path, int64_t bytes) {
  gfs::resize_file(gfs::u8path(path), (uintmax_t)bytes);
}

// Overwrites `count` bytes at `offset`. Used to corrupt a block in place, which is the
// failure a length-only framing cannot see and a checksum can.
void overwriteBytesAt(const string& path, int64_t offset, const vector<uint8_t>& bytes) {
  FILE* f = fopen(path.c_str(), "r+b");
  testAssert(f != NULL);
  testAssert(fseek(f, (long)offset, SEEK_SET) == 0);
  testAssert(fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size());
  fclose(f);
}

// The first `count` bytes of a file, for an assertion that a repair left them alone.
vector<uint8_t> readBytesAt(const string& path, int64_t offset, size_t count) {
  vector<uint8_t> bytes(count);
  FILE* f = fopen(path.c_str(), "rb");
  testAssert(f != NULL);
  testAssert(fseek(f, (long)offset, SEEK_SET) == 0);
  testAssert(count == 0 || fread(bytes.data(), 1, count, f) == count);
  fclose(f);
  return bytes;
}

//-------------------------------------------------------------------------------------
// The tests
//-------------------------------------------------------------------------------------

// Counts written are counts read back. Exactly -- this is a logic invariant, so there is no
// tolerance and no approximate comparison anywhere in it.
void testCountLogRoundTripsCountsExactly() {
  TestCommon::ScopedTempDir dir(TMP_DIR_PREFIX);
  const NNCacheCountLog log = NNCacheCountLog::forContext(dir.path(), "roundtrip");

  const NNCacheCountLogAppendResult appended = log.appendDump(deltaOf({{1, 7}, {2, 0}, {3, 4000000000u}}, 0));
  testAssert(appended.tornTailBytesDiscarded == 0);
  testAssert(appended.tailRepair == NNCacheFileTailRepair::NotNeeded);
  // A first dump writes the file header too; the block itself is the framing plus the rows.
  testAssert(appended.bytesAppended ==
             (int64_t)NNCacheCountLog::fileHeaderBytes() + NNCacheCountLog::bytesForDumpOf(3));
  testAssert(sizeOf(log.path()) == appended.bytesAppended);

  const NNCacheCountLogContents contents = log.load();
  testAssert(contents.tail() == NNCacheCountLogTail::Intact);
  testAssert(contents.discardedTailBytes() == 0);
  testAssert(contents.blocksApplied() == 1);
  testAssert(contents.recordsApplied() == 3);
  testAssert(contents.rows().size() == 3);
  testAssert(contents.unattributedObservations() == 0);

  testAssert(rowFor(contents, 1).observations == 7);
  // A pre-warmed entry that earned nothing is stored, not dropped: "level 0 held this and
  // nobody asked for it" is the signal to stop carrying it, and it is not the same fact as
  // "this key is not in the log".
  testAssert(rowFor(contents, 2).observations == 0);
  // The top of the 32-bit range round-trips, so the record's field width is exercised
  // rather than assumed.
  testAssert(rowFor(contents, 3).observations == 4000000000ull);
  for(size_t i = 0; i < contents.rows().size(); i++)
    testAssert(contents.rows()[i].sessions == 1);
}

// Merging is addition, and sessions counts the dumps a key appeared in.
void testCountLogAccumulatesAcrossDumps() {
  TestCommon::ScopedTempDir dir(TMP_DIR_PREFIX);
  const NNCacheCountLog log = NNCacheCountLog::forContext(dir.path(), "accumulate");

  log.appendDump(deltaOf({{1, 3}, {2, 5}}, 0));
  log.appendDump(deltaOf({{1, 4}, {3, 9}}, 2));
  log.appendDump(deltaOf({{1, 1}}, 3));

  const NNCacheCountLogContents contents = log.load();
  testAssert(contents.blocksApplied() == 3);
  testAssert(contents.recordsApplied() == 5);
  testAssert(contents.rows().size() == 3);
  testAssert(rowFor(contents, 1).observations == 8 && rowFor(contents, 1).sessions == 3);
  testAssert(rowFor(contents, 2).observations == 5 && rowFor(contents, 2).sessions == 1);
  testAssert(rowFor(contents, 3).observations == 9 && rowFor(contents, 3).sessions == 1);
  // Hits the cache could not attribute to any key are carried through the log rather than
  // dropped at the door, and they sum across dumps like everything else.
  testAssert(contents.unattributedObservations() == 5);
}

// A dump with nothing to say -- no key, no unattributed observation -- appends NOTHING, not
// a zero-record block: mirrors the sibling rule NNEvalContainer's dump states for itself
// (nncachedump.h). A no-traffic interval/shutdown dumper must not grow this file forever.
void testCountLogEmptyDumpAppendsNothing() {
  TestCommon::ScopedTempDir dir(TMP_DIR_PREFIX);
  const NNCacheCountLog log = NNCacheCountLog::forContext(dir.path(), "empty");

  // The very first dump of a context is empty: the file must not even be created.
  const NNCacheCountLogAppendResult firstAppend = log.appendDump(deltaOf({}, 0));
  testAssert(firstAppend.bytesAppended == 0);
  testAssert(firstAppend.tornTailBytesDiscarded == 0);
  testAssert(firstAppend.tailRepair == NNCacheFileTailRepair::NotNeeded);
  testAssert(!FileUtils::exists(log.path()));

  // A real dump creates the file and writes one block ...
  log.appendDump(deltaOf({{1, 7}}, 0));
  const int64_t sizeAfterOneDump = sizeOf(log.path());

  // ... and a second, empty dump leaves it byte-identical.
  const NNCacheCountLogAppendResult secondAppend = log.appendDump(deltaOf({}, 0));
  testAssert(secondAppend.bytesAppended == 0);
  testAssert(secondAppend.tornTailBytesDiscarded == 0);
  testAssert(secondAppend.tailRepair == NNCacheFileTailRepair::NotNeeded);
  testAssert(sizeOf(log.path()) == sizeAfterOneDump);

  const NNCacheCountLogContents contents = log.load();
  testAssert(contents.blocksApplied() == 1);
  testAssert(rowFor(contents, 1).observations == 7);

  // An empty delta that carries an unattributed-observation count IS something to say, and
  // is not suppressed: the block is written even though it has no per-key record.
  const NNCacheCountLogAppendResult unattributedOnly = log.appendDump(deltaOf({}, 4));
  testAssert(unattributedOnly.bytesAppended > 0);
  const NNCacheCountLogContents contents2 = log.load();
  testAssert(contents2.blocksApplied() == 2);
  testAssert(contents2.unattributedObservations() == 4);
}

// THE LOAD-BEARING TEST. A crash mid-dump leaves a partial block. Exactly the dumps that
// completed must survive, and the partial one must not.
void testCountLogTornTailIsDiscardedAndThePrefixSurvives() {
  TestCommon::ScopedTempDir dir(TMP_DIR_PREFIX);
  const NNCacheCountLog log = NNCacheCountLog::forContext(dir.path(), "torn");

  log.appendDump(deltaOf({{1, 10}, {2, 20}}, 0));
  log.appendDump(deltaOf({{1, 5}, {3, 30}}, 0));
  const int64_t sizeAfterTwoDumps = sizeOf(log.path());

  // The third dump names a key no earlier dump did, so its survival or absence is directly
  // observable rather than hidden inside a sum (ADR-0021 Rule 1: observe the property at
  // the site of the claim).
  log.appendDump(deltaOf({{1, 100}, {4, 400}, {5, 500}}, 7));
  const int64_t sizeAfterThreeDumps = sizeOf(log.path());
  testAssert(sizeAfterThreeDumps == sizeAfterTwoDumps + NNCacheCountLog::bytesForDumpOf(3));

  // Cut the file part-way through the third block's THIRD record: not on a record boundary
  // and not on a block boundary, which is what a process dying part-way through a write(2)
  // leaves behind.
  //
  // The cut falls where it does deliberately. The third block's first TWO records -- key 1
  // and key 4 -- sit COMPLETE inside the region that must be discarded. So an
  // implementation that framed by length alone, or that took whatever whole records
  // happened to fit, would apply them, and the assertions below would see it. Cutting
  // inside the first record instead would have made key 4's absence an accident of where
  // the byte landed rather than an observation of the property (ADR-0021 Rule 1).
  const int64_t cutAt = sizeAfterTwoDumps + (int64_t)NNCacheCountLog::blockHeaderBytes() +
                        2 * (int64_t)NNCacheCountLog::recordBytes() + 9;
  truncateTo(log.path(), cutAt);
  testAssert(sizeOf(log.path()) == cutAt);

  const NNCacheCountLogContents contents = log.load();

  // THE HEADLINE CLAIM FIRST, so a defect that applies part of a torn block goes red on the
  // count being wrong rather than on the tail's byte arithmetic.
  //
  // Exactly the two completed dumps survive, with exact totals. Key 1's 100 hits from the
  // torn block are not in its total: a partial block is not applied in part.
  testAssert(rowFor(contents, 1).observations == 15 && rowFor(contents, 1).sessions == 2);
  testAssert(rowFor(contents, 2).observations == 20 && rowFor(contents, 2).sessions == 1);
  testAssert(rowFor(contents, 3).observations == 30 && rowFor(contents, 3).sessions == 1);

  // AND THE PARTIAL DUMP DID NOT SURVIVE, observed positively: keys 4 and 5 were named ONLY
  // by the torn block, so their absence is a direct membership fact about the rows the load
  // returned. Key 4's record is whole in the file and is discarded anyway, because the
  // BLOCK it belongs to is not.
  testAssert(!hasRowFor(contents, 4));
  testAssert(!hasRowFor(contents, 5));
  testAssert(contents.rows().size() == 3);
  // The torn block's unattributed count went with it.
  testAssert(contents.unattributedObservations() == 0);

  // Then the accounting: the disposition is Truncated and names exactly how many bytes it
  // will not use.
  testAssert(contents.tail() == NNCacheCountLogTail::Truncated);
  testAssert(contents.discardedTailBytes() == cutAt - sizeAfterTwoDumps);
  testAssert(contents.blocksApplied() == 2);
  testAssert(contents.recordsApplied() == 4);
}

// A block that is the RIGHT LENGTH but wrong in its bytes -- the shape a lost page leaves,
// which a length-only framing cannot see at all.
void testCountLogAWholeButCorruptBlockIsRejectedByItsChecksum() {
  TestCommon::ScopedTempDir dir(TMP_DIR_PREFIX);
  const NNCacheCountLog log = NNCacheCountLog::forContext(dir.path(), "corrupt");

  log.appendDump(deltaOf({{1, 11}}, 0));
  const int64_t sizeAfterOneDump = sizeOf(log.path());
  log.appendDump(deltaOf({{2, 22}}, 0));

  // Zero out eight bytes inside the SECOND block's single record: the file length is
  // unchanged, every offset still lines up, and only the payload checksum can tell.
  const int64_t recordAt = sizeAfterOneDump + (int64_t)NNCacheCountLog::blockHeaderBytes();
  overwriteBytesAt(log.path(), recordAt, vector<uint8_t>(8, 0));

  const NNCacheCountLogContents contents = log.load();
  testAssert(contents.tail() == NNCacheCountLogTail::Truncated);
  testAssert(contents.discardedTailBytes() == sizeOf(log.path()) - sizeAfterOneDump);
  testAssert(contents.blocksApplied() == 1);
  testAssert(rowFor(contents, 1).observations == 11);
  testAssert(!hasRowFor(contents, 2));
}

// A corrupt block HEADER must be caught by the header's own checksum before its record
// count is believed, so a length a crash chose never reaches an allocation.
void testCountLogACorruptBlockHeaderIsRejectedBeforeItsLengthIsBelieved() {
  TestCommon::ScopedTempDir dir(TMP_DIR_PREFIX);
  const NNCacheCountLog log = NNCacheCountLog::forContext(dir.path(), "badlen");

  log.appendDump(deltaOf({{1, 11}}, 0));
  const int64_t sizeAfterOneDump = sizeOf(log.path());
  log.appendDump(deltaOf({{2, 22}}, 0));

  // Overwrite the second block's record count with 0xFFFFFFFF. If the header checksum were
  // not checked first, the loader would be handed a four-billion-record length.
  vector<uint8_t> ff(4, 0xFF);
  overwriteBytesAt(log.path(), sizeAfterOneDump + 4, ff);

  const NNCacheCountLogContents contents = log.load();
  testAssert(contents.tail() == NNCacheCountLogTail::Truncated);
  testAssert(contents.blocksApplied() == 1);
  testAssert(!hasRowFor(contents, 2));

  // SECOND LEG, and it is the one that makes the header's own checksum NECESSARY rather
  // than merely present. The leg above is caught twice over -- an absurd record count also
  // fails the bound against the bytes remaining -- so it does not isolate the checksum.
  // This one corrupts the header's unattributed-observations field instead: the length is still
  // right, the magic is still right, the payload still checksums, and the ONLY thing that
  // can see the damage is the header checksum. Without it the log would silently report a
  // fabricated unattributed count.
  {
    const NNCacheCountLog log2 = NNCacheCountLog::forContext(dir.path(), "badhdr");
    log2.appendDump(deltaOf({{1, 11}}, 0));
    const int64_t sizeAfterFirst = sizeOf(log2.path());
    log2.appendDump(deltaOf({{2, 22}}, 5));
    overwriteBytesAt(log2.path(), sizeAfterFirst + 8, vector<uint8_t>{7, 7, 7, 7, 7, 7, 7, 7});

    const NNCacheCountLogContents contents2 = log2.load();
    testAssert(contents2.tail() == NNCacheCountLogTail::Truncated);
    testAssert(contents2.blocksApplied() == 1);
    testAssert(!hasRowFor(contents2, 2));
    testAssert(contents2.unattributedObservations() == 0);
  }
}

// A torn tail must be repaired by the WRITER before it appends, or every later dump lands
// at an offset no loader reaches and is silently lost while every call reports success.
//
// AND THE REPAIR IS A TRUNCATION, asserted as a property of the FILE: the surviving prefix
// is byte-for-byte what stood there before the tear, and the new length is that prefix plus
// the appended block. This log's own files are megabytes, so nothing here is about write
// volume -- it is about the two formats stating ONE torn-tail contract, which is only one
// contract if both of them keep it.
void testCountLogTornTailIsRepairedByTruncationBeforeTheNextAppend() {
  TestCommon::ScopedTempDir dir(TMP_DIR_PREFIX);
  const NNCacheCountLog log = NNCacheCountLog::forContext(dir.path(), "repair");

  log.appendDump(deltaOf({{1, 10}, {2, 20}}, 0));
  const int64_t sizeAfterOneDump = sizeOf(log.path());
  const vector<uint8_t> intactPrefix = readBytesAt(log.path(), 0, (size_t)sizeAfterOneDump);
  log.appendDump(deltaOf({{3, 30}}, 0));
  truncateTo(log.path(), sizeOf(log.path()) - 5);
  const int64_t tornBytes = sizeOf(log.path()) - sizeAfterOneDump;

  const NNCacheCountLogAppendResult appended = log.appendDump(deltaOf({{4, 40}}, 0));
  testAssert(appended.tornTailBytesDiscarded == tornBytes);
  testAssert(appended.tailRepair == NNCacheFileTailRepair::Truncated);
  testAssert(readBytesAt(log.path(), 0, (size_t)sizeAfterOneDump) == intactPrefix);
  testAssert(sizeOf(log.path()) == sizeAfterOneDump + appended.bytesAppended);

  const NNCacheCountLogContents contents = log.load();
  testAssert(contents.tail() == NNCacheCountLogTail::Intact);
  // The surviving prefix is the first dump's own block; the new dump is the second.
  testAssert(contents.blocksApplied() == 2);
  testAssert(rowFor(contents, 1).observations == 10);
  testAssert(rowFor(contents, 2).observations == 20);
  testAssert(!hasRowFor(contents, 3));
  // THE POINT OF THE TEST: the dump written after the torn tail is readable. Without the
  // repair this row is what goes missing, silently.
  testAssert(rowFor(contents, 4).observations == 40);
}

// The state truncation newly makes reachable, and the count log's twin of the evaluation
// container's own test for it. A crash during the very first dump leaves a file that exists
// and carries nothing this build can read; the repair shortens it to zero bytes, where the
// rewrite it replaced would have left a fresh header behind. The next append must write that
// header rather than append a block to a headerless log.
//
// IT HAS A TWIN FOR A REASON. The whole truncation change rests on the two formats keeping
// ONE torn-tail contract, and a contract tested in one of its two implementations is a
// contract in one of them.
void testCountLogRepairsALogWhoseIntactPartIsEmpty() {
  TestCommon::ScopedTempDir dir(TMP_DIR_PREFIX);
  const NNCacheCountLog log = NNCacheCountLog::forContext(dir.path(), "emptyintact");

  log.appendDump(deltaOf({{1, 10}}, 0));
  // A crash before even the file header was whole.
  truncateTo(log.path(), 12);

  const NNCacheCountLogAppendResult appended = log.appendDump(deltaOf({{2, 20}}, 0));
  testAssert(appended.tornTailBytesDiscarded == 12);
  testAssert(appended.tailRepair == NNCacheFileTailRepair::Truncated);
  // The header is part of what was appended, because the file had none left.
  testAssert(sizeOf(log.path()) == appended.bytesAppended);

  const NNCacheCountLogContents contents = log.load();
  testAssert(contents.tail() == NNCacheCountLogTail::Intact);
  testAssert(contents.blocksApplied() == 1);
  testAssert(!hasRowFor(contents, 1));
  testAssert(rowFor(contents, 2).observations == 20);
}

// compactIfNeeded's TORN-BUT-UNDER-THE-MULTIPLE arm, the count log's twin of the container's:
// it shortens and rewrites nothing, and says which by name.
void testCountLogCompactIfNeededTruncatesATornTailRatherThanCompacting() {
  TestCommon::ScopedTempDir dir(TMP_DIR_PREFIX);
  const NNCacheCountLog log = NNCacheCountLog::forContext(dir.path(), "torntrunc");

  // Distinct keys per dump, so the physical record count equals the live set and the
  // over-multiple trigger cannot fire.
  log.appendDump(deltaOf({{1, 10}}, 0));
  log.appendDump(deltaOf({{2, 20}}, 0));
  const int64_t intactSize = sizeOf(log.path());
  const vector<uint8_t> intactPrefix = readBytesAt(log.path(), 0, (size_t)intactSize);

  log.appendDump(deltaOf({{3, 30}}, 0));
  truncateTo(log.path(), sizeOf(log.path()) - 5);

  testAssert(log.compactIfNeeded(NNCacheCountLog::defaultCompactionMultiple()) ==
             NNCacheFileMaintenance::TruncatedTornTail);
  testAssert(sizeOf(log.path()) == intactSize);
  testAssert(readBytesAt(log.path(), 0, (size_t)intactSize) == intactPrefix);

  const NNCacheCountLogContents contents = log.load();
  testAssert(contents.tail() == NNCacheCountLogTail::Intact);
  // TWO blocks, not one: a compaction would have collapsed them, and this did not compact.
  testAssert(contents.blocksApplied() == 2);
  testAssert(rowFor(contents, 1).observations == 10);
  testAssert(rowFor(contents, 2).observations == 20);
  testAssert(!hasRowFor(contents, 3));

  testAssert(log.compactIfNeeded(NNCacheCountLog::defaultCompactionMultiple()) ==
             NNCacheFileMaintenance::Nothing);
}

// Compaction preserves every total and shrinks the file; a crash mid-compaction leaves the
// original intact.
void testCountLogCompactionPreservesTotalsAndSurvivesACrash() {
  TestCommon::ScopedTempDir dir(TMP_DIR_PREFIX);
  const NNCacheCountLog log = NNCacheCountLog::forContext(dir.path(), "compact");

  for(int i = 0; i < 10; i++)
    log.appendDump(deltaOf({{1, 2}, {2, 3}}, 1));
  const NNCacheCountLogContents before = log.load();
  const int64_t sizeBefore = sizeOf(log.path());
  testAssert(before.recordsApplied() == 20);
  testAssert(before.rows().size() == 2);

  // A crash mid-compaction leaves a stale temp at the compaction path. The original must be
  // untouched by it, and the next compaction must overwrite the stale temp rather than read
  // it.
  const string stalePath = log.path() + ".compacting";
  {
    FILE* f = fopen(stalePath.c_str(), "wb");
    testAssert(f != NULL);
    const char garbage[] = "a half-written compaction from a process that died";
    testAssert(fwrite(garbage, 1, sizeof(garbage), f) == sizeof(garbage));
    fclose(f);
  }
  const NNCacheCountLogContents afterStale = log.load();
  testAssert(afterStale.tail() == NNCacheCountLogTail::Intact);
  testAssert(afterStale.rows().size() == 2);
  testAssert(sizeOf(log.path()) == sizeBefore);

  const NNCacheFileMaintenance compacted = log.compactIfNeeded(NNCacheCountLog::defaultCompactionMultiple());
  testAssert(compacted == NNCacheFileMaintenance::Compacted);

  const NNCacheCountLogContents after = log.load();
  testAssert(after.tail() == NNCacheCountLogTail::Intact);
  testAssert(after.blocksApplied() == 1);
  testAssert(after.recordsApplied() == 2);
  testAssert(after.rows().size() == 2);
  // Observationally identical to the log it replaced: same totals, same unattributed sum.
  testAssert(rowFor(after, 1).observations == 20 && rowFor(after, 1).sessions == 10);
  testAssert(rowFor(after, 2).observations == 30 && rowFor(after, 2).sessions == 10);
  testAssert(after.unattributedObservations() == before.unattributedObservations());
  testAssert(sizeOf(log.path()) ==
             (int64_t)NNCacheCountLog::fileHeaderBytes() + NNCacheCountLog::bytesForDumpOf(2));
  testAssert(sizeOf(log.path()) < sizeBefore);
  // The temp path is gone: the rename consumed it.
  testAssert(!FileUtils::exists(stalePath));

  // Under the multiple, a fresh compaction does not fire.
  testAssert(log.compactIfNeeded(NNCacheCountLog::defaultCompactionMultiple()) == NNCacheFileMaintenance::Nothing);
}

// Ordering is by observations. Nothing here ranks by sessions.
void testCountLogOrdersByLookupsAndNotBySessions() {
  TestCommon::ScopedTempDir dir(TMP_DIR_PREFIX);
  const NNCacheCountLog log = NNCacheCountLog::forContext(dir.path(), "ordering");

  // Key 1: one dump, many observations. Key 2: many dumps, few observations each. The two orderings
  // disagree, which is the whole point of the fixture.
  log.appendDump(deltaOf({{1, 100}, {2, 1}}, 0));
  for(int i = 0; i < 5; i++)
    log.appendDump(deltaOf({{2, 1}}, 0));

  const NNCacheCountLogContents contents = log.load();
  testAssert(rowFor(contents, 1).observations == 100 && rowFor(contents, 1).sessions == 1);
  testAssert(rowFor(contents, 2).observations == 6 && rowFor(contents, 2).sessions == 6);

  const vector<NNCacheCountRow> ordered = contents.byDescendingObservations();
  testAssert(ordered.size() == 2);
  testAssert(ordered[0].key == nthKey(1));  // more observations, fewer sessions -- and it is first
  testAssert(ordered[1].key == nthKey(2));
}

// The boundary refuses what it cannot honor and never coerces it into something plausible.
void testCountLogRefusesWhatItCannotHonor() {
  TestCommon::ScopedTempDir dir(TMP_DIR_PREFIX);

  // A context name becomes a path component, so it is validated to a closed alphabet.
  const char* badContexts[] = {"", ".", "..", "../escape", "a/b", "a\\b", "with space", "semi;colon"};
  for(size_t i = 0; i < sizeof(badContexts) / sizeof(badContexts[0]); i++) {
    bool refused = false;
    try { NNCacheCountLog::forContext(dir.path(), badContexts[i]); }
    catch(const StringError&) { refused = true; }
    testAssert(refused);
  }
  // And accepts the alphabet it declares.
  NNCacheCountLog::forContext(dir.path(), "b18c384nbt-19x19_v2.1");

  // A directory that does not exist is named, not created.
  {
    bool refused = false;
    try { NNCacheCountLog::forContext(dir.path() + "/nope", "ctx"); }
    catch(const StringError&) { refused = true; }
    testAssert(refused);
  }

  // A NotObserved ledger is refused rather than written as a dump of zero rows, which a
  // later reader would take for "this session hit nothing".
  {
    const NNCacheCountLog log = NNCacheCountLog::forContext(dir.path(), "notcounted");
    string message;
    bool refused = false;
    try { log.appendDump(NNCacheObservationDelta::notObserved()); }
    catch(const StringError& e) { refused = true; message = e.what(); }
    testAssert(refused);
    testAssert(!FileUtils::exists(log.path()));
    // ASSERTED ON THE MESSAGE, and the reason is worth stating because a bare "did it
    // throw" here was a proxy witness that a seen-red leg caught. Removing appendDump's own
    // isObserved() guard leaves the call throwing anyway, because
    // NNCacheObservationLedger::entries() already refuses under NotObserved -- so "it threw"
    // is true under both the presence and
    // the absence of the thing being tested. What the guard actually buys is the DIAGNOSIS:
    // an operator gets told the count log declined to persist a session's counts, not that
    // some accessor refused. That message is operator-facing and load-bearing, which is what
    // makes it a legitimate anchor rather than adjacent prose (ADR-0021 Rule 3).
    testAssert(message.find("NNCacheCountLog") != string::npos);
    testAssert(message.find("NotObserved") != string::npos);
  }

  // A missing file is not an error: it reads as an empty, intact log.
  {
    const NNCacheCountLog log = NNCacheCountLog::forContext(dir.path(), "neverwritten");
    const NNCacheCountLogContents contents = log.load();
    testAssert(contents.tail() == NNCacheCountLogTail::Intact);
    testAssert(contents.rows().empty());
    testAssert(contents.blocksApplied() == 0);
  }

  // A file that is not a count log is refused by name rather than read on the chance that
  // its fields line up.
  {
    const NNCacheCountLog log = NNCacheCountLog::forContext(dir.path(), "foreign");
    FILE* f = fopen(log.path().c_str(), "wb");
    testAssert(f != NULL);
    const char notALog[] = "SQLite format 3\0and then some more bytes so it is not short";
    testAssert(fwrite(notALog, 1, sizeof(notALog), f) == sizeof(notALog));
    fclose(f);
    testAssert(loadIsRefused(log));
  }

  // A file written for another context is refused rather than merged. Two contexts' counts
  // are different facts about different key sets.
  {
    const NNCacheCountLog one = NNCacheCountLog::forContext(dir.path(), "ctxone");
    one.appendDump(deltaOf({{1, 1}}, 0));
    const NNCacheCountLog two = NNCacheCountLog::forContext(dir.path(), "ctxtwo");
    gfs::copy_file(gfs::u8path(one.path()), gfs::u8path(two.path()));
    testAssert(loadIsRefused(two));
  }

  // A bumped version byte is refused, which is what the magic and version are for.
  {
    const NNCacheCountLog log = NNCacheCountLog::forContext(dir.path(), "version");
    log.appendDump(deltaOf({{1, 1}}, 0));
    overwriteBytesAt(log.path(), 8, vector<uint8_t>{99, 0, 0, 0});
    testAssert(loadIsRefused(log));
  }

  // A compaction multiple below 1 has no reading.
  {
    const NNCacheCountLog log = NNCacheCountLog::forContext(dir.path(), "multiple");
    bool refused = false;
    try { log.compactIfNeeded(0); }
    catch(const StringError&) { refused = true; }
    testAssert(refused);
  }

  // The tail disposition and the discarded byte count cannot disagree.
  {
    bool refused = false;
    try {
      NNCacheCountLogContents::of({}, 0, NNCacheCountLogTail::Truncated, 0, 0, 0);
    }
    catch(const StringError&) { refused = true; }
    testAssert(refused);
  }
}

}  // namespace

void Tests::runNNCacheCountLogTests() {
  cout << "Running nn cache count log tests" << endl;

  testCountLogRoundTripsCountsExactly();
  testCountLogAccumulatesAcrossDumps();
  testCountLogEmptyDumpAppendsNothing();
  testCountLogTornTailIsDiscardedAndThePrefixSurvives();
  testCountLogAWholeButCorruptBlockIsRejectedByItsChecksum();
  testCountLogACorruptBlockHeaderIsRejectedBeforeItsLengthIsBelieved();
  testCountLogTornTailIsRepairedByTruncationBeforeTheNextAppend();
  testCountLogRepairsALogWhoseIntactPartIsEmpty();
  testCountLogCompactIfNeededTruncatesATornTailRatherThanCompacting();
  testCountLogCompactionPreservesTotalsAndSurvivesACrash();
  testCountLogOrdersByLookupsAndNotBySessions();
  testCountLogRefusesWhatItCannotHonor();

  // The format's own strides, so the report's arithmetic is quoted from the implementation
  // rather than from a second copy of the numbers.
  cout << "count log: file header " << NNCacheCountLog::fileHeaderBytes()
       << " B, block header " << NNCacheCountLog::blockHeaderBytes()
       << " B, record " << NNCacheCountLog::recordBytes()
       << " B; a 40000-row dump is " << NNCacheCountLog::bytesForDumpOf(40000) << " B" << endl;
}
