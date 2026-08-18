#include "../tests/tests.h"

#include <cstdio>
#include <iostream>
#include <vector>

#include <ghc/filesystem.hpp>

#include "../core/fileutils.h"
#include "../core/rand.h"
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

NNCacheHitLedger ledgerOf(const vector<pair<int,uint32_t>>& serialAndHits, int64_t unrecorded) {
  vector<NNCacheHitCount> rows;
  for(size_t i = 0; i < serialAndHits.size(); i++) {
    NNCacheHitCount row;
    row.key = nthKey(serialAndHits[i].first);
    row.hits = serialAndHits[i].second;
    rows.push_back(row);
  }
  return NNCacheHitLedger::counted(std::move(rows), unrecorded);
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

// A directory that removes itself, so a failed assertion in the middle of a test does not
// leave litter behind for the next run to trip over.
class ScopedTempDir {
 public:
  ScopedTempDir() {
    Rand rand;
    path_ = "tmpnncachecountlog_" + Global::uint64ToHexString(rand.nextUInt64());
    gfs::create_directory(gfs::u8path(path_));
  }
  ~ScopedTempDir() {
    std::error_code ec;
    gfs::remove_all(gfs::u8path(path_), ec);
  }
  ScopedTempDir(const ScopedTempDir&) = delete;
  ScopedTempDir& operator=(const ScopedTempDir&) = delete;
  const string& path() const { return path_; }

 private:
  string path_;
};

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

//-------------------------------------------------------------------------------------
// The tests
//-------------------------------------------------------------------------------------

// Counts written are counts read back. Exactly -- this is a logic invariant, so there is no
// tolerance and no approximate comparison anywhere in it.
void testCountLogRoundTripsCountsExactly() {
  ScopedTempDir dir;
  const NNCacheCountLog log = NNCacheCountLog::forContext(dir.path(), "roundtrip");

  const NNCacheCountLogAppendResult appended = log.appendDump(ledgerOf({{1, 7}, {2, 0}, {3, 4000000000u}}, 0));
  testAssert(appended.tornTailBytesDiscarded == 0);
  testAssert(appended.rewroteTheFile == false);
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
  testAssert(contents.unattributedLookups() == 0);

  testAssert(rowFor(contents, 1).lookups == 7);
  // A pre-warmed entry that earned nothing is stored, not dropped: "level 0 held this and
  // nobody asked for it" is the signal to stop carrying it, and it is not the same fact as
  // "this key is not in the log".
  testAssert(rowFor(contents, 2).lookups == 0);
  // The top of the 32-bit range round-trips, so the record's field width is exercised
  // rather than assumed.
  testAssert(rowFor(contents, 3).lookups == 4000000000ull);
  for(size_t i = 0; i < contents.rows().size(); i++)
    testAssert(contents.rows()[i].sessions == 1);
}

// Merging is addition, and sessions counts the dumps a key appeared in.
void testCountLogAccumulatesAcrossDumps() {
  ScopedTempDir dir;
  const NNCacheCountLog log = NNCacheCountLog::forContext(dir.path(), "accumulate");

  log.appendDump(ledgerOf({{1, 3}, {2, 5}}, 0));
  log.appendDump(ledgerOf({{1, 4}, {3, 9}}, 2));
  log.appendDump(ledgerOf({{1, 1}}, 3));

  const NNCacheCountLogContents contents = log.load();
  testAssert(contents.blocksApplied() == 3);
  testAssert(contents.recordsApplied() == 5);
  testAssert(contents.rows().size() == 3);
  testAssert(rowFor(contents, 1).lookups == 8 && rowFor(contents, 1).sessions == 3);
  testAssert(rowFor(contents, 2).lookups == 5 && rowFor(contents, 2).sessions == 1);
  testAssert(rowFor(contents, 3).lookups == 9 && rowFor(contents, 3).sessions == 1);
  // Hits the cache could not attribute to any key are carried through the log rather than
  // dropped at the door, and they sum across dumps like everything else.
  testAssert(contents.unattributedLookups() == 5);
}

// THE LOAD-BEARING TEST. A crash mid-dump leaves a partial block. Exactly the dumps that
// completed must survive, and the partial one must not.
void testCountLogTornTailIsDiscardedAndThePrefixSurvives() {
  ScopedTempDir dir;
  const NNCacheCountLog log = NNCacheCountLog::forContext(dir.path(), "torn");

  log.appendDump(ledgerOf({{1, 10}, {2, 20}}, 0));
  log.appendDump(ledgerOf({{1, 5}, {3, 30}}, 0));
  const int64_t sizeAfterTwoDumps = sizeOf(log.path());

  // The third dump names a key no earlier dump did, so its survival or absence is directly
  // observable rather than hidden inside a sum (ADR-0021 Rule 1: observe the property at
  // the site of the claim).
  log.appendDump(ledgerOf({{1, 100}, {4, 400}, {5, 500}}, 7));
  const int64_t sizeAfterThreeDumps = sizeOf(log.path());
  testAssert(sizeAfterThreeDumps == sizeAfterTwoDumps + NNCacheCountLog::bytesForDumpOf(3));

  // Cut the file in the MIDDLE OF THE THIRD BLOCK'S SECOND RECORD -- not on a record
  // boundary, and not on a block boundary. This is what a process dying part-way through a
  // write(2) leaves behind.
  const int64_t cutAt = sizeAfterTwoDumps + (int64_t)NNCacheCountLog::blockHeaderBytes() +
                        (int64_t)NNCacheCountLog::recordBytes() + 9;
  truncateTo(log.path(), cutAt);
  testAssert(sizeOf(log.path()) == cutAt);

  const NNCacheCountLogContents contents = log.load();

  // The disposition is Truncated and names exactly how many bytes it will not use.
  testAssert(contents.tail() == NNCacheCountLogTail::Truncated);
  testAssert(contents.discardedTailBytes() == cutAt - sizeAfterTwoDumps);

  // Exactly the two completed dumps survive, with exact totals.
  testAssert(contents.blocksApplied() == 2);
  testAssert(contents.recordsApplied() == 4);
  testAssert(rowFor(contents, 1).lookups == 15 && rowFor(contents, 1).sessions == 2);
  testAssert(rowFor(contents, 2).lookups == 20 && rowFor(contents, 2).sessions == 1);
  testAssert(rowFor(contents, 3).lookups == 30 && rowFor(contents, 3).sessions == 1);

  // THE PARTIAL RECORD DID NOT SURVIVE, observed positively: keys 4 and 5 were named ONLY
  // by the torn block, so their absence is a direct membership fact about the rows the load
  // returned, and the assertion goes red the moment a partial block is applied.
  testAssert(!hasRowFor(contents, 4));
  testAssert(!hasRowFor(contents, 5));
  testAssert(contents.rows().size() == 3);
  // And key 1's 100 hits from the torn block are not in its total either -- a partial block
  // must not be applied in part.
  testAssert(rowFor(contents, 1).lookups != 115);
  // The torn block's unattributed count went with it.
  testAssert(contents.unattributedLookups() == 0);
}

// A block that is the RIGHT LENGTH but wrong in its bytes -- the shape a lost page leaves,
// which a length-only framing cannot see at all.
void testCountLogAWholeButCorruptBlockIsRejectedByItsChecksum() {
  ScopedTempDir dir;
  const NNCacheCountLog log = NNCacheCountLog::forContext(dir.path(), "corrupt");

  log.appendDump(ledgerOf({{1, 11}}, 0));
  const int64_t sizeAfterOneDump = sizeOf(log.path());
  log.appendDump(ledgerOf({{2, 22}}, 0));

  // Zero out eight bytes inside the SECOND block's single record: the file length is
  // unchanged, every offset still lines up, and only the payload checksum can tell.
  const int64_t recordAt = sizeAfterOneDump + (int64_t)NNCacheCountLog::blockHeaderBytes();
  overwriteBytesAt(log.path(), recordAt, vector<uint8_t>(8, 0));

  const NNCacheCountLogContents contents = log.load();
  testAssert(contents.tail() == NNCacheCountLogTail::Truncated);
  testAssert(contents.discardedTailBytes() == sizeOf(log.path()) - sizeAfterOneDump);
  testAssert(contents.blocksApplied() == 1);
  testAssert(rowFor(contents, 1).lookups == 11);
  testAssert(!hasRowFor(contents, 2));
}

// A corrupt block HEADER must be caught by the header's own checksum before its record
// count is believed, so a length a crash chose never reaches an allocation.
void testCountLogACorruptBlockHeaderIsRejectedBeforeItsLengthIsBelieved() {
  ScopedTempDir dir;
  const NNCacheCountLog log = NNCacheCountLog::forContext(dir.path(), "badlen");

  log.appendDump(ledgerOf({{1, 11}}, 0));
  const int64_t sizeAfterOneDump = sizeOf(log.path());
  log.appendDump(ledgerOf({{2, 22}}, 0));

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
  // This one corrupts the header's unattributed-lookups field instead: the length is still
  // right, the magic is still right, the payload still checksums, and the ONLY thing that
  // can see the damage is the header checksum. Without it the log would silently report a
  // fabricated unattributed count.
  {
    const NNCacheCountLog log2 = NNCacheCountLog::forContext(dir.path(), "badhdr");
    log2.appendDump(ledgerOf({{1, 11}}, 0));
    const int64_t sizeAfterFirst = sizeOf(log2.path());
    log2.appendDump(ledgerOf({{2, 22}}, 5));
    overwriteBytesAt(log2.path(), sizeAfterFirst + 8, vector<uint8_t>{7, 7, 7, 7, 7, 7, 7, 7});

    const NNCacheCountLogContents contents2 = log2.load();
    testAssert(contents2.tail() == NNCacheCountLogTail::Truncated);
    testAssert(contents2.blocksApplied() == 1);
    testAssert(!hasRowFor(contents2, 2));
    testAssert(contents2.unattributedLookups() == 0);
  }
}

// A torn tail must be repaired by the WRITER before it appends, or every later dump lands
// at an offset no loader reaches and is silently lost while every call reports success.
void testCountLogTornTailIsRepairedBeforeTheNextAppend() {
  ScopedTempDir dir;
  const NNCacheCountLog log = NNCacheCountLog::forContext(dir.path(), "repair");

  log.appendDump(ledgerOf({{1, 10}, {2, 20}}, 0));
  const int64_t sizeAfterOneDump = sizeOf(log.path());
  log.appendDump(ledgerOf({{3, 30}}, 0));
  truncateTo(log.path(), sizeOf(log.path()) - 5);
  const int64_t tornBytes = sizeOf(log.path()) - sizeAfterOneDump;

  const NNCacheCountLogAppendResult appended = log.appendDump(ledgerOf({{4, 40}}, 0));
  testAssert(appended.tornTailBytesDiscarded == tornBytes);
  testAssert(appended.rewroteTheFile == true);

  const NNCacheCountLogContents contents = log.load();
  testAssert(contents.tail() == NNCacheCountLogTail::Intact);
  // The repair collapsed the surviving prefix into one block; the new dump is the second.
  testAssert(contents.blocksApplied() == 2);
  testAssert(rowFor(contents, 1).lookups == 10);
  testAssert(rowFor(contents, 2).lookups == 20);
  testAssert(!hasRowFor(contents, 3));
  // THE POINT OF THE TEST: the dump written after the torn tail is readable. Without the
  // repair this row is what goes missing, silently.
  testAssert(rowFor(contents, 4).lookups == 40);
}

// Compaction preserves every total and shrinks the file; a crash mid-compaction leaves the
// original intact.
void testCountLogCompactionPreservesTotalsAndSurvivesACrash() {
  ScopedTempDir dir;
  const NNCacheCountLog log = NNCacheCountLog::forContext(dir.path(), "compact");

  for(int i = 0; i < 10; i++)
    log.appendDump(ledgerOf({{1, 2}, {2, 3}}, 1));
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

  const bool compacted = log.compactIfNeeded(NNCacheCountLog::defaultCompactionMultiple());
  testAssert(compacted == true);

  const NNCacheCountLogContents after = log.load();
  testAssert(after.tail() == NNCacheCountLogTail::Intact);
  testAssert(after.blocksApplied() == 1);
  testAssert(after.recordsApplied() == 2);
  testAssert(after.rows().size() == 2);
  // Observationally identical to the log it replaced: same totals, same unattributed sum.
  testAssert(rowFor(after, 1).lookups == 20 && rowFor(after, 1).sessions == 10);
  testAssert(rowFor(after, 2).lookups == 30 && rowFor(after, 2).sessions == 10);
  testAssert(after.unattributedLookups() == before.unattributedLookups());
  testAssert(sizeOf(log.path()) ==
             (int64_t)NNCacheCountLog::fileHeaderBytes() + NNCacheCountLog::bytesForDumpOf(2));
  testAssert(sizeOf(log.path()) < sizeBefore);
  // The temp path is gone: the rename consumed it.
  testAssert(!FileUtils::exists(stalePath));

  // Under the multiple, a fresh compaction does not fire.
  testAssert(log.compactIfNeeded(NNCacheCountLog::defaultCompactionMultiple()) == false);
}

// Ordering is by lookups. Nothing here ranks by sessions.
void testCountLogOrdersByLookupsAndNotBySessions() {
  ScopedTempDir dir;
  const NNCacheCountLog log = NNCacheCountLog::forContext(dir.path(), "ordering");

  // Key 1: one dump, many lookups. Key 2: many dumps, few lookups each. The two orderings
  // disagree, which is the whole point of the fixture.
  log.appendDump(ledgerOf({{1, 100}, {2, 1}}, 0));
  for(int i = 0; i < 5; i++)
    log.appendDump(ledgerOf({{2, 1}}, 0));

  const NNCacheCountLogContents contents = log.load();
  testAssert(rowFor(contents, 1).lookups == 100 && rowFor(contents, 1).sessions == 1);
  testAssert(rowFor(contents, 2).lookups == 6 && rowFor(contents, 2).sessions == 6);

  const vector<NNCacheCountRow> ordered = contents.byDescendingLookups();
  testAssert(ordered.size() == 2);
  testAssert(ordered[0].key == nthKey(1));  // more lookups, fewer sessions -- and it is first
  testAssert(ordered[1].key == nthKey(2));
}

// The boundary refuses what it cannot honor and never coerces it into something plausible.
void testCountLogRefusesWhatItCannotHonor() {
  ScopedTempDir dir;

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

  // A NotCounted ledger is refused rather than written as a dump of zero rows, which a
  // later reader would take for "this session hit nothing".
  {
    const NNCacheCountLog log = NNCacheCountLog::forContext(dir.path(), "notcounted");
    bool refused = false;
    try { log.appendDump(NNCacheHitLedger::notCounted()); }
    catch(const StringError&) { refused = true; }
    testAssert(refused);
    testAssert(!FileUtils::exists(log.path()));
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
    one.appendDump(ledgerOf({{1, 1}}, 0));
    const NNCacheCountLog two = NNCacheCountLog::forContext(dir.path(), "ctxtwo");
    gfs::copy_file(gfs::u8path(one.path()), gfs::u8path(two.path()));
    testAssert(loadIsRefused(two));
  }

  // A bumped version byte is refused, which is what the magic and version are for.
  {
    const NNCacheCountLog log = NNCacheCountLog::forContext(dir.path(), "version");
    log.appendDump(ledgerOf({{1, 1}}, 0));
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
  testCountLogTornTailIsDiscardedAndThePrefixSurvives();
  testCountLogAWholeButCorruptBlockIsRejectedByItsChecksum();
  testCountLogACorruptBlockHeaderIsRejectedBeforeItsLengthIsBelieved();
  testCountLogTornTailIsRepairedBeforeTheNextAppend();
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
