#include "../tests/tests.h"

#include <cstdio>
#include <functional>
#include <iostream>
#include <sstream>
#include <vector>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#include "../core/rand.h"
#include "../external/nlohmann_json/json.hpp"
#include "../main.h"
#include "../neuralnet/nncachecountlog.h"

using namespace std;
using namespace TestCommon;
using json = nlohmann::json;

// Tests for the nncountsdump subcommand and for the read-only detailed-load extension it is
// built on (NNCacheCountLog::loadDetailedUnlocked / NNCacheCountLogDetailedContents in
// nncachecountlog.h). The load-bearing question these tests answer, per row 1711's charter,
// is faithfulness: does the per-block view show exactly what a block's own bytes recorded
// (a delta), and does the accumulated view still match what load() has always computed
// (a sum across blocks) -- see nncachecountlog.cpp's scanLog, where the merge itself lives
// ("result.rows[it->second].lookups += ...").

namespace {

const char* const TMP_DIR_PREFIX = "tmpnncountsdump";

Hash128 nthKey(int serial) {
  return Hash128(
    ((uint64_t)(serial + 1)) * 0x9E3779B97F4A7C15ULL,
    ((uint64_t)(serial + 1)) * 0xD6E8FEB86659FD93ULL + 0x1234567ULL
  );
}

NNCacheHitCountDelta deltaOf(const vector<pair<int,uint32_t>>& serialAndHits, int64_t unrecorded) {
  vector<NNCacheHitCount> rows;
  for(size_t i = 0; i < serialAndHits.size(); i++) {
    NNCacheHitCount row;
    row.key = nthKey(serialAndHits[i].first);
    row.hits = serialAndHits[i].second;
    rows.push_back(row);
  }
  return NNCacheHitCountDelta::ofDeltaRows(std::move(rows), unrecorded);
}

// The one row in a block for `serial`, or NULL if that block did not mention it -- the
// per-block surface has no "one row per key" guarantee across the whole log the way
// NNCacheCountRow does, so callers ask block-by-block.
const NNCacheCountLogBlockRow* blockRowFor(const NNCacheCountLogBlock& block, int serial) {
  const Hash128 key = nthKey(serial);
  for(size_t i = 0; i < block.rows.size(); i++) {
    if(block.rows[i].key == key)
      return &block.rows[i];
  }
  return NULL;
}

const NNCacheCountRow& aggregateRowFor(const NNCacheCountLogContents& contents, int serial) {
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

void truncateFileTo(const string& path, int64_t bytes) {
  FILE* f = fopen(path.c_str(), "r+b");
  testAssert(f != NULL);
#ifdef _WIN32
  testAssert(_chsize(_fileno(f), (long)bytes) == 0);
#else
  testAssert(ftruncate(fileno(f), (off_t)bytes) == 0);
#endif
  fclose(f);
}

int64_t sizeOfFile(const string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  testAssert(f != NULL);
  testAssert(fseek(f, 0, SEEK_END) == 0);
  const long sz = ftell(f);
  testAssert(sz >= 0);
  fclose(f);
  return (int64_t)sz;
}

// Runs a MainCmds subcommand with std::cout captured, for a test that has to assert on the
// rendered text rather than on the library types underneath it -- the subcommand's own
// argument parsing, file-path splitting and printing are otherwise untested.
string captureStdout(const std::function<int()>& call, int& exitCodeOut) {
  ostringstream captured;
  streambuf* const savedBuf = cout.rdbuf(captured.rdbuf());
  exitCodeOut = call();
  cout.rdbuf(savedBuf);
  return captured.str();
}

//-------------------------------------------------------------------------------------
// loadDetailedUnlocked: per-block rows are deltas, aggregate is still the sum
//-------------------------------------------------------------------------------------

// THE LOAD-BEARING TEST. A key that appears in two blocks must show its OWN delta in each
// block's own rows, while the aggregate view sums them -- exactly the distinction the
// dumper's contract asks for (per-block rows AND an accumulated-total view).
void testDetailedLoadShowsPerBlockDeltasAndAccumulatedTotals() {
  ScopedTempDir dir(TMP_DIR_PREFIX);
  const NNCacheCountLog log = NNCacheCountLog::forContext(dir.path(), "detailed");

  // Key 1 appears in all three blocks; key 2 only in the first; key 3 only in the third.
  log.appendDump(deltaOf({{1, 3}, {2, 5}}, 0));
  log.appendDump(deltaOf({{1, 4}}, 2));
  log.appendDump(deltaOf({{1, 1}, {3, 9}}, 7));

  const NNCacheCountLogDetailedContents detailed = log.loadDetailedUnlocked();
  testAssert(detailed.blocks().size() == 3);

  // Per block: each row is that dump's own delta, never a running total.
  const NNCacheCountLogBlockRow* b0k1 = blockRowFor(detailed.blocks()[0], 1);
  testAssert(b0k1 != NULL && b0k1->lookups == 3 && b0k1->sessions == 1);
  testAssert(blockRowFor(detailed.blocks()[0], 2) != NULL);
  testAssert(blockRowFor(detailed.blocks()[0], 3) == NULL);
  testAssert(detailed.blocks()[0].unattributedLookups == 0);

  const NNCacheCountLogBlockRow* b1k1 = blockRowFor(detailed.blocks()[1], 1);
  testAssert(b1k1 != NULL && b1k1->lookups == 4 && b1k1->sessions == 1);
  testAssert(blockRowFor(detailed.blocks()[1], 2) == NULL);
  testAssert(detailed.blocks()[1].unattributedLookups == 2);

  const NNCacheCountLogBlockRow* b2k1 = blockRowFor(detailed.blocks()[2], 1);
  testAssert(b2k1 != NULL && b2k1->lookups == 1 && b2k1->sessions == 1);
  const NNCacheCountLogBlockRow* b2k3 = blockRowFor(detailed.blocks()[2], 3);
  testAssert(b2k3 != NULL && b2k3->lookups == 9 && b2k3->sessions == 1);
  testAssert(detailed.blocks()[2].unattributedLookups == 7);

  // Aggregate: the same merge load() has always computed -- key 1's lookups accumulate
  // (3+4+1=8) across the three dumps it appeared in every time, matching
  // nncachecountlog.cpp's scanLog merge ("result.rows[it->second].lookups += ...").
  const NNCacheCountLogContents& agg = detailed.aggregate();
  testAssert(agg.blocksApplied() == 3);
  testAssert(aggregateRowFor(agg, 1).lookups == 8 && aggregateRowFor(agg, 1).sessions == 3);
  testAssert(aggregateRowFor(agg, 2).lookups == 5 && aggregateRowFor(agg, 2).sessions == 1);
  testAssert(aggregateRowFor(agg, 3).lookups == 9 && aggregateRowFor(agg, 3).sessions == 1);
  testAssert(agg.unattributedLookups() == 9);
  testAssert(agg.tail() == NNCacheCountLogTail::Intact);
}

// A torn tail: everything up to the last whole block is applied and shown (both per-block
// and in the aggregate); the partial block contributes nothing and is reported as torn.
void testDetailedLoadReportsTornTailAndKeepsThePrefix() {
  ScopedTempDir dir(TMP_DIR_PREFIX);
  const NNCacheCountLog log = NNCacheCountLog::forContext(dir.path(), "torn");

  log.appendDump(deltaOf({{1, 10}, {2, 20}}, 0));
  const int64_t sizeAfterOneDump = sizeOfFile(log.path());
  log.appendDump(deltaOf({{1, 100}, {4, 400}, {5, 500}}, 0));
  const int64_t sizeAfterTwoDumps = sizeOfFile(log.path());
  testAssert(sizeAfterTwoDumps == sizeAfterOneDump + NNCacheCountLog::bytesForDumpOf(3));

  // Cut part-way through the second block's third record: not a block boundary. The bytes
  // kept past the first, whole block are the second block's own header plus its first two
  // records plus 5 stray bytes of its third -- exactly what discardedTailBytes below must
  // name.
  const int64_t keptOfSecondBlock =
    (int64_t)NNCacheCountLog::blockHeaderBytes() + 2 * (int64_t)NNCacheCountLog::recordBytes() + 5;
  truncateFileTo(log.path(), sizeAfterOneDump + keptOfSecondBlock);
  testAssert(sizeAfterOneDump + keptOfSecondBlock < sizeAfterTwoDumps);

  const NNCacheCountLogDetailedContents detailed = log.loadDetailedUnlocked();
  testAssert(detailed.blocks().size() == 1);
  testAssert(blockRowFor(detailed.blocks()[0], 1) != NULL);
  testAssert(blockRowFor(detailed.blocks()[0], 2) != NULL);

  const NNCacheCountLogContents& agg = detailed.aggregate();
  testAssert(agg.tail() == NNCacheCountLogTail::Truncated);
  testAssert(agg.discardedTailBytes() == keptOfSecondBlock);
  testAssert(aggregateRowFor(agg, 1).lookups == 10);
  testAssert(aggregateRowFor(agg, 2).lookups == 20);
}

//-------------------------------------------------------------------------------------
// The subcommand itself
//-------------------------------------------------------------------------------------

void testNNCountsDumpSubcommandHumanOutput() {
  ScopedTempDir dir(TMP_DIR_PREFIX);
  const NNCacheCountLog log = NNCacheCountLog::forContext(dir.path(), "cmdhuman");
  log.appendDump(deltaOf({{1, 7}}, 0));
  log.appendDump(deltaOf({{1, 2}, {2, 9}}, 3));

  int exitCode = -1;
  const string output = captureStdout(
    [&]() { return MainCmds::nncountsdump({"nncountsdump", log.path()}); }, exitCode
  );
  testAssert(exitCode == 0);
  testAssert(output.find("block 0 (one cache_dump's append)") != string::npos);
  testAssert(output.find("block 1 (one cache_dump's append)") != string::npos);
  testAssert(output.find(nthKey(1).toString()) != string::npos);
  // The currency line is asserted by CONSTANT, not by pasting its literal text here again:
  // the wording is owned by NNCacheCountLog::countsCurrencyDescription() and is due to
  // change under nncache-observation-currency (ledger rows 1717/1722/1723). Hard-coding
  // today's sentence into this test would make that landing break a test that has nothing
  // to say about wording.
  testAssert(output.find(NNCacheCountLog::countsCurrencyDescription()) != string::npos);
  testAssert(output.find("accumulated totals") != string::npos);
  testAssert(output.find("tail:") != string::npos);
}

void testNNCountsDumpSubcommandJsonOutput() {
  ScopedTempDir dir(TMP_DIR_PREFIX);
  const NNCacheCountLog log = NNCacheCountLog::forContext(dir.path(), "cmdjson");
  log.appendDump(deltaOf({{1, 7}, {2, 0}}, 0));
  log.appendDump(deltaOf({{1, 3}}, 1));

  int exitCode = -1;
  const string output = captureStdout(
    [&]() { return MainCmds::nncountsdump({"nncountsdump", log.path(), "--json"}); }, exitCode
  );
  testAssert(exitCode == 0);

  const json parsed = json::parse(output);
  testAssert(parsed["context"] == "cmdjson");
  testAssert(parsed["countsAre"] == NNCacheCountLog::countsCurrencyDescription());
  testAssert(parsed["blocks"].size() == 2);
  testAssert(parsed["blocks"][0]["rows"].size() == 2);
  testAssert(parsed["blocks"][1]["rows"].size() == 1);
  testAssert(parsed["totals"]["rows"] == 2);
  testAssert(parsed["totals"]["blocksApplied"] == 2);
  testAssert(parsed["totals"]["totalLookups"] == 10);
  testAssert(parsed["tail"] == "intact");
}

void testNNCountsDumpSubcommandRefusesAMissingFile() {
  ScopedTempDir dir(TMP_DIR_PREFIX);
  int exitCode = -1;
  const string output = captureStdout(
    [&]() { return MainCmds::nncountsdump({"nncountsdump", dir.path() + "/nosuchfile.nncounts"}); },
    exitCode
  );
  (void)output;
  testAssert(exitCode != 0);
}

void testNNCountsDumpSubcommandRefusesTheWrongExtension() {
  ScopedTempDir dir(TMP_DIR_PREFIX);
  const string path = dir.path() + "/notacountlog.txt";
  FILE* f = fopen(path.c_str(), "wb");
  testAssert(f != NULL);
  fclose(f);
  int exitCode = -1;
  const string output = captureStdout(
    [&]() { return MainCmds::nncountsdump({"nncountsdump", path}); }, exitCode
  );
  (void)output;
  testAssert(exitCode != 0);
}

}  // namespace

void Tests::runNNCountsDumpTests() {
  cout << "Running nncountsdump tests" << endl;

  testDetailedLoadShowsPerBlockDeltasAndAccumulatedTotals();
  testDetailedLoadReportsTornTailAndKeepsThePrefix();
  testNNCountsDumpSubcommandHumanOutput();
  testNNCountsDumpSubcommandJsonOutput();
  testNNCountsDumpSubcommandRefusesAMissingFile();
  testNNCountsDumpSubcommandRefusesTheWrongExtension();
}
