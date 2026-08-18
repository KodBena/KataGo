#include "../tests/tests.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <ghc/filesystem.hpp>

#include "../core/fileutils.h"
#include "../neuralnet/nncache.h"
#include "../neuralnet/nncachecountlog.h"

using namespace std;

namespace gfs = ghc::filesystem;

// THE MEASUREMENT of write volume per dump for the count log.
//
// It is a measurement, not an assertion, so it is deliberately not part of runtests --
// the same disposition runnncachefrozenbench already has on this branch.
//
// WHAT IS MEASURED AND WHERE THE NUMBER COMES FROM. /proc/self/io's write_bytes is the
// kernel's own count of bytes this process has caused to be sent to the storage layer. It
// is read immediately before and immediately after a single appendDump, with an fsync
// inside that dump, so the delta is what the dump cost the device and not what the format's
// arithmetic says it should have cost. The two are printed side by side precisely so a
// disagreement is visible rather than assumed away.
//
// Linux-only by construction: /proc/self/io does not exist elsewhere. On another platform
// this says so and prints no figure, rather than printing one it did not observe.

namespace {

Hash128 nthKey(int64_t serial) {
  return Hash128(
    ((uint64_t)(serial + 1)) * 0x9E3779B97F4A7C15ULL,
    ((uint64_t)(serial + 1)) * 0xD6E8FEB86659FD93ULL + 0x1234567ULL
  );
}

// The process's cumulative write_bytes, or -1 if this platform does not report it.
int64_t writeBytesSoFar() {
  FILE* f = fopen("/proc/self/io", "rb");
  if(f == NULL)
    return -1;
  char buf[4096];
  const size_t n = fread(buf, 1, sizeof(buf) - 1, f);
  fclose(f);
  buf[n] = '\0';
  const char* p = strstr(buf, "write_bytes:");
  if(p == NULL)
    return -1;
  return (int64_t)strtoll(p + strlen("write_bytes:"), NULL, 10);
}

NNCacheHitLedger ledgerOfSize(int64_t numRows, int64_t serialBase) {
  vector<NNCacheHitCount> rows;
  rows.reserve((size_t)numRows);
  for(int64_t i = 0; i < numRows; i++) {
    NNCacheHitCount row;
    row.key = nthKey(serialBase + i);
    // A plausible spread rather than a constant, so nothing about the measurement depends
    // on the values compressing -- the format does not compress, and this makes that
    // visible rather than accidental.
    row.hits = (uint32_t)(1 + (i % 97));
    rows.push_back(row);
  }
  return NNCacheHitLedger::counted(std::move(rows), 0);
}

void measureOneDumpSize(const string& dir, int64_t numRows, int serial) {
  const string context = "bench" + Global::intToString(serial);
  const NNCacheCountLog log = NNCacheCountLog::forContext(dir, context);
  const NNCacheHitLedger ledger = ledgerOfSize(numRows, 1);

  const int64_t before = writeBytesSoFar();
  const NNCacheCountLogAppendResult appended = log.appendDump(ledger);
  const int64_t after = writeBytesSoFar();

  const int64_t predicted = (int64_t)NNCacheCountLog::fileHeaderBytes() + NNCacheCountLog::bytesForDumpOf(numRows);
  cout << "rows=" << numRows
       << "  format bytes=" << appended.bytesAppended
       << "  predicted=" << predicted
       << "  file size=" << (int64_t)gfs::file_size(gfs::u8path(log.path()));
  if(before < 0 || after < 0)
    cout << "  measured write_bytes=UNAVAILABLE (no /proc/self/io on this platform)";
  else
    cout << "  measured write_bytes delta=" << (after - before);
  cout << endl;

  gfs::remove(gfs::u8path(log.path()));
}

// The second dump onto an existing file: the case the design is actually optimised for,
// where nothing but the changed set is written.
void measureSecondDumpOntoAnExistingLog(const string& dir, int64_t numRows) {
  const NNCacheCountLog log = NNCacheCountLog::forContext(dir, "benchsecond");
  log.appendDump(ledgerOfSize(numRows, 1));

  const NNCacheHitLedger ledger = ledgerOfSize(numRows, 1);
  const int64_t before = writeBytesSoFar();
  const NNCacheCountLogAppendResult appended = log.appendDump(ledger);
  const int64_t after = writeBytesSoFar();

  cout << "second dump onto an existing " << numRows << "-row log:"
       << "  format bytes=" << appended.bytesAppended
       << "  rewrote=" << (appended.rewroteTheFile ? "yes" : "no");
  if(before < 0 || after < 0)
    cout << "  measured write_bytes=UNAVAILABLE";
  else
    cout << "  measured write_bytes delta=" << (after - before);
  cout << endl;

  // And the compaction, which is the one operation whose volume is the LIVE SET rather than
  // the changed set. Reported beside the appends so the amortised cost is visible.
  const int64_t beforeCompact = writeBytesSoFar();
  log.compact();
  const int64_t afterCompact = writeBytesSoFar();
  cout << "compaction of the same log (live set " << numRows << " rows):"
       << "  file size=" << (int64_t)gfs::file_size(gfs::u8path(log.path()));
  if(beforeCompact < 0 || afterCompact < 0)
    cout << "  measured write_bytes=UNAVAILABLE";
  else
    cout << "  measured write_bytes delta=" << (afterCompact - beforeCompact);
  cout << endl;

  gfs::remove(gfs::u8path(log.path()));
}

}  // namespace

void Tests::runNNCacheCountLogBench(const string& directory) {
  cout << "Count log write-volume measurement." << endl;
  cout << "The target this is measured against: a dump bounded by the CHANGED SET, about "
       << "40000 x 24 B = 960000 B, rather than by anything larger." << endl;

  if(!FileUtils::isDirectory(directory))
    throw StringError("runnncachecountlogbench: '" + directory + "' is not an existing directory.");
  cout << "Writing under: " << directory << endl;

  if(writeBytesSoFar() < 0)
    cout << "NOTE: /proc/self/io is unavailable on this platform, so no measured figure "
         << "is reported below -- only the format's own arithmetic." << endl;

  const int64_t sizes[] = {1000, 10000, 40000, 45664, 291129};
  for(size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++)
    measureOneDumpSize(directory, sizes[i], (int)i);

  measureSecondDumpOntoAnExistingLog(directory, 40000);
}
