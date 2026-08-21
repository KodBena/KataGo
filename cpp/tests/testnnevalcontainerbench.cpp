#include "../tests/tests.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <ghc/filesystem.hpp>

#include "../core/fileutils.h"
#include "../neuralnet/nnevalcontainer.h"

using namespace std;

namespace gfs = ghc::filesystem;

// THE LOAD/APPEND I/O MEASUREMENT for the evaluation container.
//
// It is a measurement and not an assertion, so it is deliberately not part of runtests --
// the same disposition runnncachecountlogbench already has, and for the same two reasons:
// nothing here has a pass/fail answer, and it writes real files of a size no test suite
// should scatter, so the directory is named on the command line rather than defaulted.
//
// WHAT IT EXISTS TO ANSWER. The deployment's per-card containers reach 10-20 GB and the
// corpus does not fit page cache, so three costs decide whether the format is usable at
// that size, and none of them is visible from the format's own arithmetic:
//
//   THE REPAIR'S WRITE VOLUME. A torn tail -- an engine SIGKILLed mid-dump -- is repaired
//   before the next append. What that repair costs the DEVICE (not the format) is read
//   from /proc/self/io's write_bytes, which is the kernel's own count of bytes this
//   process caused to be sent to the storage layer, sampled either side of one append.
//   A repair that rewrites the file reads as the file's size here; one that truncates
//   reads as the appended block.
//
//   THE PAGE-CACHE FOOTPRINT. A load that leaves the file resident evicts the system's
//   working set. mincore(2) over a mapping of the file is the direct observation of
//   which of its pages are resident, and it is taken AFTER the load rather than inferred
//   from what the code asked the kernel for.
//
//   THE WALL TIME of a load whose file is not in page cache, which is the only load the
//   deployment ever performs. Every timed load below is preceded by an explicit
//   whole-file POSIX_FADV_DONTNEED, so "cold" is a state that was established and then
//   verified by mincore, not one that was hoped for.
//
// Linux-first by construction: /proc/self/io, mincore and posix_fadvise are Linux's. On
// another platform each figure says UNAVAILABLE and names why, rather than printing a
// number it did not observe.

namespace {

//-------------------------------------------------------------------------------------
// The substrate probes
//-------------------------------------------------------------------------------------

// A named field of /proc/self/io in bytes, or -1 if this platform does not report it.
// read_bytes and write_bytes are the kernel's own counts of what this process caused to be
// fetched from and sent to the storage layer -- which is a different fact from what the
// code asked for, since a re-read served by page cache costs nothing there.
int64_t procSelfIoField(const char* field) {
  FILE* f = fopen("/proc/self/io", "rb");
  if(f == NULL)
    return -1;
  char buf[4096];
  const size_t n = fread(buf, 1, sizeof(buf) - 1, f);
  fclose(f);
  buf[n] = '\0';
  // Matched at the start of a line, so "read_bytes" never matches inside another field.
  const string needle = string("\n") + field + ":";
  const char* p = strstr(buf, needle.c_str());
  if(p == NULL)
    return -1;
  return (int64_t)strtoll(p + needle.size(), NULL, 10);
}

int64_t writeBytesSoFar() { return procSelfIoField("write_bytes"); }
int64_t readBytesSoFar() { return procSelfIoField("read_bytes"); }

// A named field of /proc/meminfo in kB, or -1. Used for the SYSTEM-WIDE page-cache delta,
// which is the fact a per-file residency count cannot state: whether the load inflated the
// cache the rest of the machine is competing for.
int64_t meminfoKB(const char* field) {
  FILE* f = fopen("/proc/meminfo", "rb");
  if(f == NULL)
    return -1;
  vector<char> buf(65536);
  const size_t n = fread(buf.data(), 1, buf.size() - 1, f);
  fclose(f);
  buf[n] = '\0';
  // The field name is matched at the start of a line, so "Cached" never matches "SwapCached".
  const string needle = string("\n") + field + ":";
  const char* p = strstr(buf.data(), needle.c_str());
  if(p == NULL)
    return -1;
  return (int64_t)strtoll(p + needle.size(), NULL, 10);
}

struct Residency {
  bool available = false;
  int64_t residentPages = 0;
  int64_t totalPages = 0;
  int64_t pageSize = 4096;
};

// How many of `path`'s pages are in page cache RIGHT NOW, by mincore(2) over a mapping of
// it. The mapping is created and dropped inside this function and is never dereferenced, so
// the act of measuring does not itself make a page resident.
Residency residencyOf(const string& path) {
  Residency r;
#ifndef _WIN32
  const int fd = ::open(path.c_str(), O_RDONLY);
  if(fd < 0)
    return r;
  struct stat st;
  if(::fstat(fd, &st) != 0 || st.st_size <= 0) {
    ::close(fd);
    return r;
  }
  const size_t pageSize = (size_t)::sysconf(_SC_PAGESIZE);
  const size_t length = (size_t)st.st_size;
  void* map = ::mmap(NULL, length, PROT_READ, MAP_SHARED, fd, 0);
  if(map == MAP_FAILED) {
    ::close(fd);
    return r;
  }
  const size_t pages = (length + pageSize - 1) / pageSize;
  vector<unsigned char> vec(pages);
  if(::mincore(map, length, vec.data()) == 0) {
    r.available = true;
    r.totalPages = (int64_t)pages;
    r.pageSize = (int64_t)pageSize;
    for(size_t i = 0; i < pages; i++)
      r.residentPages += (vec[i] & 1) ? 1 : 0;
  }
  ::munmap(map, length);
  ::close(fd);
#else
  (void)path;
#endif
  return r;
}

// Asks the kernel to drop every page of `path` from page cache, and reports whether it did.
// This is the COLD-CACHE PROTOCOL: `echo 3 > drop_caches` needs privileges this measurement
// does not have, and a per-file DONTNEED is the unprivileged equivalent that names exactly
// the file it is about.
bool dropPageCacheFor(const string& path) {
#ifndef _WIN32
  const int fd = ::open(path.c_str(), O_RDONLY);
  if(fd < 0)
    return false;
  // The dirty pages have to reach the device before DONTNEED will evict them; without this
  // the drop silently keeps whatever was still dirty.
  (void)::fsync(fd);
  const int rc = ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
  ::close(fd);
  return rc == 0;
#else
  (void)path;
  return false;
#endif
}

void printResidency(const char* label, const Residency& r) {
  cout << "    " << label << ": ";
  if(!r.available) {
    cout << "UNAVAILABLE (no mincore on this platform)" << endl;
    return;
  }
  const double pct = r.totalPages > 0 ? 100.0 * (double)r.residentPages / (double)r.totalPages : 0.0;
  cout << r.residentPages << " / " << r.totalPages << " pages resident ("
       << pct << "%, " << ((double)(r.residentPages * r.pageSize) / 1048576.0) << " MiB)" << endl;
}

void printBytes(const char* label, int64_t bytes) {
  cout << "    " << label << ": ";
  if(bytes < 0)
    cout << "UNAVAILABLE (no /proc/self/io on this platform)" << endl;
  else
    cout << bytes << " bytes (" << ((double)bytes / 1048576.0) << " MiB)" << endl;
}

double secondsSince(const chrono::steady_clock::time_point& t0) {
  return chrono::duration<double>(chrono::steady_clock::now() - t0).count();
}

//-------------------------------------------------------------------------------------
// The synthetic container
//-------------------------------------------------------------------------------------

const char* BENCH_MODEL = "kata1-b18c384nbt-s9732312320-d4245566942";
const int BENCH_MODEL_VERSION = 14;
const int BENCH_EDGE = 19;
const bool BENCH_OWNERMAP = true;

Hash128 nthKey(int64_t serial) {
  return Hash128(
    ((uint64_t)(serial + 1)) * 0x9E3779B97F4A7C15ULL,
    ((uint64_t)(serial + 1)) * 0xD6E8FEB86659FD93ULL + 0x1234567ULL
  );
}

// One evaluation, filled from its serial so that nothing about the measurement depends on
// the bytes compressing -- the format does not compress, and a container of one repeated
// value would let a filesystem that does make these figures a fiction.
shared_ptr<NNOutput> makeOutput(int64_t serial, int nnXLen, int nnYLen, bool withOwnerMap) {
  shared_ptr<NNOutput> out = make_shared<NNOutput>();
  out->nnHash = nthKey(serial);
  const float g = (float)(serial % 1024) * 0.001f;
  out->whiteWinProb = 0.5f + g;
  out->whiteLossProb = 0.5f - g;
  out->whiteNoResultProb = 0.0f;
  out->whiteScoreMean = g * 10.0f;
  out->whiteScoreMeanSq = g * 100.0f;
  out->whiteLead = g * 5.0f;
  out->varTimeLeft = 42.5f;
  out->shorttermWinlossError = g;
  out->shorttermScoreError = g;
  out->policyOptimismUsed = 0.5f;
  out->nnXLen = nnXLen;
  out->nnYLen = nnYLen;
  const int area = nnXLen * nnYLen;
  for(int i = 0; i < NNPos::MAX_NN_POLICY_SIZE; i++)
    out->policyProbs[i] = 0.0f;
  for(int i = 0; i <= area; i++)
    out->policyProbs[i] = (float)((serial * 7 + i * 13) % 9973) * 0.0001f;
  if(withOwnerMap) {
    out->whiteOwnerMap = new float[area];
    for(int i = 0; i < area; i++)
      out->whiteOwnerMap[i] = (float)((serial * 11 + i * 17) % 2001) * 0.001f - 1.0f;
  }
  return out;
}

vector<shared_ptr<const NNOutput>> makeDump(int64_t serialBase, int64_t entriesPerDump) {
  vector<shared_ptr<const NNOutput>> entries;
  entries.reserve((size_t)entriesPerDump);
  for(int64_t i = 0; i < entriesPerDump; i++)
    entries.push_back(
      shared_ptr<const NNOutput>(makeOutput(serialBase + i, BENCH_EDGE, BENCH_EDGE, BENCH_OWNERMAP)));
  return entries;
}

int64_t sizeOfOrZero(const string& path) {
  if(!FileUtils::exists(path))
    return 0;
  return (int64_t)gfs::file_size(gfs::u8path(path));
}

}  // namespace

//-------------------------------------------------------------------------------------
// The bench
//-------------------------------------------------------------------------------------

void Tests::runNNEvalContainerLoadIOBench(
  const string& directory, int64_t targetMiB, int64_t entriesPerDump, bool doFullLoad
) {
  if(!FileUtils::isDirectory(directory))
    throw StringError("runnnevalcontainerbench: '" + directory + "' is not an existing directory.");
  if(targetMiB < 1)
    throw StringError("runnnevalcontainerbench: the target size in MiB must be at least 1.");
  if(entriesPerDump < 1)
    throw StringError("runnnevalcontainerbench: the entries per dump must be at least 1.");

  const NNEvalContainer container =
    NNEvalContainer::forContextAndModel(directory, "loadiobench", BENCH_MODEL, BENCH_MODEL_VERSION);

  // A fresh container every run: a measurement that resumed a leftover file would be
  // measuring a different file than the one it reports.
  if(FileUtils::exists(container.path()))
    gfs::remove(gfs::u8path(container.path()));

  const int64_t target = targetMiB * 1048576;
  const int64_t bytesPerEntry = NNEvalContainer::bytesForEntry(BENCH_EDGE, BENCH_EDGE, BENCH_OWNERMAP);
  cout << "=== synthesis ===" << endl;
  cout << "  target=" << targetMiB << " MiB  entriesPerDump=" << entriesPerDump
       << "  bytesPerEntry=" << bytesPerEntry << "  board=" << BENCH_EDGE << "x" << BENCH_EDGE
       << "  ownermap=" << (BENCH_OWNERMAP ? "yes" : "no") << endl;

  int64_t serial = 1;
  int64_t dumps = 0;
  double synthSeconds = 0.0;
  int64_t synthWriteBytes = 0;
  int64_t synthReadBytes = 0;
  bool ioAvailable = true;
  while(sizeOfOrZero(container.path()) < target) {
    const vector<shared_ptr<const NNOutput>> entries = makeDump(serial, entriesPerDump);
    const int64_t wBefore = writeBytesSoFar();
    const int64_t rBefore = readBytesSoFar();
    const chrono::steady_clock::time_point t0 = chrono::steady_clock::now();
    const NNEvalContainerAppendResult appended = container.appendBlock(entries);
    const double seconds = secondsSince(t0);
    const int64_t wAfter = writeBytesSoFar();
    const int64_t rAfter = readBytesSoFar();

    serial += entriesPerDump;
    dumps += 1;
    synthSeconds += seconds;
    if(wBefore < 0 || wAfter < 0 || rBefore < 0 || rAfter < 0)
      ioAvailable = false;
    else {
      synthWriteBytes += wAfter - wBefore;
      synthReadBytes += rAfter - rBefore;
    }
    cout << "  dump " << dumps << ": " << seconds << " s"
         << "  formatBytes=" << appended.bytesAppended;
    if(ioAvailable)
      cout << "  write_bytes=" << (wAfter - wBefore) << "  read_bytes=" << (rAfter - rBefore);
    cout << "  fileSize=" << sizeOfOrZero(container.path()) << endl;
  }
  const int64_t fileSize = sizeOfOrZero(container.path());
  cout << "  TOTAL: dumps=" << dumps << "  entries=" << (serial - 1)
       << "  fileSize=" << fileSize << " (" << ((double)fileSize / 1073741824.0) << " GiB)"
       << "  seconds=" << synthSeconds << endl;
  if(ioAvailable) {
    printBytes("synthesis write_bytes", synthWriteBytes);
    printBytes("synthesis read_bytes", synthReadBytes);
  }

  //-----------------------------------------------------------------------------------
  // The cold key-set scan: what an attach's first pass costs.
  //-----------------------------------------------------------------------------------
  cout << "=== cold loadIndex (the key-set scan) ===" << endl;
  {
    const bool dropped = dropPageCacheFor(container.path());
    cout << "    cold-cache protocol: whole-file POSIX_FADV_DONTNEED "
         << (dropped ? "applied" : "UNAVAILABLE") << endl;
    printResidency("residency before", residencyOf(container.path()));
    const int64_t cachedBefore = meminfoKB("Cached");
    const int64_t rBefore = readBytesSoFar();
    const chrono::steady_clock::time_point t0 = chrono::steady_clock::now();
    const NNEvalContainerIndex index = container.loadIndex();
    const double seconds = secondsSince(t0);
    const int64_t rAfter = readBytesSoFar();
    const int64_t cachedAfter = meminfoKB("Cached");
    cout << "    seconds=" << seconds
         << "  liveKeys=" << index.entries().size()
         << "  blocksApplied=" << index.blocksApplied()
         << "  entriesApplied=" << index.entriesApplied() << endl;
    if(seconds > 0.0)
      cout << "    throughput=" << ((double)fileSize / 1048576.0 / seconds) << " MiB/s" << endl;
    printBytes("read_bytes", (rBefore < 0 || rAfter < 0) ? -1 : rAfter - rBefore);
    printResidency("residency after", residencyOf(container.path()));
    if(cachedBefore >= 0 && cachedAfter >= 0)
      cout << "    /proc/meminfo Cached delta: " << (cachedAfter - cachedBefore) << " kB" << endl;
    else
      cout << "    /proc/meminfo Cached delta: UNAVAILABLE" << endl;
  }

  //-----------------------------------------------------------------------------------
  // The cold full load: every payload decoded. Optional, because it holds the whole live
  // set in memory and a machine smaller than the container cannot perform it at all --
  // which is a fact about the substrate, stated rather than crashed into.
  //-----------------------------------------------------------------------------------
  if(doFullLoad) {
    cout << "=== cold load (every payload decoded) ===" << endl;
    const bool dropped = dropPageCacheFor(container.path());
    cout << "    cold-cache protocol: whole-file POSIX_FADV_DONTNEED "
         << (dropped ? "applied" : "UNAVAILABLE") << endl;
    printResidency("residency before", residencyOf(container.path()));
    const int64_t cachedBefore = meminfoKB("Cached");
    const int64_t rBefore = readBytesSoFar();
    const chrono::steady_clock::time_point t0 = chrono::steady_clock::now();
    const NNEvalContainerContents contents = container.load();
    const double seconds = secondsSince(t0);
    const int64_t rAfter = readBytesSoFar();
    const int64_t cachedAfter = meminfoKB("Cached");
    cout << "    seconds=" << seconds
         << "  liveEntries=" << contents.entries().size()
         << "  blocksApplied=" << contents.blocksApplied() << endl;
    if(seconds > 0.0)
      cout << "    throughput=" << ((double)fileSize / 1048576.0 / seconds) << " MiB/s" << endl;
    printBytes("read_bytes", (rBefore < 0 || rAfter < 0) ? -1 : rAfter - rBefore);
    printResidency("residency after", residencyOf(container.path()));
    if(cachedBefore >= 0 && cachedAfter >= 0)
      cout << "    /proc/meminfo Cached delta: " << (cachedAfter - cachedBefore) << " kB" << endl;
    else
      cout << "    /proc/meminfo Cached delta: UNAVAILABLE" << endl;
  }
  else {
    cout << "=== cold load (every payload decoded) === SKIPPED (not requested)" << endl;
  }

  //-----------------------------------------------------------------------------------
  // The torn tail: the SIGKILL-mid-dump case, and what the repair costs the device.
  //-----------------------------------------------------------------------------------
  cout << "=== torn-tail repair ===" << endl;
  {
    const int64_t sizeBefore = sizeOfOrZero(container.path());
    // A kill mid-dump synthesised: the last block loses its final bytes, exactly as a
    // partial flush would leave it.
    const int64_t tornBy = 4096;
    gfs::resize_file(gfs::u8path(container.path()), (uintmax_t)(sizeBefore - tornBy));
    cout << "    truncated " << tornBy << " bytes off a " << sizeBefore
         << "-byte container to synthesise a kill mid-dump" << endl;

    const bool dropped = dropPageCacheFor(container.path());
    cout << "    cold-cache protocol: whole-file POSIX_FADV_DONTNEED "
         << (dropped ? "applied" : "UNAVAILABLE") << endl;

    const vector<shared_ptr<const NNOutput>> entries = makeDump(serial, entriesPerDump);
    const int64_t wBefore = writeBytesSoFar();
    const int64_t rBefore = readBytesSoFar();
    const chrono::steady_clock::time_point t0 = chrono::steady_clock::now();
    const NNEvalContainerAppendResult appended = container.appendBlock(entries);
    const double seconds = secondsSince(t0);
    const int64_t wAfter = writeBytesSoFar();
    const int64_t rAfter = readBytesSoFar();

    cout << "    seconds=" << seconds
         << "  tornTailBytesDiscarded=" << appended.tornTailBytesDiscarded
         << "  rewroteTheFile=" << (appended.rewroteTheFile ? "true" : "false")
         << "  bytesAppended=" << appended.bytesAppended << endl;
    printBytes("repair+append write_bytes", (wBefore < 0 || wAfter < 0) ? -1 : wAfter - wBefore);
    printBytes("repair+append read_bytes", (rBefore < 0 || rAfter < 0) ? -1 : rAfter - rBefore);
    cout << "    file size before torn=" << sizeBefore
         << "  after repair+append=" << sizeOfOrZero(container.path()) << endl;

    // And the proof that the repaired container still loads: the key set is read back and
    // the block count reported, so a repair that silently lost the surviving prefix is
    // visible here rather than inferred from the byte counts above.
    const NNEvalContainerIndex index = container.loadIndex();
    cout << "    after repair: tail=" << (index.tail() == NNEvalContainerTail::Intact ? "Intact" : "Truncated")
         << "  blocksApplied=" << index.blocksApplied()
         << "  entriesApplied=" << index.entriesApplied()
         << "  liveKeys=" << index.entries().size() << endl;
  }

  cout << "=== done; the container is left at " << container.path() << " ===" << endl;
}
