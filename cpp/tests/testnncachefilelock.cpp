#include "../tests/tests.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "../core/fileutils.h"
#include "../core/global.h"
#include "../neuralnet/nncachecountlog.h"
#include "../neuralnet/nncachefileformat.h"
#include "../neuralnet/nnevalcontainer.h"

using namespace std;
using namespace TestCommon;

// Correctness tests for the cross-process lock that lets several engine processes share one
// cache directory (NNCacheFileLock, nncachefileformat.h).
//
// WHAT THESE TESTS HAVE TO WITNESS, AND WHY "IT DIDN'T CRASH" IS NOT IT. The failure this
// lock prevents is silent and it is not a crash. A block is written through a reused
// per-entry buffer -- many small writes, not one -- so two unlocked appends interleave IN THE
// MIDDLE of the file. The reader then stops at the first block that fails its checksum and
// discards THE ENTIRE REMAINDER, so one interleaved append costs every block written after
// it, including blocks that were themselves perfectly good. A test that only checked "the
// file still parses" would pass on a file that had silently lost a whole dump: the tail would
// be intact precisely because everything past the damage was thrown away. So the assertions
// here are COUNTS -- blocksApplied and entriesApplied, and the count log's rows -- which is
// the only thing that distinguishes "both dumps landed" from "one dump landed and the other
// was discarded along with the evidence".
//
// WHY THE MULTI-PROCESS TESTS FORK. An advisory lock is a fact about OPEN FILE DESCRIPTIONS,
// not about threads, so a single-process test with two descriptors witnesses the exclusion
// mechanism honestly -- and two of the tests below do exactly that, because it is the sharper
// instrument for the compatibility matrix. But the property the feature exists for is stated
// about PROCESSES, and a fork is the smallest thing that produces two of them. fork() has no
// Windows equivalent; those tests are compiled out there rather than quietly reinterpreted,
// and the runner says so.

namespace {

//-------------------------------------------------------------------------------------
// Fixtures
//-------------------------------------------------------------------------------------

const char* TMP_DIR_PREFIX = "tmpnncachefilelock";
const char* MODEL = "kata1-b18c384nbt-s9732312320-d4245566942";
const int MODEL_VERSION = 14;
const char* CONTEXT = "card-5455";

// 19x19 with an ownership map, so one entry is ~2.9 KB of payload and a block of a few
// hundred is megabytes rather than kilobytes. That size is not decoration: the race these
// tests provoke is only reachable while a writer is actually inside its many small writes,
// and a block that fits in one buffer flush would close the window the test is aiming at.
const int NNXLEN = 19;
const int NNYLEN = 19;

Hash128 nthKey(int serial) {
  return Hash128(
    ((uint64_t)(serial + 1)) * 0x9E3779B97F4A7C15ULL,
    ((uint64_t)(serial + 1)) * 0xD6E8FEB86659FD93ULL + 0x1234567ULL
  );
}

shared_ptr<const NNOutput> makeOutput(int serial) {
  shared_ptr<NNOutput> out = make_shared<NNOutput>();
  out->nnHash = nthKey(serial);
  const float s = (float)serial;
  out->whiteWinProb = 0.125f;
  out->whiteLossProb = 0.25f;
  out->whiteNoResultProb = 0.0f;
  out->whiteScoreMean = -3.5f;
  out->whiteScoreMeanSq = 17.25f;
  out->whiteLead = 1.75f;
  out->varTimeLeft = 42.5f;
  out->shorttermWinlossError = 0.03125f;
  out->shorttermScoreError = 0.0625f;
  out->policyOptimismUsed = 0.5f;
  out->nnXLen = NNXLEN;
  out->nnYLen = NNYLEN;

  const int area = NNXLEN * NNYLEN;
  for(int i = 0; i < NNPos::MAX_NN_POLICY_SIZE; i++)
    out->policyProbs[i] = 0.0f;
  for(int i = 0; i <= area; i++)
    out->policyProbs[i] = (float)i * 0.5f + s;
  out->whiteOwnerMap = new float[area];
  for(int i = 0; i < area; i++)
    out->whiteOwnerMap[i] = ((float)i * 0.015625f) - 1.0f;
  return shared_ptr<const NNOutput>(out);
}

// `count` entries whose serials start at `firstSerial`. Two callers passing disjoint ranges
// produce disjoint key sets, which is what makes "both dumps landed" a countable fact rather
// than an inference from a total.
vector<shared_ptr<const NNOutput>> makeBlock(int firstSerial, int count) {
  vector<shared_ptr<const NNOutput>> out;
  out.reserve((size_t)count);
  for(int i = 0; i < count; i++)
    out.push_back(makeOutput(firstSerial + i));
  return out;
}

// The count log's half of a dump, over the same disjoint serial range. A dump writes BOTH of
// a context's files, so a witness that only looked at the container would leave the file the
// operator's admission policy actually reads out of the claim.
NNCacheHitCountDelta makeCountDelta(int firstSerial, int count) {
  vector<NNCacheHitCount> rows;
  rows.reserve((size_t)count);
  for(int i = 0; i < count; i++) {
    NNCacheHitCount row;
    row.key = nthKey(firstSerial + i);
    row.hits = (uint32_t)(i + 1);
    rows.push_back(row);
  }
  return NNCacheHitCountDelta::ofDeltaRows(rows, 0);
}

// True if this refusal names both the context and the lock file, which is the whole point of
// the message: an operator reading it has to be able to find the holder.
bool namesContextAndLockPath(const string& message, const string& directory, const string& context) {
  const string lockPath = NNCacheFileLock::pathForContext(directory, context);
  return message.find(context) != string::npos && message.find(lockPath) != string::npos;
}

// The refusal's message, or the empty string if the acquisition succeeded. Named rather than
// inlined at each site so the lock's [[nodiscard]]-shaped value is never quietly dropped in a
// bare try block.
string acquisitionRefusal(
  const string& directory, const string& context, NNCacheFileLockMode mode, int waitMs
) {
  try {
    const NNCacheFileLock lock = NNCacheFileLock::overContext(directory, context, mode, waitMs);
    (void)lock;
    return "";
  }
  catch(const StringError& e) {
    return e.what();
  }
}

int64_t nowMicros() {
  return (int64_t)chrono::duration_cast<chrono::microseconds>(
    chrono::system_clock::now().time_since_epoch()
  ).count();
}

#ifndef _WIN32

// The inode a path currently resolves to. This is what makes "the lock file was never
// renamed" a WITNESSED fact rather than an inspected one: a rename replaces the inode behind
// a name, so comparing the number across a compaction distinguishes the file that was
// replaced from the file that was not.
uint64_t inodeOf(const string& path) {
  struct stat st;
  testAssert(::stat(path.c_str(), &st) == 0);
  return (uint64_t)st.st_ino;
}

// Runs `body` in a forked child and returns its pid.
//
// The child leaves through _exit, which runs NO destructors and NO atexit handlers. That is
// deliberate and load-bearing: the parent's ScopedTempDir is on the child's stack too, and a
// child unwinding normally would delete the directory the parent is still testing.
pid_t forkChild(const std::function<void()>& body) {
  std::cout.flush();
  std::fflush(NULL);
  const pid_t pid = ::fork();
  testAssert(pid >= 0);
  if(pid == 0) {
    int status = 0;
    try {
      body();
    }
    catch(const std::exception& e) {
      std::fprintf(stderr, "child failed: %s\n", e.what());
      status = 1;
    }
    catch(...) {
      status = 1;
    }
    std::fflush(NULL);
    _exit(status);
  }
  return pid;
}

// Waits for the child and asserts it left cleanly. A child that threw is a FAILED TEST, not a
// quiet zero in a count: without this the multi-process tests would read "0 blocks applied"
// as a data-loss bug when the real event was a child that never ran.
void waitForChild(pid_t pid) {
  int status = 0;
  testAssert(::waitpid(pid, &status, 0) == pid);
  testAssert(WIFEXITED(status));
  testAssert(WEXITSTATUS(status) == 0);
}

// The children rendezvous on this file rather than starting when fork returns. Two forks are
// not simultaneous -- the first child can be finished before the second is scheduled -- and a
// race that never actually raced would witness nothing while passing.
void awaitStartSignal(const string& path) {
  for(int i = 0; i < 100000; i++) {
    if(FileUtils::exists(path))
      return;
    std::this_thread::sleep_for(chrono::microseconds(200));
  }
  throw StringError("test child: the start signal never arrived.");
}

void raiseStartSignal(const string& path) {
  std::FILE* f = std::fopen(path.c_str(), "wb");
  testAssert(f != NULL);
  testAssert(std::fclose(f) == 0);
}

// A child records when it BEGAN its attempt and when its dump RETURNED. The parent reads the
// two intervals back and reports whether they overlapped -- which is the evidence that the
// exclusion was actually exercised rather than that the two dumps happened to be sequential.
void writeInterval(const string& path, int64_t startMicros, int64_t endMicros) {
  std::FILE* f = std::fopen(path.c_str(), "wb");
  testAssert(f != NULL);
  std::fprintf(f, "%lld %lld\n", (long long)startMicros, (long long)endMicros);
  testAssert(std::fclose(f) == 0);
}

void readInterval(const string& path, int64_t& startMicros, int64_t& endMicros) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  testAssert(f != NULL);
  long long a = 0;
  long long b = 0;
  testAssert(std::fscanf(f, "%lld %lld", &a, &b) == 2);
  testAssert(std::fclose(f) == 0);
  startMicros = (int64_t)a;
  endMicros = (int64_t)b;
}

#endif  // _WIN32

//-------------------------------------------------------------------------------------
// The lock file is a file of its own
//-------------------------------------------------------------------------------------

// CRITERION 3, first half. The lock's own name is the context's, it sits beside the data
// files, and taking it touches neither of them.
void testTheLockLivesOnItsOwnFileBesideTheData() {
  ScopedTempDir dir(TMP_DIR_PREFIX);
  const string lockPath = NNCacheFileLock::pathForContext(dir.path(), CONTEXT);
  testAssert(lockPath == dir.path() + "/" + string(CONTEXT) + ".nnlock");
  testAssert(!FileUtils::exists(lockPath));

  {
    const NNCacheFileLock lock =
      NNCacheFileLock::overContext(dir.path(), CONTEXT, NNCacheFileLockMode::Exclusive);
    testAssert(lock.path() == lockPath);
    testAssert(lock.mode() == NNCacheFileLockMode::Exclusive);
    testAssert(FileUtils::exists(lockPath));
  }

  // Created but never written: its only content is its existence, so there is nothing in it a
  // torn write could damage and nothing a reader has to interpret.
  testAssert(FileUtils::exists(lockPath));
  std::FILE* f = std::fopen(lockPath.c_str(), "rb");
  testAssert(f != NULL);
  testAssert(std::fseek(f, 0, SEEK_END) == 0);
  testAssert(std::ftell(f) == 0);
  testAssert(std::fclose(f) == 0);

  // Neither data file was created by locking. The lock is not a side-channel that brings a
  // context's files into existence.
  testAssert(!FileUtils::exists(dir.path() + "/" + string(CONTEXT) + ".nncounts"));
  testAssert(!FileUtils::exists(dir.path() + "/" + string(CONTEXT) + "." + MODEL + ".nnevals"));
  cout << "Lock file " << lockPath << " is its own empty file, and locking created no data file" << endl;
}

// A context name that could not be a path component is refused by the lock exactly as it is
// by the two formats, whether or not a lock is actually taken.
void testABadContextNameIsRefusedByBothFormsAlike() {
  ScopedTempDir dir(TMP_DIR_PREFIX);
  bool pathRefused = false;
  try { (void)NNCacheFileLock::pathForContext(dir.path(), ".."); }
  catch(const StringError&) { pathRefused = true; }
  testAssert(pathRefused);

  bool acquireRefused = false;
  try {
    const NNCacheFileLock lock =
      NNCacheFileLock::overContext(dir.path(), "..", NNCacheFileLockMode::Shared);
    (void)lock;
  }
  catch(const StringError&) { acquireRefused = true; }
  testAssert(acquireRefused);
  cout << "A context name of \"..\" is refused by pathForContext and by overContext alike" << endl;
}

//-------------------------------------------------------------------------------------
// The compatibility matrix
//-------------------------------------------------------------------------------------

// CRITERION 2, the mechanism half: shared admits shared, and excludes exclusive.
//
// Two DESCRIPTIONS of one file, from one process, is a faithful instrument for this: flock
// excludes open file descriptions, and it does not care whether two of them are held by one
// process or two. What this cannot witness is the PROCESS-level property, which the forked
// tests below take up.
void testSharedLocksAdmitEachOtherAndExcludeAWriter() {
  ScopedTempDir dir(TMP_DIR_PREFIX);
  const NNCacheFileLock reader =
    NNCacheFileLock::overContext(dir.path(), CONTEXT, NNCacheFileLockMode::Shared);
  (void)reader;

  // A second reader, at zero wait, so a pass cannot be an artifact of patience.
  const NNCacheFileLock secondReader =
    NNCacheFileLock::overContext(dir.path(), CONTEXT, NNCacheFileLockMode::Shared, 0);
  testAssert(secondReader.mode() == NNCacheFileLockMode::Shared);

  const string refusal =
    acquisitionRefusal(dir.path(), CONTEXT, NNCacheFileLockMode::Exclusive, 0);
  testAssert(refusal != "");
  testAssert(namesContextAndLockPath(refusal, dir.path(), CONTEXT));
  cout << "Two shared holders coexist; a writer is refused: " << refusal << endl;
}

// The exclusive lock excludes both kinds, which is what makes a dump atomic against a
// concurrent attach as well as against a concurrent dump.
void testTheExclusiveLockExcludesReadersAndWritersAlike() {
  ScopedTempDir dir(TMP_DIR_PREFIX);
  const NNCacheFileLock writer =
    NNCacheFileLock::overContext(dir.path(), CONTEXT, NNCacheFileLockMode::Exclusive);
  (void)writer;

  const string readerRefusal =
    acquisitionRefusal(dir.path(), CONTEXT, NNCacheFileLockMode::Shared, 0);
  const string writerRefusal =
    acquisitionRefusal(dir.path(), CONTEXT, NNCacheFileLockMode::Exclusive, 0);
  testAssert(readerRefusal != "");
  testAssert(writerRefusal != "");
  testAssert(namesContextAndLockPath(readerRefusal, dir.path(), CONTEXT));
  testAssert(namesContextAndLockPath(writerRefusal, dir.path(), CONTEXT));
  cout << "An exclusive holder refuses a reader: " << readerRefusal << endl;
}

// Two DIFFERENT contexts do not contend. The unit of exclusion is the context, and a
// directory-wide lock -- the rejected alternative -- would have serialised these.
void testTwoContextsInOneDirectoryDoNotContend() {
  ScopedTempDir dir(TMP_DIR_PREFIX);
  const NNCacheFileLock a =
    NNCacheFileLock::overContext(dir.path(), "card-a", NNCacheFileLockMode::Exclusive);
  const NNCacheFileLock b =
    NNCacheFileLock::overContext(dir.path(), "card-b", NNCacheFileLockMode::Exclusive, 0);
  testAssert(a.path() != b.path());
  cout << "Exclusive locks on card-a and card-b coexist in one directory" << endl;
}

// A negative wait has no reading and is refused rather than silently treated as zero.
void testANegativeWaitIsRefused() {
  ScopedTempDir dir(TMP_DIR_PREFIX);
  bool refused = false;
  string message;
  try {
    const NNCacheFileLock lock =
      NNCacheFileLock::overContext(dir.path(), CONTEXT, NNCacheFileLockMode::Shared, -1);
    (void)lock;
  }
  catch(const StringError& e) { refused = true; message = e.what(); }
  testAssert(refused);
  cout << "A negative wait is refused: " << message << endl;
}

//-------------------------------------------------------------------------------------
// The multi-process properties
//-------------------------------------------------------------------------------------

#ifndef _WIN32

// CRITERION 4. A holder in ANOTHER PROCESS is waited out to the deadline and then reported by
// name. The deadline here is passed explicitly and is short, because what is under test is
// that the bound exists and that the refusal identifies the holder's file -- not the
// production constant, which is a policy choice about how long a real dump may take.
void testAWaitPastTheDeadlineThrowsNamingTheContextAndThePath() {
  ScopedTempDir dir(TMP_DIR_PREFIX);
  const string signal = dir.path() + "/held";
  const string directory = dir.path();

  const pid_t child = forkChild([directory, signal]() {
    const NNCacheFileLock held =
      NNCacheFileLock::overContext(directory, CONTEXT, NNCacheFileLockMode::Exclusive);
    (void)held;
    raiseStartSignal(signal);
    std::this_thread::sleep_for(chrono::milliseconds(1500));
  });

  awaitStartSignal(signal);
  const int64_t begun = nowMicros();
  const string refusal =
    acquisitionRefusal(directory, CONTEXT, NNCacheFileLockMode::Exclusive, 300);
  const int64_t waited = nowMicros() - begun;

  testAssert(refusal != "");
  testAssert(namesContextAndLockPath(refusal, directory, CONTEXT));
  // It waited, and it stopped waiting. Both halves are the contract: an immediate failure
  // would turn ordinary contention into a client-visible error, and an unbounded wait would
  // turn a wedged peer into a wedged engine carrying no diagnostic.
  testAssert(waited >= 300LL * 1000LL);
  testAssert(waited < 1400LL * 1000LL);
  cout << "Waited " << (waited / 1000) << " ms for a lock another process held, then: "
       << refusal << endl;
  waitForChild(child);
}

// CRITERION 3, second half, as a WITNESSED fact. A compaction replaces the data file's inode
// behind its name; the lock file's inode is unchanged. Had the lock been taken on the data
// file, the exclusion would have evaporated at exactly this moment -- the holder would still
// hold an inode, and it would no longer be the inode the path names.
void testCompactionReplacesTheDataInodeAndNeverTheLockInode() {
  ScopedTempDir dir(TMP_DIR_PREFIX);
  const NNEvalContainer container =
    NNEvalContainer::forContextAndModel(dir.path(), CONTEXT, MODEL, MODEL_VERSION);

  // Two blocks holding the SAME keys, so the live set is half the physical entries and the
  // compaction trigger fires.
  (void)container.appendBlock(makeBlock(0, 8));
  (void)container.appendBlock(makeBlock(0, 8));
  const string lockPath = NNCacheFileLock::pathForContext(dir.path(), CONTEXT);
  testAssert(FileUtils::exists(lockPath));

  const uint64_t lockInodeBefore = inodeOf(lockPath);
  const uint64_t dataInodeBefore = inodeOf(container.path());
  const bool compacted = container.compactIfNeeded(1);
  testAssert(compacted);
  const uint64_t lockInodeAfter = inodeOf(lockPath);
  const uint64_t dataInodeAfter = inodeOf(container.path());

  // The data file WAS replaced -- proving the rename this test is about actually happened.
  testAssert(dataInodeAfter != dataInodeBefore);
  // The lock file was NOT.
  testAssert(lockInodeAfter == lockInodeBefore);
  cout << "Compaction replaced the container inode " << dataInodeBefore << " -> " << dataInodeAfter
       << " while the lock inode stayed " << lockInodeBefore << endl;
}

// CRITERION 1. Two processes dumping the same context at once, into BOTH of its files, over
// DISJOINT key sets. Every figure asserted below is a count, because a count is the only
// thing that separates "both dumps landed" from "one dump landed and the other was discarded
// with everything after it".
void testTwoProcessesDumpingOneContextBothLand() {
  ScopedTempDir dir(TMP_DIR_PREFIX);
  const string directory = dir.path();
  const string signal = directory + "/go";
  const int perDump = 400;

  vector<pid_t> children;
  for(int child = 0; child < 2; child++) {
    const string intervalPath = directory + "/interval" + Global::intToString(child);
    const int firstSerial = child * perDump;
    children.push_back(forkChild([directory, signal, intervalPath, firstSerial, perDump]() {
      const NNEvalContainer container =
        NNEvalContainer::forContextAndModel(directory, CONTEXT, MODEL, MODEL_VERSION);
      const NNCacheCountLog log = NNCacheCountLog::forContext(directory, CONTEXT);
      const vector<shared_ptr<const NNOutput>> entries = makeBlock(firstSerial, perDump);
      const NNCacheHitCountDelta delta = makeCountDelta(firstSerial, perDump);
      awaitStartSignal(signal);
      const int64_t began = nowMicros();
      // A real dump writes both files, so the race is over both.
      (void)log.appendDump(delta);
      (void)container.appendBlock(entries);
      writeInterval(intervalPath, began, nowMicros());
    }));
  }
  raiseStartSignal(signal);
  for(size_t i = 0; i < children.size(); i++)
    waitForChild(children[i]);

  // THE CONTAINER. Two whole blocks, every entry of both, and an intact tail. blocksApplied
  // is the assertion that carries the weight: a torn-tail check alone would pass on a file
  // that had discarded one of the two dumps.
  const NNEvalContainer container =
    NNEvalContainer::forContextAndModel(directory, CONTEXT, MODEL, MODEL_VERSION);
  const NNEvalContainerContents contents = container.load();
  testAssert(contents.tail() == NNEvalContainerTail::Intact);
  testAssert(contents.discardedTailBytes() == 0);
  testAssert(contents.blocksApplied() == 2);
  testAssert(contents.entriesApplied() == (int64_t)(2 * perDump));
  testAssert((int64_t)contents.entries().size() == (int64_t)(2 * perDump));

  // THE COUNT LOG, the same claim over the other file of the same context.
  const NNCacheCountLog log = NNCacheCountLog::forContext(directory, CONTEXT);
  const NNCacheCountLogContents counts = log.load();
  testAssert(counts.tail() == NNCacheCountLogTail::Intact);
  testAssert(counts.discardedTailBytes() == 0);
  testAssert(counts.blocksApplied() == 2);
  testAssert((int64_t)counts.rows().size() == (int64_t)(2 * perDump));

  // Every key of BOTH disjoint ranges is present, checked by identity rather than by total:
  // two dumps of the same 400 keys would also total 800 physical entries.
  vector<bool> seen((size_t)(2 * perDump), false);
  for(size_t i = 0; i < contents.entries().size(); i++) {
    for(int serial = 0; serial < 2 * perDump; serial++) {
      if(contents.entries()[i]->nnHash == nthKey(serial)) {
        testAssert(!seen[(size_t)serial]);
        seen[(size_t)serial] = true;
      }
    }
  }
  for(size_t i = 0; i < seen.size(); i++)
    testAssert(seen[i]);

  // Whether the two attempts actually overlapped in time. This is EVIDENCE THAT THE TEST
  // EXERCISED THE LOCK, not a property of the lock, so it is reported rather than asserted:
  // making it an assertion would make the suite fail on a machine that happened to schedule
  // the two children sequentially, which is a fact about the scheduler and not a defect.
  int64_t startA = 0, endA = 0, startB = 0, endB = 0;
  readInterval(directory + "/interval0", startA, endA);
  readInterval(directory + "/interval1", startB, endB);
  const int64_t overlap = std::min(endA, endB) - std::max(startA, startB);
  cout << "Two processes dumped " << perDump << " entries each into one context: "
       << contents.blocksApplied() << " blocks, " << contents.entriesApplied()
       << " entries, tail Intact; count log " << counts.blocksApplied() << " blocks, "
       << counts.rows().size() << " rows" << endl;
  cout << "  attempt overlap: " << (overlap / 1000) << " ms"
       << (overlap > 0 ? " (the two dumps genuinely contended)" : " (the dumps did not overlap this run)")
       << endl;
}

// CRITERION 2. A reader running concurrently with a writer sees whole states only. Each
// observation must be an intact tail AND a whole number of blocks: a reader that caught a
// half-written block would report a torn tail, and one that caught a block header without its
// entries would report a block count without the entries to match.
//
// THE SAMPLING LOOP READS THROUGH loadIndex, AND THAT IS THE REPRESENTATIVE CALL RATHER THAN
// THE CONVENIENT ONE. loadIndex is what the level-0 attach path actually invokes -- an attach
// reads a container's key set and its locations, and decodes only the payloads its selection
// bound kept -- and it applies the same block walk, the same checksum verification and the
// same torn-tail contract load() does. It is also cheap enough to sample the file many times
// while a dump is in flight; sampling through load(), which decodes every payload of a file
// that is growing under it, is slower than the writer and caught the minimum two states.
void testAReaderConcurrentWithADumpNeverSeesAPartialBlock() {
  ScopedTempDir dir(TMP_DIR_PREFIX);
  const string directory = dir.path();
  const string signal = directory + "/go";
  const int perBlock = 250;
  const int blocks = 12;

  const pid_t writer = forkChild([directory, signal, perBlock, blocks]() {
    const NNEvalContainer container =
      NNEvalContainer::forContextAndModel(directory, CONTEXT, MODEL, MODEL_VERSION);
    awaitStartSignal(signal);
    for(int i = 0; i < blocks; i++) {
      (void)container.appendBlock(makeBlock(i * perBlock, perBlock));
      // PACED, and the pacing is what makes this test a witness at all. Written back to back,
      // the writer holds or re-takes the exclusive lock continuously and finishes all twelve
      // blocks inside a couple of the reader's poll intervals: the reader then observes the
      // empty file and the finished file and NOTHING IN BETWEEN, which passes a
      // "saw more than one state" assertion while proving nothing about concurrency. This is
      // not an artificial slowdown to make a test pass -- it is what a real engine does, since
      // a dump is an occasional explicit act and not a tight loop, and it is what puts the
      // reader inside the dump sequence where the property under test lives.
      std::this_thread::sleep_for(chrono::milliseconds(25));
    }
  });

  const NNEvalContainer container =
    NNEvalContainer::forContextAndModel(directory, CONTEXT, MODEL, MODEL_VERSION);
  raiseStartSignal(signal);

  // Sample until the writer is done, asserting the invariant on EVERY observation.
  vector<int64_t> observed;
  int64_t lastBlocks = -1;
  int status = 0;
  for(;;) {
    const NNEvalContainerIndex index = container.loadIndex();
    testAssert(index.tail() == NNEvalContainerTail::Intact);
    testAssert(index.discardedTailBytes() == 0);
    // A whole number of whole blocks, every time.
    testAssert(index.entriesApplied() == index.blocksApplied() * (int64_t)perBlock);
    // Never goes backwards: an append only ever adds.
    testAssert(index.blocksApplied() >= lastBlocks);
    if(index.blocksApplied() != lastBlocks) {
      observed.push_back(index.blocksApplied());
      lastBlocks = index.blocksApplied();
    }
    const pid_t done = ::waitpid(writer, &status, WNOHANG);
    if(done == writer)
      break;
    testAssert(done == 0);
    // A GAP BETWEEN SAMPLES, and it is not padding. A reader that re-acquires the shared lock
    // the instant it releases it leaves no interval for an exclusive acquirer to win, and
    // flock has no queue and no fairness guarantee -- so a gap-less loop STARVES THE WRITER
    // to its deadline. That was witnessed here, not theorised: an earlier form of this test
    // held no gap and the appending child threw after waiting the full 20 s. A real attach
    // does substantial work between reads, so the gap is the realistic shape; the starvation
    // boundary is a genuine property of the lock and is recorded in the audit report rather
    // than being asserted here, because a test that pinned it down would be asserting a
    // limitation as though it were a promise.
    std::this_thread::sleep_for(chrono::milliseconds(5));
  }
  testAssert(WIFEXITED(status));
  testAssert(WEXITSTATUS(status) == 0);

  const NNEvalContainerContents finalContents = container.load();
  testAssert(finalContents.tail() == NNEvalContainerTail::Intact);
  testAssert(finalContents.blocksApplied() == (int64_t)blocks);
  testAssert(finalContents.entriesApplied() == (int64_t)(blocks * perBlock));

  // THE ASSERTION WITH TEETH. The reader must have observed at least one state STRICTLY
  // BETWEEN empty and finished -- that is the only thing that proves it read while the dump
  // sequence was in flight, and therefore the only thing that makes every "intact tail, whole
  // number of blocks" assertion above a statement about concurrency rather than about a file
  // nobody was touching. "At least two distinct observations" was the earlier form of this
  // check and it was WORTHLESS: the reader saw 0 blocks and then 12, two distinct values that
  // between them span the entire dump without once landing inside it.
  bool sawAnIntermediateState = false;
  for(size_t i = 0; i < observed.size(); i++) {
    if(observed[i] > 0 && observed[i] < (int64_t)blocks)
      sawAnIntermediateState = true;
  }
  testAssert(sawAnIntermediateState);
  cout << "  observed block counts:";
  for(size_t i = 0; i < observed.size(); i++)
    cout << " " << observed[i];
  cout << endl;
  cout << "A concurrent reader took " << observed.size()
       << " distinct observations while " << blocks
       << " blocks were being appended, every one of them a whole number of whole blocks with an intact tail"
       << endl;
}

#endif  // _WIN32

}  // namespace

void Tests::runNNCacheFileLockTests() {
  cout << "Running nn cache file lock tests" << endl;

  testTheLockLivesOnItsOwnFileBesideTheData();
  testABadContextNameIsRefusedByBothFormsAlike();
  testSharedLocksAdmitEachOtherAndExcludeAWriter();
  testTheExclusiveLockExcludesReadersAndWritersAlike();
  testTwoContextsInOneDirectoryDoNotContend();
  testANegativeWaitIsRefused();

#ifndef _WIN32
  testAWaitPastTheDeadlineThrowsNamingTheContextAndThePath();
  testCompactionReplacesTheDataInodeAndNeverTheLockInode();
  testTwoProcessesDumpingOneContextBothLand();
  testAReaderConcurrentWithADumpNeverSeesAPartialBlock();
#else
  // Stated, not skipped silently: these four tests need a second process, fork() is how they
  // get one, and Windows has no equivalent. Their properties are UNWITNESSED on this platform
  // rather than assumed to hold (ADR-0002).
  cout << "  fork() is unavailable on this platform: the four multi-process tests did not run" << endl;
#endif
}
