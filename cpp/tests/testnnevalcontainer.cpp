#include "../tests/tests.h"

#include <cstdio>
#include <iostream>
#include <memory>
#include <vector>

#include <ghc/filesystem.hpp>

#include "../core/fileutils.h"
#include "../core/rand.h"
#include "../neuralnet/nncachefileformat.h"
#include "../neuralnet/nnevalcontainer.h"

using namespace std;
using namespace TestCommon;

namespace gfs = ghc::filesystem;

// Correctness tests for the append-only per-(context, model) evaluation container.
//
// Everything asserted here is a LOGIC INVARIANT and is asserted EXACTLY, with no tolerance
// (ADR-0009, Calibration). That includes the floats: a stored evaluation is not a
// measurement to be reproduced within an epsilon, it is a value that was written and must
// come back as itself, so every float comparison below is ==. A tolerance here would hide
// exactly the defect this format could plausibly have -- an endianness or a packing slip
// that moves the low bits.
//
// The load-bearing tests are the torn-tail, truncation and corrupt-block ones. A test that
// only round-trips a clean file does not test the thing this framing exists for.

namespace {

//-------------------------------------------------------------------------------------
// Fixtures
//-------------------------------------------------------------------------------------

const char* MODEL = "kata1-b18c384nbt-s9732312320-d4245566942";
const int MODEL_VERSION = 14;

// The value written into every policy slot beyond the board. It is never written to a file
// -- the point of the fixture is that the loader must NOT bring it back -- so it is a
// deliberately unmistakable number rather than a plausible one.
const float BEYOND_BOARD_SENTINEL = -12345.5f;

// A distinct key per serial, spread over both halves.
Hash128 nthKey(int serial) {
  return Hash128(
    ((uint64_t)(serial + 1)) * 0x9E3779B97F4A7C15ULL,
    ((uint64_t)(serial + 1)) * 0xD6E8FEB86659FD93ULL + 0x1234567ULL
  );
}

// One evaluation, filled deterministically from (serial, generation) so that two evaluations
// of the SAME key from different dumps are distinguishable -- which is what makes the merge
// rules observable rather than inferred.
shared_ptr<NNOutput> makeOutput(int serial, int generation, int nnXLen, int nnYLen, bool withOwnerMap) {
  shared_ptr<NNOutput> out = make_shared<NNOutput>();
  out->nnHash = nthKey(serial);
  const float g = (float)generation;
  out->whiteWinProb = 0.125f * g;
  out->whiteLossProb = 0.25f + g;
  out->whiteNoResultProb = 0.0f;
  out->whiteScoreMean = -3.5f * g;
  out->whiteScoreMeanSq = 17.25f + g;
  out->whiteLead = 1.75f * g;
  out->varTimeLeft = 42.5f;
  out->shorttermWinlossError = 0.03125f * g;
  out->shorttermScoreError = 0.0625f;
  out->policyOptimismUsed = 0.5f;
  out->nnXLen = nnXLen;
  out->nnYLen = nnYLen;

  const int area = nnXLen * nnYLen;
  for(int i = 0; i < NNPos::MAX_NN_POLICY_SIZE; i++)
    out->policyProbs[i] = BEYOND_BOARD_SENTINEL;
  for(int i = 0; i <= area; i++) {
    // Some slots negative, as illegal moves are in memory, so the sign survives the round
    // trip under observation rather than by luck.
    out->policyProbs[i] = (i % 7 == 0) ? -1.0f : ((float)i * 0.5f + g);
  }

  if(withOwnerMap) {
    out->whiteOwnerMap = new float[area];
    for(int i = 0; i < area; i++)
      out->whiteOwnerMap[i] = ((float)i * 0.015625f) - 1.0f + g;
  }
  return out;
}

shared_ptr<const NNOutput> asStored(const shared_ptr<NNOutput>& out) {
  return shared_ptr<const NNOutput>(out);
}

// The one entry for `serial`. Asserts there is exactly one -- so "one entry per key" is
// checked on every read rather than once -- and asserts it is present, so a caller reads the
// evaluation off a reference and never dereferences a maybe-null pointer.
const NNOutput& entryFor(const NNEvalContainerContents& contents, int serial) {
  const Hash128 key = nthKey(serial);
  const NNOutput* found = NULL;
  for(size_t i = 0; i < contents.entries().size(); i++) {
    if(contents.entries()[i]->nnHash == key) {
      testAssert(found == NULL);
      found = contents.entries()[i].get();
    }
  }
  testAssert(found != NULL);
  return *found;
}

// Whether the container holds this key at all. Separate from entryFor because the two are
// different questions and an absence assertion should not go through an accessor that
// asserts presence.
bool hasEntryFor(const NNEvalContainerContents& contents, int serial) {
  const Hash128 key = nthKey(serial);
  for(size_t i = 0; i < contents.entries().size(); i++) {
    if(contents.entries()[i]->nnHash == key)
      return true;
  }
  return false;
}

// Every field the format claims to carry, compared exactly. The beyond-the-board policy
// slots are checked too, and they are checked to be ZERO rather than to match the source:
// they are a property of this build's array, not of the evaluation, so the loader bringing
// the writer's sentinel back would be the format carrying a fact it has no business
// carrying.
void assertSameEvaluation(const NNOutput& expected, const NNOutput& actual) {
  testAssert(actual.nnHash == expected.nnHash);
  testAssert(actual.nnXLen == expected.nnXLen);
  testAssert(actual.nnYLen == expected.nnYLen);
  testAssert(actual.whiteWinProb == expected.whiteWinProb);
  testAssert(actual.whiteLossProb == expected.whiteLossProb);
  testAssert(actual.whiteNoResultProb == expected.whiteNoResultProb);
  testAssert(actual.whiteScoreMean == expected.whiteScoreMean);
  testAssert(actual.whiteScoreMeanSq == expected.whiteScoreMeanSq);
  testAssert(actual.whiteLead == expected.whiteLead);
  testAssert(actual.varTimeLeft == expected.varTimeLeft);
  testAssert(actual.shorttermWinlossError == expected.shorttermWinlossError);
  testAssert(actual.shorttermScoreError == expected.shorttermScoreError);
  testAssert(actual.policyOptimismUsed == expected.policyOptimismUsed);
  testAssert(actual.noisedPolicyProbs == NULL);

  const int area = expected.nnXLen * expected.nnYLen;
  for(int i = 0; i <= area; i++)
    testAssert(actual.policyProbs[i] == expected.policyProbs[i]);
  for(int i = area + 1; i < NNPos::MAX_NN_POLICY_SIZE; i++)
    testAssert(actual.policyProbs[i] == 0.0f);

  testAssert((actual.whiteOwnerMap != NULL) == (expected.whiteOwnerMap != NULL));
  if(expected.whiteOwnerMap != NULL) {
    for(int i = 0; i < area; i++)
      testAssert(actual.whiteOwnerMap[i] == expected.whiteOwnerMap[i]);
  }
}

// A directory that removes itself, so a failed assertion in the middle of a test does not
// leave litter behind for the next run to trip over.
class ScopedTempDir {
 public:
  ScopedTempDir() {
    Rand rand;
    path_ = "tmpnnevalcontainer_" + Global::uint64ToHexString(rand.nextUInt64());
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

// True if load() refused this file. A named helper rather than a bare call in a try, so the
// [[nodiscard]] on load() stays honest instead of being waived at each site.
bool loadIsRefused(const NNEvalContainer& container) {
  try {
    const NNEvalContainerContents contents = container.load();
    (void)contents;
    return false;
  }
  catch(const StringError&) {
    return true;
  }
}

// The refusal's message, or the empty string if there was none. Used where the DIAGNOSIS is
// the thing under test and not merely that something threw.
string loadRefusalMessage(const NNEvalContainer& container) {
  try {
    const NNEvalContainerContents contents = container.load();
    (void)contents;
    return "";
  }
  catch(const StringError& e) {
    return e.what();
  }
}

int64_t sizeOf(const string& path) {
  return (int64_t)gfs::file_size(gfs::u8path(path));
}

void truncateTo(const string& path, int64_t bytes) {
  gfs::resize_file(gfs::u8path(path), (uintmax_t)bytes);
}

// Overwrites `bytes` at `offset`. Used to corrupt a block in place, which is the failure a
// length-only framing cannot see at all and a checksum can.
void overwriteBytesAt(const string& path, int64_t offset, const vector<uint8_t>& bytes) {
  FILE* f = fopen(path.c_str(), "r+b");
  testAssert(f != NULL);
  testAssert(fseek(f, (long)offset, SEEK_SET) == 0);
  testAssert(fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size());
  fclose(f);
}

vector<uint8_t> readBytesAt(const string& path, int64_t offset, size_t count) {
  vector<uint8_t> bytes(count);
  FILE* f = fopen(path.c_str(), "rb");
  testAssert(f != NULL);
  testAssert(fseek(f, (long)offset, SEEK_SET) == 0);
  testAssert(fread(bytes.data(), 1, count, f) == count);
  fclose(f);
  return bytes;
}

// Little-endian, unpacked byte by byte here as well, so the test reads the file the way the
// format says a file is read rather than by reinterpreting this machine's struct layout.
uint32_t readU32(const uint8_t* p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
uint64_t readU64(const uint8_t* p) {
  uint64_t v = 0;
  for(int i = 0; i < 8; i++)
    v |= ((uint64_t)p[i]) << (8 * i);
  return v;
}

void writeU64(uint8_t* p, uint64_t v) {
  for(int i = 0; i < 8; i++)
    p[i] = (uint8_t)(v >> (8 * i));
}

NNEvalContainer containerIn(const ScopedTempDir& dir, const char* context) {
  return NNEvalContainer::forContextAndModel(dir.path(), context, MODEL, MODEL_VERSION);
}

//-------------------------------------------------------------------------------------
// The tests
//-------------------------------------------------------------------------------------

// Evaluations written are evaluations read back, exactly, WITH and WITHOUT an ownership map
// in the same file. The mixed file is the point: carriage is optional per entry, so a
// container that only ever held one kind would not exercise the flag at all.
void testEvalContainerRoundTripsWithAndWithoutOwnershipMaps() {
  ScopedTempDir dir;
  const NNEvalContainer container = containerIn(dir, "roundtrip");

  // 19x19 with a map, 19x19 without, 9x9 with, 9x9 without: the two axes crossed, so a
  // defect that only bites one combination cannot hide behind the other three.
  const shared_ptr<NNOutput> a = makeOutput(1, 1, 19, 19, true);
  const shared_ptr<NNOutput> b = makeOutput(2, 1, 19, 19, false);
  const shared_ptr<NNOutput> c = makeOutput(3, 1, 9, 9, true);
  const shared_ptr<NNOutput> d = makeOutput(4, 1, 9, 9, false);

  const vector<shared_ptr<const NNOutput>> entries = {asStored(a), asStored(b), asStored(c), asStored(d)};
  const NNEvalContainerAppendResult appended = container.appendBlock(entries);
  testAssert(appended.tornTailBytesDiscarded == 0);
  testAssert(appended.tailRepair == NNCacheFileTailRepair::NotNeeded);

  // The byte arithmetic is quoted from the implementation rather than from a second copy of
  // the numbers (ADR-0012 P1).
  const int64_t expectedBytes =
    NNEvalContainer::fileHeaderBytesFor(MODEL) +
    (int64_t)NNEvalContainer::blockHeaderBytes() +
    NNEvalContainer::bytesForEntry(19, 19, true) +
    NNEvalContainer::bytesForEntry(19, 19, false) +
    NNEvalContainer::bytesForEntry(9, 9, true) +
    NNEvalContainer::bytesForEntry(9, 9, false);
  testAssert(appended.bytesAppended == expectedBytes);
  testAssert(sizeOf(container.path()) == expectedBytes);

  const NNEvalContainerContents contents = container.load();
  testAssert(contents.tail() == NNEvalContainerTail::Intact);
  testAssert(contents.discardedTailBytes() == 0);
  testAssert(contents.blocksApplied() == 1);
  testAssert(contents.entriesApplied() == 4);
  testAssert(contents.entries().size() == 4);

  assertSameEvaluation(*a, entryFor(contents, 1));
  assertSameEvaluation(*b, entryFor(contents, 2));
  assertSameEvaluation(*c, entryFor(contents, 3));
  assertSameEvaluation(*d, entryFor(contents, 4));

  // Ownership carriage, asserted as its own membership fact rather than folded into the
  // comparison above: an entry that HAS a map has one, and one that does not, does not.
  testAssert(entryFor(contents, 1).whiteOwnerMap != NULL);
  testAssert(entryFor(contents, 2).whiteOwnerMap == NULL);
  testAssert(entryFor(contents, 3).whiteOwnerMap != NULL);
  testAssert(entryFor(contents, 4).whiteOwnerMap == NULL);

  // And the 9x9 entries' beyond-the-board policy slots came back ZERO, not as the writer's
  // sentinel: the file carries a board-sized policy and nothing about this build's array
  // width. Asserted positively at the exact slot rather than inferred from the byte count.
  testAssert(entryFor(contents, 3).policyProbs[9 * 9 + 1] == 0.0f);
  testAssert(entryFor(contents, 4).policyProbs[NNPos::MAX_NN_POLICY_SIZE - 1] == 0.0f);
}

// A second dump supersedes the first for a key it names again -- EXCEPT that an evaluation
// without an ownership map never displaces one with. That exception is the store-side face
// of the live supersession rule and is the whole reason the merge is not plain last-wins.
void testEvalContainerMergeIsLastWinsExceptAnOwnershipMapIsNeverLost() {
  ScopedTempDir dir;
  const NNEvalContainer container = containerIn(dir, "merge");

  const shared_ptr<NNOutput> a1 = makeOutput(1, 1, 19, 19, true);   // key 1, WITH a map
  const shared_ptr<NNOutput> b1 = makeOutput(2, 1, 19, 19, false);  // key 2, without
  container.appendBlock({asStored(a1), asStored(b1)});

  const shared_ptr<NNOutput> a2 = makeOutput(1, 2, 19, 19, false);  // key 1 again, WITHOUT
  const shared_ptr<NNOutput> b2 = makeOutput(2, 2, 19, 19, true);   // key 2 again, WITH
  const shared_ptr<NNOutput> c2 = makeOutput(3, 2, 19, 19, false);  // a key only block 2 names
  container.appendBlock({asStored(a2), asStored(b2), asStored(c2)});

  {
    const NNEvalContainerContents contents = container.load();
    testAssert(contents.blocksApplied() == 2);
    testAssert(contents.entriesApplied() == 5);
    testAssert(contents.entries().size() == 3);

    // THE HEADLINE CLAIM. Key 1's later, ownermap-less evaluation did NOT win: the entry
    // still carries the map, and it is still generation 1 in every field -- so this is the
    // earlier evaluation standing, not the later one with a map somehow attached.
    assertSameEvaluation(*a1, entryFor(contents, 1));
    testAssert(entryFor(contents, 1).whiteOwnerMap != NULL);

    // And the rule is not "the first write wins": key 2's later evaluation, which ADDS a
    // map, did supersede. Without this leg the assertion above would also pass under an
    // implementation that simply ignored every block after the first.
    assertSameEvaluation(*b2, entryFor(contents, 2));
    testAssert(entryFor(contents, 2).whiteOwnerMap != NULL);

    assertSameEvaluation(*c2, entryFor(contents, 3));
  }

  // A later evaluation that also carries a map supersedes normally: the exception is about
  // never LOSING a map, not about freezing an entry once it has one.
  const shared_ptr<NNOutput> a3 = makeOutput(1, 3, 19, 19, true);
  container.appendBlock({asStored(a3)});
  {
    const NNEvalContainerContents contents = container.load();
    testAssert(contents.entriesApplied() == 6);
    testAssert(contents.entries().size() == 3);
    assertSameEvaluation(*a3, entryFor(contents, 1));
  }
}

// The same never-lose-an-ownership-map rule, for a map that was NOT established in the first
// block.
//
// Why this is a leg of its own rather than another key in the test above. That test always
// establishes the protected map in block 1, so every assertion it makes is consistent with an
// implementation that only protects an entry the file OPENED with -- one that treats "carries
// a map" as a property of the original insertion rather than of the entry currently standing.
// Nothing in applyEntry is written that way, but the test could not tell, and a rule that
// holds only for the first block is exactly the shape a later refactor could introduce
// unnoticed. Here the map arrives in block 2, having itself superseded a map-less entry, and
// must still stand against block 3.
void testEvalContainerAnOwnershipMapEarnedInALaterBlockIsAlsoNeverLost() {
  ScopedTempDir dir;
  const NNEvalContainer container = containerIn(dir, "latemap");

  // Block 1: key 1 WITHOUT a map. Key 2 is a control that never has a map at all.
  const shared_ptr<NNOutput> a1 = makeOutput(1, 1, 19, 19, false);
  const shared_ptr<NNOutput> b1 = makeOutput(2, 1, 19, 19, false);
  container.appendBlock({asStored(a1), asStored(b1)});

  // Block 2: key 1 WITH a map, superseding normally. The map is now established, mid-file.
  const shared_ptr<NNOutput> a2 = makeOutput(1, 2, 19, 19, true);
  container.appendBlock({asStored(a2)});
  {
    const NNEvalContainerContents contents = container.load();
    assertSameEvaluation(*a2, entryFor(contents, 1));
    testAssert(entryFor(contents, 1).whiteOwnerMap != NULL);
  }

  // Block 3: key 1 WITHOUT a map again. It must not win.
  const shared_ptr<NNOutput> a3 = makeOutput(1, 3, 19, 19, false);
  // Key 2 rides along in the same block, map-less both times, and MUST supersede. Without it
  // this test would also pass against an implementation that ignored block 3 wholesale, and
  // key 1's survival would be an observation of nothing (ADR-0021 Rule 4: the red, and the
  // green, must be for the right reason).
  const shared_ptr<NNOutput> b3 = makeOutput(2, 3, 19, 19, false);
  container.appendBlock({asStored(a3), asStored(b3)});

  const NNEvalContainerContents contents = container.load();
  testAssert(contents.blocksApplied() == 3);
  testAssert(contents.entriesApplied() == 5);
  testAssert(contents.entries().size() == 2);

  // THE HEADLINE CLAIM: key 1 is still block 2's evaluation, map intact.
  assertSameEvaluation(*a2, entryFor(contents, 1));
  testAssert(entryFor(contents, 1).whiteOwnerMap != NULL);
  // AND block 3 was applied: its map-less entry for key 2 won, because key 2 had no map to
  // lose.
  assertSameEvaluation(*b3, entryFor(contents, 2));
  testAssert(entryFor(contents, 2).whiteOwnerMap == NULL);
  cout << "nnevals container: an ownership map earned in block 2 stood against block 3's "
          "map-less entry for the same key, while block 3's other entry superseded normally"
       << endl;
}

// THE LOAD-BEARING TEST. A crash mid-dump leaves a partial block. Exactly the dumps that
// completed must survive, and the partial one must not -- not even the entries of it that
// happen to sit whole inside the discarded region.
void testEvalContainerTornTailIsDiscardedAndThePrefixSurvives() {
  ScopedTempDir dir;
  const NNEvalContainer container = containerIn(dir, "torn");

  const shared_ptr<NNOutput> a1 = makeOutput(1, 1, 19, 19, true);
  const shared_ptr<NNOutput> b1 = makeOutput(2, 1, 19, 19, false);
  container.appendBlock({asStored(a1), asStored(b1)});
  const shared_ptr<NNOutput> c2 = makeOutput(3, 2, 9, 9, true);
  container.appendBlock({asStored(c2)});
  const int64_t sizeAfterTwoDumps = sizeOf(container.path());

  // The third dump both supersedes a key the first named AND names two keys no earlier dump
  // did, so its survival or absence is directly observable in two independent ways rather
  // than hidden inside a merge (ADR-0021 Rule 1: observe the property at the site of the
  // claim).
  const shared_ptr<NNOutput> a3 = makeOutput(1, 3, 19, 19, true);
  const shared_ptr<NNOutput> d3 = makeOutput(4, 3, 19, 19, true);
  const shared_ptr<NNOutput> e3 = makeOutput(5, 3, 9, 9, false);
  container.appendBlock({asStored(a3), asStored(d3), asStored(e3)});

  // Cut the file part-way through the third block's THIRD payload: not on a payload
  // boundary and not on a block boundary, which is what a process dying part-way through a
  // write(2) leaves behind.
  //
  // The cut falls where it does deliberately. The third block's WHOLE key index and its
  // first TWO payloads -- keys 1 and 4 -- sit COMPLETE inside the region that must be
  // discarded. So an implementation that trusted the gathered index, or that took whatever
  // whole payloads happened to fit, would apply them, and the assertions below would see it.
  // Cutting inside the index instead would have made key 4's absence an accident of where
  // the byte landed rather than an observation of the property; that case is its own leg in
  // the truncation test.
  const int64_t cutAt = sizeAfterTwoDumps + (int64_t)NNEvalContainer::blockHeaderBytes() +
                        3 * (int64_t)NNEvalContainer::entryHeaderBytes() +
                        2 * NNEvalContainer::payloadBytesFor(19, 19, true) + 9;
  testAssert(cutAt < sizeOf(container.path()));
  truncateTo(container.path(), cutAt);
  testAssert(sizeOf(container.path()) == cutAt);

  const NNEvalContainerContents contents = container.load();

  // THE HEADLINE CLAIM FIRST, so a defect that applies part of a torn block goes red on the
  // content being wrong rather than on the tail's byte arithmetic.
  //
  // Key 1 is still the FIRST dump's evaluation. The torn block's replacement for it sat
  // whole in the discarded region and was not applied in part.
  assertSameEvaluation(*a1, entryFor(contents, 1));
  assertSameEvaluation(*b1, entryFor(contents, 2));
  assertSameEvaluation(*c2, entryFor(contents, 3));

  // AND THE PARTIAL DUMP DID NOT SURVIVE, observed positively: keys 4 and 5 were named ONLY
  // by the torn block, so their absence is a direct membership fact about what the load
  // returned. Key 4's entry is whole in the file and is discarded anyway, because the BLOCK
  // it belongs to is not.
  testAssert(!hasEntryFor(contents, 4));
  testAssert(!hasEntryFor(contents, 5));
  testAssert(contents.entries().size() == 3);

  // Then the accounting: the disposition is Truncated and names exactly how many bytes it
  // will not use.
  testAssert(contents.tail() == NNEvalContainerTail::Truncated);
  testAssert(contents.discardedTailBytes() == cutAt - sizeAfterTwoDumps);
  testAssert(contents.blocksApplied() == 2);
  testAssert(contents.entriesApplied() == 3);
}

// Truncation at every structurally distinct point in the file. In each case the container
// must REFUSE the bytes it cannot account for -- typed as Truncated, with the exact byte
// count, and with nothing from the incomplete region applied -- rather than short-reading
// its way to a plausible partial answer.
void testEvalContainerTruncationRefusesRatherThanShortReading() {
  ScopedTempDir dir;
  const int64_t headerBytes = NNEvalContainer::fileHeaderBytesFor(MODEL);
  const int64_t entryBytes = NNEvalContainer::bytesForEntry(19, 19, true);

  // Each leg gets its own container, because a truncation is destructive and legs that
  // shared one file would be testing each other's leftovers.
  struct Leg { const char* context; int64_t cutTo; int64_t expectDiscarded; int64_t expectBlocks; };

  // Two dumps: one entry each. The file is header + block + entry + block + entry.
  const int64_t fullSize = headerBytes + 2 * ((int64_t)NNEvalContainer::blockHeaderBytes() + entryBytes);
  const vector<Leg> legs = {
    // Inside the fixed part of the file header: a crash during the very first dump.
    {"trunchdr", 20, 20, 0},
    // Inside the model name that follows it: the header is self-describing and incomplete.
    {"truncname", headerBytes - 1, headerBytes - 1, 0},
    // Exactly at the end of the header: a file that holds no dump at all.
    {"truncnone", headerBytes, 0, 0},
    // Inside the first block's header, before any length in it could be believed.
    {"truncblk", headerBytes + 7, 7, 0},
    // One byte short of the first entry's end.
    {"truncentry", headerBytes + (int64_t)NNEvalContainer::blockHeaderBytes() + entryBytes - 1,
     (int64_t)NNEvalContainer::blockHeaderBytes() + entryBytes - 1, 0},
    // Inside the SECOND block's gathered key index: the index claims an entry whose bytes
    // are not all there. This is a distinct case from a cut inside the payload region -- the
    // index is a length that a crash can leave half-written and that a reader is about to
    // size an array on.
    {"truncindex",
     headerBytes + (int64_t)NNEvalContainer::blockHeaderBytes() + entryBytes +
       (int64_t)NNEvalContainer::blockHeaderBytes() + (int64_t)NNEvalContainer::entryHeaderBytes() / 2,
     (int64_t)NNEvalContainer::blockHeaderBytes() + (int64_t)NNEvalContainer::entryHeaderBytes() / 2, 1},
    // One byte short of the whole file: the first dump survives, the second does not.
    {"trunclast", fullSize - 1, (int64_t)NNEvalContainer::blockHeaderBytes() + entryBytes - 1, 1},
  };

  for(size_t i = 0; i < legs.size(); i++) {
    const NNEvalContainer container = containerIn(dir, legs[i].context);
    const shared_ptr<NNOutput> first = makeOutput(1, 1, 19, 19, true);
    const shared_ptr<NNOutput> second = makeOutput(2, 1, 19, 19, true);
    container.appendBlock({asStored(first)});
    container.appendBlock({asStored(second)});
    testAssert(sizeOf(container.path()) == fullSize);

    truncateTo(container.path(), legs[i].cutTo);
    const NNEvalContainerContents contents = container.load();

    testAssert(contents.blocksApplied() == legs[i].expectBlocks);
    testAssert(contents.discardedTailBytes() == legs[i].expectDiscarded);
    testAssert(contents.tail() ==
               (legs[i].expectDiscarded > 0 ? NNEvalContainerTail::Truncated : NNEvalContainerTail::Intact));
    // The membership claim, positively: an entry from a block that did not survive is not in
    // the result. Key 2 is named only by the second dump.
    testAssert(hasEntryFor(contents, 1) == (legs[i].expectBlocks >= 1));
    testAssert(!hasEntryFor(contents, 2));
    if(legs[i].expectBlocks >= 1)
      assertSameEvaluation(*first, entryFor(contents, 1));
  }
}

// The block's key index is GATHERED: the entry headers sit contiguously right after the
// block header, so a caller can read a block's whole key set with its ownermap flags in one
// sequential read instead of walking every payload to find the next key.
//
// This is observed at the bytes, because it is a claim about the byte layout and nothing
// else would witness it: an implementation that interleaved headers with payloads would pass
// every round-trip test in this file and fail only here.
void testEvalContainerTheBlocksKeyIndexIsGatheredAfterItsHeader() {
  ScopedTempDir dir;
  const NNEvalContainer container = containerIn(dir, "keyindex");

  // Deliberately mixed shapes and ownermap carriage, so a reader of the index alone still
  // sees the right flags and sizes against entries of quite different payload lengths.
  const shared_ptr<NNOutput> a = makeOutput(1, 1, 19, 19, true);
  const shared_ptr<NNOutput> b = makeOutput(2, 1, 9, 9, false);
  const shared_ptr<NNOutput> c = makeOutput(3, 1, 19, 19, false);
  container.appendBlock({asStored(a), asStored(b), asStored(c)});

  const int64_t indexStart =
    NNEvalContainer::fileHeaderBytesFor(MODEL) + (int64_t)NNEvalContainer::blockHeaderBytes();
  const vector<uint8_t> index =
    readBytesAt(container.path(), indexStart, 3 * NNEvalContainer::entryHeaderBytes());

  const shared_ptr<NNOutput> expected[3] = {a, b, c};
  int64_t runningOffset = 0;
  for(int i = 0; i < 3; i++) {
    const uint8_t* h = index.data() + (size_t)i * NNEvalContainer::entryHeaderBytes();
    testAssert(readU64(h + 0) == expected[i]->nnHash.hash0);
    testAssert(readU64(h + 8) == expected[i]->nnHash.hash1);
    // The ownermap flag is readable from the index alone -- which is the point, since an
    // admission or merge decision wants it without touching a payload.
    testAssert((h[16] & 0x1) == (expected[i]->whiteOwnerMap != NULL ? 1 : 0));
    testAssert(h[17] == 0);
    testAssert((int)h[18] == expected[i]->nnXLen);
    testAssert((int)h[19] == expected[i]->nnYLen);
    const int64_t payloadBytes =
      NNEvalContainer::payloadBytesFor(expected[i]->nnXLen, expected[i]->nnYLen,
                                       expected[i]->whiteOwnerMap != NULL);
    testAssert((int64_t)readU32(h + 20) == payloadBytes);
    // And the payload offset is the running sum of the preceding payload sizes, so reaching
    // one entry's payload is O(1) from its header.
    testAssert((int64_t)readU64(h + 24) == runningOffset);
    runningOffset += payloadBytes;
  }

  // And the file is exactly the header, the block header, the index, and the payloads -- no
  // interleaving, nothing else.
  testAssert(sizeOf(container.path()) == indexStart + 3 * (int64_t)NNEvalContainer::entryHeaderBytes() + runningOffset);
}

// A file from a LATER VERSION of this format -- one that sets a flag bit or a reserved header
// byte v1 does not define -- is refused by name, not read as if the bits it does not
// understand were absent.
//
// HOW THIS FILE IS FORGED, because the method is the whole reason this test exists. The
// container is written by the REAL writer; then one bit is flipped on disk; then the two
// checksums that bit invalidated are recomputed BY CALLING THE PRODUCTION FUNCTIONS --
// NNCacheFileChecksum::of and NNCacheFileName::hashOf, the same functions the reader itself
// calls to verify, public in nncachefileformat.h precisely so a caller can reuse them. No
// checksum algorithm and no framing rule is restated here; the test knows only the strides
// the format publishes through NNEvalContainer's own accessors, exactly as the key-index test
// above does.
//
// This matters because the alternative -- flipping the bit and leaving the checksums stale --
// would witness nothing: the block would be discarded as corrupt, the refusal under test
// would never be reached, and the test would pass just as well with that refusal deleted. A
// forged file whose checksums are VALID is the only input that can reach it.
void testEvalContainerAFileFromALaterVersionIsRefusedNotSilentlyRead() {
  ScopedTempDir dir;

  // The one seed both checksums in this file are computed under. Read from the production
  // function, for the context this container is bound to.
  const uint64_t contextHash = NNCacheFileName::hashOf("laterversion");

  // Leg 1: an undefined FLAG BIT on an entry. The entry header sits inside the block, so the
  // block's own two checksums have to be repaired for the file to reach the flag check at
  // all.
  {
    const NNEvalContainer container = containerIn(dir, "laterversion");
    const shared_ptr<NNOutput> a = makeOutput(1, 1, 19, 19, true);
    container.appendBlock({asStored(a)});

    // It reads cleanly first, so what changes below is the one bit and nothing else.
    {
      const NNEvalContainerContents before = container.load();
      testAssert(before.tail() == NNEvalContainerTail::Intact);
      testAssert(before.entries().size() == 1);
    }

    const int64_t blockStart = NNEvalContainer::fileHeaderBytesFor(MODEL);
    const int64_t indexStart = blockStart + (int64_t)NNEvalContainer::blockHeaderBytes();
    // Everything the block's entry checksum covers: the gathered index, then the payloads.
    const int64_t regionBytes = NNEvalContainer::bytesForEntry(19, 19, true);

    // Set flags bit 1 -- reserved in v1, and NOT bit 0, so the entry's shape, its declared
    // payload size and its offset all stay exactly right. The only thing wrong with this file
    // is that it claims a feature this build does not have.
    vector<uint8_t> flagsByte = readBytesAt(container.path(), indexStart + 16, 1);
    testAssert((flagsByte[0] & 0x2) == 0);
    flagsByte[0] = (uint8_t)(flagsByte[0] | 0x2);
    overwriteBytesAt(container.path(), indexStart + 16, flagsByte);

    // Repair the entry checksum over the whole region, through the production function.
    const vector<uint8_t> region = readBytesAt(container.path(), indexStart, (size_t)regionBytes);
    vector<uint8_t> entryChecksum(8);
    writeU64(entryChecksum.data(), NNCacheFileChecksum::of(region.data(), region.size(), contextHash));
    overwriteBytesAt(container.path(), blockStart + 16, entryChecksum);

    // And the block header's checksum of itself, over its own first 24 bytes, same function,
    // same seed.
    const vector<uint8_t> blockHeader = readBytesAt(container.path(), blockStart, 24);
    vector<uint8_t> headerChecksum(8);
    writeU64(headerChecksum.data(), NNCacheFileChecksum::of(blockHeader.data(), blockHeader.size(), contextHash));
    overwriteBytesAt(container.path(), blockStart + 24, headerChecksum);

    // THE CLAIM: this is a well-formed, correctly-checksummed container that v1 must refuse.
    // Asserted on the DIAGNOSIS, because a refusal that merely said "corrupt" would mean the
    // forgery had failed and the checksums, not the flag, had caught it.
    const string message = loadRefusalMessage(container);
    testAssert(message.find("flag bits this build does not define") != string::npos);
    testAssert(message.find("later version") != string::npos);
    // Printed, because this refusal is the one the format's forward-compatibility story rests
    // on and the message is what an operator meeting a future file will actually see.
    cout << "nnevals container: a forged v1 file setting an undefined flag bit, with both "
            "block checksums valid, was refused: " << message << endl;
  }

  // Leg 2: a reserved FILE-HEADER byte. The file header carries no checksum of its own -- it
  // is verified field by field instead -- so this one needs no repair, which is worth stating
  // rather than leaving the reader to wonder why the two legs look so different.
  {
    const NNEvalContainer container = containerIn(dir, "laterheader");
    container.appendBlock({asStored(makeOutput(1, 1, 19, 19, false))});
    testAssert(!loadIsRefused(container));
    overwriteBytesAt(container.path(), 40, vector<uint8_t>{0, 0, 0, 1, 0, 0, 0, 0});
    const string message = loadRefusalMessage(container);
    testAssert(message.find("reserved file-header byte") != string::npos);
  }
}

// A block that is the RIGHT LENGTH but wrong in its bytes -- the shape a lost page leaves,
// which a length-only framing cannot see at all.
void testEvalContainerAWholeButCorruptBlockIsRejectedByItsChecksum() {
  ScopedTempDir dir;
  const NNEvalContainer container = containerIn(dir, "corrupt");

  const shared_ptr<NNOutput> a = makeOutput(1, 1, 19, 19, true);
  container.appendBlock({asStored(a)});
  const int64_t sizeAfterOneDump = sizeOf(container.path());
  const shared_ptr<NNOutput> b = makeOutput(2, 1, 19, 19, true);
  container.appendBlock({asStored(b)});

  // Zero eight bytes inside the SECOND block's single entry -- in its ownership map, well
  // past every framing field, so nothing about the file's lengths changes and only the
  // entry checksum can tell.
  const int64_t deepInsideThePayload =
    sizeAfterOneDump + (int64_t)NNEvalContainer::blockHeaderBytes() +
    (int64_t)NNEvalContainer::entryHeaderBytes() + 1000;
  overwriteBytesAt(container.path(), deepInsideThePayload, vector<uint8_t>(8, 0));

  const NNEvalContainerContents contents = container.load();
  testAssert(contents.tail() == NNEvalContainerTail::Truncated);
  testAssert(contents.discardedTailBytes() == sizeOf(container.path()) - sizeAfterOneDump);
  testAssert(contents.blocksApplied() == 1);
  assertSameEvaluation(*a, entryFor(contents, 1));
  // The corrupt block's entry is not served in a damaged form. This is the assertion that
  // matters: a checksum that only logged would still let the caller hold a wrong evaluation.
  testAssert(!hasEntryFor(contents, 2));
}

// A corrupt block HEADER must not produce a partial application, and its lengths must never
// reach an allocation unverified.
//
// HONESTLY LABELLED: unlike the count log, this format's block header carries no field that
// the entry checksum does not already cover indirectly, so the header's own checksum here is
// defence in depth rather than the only thing that can see the damage. What is asserted
// below is therefore the BEHAVIOUR -- an absurd length is refused and the intact prefix
// stands -- not a claim that one specific check is the one that caught it.
void testEvalContainerACorruptBlockHeaderDoesNotProduceAPartialApplication() {
  ScopedTempDir dir;
  const NNEvalContainer container = containerIn(dir, "badlen");

  const shared_ptr<NNOutput> a = makeOutput(1, 1, 19, 19, false);
  container.appendBlock({asStored(a)});
  const int64_t sizeAfterOneDump = sizeOf(container.path());
  const shared_ptr<NNOutput> b = makeOutput(2, 1, 19, 19, false);
  container.appendBlock({asStored(b)});

  // Overwrite the second block's entry count with 0xFFFFFFFF. Nothing may size an
  // allocation on it: four billion entry headers is 128 GB of framing that the file plainly
  // does not contain.
  overwriteBytesAt(container.path(), sizeAfterOneDump + 4, vector<uint8_t>(4, 0xFF));

  const NNEvalContainerContents contents = container.load();
  testAssert(contents.tail() == NNEvalContainerTail::Truncated);
  testAssert(contents.blocksApplied() == 1);
  assertSameEvaluation(*a, entryFor(contents, 1));
  testAssert(!hasEntryFor(contents, 2));

  // The same for an absurd payload total, which is the other length in the header.
  {
    const NNEvalContainer other = containerIn(dir, "badbytes");
    other.appendBlock({asStored(a)});
    const int64_t sizeAfterFirst = sizeOf(other.path());
    other.appendBlock({asStored(b)});
    overwriteBytesAt(other.path(), sizeAfterFirst + 8, vector<uint8_t>(8, 0xFF));

    const NNEvalContainerContents contents2 = other.load();
    testAssert(contents2.tail() == NNEvalContainerTail::Truncated);
    testAssert(contents2.blocksApplied() == 1);
    testAssert(!hasEntryFor(contents2, 2));
  }
}

// A torn tail must be repaired by the WRITER before it appends, or every later dump lands at
// an offset no loader reaches and is silently lost while every call reports success.
//
// AND THE REPAIR IS A TRUNCATION, which is asserted here as a property of the FILE rather
// than as a fact about which function was called. Three observations say it, and together
// they say it exactly: the surviving prefix is byte-for-byte what stood there before the
// tear (a rewrite would have re-encoded it into one collapsed block, changing those bytes);
// the file's new length is that prefix plus the appended block and not one byte more; and
// the append reports its tail repair as Truncated with a positive discard count. The reason
// assert this rather than leave it to the byte counts of a bench: at the deployment's 10-20
// GB per card, a repair that quietly went back to rewriting would cost a full card written
// to flash for every engine killed mid-dump, and nothing else in this suite would notice.
void testEvalContainerTornTailIsRepairedByTruncationBeforeTheNextAppend() {
  ScopedTempDir dir;
  const NNEvalContainer container = containerIn(dir, "repair");

  const shared_ptr<NNOutput> a = makeOutput(1, 1, 19, 19, true);
  const shared_ptr<NNOutput> b = makeOutput(2, 1, 9, 9, false);
  container.appendBlock({asStored(a), asStored(b)});
  const int64_t sizeAfterOneDump = sizeOf(container.path());
  // The bytes that MUST survive the repair unchanged, captured before anything is torn.
  const vector<uint8_t> intactPrefix = readBytesAt(container.path(), 0, (size_t)sizeAfterOneDump);

  const shared_ptr<NNOutput> c = makeOutput(3, 1, 19, 19, false);
  container.appendBlock({asStored(c)});
  truncateTo(container.path(), sizeOf(container.path()) - 5);
  const int64_t tornBytes = sizeOf(container.path()) - sizeAfterOneDump;

  const shared_ptr<NNOutput> d = makeOutput(4, 1, 19, 19, true);
  const NNEvalContainerAppendResult appended = container.appendBlock({asStored(d)});
  testAssert(appended.tornTailBytesDiscarded == tornBytes);
  testAssert(appended.tailRepair == NNCacheFileTailRepair::Truncated);

  // The surviving prefix, unchanged. A rewrite would have replaced these bytes with a
  // re-encoding of the live set, so this is the assertion that distinguishes the two
  // repairs rather than merely observing that one of them happened.
  testAssert(readBytesAt(container.path(), 0, (size_t)sizeAfterOneDump) == intactPrefix);
  // And nothing beyond the prefix but the one block just appended.
  testAssert(sizeOf(container.path()) == sizeAfterOneDump + appended.bytesAppended);

  const NNEvalContainerContents contents = container.load();
  testAssert(contents.tail() == NNEvalContainerTail::Intact);
  // The surviving prefix is the first dump's own block, still one block; the new dump is the
  // second.
  testAssert(contents.blocksApplied() == 2);
  assertSameEvaluation(*a, entryFor(contents, 1));
  assertSameEvaluation(*b, entryFor(contents, 2));
  testAssert(!hasEntryFor(contents, 3));
  // THE POINT OF THE TEST: the dump written after the torn tail is readable. Without the
  // repair this entry is what goes missing, silently.
  assertSameEvaluation(*d, entryFor(contents, 4));
}

// The state truncation newly makes reachable: a container whose INTACT PART IS EMPTY. A
// crash during the very first dump leaves a file that exists and carries nothing this build
// can read, and the repair shortens it to zero bytes -- where the rewrite it replaced would
// have left a fresh header behind. The next append must write that header rather than
// append a block to a headerless file, and a file that never gets its header is one no
// loader will ever accept, silently, forever.
void testEvalContainerRepairsAContainerWhoseIntactPartIsEmpty() {
  ScopedTempDir dir;
  const NNEvalContainer container = containerIn(dir, "emptyintact");

  const shared_ptr<NNOutput> a = makeOutput(1, 1, 19, 19, true);
  container.appendBlock({asStored(a)});
  // A crash before even the file header was whole: everything after a few bytes is gone.
  truncateTo(container.path(), 12);

  const shared_ptr<NNOutput> b = makeOutput(2, 1, 19, 19, false);
  const NNEvalContainerAppendResult appended = container.appendBlock({asStored(b)});
  testAssert(appended.tornTailBytesDiscarded == 12);
  testAssert(appended.tailRepair == NNCacheFileTailRepair::Truncated);
  // The header is part of what was appended, because the file had none left.
  testAssert(sizeOf(container.path()) == appended.bytesAppended);

  const NNEvalContainerContents contents = container.load();
  testAssert(contents.tail() == NNEvalContainerTail::Intact);
  testAssert(contents.blocksApplied() == 1);
  testAssert(!hasEntryFor(contents, 1));
  assertSameEvaluation(*b, entryFor(contents, 2));
}

// Compaction preserves the merged live set -- including the never-lose-an-ownermap rule --
// shrinks the file, and a crash mid-compaction leaves the original intact.
void testEvalContainerCompactionPreservesTheLiveSetAndSurvivesAStaleTemp() {
  ScopedTempDir dir;
  const NNEvalContainer container = containerIn(dir, "compact");

  const shared_ptr<NNOutput> a1 = makeOutput(1, 1, 19, 19, true);
  const shared_ptr<NNOutput> b1 = makeOutput(2, 1, 9, 9, false);
  container.appendBlock({asStored(a1), asStored(b1)});
  for(int i = 2; i <= 6; i++) {
    // Each later dump re-states key 1 WITHOUT a map -- which must never win -- and key 2
    // with a fresh generation, which must.
    const shared_ptr<NNOutput> aN = makeOutput(1, i, 19, 19, false);
    const shared_ptr<NNOutput> bN = makeOutput(2, i, 9, 9, false);
    container.appendBlock({asStored(aN), asStored(bN)});
  }
  const shared_ptr<NNOutput> b6 = makeOutput(2, 6, 9, 9, false);

  const int64_t sizeBefore = sizeOf(container.path());
  {
    const NNEvalContainerContents before = container.load();
    testAssert(before.entriesApplied() == 12);
    testAssert(before.entries().size() == 2);
  }

  // A crash mid-compaction leaves a stale temp at the compaction path. The original must be
  // untouched by it, and the next compaction must overwrite the stale temp rather than read
  // it.
  const string stalePath = container.path() + ".compacting";
  {
    FILE* f = fopen(stalePath.c_str(), "wb");
    testAssert(f != NULL);
    const char garbage[] = "a half-written compaction from a process that died";
    testAssert(fwrite(garbage, 1, sizeof(garbage), f) == sizeof(garbage));
    fclose(f);
  }
  {
    const NNEvalContainerContents afterStale = container.load();
    testAssert(afterStale.tail() == NNEvalContainerTail::Intact);
    testAssert(afterStale.entries().size() == 2);
    testAssert(sizeOf(container.path()) == sizeBefore);
  }

  testAssert(container.compactIfNeeded(NNEvalContainer::defaultCompactionMultiple()) == NNCacheFileMaintenance::Compacted);

  const NNEvalContainerContents after = container.load();
  testAssert(after.tail() == NNEvalContainerTail::Intact);
  testAssert(after.blocksApplied() == 1);
  testAssert(after.entriesApplied() == 2);
  testAssert(after.entries().size() == 2);
  // Observationally identical to the container it replaced: key 1 is still the generation-1
  // evaluation WITH its map, and key 2 is the newest one.
  assertSameEvaluation(*a1, entryFor(after, 1));
  testAssert(entryFor(after, 1).whiteOwnerMap != NULL);
  assertSameEvaluation(*b6, entryFor(after, 2));

  testAssert(sizeOf(container.path()) ==
             NNEvalContainer::fileHeaderBytesFor(MODEL) +
             (int64_t)NNEvalContainer::blockHeaderBytes() +
             NNEvalContainer::bytesForEntry(19, 19, true) +
             NNEvalContainer::bytesForEntry(9, 9, false));
  testAssert(sizeOf(container.path()) < sizeBefore);
  // The temp path is gone: the rename consumed it.
  testAssert(!FileUtils::exists(stalePath));

  // Under the multiple, a fresh compaction does not fire.
  testAssert(container.compactIfNeeded(NNEvalContainer::defaultCompactionMultiple()) == NNCacheFileMaintenance::Nothing);
}

// The boundary refuses what it cannot honor and never coerces it into something plausible.
void testEvalContainerRefusesWhatItCannotHonor() {
  ScopedTempDir dir;

  // Both names become path components, so both are validated to a closed alphabet.
  const char* badNames[] = {"", ".", "..", "../escape", "a/b", "a\\b", "with space", "semi;colon"};
  for(size_t i = 0; i < sizeof(badNames) / sizeof(badNames[0]); i++) {
    bool refusedContext = false;
    try { NNEvalContainer::forContextAndModel(dir.path(), badNames[i], MODEL, MODEL_VERSION); }
    catch(const StringError&) { refusedContext = true; }
    testAssert(refusedContext);

    bool refusedModel = false;
    try { NNEvalContainer::forContextAndModel(dir.path(), "ctx", badNames[i], MODEL_VERSION); }
    catch(const StringError&) { refusedModel = true; }
    testAssert(refusedModel);
  }
  // And accepts the alphabet it declares, including a real internalName.
  NNEvalContainer::forContextAndModel(dir.path(), "card-5455", MODEL, MODEL_VERSION);

  // A directory that does not exist is named, not created.
  {
    bool refused = false;
    try { NNEvalContainer::forContextAndModel(dir.path() + "/nope", "ctx", MODEL, MODEL_VERSION); }
    catch(const StringError&) { refused = true; }
    testAssert(refused);
  }

  // A missing file is not an error: it reads as an empty, intact container.
  {
    const NNEvalContainer container = containerIn(dir, "neverwritten");
    const NNEvalContainerContents contents = container.load();
    testAssert(contents.tail() == NNEvalContainerTail::Intact);
    testAssert(contents.entries().empty());
    testAssert(contents.blocksApplied() == 0);
  }

  // An entry carrying noisedPolicyProbs is refused rather than written without it, and the
  // refusal happens before the file is touched, so a rejected dump leaves no half-written
  // block behind.
  {
    const NNEvalContainer container = containerIn(dir, "noised");
    shared_ptr<NNOutput> noised = makeOutput(1, 1, 19, 19, false);
    noised->noisedPolicyProbs = new float[NNPos::MAX_NN_POLICY_SIZE];
    for(int i = 0; i < NNPos::MAX_NN_POLICY_SIZE; i++)
      noised->noisedPolicyProbs[i] = 0.0f;
    string message;
    bool refused = false;
    try { container.appendBlock({asStored(noised)}); }
    catch(const StringError& e) { refused = true; message = e.what(); }
    testAssert(refused);
    testAssert(!FileUtils::exists(container.path()));
    // Asserted on the DIAGNOSIS, because "it threw" would be true of several unrelated
    // defects. The message is operator-facing and load-bearing, which is what makes it a
    // legitimate anchor rather than adjacent prose (ADR-0021 Rule 3).
    testAssert(message.find("noisedPolicyProbs") != string::npos);
  }

  // A null entry is refused rather than dereferenced.
  {
    const NNEvalContainer container = containerIn(dir, "nullentry");
    bool refused = false;
    try { container.appendBlock({shared_ptr<const NNOutput>()}); }
    catch(const StringError&) { refused = true; }
    testAssert(refused);
    testAssert(!FileUtils::exists(container.path()));
  }

  // A board this build cannot represent is refused by name, not truncated into one it can.
  {
    const NNEvalContainer container = containerIn(dir, "bigboard");
    shared_ptr<NNOutput> big = makeOutput(1, 1, 19, 19, false);
    big->nnXLen = Board::MAX_LEN + 1;
    string message;
    bool refused = false;
    try { container.appendBlock({asStored(big)}); }
    catch(const StringError& e) { refused = true; message = e.what(); }
    testAssert(refused);
    testAssert(message.find("at most") != string::npos);
    // Put it back before the destructor runs over an ownermap sized for the real board.
    big->nnXLen = 19;
  }

  // A file that is not a container is refused by name rather than read on the chance that
  // its fields line up.
  {
    const NNEvalContainer container = containerIn(dir, "foreign");
    FILE* f = fopen(container.path().c_str(), "wb");
    testAssert(f != NULL);
    const char notAContainer[] = "SQLite format 3\0and then some more bytes so that it is not short at all";
    testAssert(fwrite(notAContainer, 1, sizeof(notAContainer), f) == sizeof(notAContainer));
    fclose(f);
    testAssert(loadIsRefused(container));
  }

  // A bumped version byte is refused, which is what the magic and version are for.
  {
    const NNEvalContainer container = containerIn(dir, "version");
    container.appendBlock({asStored(makeOutput(1, 1, 19, 19, false))});
    overwriteBytesAt(container.path(), 8, vector<uint8_t>{99, 0, 0, 0});
    testAssert(loadIsRefused(container));
  }

  // A container written for another CONTEXT is refused rather than merged.
  {
    const NNEvalContainer one = containerIn(dir, "ctxone");
    one.appendBlock({asStored(makeOutput(1, 1, 19, 19, false))});
    const NNEvalContainer two = containerIn(dir, "ctxtwo");
    gfs::copy_file(gfs::u8path(one.path()), gfs::u8path(two.path()));
    testAssert(loadIsRefused(two));
  }

  // THE LOAD-BEARING REFUSAL. A container written for another MODEL is refused, by name,
  // even though its bytes are a perfectly well-formed container and every key in it is a
  // key this model could legitimately have produced -- because the NN cache key names no
  // net, so nothing in the entries themselves could ever catch this.
  {
    const NNEvalContainer mine = containerIn(dir, "modelone");
    mine.appendBlock({asStored(makeOutput(1, 1, 19, 19, false))});
    const NNEvalContainer theirs =
      NNEvalContainer::forContextAndModel(dir.path(), "modelone", "some-other-net-v3", MODEL_VERSION);
    gfs::copy_file(gfs::u8path(mine.path()), gfs::u8path(theirs.path()));
    const string message = loadRefusalMessage(theirs);
    testAssert(message.find("some-other-net-v3") != string::npos);
    testAssert(message.find(MODEL) != string::npos);
  }

  // And the same file read under the right model name but a different model VERSION: the
  // name would have matched, and the outputs' meaning still depends on the version that
  // produced them.
  {
    const NNEvalContainer v14 = containerIn(dir, "modelversion");
    v14.appendBlock({asStored(makeOutput(1, 1, 19, 19, false))});
    const NNEvalContainer v9 =
      NNEvalContainer::forContextAndModel(dir.path(), "modelversion", MODEL, 9);
    testAssert(loadIsRefused(v9));
  }

  // A compaction multiple below 1 has no reading.
  {
    const NNEvalContainer container = containerIn(dir, "multiple");
    bool refused = false;
    try { container.compactIfNeeded(0); }
    catch(const StringError&) { refused = true; }
    testAssert(refused);
  }

  // The tail disposition and the discarded byte count cannot disagree.
  {
    bool refused = false;
    try {
      NNEvalContainerContents contents =
        NNEvalContainerContents::of({}, NNEvalContainerTail::Truncated, 0, 0, 0);
      (void)contents;
    }
    catch(const StringError&) { refused = true; }
    testAssert(refused);
  }
}

}  // namespace

void Tests::runNNEvalContainerTests() {
  cout << "Running nn eval container tests" << endl;

  testEvalContainerRoundTripsWithAndWithoutOwnershipMaps();
  testEvalContainerMergeIsLastWinsExceptAnOwnershipMapIsNeverLost();
  testEvalContainerAnOwnershipMapEarnedInALaterBlockIsAlsoNeverLost();
  testEvalContainerTornTailIsDiscardedAndThePrefixSurvives();
  testEvalContainerTruncationRefusesRatherThanShortReading();
  testEvalContainerTheBlocksKeyIndexIsGatheredAfterItsHeader();
  testEvalContainerAFileFromALaterVersionIsRefusedNotSilentlyRead();
  testEvalContainerAWholeButCorruptBlockIsRejectedByItsChecksum();
  testEvalContainerACorruptBlockHeaderDoesNotProduceAPartialApplication();
  testEvalContainerTornTailIsRepairedByTruncationBeforeTheNextAppend();
  testEvalContainerRepairsAContainerWhoseIntactPartIsEmpty();
  testEvalContainerCompactionPreservesTheLiveSetAndSurvivesAStaleTemp();
  testEvalContainerRefusesWhatItCannotHonor();

  // The format's own strides and sizes, so a report's arithmetic is quoted from the
  // implementation rather than from a second copy of the numbers.
  cout << "nnevals container: file header " << NNEvalContainer::fileHeaderBytesFor(MODEL)
       << " B for this model name, block header " << NNEvalContainer::blockHeaderBytes()
       << " B, entry header " << NNEvalContainer::entryHeaderBytes()
       << " B; a 19x19 entry is " << NNEvalContainer::bytesForEntry(19, 19, false)
       << " B without an ownership map and " << NNEvalContainer::bytesForEntry(19, 19, true)
       << " B with one" << endl;
}
