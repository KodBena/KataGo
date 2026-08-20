#include "../tests/tests.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "../core/timer.h"
#include "../neuralnet/nncacheprobed.h"

using namespace std;
using namespace NNCacheProbed;

// The inline-tag measurement, and nothing else.
//
// THE QUESTION. A probed table must reject `ways` candidate slots to conclude a miss.
// Each slot holds a shared_ptr<NNOutput>; the key lives at offset 0 of the NNOutput,
// which is a separate heap block of 1528 bytes (2972 with an ownership map). So
// without an inline tag, rejecting one occupied slot costs a dereference into a
// different, cold cache line, and rejecting a whole probe set costs `ways` of them.
// With a 4-byte tag stored beside the pointer, a rejection reads a word the probe
// already pulled in.
//
// THE MEASUREMENT. Both variants of the SAME table template are instantiated here --
// this is the only reason NNCacheTableProbed lives in a header -- and both are built
// in one process, from one build, populated from the same key stream, and timed on the
// same lookup stream. The arms are run in alternating order across rounds, because an
// earlier measurement on this programme produced a statistically significant result
// that turned out to be nothing but always running one arm first.
//
// This is a measurement, so it is not part of runtests: it is reached by the
// `runnncachebench` subcommand, and its numbers are meaningless unless the box is
// otherwise idle.

namespace {

typedef NNCacheTableProbed<LinearProbe,LruEviction,true> TaggedTable;
typedef NNCacheTableProbed<LinearProbe,LruEviction,false> UntaggedTable;

static const int SIZE_POW = 17;   // 131072 slots
static const int POOL_POW = 11;   // 2048 regions of 64 slots
static const int64_t FILL_MULTIPLIER = 3;   // inserts per slot, to drive occupancy up
static const int64_t LOOKUPS = 1000000;
static const int ROUNDS = 4;
// The cost the tag removes is one pointer chase per REJECTED way, so it must scale
// with associativity. Sweeping the whole range the config accepts here is what makes
// "the tag is a prerequisite" a claim about probing rather than about one setting.
static const int WAYS_SWEEP[4] = {2,4,8,16};

// A key stream with no structure the table could exploit: splitmix64 over a counter.
static Hash128 streamKey(uint64_t serial) {
  uint64_t s0 = serial * 2 + 1;
  uint64_t s1 = serial * 2 + 2;
  return Hash128(splitMix64(s0), splitMix64(s1));
}

static shared_ptr<NNOutput> streamEntry(Hash128 hash) {
  shared_ptr<NNOutput> p = make_shared<NNOutput>();
  p->nnHash = hash;
  p->nnXLen = 19;
  p->nnYLen = 19;
  return p;
}

template<class Table>
static void populate(Table& table, int64_t inserts) {
  for(int64_t i = 0; i<inserts; i++)
    table.set(streamEntry(streamKey((uint64_t)i)));
}

// Times `lookups` gets over keys drawn from [keyBase, keyBase+keyRange). The miss arm
// draws from a range that was never inserted, so every probe set must be rejected in
// full; the hit arm draws from the most recently inserted keys, which are the ones
// still resident, so a probe usually stops early.
template<class Table>
static double timeLookups(
  Table& table, int64_t lookups, int64_t keyBase, int64_t keyRange, int64_t& foundOut
) {
  shared_ptr<NNOutput> got;
  int64_t found = 0;
  ClockTimer timer;
  for(int64_t i = 0; i<lookups; i++) {
    if(table.get(streamKey((uint64_t)(keyBase + (i % keyRange))),got))
      found += 1;
  }
  const double seconds = timer.getSeconds();
  foundOut = found;
  return seconds;
}

static double meanOf(const vector<double>& v) {
  double sum = 0.0;
  for(size_t i = 0; i<v.size(); i++) sum += v[i];
  return sum / (double)v.size();
}

static double sdOf(const vector<double>& v) {
  if(v.size() < 2) return 0.0;
  const double mean = meanOf(v);
  double ss = 0.0;
  for(size_t i = 0; i<v.size(); i++) ss += (v[i]-mean)*(v[i]-mean);
  return sqrt(ss / (double)(v.size()-1));
}

// One A/B at one associativity, on one lookup stream. Both arms are populated from the
// same key stream and run in ALTERNATING order across rounds, because an earlier
// measurement on this programme produced a significant, entirely spurious effect from
// nothing but always running one arm first.
struct ArmResult { double taggedMean; double taggedSd; double untaggedMean; double untaggedSd; };

static ArmResult runArms(
  TaggedTable& tagged, UntaggedTable& untagged,
  int64_t keyBase, int64_t keyRange, int64_t expectedFound
) {
  vector<double> taggedNs;
  vector<double> untaggedNs;
  for(int round = 0; round<ROUNDS; round++) {
    int64_t found = 0;
    double a = 0.0, b = 0.0;
    if(round % 2 == 0) {
      a = timeLookups(tagged,LOOKUPS,keyBase,keyRange,found);
      testAssert(found == expectedFound);
      b = timeLookups(untagged,LOOKUPS,keyBase,keyRange,found);
      testAssert(found == expectedFound);
    }
    else {
      b = timeLookups(untagged,LOOKUPS,keyBase,keyRange,found);
      testAssert(found == expectedFound);
      a = timeLookups(tagged,LOOKUPS,keyBase,keyRange,found);
      testAssert(found == expectedFound);
    }
    taggedNs.push_back(a * 1.0e9 / (double)LOOKUPS);
    untaggedNs.push_back(b * 1.0e9 / (double)LOOKUPS);
  }
  ArmResult r;
  r.taggedMean = meanOf(taggedNs);
  r.taggedSd = sdOf(taggedNs);
  r.untaggedMean = meanOf(untaggedNs);
  r.untaggedSd = sdOf(untaggedNs);
  return r;
}

static void printArm(const char* what, int ways, const ArmResult& r) {
  cout << "  ways=" << ways << " " << what
       << ": tagged " << r.taggedMean << " +/- " << r.taggedSd
       << " ns/op, untagged " << r.untaggedMean << " +/- " << r.untaggedSd
       << " ns/op, untagged/tagged = " << (r.untaggedMean / r.taggedMean)
       << ", cost of no tag = " << (r.untaggedMean - r.taggedMean) << " ns/op" << endl;
}

}  // namespace

void Tests::runNNCacheBench() {
  const int64_t slots = ((int64_t)1) << SIZE_POW;
  const int64_t inserts = slots * FILL_MULTIPLIER;

  cout << "nn cache inline-tag benchmark" << endl;
  cout << "  table: linearprobe, lru, 2^" << SIZE_POW << " slots in 2^" << POOL_POW
       << " lock regions (" << (slots >> POOL_POW) << " slots per region)" << endl;
  cout << "  each arm populated with " << inserts << " distinct keys; sizeof(NNOutput) = "
       << sizeof(NNOutput) << " B, so each arm's resident payload is about "
       << (slots * (int64_t)sizeof(NNOutput)) / (1024*1024) << " MiB" << endl;
  cout << "  " << LOOKUPS << " lookups per round, " << ROUNDS
       << " rounds per arm, arms run in alternating order" << endl;
  cout << "  slot bytes as shipped (lru stamp + shared_ptr + tag): "
       << TaggedTable::slotBytes() << " B; removing the tag field would save 8 B/slot,"
       << " not 4, because of alignment" << endl;

  // The miss stream draws keys that were never inserted, so every probe set must be
  // rejected in full. The hit stream draws from the last `slots` keys inserted, which
  // under LRU are the ones still resident.
  const int64_t missBase = inserts * 4;
  const int64_t hitBase = inserts - slots;

  for(int w = 0; w<4; w++) {
    const int ways = WAYS_SWEEP[w];
    TaggedTable tagged(SIZE_POW,POOL_POW,ways);
    UntaggedTable untagged(SIZE_POW,POOL_POW,ways);
    populate(tagged,inserts);
    populate(untagged,inserts);

    const ArmResult missArm = runArms(tagged,untagged,missBase,LOOKUPS,0);
    printArm("miss",ways,missArm);

    // How many of the hit stream are actually resident is a property of the table, not
    // of the tag, so both arms must agree on it -- and they do, or the assertion inside
    // runArms would have fired. Measure it once so the hit rate is on the record.
    shared_ptr<NNOutput> got;
    int64_t resident = 0;
    for(int64_t i = 0; i<LOOKUPS; i++)
      resident += tagged.get(streamKey((uint64_t)(hitBase + (i % LOOKUPS))),got) ? 1 : 0;
    const ArmResult hitArm = runArms(tagged,untagged,hitBase,LOOKUPS,resident);
    cout << "  ways=" << ways << " hit stream resident fraction: "
         << (double)resident / (double)LOOKUPS << endl;
    printArm("hit ",ways,hitArm);
  }
}
