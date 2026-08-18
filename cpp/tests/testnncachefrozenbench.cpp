#include "../tests/tests.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include "../core/timer.h"
#include "../neuralnet/nncachefrozen.h"

using namespace std;

// The SPEC.md 8 performance floor for the frozen level-0 cache, and nothing else.
//
// THE FLOOR, restated so this file can be read alone. Single-threaded lookup latency must
// not exceed the prototype's at the corpus sizes that actually occur. The prototype's
// published figures (SPEC.md 8.2) are 13.255 ns/op at 4,096 keys, 27.526 at 32,768, 73.884
// at 262,144 and 109.745 at 524,288, interpolating to roughly 30-35 ns/op at the median
// real card of 45,664 and roughly 80 ns/op at the largest of 291,129.
//
// THE COMPARISON IS CROSS-SESSION, AND THAT IS A REAL WEAKNESS. SPEC.md 8.3 asks for both
// arms in one process wherever possible and says that where it is not -- and here it is
// not, because the prototype may not be built under the clean-room constraint -- the
// comparison against a published number must be stated as the materially weaker evidence
// it is rather than dressed up as a paired measurement. It is stated here and in the
// report.
//
// This is a measurement, so it is not part of runtests: it is reached by the
// `runnncachefrozenbench` subcommand, and its numbers are meaningless unless the box is
// otherwise idle and did not page during the window.

namespace {

// SPEC.md 8.3's own list: the first, second and fifth line up with the published table's
// rows, and the others are the real per-card percentiles of SPEC.md 8.1 (median 45,664,
// 90th percentile 121,796, maximum 291,129).
const int CORPUS_SIZES[6] = {4096, 32768, 45664, 121796, 262144, 291129};
const int64_t LOOKUPS_PER_REP = 1000000;
const int REPS = 15;

uint64_t splitmix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  uint64_t z = x;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

// Two disjoint key streams. `resident` builds the corpus; `foreign` is drawn from a
// serial range the corpus never uses, so every foreign key is a genuine absent query.
Hash128 residentKey(int64_t serial) {
  return Hash128(splitmix((uint64_t)serial * 2 + 1), splitmix((uint64_t)serial * 2 + 2));
}
Hash128 foreignKey(int64_t serial) {
  return residentKey(serial + 1000000000LL);
}

// A real NNOutput with no ownership map: 1528 bytes with the key at offset 0, which is the
// same size and offset SPEC.md 8.2 says the published floor was measured with.
shared_ptr<NNOutput> outputFor(Hash128 hash) {
  shared_ptr<NNOutput> p = make_shared<NNOutput>();
  p->nnHash = hash;
  p->nnXLen = 19;
  p->nnYLen = 19;
  return p;
}

double meanOf(const vector<double>& v) {
  double sum = 0.0;
  for(size_t i = 0; i < v.size(); i++) sum += v[i];
  return sum / (double)v.size();
}
double sdOf(const vector<double>& v) {
  if(v.size() < 2) return 0.0;
  const double mean = meanOf(v);
  double ss = 0.0;
  for(size_t i = 0; i < v.size(); i++) ss += (v[i] - mean) * (v[i] - mean);
  return sqrt(ss / (double)(v.size() - 1));
}

struct Reading {
  double mean; double sd; double min; double max;
};

Reading summarize(const vector<double>& v) {
  Reading r;
  r.mean = meanOf(v);
  r.sd = sdOf(v);
  r.min = *std::min_element(v.begin(), v.end());
  r.max = *std::max_element(v.begin(), v.end());
  return r;
}

void report(const char* what, int n, const Reading& r) {
  cout << "  n=" << n << " " << what
       << ": mean " << r.mean << " ns/op, sd " << r.sd
       << ", min " << r.min << ", max " << r.max
       << "  (n=" << REPS << " reps of " << LOOKUPS_PER_REP << " lookups)" << endl;
}

}  // namespace

void Tests::runNNCacheFrozenBench() {
  cout << "SPEC.md 8 performance floor for the frozen level-0 cache." << endl;
  cout << "Single-threaded, evaluations RETRIEVED (not membership-only), queries drawn "
       << "uniformly from the resident key set." << endl;
  cout << "The prototype figures this is compared against are PUBLISHED (SPEC.md 8.2), not "
       << "measured here: 13.255 ns/op at 4096, 27.526 at 32768, 73.884 at 262144, "
       << "109.745 at 524288, interpolating to ~30-35 at the 45664 median and ~80 at the "
       << "291129 maximum. A cross-session comparison against a published number is "
       << "materially weaker evidence than a paired one." << endl;

  for(int si = 0; si < 6; si++) {
    const int n = CORPUS_SIZES[si];
    vector<shared_ptr<NNOutput>> outputs;
    outputs.reserve((size_t)n);
    for(int i = 0; i < n; i++)
      outputs.push_back(outputFor(residentKey(i)));
    unique_ptr<NNCacheFrozen> frozen = NNCacheFrozen::build(std::move(outputs));

    cout << "n=" << n << " built; structure "
         << ((double)frozen->structureBytes() / (double)n) << " B/entry" << endl;

    vector<double> hitNs;
    vector<double> missNs;
    shared_ptr<NNOutput> got;
    for(int rep = 0; rep < REPS; rep++) {
      // Hit path. The index is scattered rather than swept, so the stream is not
      // prefetchable and the measurement is of a random-access lookup, which is what a
      // search does.
      {
        int64_t found = 0;
        ClockTimer timer;
        for(int64_t i = 0; i < LOOKUPS_PER_REP; i++) {
          const int64_t which = (int64_t)(splitmix((uint64_t)(rep * LOOKUPS_PER_REP + i)) % (uint64_t)n);
          if(frozen->get(residentKey(which), got))
            found += 1;
        }
        const double seconds = timer.getSeconds();
        // Every query hit a resident key, so anything else means the measurement was not
        // measuring what it says it was.
        testAssert(found == LOOKUPS_PER_REP);
        hitNs.push_back(seconds * 1.0e9 / (double)LOOKUPS_PER_REP);
      }
      // Absent path (SPEC.md 2.4, 8.3). Not part of the floor, but a pathological miss
      // path is a defect the hit numbers would hide.
      {
        int64_t found = 0;
        ClockTimer timer;
        for(int64_t i = 0; i < LOOKUPS_PER_REP; i++) {
          const int64_t which = (int64_t)(splitmix((uint64_t)(rep * LOOKUPS_PER_REP + i)) % (uint64_t)n);
          if(frozen->get(foreignKey(which), got))
            found += 1;
        }
        const double seconds = timer.getSeconds();
        testAssert(found == 0);
        missNs.push_back(seconds * 1.0e9 / (double)LOOKUPS_PER_REP);
      }
    }
    report("hit ", n, summarize(hitNs));
    report("miss", n, summarize(missNs));
    cout << flush;
  }
  cout << "Done" << endl;
}
