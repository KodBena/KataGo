#include "../tests/tests.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include "../core/timer.h"
#include "../neuralnet/nncache.h"
#include "../neuralnet/nncachefrozen.h"
#include "../neuralnet/nncachetwolevel.h"

using namespace std;

// WHAT THE ORDERED RESOLUTION LIST COSTS THE MISS PATH, measured rather than argued.
//
// THE CLAIM UNDER TEST is the acceptance bar for generalising the two-level table from one
// level 0 to a list: the miss path must not regress against the single-source shape that
// preceded it, and each additional attached source must cost about one CHD probe -- the
// ~40-110 ns at real sizes the design's own arithmetic predicts, against the 2.4-2.8 ms an
// avoided evaluation costs. The bar is NO REGRESSION, not no cost.
//
// THE METHOD, stated because a number without one is an opinion (ADR-0009).
//
//   WHAT THIS FILE MEASURES, AND WHAT IT DOES NOT. It measures the LIST: the miss path at
//   one through four attached sources, so the per-source cost is read off the differences
//   between its own arms, in one process, over the same key streams. Arm A0 -- the frozen
//   source probed directly through a raw pointer, then level 1 -- is reported beside them as
//   the floor of the shape, and is NOT a baseline for the list: it also skips the virtual
//   NNCacheTable::get dispatch the two-level table paid before this change and pays after
//   it, so quoting it as the baseline would charge the list for a cost it did not introduce.
//
//   THE NO-REGRESSION COMPARISON AGAINST THE PRE-CHANGE TABLE IS NOT IN THIS FILE, and that
//   is deliberate. The only faithful reference is the pre-change implementation itself, and
//   the honest way to have it is to compile it -- verbatim, from the commit before this one
//   -- into the same binary and interleave the two arms rep by rep. That was done once, and
//   its result is recorded in nncachetwolevel.h beside the code it is about. The scaffold is
//   NOT kept in this tree, because a verbatim copy of deleted code committed beside its
//   replacement is a second home for one fact that nothing keeps honest (ADR-0012 P1); it is
//   kept, with the recipe to regenerate it, under
//   audit-reports/impl-level0-multi-source-measurement/. A hand-written stand-in for the
//   pre-change table was tried first and is exactly what this note exists to warn the next
//   reader off: it omitted the hit ledger and the object layout that came with it, and it
//   read 5-9 ns FASTER than the real thing, which would have turned a real cost into an
//   invented one (or hidden it) depending on which way the stub happened to differ.
//
//   THE MISS PATH IS THE QUANTITY. Queries are drawn from a key stream disjoint from every
//   source's, so every lookup traverses every attached source and then level 1 and finds
//   nothing. That is the path the extra sources are on; a hit arm is reported beside it
//   because a hit stops at the first holder and therefore should NOT move with the source
//   count, which is a check on the measurement itself rather than on the code.
//
//   REPORTED WITH ITS SPREAD. Each arm is n=REPS repetitions of LOOKUPS_PER_REP lookups;
//   mean, sample standard deviation, min and max are all printed, so a reader can see
//   whether a difference between arms is larger than the noise rather than being told it is
//   (ADR-0009's behavioural tier: a difference is a number with an uncertainty beside it).
//
//   THE SUBSTRATE IS NOT CONTROLLED BY THIS FILE. It is a measurement, so it is not part of
//   runtests: it is reached by the `runnncachetwolevelbench` subcommand, it allocates a few
//   hundred megabytes, and its numbers mean nothing unless the box was otherwise idle and
//   did not page during the window. Whoever quotes it says which.

namespace {

// The median real card in the operator's corpus (SPEC.md 8.1), so a "per attached source"
// figure is a figure at the size sources actually are.
const int ENTRIES_PER_SOURCE = 45664;
const int MAX_SOURCES = 4;
const int64_t LOOKUPS_PER_REP = 1000000;
const int REPS = 11;

uint64_t splitmix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  uint64_t z = x;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

// Disjoint key streams: source s owns serials [s*ENTRIES, (s+1)*ENTRIES), and the foreign
// stream is drawn from a range no source uses, so every foreign key is a genuine miss in
// every attached source.
Hash128 keyOf(int64_t serial) {
  return Hash128(splitmix((uint64_t)serial * 2 + 1), splitmix((uint64_t)serial * 2 + 2));
}
Hash128 foreignKey(int64_t serial) {
  return keyOf(serial + 1000000000LL);
}

unique_ptr<NNOutput> outputFor(Hash128 hash) {
  unique_ptr<NNOutput> p(new NNOutput());
  p->nnHash = hash;
  p->nnXLen = 19;
  p->nnYLen = 19;
  return p;
}

unique_ptr<NNCacheFrozen> sourceNumber(int which) {
  vector<unique_ptr<NNOutput>> outputs;
  outputs.reserve((size_t)ENTRIES_PER_SOURCE);
  for(int i = 0; i < ENTRIES_PER_SOURCE; i++)
    outputs.push_back(outputFor(keyOf((int64_t)which * ENTRIES_PER_SOURCE + i)));
  return NNCacheFrozen::build(std::move(outputs));
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

void report(const string& what, const vector<double>& ns) {
  cout << "  " << what
       << ": mean " << meanOf(ns) << " ns/op, sd " << sdOf(ns)
       << ", min " << *std::min_element(ns.begin(), ns.end())
       << ", max " << *std::max_element(ns.begin(), ns.end()) << endl;
}

}  // namespace

void Tests::runNNCacheTwoLevelBench() {
  cout << "The ordered resolution list's cost on the level-0 miss path." << endl;
  cout << "In one process over the same sources and key streams. Arm A0 probes the frozen "
       << "source directly (no virtual dispatch) and is the floor of the shape, not a "
       << "baseline for the list; arm B is the list at 1.." << MAX_SOURCES
       << " attached sources, and the per-source cost is the difference between its arms. "
       << ENTRIES_PER_SOURCE << " entries per source, disjoint key sets." << endl;
  cout << "Numbers are meaningless unless the box was idle and did not page." << endl;

  vector<unique_ptr<NNCacheFrozen>> built;
  for(int s = 0; s < MAX_SOURCES; s++)
    built.push_back(sourceNumber(s));
  NNCacheFrozen* firstSource = built[0].get();

  // Query streams built OUTSIDE the timed window: constructing a key costs two splitmix
  // rounds, which is real work of the same order as the lookup it feeds.
  vector<Hash128> missQueries((size_t)LOOKUPS_PER_REP);
  vector<Hash128> hitQueries((size_t)LOOKUPS_PER_REP);
  for(int64_t i = 0; i < LOOKUPS_PER_REP; i++) {
    missQueries[(size_t)i] = foreignKey((int64_t)(splitmix((uint64_t)i) % (uint64_t)ENTRIES_PER_SOURCE));
    // Drawn from the FIRST source, so the hit arm stops at the first holder no matter how
    // many sources are attached -- which is why it should not move with the source count.
    hitQueries[(size_t)i] = keyOf((int64_t)(splitmix((uint64_t)i + 7) % (uint64_t)ENTRIES_PER_SOURCE));
  }

  // Arm A0: the raw floor, no virtual dispatch. Reported, not used as the baseline.
  unique_ptr<NNCacheTable> referenceLevelOne = NNCacheTable::create(NNCacheConfig::statusQuo(16, 2));
  {
    vector<double> missNs;
    vector<double> hitNs;
    for(int rep = 0; rep < REPS; rep++) {
      shared_ptr<NNOutput> got;
      ClockTimer timer;
      int64_t found = 0;
      for(int64_t i = 0; i < LOOKUPS_PER_REP; i++) {
        if(firstSource->get(missQueries[(size_t)i], got)) found++;
        else if(referenceLevelOne->get(missQueries[(size_t)i], got)) found++;
      }
      missNs.push_back(timer.getSeconds() * 1.0e9 / (double)LOOKUPS_PER_REP);
      if(found != 0) throw StringError("bench: the miss stream hit something");

      ClockTimer hitTimer;
      for(int64_t i = 0; i < LOOKUPS_PER_REP; i++) {
        if(firstSource->get(hitQueries[(size_t)i], got)) found++;
        else if(referenceLevelOne->get(hitQueries[(size_t)i], got)) found++;
      }
      hitNs.push_back(hitTimer.getSeconds() * 1.0e9 / (double)LOOKUPS_PER_REP);
      if(found != LOOKUPS_PER_REP) throw StringError("bench: the hit stream missed something");
    }
    report("arm A0 direct probe, no vcall  miss", missNs);
    report("arm A0 direct probe, no vcall  hit ", hitNs);
  }

  // Arms B: the list, at one through MAX_SOURCES attached sources.
  unique_ptr<NNCacheTwoLevelTable> table =
    makeTwoLevelNNCacheTable(std::move(built[0]), NNCacheTable::create(NNCacheConfig::statusQuo(16, 2)), 16);
  for(int attached = 1; attached <= MAX_SOURCES; attached++) {
    if(attached > 1)
      (void)table->attachLevelZero(std::move(built[(size_t)attached - 1]));
    if((int)table->numLevelZeroSources() != attached)
      throw StringError("bench: the list is not the size this arm says it is");

    vector<double> missNs;
    vector<double> hitNs;
    for(int rep = 0; rep < REPS; rep++) {
      shared_ptr<NNOutput> got;
      int64_t found = 0;
      ClockTimer timer;
      for(int64_t i = 0; i < LOOKUPS_PER_REP; i++)
        if(table->get(missQueries[(size_t)i], got)) found++;
      missNs.push_back(timer.getSeconds() * 1.0e9 / (double)LOOKUPS_PER_REP);
      if(found != 0) throw StringError("bench: the miss stream hit something");

      ClockTimer hitTimer;
      for(int64_t i = 0; i < LOOKUPS_PER_REP; i++)
        if(table->get(hitQueries[(size_t)i], got)) found++;
      hitNs.push_back(hitTimer.getSeconds() * 1.0e9 / (double)LOOKUPS_PER_REP);
      if(found != LOOKUPS_PER_REP) throw StringError("bench: the hit stream missed something");
    }
    report("arm B  list, " + Global::intToString(attached) + " source(s) miss", missNs);
    report("arm B  list, " + Global::intToString(attached) + " source(s) hit ", hitNs);
  }
}
