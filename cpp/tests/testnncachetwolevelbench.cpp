#include "../tests/tests.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <optional>
#include <vector>

#include "../core/timer.h"
#include "../neuralnet/nncache.h"
#include "../neuralnet/nncachefrozen.h"
#include "../neuralnet/nncachetwolevel.h"
#include "../tests/testcacheswapseam.h"

using namespace std;

// The declared test seam for the level-0 swap door; see tests/testcacheswapseam.h. One permit,
// minted once for this file, spent at every attach and detach below.
static const NNCacheLevelZeroSwapPermit SWAP_PERMIT = NNCacheLevelZeroSwapTestSeam::permit();

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

// THE LARGEST REAL CARD IN THE OPERATOR'S CORPUS, for the attach arm: the reconcile is
// O(entries), so the figure that matters is the one at the size the walk is longest.
const int LARGEST_CARD_ENTRIES = 291129;
const int ATTACH_REPS = 5;

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
      (void)table->attachLevelZero(SWAP_PERMIT, std::move(built[(size_t)attached - 1]), std::optional<NNCacheContextId>());
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

  // ARM C: WHAT ATTACH'S RECONCILE COSTS, which is the one cost the attach contract added.
  //
  // Attach asks level 1, through NNCacheTable::contains, about every entry of the arriving
  // source, and shadows the ones level 1 already owns. So the quantity is per ENTRY and it has
  // two regimes, and both are measured because the design's own budget was argued from an
  // estimate of one of them:
  //
  //   NOTHING OWNED -- every probe misses. This is the ordinary first attach of a session.
  //   EVERYTHING OWNED -- every probe hits, shadows, and folds a count into the ledger. This is
  //   the worst case the contract can produce: a card re-attached after the session has already
  //   re-evaluated all of it.
  //
  // Reported in MILLISECONDS PER ATTACH at the two real card sizes, because that is the currency
  // the session-boundary budget is stated in, with the per-entry figure beside it. Each rep
  // builds a fresh source OUTSIDE the timed window -- attach consumes it -- and detaches after,
  // so the arms do not accumulate sources.
  {
    const int sizes[2] = {ENTRIES_PER_SOURCE, LARGEST_CARD_ENTRIES};
    for(int owned = 0; owned <= 1; owned++) {
      for(int which = 0; which < 2; which++) {
        const int n = sizes[which];
        vector<double> ms;
        int64_t reconciled = 0;
        for(int rep = 0; rep < ATTACH_REPS; rep++) {
          // A fresh key range per rep and per arm, so no rep is served by another's state.
          const int64_t base = (int64_t)2000000000LL + (int64_t)(owned * 4 + which) * 1000000LL +
                               (int64_t)rep * 300000LL;
          vector<unique_ptr<NNOutput>> outputs;
          outputs.reserve((size_t)n);
          for(int i = 0; i < n; i++)
            outputs.push_back(outputFor(keyOf(base + i)));
          unique_ptr<NNCacheFrozen> source = NNCacheFrozen::build(std::move(outputs));
          // A level 1 of its own per rep, big enough to hold the whole card without collisions
          // deciding the answer: what is being measured is the probe, not an eviction policy.
          unique_ptr<NNCacheTwoLevelTable> attachTable = makeTwoLevelNNCacheTable(
            NNCacheFrozen::build(vector<unique_ptr<NNOutput>>()),
            NNCacheTable::create(NNCacheConfig::statusQuo(22, 10)),
            16
          );
          if(owned == 1) {
            for(int i = 0; i < n; i++) {
              shared_ptr<NNOutput> p = make_shared<NNOutput>();
              p->nnHash = keyOf(base + i);
              p->nnXLen = 19;
              p->nnYLen = 19;
              attachTable->set(p);
            }
          }
          ClockTimer timer;
          const NNCacheLevelZeroAttachment attachment =
            attachTable->attachLevelZero(SWAP_PERMIT, std::move(source), std::optional<NNCacheContextId>());
          ms.push_back(timer.getSeconds() * 1.0e3);
          // NOT AN EQUALITY on the owned arm: level 1 here is the shipped direct-mapped table, so
          // a few thousand of the keys offered to it displaced each other and are genuinely not
          // owned. What the arm needs is that the reconcile really did the work -- almost every
          // probe hitting, shadowing and folding a count -- and the count it actually reconciled
          // is printed rather than assumed.
          reconciled = attachment.entriesLevelOneAlreadyOwned;
          if(owned == 0 && reconciled != 0)
            throw StringError("bench: the empty-level-1 arm reconciled something");
          if(owned == 1 && reconciled < (int64_t)n * 9 / 10)
            throw StringError("bench: the owned arm did not reconcile the card it had set up");
        }
        vector<double> perEntryNs;
        for(size_t i = 0; i < ms.size(); i++)
          perEntryNs.push_back(ms[i] * 1.0e6 / (double)n);
        const string label =
          string("arm C  attach ") + Global::intToString(n) + " entries, level 1 owns " +
          (owned == 1 ? "all" : "none");
        cout << "  " << label << " (" << reconciled << " reconciled): mean " << meanOf(ms) << " ms/attach, sd " << sdOf(ms)
             << ", min " << *std::min_element(ms.begin(), ms.end())
             << ", max " << *std::max_element(ms.begin(), ms.end()) << endl;
        report(label + " (per entry)", perEntryNs);
      }
    }
  }

  // ARM D: WHAT OBSERVATION COUNTING COSTS ON THE PATH MCTS HAMMERS.
  //
  // NNEvaluator::evaluate mints one NNCachePresentation per demand via NNCacheTable::present, before any level is
  // consulted, so the added cost sits on the same loop the arms above measure. Three states,
  // because they are three different costs and a single figure would hide the one that
  // matters most:
  //
  //   D0 NO CONTEXT EVER ATTACHED. The overwhelmingly common configuration -- plain katago
  //   play, no cache directory -- and the one the design is REQUIRED to leave
  //   indistinguishable from baseline. present() is inlined and the ledger pointer is null for
  //   the life of the process, so what is being measured is one always-not-taken branch.
  //
  //   D1 A CONTEXT ATTACHED, THE REQUEST NAMING NONE. One more test and no work: an
  //   observation belongs to a card or to no file at all.
  //
  //   D2 A CONTEXT ATTACHED AND NAMED. The real cost: one pooled mutex, one bounded probe over
  //   32-byte rows, one increment. This is what a study session actually pays.
  //
  // MEASURED AS A DIFFERENCE IN ONE PROCESS, INTERLEAVED REP BY REP against the same loop
  // WITHOUT the mint, over the same key stream, against the same table. That is a
  // stronger protocol than an A/B across two builds for this particular quantity, and
  // deliberately so: a cross-build A/B of a few nanoseconds is exactly the shape a compiler
  // flag mismatch fakes, and this project has already been handed one fake 2x regression that
  // way. Here both arms are the same binary, the same optimisation, the same instruction cache
  // -- the only difference between them is the call under test.
  //
  // The baseline arm is measured FIRST in each rep and the observing arm SECOND, and then the
  // order is reversed on odd reps, so a warm-up asymmetry cannot be read as the effect.
  {
    cout << "arm D: what NNCacheTable::present costs per evaluation request, as a difference "
         << "measured in one process against the same loop without it." << endl;

    // A plain single-level table: level 0 is irrelevant to this arm, and using one would put
    // the resolution list's own cost inside both arms for nothing.
    unique_ptr<NNCacheTable> plain = NNCacheTable::create(NNCacheConfig::statusQuo(16, 2));
    const NNCacheAttribution unattributed = NNCacheAttribution::noAttributableContext();

    // D0 first, and it must run BEFORE any context is attached: attaching one allocates the
    // ledger for the life of the table, and there is no way back.
    vector<double> baseNs0;
    vector<double> obsNs0;
    for(int rep = 0; rep < REPS; rep++) {
      shared_ptr<NNOutput> got;
      int64_t found = 0;
      const bool baseFirst = (rep % 2) == 0;
      double base = 0.0;
      double obs = 0.0;
      for(int which = 0; which < 2; which++) {
        const bool doBase = (which == 0) == baseFirst;
        ClockTimer timer;
        if(doBase) {
          for(int64_t i = 0; i < LOOKUPS_PER_REP; i++)
            if(plain->get(missQueries[(size_t)i], got)) found++;
          base = timer.getSeconds() * 1.0e9 / (double)LOOKUPS_PER_REP;
        }
        else {
          for(int64_t i = 0; i < LOOKUPS_PER_REP; i++) {
            (void)plain->present(missQueries[(size_t)i], unattributed);
            if(plain->get(missQueries[(size_t)i], got)) found++;
          }
          obs = timer.getSeconds() * 1.0e9 / (double)LOOKUPS_PER_REP;
        }
      }
      if(found != 0) throw StringError("bench: the miss stream hit something");
      baseNs0.push_back(base);
      obsNs0.push_back(obs);
    }
    report("arm D0 no context, get only          ", baseNs0);
    report("arm D0 no context, observe + get     ", obsNs0);
    cout << "  arm D0 added cost (minima): "
         << (*std::min_element(obsNs0.begin(), obsNs0.end()) - *std::min_element(baseNs0.begin(), baseNs0.end()))
         << " ns/request" << endl;

    // Now a context exists, so the ledger is allocated and D1/D2 measure the real structure.
    const NNCacheContextId card = plain->attachCacheContext("bench-card");
    const NNCacheAttribution attributed = NNCacheAttribution::toContext(card);
    const NNCacheAttribution* const arms[2] = {&unattributed, &attributed};
    const char* const armNames[2] = {"arm D1 context attached, unattributed", "arm D2 context attached, named       "};
    for(int arm = 0; arm < 2; arm++) {
      vector<double> baseNs;
      vector<double> obsNs;
      for(int rep = 0; rep < REPS; rep++) {
        shared_ptr<NNOutput> got;
        int64_t found = 0;
        const bool baseFirst = (rep % 2) == 0;
        double base = 0.0;
        double obs = 0.0;
        for(int which = 0; which < 2; which++) {
          const bool doBase = (which == 0) == baseFirst;
          ClockTimer timer;
          if(doBase) {
            for(int64_t i = 0; i < LOOKUPS_PER_REP; i++)
              if(plain->get(missQueries[(size_t)i], got)) found++;
            base = timer.getSeconds() * 1.0e9 / (double)LOOKUPS_PER_REP;
          }
          else {
            for(int64_t i = 0; i < LOOKUPS_PER_REP; i++) {
              (void)plain->present(missQueries[(size_t)i], *arms[arm]);
              if(plain->get(missQueries[(size_t)i], got)) found++;
            }
            obs = timer.getSeconds() * 1.0e9 / (double)LOOKUPS_PER_REP;
          }
        }
        if(found != 0) throw StringError("bench: the miss stream hit something");
        baseNs.push_back(base);
        obsNs.push_back(obs);
      }
      report(string(armNames[arm]) + " get only     ", baseNs);
      report(string(armNames[arm]) + " observe + get", obsNs);
      cout << "  " << armNames[arm] << " added cost (minima): "
           << (*std::min_element(obsNs.begin(), obsNs.end()) - *std::min_element(baseNs.begin(), baseNs.end()))
           << " ns/request" << endl;
    }
    // What D2 actually recorded, so the arm is not silently measuring an early-return: the
    // stream is 1e6 lookups over ENTRIES_PER_SOURCE distinct keys, and the ledger's overflow
    // counter says whether every one of them found a row.
    const NNCacheObservationLedger observed = plain->harvestObservationCountsFor(card);
    cout << "  arm D2 recorded " << observed.entries().size() << " distinct keys, "
         << observed.unrecordedObservations() << " observations with no row available" << endl;

    // ARM D3: WHERE ARM D2'S COST ACTUALLY GOES, because a bare "55 ns" with no attribution is
    // a number nobody can act on (ADR-0009: measure the mechanism before restructuring for it).
    //
    // Two candidates -- the pooled mutex, and a random access into a table far larger than any
    // cache -- separated by varying ONLY the second. Both arms run the recorder alone, take the
    // same lock, and execute the same probe; they differ in whether the rows they touch are
    // resident. The DIFFERENCE is the memory. What is LEFT at the small size is the lock, the
    // hash mix and the compare.
    //
    // THE SMALL ARM'S KEY STREAM IS BOUNDED TO KEEP IT A CONTROL. A 2^12-row table fed the full
    // 45,664-key stream overflows its probe window on almost every call and walks all 16 slots,
    // which is MORE instruction work rather than less -- the first version of this arm did
    // exactly that and its number could not be read as "the same work without the memory". The
    // stream here is cut to a distinct-key count that leaves the small table at the same load
    // factor the production one sits at, so the probe lengths match and the arms are comparable.
    {
      const int smallPow = 12;
      const int64_t smallDistinctKeys =
        (((int64_t)1) << smallPow) * NNCacheObservationRecorder::maxLoadFactorPercent() / 100;
      for(int arm = 0; arm < 2; arm++) {
        const bool production = arm == 0;
        const int pow2 = production ? NNCacheObservationRecorder::defaultPowerOfTwo() : smallPow;
        const int mutexPow = pow2 < 10 ? pow2 : 10;
        NNCacheObservationRecorder recorder(pow2, mutexPow);
        // The production arm sees the whole stream; the control sees a bounded slice of the same
        // keys, so the two differ in table size and in nothing about the keys themselves.
        vector<Hash128> stream((size_t)LOOKUPS_PER_REP);
        for(int64_t i = 0; i < LOOKUPS_PER_REP; i++)
          stream[(size_t)i] = production
            ? missQueries[(size_t)i]
            : missQueries[(size_t)(splitmix((uint64_t)i) % (uint64_t)smallDistinctKeys)];
        vector<double> ns;
        for(int rep = 0; rep < REPS; rep++) {
          ClockTimer timer;
          for(int64_t i = 0; i < LOOKUPS_PER_REP; i++)
            recorder.observe(stream[(size_t)i], 0);
          ns.push_back(timer.getSeconds() * 1.0e9 / (double)LOOKUPS_PER_REP);
        }
        report(
          string("arm D3 recorder alone, 2^") + Global::intToString(pow2) + " rows (" +
          Global::int64ToString((int64_t)observationRecorderBytes(pow2) / 1024) + " KiB), " +
          Global::int64ToString(recorder.unrecordedObservations()) + " with no row",
          ns
        );
      }
    }
  }
}
