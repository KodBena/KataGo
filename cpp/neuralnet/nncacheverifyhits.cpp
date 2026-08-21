#include "../neuralnet/nncacheverifyhits.h"

// EMPTY TRANSLATION UNIT IN THE DEFAULT BUILD. See the header: this file is in the source list
// unconditionally so it cannot rot, and contributes not one symbol unless the option is on.
// The same shape as cpp/game/lt9_census.cpp.
#ifdef KATAGO_NNCACHE_VERIFY_HITS

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

#include "../core/config_parser.h"
#include "../core/global.h"
#include "../core/logger.h"
#include "../neuralnet/nninputs.h"

using namespace std;

//-------------------------------------------------------------------------------------
// Tolerances
//-------------------------------------------------------------------------------------

NNCacheHitVerifyTolerances NNCacheHitVerifyTolerances::defaults() {
  // cpp/tests/testnn.cpp:8-14, the fp32 arm of approxEqual. Not a new number.
  NNCacheHitVerifyTolerances tol;
  tol.relative = 1e-4;
  tol.floor = 1.0;
  return tol;
}

NNCacheHitVerifyTolerances NNCacheHitVerifyTolerances::fromConfig(ConfigParser& cfg) {
  NNCacheHitVerifyTolerances tol = defaults();
  // Bounded at the parse, not checked after: a zero or negative relative would make every
  // comparison a mismatch, and a huge one would make the instrument report success without
  // observing anything. Both are worse than a refusal (ADR-0002).
  if(cfg.contains("nnCacheVerifyHitsTolRelative"))
    tol.relative = cfg.getDouble("nnCacheVerifyHitsTolRelative", 1e-12, 1.0);
  if(cfg.contains("nnCacheVerifyHitsTolFloor"))
    tol.floor = cfg.getDouble("nnCacheVerifyHitsTolFloor", 0.0, 1e6);
  return tol;
}

string NNCacheHitVerifyTolerances::describe() const {
  return Global::strprintf(
    "|served-recomputed| < %.3g * max(|served|,|recomputed|,%.3g)", relative, floor
  );
}

//-------------------------------------------------------------------------------------
// The recompute scope
//-------------------------------------------------------------------------------------

namespace {
  // Nesting depth rather than a bool: a future caller that wraps one recompute inside another
  // would silently clear the flag on the inner scope's exit, which is the stale-tag defect
  // NNEvaluator::evaluate already forecloses for its own two per-request tags.
  thread_local int t_recomputeDepth = 0;
}

bool NNCacheHitVerifier::inRecompute() {
  return t_recomputeDepth > 0;
}

NNCacheHitVerifier::RecomputeScope::RecomputeScope() {
  t_recomputeDepth += 1;
}

NNCacheHitVerifier::RecomputeScope::~RecomputeScope() {
  t_recomputeDepth -= 1;
}

//-------------------------------------------------------------------------------------
// The verifier
//-------------------------------------------------------------------------------------

NNCacheHitVerifier::NNCacheHitVerifier(
  const NNCacheHitVerifyTolerances& tolerances,
  bool verifyResidentOrigin,
  Logger* logger
)
  : tolerances_(tolerances),
    verifyResidentOrigin_(verifyResidentOrigin),
    logger_(logger),
    mutex_(),
    verifiedHits_(0),
    verifiedFallThroughHits_(0),
    mismatches_(0),
    skippedNondeterministicSymmetry_(0),
    skippedResidentOrigin_(0),
    skippedNoRecompute_(0),
    recomputesThatThrew_(0),
    worstDeviationRatio_(0.0),
    // "none" AND NOT THE EMPTY STRING. This is the field a reader checks first, and an empty
    // string reads as "missing" rather than as "nothing deviated" (audit note, 2026-08-21).
    worstChannel_("none"),
    worstKey_("none")
{
  // SAYS SO ON ARMING. A build that silently carries this would be indistinguishable from the
  // default build in its own log, which is exactly the confusion the compile-time gate exists
  // to prevent.
  const string line =
    "NNCACHE HIT VERIFY IS ARMED (KATAGO_NNCACHE_VERIFY_HITS build -- NOT a shipping build; "
    "every verified hit costs a full extra forward pass). Tolerance: " + tolerances_.describe() +
    "; resident (level-1) hits are " + (verifyResidentOrigin_ ? "ALSO verified" : "not verified") + ".";
  if(logger_ != NULL)
    logger_->write(line);
  else
    cerr << line << endl;
}

NNCacheHitVerifier::~NNCacheHitVerifier() {
}

bool NNCacheHitVerifier::shouldVerify(NNCacheHitOrigin origin) {
  if(origin == NNCacheHitOrigin::LevelZeroPersisted || verifyResidentOrigin_)
    return true;
  // COUNTED ONLY WHERE IT IS ACTUALLY SKIPPED. The increment used to sit above the opt-in test,
  // so with nnCacheVerifyHitsIncludeResident=true a resident hit was both verified and reported
  // as skipped -- a field whose own name and published JSON key say "not compared" carrying
  // hits that were (audit finding, 2026-08-21).
  lock_guard<std::mutex> lock(mutex_);
  skippedResidentOrigin_ += 1;
  return false;
}

void NNCacheHitVerifier::countSkippedNondeterministicSymmetry() {
  lock_guard<std::mutex> lock(mutex_);
  skippedNondeterministicSymmetry_ += 1;
}

void NNCacheHitVerifier::countSkippedNoRecompute() {
  lock_guard<std::mutex> lock(mutex_);
  skippedNoRecompute_ += 1;
}

void NNCacheHitVerifier::countRecomputeThrew() {
  lock_guard<std::mutex> lock(mutex_);
  recomputesThatThrew_ += 1;
}

void NNCacheHitVerifier::checkChannel(
  const char* channelName,
  int slot,
  double servedValue,
  double recomputedValue,
  WorstChannel& worst
) const {
  // NON-FINITE VALUES ARE DECIDED FIRST, AND NOT BY ARITHMETIC. This is the hole an audit found
  // and reproduced (2026-08-21): with served = +inf the allowance is also inf, the deviation is
  // inf, and the ratio is inf/inf = NaN -- which is FALSE against every comparison, so the fold
  // below skips it and `held` concludes true. The grossest corruption an f32 can carry was the
  // one value this instrument could not see. It is not exotic either: a mis-offset read hands
  // getF32 an arbitrary 32-bit pattern, and about one in 256 of those has exponent 0xFF.
  //
  // THE RULE: two non-finite values agree only if they are THE SAME non-finite value -- both
  // NaN, both +inf, both -inf. That much is the format faithfully round-tripping what was
  // written, which putF32/getF32 exist to do. Anything else is a mismatch, forced to infinity so
  // it cannot be argued down by a tolerance.
  const bool servedFinite = std::isfinite(servedValue);
  const bool recomputedFinite = std::isfinite(recomputedValue);
  if(!servedFinite || !recomputedFinite) {
    const bool sameNonFinite =
      (std::isnan(servedValue) && std::isnan(recomputedValue)) ||
      (servedValue == recomputedValue);  // catches +inf/+inf and -inf/-inf
    if(!sameNonFinite && std::numeric_limits<double>::infinity() > worst.ratio) {
      worst.ratio = std::numeric_limits<double>::infinity();
      worst.name = slot < 0 ? string(channelName) : (string(channelName) + "[" + Global::intToString(slot) + "]");
      worst.served = servedValue;
      worst.recomputed = recomputedValue;
      worst.allowance = 0.0;
    }
    return;
  }

  const double magnitude = std::max(std::fabs(servedValue), std::fabs(recomputedValue));
  // Positive by construction: fromConfig bounds relative strictly above zero, and a zero floor
  // still leaves the magnitude term -- but a channel that is exactly zero on both sides with a
  // zero floor would divide by zero, so the guard is not decorative.
  const double allowance = std::max(tolerances_.relative * std::max(magnitude, tolerances_.floor), 1e-300);
  const double deviation = std::fabs(servedValue - recomputedValue);
  const double ratio = deviation / allowance;
  if(ratio > worst.ratio) {
    worst.ratio = ratio;
    worst.name = slot < 0 ? string(channelName) : (string(channelName) + "[" + Global::intToString(slot) + "]");
    worst.served = servedValue;
    worst.recomputed = recomputedValue;
    worst.allowance = allowance;
  }
}

bool NNCacheHitVerifier::compare(
  Hash128 key,
  int symmetryUsed,
  const NNOutput& served,
  const NNOutput& recomputed,
  int policySize,
  bool viaOwnerMapFallThrough
) {
  WorstChannel worst;
  worst.name = "none";
  worst.ratio = 0.0;
  worst.served = 0.0;
  worst.recomputed = 0.0;
  worst.allowance = 0.0;

  checkChannel("whiteWinProb", -1, served.whiteWinProb, recomputed.whiteWinProb, worst);
  checkChannel("whiteLossProb", -1, served.whiteLossProb, recomputed.whiteLossProb, worst);
  checkChannel("whiteNoResultProb", -1, served.whiteNoResultProb, recomputed.whiteNoResultProb, worst);
  checkChannel("whiteScoreMean", -1, served.whiteScoreMean, recomputed.whiteScoreMean, worst);
  checkChannel("whiteScoreMeanSq", -1, served.whiteScoreMeanSq, recomputed.whiteScoreMeanSq, worst);
  checkChannel("whiteLead", -1, served.whiteLead, recomputed.whiteLead, worst);
  checkChannel("varTimeLeft", -1, served.varTimeLeft, recomputed.varTimeLeft, worst);
  checkChannel("shorttermWinlossError", -1, served.shorttermWinlossError, recomputed.shorttermWinlossError, worst);
  checkChannel("shorttermScoreError", -1, served.shorttermScoreError, recomputed.shorttermScoreError, worst);
  checkChannel("policyOptimismUsed", -1, served.policyOptimismUsed, recomputed.policyOptimismUsed, worst);

  // THE BOARD SHAPE IS ITSELF A PERSISTED FIELD, so a decoder that lost it is caught here and
  // not by a policy loop that would then read the wrong number of slots.
  if(served.nnXLen != recomputed.nnXLen || served.nnYLen != recomputed.nnYLen) {
    worst.name = "nnXLen/nnYLen";
    worst.ratio = std::numeric_limits<double>::infinity();
    worst.served = served.nnXLen * 1000.0 + served.nnYLen;
    worst.recomputed = recomputed.nnXLen * 1000.0 + recomputed.nnYLen;
  }
  else {
    for(int i = 0; i < policySize; i++)
      checkChannel("policyProbs", i, served.policyProbs[i], recomputed.policyProbs[i], worst);

    // THE OWNERMAP-PRESENT FLAG IS PERSISTED TOO. A disagreement about whether there IS an
    // ownership map is the same defect class as a wrong value in one, so it is a mismatch and
    // not a reason to skip the channel. The recompute always asks for a map exactly when the
    // served entry has one, so a disagreement here is the format's and nobody else's.
    const bool servedHasOwnership = served.whiteOwnerMap != NULL;
    const bool recomputedHasOwnership = recomputed.whiteOwnerMap != NULL;
    if(servedHasOwnership != recomputedHasOwnership) {
      worst.name = "whiteOwnerMap presence";
      worst.ratio = std::numeric_limits<double>::infinity();
      worst.served = servedHasOwnership ? 1.0 : 0.0;
      worst.recomputed = recomputedHasOwnership ? 1.0 : 0.0;
    }
    else if(servedHasOwnership) {
      const int ownershipSize = served.nnXLen * served.nnYLen;
      for(int i = 0; i < ownershipSize; i++)
        checkChannel("whiteOwnerMap", i, served.whiteOwnerMap[i], recomputed.whiteOwnerMap[i], worst);
    }
  }

  const bool held = worst.ratio <= 1.0;
  const string keyStr = Global::uint64ToHexString(key.hash0) + Global::uint64ToHexString(key.hash1);

  int64_t verifiedHitsSnapshot = 0;
  int64_t mismatchesSnapshot = 0;
  {
    lock_guard<std::mutex> lock(mutex_);
    verifiedHits_ += 1;
    if(viaOwnerMapFallThrough)
      verifiedFallThroughHits_ += 1;
    if(!held)
      mismatches_ += 1;
    if(worst.ratio > worstDeviationRatio_) {
      worstDeviationRatio_ = worst.ratio;
      worstChannel_ = worst.name;
      worstKey_ = keyStr;
    }
    verifiedHitsSnapshot = verifiedHits_;
    mismatchesSnapshot = mismatches_;
  }

  if(!held) {
    // LOUD AND STRUCTURED, and it does NOT change the served result -- see the header. Every
    // field a reader needs to act is on the one line: which key, which channel, both values,
    // the allowance the channel actually had, and the symmetry the recompute ran under (which
    // is the one thing that can make a mismatch a false positive, so it is stated rather than
    // left to be asked about).
    const string line = Global::strprintf(
      "NNCACHE HIT VERIFY MISMATCH: key=%s worstChannel=%s served=%.9g recomputed=%.9g "
      "deviation=%.6g allowance=%.6g ratio=%.6g (tolerance %s) recomputeSymmetry=%d. "
      "The SERVED result was returned to the caller UNCHANGED -- this is an observer, not a "
      "corrector. Running totals: verifiedHits=%lld mismatches=%lld.",
      keyStr.c_str(), worst.name.c_str(), worst.served, worst.recomputed,
      std::fabs(worst.served - worst.recomputed), worst.allowance, worst.ratio,
      tolerances_.describe().c_str(), symmetryUsed,
      (long long)verifiedHitsSnapshot, (long long)mismatchesSnapshot
    );
    if(logger_ != NULL)
      logger_->write(line);
    else
      cerr << line << endl;
  }

  return held;
}

NNCacheHitVerifyStats NNCacheHitVerifier::stats() const {
  lock_guard<std::mutex> lock(mutex_);
  NNCacheHitVerifyStats out;
  out.verifiedHits = verifiedHits_;
  out.verifiedFallThroughHits = verifiedFallThroughHits_;
  out.mismatches = mismatches_;
  out.skippedNondeterministicSymmetry = skippedNondeterministicSymmetry_;
  out.skippedResidentOrigin = skippedResidentOrigin_;
  out.skippedNoRecompute = skippedNoRecompute_;
  out.recomputesThatThrew = recomputesThatThrew_;
  out.worstDeviationRatio = worstDeviationRatio_;
  out.worstChannel = worstChannel_;
  out.worstKey = worstKey_;
  return out;
}

#endif  // KATAGO_NNCACHE_VERIFY_HITS
