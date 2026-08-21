#ifndef NEURALNET_NNCACHEVERIFYHITS_H_
#define NEURALNET_NNCACHEVERIFYHITS_H_

// THE MOST OBVIOUS CORRECTNESS TEST OF A CACHE-WARMING SYSTEM, AND IT IS A DEBUG BUILD:
// does a hit served out of a PERSISTED level 0 carry the same numbers the net would compute
// for that position right now?
//
// WHAT THIS FILE IS FOR, in one sentence: everything the level-0 hit path needs in order to
// re-run the forward pass beside a hit and say, in numbers, whether the round trip through
// the file format changed the evaluation -- and nothing else, so the default build can
// contain not one byte of it.
//
// THE GATE IS COMPILE-TIME AND THE WHOLE FILE IS INSIDE IT. Everything below the #ifdef is
// absent from a build configured without -DKATAGO_NNCACHE_VERIFY_HITS=ON: no type, no
// symbol, no branch, nothing to argue about at bench time. The shape is copied from the LT-9
// census (KATAGO_LT9_CENSUS in cpp/CMakeLists.txt): a CMake option defaulting OFF, a
// target_compile_definitions inside an if(), and a .cpp that is in the source list
// unconditionally and is an EMPTY translation unit when the option is off -- so a configure
// that forgets the source list cannot happen and the file cannot rot. Operator amendment,
// ledger row 1978.
//
// WHAT IS NOT COPIED FROM THAT PRECEDENT, because it does not exist: the census's OFF-absence
// was argued structurally and never observed. This option's is OBSERVED, by symbol inspection
// of the default binary. See the arc's report.
//
// -----------------------------------------------------------------------------------------
// WHY A RECOMPUTE AND NOT A CHECKSUM
//
// The file format already checksums its blocks, and a torn or bit-rotted file is caught at
// load. That check answers "are these the bytes that were written". It does NOT answer "does
// the entry the loader built out of those bytes mean what the evaluation that was written
// meant" -- a field read at the wrong offset, a scalar order that drifted between encoder and
// decoder, a policy length computed from the wrong board dimension, an ownership flag misread:
// every one of those produces a CHECKSUM-VALID file whose deserialized evaluation is wrong.
// The only instrument that observes THAT property is the net itself (ADR-0021: the witness
// observes the property, not a symptom). So this recomputes.
//
// -----------------------------------------------------------------------------------------
// TOLERANCE, AND WHY IT IS NOT THE CATEGORY ERROR THE CONTAINER SUITES REFUSE
//
// cpp/tests/testnnevalcontainer.cpp asserts its float round trips with ==, in terms, and says
// a tolerance there would hide the bug class. That is right, and it is A DIFFERENT COMPARISON.
// That suite compares an NNOutput against ITS OWN bytes after a write and a read: the same
// value, moved. Nothing numeric happens in between, so equality is the bar and putF32/getF32
// exist to make it reachable.
//
// THIS compares a deserialized evaluation against A SECOND FORWARD PASS. That crosses a
// numerics boundary -- the recompute rides a batch of a different shape than the batch the
// original rode, and a GEMM's reduction order depends on its batch, so the two agree to a few
// ulps and not to the bit. On a FP16 backend they agree to considerably less than that.
// Pinning this bit-exactly is the category error ADR-0009/ADR-0012 P6 names: it would report
// legitimate batching as corruption and forbid the very deployment shape the cache exists for.
//
// AND THE TOLERANCE COSTS NO SENSITIVITY, which is the part that matters. The defect class
// this instrument exists to catch -- a field read at the wrong offset, a transposed scalar
// order, a policy length off by a board -- produces GROSS errors: a winrate where a lead
// should be, a policy slot holding an ownership value. Those are O(1) to O(100) deviations
// against an allowance of O(1e-4). There are three or more orders of magnitude between the
// noise being tolerated and the smallest defect being looked for, which is what makes this a
// tolerance rather than a hedge.
//
// THE MODEL IS THE CODEBASE'S OWN, NOT A NEW ONE. cpp/tests/testnn.cpp's approxEqual is this
// repository's forward-comparison precedent and it is relative-with-a-floor:
//
//     fp32:  |x-y| < 0.0001 * max(|x|, |y|, 1.0)
//     fp16:  |x-y| < 0.03   * max(|x|, |y|, 3.0)
//
// The same two numbers are used here, for the same reason, and the fp16 pair is exactly the
// GPU guidance: a CUDA FP16 bring-up sets relative=0.03 and floor=3.0 and is using this
// codebase's own measured FP16 allowance rather than a guess. Both are config keys in a verify
// build so that bring-up needs no recompile.
//
// WHY ONE RELATIVE FACTOR RATHER THAN NINE HAND-PICKED PER-CHANNEL CONSTANTS: the channels
// span five units and four orders of magnitude -- a probability in [0,1], a lead in points, a
// scoreMeanSq in points-squared running into the thousands, a varTimeLeft in turns. A relative
// allowance with a floor derives each channel's effective tolerance FROM ITS OWN MAGNITUDE,
// which is the one rule that is right for all of them; nine constants would be nine numbers to
// justify, drift and get wrong (ADR-0012 P1). The effective per-channel tolerances that fall
// out at the fp32 defaults, for the record:
//
//     probabilities (win/loss/noResult, policy slot, ownership slot)   1e-4 absolute
//     shorttermWinlossError, policyOptimismUsed                        1e-4 absolute
//     whiteScoreMean, whiteLead, shorttermScoreError  (|v| up to ~400) 1e-4 .. 4e-2
//     whiteScoreMeanSq                                (|v| up to ~2e5) 1e-4 .. 2e1
//     varTimeLeft                                     (|v| up to ~1e3) 1e-4 .. 1e-1
//
// -----------------------------------------------------------------------------------------
// SYMMETRY: THE HONEST GAP, STATED RATHER THAN PAPERED OVER
//
// A cached evaluation DOES NOT RECORD THE SYMMETRY IT WAS COMPUTED UNDER. NNOutput has no
// such field, the file format persists no such field, and NNInputs::getHash does not fold
// symmetry into the cache key -- which is exactly why NNEvaluator::averageMultipleSymmetries
// must pass skipCache. A neural net is not exactly equivariant, so two symmetries of one
// position give visibly different policies: comparing a hit computed under symmetry 5 against
// a recompute under symmetry 0 measures the net's non-equivariance, which is orders of
// magnitude larger than any deserialization defect and would drown the thing being looked for.
//
// SO VERIFICATION IS REFUSED WHERE THE SYMMETRY IS NOT PINNED, rather than performed and
// caveated. Concretely: an evaluator running with nnRandomize=true assigns each unspecified
// row a random symmetry on the server thread, so neither the cached entry's symmetry nor the
// recompute's is knowable. Under that configuration this verifier verifies NOTHING and says
// so -- it counts the hit into skippedNondeterministicSymmetry and reports that count beside
// verifiedHits, so a run that verified zero hits cannot be mistaken for a run that verified
// many and found nothing (ADR-0002).
//
// WHAT IS COVERED, THEN: an evaluator with nnRandomize=false (or nnForcedSymmetry set), where
// every unspecified evaluation takes currentDefaultSymmetry, and the recompute is issued with
// that same symmetry named EXPLICITLY rather than left unspecified. A request that named its
// own symmetry is covered too, because the recompute reuses the request's symmetry.
//
// WHAT IS STILL NOT COVERED, and cannot be without a format change: that the SESSION THAT
// WROTE the entry also ran with the same pinned symmetry. Nothing in the file says. A store
// written under nnRandomize=true and verified under nnRandomize=false will report mismatches
// that are the net's non-equivariance and not a deserialization defect. The verify build's
// own witnesses build their store with nnRandomize=false for this reason, and the mismatch
// report names the symmetry the recompute used so a reader can tell the two apart. Recording
// the symmetry per entry is the real fix; it is a format change and is filed, not smuggled in
// here.
//
// -----------------------------------------------------------------------------------------
// THE INSTRUMENT STAYS OUTSIDE THE INSTRUMENT
//
// A mismatch NEVER alters the served result. This is an observer: it counts, it logs, and the
// evaluation the caller receives is the one the cache served, mismatch or not. Two further
// consequences, both enforced rather than remembered:
//
//   THE RECOMPUTE DOES NOT STORE. NNEvaluator::evaluate ends by setting its result into the
//   table even when skipCache was passed, so a naive recompute would write its own answer over
//   the level-0 entry -- shadowing it, moving the key to level 1, and silently ending
//   verification of that key after the first hit. RecomputeScope suppresses that store for the
//   duration of the nested call.
//
//   THE RECOMPUTE IS NOT A NEW PRESENTATION. It rides the existing
//   NNCachePresentationRole::ServesACountedPresentation channel, the same one
//   averageMultipleSymmetries uses to fan one demand out over several forward passes, so the
//   count log records one observation for a verified hit and not two.
//
// The recompute DOES cost a real forward pass and DOES increment the evaluator's row counters.
// That is why a verify build never enters a timing measurement, and why the process-level
// witnesses that assert "this process did zero neural net work" run on the default build.

#ifdef KATAGO_NNCACHE_VERIFY_HITS

#include <cstdint>
#include <mutex>
#include <string>

#include "../core/hash.h"

class ConfigParser;
class Logger;
struct NNOutput;

// WHICH LEVEL ANSWERED. The whole reason this enum exists is that the level-0 case is the one
// worth a forward pass: those bytes came off disk and through a decoder, and every other kind
// of hit is a pointer to an NNOutput this process itself computed.
enum class NNCacheHitOrigin {
  // A persisted level-0 source, loaded from a container file and decoded, resolved the key.
  // THE POINT.
  LevelZeroPersisted,
  // Ordinary resident level-1 memory resolved it. Verifiable behind the same flag, opt-in,
  // because it exercises no deserialization at all -- see nnCacheVerifyHitsIncludeResident.
  LevelOneResident,
};

// THE ALLOWANCE, IN THE CODEBASE'S OWN SHAPE: |x-y| < relative * max(|x|, |y|, floor).
// See the header comment for why one relative factor and not nine per-channel constants, and
// for the fp32 and fp16 pairs this repository already uses (cpp/tests/testnn.cpp:8-14).
struct NNCacheHitVerifyTolerances {
  double relative;
  double floor;

  // fp32 / deterministic-CPU defaults: 1e-4 and 1.0, from testnn.cpp's approxEqual.
  [[nodiscard]] static NNCacheHitVerifyTolerances defaults();

  // Both keys optional, each defaulting to defaults():
  //
  //   nnCacheVerifyHitsTolRelative   (default 1e-4; set 0.03 for CUDA FP16)
  //   nnCacheVerifyHitsTolFloor      (default 1.0;  set 3.0  for CUDA FP16)
  //
  // They exist so a GPU bring-up of this instrument needs no recompile of the thing being
  // brought up. Refuses a non-positive relative or a negative floor rather than silently
  // verifying nothing or everything (ADR-0002).
  [[nodiscard]] static NNCacheHitVerifyTolerances fromConfig(ConfigParser& cfg);

  // One line, for the log the verifier writes when it arms itself. A tolerance nobody can read
  // off the run's own output is a tolerance a later reader has to go find in a header.
  [[nodiscard]] std::string describe() const;
};

// WHAT A VERIFY BUILD ADDS TO cache_stats. Present only in a verify build, by construction:
// the type that carries them does not exist otherwise.
struct NNCacheHitVerifyStats {
  // Hits that were actually compared against a fresh forward pass.
  int64_t verifiedHits;
  // Of those, how many exceeded tolerance on at least one channel.
  int64_t mismatches;
  // Level-0 hits NOT compared, because the evaluator's symmetry is not pinned. Reported
  // beside verifiedHits precisely so a zero-coverage run is legible as one.
  int64_t skippedNondeterministicSymmetry;
  // Resident (level-1) hits not compared, because verifying them is opt-in.
  int64_t skippedResidentOrigin;
  // The largest deviation/allowance ratio seen across every channel of every comparison. <= 1
  // means every comparison held. Reported as a RATIO rather than a raw deviation because the
  // channels are in five different units and a raw maximum over them means nothing. Borrowed
  // from testnnevalcanary.cpp's worstRatioVsLimits, which reports headroom rather than a bare
  // pass/fail for the same reason.
  double worstDeviationRatio;
  // Which channel that worst ratio was on, and on which key, so the worst case is addressable
  // and not just a number.
  std::string worstChannel;
  std::string worstKey;
};

// THE VERIFIER. One per NNEvaluator, owned by it, consulted only from the cache-hit branch.
//
// THREAD SAFETY: compare() is called from every search thread that takes a hit. The counters
// are guarded by a mutex rather than made atomic, because the worst-case triple
// (ratio, channel, key) must move together or the report names the wrong channel for the wrong
// key -- three independently atomic fields are three writers of one fact (ADR-0012 P1). The
// lock is taken once per verified hit, beside a full forward pass that costs milliseconds.
class NNCacheHitVerifier {
 public:
  NNCacheHitVerifier(const NNCacheHitVerifyTolerances& tolerances, bool verifyResidentOrigin, Logger* logger);
  ~NNCacheHitVerifier();

  NNCacheHitVerifier(const NNCacheHitVerifier&) = delete;
  NNCacheHitVerifier& operator=(const NNCacheHitVerifier&) = delete;

  // Should a hit of this origin be recomputed at all? False for a resident hit unless the
  // operator opted in, and the refusal is COUNTED rather than silent.
  [[nodiscard]] bool shouldVerify(NNCacheHitOrigin origin);

  // The symmetry could not be pinned, so nothing is compared. Counted, and reported.
  void countSkippedNondeterministicSymmetry();

  // THE COMPARISON. `served` is what the cache handed the caller; `recomputed` is what a fresh
  // forward pass under `symmetryUsed` produced for the same position. `policySize` is
  // nnXLen*nnYLen+1 for the board actually evaluated -- the slots beyond it are a property of
  // this build's array and not of the evaluation, which is why the file format does not
  // persist them and why comparing them would compare uninitialized memory.
  //
  // OWNERSHIP IS COMPARED WHEN BOTH SIDES HAVE IT, and its ABSENCE ON ONE SIDE ONLY is itself
  // a mismatch -- the ownermap-present flag is a persisted field, and a decoder that lost it
  // is exactly the defect class this exists for.
  //
  // Returns true iff every channel held. Never modifies either output.
  bool compare(
    Hash128 key,
    int symmetryUsed,
    const NNOutput& served,
    const NNOutput& recomputed,
    int policySize
  );

  [[nodiscard]] NNCacheHitVerifyStats stats() const;

  // IS THIS THREAD INSIDE A VERIFICATION RECOMPUTE RIGHT NOW? Consulted at exactly two places
  // in NNEvaluator::evaluate: to suppress the terminal cache store, and to refuse a nested
  // verification. Thread-local rather than a parameter because the nested call is
  // NNEvaluator::evaluate itself, and threading a parameter through its public signature would
  // put a verify-build-only argument on a function fifteen other files call.
  [[nodiscard]] static bool inRecompute();

  // RAII. Marks this thread as inside a recompute for its lifetime; nests safely.
  class RecomputeScope {
   public:
    RecomputeScope();
    ~RecomputeScope();
    RecomputeScope(const RecomputeScope&) = delete;
    RecomputeScope& operator=(const RecomputeScope&) = delete;
  };

 private:
  // ONE COMPARISON'S RUNNING WORST CASE, carried as one value because its four fields are one
  // fact: naming a channel beside a ratio and a pair of values that came from a different
  // channel is the two-writers-of-one-truth defect in miniature (ADR-0012 P1).
  struct WorstChannel {
    std::string name;
    double ratio;
    double served;
    double recomputed;
    double allowance;
  };

  // Compares one channel against the allowance and folds it into this comparison's worst case.
  // `channelName` is a static string; `slot` is -1 for a scalar channel and the index for a
  // policy or ownership slot, so the report can name the exact slot that moved.
  void checkChannel(
    const char* channelName,
    int slot,
    double servedValue,
    double recomputedValue,
    WorstChannel& worst
  ) const;

  const NNCacheHitVerifyTolerances tolerances_;
  const bool verifyResidentOrigin_;
  Logger* const logger_;

  mutable std::mutex mutex_;
  int64_t verifiedHits_;
  int64_t mismatches_;
  int64_t skippedNondeterministicSymmetry_;
  int64_t skippedResidentOrigin_;
  double worstDeviationRatio_;
  std::string worstChannel_;
  std::string worstKey_;
};

#endif  // KATAGO_NNCACHE_VERIFY_HITS

#endif  // NEURALNET_NNCACHEVERIFYHITS_H_
