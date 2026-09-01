#ifndef NEURALNET_NNCACHETRACE_H_
#define NEURALNET_NNCACHETRACE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "../neuralnet/nncache.h"

// Recording the cache's operation stream, so that a policy sweep replays one real
// workload instead of guessing at a synthetic one.
//
// WHY A TRACE AND NOT A LIVE SWEEP. The quantities that decide a capacity-management
// question -- hit rate, resident bytes, occupancy -- depend on WHICH KEYS ARRIVE AND IN
// WHAT ORDER, not on how fast the machine ran. Running the real engine once per policy
// configuration would re-pay the neural-net cost, which is three to four orders of
// magnitude larger than the cache path, for a quantity the neural net does not affect.
// Recording the stream once and replaying it against every configuration measures every
// configuration against THE SAME workload, which a per-configuration live run could not
// promise even in principle.
//
// THE ASSUMPTION THAT BUYS THIS, STATED SO IT CAN BE REFUSED. Replay assumes the
// operation stream is the same whichever cache policy is in force. It is exact for a
// single-threaded, fixed-visit search: the cache returns the identical NNOutput a miss
// would have computed -- the symmetry choice is inside the hash -- so a hit changes how
// long a visit takes and nothing about which position is visited next. It is NOT exact
// for a multi-threaded search, where thread interleaving already makes the stream
// nondeterministic run to run. So a replayed hit rate is a claim about the recorded
// workload, and its transfer to a live multi-threaded run is an inference, not a
// measurement.
//
// WHERE THIS SITS. It is a DECORATOR over any NNCacheTable, exactly like the admission
// filter, and it is constructed only when an environment variable asked for it. Nothing
// on the shipped get/set path tests for it, because when it is not asked for it is not
// there -- the alternative, a branch inside each of the four table implementations, would
// put the instrument inside the instrument on the one path this whole programme is
// measuring (ADR-0009).
//
// v3 ADDENDUM -- LONG-RUN CAPTURE. v1 and v2 both serialize every operation through one
// mutex and one file: exact global order, at the cost of turning every get()/set() into
// a wait for whichever thread got there first. That cost was accepted because capture
// runs were short. A capture meant to run for hours to days at close to untraced speed
// can't accept it -- not just for throughput, but because a heavily contended lock can
// itself perturb thread scheduling enough to change which thread reaches which position
// when, which is precisely the interleaving this file exists to record faithfully.
//
// v3 answers this by splitting the single (mutex, file) pair into `numShards`
// independent ones, chosen by hashing the accessed key -- not by thread id, and not by
// round robin. Hashing the key means the same key always lands in the same shard, from
// every thread, forever: a shard's own file is therefore still in EXACT order for
// repeated accesses to any one key, and no coordination between shards is ever needed,
// because the one thing that would require it -- the same key observed "at once" via two
// different shards -- cannot happen. What is given up is exact order ACROSS different
// keys: two records in different shards are ordered only by a timestamp attached at
// write time and reconstructed by readTrace() on read, which is an approximation of the
// true interleaving rather than a record of it. See readTrace's `orderIsApproximate`.
struct NNCacheTraceRecord {
  uint64_t hash0;
  uint64_t hash1;
  uint32_t flags;
  uint32_t bytes;  // nnOutputFootprintBytes of the payload, on a set; 0 on a get
};
// The v2/v3 writer serializes this with one write of the whole struct (see
// nncachetrace.cpp), which is only correct if the compiler has inserted no padding.
// True on every ABI this project targets given the field order above, but asserted
// rather than assumed.
static_assert(sizeof(NNCacheTraceRecord) == 24, "NNCacheTraceRecord must stay 24 bytes with no padding");

namespace NNCacheTrace {
  // bit 0: this op was a set (clear = a get). bit 1: on a get, this get was a hit.
  static const uint32_t FLAG_IS_SET = 1u;
  static const uint32_t FLAG_HIT = 2u;

  // 16 bytes, written once at the head of a v2 file. A v1 file starts with a hash, so
  // the odds of a v1 file opening with these exact bytes are 2^-128.
  static const char* const MAGIC_V2 = "KATAGONNCACHET2\n";
  static const size_t MAGIC_LEN = 16;
  static const size_t RECORD_LEN = 24;
  static const size_t RECORD_LEN_V1 = 20;

  // A v3 trace is not a new record format so much as a new FILE LAYOUT: `path` becomes
  // a small manifest (this magic + a uint32 shard count) and the records themselves live
  // in sibling files path.s0000, path.s0001, .... Each shard record is
  // [timestampNanos:8][hash0:8][hash1:8][flags:4][bytes:4] = 32 bytes, with no per-shard
  // magic of its own -- the manifest is the only thing that says these files are v3 and
  // how many of them there are, so a shard file is only meaningful next to its manifest.
  static const char* const MAGIC_V3_MANIFEST = "KATAGONNCACHET3M";
  static const size_t RECORD_LEN_V3 = 32;

  // The environment variable that turns recording on, checked once per table
  // construction and never again. Named here so the command, the docs and the
  // implementation cannot drift (ADR-0012 P1).
  static const char* const TRACE_ENV = "KATAGO_NNCACHE_TRACE";

  // Wraps `inner` so that every get and set is appended to `path`. Throws if the file
  // cannot be opened -- a tracing run that silently produced no trace would be the
  // worst possible outcome (ADR-0002).
  //
  // numShards: how many independent (lock, file) pairs to spread capture across. 0 (the
  // default) asks for a count derived from the machine's apparent concurrency; any
  // other value is rounded up to a power of two and clamped to a range that stays well
  // under typical per-process open-file-descriptor limits. Pass 1 to recover the old
  // single-writer, exact-global-order behavior exactly -- worth doing for a short
  // debug capture where that exactness matters more than long-run throughput. The
  // actual count used is always printed to stderr on construction.
  std::unique_ptr<NNCacheTable> wrapWithTrace(
    std::unique_ptr<NNCacheTable> inner, const std::string& path, size_t numShards = 0
  );

  // Reads a whole trace into memory, accepting v1, v2, and v3 (sharded) files
  // transparently based on `path`'s own header. The caller does not need to know which
  // version produced it.
  //
  // `assumedBytesIfV1` is what a v1 record's missing footprint is taken to be; it is
  // required rather than defaulted, because a silent default here would turn an
  // assumption into a number nobody could find later. `wasV1` reports which format was
  // read, so a caller can stamp its results with the fact that its byte figures are
  // assumed rather than measured.
  //
  // `orderIsApproximate` reports whether the returned order is exact (v1, v2, or a v3
  // file captured with numShards==1) or reconstructed by merging independently-ordered
  // shards by timestamp (any v3 file with numShards>1) -- the same reasoning as wasV1:
  // a caller's results should be stamped with which kind of order they rest on, not
  // have that distinction silently disappear.
  std::vector<NNCacheTraceRecord> readTrace(
    const std::string& path, uint32_t assumedBytesIfV1, bool& wasV1, bool& orderIsApproximate
  );
  // Convenience overload for existing call sites: identical to the above with
  // orderIsApproximate discarded. No existing caller needs to change.
  std::vector<NNCacheTraceRecord> readTrace(
    const std::string& path, uint32_t assumedBytesIfV1, bool& wasV1
  );
}

#endif  // NEURALNET_NNCACHETRACE_H_
