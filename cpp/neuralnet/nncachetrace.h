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

// One appended record per cache operation. Little-endian, 24 bytes, fixed forever.
//
// This is version 2 of a format whose version 1 -- the same fields without `bytes` --
// was written by an earlier branch of this investigation and whose files still exist. A
// v1 file carries no footprint, so a replay of one must be told what to assume an entry
// costs; a v2 file carries the real measured footprint of every stored payload, which is
// what makes "resident bytes actually used" a measurement rather than a multiplication.
// The two are told apart by the magic header below, which v1 files do not have.
struct NNCacheTraceRecord {
  uint64_t hash0;
  uint64_t hash1;
  uint32_t flags;
  uint32_t bytes;  // nnOutputFootprintBytes of the payload, on a set; 0 on a get
};

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

  // The environment variable that turns recording on, checked once per table
  // construction and never again. Named here so the command, the docs and the
  // implementation cannot drift (ADR-0012 P1).
  static const char* const TRACE_ENV = "KATAGO_NNCACHE_TRACE";

  // Wraps `inner` so that every get and set is appended to `path`. Throws if the file
  // cannot be opened -- a tracing run that silently produced no trace would be the
  // worst possible outcome (ADR-0002).
  std::unique_ptr<NNCacheTable> wrapWithTrace(std::unique_ptr<NNCacheTable> inner, const std::string& path);

  // Reads a whole trace into memory, accepting both versions. `assumedBytesIfV1` is what
  // a v1 record's missing footprint is taken to be; it is required rather than defaulted,
  // because a silent default here would turn an assumption into a number nobody could
  // find later. `wasV1` reports which format was read, so a caller can stamp its results
  // with the fact that its byte figures are assumed rather than measured.
  std::vector<NNCacheTraceRecord> readTrace(
    const std::string& path, uint32_t assumedBytesIfV1, bool& wasV1
  );
}

#endif  // NEURALNET_NNCACHETRACE_H_
