#include "../neuralnet/nncachetrace.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>

#include "../core/fileutils.h"

using namespace std;

// See nncachetrace.h for why recording is a decorator, what a replay of its output does
// and does not establish, and what changed (file layout, not record format) in v3.

// A shard's own fileMutex already provides all the exclusion a write to its file needs.
// Plain fwrite pays for a second, redundant lock internal to libc on top of that --
// harmless under low contention, a real cost under the contention this class exists to
// be used under. Use the unlocked variant where one exists; fall back to plain fwrite
// (correct, just not free of the redundant lock) wherever it doesn't.
#if defined(_WIN32)
  #define KATAGO_TRACE_FWRITE _fwrite_nolock
#elif defined(__GLIBC__) || defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__)
  #define KATAGO_TRACE_FWRITE fwrite_unlocked
#else
  #define KATAGO_TRACE_FWRITE fwrite
#endif

namespace {

// Reads `file` (already positioned) as a stream of fixed-length records, calling
// unpackRecord(bytes) once per whole record. Throws if the file ends in a partial
// record -- a capture that was killed mid-record is a fact about the run, not a
// rounding error (ADR-0015 Rule 3). Shared by the v1/v2 path and every v3 shard so that
// fact is caught the same way regardless of which format produced it.
void readFixedLengthRecords(
  std::FILE* file, const string& pathForErrors, size_t recLen,
  const std::function<void(const char*)>& unpackRecord
) {
  vector<char> buf(recLen * 65536);
  size_t leftover = 0;
  while(true) {
    const size_t got = fread(buf.data()+leftover, 1, buf.size()-leftover, file);
    if(got == 0 && leftover == 0)
      break;
    const size_t have = leftover + got;
    const size_t whole = (have / recLen) * recLen;
    for(size_t off = 0; off < whole; off += recLen)
      unpackRecord(buf.data()+off);
    leftover = have - whole;
    if(leftover > 0)
      memmove(buf.data(), buf.data()+whole, leftover);
    if(got == 0)
      break;
  }
  if(leftover != 0)
    throw StringError(
      "readTrace: " + pathForErrors + " ends in a partial record (" + Global::uint64ToString((uint64_t)leftover) +
      " trailing bytes of a " + Global::uint64ToString((uint64_t)recLen) + "-byte record). The capture "
      "did not finish; treat this trace as no trace rather than as a short one."
    );
}

size_t nextPow2(size_t x) {
  size_t p = 1;
  while(p < x) p <<= 1;
  return p;
}

// How many (mutex, file) shards to use. 0 asks for a count derived from the machine's
// apparent concurrency -- 4x hardware_concurrency() gives enough shards that two
// threads landing on the same one at the same moment is rare, without opening an
// unreasonable number of files. Any request (default-derived or explicit) is rounded up
// to a power of two, for the cheap "hash & mask" shard selection in append() below, and
// clamped to stay well under typical per-process open-file-descriptor limits (usually
// 1024+; 128 leaves ample room for whatever else the process has open).
size_t chooseShardCount(size_t requested) {
  if(requested == 0) {
    size_t hc = std::thread::hardware_concurrency();
    if(hc == 0) hc = 8;  // hardware_concurrency() is allowed to return 0 if it doesn't know
    requested = hc * 4;
  }
  size_t n = nextPow2(requested);
  if(n < 1) n = 1;
  if(n > 128) n = 128;
  return n;
}

string shardPath(const string& base, size_t idx) {
  char suffix[16];
  std::snprintf(suffix, sizeof(suffix), ".s%04u", (unsigned)idx);
  return base + suffix;
}

// Appends one record per operation. v1/v2 used one (mutex, file) pair for the whole
// table; v3 uses `numShards` of them, chosen by hashing the accessed key -- see
// nncachetrace.h's v3 addendum for why key hashing (not thread id, not round robin) is
// what makes this need no coordination between shards at all.
class NNCacheTableTracing final : public NNCacheTable {
  struct Shard {
    std::vector<char> fileBuf;
    std::FILE* file = nullptr;
    std::mutex fileMutex;
    // Owns its own file: if construction throws partway through opening shard N+1, the
    // shards 0..N already open must still be closed during stack unwinding, and nothing
    // else will do that for them since the outer object never finishes constructing.
    ~Shard() { if(file) fclose(file); }
  };

  std::unique_ptr<NNCacheTable> inner;
  std::vector<std::unique_ptr<Shard>> shards;  // unique_ptr: Shard holds a mutex, so it can't be relocated by a vector resize
  size_t shardMask = 0;  // numShards - 1; numShards is always a power of two

  void append(Hash128 hash, uint32_t flags, uint32_t bytes) {
    // Same key, same shard, always -- see the class comment. This is what lets every
    // shard's file stay in exact order for that shard's own keys without any of the
    // shards ever needing to know about each other.
    Shard& shard = *shards[static_cast<size_t>(hash.hash0) & shardMask];

    // Wall-clock-ish, not an exact global sequence number: readTrace() only needs to
    // reconstruct an order good enough to tune a replacement policy by, not the literal
    // order a fully serialized capture would have produced (see nncachetrace.h). A
    // steady_clock read costs about the same as the single atomic increment an exact
    // global counter would need, but a counter would put every shard back into
    // contention with every other one on that one shared cache line, undoing the reason
    // shards exist. This also avoids the invariant/synced-TSC assumption a raw rdtsc
    // read would carry -- an acceptable trade given this is not itself the multi-hour
    // hot path, just one clock read per operation on it.
    const uint64_t nowNanos = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
      ).count()
    );

    char buf[32];
    memcpy(buf,    &nowNanos,   8);
    memcpy(buf+8,  &hash.hash0, 8);
    memcpy(buf+16, &hash.hash1, 8);
    memcpy(buf+24, &flags,      4);
    memcpy(buf+28, &bytes,      4);

    std::lock_guard<std::mutex> lock(shard.fileMutex);
    // A short write means the trace on disk no longer corresponds to what happened, and
    // every number a replay of it produces would be wrong in a way nothing downstream
    // could detect. Refuse at the point of loss (ADR-0002 rung 1).
    if(KATAGO_TRACE_FWRITE(buf,32,1,shard.file) != 1)
      throw StringError("NNCacheTableTracing: short write to a trace shard; the trace is now incomplete.");
  }

 public:
  NNCacheTableTracing(std::unique_ptr<NNCacheTable> innerArg, const string& path, size_t numShardsArg)
    :inner(std::move(innerArg))
  {
    if(inner == nullptr)
      throw StringError("NNCacheTableTracing: no inner table to trace.");

    const size_t numShards = chooseShardCount(numShardsArg);
    shardMask = numShards - 1;

    // The manifest, written before any shard file exists, so a reader can never observe
    // a manifest whose shards aren't there yet.
    {
      std::FILE* manifest = fopen(path.c_str(), "wb");
      if(manifest == NULL)
        throw StringError(string("NNCacheTableTracing: ") + NNCacheTrace::TRACE_ENV +
          " named a path that could not be opened for writing: " + path);
      const uint32_t numShards32 = (uint32_t)numShards;
      const bool ok =
        fwrite(NNCacheTrace::MAGIC_V3_MANIFEST,1,NNCacheTrace::MAGIC_LEN,manifest) == NNCacheTrace::MAGIC_LEN &&
        fwrite(&numShards32,4,1,manifest) == 1;
      fclose(manifest);
      if(!ok)
        throw StringError("NNCacheTableTracing: could not write the trace manifest to " + path);
    }

    shards.resize(numShards);
    for(size_t i = 0; i < numShards; i++) {
      shards[i].reset(new Shard());
      Shard& shard = *shards[i];
      const string sp = shardPath(path, i);
      shard.file = fopen(sp.c_str(), "wb");
      if(shard.file == NULL)
        throw StringError("NNCacheTableTracing: could not open trace shard for writing: " + sp);
      // 64KiB per shard: with up to 128 shards that's at most 8MiB total, comfortably
      // small, while still cutting how often each shard's stdio buffer fills and forces
      // a flush relative to the default (typically a few KiB).
      shard.fileBuf.resize(1 << 16);
      if(setvbuf(shard.file, shard.fileBuf.data(), _IOFBF, shard.fileBuf.size()) != 0)
        throw StringError("NNCacheTableTracing: setvbuf failed for " + sp);
    }

    // A capture run is not a timing run: this run's cache-op timings, and to a lesser
    // extent its thread interleaving, are not representative of an untraced run. Say so
    // where the operator will see it.
    fprintf(stderr,
      "NNCacheTableTracing: recording to %s across %zu shard file(s); this run's cache "
      "timings are not representative of an untraced run.\n",
      path.c_str(), numShards);
  }

  // No explicit destructor needed: Shard now closes its own file (see above), so the
  // default (which destroys `shards`, which destroys each Shard) is correct on both the
  // normal path and every exception path.
  ~NNCacheTableTracing() override = default;

  bool get(Hash128 nnHash, std::shared_ptr<NNOutput>& ret) override {
    // getRaw, not inner->get: inner's static type is NNCacheTable, the base, and the raw
    // get/set are protected -- see nncache.h's own comment on why a decorator needs the
    // static forwarder rather than plain protected access.
    const bool found = NNCacheTable::getRaw(*inner, nnHash, ret);
    append(nnHash, found ? NNCacheTrace::FLAG_HIT : 0u, 0u);
    return found;
  }

  void set(const std::shared_ptr<NNOutput>& p) override {
    // The footprint is recorded from the payload actually offered, not assumed from a
    // constant: whether this entry carries an ownership map is precisely the thing a
    // replay must not have to guess.
    append(p->nnHash, NNCacheTrace::FLAG_IS_SET, (uint32_t)nnOutputFootprintBytes(*p));
    NNCacheTable::setRaw(*inner, p);
  }

  // DELEGATED AND DELIBERATELY NOT RECORDED. See nncachetrace.h / ADR-0009: a
  // containment probe stores, retrieves, and moves nothing in any policy's state, so
  // recording it would put a session-boundary act into the stream as though it were a
  // client's lookup.
  bool contains(Hash128 nnHash) const override { return inner->contains(nnHash); }

  void clear() override { inner->clear(); }
  NNCacheStats stats() const override { return inner->stats(); }
};

}  // namespace

unique_ptr<NNCacheTable> NNCacheTrace::wrapWithTrace(unique_ptr<NNCacheTable> inner, const string& path, size_t numShards) {
  return unique_ptr<NNCacheTable>(new NNCacheTableTracing(std::move(inner), path, numShards));
}

vector<NNCacheTraceRecord> NNCacheTrace::readTrace(
  const string& path, uint32_t assumedBytesIfV1, bool& wasV1, bool& orderIsApproximate
) {
  std::FILE* file = fopen(path.c_str(),"rb");
  if(file == NULL)
    throw StringError("readTrace: could not open trace file for reading: " + path);

  char magic[NNCacheTrace::MAGIC_LEN];
  const size_t magicRead = fread(magic,1,NNCacheTrace::MAGIC_LEN,file);
  const bool isV2 = magicRead == NNCacheTrace::MAGIC_LEN &&
    memcmp(magic,NNCacheTrace::MAGIC_V2,NNCacheTrace::MAGIC_LEN) == 0;
  const bool isV3Manifest = magicRead == NNCacheTrace::MAGIC_LEN &&
    memcmp(magic,NNCacheTrace::MAGIC_V3_MANIFEST,NNCacheTrace::MAGIC_LEN) == 0;

  if(isV3Manifest) {
    uint32_t numShards = 0;
    const bool readCount = fread(&numShards,4,1,file) == 1;
    fclose(file);
    if(!readCount || numShards == 0)
      throw StringError("readTrace: " + path + " is a v3 manifest but its shard count could not be read.");

    wasV1 = false;
    orderIsApproximate = numShards > 1;

    struct RecWithTime { uint64_t timestamp; NNCacheTraceRecord rec; };
    vector<RecWithTime> all;
    for(uint32_t s = 0; s < numShards; s++) {
      const string sp = shardPath(path, s);
      std::FILE* sf = fopen(sp.c_str(),"rb");
      if(sf == NULL)
        throw StringError("readTrace: " + path + " names " + Global::uint64ToString((uint64_t)numShards) +
          " shard(s) but " + sp + " could not be opened.");
      readFixedLengthRecords(sf, sp, NNCacheTrace::RECORD_LEN_V3, [&](const char* p){
        RecWithTime rwt;
        memcpy(&rwt.timestamp, p,    8);
        memcpy(&rwt.rec.hash0, p+8,  8);
        memcpy(&rwt.rec.hash1, p+16, 8);
        memcpy(&rwt.rec.flags, p+24, 4);
        memcpy(&rwt.rec.bytes, p+28, 4);
        all.push_back(rwt);
      });
      fclose(sf);
    }
    // Stable: two records can genuinely tie in timestamp (clock resolution is finite),
    // and ties then keep whatever order the shard loop above read them in -- arbitrary
    // across shards, but deterministic and reproducible for a given set of trace files.
    std::stable_sort(all.begin(), all.end(), [](const RecWithTime& a, const RecWithTime& b){
      return a.timestamp < b.timestamp;
    });
    vector<NNCacheTraceRecord> out;
    out.reserve(all.size());
    for(auto& rwt : all) out.push_back(rwt.rec);
    if(out.empty())
      throw StringError("readTrace: " + path + " contains no records across " +
        Global::uint64ToString((uint64_t)numShards) + " shard(s).");
    return out;
  }

  // --- v1 / v2: one file, one writer, already in exact order. ---
  wasV1 = !isV2;
  orderIsApproximate = false;
  if(!isV2) {
    // No header, so those bytes were the first record's first bytes. Rewind and read
    // the whole file as 20-byte records.
    if(fseek(file,0,SEEK_SET) != 0) {
      fclose(file);
      throw StringError("readTrace: could not rewind " + path);
    }
  }
  const size_t recLen = isV2 ? NNCacheTrace::RECORD_LEN : NNCacheTrace::RECORD_LEN_V1;
  vector<NNCacheTraceRecord> out;
  readFixedLengthRecords(file, path, recLen, [&](const char* p){
    NNCacheTraceRecord rec;
    memcpy(&rec.hash0, p,    8);
    memcpy(&rec.hash1, p+8,  8);
    memcpy(&rec.flags, p+16, 4);
    if(isV2)
      memcpy(&rec.bytes, p+20, 4);
    else
      rec.bytes = (rec.flags & NNCacheTrace::FLAG_IS_SET) != 0 ? assumedBytesIfV1 : 0u;
    out.push_back(rec);
  });
  fclose(file);

  if(out.empty())
    throw StringError("readTrace: " + path + " contains no records.");
  return out;
}

vector<NNCacheTraceRecord> NNCacheTrace::readTrace(
  const string& path, uint32_t assumedBytesIfV1, bool& wasV1
) {
  bool orderIsApproximate;
  return readTrace(path, assumedBytesIfV1, wasV1, orderIsApproximate);
}

