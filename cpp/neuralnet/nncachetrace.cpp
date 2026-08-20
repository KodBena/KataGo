#include "../neuralnet/nncachetrace.h"

#include <cstdio>
#include <cstring>
#include <mutex>

#include "../core/fileutils.h"

using namespace std;

// See nncachetrace.h for why recording is a decorator and what a replay of its output
// does and does not establish.

namespace {

// Appends one record per operation, under its own mutex.
//
// The mutex is the honest cost and it is why this must never wrap a table during a
// TIMED run: it serialises every cache operation through one lock, which is exactly the
// contention the mutex pool exists to avoid. A capture run is not a timing run, and the
// loud line this prints on construction says so where the operator will see it.
class NNCacheTableTracing final : public NNCacheTable {
  std::unique_ptr<NNCacheTable> inner;
  std::FILE* file;
  std::mutex fileMutex;

  void append(Hash128 hash, uint32_t flags, uint32_t bytes) {
    NNCacheTraceRecord rec;
    rec.hash0 = hash.hash0;
    rec.hash1 = hash.hash1;
    rec.flags = flags;
    rec.bytes = bytes;
    std::lock_guard<std::mutex> lock(fileMutex);
    // A short write means the trace on disk no longer corresponds to what happened, and
    // every number a replay of it produces would be wrong in a way nothing downstream
    // could detect. Refuse at the point of loss (ADR-0002 rung 1).
    if(fwrite(&rec.hash0,8,1,file) != 1 ||
       fwrite(&rec.hash1,8,1,file) != 1 ||
       fwrite(&rec.flags,4,1,file) != 1 ||
       fwrite(&rec.bytes,4,1,file) != 1)
      throw StringError("NNCacheTableTracing: short write to the trace file; the trace is now incomplete.");
  }

 public:
  NNCacheTableTracing(std::unique_ptr<NNCacheTable> innerArg, const string& path)
    :inner(std::move(innerArg)), file(NULL)
  {
    if(inner == nullptr)
      throw StringError("NNCacheTableTracing: no inner table to trace.");
    file = fopen(path.c_str(),"wb");
    if(file == NULL)
      throw StringError(
        string("NNCacheTableTracing: ") + NNCacheTrace::TRACE_ENV +
        " named a path that could not be opened for writing: " + path
      );
    if(fwrite(NNCacheTrace::MAGIC_V2,1,NNCacheTrace::MAGIC_LEN,file) != NNCacheTrace::MAGIC_LEN)
      throw StringError("NNCacheTableTracing: could not write the trace header to " + path);
  }

  ~NNCacheTableTracing() override {
    if(file != NULL)
      fclose(file);
  }

  bool get(Hash128 nnHash, std::shared_ptr<NNOutput>& ret) override {
    const bool found = inner->get(nnHash,ret);
    append(nnHash, found ? NNCacheTrace::FLAG_HIT : 0u, 0u);
    return found;
  }

  void set(const std::shared_ptr<NNOutput>& p) override {
    // The footprint is recorded from the payload actually offered, not assumed from a
    // constant: whether this entry carries an ownership map is precisely the thing a
    // replay must not have to guess.
    append(p->nnHash, NNCacheTrace::FLAG_IS_SET, (uint32_t)nnOutputFootprintBytes(*p));
    inner->set(p);
  }

  // DELEGATED AND DELIBERATELY NOT RECORDED. The trace exists so a replay can re-run this run's
  // cache operations under a different policy, and a containment probe is not one of them: it
  // stores nothing, retrieves nothing, and moves nothing in any policy's state, so a replay's
  // numbers are identical whether it happened or not. Recording it would instead put a
  // session-boundary act into the stream as though it were a client's lookup, which is the one
  // thing a trace must not do (ADR-0009).
  bool contains(Hash128 nnHash) const override { return inner->contains(nnHash); }

  void clear() override { inner->clear(); }
  NNCacheStats stats() const override { return inner->stats(); }
};

}  // namespace

unique_ptr<NNCacheTable> NNCacheTrace::wrapWithTrace(unique_ptr<NNCacheTable> inner, const string& path) {
  return unique_ptr<NNCacheTable>(new NNCacheTableTracing(std::move(inner), path));
}

vector<NNCacheTraceRecord> NNCacheTrace::readTrace(
  const string& path, uint32_t assumedBytesIfV1, bool& wasV1
) {
  std::FILE* file = fopen(path.c_str(),"rb");
  if(file == NULL)
    throw StringError("readTrace: could not open trace file for reading: " + path);

  char magic[NNCacheTrace::MAGIC_LEN];
  const size_t magicRead = fread(magic,1,NNCacheTrace::MAGIC_LEN,file);
  const bool isV2 = magicRead == NNCacheTrace::MAGIC_LEN &&
    memcmp(magic,NNCacheTrace::MAGIC_V2,NNCacheTrace::MAGIC_LEN) == 0;
  wasV1 = !isV2;
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
  vector<char> buf(recLen * 65536);
  size_t leftover = 0;
  while(true) {
    const size_t got = fread(buf.data() + leftover, 1, buf.size() - leftover, file);
    if(got == 0 && leftover == 0)
      break;
    const size_t have = leftover + got;
    const size_t whole = (have / recLen) * recLen;
    for(size_t off = 0; off < whole; off += recLen) {
      NNCacheTraceRecord rec;
      memcpy(&rec.hash0, buf.data()+off, 8);
      memcpy(&rec.hash1, buf.data()+off+8, 8);
      memcpy(&rec.flags, buf.data()+off+16, 4);
      if(isV2)
        memcpy(&rec.bytes, buf.data()+off+20, 4);
      else
        rec.bytes = (rec.flags & NNCacheTrace::FLAG_IS_SET) != 0 ? assumedBytesIfV1 : 0u;
      out.push_back(rec);
    }
    leftover = have - whole;
    if(leftover > 0)
      memmove(buf.data(), buf.data()+whole, leftover);
    if(got == 0)
      break;
  }
  fclose(file);

  // A trailing partial record means the file was truncated -- a capture that was killed,
  // most likely by the substrate. That is a fact about the run, not a rounding error, so
  // it is refused rather than silently dropped (ADR-0015 Rule 3: a substrate verdict is
  // never read as ordinary noise).
  if(leftover != 0)
    throw StringError(
      "readTrace: " + path + " ends in a partial record (" + Global::uint64ToString((uint64_t)leftover) +
      " trailing bytes of a " + Global::uint64ToString((uint64_t)recLen) + "-byte record). The capture "
      "did not finish; treat this trace as no trace rather than as a short one."
    );
  if(out.empty())
    throw StringError("readTrace: " + path + " contains no records.");
  return out;
}
