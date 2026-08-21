#ifndef NEURALNET_NNCACHEFILEFORMAT_H_
#define NEURALNET_NNCACHEFILEFORMAT_H_

#include <cstdint>
#include <cstdio>
#include <cstddef>
#include <string>
#include <vector>

// The three facts that every on-disk cache file in this family shares, in one home.
//
// The family is the per-context count log (<context>.nncounts, nncachecountlog.h) and the
// per-(context, model) evaluation container (<context>.<model>.nnevals,
// nnevalcontainer.h). They are two formats, not one, and neither reads the other's bytes.
// But three things about them are ONE fact rather than two, and each is here for a reason
// that is not tidiness:
//
//   THE LITTLE-ENDIAN PACKING. Both formats are a fact about bytes rather than about a
//   compiler's struct layout, and both say so. Two byte-by-byte packers would be two
//   authors of one convention (ADR-0012 P1/P7).
//
//   THE CHECKSUM. Same reason, plus: the checksum is what the torn-tail contract of both
//   formats rests on, and a divergence between two hand-copied implementations of it would
//   surface as a file that one build wrote and another rejected.
//
//   THE PATH-COMPONENT NAME. A context name is a component of BOTH files' paths, and an
//   attach joins the two files by context. The context hash the two headers carry must be
//   the SAME function of the SAME validated name, or the join is between two different
//   notions of "card-5455". This one is not merely economical to share -- sharing it is the
//   contract.
//
//   THE DURABILITY PRIMITIVES. Both formats' crash story is append-plus-fsync, an atomic
//   rename for rewrites, and a TRUNCATION back to the end of the last intact block for
//   torn-tail repair; and both need a file handle that survives a throw between opening
//   and closing. One RAII handle, one directory fsync and one truncation, not two.
//
//   THE PAGE-CACHE POSTURE. Both formats are streamed front to back and neither is ever
//   re-read: the deployment's corpus is ~100 GB against a machine that cannot cache a
//   fraction of it, so every byte either format reads is a byte of somebody else's working
//   set evicted unless the kernel is told otherwise. The advice is one fact about how this
//   file family is used, not two, and BOTH formats' scans take it -- which is stated here
//   because for a while only one of them did, on the reasoning that a count log is small.
//   That reasoning does not survive its own arithmetic: a card holding 10-20 GB of
//   evaluations has millions of keys, and at 24 bytes a record its count log is hundreds of
//   megabytes, not the few this header once assumed.
//
// Nothing here knows what either format's records mean. It is the byte layer under both.

//-------------------------------------------------------------------------------------
// Little-endian packing
//-------------------------------------------------------------------------------------

namespace NNCacheFileBytes {
  void put32(uint8_t* p, uint32_t v);
  void put64(uint8_t* p, uint64_t v);
  uint32_t get32(const uint8_t* p);
  uint64_t get64(const uint8_t* p);

  // The IEEE-754 bit pattern, moved verbatim. A float is stored as its bits rather than
  // re-derived through any arithmetic, so a NaN, an infinity and a negative zero all come
  // back as themselves rather than as something plausible.
  void putF32(uint8_t* p, float v);
  float getF32(const uint8_t* p);
}

//-------------------------------------------------------------------------------------
// The checksum
//-------------------------------------------------------------------------------------

// FNV-1a over the bytes, then an avalanche finalizer.
//
// It is a CORRUPTION DETECTOR, not a MAC: it defends against a crash, never against an
// adversary. Two properties are what the torn-tail contract needs, and both are why it is
// FNV rather than a sum or an XOR. A run of zero bytes -- the shape a page that never
// reached the device leaves behind -- still advances the state, because the multiply runs
// whatever the byte was. And the finalizer means a single changed bit anywhere in the input
// changes about half the output bits, so a block that lost its last few bytes to a partial
// flush does not checksum to the same value as the whole one.
//
// It is an ACCUMULATOR and not only a one-shot because the evaluation container's blocks are
// tens of megabytes: a block is checksummed by streaming it through a bounded buffer, so
// that verifying a block never requires holding the block.
class NNCacheFileChecksum {
 public:
  explicit NNCacheFileChecksum(uint64_t seed);

  void update(const uint8_t* data, size_t len);
  // The finalizer, applied to a copy of the state, so a checksum can be read without ending
  // the accumulation.
  [[nodiscard]] uint64_t finish() const;

  // The one-shot form, for a buffer already in hand.
  [[nodiscard]] static uint64_t of(const uint8_t* data, size_t len, uint64_t seed);

 private:
  uint64_t state_;
};

//-------------------------------------------------------------------------------------
// The path-component name boundary
//-------------------------------------------------------------------------------------

namespace NNCacheFileName {
  // ASCII letters, digits, '.', '_' and '-', nonempty, at most this many characters, and
  // neither "." nor "..".
  size_t maxLength();

  // Throws StringError, naming `whoIsAsking` and `whatItIs` (e.g. "context name",
  // "model name"), if `name` is not in that alphabet.
  //
  // A name here becomes a component of a path. A path expression is an interpreter and
  // there is no typed value-carrier to hand a component to, so the sanctioned move is a
  // strict validation to a closed alphabet that REFUSES what it cannot honor -- never an
  // escape, never a rewrite into something acceptable (ADR-0012, the 2026-07-18
  // interpreter-boundary amendment). Without it a context of ".." writes outside the
  // directory the caller named.
  void verify(const std::string& name, const char* whoIsAsking, const char* whatItIs);

  // The 64-bit identity a file header stores for the name it was written for. Both formats
  // store it under the same function, because an attach joins a context's two files by it.
  uint64_t hashOf(const std::string& name);
}

//-------------------------------------------------------------------------------------
// The durability primitives
//-------------------------------------------------------------------------------------

// Owns a FILE* for its scope. The whole reason it exists is that the format code throws in
// a dozen places between opening and closing.
//
// WHY C STDIO AND NOT std::ofstream, since ADR-0012 P9 asks for a measured reason or a real
// named constraint rather than a habit. The constraint is real and is named: every
// durability claim these formats make rests on fsync, fsync needs the underlying file
// descriptor, and std::ofstream exposes no portable way to obtain one. std::filebuf's
// descriptor is an implementation extension where it exists at all. So the write path uses
// FILE*, held here so a throw cannot leak it, and the read path uses FILE* too rather than
// mixing two I/O stacks over one format.
class NNCacheFileHandle {
 public:
  NNCacheFileHandle(const std::string& path, const char* mode);
  ~NNCacheFileHandle();
  NNCacheFileHandle(const NNCacheFileHandle&) = delete;
  NNCacheFileHandle& operator=(const NNCacheFileHandle&) = delete;

  bool isOpen() const { return file_ != nullptr; }
  std::FILE* get() const { return file_; }

  // THE READ POSTURE FOR A FILE THAT IS STREAMED ONCE AND NEVER RE-READ. Call this
  // immediately after opening for reading and before the first read; it does two things
  // that are one decision:
  //
  //   A LARGE STDIO BUFFER, because the default one is the filesystem's block size --
  //   4096 bytes on the deployment's filesystem -- and that is the granularity every read
  //   and every seek inside this file then takes. MEASURED, on a 44.5 MB three-block
  //   container: 12,111 read syscalls, of which 11,426 were 4096 bytes. The bytes fetched
  //   from the device were already exactly the file's size -- the kernel's readahead sees
  //   to that -- so what the small buffer costs is syscalls, and at the deployment's
  //   10-20 GB per card that is millions of them per load.
  //
  //   POSIX_FADV_SEQUENTIAL, which says the same thing to the kernel that the buffer says
  //   to stdio.
  //
  // Returns false if the buffer could not be installed, so a caller that cares can say so;
  // the advice itself is best-effort and is a no-op where the platform has none.
  bool useSequentialStreamBuffer();

  // Tells the kernel that `bytes` bytes from `offset` are done with and need not be kept in
  // page cache. This is DROP-BEHIND, and it is the only thing standing between a container
  // load and the eviction of the machine's working set: the corpus is ~100 GB against a page
  // cache that holds a fraction of one card, so a load that leaves its bytes resident buys
  // nothing (they are never re-read) and costs everything else that was cached.
  //
  // Best-effort and silent about the KERNEL's answer: a kernel that declines, or a platform
  // with no such call, leaves the pages resident, which is a performance fact and never a
  // correctness one.
  //
  // LOUD about the CALLER, though, and that is why the handle records its open mode. This is
  // deliberately not fsync-coupled: it is only ever correct on a handle opened for reading,
  // where there is nothing dirty to lose. On a write handle, DONTNEED against not-yet-flushed
  // pages would be a request to discard data this process has not finished writing, so a
  // write handle is REFUSED by name rather than served (ADR-0002). The precondition used to
  // be a sentence in this comment that a future caller could break in silence; now it is a
  // fact the type carries.
  void dropCachedRange(int64_t offset, int64_t bytes);

  // Flushes the stdio buffer and then forces the bytes to the device. Returns false, rather
  // than throwing, so the caller names the file in its own message.
  bool flushAndSync();

  // Closes early, so a rename can follow on platforms that will not rename an open file.
  // Idempotent.
  bool closeNow();

  // The stdio buffer useSequentialStreamBuffer installs, in bytes.
  static size_t sequentialStreamBufferBytes();

 private:
  std::FILE* file_;
  // Whether the mode this was opened with can write. Read from the mode string at
  // construction and never re-derived, because it is what dropCachedRange refuses on.
  bool canWrite_;
  // The path, for a refusal that has to name the file it is about.
  std::string path_;
  // The storage setvbuf was handed. It is owned HERE because setvbuf does not copy: the
  // buffer must outlive every read through the stream, and the only object whose lifetime
  // is already exactly that is this handle.
  std::vector<char> buffer_;
};

namespace NNCacheFileSync {
  // Forces a rename to be durable, not merely the contents of the two files it renamed.
  //
  // Without this a crash can leave the old name pointing at the old inode even though both
  // files' data are on the device, because the directory entry itself was still in cache.
  // There is no portable Windows equivalent -- a directory is not openable as a file there
  // -- so on Windows the rename's durability is whatever the filesystem gives, which is
  // stated here rather than assumed away.
  void directoryOf(const std::string& path);
}

// TORN-TAIL REPAIR: shortening a file back to the end of its last intact block.
//
// WHAT THIS REPLACES AND WHY IT IS NOT A MICRO-OPTIMISATION. Both formats used to repair a
// torn tail by rewriting the whole file through a temp sibling and an atomic rename -- the
// same machinery compaction uses. That is correct and it is ruinously expensive here,
// because of the lifecycle rather than the format: engines are long-lived and arbitrarily
// SIGKILLed, interval dumps run every fifteen minutes, and a per-card container reaches
// 10-20 GB. Every kill that lands inside a dump therefore cost a full rewrite of the card at
// the next dump -- tens of gigabytes written to flash to discard a few kilobytes that were
// going to be discarded either way, since the rewrite path already threw the torn bytes
// away. Truncation discards exactly the same bytes and writes none.
//
// THE SURVIVING DATA IS BYTE-FOR-BYTE THE SAME, and that is what makes the substitution
// legitimate rather than merely cheaper: the rewrite re-encoded the live set, which is a
// SUBSET of the intact prefix (last-wins merging drops superseded entries); truncation keeps
// the intact prefix itself. Neither keeps a byte of the torn tail. What truncation does NOT
// do is compact -- the file keeps its superseded entries -- and that is correct, because
// compaction is a separate decision with its own trigger, and a repair that silently
// compacted would make every kill a compaction.
//
// THE CRASH STORY. ftruncate is a metadata operation the filesystem journals; a crash during
// it leaves the file at its old length or its new one, and the old length is simply the torn
// tail again, which the next append repairs identically. The fsync afterwards is what makes
// the new length durable rather than merely visible, so a crash right after the repair
// cannot resurrect the tail and strand the block appended past it.
namespace NNCacheFileTruncate {
  // Shortens `path` to exactly `bytes` and forces the new length to the device. Throws
  // StringError naming the file, the length and the errno if it cannot -- a repair that
  // silently did not happen would let the next append land past a torn tail, which is the
  // one outcome the repair exists to prevent (ADR-0002).
  //
  // Refuses a negative length, and refuses to GROW a file: this is a repair primitive, and
  // ftruncate's zero-filling extension has no reading in either format.
  void toLength(const std::string& path, int64_t bytes);
}

// WHAT A WRITE PATH DID ABOUT THE TAIL IT WAS HANDED. Every append in this file family
// scans before it writes, and this is what that scan's verdict authorised.
//
// It is a TYPE and not a bool because the bool it replaces became a lie. The old result
// carried `rewroteTheFile`, which meant "the repair rewrote the whole file"; once the repair
// became a truncation, no writer could ever set it and its documentation promised a fact the
// code could no longer produce. An operator reading a dump report was left to infer that a
// repair happened at all from a byte count being positive beside a flag that was always
// false. The disposition says it outright, and the byte count says how much.
enum class NNCacheFileTailRepair {
  // The tail was intact. Nothing was discarded and nothing was written but the append.
  NotNeeded,
  // The file was shortened back to the end of its last intact block.
  Truncated,
};

// WHAT A MAINTENANCE PASS DID TO A FILE -- the three-valued answer compactIfNeeded owes its
// caller, in place of a bool that could not tell two of them apart.
//
// The bool said "did it compact". After the repair became a truncation that was wrong for
// one of its two true cases: a torn tail under the size trigger now shortens the file and
// rewrites nothing, and reporting that to an operator as "compacted" describes a full
// rewrite of a 10-20 GB card that did not happen. The three cases are genuinely three, so
// they are three values (ADR-0000).
enum class NNCacheFileMaintenance {
  // The size trigger did not fire and the tail was intact. The file was not touched.
  Nothing,
  // The tail was torn and the size trigger did not fire: shortened, not rewritten.
  TruncatedTornTail,
  // The size trigger fired: the whole file rewritten as its header plus one block. This
  // subsumes a repair, since it writes a fresh file from the intact part.
  Compacted,
};

namespace NNCacheFileReport {
  // The disposition in one lowercase word, for a report or a JSON field: "notNeeded",
  // "truncated"; "nothing", "truncatedTornTail", "compacted".
  const char* nameOf(NNCacheFileTailRepair repair);
  const char* nameOf(NNCacheFileMaintenance maintenance);
}

//-------------------------------------------------------------------------------------
// The cross-process lock
//-------------------------------------------------------------------------------------

// WHOSE TURN IT IS TO WRITE A CONTEXT'S FILES, when more than one engine process shares a
// cache directory.
//
// WHAT THIS REPLACES. Both formats used to open with the sentence "ONE WRITER ... stated
// rather than defended against, because the deployment is one engine process." That was a
// true scoping decision and it has expired: a proxy fanning queries across several KataGo
// LEAFs on one host puts several writers on one directory, and the failure mode is not the
// benign one the sentence implies. Two appenders do not merely lose the loser's dump. A
// block is written through a reused per-entry buffer -- many small writes, not one -- so
// concurrent appends interleave IN THE MIDDLE of the file, and the reader stops at the
// first block that fails its checksum and discards THE ENTIRE REMAINDER. One interleaved
// append therefore costs every block written after it, including blocks that were
// themselves perfectly good.
//
// THE LOCK IS ON A FILE OF ITS OWN (<context>.nnlock) AND THAT IS THE WHOLE POINT. The
// obvious implementation -- lock the count log, or lock the container -- is wrong for a
// reason that only shows up in the one operation that most needs the lock. Compaction
// writes a temp sibling and renames it over the data path. A lock held on the data file's
// inode survives that rename, but the inode no longer has the name: the next process to
// open the path gets the NEW inode and locks something nobody else is holding. Mutual
// exclusion would evaporate precisely during compaction. A file that is only ever created
// and locked -- never renamed, never truncated, never written -- has no such failure mode.
//
// THE UNIT IS THE CONTEXT, not the directory and not the individual file. A dump touches
// exactly one context's two files, so a context-wide lock is the smallest one that makes a
// dump atomic against another dump, and a directory-wide lock would serialise unrelated
// cards for no benefit.
//
// SHARED FOR READERS, EXCLUSIVE FOR WRITERS. An attach reads both files while another
// process may be appending, and a reader that observes a half-written block sees a torn
// tail and silently discards everything after it -- the same data loss as the writer/writer
// case, reached from the other side. Several engines attached to one card and none dumping
// is the common case, and it stays contention-free.
//
// A FILESYSTEM THAT CANNOT LOCK IS AN ERROR, NEVER A SHRUG. If the lock cannot be taken
// because the filesystem does not implement locking, this throws and names the errno rather
// than proceeding unlocked. Proceeding unlocked is exactly the silent-corruption case the
// class exists to prevent, so degrading to it on the platforms least able to afford it
// would invert the contract (ADR-0002).
enum class NNCacheFileLockMode {
  // Concurrent readers admitted; excludes every writer.
  Shared,
  // Excludes every other holder, reader or writer.
  Exclusive,
};

class NNCacheFileLock {
 public:
  // Acquires `mode` over `context`'s files in `directory`, creating the lock file if it is
  // not there. Waits up to `waitMilliseconds` for a conflicting holder to finish, then
  // throws StringError naming the context and the lock path.
  //
  // WHY A BOUNDED WAIT AND NOT EITHER EXTREME. Failing immediately would turn ordinary,
  // expected contention -- two engines dumping the same card seconds apart -- into a client-
  // visible error. Blocking forever would turn a wedged peer into a wedged engine carrying
  // no diagnostic. The bound is what makes contention survivable and a stuck holder
  // reportable.
  static NNCacheFileLock overContext(
    const std::string& directory,
    const std::string& context,
    NNCacheFileLockMode mode,
    int waitMilliseconds
  );
  // The same, at defaultWaitMilliseconds().
  static NNCacheFileLock overContext(
    const std::string& directory,
    const std::string& context,
    NNCacheFileLockMode mode
  );

  ~NNCacheFileLock();
  NNCacheFileLock(NNCacheFileLock&&) noexcept;
  NNCacheFileLock& operator=(NNCacheFileLock&&) noexcept;
  NNCacheFileLock(const NNCacheFileLock&) = delete;
  NNCacheFileLock& operator=(const NNCacheFileLock&) = delete;

  // The lock file this holds, for a message that has to name it.
  const std::string& path() const { return path_; }
  NNCacheFileLockMode mode() const { return mode_; }

  // How long overContext waits before it throws, when not told otherwise.
  static int defaultWaitMilliseconds();
  // The path overContext would lock, without acquiring it -- for a report, a test, or a
  // refusal that has to name the file. Validates `context` exactly as the acquiring form
  // does, so a bad name is refused identically whether or not a lock is taken.
  static std::string pathForContext(const std::string& directory, const std::string& context);

 private:
  NNCacheFileLock(std::string path, NNCacheFileLockMode mode);
  void release() noexcept;

  std::string path_;
  NNCacheFileLockMode mode_;
#ifdef _WIN32
  // A HANDLE, kept as void* so this header does not drag in windows.h.
  void* handle_;
#else
  int fd_;
#endif
};

#endif  // NEURALNET_NNCACHEFILEFORMAT_H_
