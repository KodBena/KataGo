#ifndef NEURALNET_NNCACHEFILEFORMAT_H_
#define NEURALNET_NNCACHEFILEFORMAT_H_

#include <cstdint>
#include <cstdio>
#include <cstddef>
#include <string>

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
//   THE DURABILITY PRIMITIVES. Both formats' crash story is append-plus-fsync with an
//   atomic rename for rewrites, and both need a file handle that survives a throw between
//   opening and closing. One RAII handle and one directory fsync, not two.
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

  // Flushes the stdio buffer and then forces the bytes to the device. Returns false, rather
  // than throwing, so the caller names the file in its own message.
  bool flushAndSync();

  // Closes early, so a rename can follow on platforms that will not rename an open file.
  // Idempotent.
  bool closeNow();

 private:
  std::FILE* file_;
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
