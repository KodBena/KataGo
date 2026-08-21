#include "../neuralnet/nncachefileformat.h"

#include <cstring>

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#include <windows.h>
#else
#include <unistd.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#endif

#include <cerrno>
#include <chrono>
#include <thread>

#include "../core/global.h"

// The byte layer shared by the count log and the evaluation container. See
// nncachefileformat.h for why each of the four pieces has one home rather than two.

namespace {
  const size_t MAX_NAME_LEN = 128;

  // How long an acquisition waits out a conflicting holder before it gives up and says so.
  // Sized against what it is waiting FOR: the longest legitimate holder is a dump of a large
  // container, which is bounded by the fsync of a file the header sizes at ~69-135 MB for a
  // median card and ~6x that at the largest. Twenty seconds clears that with room spare on any
  // plausible device, and anything still holding after it is not slow but stuck -- which is a
  // fact worth reporting rather than waiting on.
  const int DEFAULT_LOCK_WAIT_MS = 20000;
  // The retry schedule. Advisory locks carry no wait-and-notify primitive that is portable
  // across the platforms KataGo builds on, so acquisition polls -- and the interval is a
  // BACKOFF rather than a flat period for a reason that was measured rather than assumed.
  //
  // WHAT A FLAT 20 MS COST, WITNESSED. A reader holds the shared lock for as long as its read
  // takes and then re-acquires; the gap in which an exclusive acquirer can win is the interval
  // BETWEEN one reader's release and its next acquisition. A writer polling every 20 ms gets
  // fifty chances a second to land in that gap, and against a reader whose gap is a fraction
  // of a millisecond it essentially never does: a test in which one process appended blocks
  // while another sampled the same container in a loop starved the writer for the entire
  // 20-second deadline and then threw.
  //
  // A FAST FIRST RETRY IS WHAT MAKES ORDINARY CONTENTION CHEAP, AND THE CAP IS WHAT KEEPS A
  // LONG WAIT FROM SPINNING. Brief contention -- the common case, two dumps seconds apart --
  // is resolved in about a millisecond instead of up to twenty. A genuinely long wait settles
  // onto the 20 ms ceiling, so a full timeout still costs about a thousand syscalls rather
  // than twenty thousand.
  //
  // THIS IMPROVES THE ODDS AND DOES NOT MAKE THEM CERTAIN, which is stated here rather than
  // left for a reader to discover: flock offers no fairness guarantee and no queue, so a
  // continuous procession of shared holders with no gap at all can still starve an exclusive
  // acquirer to its deadline. The deadline is what makes that survivable -- it is reported by
  // name rather than waited on forever (ADR-0002).
  const int LOCK_POLL_MIN_MS = 1;
  const int LOCK_POLL_MAX_MS = 20;

  // The stdio buffer a sequential read stream is given. One mebibyte, chosen against what it
  // replaces rather than as a round number: the default is the filesystem's block size, 4096
  // bytes here, and a 256x buffer turns the measured 12,111 read syscalls of a 44.5 MB
  // container into the low hundreds. Larger buys progressively less -- the device bytes are
  // already exactly the file's size at any of these sizes, so all that is being bought is
  // syscalls -- and costs a megabyte per concurrently-open container, of which a multi-model
  // host has one per (context, model) pair being read.
  const size_t SEQUENTIAL_STREAM_BUFFER_BYTES = 1 << 20;
}

//-------------------------------------------------------------------------------------
// Little-endian packing
//-------------------------------------------------------------------------------------

void NNCacheFileBytes::put32(uint8_t* p, uint32_t v) {
  p[0] = (uint8_t)(v      ); p[1] = (uint8_t)(v >>  8);
  p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}

void NNCacheFileBytes::put64(uint8_t* p, uint64_t v) {
  for(int i = 0; i < 8; i++)
    p[i] = (uint8_t)(v >> (8 * i));
}

uint32_t NNCacheFileBytes::get32(const uint8_t* p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

uint64_t NNCacheFileBytes::get64(const uint8_t* p) {
  uint64_t v = 0;
  for(int i = 0; i < 8; i++)
    v |= ((uint64_t)p[i]) << (8 * i);
  return v;
}

void NNCacheFileBytes::putF32(uint8_t* p, float v) {
  static_assert(sizeof(float) == 4, "this format stores a float as its 4 IEEE-754 bytes");
  uint32_t bits;
  std::memcpy(&bits, &v, 4);
  put32(p, bits);
}

float NNCacheFileBytes::getF32(const uint8_t* p) {
  const uint32_t bits = get32(p);
  float v;
  std::memcpy(&v, &bits, 4);
  return v;
}

//-------------------------------------------------------------------------------------
// The checksum
//-------------------------------------------------------------------------------------

NNCacheFileChecksum::NNCacheFileChecksum(uint64_t seed)
  :state_(0xcbf29ce484222325ULL ^ seed)
{}

void NNCacheFileChecksum::update(const uint8_t* data, size_t len) {
  uint64_t h = state_;
  for(size_t i = 0; i < len; i++) {
    h ^= (uint64_t)data[i];
    h *= 0x100000001b3ULL;
  }
  state_ = h;
}

uint64_t NNCacheFileChecksum::finish() const {
  uint64_t h = state_;
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdULL;
  h ^= h >> 33;
  h *= 0xc4ceb9fe1a85ec53ULL;
  h ^= h >> 33;
  return h;
}

uint64_t NNCacheFileChecksum::of(const uint8_t* data, size_t len, uint64_t seed) {
  NNCacheFileChecksum sum(seed);
  sum.update(data, len);
  return sum.finish();
}

//-------------------------------------------------------------------------------------
// The path-component name boundary
//-------------------------------------------------------------------------------------

size_t NNCacheFileName::maxLength() { return MAX_NAME_LEN; }

void NNCacheFileName::verify(const std::string& name, const char* whoIsAsking, const char* whatItIs) {
  const std::string who = std::string(whoIsAsking) + ": the " + whatItIs;
  if(name.empty())
    throw StringError(who + " is empty.");
  if(name.size() > MAX_NAME_LEN)
    throw StringError(
      who + " is " + Global::uint64ToString(name.size()) +
      " characters; at most " + Global::uint64ToString(MAX_NAME_LEN) + " are allowed."
    );
  if(name == "." || name == "..")
    throw StringError(who + " is '" + name + "', which is not a usable name.");
  for(size_t i = 0; i < name.size(); i++) {
    const char c = name[i];
    const bool ok =
      (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
      c == '.' || c == '_' || c == '-';
    if(!ok)
      throw StringError(
        who + " contains '" + std::string(1, c) +
        "' at position " + Global::uint64ToString(i) +
        "; it may hold only ASCII letters, digits, '.', '_' and '-'."
      );
  }
}

uint64_t NNCacheFileName::hashOf(const std::string& name) {
  return NNCacheFileChecksum::of((const uint8_t*)name.data(), name.size(), 0);
}

//-------------------------------------------------------------------------------------
// The durability primitives
//-------------------------------------------------------------------------------------

namespace {
  // Whether a stdio mode string opens a stream that can write. Every mode but a bare "r"
  // family can: "w", "a" and any mode carrying '+' all write. Read once at construction
  // rather than re-parsed, and used only to refuse a page-cache drop on a write handle.
  bool modeCanWrite(const char* mode) {
    if(mode == nullptr || mode[0] == '\0')
      return true;  // an unreadable mode is treated as writable, which is the safe answer
    if(mode[0] != 'r')
      return true;
    for(const char* p = mode; *p != '\0'; p++) {
      if(*p == '+')
        return true;
    }
    return false;
  }
}

NNCacheFileHandle::NNCacheFileHandle(const std::string& path, const char* mode)
  :file_(std::fopen(path.c_str(), mode)),
   canWrite_(modeCanWrite(mode)),
   path_(path)
{}

NNCacheFileHandle::~NNCacheFileHandle() {
  if(file_ != nullptr)
    std::fclose(file_);
}

bool NNCacheFileHandle::flushAndSync() {
  if(file_ == nullptr)
    return false;
  if(std::fflush(file_) != 0)
    return false;
#ifdef _WIN32
  return _commit(_fileno(file_)) == 0;
#else
  return ::fsync(::fileno(file_)) == 0;
#endif
}

size_t NNCacheFileHandle::sequentialStreamBufferBytes() {
  return SEQUENTIAL_STREAM_BUFFER_BYTES;
}

bool NNCacheFileHandle::useSequentialStreamBuffer() {
  if(file_ == nullptr)
    return false;
  buffer_.resize(SEQUENTIAL_STREAM_BUFFER_BYTES);
  if(std::setvbuf(file_, buffer_.data(), _IOFBF, buffer_.size()) != 0) {
    // The stream keeps whatever buffer it had, so the storage is released rather than left
    // held for a setvbuf that did not take it.
    buffer_.clear();
    buffer_.shrink_to_fit();
    return false;
  }
#ifndef _WIN32
#ifdef POSIX_FADV_SEQUENTIAL
  (void)::posix_fadvise(::fileno(file_), 0, 0, POSIX_FADV_SEQUENTIAL);
#endif
#endif
  return true;
}

void NNCacheFileHandle::dropCachedRange(int64_t offset, int64_t bytes) {
  // A WRITE HANDLE IS REFUSED BY NAME. On a stream that can write, the pages this would
  // discard may be pages this process has written and not yet flushed, and asking the kernel
  // to forget them is asking it to lose data. It is a caller error rather than a runtime
  // condition, so it is a refusal and not a silent no-op (ADR-0002).
  if(canWrite_)
    throw StringError(
      "NNCacheFileHandle: a page-cache drop was requested on " + path_ +
      ", which is open for writing. Dropping cached pages of a stream that can write risks "
      "discarding bytes this process has not flushed; this is only ever correct on a handle "
      "opened for reading."
    );
  if(file_ == nullptr || offset < 0 || bytes <= 0)
    return;
#ifndef _WIN32
#ifdef POSIX_FADV_DONTNEED
  // The stdio buffer may still hold bytes the caller has not consumed; the advice is about
  // page cache and does not touch the stream's own position or buffer, so no flush is needed
  // and none is done.
  (void)::posix_fadvise(::fileno(file_), (off_t)offset, (off_t)bytes, POSIX_FADV_DONTNEED);
#else
  (void)offset; (void)bytes;
#endif
#else
  (void)offset; (void)bytes;
#endif
}

bool NNCacheFileHandle::closeNow() {
  if(file_ == nullptr)
    return true;
  const bool ok = std::fclose(file_) == 0;
  file_ = nullptr;
  return ok;
}

void NNCacheFileSync::directoryOf(const std::string& path) {
#ifndef _WIN32
  const size_t slash = path.find_last_of('/');
  const std::string dir = slash == std::string::npos ? std::string(".") : path.substr(0, slash);
  const int fd = ::open(dir.c_str(), O_RDONLY);
  if(fd < 0)
    return;
  (void)::fsync(fd);
  (void)::close(fd);
#else
  (void)path;
#endif
}

void NNCacheFileTruncate::toLength(const std::string& path, int64_t bytes) {
  if(bytes < 0)
    throw StringError(
      "NNCacheFileTruncate: a length of " + Global::int64ToString(bytes) + " for " + path +
      " has no reading."
    );

#ifdef _WIN32
  const int fd = ::_open(path.c_str(), _O_WRONLY | _O_BINARY);
  if(fd < 0)
    throw StringError(
      "NNCacheFileTruncate: could not open " + path + " to shorten it to " +
      Global::int64ToString(bytes) + " bytes: " + std::strerror(errno) + "."
    );
  const __int64 currentLength = ::_filelengthi64(fd);
  if(currentLength < 0) {
    ::_close(fd);
    throw StringError("NNCacheFileTruncate: could not size " + path + " before shortening it.");
  }
  if((int64_t)currentLength < bytes) {
    ::_close(fd);
    throw StringError(
      "NNCacheFileTruncate: " + path + " is " + Global::int64ToString((int64_t)currentLength) +
      " bytes and was asked to become " + Global::int64ToString(bytes) +
      ". This shortens a file to the end of its last intact block; it never grows one."
    );
  }
  if(::_chsize_s(fd, (__int64)bytes) != 0) {
    ::_close(fd);
    throw StringError(
      "NNCacheFileTruncate: could not shorten " + path + " to " + Global::int64ToString(bytes) +
      " bytes: " + std::strerror(errno) + "."
    );
  }
  const bool synced = ::_commit(fd) == 0;
  ::_close(fd);
#else
  const int fd = ::open(path.c_str(), O_WRONLY);
  if(fd < 0)
    throw StringError(
      "NNCacheFileTruncate: could not open " + path + " to shorten it to " +
      Global::int64ToString(bytes) + " bytes: " + std::strerror(errno) + "."
    );
  struct stat st;
  if(::fstat(fd, &st) != 0) {
    ::close(fd);
    throw StringError("NNCacheFileTruncate: could not size " + path + " before shortening it.");
  }
  if((int64_t)st.st_size < bytes) {
    ::close(fd);
    throw StringError(
      "NNCacheFileTruncate: " + path + " is " + Global::int64ToString((int64_t)st.st_size) +
      " bytes and was asked to become " + Global::int64ToString(bytes) +
      ". This shortens a file to the end of its last intact block; it never grows one."
    );
  }
  if(::ftruncate(fd, (off_t)bytes) != 0) {
    const int err = errno;
    ::close(fd);
    throw StringError(
      "NNCacheFileTruncate: could not shorten " + path + " to " + Global::int64ToString(bytes) +
      " bytes: " + std::strerror(err) + "."
    );
  }
  // The new LENGTH is what has to be durable. Without this the tail can come back after a
  // crash, and a block appended past it would be stranded at an offset no loader reaches --
  // exactly the failure the repair exists to prevent.
  const bool synced = ::fsync(fd) == 0;
  const int syncErr = errno;
  ::close(fd);
  if(!synced)
    throw StringError(
      "NNCacheFileTruncate: shortened " + path + " to " + Global::int64ToString(bytes) +
      " bytes but could not fsync the new length: " + std::strerror(syncErr) + "."
    );
#endif
#ifdef _WIN32
  if(!synced)
    throw StringError(
      "NNCacheFileTruncate: shortened " + path + " to " + Global::int64ToString(bytes) +
      " bytes but could not commit the new length."
    );
#endif
  // The length lives in the inode, not the directory entry, so no directory fsync is needed
  // -- unlike the rename this replaces, which is a directory operation. Stated because the
  // asymmetry with rewriteAsOneBlock's directory fsync is otherwise an apparent omission.
}

const char* NNCacheFileReport::nameOf(NNCacheFileTailRepair repair) {
  switch(repair) {
    case NNCacheFileTailRepair::NotNeeded: return "notNeeded";
    case NNCacheFileTailRepair::Truncated: return "truncated";
  }
  // Unreachable for a value of the enum; present so a future value added without visiting
  // this switch is a compile warning here rather than a silent empty string at the operator.
  throw StringError("NNCacheFileReport: a tail-repair disposition has no name.");
}

const char* NNCacheFileReport::nameOf(NNCacheFileMaintenance maintenance) {
  switch(maintenance) {
    case NNCacheFileMaintenance::Nothing: return "nothing";
    case NNCacheFileMaintenance::TruncatedTornTail: return "truncatedTornTail";
    case NNCacheFileMaintenance::Compacted: return "compacted";
  }
  throw StringError("NNCacheFileReport: a maintenance disposition has no name.");
}

//-------------------------------------------------------------------------------------
// The cross-process lock
//-------------------------------------------------------------------------------------

std::string NNCacheFileLock::pathForContext(const std::string& directory, const std::string& context) {
  NNCacheFileName::verify(context, "NNCacheFileLock", "context name");
  return directory + "/" + context + ".nnlock";
}

int NNCacheFileLock::defaultWaitMilliseconds() {
  return DEFAULT_LOCK_WAIT_MS;
}

NNCacheFileLock::NNCacheFileLock(std::string path, NNCacheFileLockMode mode)
  :path_(std::move(path)),
   mode_(mode),
#ifdef _WIN32
   handle_(nullptr)
#else
   fd_(-1)
#endif
{}

NNCacheFileLock::NNCacheFileLock(NNCacheFileLock&& other) noexcept
  :path_(std::move(other.path_)),
   mode_(other.mode_),
#ifdef _WIN32
   handle_(other.handle_)
#else
   fd_(other.fd_)
#endif
{
#ifdef _WIN32
  other.handle_ = nullptr;
#else
  other.fd_ = -1;
#endif
}

NNCacheFileLock& NNCacheFileLock::operator=(NNCacheFileLock&& other) noexcept {
  if(this != &other) {
    release();
    path_ = std::move(other.path_);
    mode_ = other.mode_;
#ifdef _WIN32
    handle_ = other.handle_;
    other.handle_ = nullptr;
#else
    fd_ = other.fd_;
    other.fd_ = -1;
#endif
  }
  return *this;
}

NNCacheFileLock::~NNCacheFileLock() {
  release();
}

// Unlocking is implicit in closing the descriptor on both platforms, but it is done
// explicitly first so that the release is not resting on that equivalence.
void NNCacheFileLock::release() noexcept {
#ifdef _WIN32
  if(handle_ != nullptr) {
    OVERLAPPED overlapped;
    std::memset(&overlapped, 0, sizeof(overlapped));
    (void)::UnlockFileEx((HANDLE)handle_, 0, MAXDWORD, MAXDWORD, &overlapped);
    (void)::CloseHandle((HANDLE)handle_);
    handle_ = nullptr;
  }
#else
  if(fd_ >= 0) {
    (void)::flock(fd_, LOCK_UN);
    (void)::close(fd_);
    fd_ = -1;
  }
#endif
}

NNCacheFileLock NNCacheFileLock::overContext(
  const std::string& directory,
  const std::string& context,
  NNCacheFileLockMode mode
) {
  return overContext(directory, context, mode, DEFAULT_LOCK_WAIT_MS);
}

NNCacheFileLock NNCacheFileLock::overContext(
  const std::string& directory,
  const std::string& context,
  NNCacheFileLockMode mode,
  int waitMilliseconds
) {
  if(waitMilliseconds < 0)
    throw StringError(
      "NNCacheFileLock: a negative wait of " + Global::intToString(waitMilliseconds) +
      " ms has no reading; pass 0 to attempt the lock once."
    );

  const std::string path = pathForContext(directory, context);
  const char* const what = mode == NNCacheFileLockMode::Exclusive ? "exclusive" : "shared";
  NNCacheFileLock lock(path, mode);

  // The lock file is CREATED but never written, and that is deliberate: its only content is
  // its existence, so there is nothing in it a torn write could damage and nothing a reader
  // has to interpret.
#ifdef _WIN32
  const HANDLE handle = ::CreateFileA(
    path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
    NULL, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL
  );
  if(handle == INVALID_HANDLE_VALUE)
    throw StringError(
      "NNCacheFileLock: could not open lock file " + path + " for context '" + context +
      "' (Windows error " + Global::int64ToString((int64_t)::GetLastError()) + ")."
    );
  lock.handle_ = (void*)handle;
  const DWORD flags =
    LOCKFILE_FAIL_IMMEDIATELY | (mode == NNCacheFileLockMode::Exclusive ? LOCKFILE_EXCLUSIVE_LOCK : 0);
#else
  lock.fd_ = ::open(path.c_str(), O_RDWR | O_CREAT, 0644);
  if(lock.fd_ < 0)
    throw StringError(
      "NNCacheFileLock: could not open lock file " + path + " for context '" + context +
      "': " + std::strerror(errno) + "."
    );
  const int op = (mode == NNCacheFileLockMode::Exclusive ? LOCK_EX : LOCK_SH) | LOCK_NB;
#endif

  const std::chrono::steady_clock::time_point deadline =
    std::chrono::steady_clock::now() + std::chrono::milliseconds(waitMilliseconds);
  int pollMs = LOCK_POLL_MIN_MS;
  for(;;) {
#ifdef _WIN32
    OVERLAPPED overlapped;
    std::memset(&overlapped, 0, sizeof(overlapped));
    if(::LockFileEx((HANDLE)lock.handle_, flags, 0, MAXDWORD, MAXDWORD, &overlapped))
      return lock;
    const DWORD err = ::GetLastError();
    const bool heldByAnother = err == ERROR_LOCK_VIOLATION || err == ERROR_IO_PENDING;
    if(!heldByAnother)
      throw StringError(
        "NNCacheFileLock: could not take the " + std::string(what) + " lock on " + path +
        " (Windows error " + Global::int64ToString((int64_t)err) + "). The cache directory must be "
        "on a filesystem that implements locking: KataGo will not append to a shared cache file it "
        "cannot lock, because an unlocked concurrent append interleaves mid-file and costs every "
        "block written after it."
      );
#else
    if(::flock(lock.fd_, op) == 0)
      return lock;
    const int err = errno;
    // EWOULDBLOCK is the one refusal that means "someone else holds it"; every other errno
    // means the lock cannot be taken AT ALL, which is a different fact with a different
    // remedy and must not be retried until the deadline as though it were contention.
    // EAGAIN is named beside it because POSIX permits the two to be distinct values, and on
    // a platform where they are, the one this flock returns is not fixed by the standard.
    if(err != EWOULDBLOCK && err != EAGAIN && err != EINTR)
      throw StringError(
        "NNCacheFileLock: could not take the " + std::string(what) + " lock on " + path +
        ": " + std::strerror(err) + ". The cache directory must be on a filesystem that "
        "implements advisory locking: KataGo will not append to a shared cache file it cannot "
        "lock, because an unlocked concurrent append interleaves mid-file and costs every block "
        "written after it."
      );
#endif
    if(std::chrono::steady_clock::now() >= deadline)
      throw StringError(
        "NNCacheFileLock: waited " + Global::intToString(waitMilliseconds) + " ms for the " +
        std::string(what) + " lock on " + path + " (context '" + context + "') and another "
        "process still holds it. A dump or a compaction of this context is either still running "
        "elsewhere or has left the lock behind; no bytes of this context's files were touched."
      );
    std::this_thread::sleep_for(std::chrono::milliseconds(pollMs));
    if(pollMs < LOCK_POLL_MAX_MS)
      pollMs = pollMs * 2 < LOCK_POLL_MAX_MS ? pollMs * 2 : LOCK_POLL_MAX_MS;
  }
}
