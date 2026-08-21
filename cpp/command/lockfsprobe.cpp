#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>

#ifndef _WIN32
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "../core/global.h"
#include "../main.h"
#include "../neuralnet/nncachefileformat.h"

// DOES flock ACTUALLY EXCLUDE ON THIS DIRECTORY'S FILESYSTEM? One command, one verdict, an
// exit status that carries it.
//
// WHY THIS EXISTS. NNCacheFileLock is what keeps several engine processes sharing one cache
// directory from interleaving their appends mid-file, and it is built on advisory locking:
// flock on POSIX, LockFileEx on Windows. Advisory locking is a property of the FILESYSTEM,
// not of the operating system. A local ext4/xfs/btrfs directory has it. An NFS mount, a
// FUSE filesystem, an sshfs mount, a container overlay or a network share may not -- and the
// ways it can be absent are not all loud. The dangerous shape is not the mount that returns
// ENOLCK, which NNCacheFileLock already refuses and names; it is the mount whose flock
// RETURNS ZERO FOR EVERYONE, so every process believes it holds the exclusive lock and the
// corruption the lock exists to prevent happens with the lock code running perfectly.
//
// WHICH IS WHY A SINGLE PROCESS CANNOT ANSWER THIS. One process calling flock() and getting
// 0 back has learned nothing at all: that is precisely the observation a no-op flock also
// produces. The only evidence that means anything is TWO REAL PROCESSES CONTENDING and an
// OBSERVED ORDERING between them -- one of them made to wait, and seen to proceed only after
// the other let go. So this forks, and the verdict rests on timestamps taken in two
// different processes on the one monotonic clock they share.
//
// IT LOCKS THROUGH NNCacheFileLock, NOT THROUGH A LOCAL COPY OF flock(). The question worth
// answering is not "does this filesystem implement some locking" but "does the code path
// production takes exclude here" -- the same class, the same lock-file naming, the same
// O_RDWR|O_CREAT open, the same modes. A probe that reimplemented the call could pass while
// the real path failed, which is the one outcome that would make the probe worse than
// nothing (ADR-0021: the witness observes the property, not a symptom).
//
// THE FOUR CHECKS, and what each one rules out:
//
//   UNCONTENDED. One process alone takes the exclusive lock, then the shared lock. This is
//   the check that catches the filesystem which cannot lock AT ALL: the errno path, where
//   NNCacheFileLock throws naming strerror(errno). It runs FIRST so that every throw in the
//   contended checks below has exactly one possible reading -- contention -- and no message
//   text has to be parsed to tell the two apart.
//
//   EXCLUDES-WRITER and EXCLUDES-READER. While the parent holds the exclusive lock, the
//   child attempts the exclusive lock and then the shared lock, each with a zero wait. Both
//   must be REFUSED. An acquisition here is the no-op-flock filesystem, caught red-handed
//   with two processes holding one exclusive lock.
//
//   ORDERING. The child then attempts the exclusive lock BLOCKING, while the parent goes on
//   holding it for a further hold period. The parent timestamps its release; the child
//   timestamps its acquisition; the child's must come after the parent's. This is the
//   positive half -- not merely "the child was refused" but "the child was made to wait and
//   then got in", which is the actual property production depends on.
//
//   SHARED-SHARES. Two processes hold the shared lock at once. A filesystem where this is
//   refused is not incorrect -- readers serialising costs throughput, not integrity -- so it
//   is reported as a WARNING and does not change the verdict. It is checked because without
//   it a filesystem whose flock always failed could not be distinguished from a healthy one
//   by the refusals above alone; the UNCONTENDED check is the primary guard against that and
//   this is the second.
//
// EXIT STATUS: 0 the lock excludes here, 1 it does not (or the filesystem cannot lock at
// all), 2 the probe could not run (bad arguments, missing directory, fork failure). 1 and 2
// are distinct because they call for different responses: 1 says "do not point a shared
// cache at this directory", 2 says "the probe did not answer".

namespace {

const char* const USAGE =
  "Usage: katago lockfsprobe <directory> [-context <name>] [-hold-ms <milliseconds>]\n"
  "\n"
  "Answers whether NNCacheFileLock actually excludes on <directory>'s filesystem, by\n"
  "contending for a lock file there between two real processes.\n"
  "\n"
  "  -context <name>      Lock-file context name; the probe locks <directory>/<name>.nnlock\n"
  "                       and removes it afterwards. Default \"lockfsprobe\". Point this at a\n"
  "                       name no live engine is using.\n"
  "  -hold-ms <ms>        How long the holding process keeps the lock while the other waits.\n"
  "                       Default 300. Larger only makes the probe slower.\n"
  "\n"
  "Exit 0 = the lock EXCLUDES here, 1 = it does NOT (or the filesystem cannot lock),\n"
  "2 = the probe could not run.\n";

const int DEFAULT_HOLD_MS = 300;
// The waiting child gives up long after the parent's hold, so that a slow filesystem shows
// up as a slow acquisition rather than as a deadline the probe itself imposed.
const int BLOCKING_WAIT_MULTIPLE = 40;
const int BLOCKING_WAIT_FLOOR_MS = 10000;

// The one clock both processes read. It is set before the fork, so the child inherits the
// same origin by copy and the two processes' milliseconds are comparable without any
// handshake -- steady_clock is CLOCK_MONOTONIC, which is system-wide on the platforms that
// have fork.
std::chrono::steady_clock::time_point probeEpoch;

double msSinceEpoch() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - probeEpoch).count();
}

#ifndef _WIN32

// What one forked child observed. Written to a pipe as raw bytes rather than as text,
// because there is no parsing to get wrong and the struct is POD by construction.
struct ChildReport {
  // 1 if the lock was ACQUIRED, 0 if it was refused. For the contended checks, 1 is failure.
  int32_t nonblockingExclusiveAcquired;
  int32_t nonblockingSharedAcquired;
  int32_t blockingExclusiveAcquired;
  int32_t sharedAcquired;
  // When the blocking exclusive acquisition happened, on the shared clock.
  double blockingAcquireMs;
  // The last refusal's text, so the parent can print the filesystem's own words.
  char message[600];
};

void setMessage(ChildReport& report, const std::string& text) {
  const size_t n = text.size() < sizeof(report.message) - 1 ? text.size() : sizeof(report.message) - 1;
  std::memcpy(report.message, text.data(), n);
  report.message[n] = '\0';
}

bool writeAll(int fd, const void* data, size_t len) {
  const char* p = (const char*)data;
  while(len > 0) {
    const ssize_t n = ::write(fd, p, len);
    if(n < 0) {
      if(errno == EINTR)
        continue;
      return false;
    }
    p += n;
    len -= (size_t)n;
  }
  return true;
}

bool readAll(int fd, void* data, size_t len) {
  char* p = (char*)data;
  while(len > 0) {
    const ssize_t n = ::read(fd, p, len);
    if(n < 0) {
      if(errno == EINTR)
        continue;
      return false;
    }
    if(n == 0)
      return false;
    p += n;
    len -= (size_t)n;
  }
  return true;
}

// Attempts a lock and says only whether it was acquired, releasing it immediately. Every
// throw is reported as a refusal HERE and interpreted by the caller, because by the time the
// contended checks run the uncontended check has already established that this filesystem
// can lock, so a throw can only be contention.
bool tryLock(
  const std::string& directory,
  const std::string& context,
  NNCacheFileLockMode mode,
  int waitMs,
  std::string& refusal
) {
  try {
    NNCacheFileLock lock = NNCacheFileLock::overContext(directory, context, mode, waitMs);
    (void)lock;
    return true;
  }
  catch(const StringError& e) {
    refusal = std::string(e.what());
    return false;
  }
}

// The contended check. The parent holds an exclusive lock across the whole of this; the
// child it forks does the attempting.
//
// The child NEVER RETURNS -- it leaves by _exit, so that no destructor runs in it. That is
// not tidiness: fork duplicates the parent's lock file descriptor into the child, the
// duplicate shares the parent's open file description, and running the inherited
// NNCacheFileLock's destructor in the child would flock(LOCK_UN) the description the PARENT
// is holding its lock on. The parent's lock would evaporate halfway through the check and
// the probe would report exclusion failing on a filesystem where it works perfectly.
bool runExclusionCheck(
  const std::string& directory,
  const std::string& context,
  int holdMs,
  ChildReport& report,
  double& parentReleaseMs,
  std::string& error
) {
  int readyPipe[2];
  int reportPipe[2];
  if(::pipe(readyPipe) != 0) {
    error = std::string("pipe() failed: ") + std::strerror(errno);
    return false;
  }
  if(::pipe(reportPipe) != 0) {
    error = std::string("pipe() failed: ") + std::strerror(errno);
    ::close(readyPipe[0]);
    ::close(readyPipe[1]);
    return false;
  }

  const int blockingWaitMs =
    holdMs * BLOCKING_WAIT_MULTIPLE > BLOCKING_WAIT_FLOOR_MS ? holdMs * BLOCKING_WAIT_MULTIPLE : BLOCKING_WAIT_FLOOR_MS;

  std::cout.flush();
  std::cerr.flush();

  // Scope so that the parent's lock is released at a point this function CHOOSES, timestamped
  // immediately before, rather than wherever the function happens to end.
  {
    NNCacheFileLock held = NNCacheFileLock::overContext(directory, context, NNCacheFileLockMode::Exclusive, 0);
    (void)held;

    const pid_t pid = ::fork();
    if(pid < 0) {
      error = std::string("fork() failed: ") + std::strerror(errno);
      ::close(readyPipe[0]);
      ::close(readyPipe[1]);
      ::close(reportPipe[0]);
      ::close(reportPipe[1]);
      return false;
    }

    if(pid == 0) {
      ::close(readyPipe[0]);
      ::close(reportPipe[0]);
      ChildReport mine;
      std::memset(&mine, 0, sizeof(mine));

      std::string refusal;
      mine.nonblockingExclusiveAcquired =
        tryLock(directory, context, NNCacheFileLockMode::Exclusive, 0, refusal) ? 1 : 0;
      if(mine.nonblockingExclusiveAcquired == 0)
        setMessage(mine, refusal);
      mine.nonblockingSharedAcquired =
        tryLock(directory, context, NNCacheFileLockMode::Shared, 0, refusal) ? 1 : 0;

      // Only now does the parent start its hold timer, so the two attempts above are known
      // to have happened while the parent held the lock rather than merely believed to.
      const char ready = 'r';
      (void)writeAll(readyPipe[1], &ready, 1);
      ::close(readyPipe[1]);

      try {
        NNCacheFileLock got =
          NNCacheFileLock::overContext(directory, context, NNCacheFileLockMode::Exclusive, blockingWaitMs);
        (void)got;
        mine.blockingAcquireMs = msSinceEpoch();
        mine.blockingExclusiveAcquired = 1;
      }
      catch(const StringError& e) {
        mine.blockingExclusiveAcquired = 0;
        setMessage(mine, std::string(e.what()));
      }

      (void)writeAll(reportPipe[1], &mine, sizeof(mine));
      ::close(reportPipe[1]);
      ::_exit(0);
    }

    ::close(readyPipe[1]);
    ::close(reportPipe[1]);

    char ready = '\0';
    if(!readAll(readyPipe[0], &ready, 1)) {
      error = "the forked child exited before reporting its non-blocking attempts";
      ::close(readyPipe[0]);
      ::close(reportPipe[0]);
      (void)::waitpid(pid, NULL, 0);
      return false;
    }
    ::close(readyPipe[0]);

    std::this_thread::sleep_for(std::chrono::milliseconds(holdMs));
    parentReleaseMs = msSinceEpoch();
  }

  std::memset(&report, 0, sizeof(report));
  const bool got = readAll(reportPipe[0], &report, sizeof(report));
  ::close(reportPipe[0]);
  int status = 0;
  (void)::waitpid(-1, &status, 0);
  if(!got) {
    error = "the forked child exited without writing its report";
    return false;
  }
  return true;
}

// Two processes holding the SHARED lock at once. Same _exit discipline and same reason.
bool runSharedCheck(
  const std::string& directory,
  const std::string& context,
  ChildReport& report,
  std::string& error
) {
  int reportPipe[2];
  if(::pipe(reportPipe) != 0) {
    error = std::string("pipe() failed: ") + std::strerror(errno);
    return false;
  }

  std::cout.flush();
  std::cerr.flush();

  {
    NNCacheFileLock held = NNCacheFileLock::overContext(directory, context, NNCacheFileLockMode::Shared, 0);
    (void)held;

    const pid_t pid = ::fork();
    if(pid < 0) {
      error = std::string("fork() failed: ") + std::strerror(errno);
      ::close(reportPipe[0]);
      ::close(reportPipe[1]);
      return false;
    }
    if(pid == 0) {
      ::close(reportPipe[0]);
      ChildReport mine;
      std::memset(&mine, 0, sizeof(mine));
      std::string refusal;
      mine.sharedAcquired = tryLock(directory, context, NNCacheFileLockMode::Shared, 0, refusal) ? 1 : 0;
      if(mine.sharedAcquired == 0)
        setMessage(mine, refusal);
      (void)writeAll(reportPipe[1], &mine, sizeof(mine));
      ::close(reportPipe[1]);
      ::_exit(0);
    }
    ::close(reportPipe[1]);

    std::memset(&report, 0, sizeof(report));
    const bool got = readAll(reportPipe[0], &report, sizeof(report));
    ::close(reportPipe[0]);
    int status = 0;
    (void)::waitpid(pid, &status, 0);
    if(!got) {
      error = "the forked child exited without writing its report";
      return false;
    }
  }
  return true;
}

#endif  // _WIN32

void line(const char* mark, const std::string& text) {
  std::cout << "  [" << mark << "] " << text << std::endl;
}

}  // namespace

int MainCmds::lockfsprobe(const std::vector<std::string>& args) {
  std::string directory;
  std::string context = "lockfsprobe";
  int holdMs = DEFAULT_HOLD_MS;

  for(size_t i = 1; i < args.size(); i++) {
    const std::string& arg = args[i];
    if(arg == "-help" || arg == "--help" || arg == "-h") {
      std::cout << USAGE;
      return 2;
    }
    else if(arg == "-context" || arg == "--context") {
      if(i + 1 >= args.size()) {
        std::cerr << "lockfsprobe: -context needs a value." << std::endl << USAGE;
        return 2;
      }
      context = args[++i];
    }
    else if(arg == "-hold-ms" || arg == "--hold-ms") {
      if(i + 1 >= args.size()) {
        std::cerr << "lockfsprobe: -hold-ms needs a value." << std::endl << USAGE;
        return 2;
      }
      try {
        holdMs = Global::stringToInt(args[++i]);
      }
      catch(const StringError& e) {
        std::cerr << "lockfsprobe: -hold-ms: " << e.what() << std::endl;
        return 2;
      }
      if(holdMs < 1 || holdMs > 60000) {
        std::cerr << "lockfsprobe: -hold-ms must be between 1 and 60000." << std::endl;
        return 2;
      }
    }
    else if(arg.size() > 0 && arg[0] == '-') {
      std::cerr << "lockfsprobe: unknown option " << arg << std::endl << USAGE;
      return 2;
    }
    else if(directory.empty()) {
      directory = arg;
    }
    else {
      std::cerr << "lockfsprobe: more than one directory given." << std::endl << USAGE;
      return 2;
    }
  }

  if(directory.empty()) {
    std::cerr << "lockfsprobe: no directory given." << std::endl << USAGE;
    return 2;
  }
  // A trailing slash would make the lock path "dir//name.nnlock", which is the same file but
  // a different string in every message the probe prints.
  while(directory.size() > 1 && directory[directory.size() - 1] == '/')
    directory.erase(directory.size() - 1);

  std::string lockPath;
  try {
    lockPath = NNCacheFileLock::pathForContext(directory, context);
  }
  catch(const StringError& e) {
    std::cerr << "lockfsprobe: " << e.what() << std::endl;
    return 2;
  }

  probeEpoch = std::chrono::steady_clock::now();

  std::cout << "lockfsprobe: does NNCacheFileLock exclude on this filesystem?" << std::endl;
  std::cout << "  directory: " << directory << std::endl;
  std::cout << "  lock file: " << lockPath << std::endl;
  std::cout << "  hold:      " << holdMs << " ms" << std::endl;
  std::cout << std::endl;

#ifdef _WIN32
  std::cout << "  [UNEXERCISED] This probe forks a second process, and there is no fork on Windows."
            << std::endl
            << "                It is compiled out here rather than reporting a verdict it did not"
            << std::endl
            << "                observe. Run it on the host that will hold the shared cache." << std::endl;
  std::cout << std::endl << "LOCKFS VERDICT: UNKNOWN (probe not available on this platform)" << std::endl;
  return 2;
#else

  {
    struct stat st;
    if(::stat(directory.c_str(), &st) != 0) {
      std::cerr << "lockfsprobe: cannot stat " << directory << ": " << std::strerror(errno) << std::endl;
      std::cout << std::endl << "LOCKFS VERDICT: UNKNOWN (the directory is not there)" << std::endl;
      return 2;
    }
    if(!S_ISDIR(st.st_mode)) {
      std::cerr << "lockfsprobe: " << directory << " is not a directory." << std::endl;
      std::cout << std::endl << "LOCKFS VERDICT: UNKNOWN (not a directory)" << std::endl;
      return 2;
    }
  }

  bool failed = false;
  bool warned = false;

  // CHECK 1: UNCONTENDED. Establishes that this filesystem implements locking at all, so
  // that every refusal below reads as contention and nothing has to parse a message.
  {
    std::string refusal;
    if(!tryLock(directory, context, NNCacheFileLockMode::Exclusive, 0, refusal)) {
      line("FAIL", "uncontended exclusive lock: REFUSED with nobody holding it.");
      line("    ", refusal);
      std::cout << std::endl
                << "LOCKFS VERDICT: UNSUPPORTED -- the lock cannot be taken here at all." << std::endl
                << "The refusal above is this filesystem's own words. It may not implement locking, or"
                << std::endl
                << "the directory may not be writable by this user; either way a shared NN cache cannot"
                << std::endl
                << "live here, and KataGo will refuse every dump and every attach against it, loudly,"
                << std::endl
                << "with that same message." << std::endl;
      return 1;
    }
    line("PASS", "uncontended exclusive lock acquired and released.");
    if(!tryLock(directory, context, NNCacheFileLockMode::Shared, 0, refusal)) {
      line("FAIL", "uncontended shared lock: REFUSED with nobody holding it.");
      line("    ", refusal);
      std::cout << std::endl
                << "LOCKFS VERDICT: UNSUPPORTED -- this filesystem cannot take a shared lock."
                << std::endl;
      return 1;
    }
    line("PASS", "uncontended shared lock acquired and released.");
  }

  // CHECKS 2-4: CONTENDED. Two real processes.
  ChildReport report;
  double parentReleaseMs = 0.0;
  std::string error;
  if(!runExclusionCheck(directory, context, holdMs, report, parentReleaseMs, error)) {
    std::cerr << "lockfsprobe: " << error << std::endl;
    std::cout << std::endl << "LOCKFS VERDICT: UNKNOWN (the probe could not run its two processes)" << std::endl;
    return 2;
  }

  if(report.nonblockingExclusiveAcquired != 0) {
    line("FAIL", "a second process TOOK THE EXCLUSIVE LOCK while this one held it.");
    line("    ", "Two processes held one exclusive lock at once: flock returns success here without "
                "excluding anything.");
    failed = true;
  }
  else {
    line("PASS", "a second process was refused the exclusive lock while this one held it.");
  }

  if(report.nonblockingSharedAcquired != 0) {
    line("FAIL", "a second process TOOK THE SHARED LOCK while this one held the exclusive lock.");
    line("    ", "A writer does not exclude readers here, so an attach can read a half-written block.");
    failed = true;
  }
  else {
    line("PASS", "a second process was refused the shared lock while this one held the exclusive lock.");
  }

  if(report.blockingExclusiveAcquired == 0) {
    line("FAIL", "a second process waiting for the exclusive lock never got it, even after release.");
    line("    ", std::string(report.message));
    failed = true;
  }
  else if(report.blockingAcquireMs < parentReleaseMs) {
    line("FAIL", "a second process acquired the exclusive lock BEFORE this one released it.");
    line("    ", "release at " + Global::doubleToString(parentReleaseMs) + " ms, second process acquired at " +
                Global::doubleToString(report.blockingAcquireMs) + " ms.");
    failed = true;
  }
  else {
    line("PASS", "a second process waited and then acquired: released at " +
                 Global::doubleToString(parentReleaseMs) + " ms, acquired at " +
                 Global::doubleToString(report.blockingAcquireMs) + " ms (waited " +
                 Global::doubleToString(report.blockingAcquireMs - parentReleaseMs) + " ms past release).");
  }

  ChildReport sharedReport;
  if(!runSharedCheck(directory, context, sharedReport, error)) {
    std::cerr << "lockfsprobe: " << error << std::endl;
    std::cout << std::endl << "LOCKFS VERDICT: UNKNOWN (the probe could not run its two processes)" << std::endl;
    return 2;
  }
  if(sharedReport.sharedAcquired != 0) {
    line("PASS", "two processes held the shared lock at once; readers do not serialise.");
  }
  else {
    line("WARN", "a second process could not take the shared lock while this one held it.");
    line("    ", std::string(sharedReport.message));
    line("    ", "Integrity is intact -- this is stricter than needed, not looser -- but every attach "
                "will serialise against every other attach here.");
    warned = true;
  }

  // Best-effort: the probe's own lock file is litter in a directory the operator is watching.
  // A failure to remove it is not worth a word, since its presence harms nothing.
  (void)::unlink(lockPath.c_str());

  std::cout << std::endl;
  if(failed) {
    std::cout << "LOCKFS VERDICT: UNSUPPORTED -- flock does NOT exclude on " << directory << std::endl
              << "Do NOT point a shared NN cache directory here. Two engine processes would each"
              << std::endl
              << "believe they hold the lock, their appends would interleave mid-file, and the reader"
              << std::endl
              << "would discard every block after the first torn one." << std::endl;
    return 1;
  }
  std::cout << "LOCKFS VERDICT: SUPPORTED -- flock excludes on " << directory << std::endl
            << "Two processes contended for the lock and the second one was made to wait." << std::endl;
  if(warned)
    std::cout << "(with one WARNING above: shared locks do not share here)" << std::endl;
  return 0;
#endif
}
