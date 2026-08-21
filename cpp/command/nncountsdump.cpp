#include <iostream>
#include <optional>

#include "../core/fileutils.h"
#include "../core/global.h"
#include "../external/nlohmann_json/json.hpp"
#include "../main.h"
#include "../neuralnet/nncachecountlog.h"
#include "../neuralnet/nncachefileformat.h"

// RENDERS A .nncounts FILE HUMAN-READABLY, from the storage side rather than the engine
// side. See nncachecountlog.h for what the format is and why; this file only reads it and
// prints it, through NNCacheCountLog::loadDetailedUnlocked() -- the SAME scan load() uses
// (nncachecountlog.cpp's scanLog), asked to also keep each block's own rows. There is no
// second parser of the byte layout here, by design: a hand-rolled reader that agreed with
// the real one today would drift the day the format's framing next changes.
//
// WHAT THE NUMBERS ARE, AND WHAT THEY ARE NOT (ledger rows 1651/1654 named this trap once
// already, so it is stated here rather than left to be inferred): every count this tool
// prints is a RECORDED RETRIEVAL -- a cache hit for a key some earlier dump had already
// stored, counted only from the dump onward that first noticed it. It is not a count of
// evaluations, and it is not a forward-pass estimate. A key evaluated once and never looked
// up again earns no row at all, in this file or in this tool's output.
//
// LOCKING. Every other public NNCacheCountLog operation takes the context's cross-process
// lock; this tool takes it too, WHEN IT CAN, with a bounded wait rather than the engine's
// default one -- an operator running this on the storage side of a mount (see
// deploy/kataproxy-gpu-topology/README.md's transport-agnostic section) may be looking at a
// directory where the engine-side lock does not bind at all, and a tool that refused there
// would be useless exactly where it is most wanted. A lock that cannot be taken in time is a
// printed caveat, never a failure: the read itself is safe unlocked too, because a torn tail
// -- the one thing an unlocked read concurrent with a live append can see -- is a reported
// disposition here, not a wrong number (see loadDetailedUnlocked's own comment).

using namespace std;
using json = nlohmann::json;

namespace {

const int BOUNDED_LOCK_WAIT_MS = 2000;

const char* const USAGE =
  "Usage: katago nncountsdump <file.nncounts> [--json]\n"
  "\n"
  "Renders a .nncounts (NN cache count log) file's contents human-readably.\n"
  "\n"
  "  --json    Emit the same data as one JSON object instead of the human-readable report.\n"
  "\n"
  "Read-only. Takes the context's shared lock with a bounded wait if it can; if it cannot,\n"
  "proceeds unlocked with a printed caveat rather than failing -- see nncountsdump.cpp.\n";

// Splits a path into the directory NNCacheCountLog::forContext expects and the context name
// its own path-building already derives from it (directory + "/" + context + ".nncounts").
// Recovering exactly that split is what lets this tool hand the file to forContext at all --
// forContext has no "open this literal path" door, on purpose: the context name is a
// validated identity, not a filename to be taken on faith (nncachecountlog.h's
// verifyContextName). A path this split cannot parse is refused here, by name, before
// anything is opened.
struct SplitPath {
  string directory;
  string context;
};

optional<SplitPath> splitCountsPath(const string& filePath, string& refusalReason) {
  const string suffix = ".nncounts";
  if(filePath.size() <= suffix.size() || filePath.compare(filePath.size() - suffix.size(), suffix.size(), suffix) != 0) {
    refusalReason = "'" + filePath + "' does not end in .nncounts; this tool only reads that format.";
    return optional<SplitPath>();
  }
  const size_t lastSlash = filePath.find_last_of('/');
  SplitPath split;
  if(lastSlash == string::npos) {
    split.directory = ".";
    split.context = filePath.substr(0, filePath.size() - suffix.size());
  }
  else {
    split.directory = filePath.substr(0, lastSlash);
    if(split.directory.empty())
      split.directory = "/";
    split.context = filePath.substr(lastSlash + 1, filePath.size() - lastSlash - 1 - suffix.size());
  }
  if(split.context.empty()) {
    refusalReason = "'" + filePath + "' has no context name before .nncounts.";
    return optional<SplitPath>();
  }
  return optional<SplitPath>(split);
}

string tailVerdict(const NNCacheCountLogContents& contents) {
  if(contents.tail() == NNCacheCountLogTail::Intact)
    return "intact";
  return "torn (" + Global::int64ToString(contents.discardedTailBytes()) + " bytes discarded)";
}

void printHuman(
  const string& filePath,
  const SplitPath& split,
  uint64_t contextHash,
  const optional<string>& lockCaveat,
  const NNCacheCountLogDetailedContents& detailed
) {
  const NNCacheCountLogContents& agg = detailed.aggregate();

  cout << "nncountsdump: " << filePath << endl;
  cout << "  context:            " << split.context << endl;
  cout << "  context name hash:  0x" << Global::uint64ToHexString(contextHash) << endl;
  if(lockCaveat.has_value())
    cout << "  lock:               NOT HELD -- proceeding read-only unlocked (" << *lockCaveat << ")" << endl;
  else
    cout << "  lock:               held (shared)" << endl;
  cout << endl;
  // The currency sentence lives ONE place, on the format type itself, so a later
  // re-denomination of what this log counts (tracked: ledger rows 1717/1722/1723) changes
  // this tool's wording by changing that one definition rather than by editing this call
  // site to match it.
  cout << "These counts are " << NNCacheCountLog::countsCurrencyDescription() << endl;
  cout << endl;

  for(size_t b = 0; b < detailed.blocks().size(); b++) {
    const NNCacheCountLogBlock& block = detailed.blocks()[b];
    cout << "block " << b << " (one cache_dump's append):" << endl;
    for(size_t i = 0; i < block.rows.size(); i++) {
      const NNCacheCountLogBlockRow& row = block.rows[i];
      cout << "  " << row.key.toString() << "  observations=" << row.observations << "  sessions=" << row.sessions << endl;
    }
    if(block.rows.empty())
      cout << "  (no rows)" << endl;
    cout << "  unattributed observations this block: " << block.unattributedObservations << endl;
    cout << endl;
  }
  if(detailed.blocks().empty())
    cout << "(no blocks applied)" << endl << endl;

  cout << "accumulated totals (merged across every block above):" << endl;
  const vector<NNCacheCountRow> sorted = agg.byDescendingObservations();
  uint64_t totalObservations = 0;
  for(size_t i = 0; i < sorted.size(); i++) {
    cout << "  " << sorted[i].key.toString() << "  observations=" << sorted[i].observations
         << "  sessions=" << sorted[i].sessions << endl;
    totalObservations += sorted[i].observations;
  }
  if(sorted.empty())
    cout << "  (no rows)" << endl;
  cout << endl;

  cout << "totals:" << endl;
  cout << "  rows:                  " << agg.rows().size() << endl;
  cout << "  blocks applied:        " << agg.blocksApplied() << endl;
  cout << "  records applied:       " << agg.recordsApplied() << endl;
  cout << "  total observations:         " << totalObservations << endl;
  cout << "  unattributed observations:  " << agg.unattributedObservations() << endl;
  cout << "  tail:                  " << tailVerdict(agg) << endl;
}

json toJson(
  const string& filePath,
  const SplitPath& split,
  uint64_t contextHash,
  const optional<string>& lockCaveat,
  const NNCacheCountLogDetailedContents& detailed
) {
  const NNCacheCountLogContents& agg = detailed.aggregate();

  json out;
  out["file"] = filePath;
  out["context"] = split.context;
  out["contextNameHash"] = Global::uint64ToHexString(contextHash);
  out["lockHeld"] = !lockCaveat.has_value();
  if(lockCaveat.has_value())
    out["lockCaveat"] = *lockCaveat;
  out["countsAre"] = NNCacheCountLog::countsCurrencyDescription();

  json blocks = json::array();
  for(size_t b = 0; b < detailed.blocks().size(); b++) {
    const NNCacheCountLogBlock& block = detailed.blocks()[b];
    json blockJson;
    blockJson["block"] = (int64_t)b;
    json rows = json::array();
    for(size_t i = 0; i < block.rows.size(); i++) {
      json row;
      row["key"] = block.rows[i].key.toString();
      row["observations"] = block.rows[i].observations;
      row["sessions"] = block.rows[i].sessions;
      rows.push_back(row);
    }
    blockJson["rows"] = rows;
    blockJson["unattributedObservations"] = block.unattributedObservations;
    blocks.push_back(blockJson);
  }
  out["blocks"] = blocks;

  json accumulated = json::array();
  uint64_t totalObservations = 0;
  const vector<NNCacheCountRow> sorted = agg.byDescendingObservations();
  for(size_t i = 0; i < sorted.size(); i++) {
    json row;
    row["key"] = sorted[i].key.toString();
    row["observations"] = sorted[i].observations;
    row["sessions"] = sorted[i].sessions;
    accumulated.push_back(row);
    totalObservations += sorted[i].observations;
  }
  out["accumulatedTotals"] = accumulated;

  json totals;
  totals["rows"] = (int64_t)agg.rows().size();
  totals["blocksApplied"] = agg.blocksApplied();
  totals["recordsApplied"] = agg.recordsApplied();
  totals["totalObservations"] = totalObservations;
  totals["unattributedObservations"] = agg.unattributedObservations();
  out["totals"] = totals;

  out["tail"] = agg.tail() == NNCacheCountLogTail::Intact ? "intact" : "torn";
  out["discardedTailBytes"] = agg.discardedTailBytes();

  return out;
}

}  // namespace

int MainCmds::nncountsdump(const vector<string>& args) {
  string filePath;
  bool wantsJson = false;

  for(size_t i = 1; i < args.size(); i++) {
    const string& arg = args[i];
    if(arg == "-help" || arg == "--help" || arg == "-h") {
      cout << USAGE;
      return 0;
    }
    else if(arg == "-json" || arg == "--json") {
      wantsJson = true;
    }
    else if(arg.size() > 0 && arg[0] == '-') {
      cerr << "nncountsdump: unknown option " << arg << endl << USAGE;
      return 1;
    }
    else if(filePath.empty()) {
      filePath = arg;
    }
    else {
      cerr << "nncountsdump: more than one file given." << endl << USAGE;
      return 1;
    }
  }

  if(filePath.empty()) {
    cerr << "nncountsdump: no file given." << endl << USAGE;
    return 1;
  }

  // TEACHING REFUSAL 1: the file has to exist. NNCacheCountLog::load() treats a missing
  // file as the normal answer "no dump has happened yet", which is right for an engine that
  // is about to create it -- but this tool names an existing file an operator wants to
  // inspect, and a typo in the path should not print an empty, all-zero report as if it
  // were the honest contents of a card nobody has dumped.
  if(!FileUtils::exists(filePath)) {
    cerr << "nncountsdump: " << filePath << " does not exist." << endl;
    return 1;
  }

  // TEACHING REFUSAL 2: the path has to parse into a directory and a context the way every
  // .nncounts path on disk was built (nncachecountlog.cpp's forContext: directory + "/" +
  // context + ".nncounts"). A file that does not fit that shape is not this format.
  string refusalReason;
  const optional<SplitPath> split = splitCountsPath(filePath, refusalReason);
  if(!split.has_value()) {
    cerr << "nncountsdump: " << refusalReason << endl;
    return 1;
  }

  optional<NNCacheCountLog> log;
  try {
    log.emplace(NNCacheCountLog::forContext(split->directory, split->context));
  }
  catch(const StringError& e) {
    cerr << "nncountsdump: " << e.what() << endl;
    return 1;
  }

  // Try the shared lock with a BOUNDED wait -- long enough to ride out an ordinary
  // in-progress dump, short enough that a directory whose lock does not bind (a network or
  // FUSE mount with no working flock -- the exact case lockfsprobe exists to catch) does not
  // hang this tool. Failing to acquire it is a caveat, never a refusal: see
  // loadDetailedUnlocked's own comment for why an unlocked read is still safe here.
  optional<NNCacheFileLock> lock;
  optional<string> lockCaveat;
  try {
    lock.emplace(NNCacheFileLock::overContext(
      split->directory, split->context, NNCacheFileLockMode::Shared, BOUNDED_LOCK_WAIT_MS));
  }
  catch(const StringError& e) {
    lockCaveat = string(e.what());
  }

  optional<NNCacheCountLogDetailedContents> detailed;
  try {
    detailed.emplace(log->loadDetailedUnlocked());
  }
  catch(const StringError& e) {
    cerr << "nncountsdump: " << e.what() << endl;
    return 1;
  }

  // Release before printing: holding it any longer than the read itself buys nothing and
  // only extends how long this tool can block a would-be writer on the bounded-wait path.
  lock.reset();

  if(wantsJson)
    cout << toJson(filePath, *split, log->contextHash(), lockCaveat, *detailed).dump(2) << endl;
  else
    printHuman(filePath, *split, log->contextHash(), lockCaveat, *detailed);

  return 0;
}
