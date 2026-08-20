#include <fstream>
#include <iostream>

#include "../core/global.h"
#include "../main.h"
#include "../neuralnet/nncachefrozen.h"

// The conformance driver for cpp/spec/chd/SPEC.md 7.2, the executable
// cpp/spec/chd/tools/check_vectors.py drives.
//
// It targets NNCacheFrozenIndex rather than the whole level-0 structure because the
// vectors carry keys and queries and no evaluations: they test resolution, not payload
// carriage (SPEC.md 7.2). That is exactly the seam the index/payload split exists at.
//
// Contract, verbatim from the spec:
//   <driver> build-and-query <keys-file> <queries-file>
//       Build over the keys IN FILE ORDER, line 1 -> entry 0. For each query in order
//       print one line to stdout: "MEMBER <i>" or "ABSENT". Print nothing else to stdout.
//       Exit 0.
//   <driver> build-only <keys-file>
//       Attempt construction only. Exit 0 on success; on refusal exit non-zero AND write a
//       diagnostic naming the reason to stderr.

namespace {

// Two lowercase 16-digit hex fields separated by one space: hash0 then hash1. A file with
// zero lines is a valid empty key set (SPEC.md 7.1).
std::vector<Hash128> readKeyFile(const std::string& path) {
  std::ifstream in(path);
  if(!in.good())
    throw StringError("chdvectordriver: could not open " + path);
  std::vector<Hash128> keys;
  std::string line;
  while(std::getline(in, line)) {
    if(line.size() > 0 && line[line.size() - 1] == '\r')
      line.erase(line.size() - 1);
    if(line.size() == 0)
      continue;
    const size_t space = line.find(' ');
    if(space == std::string::npos)
      throw StringError("chdvectordriver: malformed key line in " + path + ": '" + line + "'");
    const uint64_t hash0 = Global::hexStringToUInt64(Global::trim(line.substr(0, space)));
    const uint64_t hash1 = Global::hexStringToUInt64(Global::trim(line.substr(space + 1)));
    keys.push_back(Hash128(hash0, hash1));
  }
  return keys;
}

}  // namespace

static int runDriver(const std::vector<std::string>& args) {
  if(args.size() >= 3 && args[1] == "build-and-query") {
    if(args.size() != 4) {
      std::cerr << "chdvectordriver build-and-query <keys-file> <queries-file>" << std::endl;
      return 2;
    }
    const std::vector<Hash128> keys = readKeyFile(args[2]);
    const std::vector<Hash128> queries = readKeyFile(args[3]);
    // A construction refusal here is a driver failure for a BUILD_OK case, and the
    // exception carries it out with a nonempty stderr, which is the right outcome either
    // way. Nothing but answer lines reaches stdout.
    const NNCacheFrozenIndex index = NNCacheFrozenIndex::build(keys);
    std::string out;
    out.reserve(queries.size() * 10);
    for(size_t i = 0; i < queries.size(); i++) {
      const std::optional<uint32_t> found = index.find(queries[i]);
      if(found.has_value()) {
        out += "MEMBER ";
        out += Global::uint64ToString((uint64_t)found.value());
        out += "\n";
      }
      else {
        out += "ABSENT\n";
      }
    }
    std::cout << out << std::flush;
    return 0;
  }

  if(args.size() >= 2 && args[1] == "build-only") {
    if(args.size() != 3) {
      std::cerr << "chdvectordriver build-only <keys-file>" << std::endl;
      return 2;
    }
    const std::vector<Hash128> keys = readKeyFile(args[2]);
    const NNCacheFrozenIndex index = NNCacheFrozenIndex::build(keys);
    (void)index;
    return 0;
  }

  std::cerr << "chdvectordriver (build-and-query <keys-file> <queries-file> | build-only <keys-file>)"
            << std::endl;
  return 2;
}

int MainCmds::chdvectordriver(const std::vector<std::string>& args) {
  // A construction refusal arrives here as an exception (SPEC.md 5). The driver turns it
  // into what SPEC.md 7.2 grades a BUILD_FAIL case on: a non-zero exit AND a diagnostic
  // naming the reason on stderr. Nothing is written to stdout on this path.
  try {
    return runDriver(args);
  }
  catch(const std::exception& e) {
    std::cerr << "chdvectordriver: construction refused: " << e.what() << std::endl;
    return 1;
  }
}
