#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>

#include "../core/datetime.h"
#include "../core/fileutils.h"
#include "../core/global.h"
#include "../core/timer.h"
#include "../main.h"
#include "../neuralnet/nncache.h"
#include "../neuralnet/nncachetrace.h"
#include "../program/setup.h"

using namespace std;

// The nn-cache policy sweep.
//
// WHAT THIS ANSWERS. For each point of the policy matrix -- collision scheme x ways x
// eviction x admission, plus the direct-mapped baseline, at a range of table sizes and
// with ownership maps present or absent -- what hit rate does it deliver, how many bytes
// is it actually holding to deliver it, and at what occupancy. Occupancy is reported
// beside every hit rate because a hit rate without it is uninterpretable: a replacement
// policy has nothing to do in a nearly-empty table, and every earlier measurement in
// this programme sat at 1.7% occupancy, which is not the regime the question is about.
//
// OCCUPANCY IS THE INDEPENDENT VARIABLE. Table size is swept as the cheap MEANS of
// reaching a range of occupancies against one fixed workload, not as the question. A
// fixed trace of N distinct keys fills a 2^k table to roughly N/2^k, so sweeping k
// downward walks the occupancy axis from empty to saturated at a fraction of the memory
// a big table would cost -- and it produces a CURVE of policy benefit against occupancy,
// from which a reader can locate whatever table size he actually runs.
//
// WHAT IT DOES NOT ANSWER. Visits per second. That comes from `katago benchmark` over a
// config carrying the nnCache* keys, which is the path the operator already trusts, and
// the runbook joins the two by configuration. Speed is an affordability check here, not
// a decision input -- the whole cache path was measured at 0.0077% of cycles -- so it is
// deliberately not allowed to shape this instrument. What IS measured here is cache
// operations per second, which is the cache path's own throughput and is the quantity
// this binary can honestly produce single-threaded on an idle core.
//
// THE WORKLOAD IS A REPLAY. See nncachetrace.h for the format, for why a replay rather
// than a live run per configuration, and -- importantly -- for the assumption a replay
// rests on and where that assumption stops holding.

namespace {

//-------------------------------------------------------------------------------------
// The results file
//-------------------------------------------------------------------------------------
// NDJSON: one JSON object per line, first line a "substrate" record and every later line
// a "config" record. One file per run, because a sweep is one experiment and its
// substrate is one fact about it; and self-describing, because these runs happen on a
// machine nobody who later reads the file can inspect.

static string jsonEscape(const string& s) {
  string out;
  for(size_t i = 0; i<s.size(); i++) {
    const char c = s[i];
    if(c == '"' || c == '\\') { out += '\\'; out += c; }
    else if(c == '\n') out += "\\n";
    else if(c == '\r') out += "\\r";
    else if(c == '\t') out += "\\t";
    else if((unsigned char)c < 0x20) out += ' ';
    else out += c;
  }
  return out;
}

// Accumulates key/value pairs and prints one JSON object. Values are pre-rendered, so
// the caller decides whether a field is a string, a number or a boolean, and a number
// never arrives quoted by accident.
class JsonLine {
  vector<pair<string,string>> fields;
 public:
  void str(const string& k, const string& v) { fields.push_back(make_pair(k, "\"" + jsonEscape(v) + "\"")); }
  void num(const string& k, int64_t v) { fields.push_back(make_pair(k, Global::int64ToString(v))); }
  void real(const string& k, double v) {
    ostringstream o;
    o << setprecision(10) << v;
    // A non-finite value is a defect upstream, and writing `nan` would produce a file no
    // JSON reader accepts -- fail where the number is made, not where it is parsed.
    const string s = o.str();
    if(s.find("nan") != string::npos || s.find("inf") != string::npos)
      throw StringError("benchnncachepolicy: refusing to emit a non-finite value for field '" + k + "'");
    fields.push_back(make_pair(k, s));
  }
  void boolean(const string& k, bool v) { fields.push_back(make_pair(k, v ? "true" : "false")); }
  string render() const {
    string out = "{";
    for(size_t i = 0; i<fields.size(); i++) {
      if(i > 0) out += ",";
      out += "\"" + fields[i].first + "\":" + fields[i].second;
    }
    return out + "}";
  }
};

//-------------------------------------------------------------------------------------
// Substrate
//-------------------------------------------------------------------------------------
// ADR-0015 Rule 4: a run offered as evidence states the machine state it ran under. Every
// field here is read from the machine at run time rather than passed in, because a
// substrate field the operator types is a substrate field the operator can get wrong.

static string readFirstLine(const string& path) {
  ifstream in(path.c_str());
  if(!in.good())
    return "UNREADABLE";
  string line;
  if(!std::getline(in,line))
    return "EMPTY";
  return line;
}

static string readCommandOutput(const string& cmd) {
  FILE* p = popen(cmd.c_str(),"r");
  if(p == NULL)
    return "UNREADABLE";
  string out;
  char buf[512];
  while(fgets(buf,sizeof(buf),p) != NULL)
    out += buf;
  pclose(p);
  while(!out.empty() && (out[out.size()-1] == '\n' || out[out.size()-1] == '\r'))
    out.resize(out.size()-1);
  return out.empty() ? "UNREADABLE" : out;
}

// pswpin/pswpout at a moment. Sampled ONCE before the whole sweep and ONCE after it, and
// the delta reported in the results file: a spot check inside a long window proves
// nothing about the window, and that error has already been made and reported on this
// programme. A nonzero pswpin delta disqualifies every timing figure in the file, which
// is why the raw pair is written out rather than a verdict this binary invented.
static bool readSwapCounters(int64_t& pswpin, int64_t& pswpout) {
  ifstream in("/proc/vmstat");
  if(!in.good())
    return false;
  pswpin = -1; pswpout = -1;
  string key; int64_t val;
  while(in >> key >> val) {
    if(key == "pswpin") pswpin = val;
    else if(key == "pswpout") pswpout = val;
  }
  return pswpin >= 0 && pswpout >= 0;
}

//-------------------------------------------------------------------------------------
// The matrix
//-------------------------------------------------------------------------------------

struct PolicyPoint {
  string label;              // how this point is named in the results file
  NNCacheCollisionScheme scheme;
  int ways;                  // 1 under direct and chain
  NNCacheEvictionPolicy eviction;
  bool hasEviction;          // false only under direct
  NNCacheAdmissionPolicy admission;
};

static string schemeName(NNCacheCollisionScheme s) {
  switch(s) {
  case NNCacheCollisionScheme::Direct: return "direct";
  case NNCacheCollisionScheme::LinearProbe: return "linearprobe";
  case NNCacheCollisionScheme::QuadraticProbe: return "quadraticprobe";
  case NNCacheCollisionScheme::Chain: return "chain";
  default: break;
  }
  throw StringError("benchnncachepolicy: unhandled collision scheme");
}

static string evictionName(NNCacheEvictionPolicy e) {
  switch(e) {
  case NNCacheEvictionPolicy::Random: return "random";
  case NNCacheEvictionPolicy::Lru: return "lru";
  case NNCacheEvictionPolicy::Lfu: return "lfu";
  case NNCacheEvictionPolicy::None: return "none";
  default: break;
  }
  throw StringError("benchnncachepolicy: unhandled eviction policy");
}

static NNCacheEvictionPolicy evictionFromName(const string& s) {
  if(s == "random") return NNCacheEvictionPolicy::Random;
  if(s == "lru") return NNCacheEvictionPolicy::Lru;
  if(s == "lfu") return NNCacheEvictionPolicy::Lfu;
  throw StringError(
    "benchnncachepolicy: -evictions accepts (random|lru|lfu), got '" + s + "'. "
    "'none' is not a sweepable policy: it names the absence of a choice, and every shape "
    "in this matrix makes one."
  );
}

static NNCacheCollisionScheme schemeFromName(const string& s) {
  if(s == "direct") return NNCacheCollisionScheme::Direct;
  if(s == "linearprobe") return NNCacheCollisionScheme::LinearProbe;
  if(s == "quadraticprobe") return NNCacheCollisionScheme::QuadraticProbe;
  if(s == "chain") return NNCacheCollisionScheme::Chain;
  throw StringError(
    "benchnncachepolicy: -collisions accepts (direct|linearprobe|quadraticprobe|chain), got '" + s + "'"
  );
}

static vector<string> splitList(const string& s) {
  vector<string> out;
  string cur;
  for(size_t i = 0; i<s.size(); i++) {
    if(s[i] == ',') { if(!cur.empty()) out.push_back(cur); cur.clear(); }
    else cur += s[i];
  }
  if(!cur.empty()) out.push_back(cur);
  if(out.empty())
    throw StringError("benchnncachepolicy: empty comma-separated list");
  return out;
}

//-------------------------------------------------------------------------------------
// One replayed configuration
//-------------------------------------------------------------------------------------

struct ReplayResult {
  int64_t gets;
  int64_t hits;
  int64_t sets;
  // A get that missed a key this table HELD EARLIER in this same replay. This is the
  // quantity that bounds what any replacement policy could possibly buy: a re-miss is a
  // hit the table gave away, and a policy can only ever recover re-misses. A workload
  // whose re-miss count is near zero is one where every policy must score the same,
  // whatever its hit rate looks like.
  int64_t reMisses;
  double seconds;
  NNCacheStats stats;
};

// A payload carrying the recorded footprint. nnOutputFootprintBytes counts the
// separately-allocated ownership map when whiteOwnerMap is non-null, so reproducing a
// recorded footprint means actually attaching one -- there is no way to fake the number
// without also faking what the table is asked to hold, and faking it is precisely how a
// byte budget gets validated against a fiction.
static shared_ptr<NNOutput> payloadFor(Hash128 hash, uint32_t recordedBytes, bool& attachedOwnerMap) {
  shared_ptr<NNOutput> p = make_shared<NNOutput>();
  p->nnHash = hash;
  p->nnXLen = 19;
  p->nnYLen = 19;
  attachedOwnerMap = recordedBytes >= (uint32_t)(sizeof(NNOutput) + 19*19*sizeof(float));
  if(attachedOwnerMap)
    p->whiteOwnerMap = new float[19*19];
  return p;
}

static ReplayResult replayOne(
  const NNCacheConfig& config,
  const vector<NNCacheTraceRecord>& trace,
  bool forceOwnerMap,
  bool forceNoOwnerMap
) {
  unique_ptr<NNCacheTable> table = NNCacheTable::create(config);

  // Keys this table has held at some point in this replay, for the re-miss count. This
  // is the harness's own bookkeeping and is deliberately outside the table, so nothing
  // measured here depends on the table maintaining a statistic for us.
  std::set<uint64_t> everHeld;

  ReplayResult r = {0,0,0,0,0.0,{0,0,0,0}};
  shared_ptr<NNOutput> got;
  ClockTimer timer;
  for(size_t i = 0; i<trace.size(); i++) {
    const NNCacheTraceRecord& rec = trace[i];
    const Hash128 hash(rec.hash0, rec.hash1);
    if((rec.flags & NNCacheTrace::FLAG_IS_SET) != 0) {
      uint32_t bytes = rec.bytes;
      if(forceOwnerMap) bytes = (uint32_t)(sizeof(NNOutput) + 19*19*sizeof(float));
      if(forceNoOwnerMap) bytes = (uint32_t)sizeof(NNOutput);
      bool attached = false;
      table->set(payloadFor(hash,bytes,attached));
      r.sets += 1;
      everHeld.insert(rec.hash0 ^ rec.hash1);
    }
    else {
      const bool found = table->get(hash,got);
      r.gets += 1;
      if(found)
        r.hits += 1;
      else if(everHeld.count(rec.hash0 ^ rec.hash1) != 0)
        r.reMisses += 1;
    }
  }
  r.seconds = timer.getSeconds();
  // Taken AFTER the timer stops: stats() walks the whole table and takes every region
  // lock, so folding it into the timed window would measure the reporting call.
  r.stats = table->stats();
  return r;
}

//-------------------------------------------------------------------------------------
// The memory preflight
//-------------------------------------------------------------------------------------
// ADR-0015 Rule 1: a heavy run checks its headroom and refuses loudly rather than
// starting something the kernel will adjudicate. This box has already produced one OOM
// kill on this programme, and cgroup memory caps are inert here -- the memory controller
// is not delegated to the user slice -- so the check has to be arithmetic done before
// allocating, not a limit the system will enforce.

static int64_t worstCasePeakBytes(
  const vector<int>& tablePows, int mutexPow, int64_t distinctKeys, int64_t maxEntryBytes, bool anyChain,
  int64_t chainBudget
) {
  // The peak is one configuration at a time -- tables are built and destroyed in turn --
  // so it is the largest single configuration, not the sum.
  int64_t peak = 0;
  for(size_t i = 0; i<tablePows.size(); i++) {
    const int64_t slots = ((int64_t)1) << tablePows[i];
    // Slot array at the widest slot the matrix can ask for (an LRU/LFU probed slot is 32
    // bytes on this toolchain; use 32 rather than reading it, so the preflight never
    // depends on the thing it is protecting).
    const int64_t slotArray = slots * 32;
    // Resident payloads are capped by BOTH what the table can hold and how many distinct
    // keys the trace has: a 2^21 table replayed against 442k distinct keys never holds
    // more than 442k entries, and pretending otherwise would refuse runs that fit.
    const int64_t residentCap = std::min(slots, distinctKeys);
    int64_t here = slotArray + residentCap * maxEntryBytes;
    if(anyChain) {
      const int64_t chainHere = slots * 8 + std::min(chainBudget, distinctKeys * maxEntryBytes);
      here = std::max(here, chainHere);
    }
    // The ghost table under second-sighting, and the mutex pool.
    here += slots * 4;
    here += (((int64_t)1) << mutexPow) * 64;
    peak = std::max(peak,here);
  }
  return peak;
}

}  // namespace

//-------------------------------------------------------------------------------------

int MainCmds::benchnncachepolicy(const vector<string>& args) {
  string tracePath;
  string outPath;
  string tablePowsStr = "17,18,19,20";
  string waysStr = "2,4,8,16";
  string collisionsStr = "direct,linearprobe,quadraticprobe,chain";
  string evictionsStr = "random,lru,lfu";
  string admissionsStr = "always,secondsighting";
  string ownershipStr = "trace";
  string note;
  string backendName = "none (replay: no neural net is loaded)";
  int mutexPow = -1;
  int64_t maxBytesBudget = -1;
  int64_t chainBudget = -1;
  int64_t assumedV1Bytes = (int64_t)sizeof(NNOutput);
  int64_t maxOps = -1;

  // main.cpp hands every subcommand its own name as args[0] (the TCLAP-based commands
  // consume it as the "program name" a usage message prints). This command parses its
  // own flags, so it skips that leading non-flag token rather than choking on it.
  const size_t argStart = (!args.empty() && !args[0].empty() && args[0][0] != '-') ? 1 : 0;
  for(size_t i = argStart; i<args.size(); i++) {
    const string& a = args[i];
    const bool hasNext = i+1 < args.size();
    if(a == "-trace" && hasNext) tracePath = args[++i];
    else if(a == "-out" && hasNext) outPath = args[++i];
    else if(a == "-table-pows" && hasNext) tablePowsStr = args[++i];
    else if(a == "-ways" && hasNext) waysStr = args[++i];
    else if(a == "-collisions" && hasNext) collisionsStr = args[++i];
    else if(a == "-evictions" && hasNext) evictionsStr = args[++i];
    else if(a == "-admissions" && hasNext) admissionsStr = args[++i];
    else if(a == "-ownership" && hasNext) ownershipStr = args[++i];
    else if(a == "-mutex-pow" && hasNext) mutexPow = Global::stringToInt(args[++i]);
    else if(a == "-max-bytes" && hasNext) maxBytesBudget = Global::stringToInt64(args[++i]);
    else if(a == "-chain-budget" && hasNext) chainBudget = Global::stringToInt64(args[++i]);
    else if(a == "-assume-v1-entry-bytes" && hasNext) assumedV1Bytes = Global::stringToInt64(args[++i]);
    else if(a == "-max-ops" && hasNext) maxOps = Global::stringToInt64(args[++i]);
    else if(a == "-backend-name" && hasNext) backendName = args[++i];
    else if(a == "-note" && hasNext) note = args[++i];
    else
      throw StringError(
        "benchnncachepolicy: unrecognized argument '" + a + "'.\n"
        "Required:\n"
        "  -trace PATH        an nn-cache trace (see KATAGO_NNCACHE_TRACE)\n"
        "  -out PATH          where the NDJSON results file is written\n"
        "  -max-bytes N       the memory budget this run may not exceed, in bytes.\n"
        "                     Required, with no default: the whole reason this flag exists is\n"
        "                     that assuming a number is how the previous plan died.\n"
        "Optional:\n"
        "  -table-pows L      default 17,18,19,20 -- swept to reach a RANGE OF OCCUPANCIES\n"
        "  -ways L            default 2,4,8,16 (probed schemes only)\n"
        "  -collisions L      default direct,linearprobe,quadraticprobe,chain\n"
        "  -evictions L       default random,lru,lfu\n"
        "  -admissions L      default always,secondsighting\n"
        "  -ownership W       trace | off | on | both   (default trace: use each set's\n"
        "                     RECORDED footprint. off/on override every entry; both runs each\n"
        "                     configuration twice, once each way.)\n"
        "  -mutex-pow N       default max(0, tablePow-4), per table size, so that 16 ways always\n"
        "                     fits a lock region. Recorded per configuration either way.\n"
        "  -chain-budget N    byte budget for chain shapes. Default: the bytes a direct table of\n"
        "                     the same tablePow would hold at the trace's mean entry size, so\n"
        "                     chain is compared against the others at equal memory rather than\n"
        "                     at an arbitrary budget.\n"
        "  -assume-v1-entry-bytes N   what a v1 (headerless, 20-byte) trace's missing footprint\n"
        "                     is taken to be. Default sizeof(NNOutput). Stamped in the results.\n"
        "  -max-ops N         replay only the first N trace records (a smoke test knob)\n"
        "  -backend-name S    recorded verbatim in the substrate line; say what BUILT the trace\n"
        "  -note S            free text recorded in the substrate line\n"
      );
  }

  if(tracePath.empty())
    throw StringError("benchnncachepolicy: -trace is required. Run with an unknown flag for the full usage.");
  if(outPath.empty())
    throw StringError("benchnncachepolicy: -out is required. Run with an unknown flag for the full usage.");
  if(maxBytesBudget <= 0)
    throw StringError(
      "benchnncachepolicy: -max-bytes is REQUIRED and has no default. It is the memory this run may\n"
      "not exceed, in bytes. There is no default because the sweep this replaces was planned against\n"
      "an assumed memory figure that turned out to be wrong, and the assumption is what cost the plan.\n"
      "State the budget for THIS machine. On a host whose RAM is shared with a VM or other tasks, the\n"
      "budget is what is FREE, not what is installed."
    );

  vector<int> tablePows;
  {
    vector<string> parts = splitList(tablePowsStr);
    for(size_t i = 0; i<parts.size(); i++) {
      const int p = Global::stringToInt(parts[i]);
      if(p < 1 || p > 30)
        throw StringError("benchnncachepolicy: -table-pows entry out of range 1..30: " + parts[i]);
      tablePows.push_back(p);
    }
  }
  vector<int> waysList;
  {
    vector<string> parts = splitList(waysStr);
    for(size_t i = 0; i<parts.size(); i++)
      waysList.push_back(Global::stringToInt(parts[i]));
  }
  vector<NNCacheCollisionScheme> schemes;
  {
    vector<string> parts = splitList(collisionsStr);
    for(size_t i = 0; i<parts.size(); i++)
      schemes.push_back(schemeFromName(parts[i]));
  }
  vector<NNCacheEvictionPolicy> evictions;
  {
    vector<string> parts = splitList(evictionsStr);
    for(size_t i = 0; i<parts.size(); i++)
      evictions.push_back(evictionFromName(parts[i]));
  }
  vector<NNCacheAdmissionPolicy> admissions;
  {
    vector<string> parts = splitList(admissionsStr);
    for(size_t i = 0; i<parts.size(); i++) {
      if(parts[i] == "always") admissions.push_back(NNCacheAdmissionPolicy::Always);
      else if(parts[i] == "secondsighting") admissions.push_back(NNCacheAdmissionPolicy::SecondSighting);
      else throw StringError("benchnncachepolicy: -admissions accepts (always|secondsighting), got '" + parts[i] + "'");
    }
  }
  // ownership: which footprint each set is replayed with.
  vector<string> ownershipModes;
  if(ownershipStr == "both") { ownershipModes.push_back("off"); ownershipModes.push_back("on"); }
  else if(ownershipStr == "trace" || ownershipStr == "off" || ownershipStr == "on")
    ownershipModes.push_back(ownershipStr);
  else
    throw StringError("benchnncachepolicy: -ownership accepts (trace|off|on|both), got '" + ownershipStr + "'");

  //---- Read the trace ----
  cerr << "benchnncachepolicy: reading trace " << tracePath << endl;
  bool traceWasV1 = false;
  vector<NNCacheTraceRecord> trace =
    NNCacheTrace::readTrace(tracePath, (uint32_t)assumedV1Bytes, traceWasV1);
  if(maxOps > 0 && (int64_t)trace.size() > maxOps)
    trace.resize((size_t)maxOps);

  int64_t traceGets = 0, traceHits = 0, traceSets = 0, traceOwnerSets = 0, traceSetBytes = 0;
  std::set<uint64_t> distinctSetKeys;
  for(size_t i = 0; i<trace.size(); i++) {
    if((trace[i].flags & NNCacheTrace::FLAG_IS_SET) != 0) {
      traceSets += 1;
      traceSetBytes += trace[i].bytes;
      if(trace[i].bytes >= (uint32_t)(sizeof(NNOutput) + 19*19*sizeof(float)))
        traceOwnerSets += 1;
      distinctSetKeys.insert(trace[i].hash0 ^ trace[i].hash1);
    }
    else {
      traceGets += 1;
      if((trace[i].flags & NNCacheTrace::FLAG_HIT) != 0)
        traceHits += 1;
    }
  }
  const int64_t distinctKeys = (int64_t)distinctSetKeys.size();
  const int64_t meanSetBytes = traceSets > 0 ? traceSetBytes / traceSets : (int64_t)sizeof(NNOutput);

  //---- The memory preflight, before anything large is allocated ----
  const int64_t maxEntryBytes =
    ownershipStr == "off" ? (int64_t)sizeof(NNOutput)
    : (int64_t)(sizeof(NNOutput) + 19*19*sizeof(float));
  bool anyChain = false;
  for(size_t i = 0; i<schemes.size(); i++)
    if(schemes[i] == NNCacheCollisionScheme::Chain) anyChain = true;
  int minMutexPow = 30;
  for(size_t i = 0; i<tablePows.size(); i++)
    minMutexPow = std::min(minMutexPow, mutexPow >= 0 ? mutexPow : std::max(0,tablePows[i]-4));
  const int64_t effectiveChainBudget =
    chainBudget > 0 ? chainBudget : (((int64_t)1) << tablePows[tablePows.size()-1]) * meanSetBytes;
  const int64_t peak = worstCasePeakBytes(
    tablePows, minMutexPow, distinctKeys, maxEntryBytes, anyChain, effectiveChainBudget
  );
  cerr << "benchnncachepolicy: worst-case peak footprint of one configuration = "
       << (peak / (1024*1024)) << " MiB; budget = " << (maxBytesBudget / (1024*1024)) << " MiB" << endl;
  if(peak > maxBytesBudget)
    throw StringError(
      "benchnncachepolicy: REFUSING to start. The largest configuration in this matrix needs about " +
      Global::int64ToString(peak / (1024*1024)) + " MiB, and -max-bytes allows " +
      Global::int64ToString(maxBytesBudget / (1024*1024)) + " MiB.\n"
      "The arithmetic, so it can be checked rather than trusted: the biggest table size asked for is 2^" +
      Global::intToString(tablePows[tablePows.size()-1]) + " slots at up to 32 bytes a slot, plus at most " +
      Global::int64ToString(std::min((int64_t)1 << tablePows[tablePows.size()-1], distinctKeys)) +
      " resident payloads (the trace holds " + Global::int64ToString(distinctKeys) +
      " distinct keys, so a bigger table cannot hold more than that) at up to " +
      Global::int64ToString(maxEntryBytes) + " bytes each.\n"
      "This is refused rather than attempted because a run that swaps produces a lookup curve that looks\n"
      "like a memory-hierarchy effect and is not one, and because this class of box has already produced\n"
      "an OOM kill on this programme. Lower -table-pows, pass -ownership off, cut the trace with\n"
      "-max-ops, or raise -max-bytes if the machine really has the headroom."
    );

  //---- The substrate line ----
  int64_t pswpinBefore = -1, pswpoutBefore = -1;
  const bool swapReadable = readSwapCounters(pswpinBefore,pswpoutBefore);

  ofstream out(outPath.c_str());
  if(!out.good())
    throw StringError("benchnncachepolicy: could not open -out for writing: " + outPath);

  {
    JsonLine j;
    j.str("record","substrate");
    j.str("schema","katago-nncache-policy-sweep/1");
    j.str("written_at", DateTime::getDateString());
    j.str("host", readCommandOutput("hostname"));
    j.str("kernel", readCommandOutput("uname -r"));
    j.str("cpu_model", readCommandOutput("grep -m1 '^model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//'"));
    j.str("nproc", readCommandOutput("nproc"));
    j.str("meminfo_memtotal", readCommandOutput("grep -m1 MemTotal /proc/meminfo"));
    j.str("meminfo_memavailable", readCommandOutput("grep -m1 MemAvailable /proc/meminfo"));
    // THP: recorded because the hierarchy work measured it worth 17-28% INSIDE THIS
    // GUEST -- larger than doubling the table -- and nothing in KataGo controls it.
    // Whether that figure transfers to bare metal is a separate and open question: this
    // guest's memory is itself backed by host hugepages, so guest-level and host-level
    // page sizing are not the same question. Recorded so a reader can tell which state a
    // number was taken under; not interpreted here.
    j.str("thp_enabled", readFirstLine("/sys/kernel/mm/transparent_hugepage/enabled"));
    j.str("thp_defrag", readFirstLine("/sys/kernel/mm/transparent_hugepage/defrag"));
    j.str("hugepages_total", readCommandOutput("grep -m1 HugePages_Total /proc/meminfo"));
    j.str("ulimit_v", readCommandOutput("bash -c 'ulimit -v'"));
    j.str("ulimit_l", readCommandOutput("bash -c 'ulimit -l'"));
    j.str("loadavg", readFirstLine("/proc/loadavg"));
    j.str("idle_pct_before", readCommandOutput(
      "awk '/^cpu /{t=0;for(i=2;i<=NF;i++)t+=$i;printf \"%.1f\", 100*$5/t}' /proc/stat"
    ));
    j.num("pswpin_before", pswpinBefore);
    j.num("pswpout_before", pswpoutBefore);
    j.boolean("swap_counters_readable", swapReadable);
    j.str("katago_version", Version::getKataGoVersionFullInfo());
    j.str("git_revision", Version::getGitRevisionWithBackend());
    j.str("backend_that_built_the_trace", backendName);
    j.str("trace_path", tracePath);
    j.str("trace_format", traceWasV1 ? "v1-20byte-headerless" : "v2-24byte");
    j.boolean("trace_footprints_are_measured", !traceWasV1);
    if(traceWasV1)
      j.num("assumed_entry_bytes_for_v1", assumedV1Bytes);
    j.num("trace_records", (int64_t)trace.size());
    j.num("trace_gets", traceGets);
    j.num("trace_hits_as_recorded", traceHits);
    j.num("trace_sets", traceSets);
    j.num("trace_sets_with_ownermap", traceOwnerSets);
    j.num("trace_distinct_set_keys", distinctKeys);
    j.num("trace_mean_set_bytes", meanSetBytes);
    j.num("sizeof_NNOutput", (int64_t)sizeof(NNOutput));
    j.num("ownermap_bytes_19x19", (int64_t)(19*19*sizeof(float)));
    j.str("ownership_mode_arg", ownershipStr);
    j.num("max_bytes_budget", maxBytesBudget);
    j.num("preflight_peak_bytes", peak);
    j.num("chain_budget_bytes", effectiveChainBudget);
    j.boolean("chain_budget_was_derived", chainBudget <= 0);
    j.str("note", note);
    // Said in the file itself, because the file is the only thing a later reader has.
    j.str("what_this_file_is",
      "One NN-cache policy sweep. Each config line replays the SAME recorded cache operation stream "
      "through one cache configuration and reports what it held and what it hit. Occupancy is the "
      "quantity that makes a hit rate interpretable: a replacement policy has nothing to do in a "
      "nearly-empty table, so table size is swept as a means of reaching a range of occupancies, "
      "not as a question in itself. Visits/s is NOT here: it comes from 'katago benchmark' over a "
      "config carrying the same nnCache* keys, joined to these rows by configuration."
    );
    out << j.render() << "\n";
    out.flush();
  }

  //---- The sweep ----
  int64_t configsRun = 0;
  for(size_t oi = 0; oi<ownershipModes.size(); oi++) {
    const string ownMode = ownershipModes[oi];
    const bool forceOn = ownMode == "on";
    const bool forceOff = ownMode == "off";
    for(size_t ti = 0; ti<tablePows.size(); ti++) {
      const int tablePow = tablePows[ti];
      const int poolPow = mutexPow >= 0 ? mutexPow : std::max(0, tablePow-4);

      // Enumerate the matrix points valid at this table size.
      vector<PolicyPoint> points;
      for(size_t si = 0; si<schemes.size(); si++) {
        const NNCacheCollisionScheme scheme = schemes[si];
        for(size_t ai = 0; ai<admissions.size(); ai++) {
          const NNCacheAdmissionPolicy adm = admissions[ai];
          if(scheme == NNCacheCollisionScheme::Direct) {
            PolicyPoint p = {"direct", scheme, 1, NNCacheEvictionPolicy::None, false, adm};
            points.push_back(p);
          }
          else if(scheme == NNCacheCollisionScheme::Chain) {
            for(size_t ei = 0; ei<evictions.size(); ei++) {
              PolicyPoint p = {"chain", scheme, 1, evictions[ei], true, adm};
              points.push_back(p);
            }
          }
          else {
            for(size_t wi = 0; wi<waysList.size(); wi++) {
              for(size_t ei = 0; ei<evictions.size(); ei++) {
                PolicyPoint p = {schemeName(scheme), scheme, waysList[wi], evictions[ei], true, adm};
                points.push_back(p);
              }
            }
          }
        }
      }

      for(size_t pi = 0; pi<points.size(); pi++) {
        const PolicyPoint& pt = points[pi];

        // Build the config. A shape this table size cannot honor -- ways beyond a lock
        // region, a chain budget too small for one entry per region -- is REPORTED as a
        // refusal line rather than skipped, because a hole in a matrix that a reader
        // cannot tell from an unrun cell is worse than no matrix.
        string refusal;
        unique_ptr<NNCacheConfig> config;
        try {
          NNCacheShape shape = NNCacheShape::directMapped();
          if(pt.scheme == NNCacheCollisionScheme::Chain)
            shape = NNCacheShape::chained(effectiveChainBudget, pt.eviction);
          else if(pt.scheme != NNCacheCollisionScheme::Direct)
            shape = NNCacheShape::probed(pt.scheme, pt.ways, pt.eviction);
          NNCacheConfig c = {tablePow, poolPow, shape, pt.admission};
          config = unique_ptr<NNCacheConfig>(new NNCacheConfig(c));
        }
        catch(const StringError& e) { refusal = e.what(); }

        JsonLine j;
        j.str("record","config");
        j.str("collision", schemeName(pt.scheme));
        j.num("ways", pt.scheme == NNCacheCollisionScheme::Direct || pt.scheme == NNCacheCollisionScheme::Chain
              ? 1 : pt.ways);
        j.str("eviction", pt.hasEviction ? evictionName(pt.eviction) : "n/a-direct-is-1-way");
        j.str("admission", pt.admission == NNCacheAdmissionPolicy::Always ? "always" : "secondsighting");
        j.num("table_pow", tablePow);
        j.num("table_slots", ((int64_t)1) << tablePow);
        j.num("mutex_pool_pow", poolPow);
        j.str("ownership_mode", ownMode);

        if(!refusal.empty()) {
          j.str("status","REFUSED");
          j.str("refusal", refusal);
          out << j.render() << "\n";
          out.flush();
          continue;
        }

        ReplayResult r;
        try {
          r = replayOne(*config, trace, forceOn, forceOff);
        }
        catch(const StringError& e) {
          j.str("status","REFUSED_AT_CONSTRUCTION");
          j.str("refusal", e.what());
          out << j.render() << "\n";
          out.flush();
          continue;
        }

        j.str("status","OK");
        j.num("gets", r.gets);
        j.num("hits", r.hits);
        j.real("hit_rate", r.gets > 0 ? (double)r.hits / (double)r.gets : 0.0);
        j.num("sets_offered", r.sets);
        j.num("re_misses", r.reMisses);
        j.real("re_miss_rate_of_gets", r.gets > 0 ? (double)r.reMisses / (double)r.gets : 0.0);
        j.num("resident_entries", r.stats.residentEntries);
        j.num("resident_payload_bytes", r.stats.residentPayloadBytes);
        j.num("fixed_structure_bytes", r.stats.fixedStructureBytes);
        j.num("total_resident_bytes", r.stats.residentPayloadBytes + r.stats.fixedStructureBytes);
        j.num("capacity_slots", r.stats.capacitySlots);
        // 0 capacity means a chained table, which is bounded by bytes and has no slot
        // occupancy. Emitting a ratio there would be inventing a denominator.
        if(r.stats.capacitySlots > 0)
          j.real("occupancy", (double)r.stats.residentEntries / (double)r.stats.capacitySlots);
        else
          j.str("occupancy","n/a-chain-is-byte-bounded");
        j.real("bytes_per_resident_entry",
               r.stats.residentEntries > 0
               ? (double)r.stats.residentPayloadBytes / (double)r.stats.residentEntries : 0.0);
        j.real("seconds", r.seconds);
        j.real("cache_ops_per_sec", r.seconds > 0.0 ? (double)(r.gets + r.sets) / r.seconds : 0.0);
        out << j.render() << "\n";
        out.flush();
        configsRun += 1;
        cerr << "  [" << configsRun << "] " << schemeName(pt.scheme) << " ways=" << pt.ways
             << " " << (pt.hasEviction ? evictionName(pt.eviction) : "-")
             << " 2^" << tablePow << " own=" << ownMode
             << " hit=" << ((double)r.hits/(double)std::max((int64_t)1,r.gets))
             << " occ=" << (r.stats.capacitySlots > 0
                            ? (double)r.stats.residentEntries/(double)r.stats.capacitySlots : -1.0)
             << endl;
      }
    }
  }

  //---- The closing substrate line: the swap delta ACROSS THE WHOLE WINDOW ----
  {
    int64_t pswpinAfter = -1, pswpoutAfter = -1;
    readSwapCounters(pswpinAfter,pswpoutAfter);
    JsonLine j;
    j.str("record","substrate_after");
    j.num("pswpin_after", pswpinAfter);
    j.num("pswpout_after", pswpoutAfter);
    j.num("pswpin_delta", swapReadable && pswpinAfter >= 0 ? pswpinAfter - pswpinBefore : -1);
    j.num("pswpout_delta", swapReadable && pswpoutAfter >= 0 ? pswpoutAfter - pswpoutBefore : -1);
    j.str("idle_pct_after", readCommandOutput(
      "awk '/^cpu /{t=0;for(i=2;i<=NF;i++)t+=$i;printf \"%.1f\", 100*$5/t}' /proc/stat"
    ));
    j.num("configs_run", configsRun);
    j.str("how_to_read_the_swap_delta",
      "Sampled across the WHOLE sweep, not spot-checked inside it. A nonzero pswpin_delta means "
      "some measured window stalled on a swap read and every cache_ops_per_sec figure in this file "
      "is disqualified; hit_rate, occupancy and the byte figures are NOT timing quantities and "
      "survive it. A nonzero pswpout_delta with pswpin_delta zero is DEGRADED, not disqualified: "
      "reclaim stole cycles but no measured loop stalled on a read."
    );
    out << j.render() << "\n";
  }
  out.close();
  cerr << "benchnncachepolicy: wrote " << configsRun << " configuration rows to " << outPath << endl;
  return 0;
}
