// gen_vectors.cpp -- generates the frozen conformance vectors under cpp/spec/chd/vectors/
// by RUNNING the archived prototype's construction and lookup code, not by transcribing it.
//
// WHY THIS FILE INCLUDES A PATH OUTSIDE THE REPOSITORY
// ---------------------------------------------------
// The single #include below reaches into /home/bork/kata_fork_archived, which the
// implementer of the new CHD structure is forbidden to open. That is deliberate and it is
// the whole point of the arrangement: the vectors in cpp/spec/chd/vectors/ are the
// *observed behaviour* of that code, and this generator is the instrument that observed it.
// No prototype source is copied into this repository. Anyone re-running this generator
// needs read access to the archive; anyone merely *checking* an implementation against the
// vectors does not, and must not.
//
// The observation site is the lookup predicate itself (ADR-0021 Rule 1): this file computes
// each expected answer through the prototype's own query function plus the stored-key
// verification that its cache wrapper applies, which is exactly where the member/absent
// decision is made. It does not re-derive that decision from a reading of the code.
//
// Build: see build.sh in this directory.

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cctype>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <algorithm>

#include "../../../core/hash.h"

// The archived prototype's construction and query code, unmodified, included from the
// read-only archive. sha256 of that file at generation time is recorded in MANIFEST.tsv.
#include "/home/bork/kata_fork_archived/cpp/neuralnet/chd_hash.h"

// ---------------------------------------------------------------------------------------
// Deterministic key generation. splitmix64, so a reader can regenerate any synthetic
// corpus from its named seed without this program.
// ---------------------------------------------------------------------------------------
static uint64_t sm64(uint64_t& s) {
  uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

static std::vector<Hash128> synthKeys(uint64_t seed, size_t n) {
  uint64_t s = seed;
  std::vector<Hash128> v;
  v.reserve(n);
  for(size_t i = 0; i < n; i++) {
    uint64_t a = sm64(s);
    uint64_t b = sm64(s);
    v.push_back(Hash128(a, b));
  }
  return v;
}

static std::string keyHex(const Hash128& h) {
  char buf[40];
  snprintf(buf, sizeof(buf), "%016llx %016llx",
           (unsigned long long)h.hash0, (unsigned long long)h.hash1);
  return std::string(buf);
}

// ---------------------------------------------------------------------------------------
// The observation. This mirrors the membership decision the prototype's frozen-cache
// wrapper makes: resolve the key through the perfect-hash query, then verify the stored key
// at the resolved position before believing the answer.
// ---------------------------------------------------------------------------------------
struct Answer {
  bool member;
  uint32_t index;      // valid iff member
  bool landedOccupied; // diagnostic: the query resolved to a position holding a real entry
};

static Answer observe(const Hash128& q, const std::vector<Hash128>& keys, const CHDTable& t) {
  Answer a{false, 0, false};
  if(t.table_size == 0) return a;
  uint32_t idx = query_chd_table(q, t);
  a.landedOccupied = (idx < keys.size());
  if(idx < keys.size() && keys[idx] == q) {
    a.member = true;
    a.index = idx;
  }
  return a;
}

// ---------------------------------------------------------------------------------------
// Case emission
// ---------------------------------------------------------------------------------------
static std::string OUTDIR;

struct CaseStat {
  std::string name;
  std::string outcome;
  size_t nkeys = 0, nqueries = 0, nmember = 0, nabsent = 0, nabsentOccupied = 0;
  uint32_t bucket_count = 0, table_size = 0;
  std::string note;
};
static std::vector<CaseStat> STATS;

static void writeLines(const std::string& path, const std::vector<std::string>& lines) {
  std::ofstream f(path, std::ios::binary);
  if(!f) throw std::runtime_error("cannot write " + path);
  for(const std::string& l : lines) f << l << "\n";
}

// Emits a BUILD_OK case: keys, queries, expected answers.
static void emitOk(const std::string& name,
                   const std::string& note,
                   const std::vector<Hash128>& keys,
                   const std::vector<Hash128>& queries) {
  CHDTable t = build_chd_table(keys);

  std::vector<std::string> keyLines, qLines, eLines;
  for(const Hash128& k : keys) keyLines.push_back(keyHex(k));
  CaseStat st;
  st.name = name; st.outcome = "BUILD_OK"; st.note = note;
  st.nkeys = keys.size();
  st.bucket_count = t.bucket_count; st.table_size = t.table_size;

  for(const Hash128& q : queries) {
    Answer a = observe(q, keys, t);
    qLines.push_back(keyHex(q));
    if(a.member) {
      char buf[32]; snprintf(buf, sizeof(buf), "MEMBER %u", a.index);
      eLines.push_back(buf);
      st.nmember++;
    } else {
      eLines.push_back("ABSENT");
      st.nabsent++;
      if(a.landedOccupied) st.nabsentOccupied++;
    }
  }
  st.nqueries = queries.size();

  writeLines(OUTDIR + "/" + name + ".keys", keyLines);
  writeLines(OUTDIR + "/" + name + ".queries", qLines);
  writeLines(OUTDIR + "/" + name + ".expected", eLines);
  {
    std::ofstream f(OUTDIR + "/" + name + ".meta", std::ios::binary);
    f << "name\t" << name << "\n";
    f << "outcome\tBUILD_OK\n";
    f << "n_keys\t" << st.nkeys << "\n";
    f << "n_queries\t" << st.nqueries << "\n";
    f << "n_expect_member\t" << st.nmember << "\n";
    f << "n_expect_absent\t" << st.nabsent << "\n";
    f << "note\t" << note << "\n";
  }
  STATS.push_back(st);
  fprintf(stderr, "[case] %-28s BUILD_OK n=%zu q=%zu member=%zu absent=%zu absent_hit_occupied_slot=%zu\n",
          name.c_str(), st.nkeys, st.nqueries, st.nmember, st.nabsent, st.nabsentOccupied);
}

// Emits a BUILD_FAIL case: construction must be refused, loudly.
// The observation is the throw firing (ADR-0021 Rule 2: a negative claim gets a tripwire
// whose FIRING is the observation, not a downstream symptom).
static void emitFail(const std::string& name,
                     const std::string& note,
                     const std::vector<Hash128>& keys) {
  bool threw = false;
  std::string what;
  try {
    CHDTable t = build_chd_table(keys);
    (void)t;
  } catch(const std::exception& e) {
    threw = true;
    what = e.what();
  }
  if(!threw) {
    fprintf(stderr, "[case] %-28s EXPECTED A REFUSAL AND DID NOT GET ONE -- aborting\n", name.c_str());
    throw std::runtime_error("case " + name + " did not refuse construction");
  }
  std::vector<std::string> keyLines;
  for(const Hash128& k : keys) keyLines.push_back(keyHex(k));
  writeLines(OUTDIR + "/" + name + ".keys", keyLines);
  {
    std::ofstream f(OUTDIR + "/" + name + ".meta", std::ios::binary);
    f << "name\t" << name << "\n";
    f << "outcome\tBUILD_FAIL\n";
    f << "n_keys\t" << keys.size() << "\n";
    f << "n_queries\t0\n";
    f << "note\t" << note << "\n";
  }
  CaseStat st; st.name = name; st.outcome = "BUILD_FAIL"; st.nkeys = keys.size(); st.note = note;
  STATS.push_back(st);
  fprintf(stderr, "[case] %-28s BUILD_FAIL refused: %s\n", name.c_str(), what.c_str());
}

// ---------------------------------------------------------------------------------------
// Query-set builders
// ---------------------------------------------------------------------------------------

// Every resident key, in input order.
static std::vector<Hash128> allMembers(const std::vector<Hash128>& keys) { return keys; }

// Absent keys generated from an independent stream, filtered against the resident set.
static std::vector<Hash128> absentRandom(const std::vector<Hash128>& keys, uint64_t seed, size_t n) {
  std::set<std::pair<uint64_t,uint64_t>> res;
  for(const Hash128& k : keys) res.insert({k.hash0, k.hash1});
  std::vector<Hash128> out;
  uint64_t s = seed;
  while(out.size() < n) {
    uint64_t a = sm64(s), b = sm64(s);
    if(res.count({a,b})) continue;
    out.push_back(Hash128(a,b));
  }
  return out;
}

// One-bit perturbations of resident keys: the nastiest absent keys, because they differ
// from a real member in a single bit and a lookup that skips key verification will often
// still land on that member's position.
static std::vector<Hash128> absentOneBitFlips(const std::vector<Hash128>& keys, size_t maxOut) {
  std::set<std::pair<uint64_t,uint64_t>> res;
  for(const Hash128& k : keys) res.insert({k.hash0, k.hash1});
  std::vector<Hash128> out;
  for(size_t i = 0; i < keys.size() && out.size() < maxOut; i++) {
    for(int bit = 0; bit < 128 && out.size() < maxOut; bit += 17) {
      Hash128 k = keys[i];
      if(bit < 64) k.hash0 ^= (1ULL << bit); else k.hash1 ^= (1ULL << (bit - 64));
      if(res.count({k.hash0, k.hash1})) continue;
      out.push_back(k);
    }
  }
  return out;
}

static std::vector<Hash128> concat(std::vector<Hash128> a, const std::vector<Hash128>& b) {
  a.insert(a.end(), b.begin(), b.end());
  return a;
}

// ---------------------------------------------------------------------------------------
// Real-key corpora, read from files the caller produced with psql (read-only).
// Format: one row per line, the 128-bit hash_id in hex.
// ---------------------------------------------------------------------------------------
static std::vector<Hash128> readRealKeys(const std::string& path) {
  std::ifstream f(path);
  if(!f) throw std::runtime_error("cannot read real key file " + path);
  std::vector<Hash128> out;
  std::string line;
  while(std::getline(f, line)) {
    std::string h;
    for(char c : line) if(isxdigit((unsigned char)c)) h.push_back((char)tolower(c));
    if(h.size() < 32) continue;
    h = h.substr(h.size() - 32);
    uint64_t hi = std::stoull(h.substr(0,16), nullptr, 16);
    uint64_t lo = std::stoull(h.substr(16,16), nullptr, 16);
    out.push_back(Hash128(hi, lo));
  }
  return out;
}

int main(int argc, char** argv) {
  if(argc < 2) { fprintf(stderr, "usage: gen_vectors <outdir> [realkeyfile ...]\n"); return 2; }
  OUTDIR = argv[1];

  // --- degenerate and small sizes -------------------------------------------------------
  {
    std::vector<Hash128> empty;
    emitOk("n0_empty", "empty key set; every lookup must report absent",
           empty, absentRandom(empty, 0xE1, 8));
  }
  for(size_t n : {size_t(1), size_t(2), size_t(3), size_t(4), size_t(5), size_t(8), size_t(9),
                  size_t(16), size_t(17)}) {
    std::vector<Hash128> keys = synthKeys(0x5EED0000ULL + n, n);
    std::vector<Hash128> q = concat(allMembers(keys),
                                    concat(absentRandom(keys, 0xA000 + n, 64),
                                           absentOneBitFlips(keys, 64)));
    char nm[64]; snprintf(nm, sizeof(nm), "n%zu_small", n);
    emitOk(nm, "small key set spanning the four-keys-per-bucket boundary", keys, q);
  }

  // --- mid sizes -------------------------------------------------------------------------
  for(size_t n : {size_t(100), size_t(1000), size_t(10000)}) {
    std::vector<Hash128> keys = synthKeys(0xBEEF0000ULL + n, n);
    std::vector<Hash128> q = concat(allMembers(keys),
                                    concat(absentRandom(keys, 0xC000 + n, n),
                                           absentOneBitFlips(keys, n)));
    char nm[64]; snprintf(nm, sizeof(nm), "n%zu_uniform", n);
    emitOk(nm, "uniform 128-bit keys; queries are all members, an equal mass of random "
               "absent keys, and an equal mass of one-bit perturbations of members", keys, q);
  }

  // --- absent-key stress: a corpus queried entirely with foreign keys --------------------
  {
    std::vector<Hash128> keys = synthKeys(0x1111, 5000);
    std::vector<Hash128> foreign = synthKeys(0x2222, 20000); // independent stream
    std::set<std::pair<uint64_t,uint64_t>> res;
    for(const Hash128& k : keys) res.insert({k.hash0,k.hash1});
    std::vector<Hash128> q;
    for(const Hash128& f : foreign) if(!res.count({f.hash0,f.hash1})) q.push_back(f);
    emitOk("absent_foreign_corpus",
           "every query is a key from an independently generated corpus; all must be absent",
           keys, q);
  }
  {
    std::vector<Hash128> keys = synthKeys(0x3333, 5000);
    emitOk("absent_onebit_only",
           "every query is a member with exactly one bit flipped; all must be absent",
           keys, absentOneBitFlips(keys, 20000));
  }

  // --- structured key sets ---------------------------------------------------------------
  {
    // One shared-hash1 pair, deliberately placed in two different buckets. Legal: the
    // distinctness requirement is per bucket, not global.
    std::vector<Hash128> keys = synthKeys(0x7770000, 1000);
    keys[10]  = Hash128(0x0000000000000000ULL, 0xDEADBEEFCAFEBABEULL); // lowest bucket
    keys[500] = Hash128(0xFFFFFFFFFFFFFFFFULL, 0xDEADBEEFCAFEBABEULL); // highest bucket
    emitOk("shared_hash1_distinct_buckets",
           "one pair of distinct keys sharing a hash1 value, placed at opposite ends of the "
           "hash0 range so they fall in different buckets; construction must succeed",
           keys, concat(allMembers(keys), absentRandom(keys, 0x778, 1000)));
  }
  {
    // Distinct keys sharing hash0, therefore all in one bucket.
    std::vector<Hash128> keys;
    uint64_t s = 0x888;
    for(size_t i = 0; i < 12; i++) keys.push_back(Hash128(0x0123456789ABCDEFULL, sm64(s)));
    emitOk("shared_hash0_one_bucket",
           "12 distinct keys sharing one hash0, so all land in the same bucket",
           keys, concat(allMembers(keys), absentRandom(keys, 0x889, 256)));
  }
  {
    std::vector<Hash128> keys = {
      Hash128(0,0),
      Hash128(0, 1),
      Hash128(1, 2),
      Hash128(UINT64_MAX, UINT64_MAX),
      Hash128(UINT64_MAX, 3),
      Hash128(0, UINT64_MAX - 1),
      Hash128(0x8000000000000000ULL, 0x8000000000000000ULL),
      Hash128(0x7FFFFFFFFFFFFFFFULL, 0x7FFFFFFFFFFFFFFFULL),
    };
    emitOk("extremal_keys",
           "all-zero, all-ones and sign-bit-boundary key values, with hash1 values kept "
           "pairwise distinct so the per-bucket precondition holds",
           keys, concat(allMembers(keys), absentRandom(keys, 0x999, 256)));
  }

  // --- refusals ---------------------------------------------------------------------------
  {
    std::vector<Hash128> keys = synthKeys(0xD00D, 100);
    keys.push_back(keys[7]); // exact duplicate
    emitFail("dup_exact",
             "the key at input position 7 appears a second time at the end; construction "
             "must be refused", keys);
  }
  {
    // Two distinct keys with identical hash1 in the same bucket. With n=2 there is one
    // bucket, so they necessarily collide for every seed.
    std::vector<Hash128> keys = {
      Hash128(0x0000000000000001ULL, 0xAAAAAAAAAAAAAAAAULL),
      Hash128(0x0000000000000002ULL, 0xAAAAAAAAAAAAAAAAULL),
    };
    emitFail("dup_hash1_same_bucket",
             "two DISTINCT keys sharing a hash1 value and landing in the same bucket; "
             "construction must be refused", keys);
  }
  {
    // The same defect at scale: 64 distinct keys all sharing one hash1 value. With four
    // keys per bucket the pigeonhole guarantees bucket-mates, so this is refused too.
    std::vector<Hash128> keys;
    uint64_t s = 0x777;
    for(size_t i = 0; i < 64; i++) keys.push_back(Hash128(sm64(s), 0xDEADBEEFCAFEBABEULL));
    emitFail("hash1_degenerate_corpus",
             "64 DISTINCT keys all sharing one hash1 value; bucket-mates are unavoidable and "
             "construction must be refused", keys);
  }

  // --- real corpora -----------------------------------------------------------------------
  for(int i = 2; i < argc; i++) {
    std::string path = argv[i];
    std::vector<Hash128> keys = readRealKeys(path);
    std::string base = path.substr(path.find_last_of('/') + 1);
    size_t dot = base.find_last_of('.');
    if(dot != std::string::npos) base = base.substr(0, dot);
    size_t nq = std::min<size_t>(keys.size(), 20000);
    std::vector<Hash128> q = concat(allMembers(keys),
                                    concat(absentRandom(keys, 0xF00D + i, nq),
                                           absentOneBitFlips(keys, nq)));
    emitOk(base,
           "real nnHash corpus from the operator's database, in the order the prototype's "
           "SQL selection delivers it (num_refs descending)",
           keys, q);
  }

  // --- diagnostic sidecar (NOT acceptance) --------------------------------------------------
  {
    std::ofstream f(OUTDIR + "/DIAGNOSTIC-internals.tsv", std::ios::binary);
    f << "# NOT AN ACCEPTANCE ARTIFACT. These are internals of the archived prototype's\n"
      << "# construction, recorded only as a debugging aid. A conforming implementation is\n"
      << "# free to differ in every column here. Nothing checks these.\n"
      << "case\toutcome\tn_keys\tproto_bucket_count\tproto_table_size\tn_queries\t"
         "n_expect_member\tn_expect_absent\tabsent_landing_on_occupied_slot\n";
    for(const CaseStat& s : STATS) {
      f << s.name << "\t" << s.outcome << "\t" << s.nkeys << "\t"
        << s.bucket_count << "\t" << s.table_size << "\t" << s.nqueries << "\t"
        << s.nmember << "\t" << s.nabsent << "\t" << s.nabsentOccupied << "\n";
    }
  }
  fprintf(stderr, "[gen] %zu cases written to %s\n", STATS.size(), OUTDIR.c_str());
  return 0;
}
