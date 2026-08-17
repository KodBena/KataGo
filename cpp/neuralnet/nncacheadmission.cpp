#include "../neuralnet/nncacheimpl.h"

#include <atomic>

using namespace std;

// Second-sighting admission.
//
// A key offered to the table for the first time is NOT stored; it is only remembered.
// A key offered again is stored. This is the in-memory form of the filter the
// operator's own persisted-cache prototype already applies in SQL as
// `WHERE num_refs > 1`, whose trade was measured on his real data at 44.787% of
// entries discarded for 7.902% of lifetime references.
//
// It is a DECORATOR rather than a branch inside each table because admission is
// orthogonal to collision resolution: writing it once covers direct mapping, both
// probe schemes and chaining, and none of those four has to know it exists.
//
// ---------------------------------------------------------------------------
// The ghost set and its memory bound
// ---------------------------------------------------------------------------
// The keys seen once live in a direct-mapped array of 32-bit tags with NO payloads
// and no chaining, so the bound is exact, fixed at construction, and never grows:
//
//     4 bytes * 2^nnCacheSizePowerOfTwo
//
// which is 8 MiB at the operator's k=21 and 32 MiB at k=23. It is derived from a knob
// that already exists rather than introducing a new one, and it is a quarter of the
// slot-pointer array it sits beside -- against a payload of ~2972 bytes per resident
// entry it is about 0.13% of what the cache costs.
//
// A ghost slot is indexed by hash1 and tagged from the high half of hash0, so neither
// quantity collides with the low bits of hash0 that the cache tables use for their own
// index. The tag is forced nonzero so that "never written" stays distinguishable from
// a real key; that costs one bit, i.e. a 2^-31 chance of admitting a key on its first
// sighting because a different key happened to leave a matching tag.
//
// Two deliberate imprecisions, both of them hit-rate effects and neither of them a
// correctness effect:
//
//   * The array is read and written with RELAXED atomics and no lock. Two threads
//     racing on one ghost slot can lose an update, which costs at most one admission.
//     Taking a lock to protect a hint would be paying for a guarantee the structure
//     does not need.
//   * The tag is NOT cleared when a key is admitted. A key that is admitted, later
//     evicted, and offered again is therefore re-admitted immediately rather than
//     having to be sighted twice more. Clearing it would make an entry that the cache
//     itself dropped harder to get back than a brand-new one, which is backwards.

namespace {

class NNCacheTableSecondSighting final : public NNCacheTable {
  std::unique_ptr<NNCacheTable> inner;
  std::atomic<uint32_t>* ghost;
  uint64_t ghostMask;

 public:
  NNCacheTableSecondSighting(std::unique_ptr<NNCacheTable> innerArg, int sizePowerOfTwo)
    :inner(std::move(innerArg)), ghost(NULL), ghostMask(0)
  {
    if(sizePowerOfTwo < 0 || sizePowerOfTwo > 63)
      throw StringError("NNCacheTable: Invalid sizePowerOfTwo: " + Global::intToString(sizePowerOfTwo));
    const uint64_t numGhostSlots = ((uint64_t)1) << sizePowerOfTwo;
    ghostMask = numGhostSlots - 1;
    ghost = new std::atomic<uint32_t>[numGhostSlots]();
  }

  ~NNCacheTableSecondSighting() override {
    delete[] ghost;
  }

  // A read is unaffected: admission decides what gets stored, never what gets found.
  bool get(Hash128 nnHash, std::shared_ptr<NNOutput>& ret) override {
    return inner->get(nnHash,ret);
  }

  void set(const std::shared_ptr<NNOutput>& p) override {
    const Hash128 nnHash = p->nnHash;
    const uint64_t idx = nnHash.hash1 & ghostMask;
    const uint32_t tag = (uint32_t)(nnHash.hash0 >> 32) | 1u;
    if(ghost[idx].load(std::memory_order_relaxed) == tag) {
      inner->set(p);
      return;
    }
    ghost[idx].store(tag, std::memory_order_relaxed);
  }

  // The inner table's own snapshot, with the ghost array added to the fixed cost. The
  // ghost is real resident memory the operator pays for and it belongs to no other
  // layer, so leaving it out would make the admission filter look free.
  NNCacheStats stats() const override {
    NNCacheStats s = inner->stats();
    s.fixedStructureBytes += (int64_t)(sizeof(std::atomic<uint32_t>) * (ghostMask + 1));
    return s;
  }

  void clear() override {
    inner->clear();
    for(uint64_t i = 0; i <= ghostMask; i++)
      ghost[i].store(0, std::memory_order_relaxed);
  }
};

}  // namespace

size_t secondSightingGhostBytes(int sizePowerOfTwo) {
  if(sizePowerOfTwo < 0 || sizePowerOfTwo > 63)
    throw StringError("secondSightingGhostBytes: Invalid sizePowerOfTwo: " + Global::intToString(sizePowerOfTwo));
  return sizeof(std::atomic<uint32_t>) * (((size_t)1) << sizePowerOfTwo);
}

unique_ptr<NNCacheTable> makeSecondSightingNNCacheTable(unique_ptr<NNCacheTable> inner, int sizePowerOfTwo) {
  if(inner == nullptr)
    throw StringError("makeSecondSightingNNCacheTable: no inner table to admit into.");
  return unique_ptr<NNCacheTable>(new NNCacheTableSecondSighting(std::move(inner), sizePowerOfTwo));
}
