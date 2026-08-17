#include "../neuralnet/nncacheimpl.h"

#include "../neuralnet/nncacheprobed.h"

using namespace std;
using namespace NNCacheProbed;

// The shipped instantiations: two probe sequences times three eviction policies,
// always with the inline tag. The UseTag=false variant exists only in the benchmark
// translation unit, which is the whole reason NNCacheTableProbed lives in a header.

namespace {

template<class Probe>
unique_ptr<NNCacheTable> makeWithProbe(const NNCacheConfig& config) {
  const int size = config.sizePowerOfTwo;
  const int pool = config.mutexPoolSizePowerOfTwo;
  const int ways = config.shape.ways();
  switch(config.shape.eviction()) {
  case NNCacheEvictionPolicy::Random:
    return unique_ptr<NNCacheTable>(new NNCacheTableProbed<Probe,RandomEviction,true>(size,pool,ways));
  case NNCacheEvictionPolicy::Lru:
    return unique_ptr<NNCacheTable>(new NNCacheTableProbed<Probe,LruEviction,true>(size,pool,ways));
  case NNCacheEvictionPolicy::Lfu:
    return unique_ptr<NNCacheTable>(new NNCacheTableProbed<Probe,LfuEviction,true>(size,pool,ways));
  case NNCacheEvictionPolicy::None:
    break;
  }
  // NNCacheShape::probed already refuses None, so reaching here means the shape type
  // was bypassed. Refuse rather than substitute a policy the operator did not ask for.
  throw StringError(
    "makeProbedNNCacheTable: eviction policy '" + string(NNCacheConfig::KEY_EVICTION) +
    "' has no probed implementation; a probed table must choose a victim."
  );
}

}  // namespace

unique_ptr<NNCacheTable> makeProbedNNCacheTable(const NNCacheConfig& config) {
  switch(config.shape.scheme()) {
  case NNCacheCollisionScheme::LinearProbe: return makeWithProbe<LinearProbe>(config);
  case NNCacheCollisionScheme::QuadraticProbe: return makeWithProbe<QuadraticProbe>(config);
  case NNCacheCollisionScheme::Direct:
  case NNCacheCollisionScheme::Chain:
    break;
  }
  throw StringError(
    "makeProbedNNCacheTable: '" + config.shape.toString() + "' is not a probed shape."
  );
}
