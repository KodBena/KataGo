#ifndef TESTS_NNCACHETABLETESTACCESS_H_
#define TESTS_NNCACHETABLETESTACCESS_H_

#include <memory>

#include "../core/hash.h"
#include "../neuralnet/nncache.h"

// THE ONE DOOR every existing get/set-driven unit test reaches NNCacheTable's raw, hash-keyed
// get(Hash128,...)/set(shared_ptr<NNOutput>,...) through, now that those two are protected
// (nncache.h narrows them so production code outside the presentation-minted
// get(NNCachePresentation,...)/set(NNCachePresentation,...,attribution) path cannot call
// them). ONE named friend rather than one per test file or per test: every test that wants
// the raw surface forwards through these two static wrappers, and nothing else outside the
// NNCacheTable class hierarchy gets the same door. Mirrors NNCacheTableBenchAccess, the
// production side's own narrow exception, in cpp/command/benchnncachepolicy.cpp.
//
// Test-tree only BY CONVENTION AND REVIEW, NOT BY BUILD ENFORCEMENT -- named honestly, in the
// same terms cpp/tests/testcacheswapseam.h already uses for its own test seam: this repo
// compiles every test .cpp and every production .cpp into the one `katago` CMake target, with
// no per-target include isolation, so nothing in the build stops a production translation unit
// from writing `#include "../tests/nncachetabletestaccess.h"` and reaching the raw surface
// through it. That is deliberately left as a review surface rather than closed further (an
// out-of-frame audit of the change that added this file raised exactly this): it is one
// greppable, obviously-out-of-place include line in a file that has no business reaching into
// tests/, not a relaxation of the boundary itself.

class NNCacheTableTestAccess {
 public:
  static bool get(NNCacheTable& table, Hash128 nnHash, std::shared_ptr<NNOutput>& ret) {
    return table.get(nnHash, ret);
  }
  static void set(NNCacheTable& table, const std::shared_ptr<NNOutput>& p) {
    table.set(p);
  }
};

#endif  // TESTS_NNCACHETABLETESTACCESS_H_
