#ifndef NEURALNET_NNCACHEIMPL_H_
#define NEURALNET_NNCACHEIMPL_H_

#include <memory>

#include "../neuralnet/nncache.h"

// The non-default cache shapes, reached only through these factories.
//
// Each shape lives in its own translation unit rather than piling into nncache.cpp:
// the probed table, the chained table and the admission filter own three unrelated
// concerns (associativity plus victim choice; a byte budget; whether an entry is
// stored at all), and one file per owner is the shape ADR-0012 P3 asks for.
// nncache.cpp keeps the vocabulary, the config Port, the footprint accounting, the
// direct-mapped table, and the seam that composes these.

// An open-addressed table: config.shape.ways() candidate slots per key, reached by
// config.shape.scheme()'s probe sequence, giving up config.shape.eviction()'s choice
// of victim when all of them are taken. Throws if the shape cannot be honored --
// notably if ways exceeds the number of slots one lock region holds.
std::unique_ptr<NNCacheTable> makeProbedNNCacheTable(const NNCacheConfig& config);

// Separate chaining with no collision-driven eviction at all, held instead to
// config.shape.maxBytes() of resident cache content. Throws if the budget is too
// small to hold even one entry per lock region.
std::unique_ptr<NNCacheTable> makeChainedNNCacheTable(const NNCacheConfig& config);

// Wraps a table so that a key is stored only from its second sighting onward. The
// keys seen once are remembered in a fixed-size, payload-free ghost table sized from
// sizePowerOfTwo; it is allocated once and never grows.
std::unique_ptr<NNCacheTable> makeSecondSightingNNCacheTable(
  std::unique_ptr<NNCacheTable> inner,
  int sizePowerOfTwo
);

// A 1-way direct-mapped table whose collisions are resolved by config.shape.replacement()
// rather than by unconditionally taking the newcomer. Reached ONLY when that rule is not
// Always -- the shipped NNCacheTableDirect is untouched and still serves the default -- so
// nothing on the default get/set path pays for this axis existing.
//
// Throws if the shape is not direct-mapped, or if it is direct-mapped with the Always
// rule, which is NNCacheTableDirect's job and not this one's.
std::unique_ptr<NNCacheTable> makeSightingDirectNNCacheTable(const NNCacheConfig& config);

// The exact resident cost of the sighting-count ghost table the two count-comparing
// replacement rules allocate. Named here so the bound is stated in one place and can be
// asserted rather than estimated (ADR-0012 P1).
//
// It is numerically the same 4 bytes per slot as secondSightingGhostBytes below, and it is
// deliberately a SEPARATE function rather than a call to that one: the two ghosts hold
// different facts (a one-bit "has been stored before" against an eight-bit presentation
// count) and a table that selects both policies allocates both. Sharing the arithmetic
// would make a future change to either silently change the other.
size_t sightingCountGhostBytes(int sizePowerOfTwo);

// The per-region share of a chained table's byte budget, and what one entry costs it.
// Both re-export the chained table's own arithmetic so a caller or a test asserts
// against the implementation rather than against a second copy of it (ADR-0012 P1).
int64_t chainedRegionBudgetBytes(int64_t maxBytes, int sizePowerOfTwo, int mutexPoolSizePowerOfTwo);
size_t chainedEntryBytes(const NNOutput& out);

// The exact resident cost of the ghost table the wrapper above allocates. Named here
// so the bound is stated in one place and can be asserted rather than estimated
// (ADR-0012 P1).
size_t secondSightingGhostBytes(int sizePowerOfTwo);

#endif  // NEURALNET_NNCACHEIMPL_H_
