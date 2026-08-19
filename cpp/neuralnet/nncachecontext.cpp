#include "../neuralnet/nncachecontext.h"

#include <atomic>
#include <mutex>

#include "../core/global.h"
#include "../neuralnet/nncachefileformat.h"
#include "../search/mutexpool.h"

// See nncachecontext.h for what a context is and, more importantly, for what this engine
// refuses to understand about one.

using namespace std;

namespace {

// Distinct for every set constructed in this process, so an id minted by one set is
// recognisable as foreign to another. It is a process-local identity and is deliberately not
// persisted anywhere: nothing on disk keys off it, and a restarted engine's sets are new.
uint64_t nextContextSetId() {
  static std::atomic<uint64_t> counter(1);
  return counter.fetch_add(1, std::memory_order_relaxed);
}

string quotedList(const vector<string>& names) {
  if(names.empty())
    return "none";
  string out;
  for(size_t i = 0; i < names.size(); i++) {
    if(i > 0)
      out += ", ";
    out += "'" + names[i] + "'";
  }
  return out;
}

}  // namespace

//-------------------------------------------------------------------------------------
// The attribution
//-------------------------------------------------------------------------------------

NNCacheAttribution::NNCacheAttribution() : id_() {}
NNCacheAttribution::NNCacheAttribution(std::optional<NNCacheContextId> id) : id_(id) {}

NNCacheAttribution NNCacheAttribution::toContext(NNCacheContextId id) {
  return NNCacheAttribution(std::optional<NNCacheContextId>(id));
}

NNCacheAttribution NNCacheAttribution::noAttributableContext() {
  return NNCacheAttribution(std::optional<NNCacheContextId>());
}

NNCacheContextId NNCacheAttribution::contextId() const {
  if(!id_.has_value())
    throw StringError(
      "NNCacheAttribution: this entry has no attributable context, so there is no context id "
      "to hand out. Check isToContext() first; a fabricated id would file the entry under a "
      "context nothing said it belonged to."
    );
  return id_.value();
}

//-------------------------------------------------------------------------------------
// The resolution
//-------------------------------------------------------------------------------------

NNCacheContextResolution::NNCacheContextResolution(
  std::optional<NNCacheAttribution> attribution, std::optional<string> refusal
)
  :attribution_(std::move(attribution)), refusal_(std::move(refusal))
{}

NNCacheContextResolution NNCacheContextResolution::resolved(NNCacheAttribution attribution) {
  return NNCacheContextResolution(std::optional<NNCacheAttribution>(attribution), std::optional<string>());
}

NNCacheContextResolution NNCacheContextResolution::refused(string message) {
  return NNCacheContextResolution(std::optional<NNCacheAttribution>(), std::optional<string>(std::move(message)));
}

//-------------------------------------------------------------------------------------
// The name space
//-------------------------------------------------------------------------------------

NNCacheContextSet::NNCacheContextSet() : setId_(nextContextSetId()), names_() {}

NNCacheContextId NNCacheContextSet::attach(const string& name) {
  // The same closed-alphabet boundary the two file formats validate their path components
  // at, read from its one home rather than re-authored here (ADR-0012 P1).
  NNCacheFileName::verify(name, "NNCacheContextSet", "context name");
  for(size_t i = 0; i < names_.size(); i++) {
    if(names_[i] == name)
      throw StringError(
        "NNCacheContextSet: context '" + name + "' is already attached. Two attachments under "
        "one name have no single answer to which of them an entry was earned by, and picking "
        "either files the entry under a context the client did not mean."
      );
  }
  names_.push_back(name);
  return NNCacheContextId(setId_, names_.size() - 1);
}

bool NNCacheContextSet::owns(const NNCacheContextId& id) const {
  return id.setId() == setId_ && id.index() < names_.size();
}

const string& NNCacheContextSet::nameOf(const NNCacheContextId& id) const {
  if(!owns(id))
    throw StringError(
      "NNCacheContextSet: this context id was minted by a different cache's context set and "
      "names nothing here. Reading it against this set would return whichever context "
      "happens to sit at the same position, which is the wrong-context service the id's "
      "carried set identity exists to make impossible."
    );
  return names_[id.index()];
}

NNCacheContextResolution NNCacheContextSet::resolveForRequest(const std::optional<string>& requested) const {
  if(requested.has_value()) {
    for(size_t i = 0; i < names_.size(); i++) {
      if(names_[i] == requested.value())
        return NNCacheContextResolution::resolved(NNCacheAttribution::toContext(NNCacheContextId(setId_, i)));
    }
    return NNCacheContextResolution::refused(
      "Unknown cacheContext '" + requested.value() + "'. Attached contexts for this model: " +
      quotedList(names_) + "."
    );
  }
  if(names_.size() == 1)
    return NNCacheContextResolution::resolved(NNCacheAttribution::toContext(NNCacheContextId(setId_, 0)));
  return NNCacheContextResolution::resolved(NNCacheAttribution::noAttributableContext());
}

//-------------------------------------------------------------------------------------
// The harvested surface
//-------------------------------------------------------------------------------------

NNCacheAttributionLedger::NNCacheAttributionLedger(
  NNCacheAttributionDisposition disposition,
  vector<NNCacheAttributionRow> rows,
  int64_t noAttributableContextEntries,
  int64_t unrecordedAttributions
)
  :disposition_(disposition),
   rows_(std::move(rows)),
   noAttributableContextEntries_(noAttributableContextEntries),
   unrecordedAttributions_(unrecordedAttributions)
{}

NNCacheAttributionLedger NNCacheAttributionLedger::notAttributed() {
  return NNCacheAttributionLedger(NNCacheAttributionDisposition::NotAttributed, vector<NNCacheAttributionRow>(), 0, 0);
}

NNCacheAttributionLedger NNCacheAttributionLedger::attributed(
  vector<NNCacheAttributionRow> rows, int64_t noAttributableContextEntries, int64_t unrecordedAttributions
) {
  return NNCacheAttributionLedger(
    NNCacheAttributionDisposition::Attributed, std::move(rows), noAttributableContextEntries, unrecordedAttributions
  );
}

const vector<NNCacheAttributionRow>& NNCacheAttributionLedger::rows() const {
  if(disposition_ != NNCacheAttributionDisposition::Attributed)
    throw StringError(
      "NNCacheAttributionLedger: no context has been attached to this cache, so it attributes "
      "nothing and has no rows to hand out. Check disposition() before asking; an empty row "
      "list would be indistinguishable from a session that earned nothing."
    );
  return rows_;
}

int64_t NNCacheAttributionLedger::noAttributableContextEntries() const {
  if(disposition_ != NNCacheAttributionDisposition::Attributed)
    throw StringError(
      "NNCacheAttributionLedger: no context has been attached to this cache, so it has no "
      "unattributed count to report either."
    );
  return noAttributableContextEntries_;
}

int64_t NNCacheAttributionLedger::unrecordedAttributions() const {
  if(disposition_ != NNCacheAttributionDisposition::Attributed)
    throw StringError(
      "NNCacheAttributionLedger: no context has been attached to this cache, so it has no "
      "unrecorded attribution count to report either."
    );
  return unrecordedAttributions_;
}

//-------------------------------------------------------------------------------------
// The recorder
//-------------------------------------------------------------------------------------

class NNCacheAttributionRecorder::Impl {
 public:
  // How many slots forward of home a key may be placed. A bounded window keeps the write
  // O(1) and keeps a nearly-full recorder from degrading into a scan; overflowing it is
  // reported, not absorbed.
  static const uint32_t PROBE_WINDOW = 16;

  Impl(int powerOfTwo, int mutexPoolSizePowerOfTwo)
    :rows_(((size_t)1) << powerOfTwo),
     mask_((((uint64_t)1) << powerOfTwo) - 1),
     mutexPool_(((uint32_t)1) << mutexPoolSizePowerOfTwo),
     mutexMask_((((uint32_t)1) << mutexPoolSizePowerOfTwo) - 1),
     noAttributable_(0),
     unrecorded_(0)
  {}

  void record(Hash128 key, const NNCacheAttribution& attribution, NNCacheEntryProvenance provenance) {
    if(!attribution.isToContext()) {
      // Counted, never guessed into whichever context happens to be attached.
      noAttributable_.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    const NNCacheContextId id = attribution.contextId();
    // The row holds the context's position in a 32-bit field, which is what buys the
    // persisted mark its bytes for free (see Row). A position past that is refused rather
    // than truncated into a different context's index (ADR-0002).
    if(id.index() > (size_t)0xFFFFFFFFu)
      throw StringError(
        "NNCacheAttributionRecorder: a context index past 2^32 cannot be recorded. Truncating "
        "it would file the entry under whichever context sits at the wrapped position."
      );
    const uint64_t home = key.hash0 & mask_;
    std::mutex& mutex = mutexPool_.getMutex((uint32_t)home & mutexMask_);
    std::lock_guard<std::mutex> lock(mutex);
    for(uint32_t step = 0; step < PROBE_WINDOW; step++) {
      Row& row = rows_[(home + step) & mask_];
      // setIdPlusOne == 0 marks a free row: a written row always carries a real set id, and
      // nextContextSetId never issues 0, so no occupied row can be mistaken for a free one.
      if(row.setIdPlusOne == 0 || row.key == key) {
        row.key = key;
        row.setIdPlusOne = id.setId() + 1;
        row.contextIndex = (uint32_t)id.index();
        // ASSIGNED, NEVER OR-ED. LiveEvaluation clears an existing mark: the entry the table
        // now holds is not the entry on disk. See the header.
        row.persisted = (provenance == NNCacheEntryProvenance::LoadedFromContainer);
        return;
      }
    }
    unrecorded_.fetch_add(1, std::memory_order_relaxed);
  }

  int64_t markPersisted(const NNCacheContextId& context, const std::vector<Hash128>& keys) {
    int64_t marked = 0;
    for(size_t k = 0; k < keys.size(); k++) {
      const Hash128 key = keys[k];
      const uint64_t home = key.hash0 & mask_;
      std::mutex& mutex = mutexPool_.getMutex((uint32_t)home & mutexMask_);
      std::lock_guard<std::mutex> lock(mutex);
      for(uint32_t step = 0; step < PROBE_WINDOW; step++) {
        Row& row = rows_[(home + step) & mask_];
        if(row.setIdPlusOne == 0)
          break;
        if(row.key != key)
          continue;
        // Only this context's own row. A key another context also earned is that context's
        // to persist into its own file, and marking it here would drop it from that file.
        if(row.setIdPlusOne == context.setId() + 1 && row.contextIndex == (uint32_t)context.index()) {
          row.persisted = true;
          marked += 1;
        }
        break;
      }
    }
    return marked;
  }

  std::vector<Hash128> unpersistedKeysFor(const NNCacheContextId& context) const {
    std::vector<Hash128> out;
    for(size_t i = 0; i < rows_.size(); i++) {
      std::mutex& mutex = mutexPool_.getMutex((uint32_t)i & mutexMask_);
      std::lock_guard<std::mutex> lock(mutex);
      if(rows_[i].setIdPlusOne == 0)
        continue;
      if(rows_[i].setIdPlusOne != context.setId() + 1 || rows_[i].contextIndex != (uint32_t)context.index())
        continue;
      if(rows_[i].persisted)
        continue;
      out.push_back(rows_[i].key);
    }
    return out;
  }

  vector<NNCacheAttributionRow> harvest(const NNCacheContextSet& contexts) const {
    vector<NNCacheAttributionRow> out;
    for(size_t i = 0; i < rows_.size(); i++) {
      std::mutex& mutex = mutexPool_.getMutex((uint32_t)i & mutexMask_);
      std::lock_guard<std::mutex> lock(mutex);
      if(rows_[i].setIdPlusOne == 0)
        continue;
      // A row minted by another set, or naming a position this set never attached, is a
      // wrong-context service waiting to happen. It is refused by name rather than read as
      // whichever context sits at the same position (ADR-0002); the table's set path checks
      // the same ownership before a row is ever written, so reaching this is a defect in
      // that path, not a client's doing.
      if(rows_[i].setIdPlusOne != contexts.id() + 1 || (size_t)rows_[i].contextIndex >= contexts.names().size())
        throw StringError(
          "NNCacheAttributionRecorder: a recorded attribution names a context this cache's "
          "context set did not mint. It is not resolved against this set's names."
        );
      NNCacheAttributionRow row;
      row.key = rows_[i].key;
      row.context = contexts.names()[rows_[i].contextIndex];
      out.push_back(row);
    }
    return out;
  }

  vector<Hash128> keysFor(const NNCacheContextId& context) const {
    vector<Hash128> out;
    for(size_t i = 0; i < rows_.size(); i++) {
      std::mutex& mutex = mutexPool_.getMutex((uint32_t)i & mutexMask_);
      std::lock_guard<std::mutex> lock(mutex);
      if(rows_[i].setIdPlusOne == 0)
        continue;
      if(rows_[i].setIdPlusOne != context.setId() + 1 || rows_[i].contextIndex != (uint32_t)context.index())
        continue;
      out.push_back(rows_[i].key);
    }
    return out;
  }

  // The row type's own size, exposed so the public rowBytes() reads it from the type rather
  // than from a second copy of the arithmetic.
  static size_t rowBytes() { return sizeof(Row); }

  int64_t noAttributableContextEntries() const { return noAttributable_.load(std::memory_order_relaxed); }
  int64_t unrecordedAttributions() const { return unrecorded_.load(std::memory_order_relaxed); }
  size_t structureBytes() const {
    return rows_.size() * rowBytes() + ((size_t)mutexMask_ + 1) * sizeof(std::mutex);
  }

 private:
  // 32 BYTES, AND THE PERSISTED MARK COSTS NONE OF THEM. The row is 16 + 8 + 4 + 1 = 29
  // bytes of fields in a structure whose alignment is 8, so it occupies 32 either way: the
  // mark lives in bytes this row was already paying for. contextIndex is 32-bit rather than
  // size_t for exactly that reason, and record() refuses an index that would not fit rather
  // than wrapping one. The relation is asserted in testnncachedump.cpp against sizeof(Row),
  // not trusted from this comment -- an earlier comment in this file named a per-row byte
  // figure that was not the real one.
  struct Row {
    Hash128 key;
    // The minting set's id, plus one, so that zero is free rather than a legal set id.
    uint64_t setIdPlusOne;
    uint32_t contextIndex;
    // Whether this key's bytes are already in the context's evaluation container. See
    // NNCacheEntryProvenance.
    bool persisted;
    Row() : key(), setIdPlusOne(0), contextIndex(0), persisted(false) {}
  };

  std::vector<Row> rows_;
  uint64_t mask_;
  mutable MutexPool mutexPool_;
  uint32_t mutexMask_;
  std::atomic<int64_t> noAttributable_;
  std::atomic<int64_t> unrecorded_;
};

// 2^20 rows. See the header for why this is a working size and not a bound, and for why the
// load factor rather than the raw row count is what decides it: at sizingReferenceKeys() the
// structure sits at 1048576 rows against 291129 keys, which is under maxLoadFactorPercent().
// 2^19 would not be -- 524288 rows against 291129 keys is 55.5% occupancy, past the bar the
// bounded probe window is held to. The relation is asserted in testnncachecontext.cpp rather
// than left in this comment to be trusted.
int NNCacheAttributionRecorder::defaultPowerOfTwo() {
  return 20;
}

// The largest per-card key count in the operator's corpus (cache-corpus-stats.wiki: 251 cards,
// median 45664, p90 121796, max 291129, over the num_refs > 1 subset a level 0 is built from).
int64_t NNCacheAttributionRecorder::sizingReferenceKeys() {
  return 291129;
}

int NNCacheAttributionRecorder::maxLoadFactorPercent() {
  return 50;
}

size_t NNCacheAttributionRecorder::rowBytes() {
  return Impl::rowBytes();
}

size_t attributionRecorderBytes(int powerOfTwo) {
  return (((size_t)1) << powerOfTwo) * NNCacheAttributionRecorder::rowBytes();
}

NNCacheAttributionRecorder::NNCacheAttributionRecorder(int powerOfTwo, int mutexPoolSizePowerOfTwo)
  :impl_(nullptr)
{
  if(powerOfTwo < 0 || powerOfTwo > 40)
    throw StringError(
      "NNCacheAttributionRecorder: powerOfTwo must be between 0 and 40; got " +
      Global::intToString(powerOfTwo) + "."
    );
  if(mutexPoolSizePowerOfTwo < 0 || mutexPoolSizePowerOfTwo > powerOfTwo)
    throw StringError(
      "NNCacheAttributionRecorder: mutexPoolSizePowerOfTwo must be between 0 and powerOfTwo."
    );
  impl_.reset(new Impl(powerOfTwo, mutexPoolSizePowerOfTwo));
}

NNCacheAttributionRecorder::~NNCacheAttributionRecorder() {}

void NNCacheAttributionRecorder::record(
  Hash128 key, const NNCacheAttribution& attribution, NNCacheEntryProvenance provenance
) {
  impl_->record(key, attribution, provenance);
}

void NNCacheAttributionRecorder::record(Hash128 key, const NNCacheAttribution& attribution) {
  impl_->record(key, attribution, NNCacheEntryProvenance::LiveEvaluation);
}

int64_t NNCacheAttributionRecorder::markPersisted(
  const NNCacheContextId& context, const vector<Hash128>& keys
) {
  return impl_->markPersisted(context, keys);
}

vector<Hash128> NNCacheAttributionRecorder::unpersistedKeysFor(const NNCacheContextId& context) const {
  return impl_->unpersistedKeysFor(context);
}

vector<NNCacheAttributionRow> NNCacheAttributionRecorder::harvest(const NNCacheContextSet& contexts) const {
  return impl_->harvest(contexts);
}

vector<Hash128> NNCacheAttributionRecorder::keysFor(const NNCacheContextId& context) const {
  return impl_->keysFor(context);
}

int64_t NNCacheAttributionRecorder::noAttributableContextEntries() const {
  return impl_->noAttributableContextEntries();
}

int64_t NNCacheAttributionRecorder::unrecordedAttributions() const {
  return impl_->unrecordedAttributions();
}

size_t NNCacheAttributionRecorder::structureBytes() const {
  return impl_->structureBytes();
}
