#include "../neuralnet/nnevalcontainer.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <optional>
#include <utility>

#include "../core/fileutils.h"
#include "../core/global.h"
#include "../neuralnet/nncachefileformat.h"

// The append-only per-(context, model) evaluation container. See nnevalcontainer.h for the
// format and for why it is the shape it is; this file is the mechanism.
//
// The framing discipline is the count log's, deliberately: magic and version and strides in
// the header, per-dump blocks, a block header that checksums itself before its lengths are
// believed, little-endian fields packed byte by byte, the reader never repairing and the
// writer always doing so, and rewrites through a temp file and an atomic rename. One file
// family, one crash-safety story. What is new here is the payload -- variable-length,
// board-sized, optionally carrying an ownership map -- and the two consequences of its
// SIZE: a median block is ~69 MB, so neither reading nor writing a block may require
// holding it.

namespace {

using NNCacheFileBytes::put32;
using NNCacheFileBytes::put64;
using NNCacheFileBytes::putF32;
using NNCacheFileBytes::get32;
using NNCacheFileBytes::get64;
using NNCacheFileBytes::getF32;

//-------------------------------------------------------------------------------------
// The format's constants, in one place
//-------------------------------------------------------------------------------------

const char FILE_MAGIC[8] = {'K','G','N','N','E','V','A','L'};
const uint32_t FORMAT_VERSION = 1;
// The fixed part of the file header. The model name follows it, and the header's own
// "file header bytes" field is the sum, which is what makes the header self-describing.
const size_t FIXED_FILE_HEADER_BYTES = 48;
const size_t BLOCK_HEADER_BYTES = 32;
// The bytes of a block header that the header's own checksum covers: everything before the
// checksum field itself.
const size_t BLOCK_HEADER_CHECKED_BYTES = 24;
const size_t ENTRY_HEADER_BYTES = 32;
const uint32_t BLOCK_MAGIC = 0x4245474Bu;  // 'K','G','E','B' little-endian
const int DEFAULT_COMPACTION_MULTIPLE = 4;

// The ten scalars of an NNOutput, in the order the payload stores them.
const size_t NUM_SCALARS = 10;
// The only flag bit v1 defines. Every other bit is reserved and must be zero; a set one is
// a file from a future version and is refused rather than ignored.
const uint16_t FLAG_HAS_OWNERMAP = 0x1;

// The buffer a block is streamed through when its checksum is verified. Bounded on purpose:
// verifying a 69 MB block must not require 69 MB.
const size_t STREAM_BUFFER_BYTES = 1 << 16;

//-------------------------------------------------------------------------------------
// 64-bit file offsets
//-------------------------------------------------------------------------------------

// std::ftell and std::fseek speak `long`, which is 32 bits on Windows -- and a container
// for the operator's largest card is ~863 MB per full copy, so a file that has grown a few
// blocks past 2 GB before its next compaction is an ordinary state here, not an exotic one.
// These use the 64-bit entry points on every platform, so an offset past 2 GB is a real
// offset rather than a silently wrapped one.
bool seekTo(std::FILE* f, int64_t offset) {
#ifdef _WIN32
  return _fseeki64(f, (__int64)offset, SEEK_SET) == 0;
#else
  return fseeko(f, (off_t)offset, SEEK_SET) == 0;
#endif
}

int64_t sizeOfOpenFile(std::FILE* f, const std::string& path) {
#ifdef _WIN32
  if(_fseeki64(f, 0, SEEK_END) != 0)
    throw StringError("NNEvalContainer: could not seek " + path + ".");
  const __int64 endPos = _ftelli64(f);
#else
  if(fseeko(f, 0, SEEK_END) != 0)
    throw StringError("NNEvalContainer: could not seek " + path + ".");
  const off_t endPos = ftello(f);
#endif
  if(endPos < 0)
    throw StringError("NNEvalContainer: could not size " + path + ".");
  return (int64_t)endPos;
}

//-------------------------------------------------------------------------------------
// Shapes
//-------------------------------------------------------------------------------------

// The largest board edge this format can frame at all. The u8 field is the format's bound;
// Board::MAX_LEN is the build's, and the two are checked separately so a refusal names the
// one that actually bit.
const int MAX_FRAMEABLE_EDGE = 255;

// Refuses, naming both numbers, a shape this build or this format cannot represent. It is a
// refusal and not a coercion: a file written by a build with a larger COMPILE_MAX_BOARD_LEN
// is a legitimate file that this build cannot honor, and saying so is the only honest answer
// (ADR-0012 P2).
void verifyShape(int nnXLen, int nnYLen, const std::string& whatFor) {
  if(nnXLen < 1 || nnYLen < 1)
    throw StringError(
      "NNEvalContainer: " + whatFor + " has a board of " + Global::intToString(nnXLen) + "x" +
      Global::intToString(nnYLen) + ", which is not a board."
    );
  if(nnXLen > MAX_FRAMEABLE_EDGE || nnYLen > MAX_FRAMEABLE_EDGE)
    throw StringError(
      "NNEvalContainer: " + whatFor + " has a board of " + Global::intToString(nnXLen) + "x" +
      Global::intToString(nnYLen) + "; this format frames an edge of at most " +
      Global::intToString(MAX_FRAMEABLE_EDGE) + "."
    );
  if(nnXLen > Board::MAX_LEN || nnYLen > Board::MAX_LEN)
    throw StringError(
      "NNEvalContainer: " + whatFor + " has a board of " + Global::intToString(nnXLen) + "x" +
      Global::intToString(nnYLen) + "; this build was compiled for an edge of at most " +
      Global::intToString(Board::MAX_LEN) + " and cannot represent it."
    );
  // Belt and braces against a future where the policy array stops being area+1: the entry
  // is refused rather than written into an array it does not fit.
  if((int64_t)nnXLen * (int64_t)nnYLen + 1 > (int64_t)NNPos::MAX_NN_POLICY_SIZE)
    throw StringError(
      "NNEvalContainer: " + whatFor + " needs " +
      Global::int64ToString((int64_t)nnXLen * (int64_t)nnYLen + 1) +
      " policy slots and this build's array holds " +
      Global::intToString(NNPos::MAX_NN_POLICY_SIZE) + "."
    );
}

int64_t payloadBytesOf(int nnXLen, int nnYLen, bool hasOwnerMap) {
  const int64_t area = (int64_t)nnXLen * (int64_t)nnYLen;
  const int64_t values = (int64_t)NUM_SCALARS + (area + 1) + (hasOwnerMap ? area : 0);
  return values * 4;
}

//-------------------------------------------------------------------------------------
// Header encoding
//-------------------------------------------------------------------------------------

std::vector<uint8_t> encodeFileHeader(
  uint64_t contextHash,
  int modelVersion,
  const std::string& modelInternalName
) {
  std::vector<uint8_t> out(FIXED_FILE_HEADER_BYTES + modelInternalName.size(), 0);
  std::memcpy(out.data(), FILE_MAGIC, 8);
  put32(out.data() + 8, FORMAT_VERSION);
  put32(out.data() + 12, (uint32_t)out.size());
  put32(out.data() + 16, (uint32_t)ENTRY_HEADER_BYTES);
  put32(out.data() + 20, (uint32_t)BLOCK_HEADER_BYTES);
  put64(out.data() + 24, contextHash);
  put32(out.data() + 32, (uint32_t)modelVersion);
  put32(out.data() + 36, (uint32_t)modelInternalName.size());
  // [40..48) reserved, already zero.
  std::memcpy(out.data() + FIXED_FILE_HEADER_BYTES, modelInternalName.data(), modelInternalName.size());
  return out;
}

// Refuses, naming the field, anything this build did not write. This is the boundary that
// keeps another format, another version, another context's counts, or ANOTHER MODEL'S
// EVALUATIONS from being read as if they were this container's (ADR-0012 P2: translate and
// validate, never coerce). The model check is the load-bearing one: the cache key names no
// net, so a container read under the wrong model would serve one net's evaluations as
// another's, silently, and the header is the only place that fact lives.
void verifyFileHeader(
  const std::vector<uint8_t>& hdr,
  const std::string& path,
  const std::string& context,
  uint64_t contextHash,
  const std::string& modelInternalName,
  int modelVersion
) {
  if(std::memcmp(hdr.data(), FILE_MAGIC, 8) != 0)
    throw StringError("NNEvalContainer: " + path + " does not begin with the evaluation-container magic; it is not an nnevals container.");
  const uint32_t version = get32(hdr.data() + 8);
  if(version != FORMAT_VERSION)
    throw StringError(
      "NNEvalContainer: " + path + " is format version " + Global::uint64ToString(version) +
      " and this build reads version " + Global::uint64ToString(FORMAT_VERSION) + " only."
    );
  const uint32_t entryHeaderBytes = get32(hdr.data() + 16);
  if(entryHeaderBytes != (uint32_t)ENTRY_HEADER_BYTES)
    throw StringError(
      "NNEvalContainer: " + path + " declares a " + Global::uint64ToString(entryHeaderBytes) +
      "-byte entry header; this build writes " + Global::uint64ToString(ENTRY_HEADER_BYTES) + "."
    );
  const uint32_t blockHeaderBytes = get32(hdr.data() + 20);
  if(blockHeaderBytes != (uint32_t)BLOCK_HEADER_BYTES)
    throw StringError(
      "NNEvalContainer: " + path + " declares a " + Global::uint64ToString(blockHeaderBytes) +
      "-byte block header; this build writes " + Global::uint64ToString(BLOCK_HEADER_BYTES) + "."
    );
  const uint64_t storedContext = get64(hdr.data() + 24);
  if(storedContext != contextHash)
    throw StringError(
      "NNEvalContainer: " + path + " was written for a different context than '" + context +
      "'. Two contexts' evaluations are not merged."
    );
  const uint32_t storedModelVersion = get32(hdr.data() + 32);
  if(storedModelVersion != (uint32_t)modelVersion)
    throw StringError(
      "NNEvalContainer: " + path + " was written by model version " +
      Global::uint64ToString(storedModelVersion) + " and this evaluator is model version " +
      Global::intToString(modelVersion) +
      ". A net's outputs are read under the version that produced them, never another."
    );
  const uint32_t nameLen = get32(hdr.data() + 36);
  const uint32_t declaredHeaderBytes = get32(hdr.data() + 12);
  if(declaredHeaderBytes != (uint32_t)FIXED_FILE_HEADER_BYTES + nameLen)
    throw StringError(
      "NNEvalContainer: " + path + " declares a " + Global::uint64ToString(declaredHeaderBytes) +
      "-byte file header but a " + Global::uint64ToString(nameLen) +
      "-character model name, which do not agree."
    );
  for(size_t i = 40; i < 48; i++) {
    if(hdr[i] != 0)
      throw StringError("NNEvalContainer: " + path + " sets a reserved file-header byte; it is not a version this build reads.");
  }
  const std::string storedName((const char*)hdr.data() + FIXED_FILE_HEADER_BYTES, nameLen);
  if(storedName != modelInternalName)
    throw StringError(
      "NNEvalContainer: " + path + " holds evaluations by model '" + storedName +
      "' and was opened for model '" + modelInternalName +
      "'. The NN cache key names no net, so one net's evaluations are never read as another's."
    );
}

void encodeBlockHeader(
  uint8_t* out,
  uint32_t entryCount,
  uint64_t totalPayloadBytes,
  uint64_t entryChecksum,
  uint64_t contextHash
) {
  put32(out + 0, BLOCK_MAGIC);
  put32(out + 4, entryCount);
  put64(out + 8, totalPayloadBytes);
  put64(out + 16, entryChecksum);
  put64(out + 24, NNCacheFileChecksum::of(out, BLOCK_HEADER_CHECKED_BYTES, contextHash));
}

//-------------------------------------------------------------------------------------
// Entry encoding and decoding
//-------------------------------------------------------------------------------------

// Verifies one entry and states the payload size its shape implies. Every refusal the write
// path can raise is raised here, and the write path calls this for every entry BEFORE it
// opens the file, so a rejected dump never leaves a half-written block behind.
int64_t verifiedPayloadBytesOf(const NNOutput& out, const std::string& path) {
  // noisedPolicyProbs has no flag, so a file cannot carry it, so an entry that has one is
  // refused rather than written without it. Search-time noise attached to a tree node is not
  // a fact about the position, and persisting it would replay one session's noise as
  // another's evaluation (ADR-0002: never silently drop what you were handed).
  if(out.noisedPolicyProbs != NULL)
    throw StringError(
      "NNEvalContainer: " + path + ": key " + out.nnHash.toString() +
      " carries noisedPolicyProbs, which this format does not store. Search-time noise is not "
      "a fact about the position; persist the un-noised evaluation instead."
    );
  verifyShape(out.nnXLen, out.nnYLen, "key " + out.nnHash.toString());
  return payloadBytesOf(out.nnXLen, out.nnYLen, out.whiteOwnerMap != NULL);
}

// Encodes one entry's HEADER, into the gathered header array at `p`. The payload offset is
// supplied by the caller because it is the running sum of the preceding payload sizes and
// that sum has one owner -- the loop writing the array -- rather than being recomputed here.
void encodeEntryHeader(uint8_t* p, const NNOutput& out, int64_t payloadBytes, int64_t payloadOffset) {
  const bool hasOwnerMap = out.whiteOwnerMap != NULL;
  put64(p + 0, out.nnHash.hash0);
  put64(p + 8, out.nnHash.hash1);
  p[16] = (uint8_t)(hasOwnerMap ? FLAG_HAS_OWNERMAP : 0);
  p[17] = 0;
  p[18] = (uint8_t)out.nnXLen;
  p[19] = (uint8_t)out.nnYLen;
  put32(p + 20, (uint32_t)payloadBytes);
  put64(p + 24, (uint64_t)payloadOffset);
}

// Encodes one entry's PAYLOAD into `buf`, which is REUSED across entries so that writing a
// block of 45,664 entries never builds a 69 MB image in memory.
void encodePayload(std::vector<uint8_t>& buf, const NNOutput& out, int64_t payloadBytes) {
  const bool hasOwnerMap = out.whiteOwnerMap != NULL;
  const int64_t area = (int64_t)out.nnXLen * (int64_t)out.nnYLen;
  buf.assign((size_t)payloadBytes, 0);

  uint8_t* v = buf.data();
  const float scalars[NUM_SCALARS] = {
    out.whiteWinProb, out.whiteLossProb, out.whiteNoResultProb, out.whiteScoreMean,
    out.whiteScoreMeanSq, out.whiteLead, out.varTimeLeft, out.shorttermWinlossError,
    out.shorttermScoreError, out.policyOptimismUsed
  };
  for(size_t i = 0; i < NUM_SCALARS; i++)
    putF32(v + i * 4, scalars[i]);
  v += NUM_SCALARS * 4;
  // Board-sized, plus the pass slot. The slots beyond it in policyProbs are a property of
  // this build's array and not of the evaluation, so they are not written.
  for(int64_t i = 0; i <= area; i++)
    putF32(v + (size_t)i * 4, out.policyProbs[i]);
  v += (size_t)(area + 1) * 4;
  if(hasOwnerMap) {
    for(int64_t i = 0; i < area; i++)
      putF32(v + (size_t)i * 4, out.whiteOwnerMap[i]);
  }
}

// EVERYTHING ONE 32-BYTE ENTRY HEADER SAYS, verified. The one place an entry header is
// read, so the key-set scan and the payload decode cannot come to different conclusions
// about the same 32 bytes (ADR-0012 P1).
//
// Every refusal below is judged only AFTER the block's checksum has proven these are the
// bytes the writer wrote, so a refusal here means the file is genuinely not one this build
// can read -- never that a crash happened.
struct DecodedEntryHeader {
  Hash128 key;
  int nnXLen;
  int nnYLen;
  bool hasOwnerMap;
  int64_t payloadBytes;
  // The offset this header states for its payload within its block's payload region. It is
  // a DERIVED quantity -- the running sum of the preceding payload sizes -- and is checked
  // against that sum by whoever owns the sum, which is the block walk, not this function.
  int64_t payloadOffset;
};

DecodedEntryHeader decodeEntryHeader(const uint8_t* header, const std::string& path) {
  DecodedEntryHeader h;
  h.key = Hash128(get64(header + 0), get64(header + 8));
  const uint16_t flags = (uint16_t)((uint16_t)header[16] | ((uint16_t)header[17] << 8));
  if((flags & (uint16_t)~(uint16_t)FLAG_HAS_OWNERMAP) != 0)
    throw StringError(
      "NNEvalContainer: " + path + ": entry " + h.key.toString() + " sets flag bits this build "
      "does not define. It was written by a later version of this format."
    );
  h.hasOwnerMap = (flags & FLAG_HAS_OWNERMAP) != 0;
  h.nnXLen = (int)header[18];
  h.nnYLen = (int)header[19];
  verifyShape(h.nnXLen, h.nnYLen, path + ": entry " + h.key.toString());
  h.payloadBytes = (int64_t)get32(header + 20);
  h.payloadOffset = (int64_t)get64(header + 24);

  const int64_t expected = payloadBytesOf(h.nnXLen, h.nnYLen, h.hasOwnerMap);
  if(h.payloadBytes != expected)
    throw StringError(
      "NNEvalContainer: " + path + ": entry " + h.key.toString() + " declares " +
      Global::int64ToString(h.payloadBytes) + " payload bytes but its " +
      Global::intToString(h.nnXLen) + "x" + Global::intToString(h.nnYLen) + " shape" +
      (h.hasOwnerMap ? " with an ownership map" : " without an ownership map") + " is " +
      Global::int64ToString(expected) + " bytes."
    );
  return h;
}

// Fills `out` from one entry's verified header and its verified payload bytes.
//
// THE OWNERSHIP-MAP STORAGE IS THE CALLER'S, and this function refuses to proceed if the
// caller did not supply exactly what the header says is needed. The two callers allocate it
// in incompatible ways -- the ordinary read path with new[], which ~NNOutput will delete[],
// and the level-0 loader from a contiguous arena block, which ~NNOutput must NEVER touch --
// so a reader that allocated on the caller's behalf would have to guess which, and a wrong
// guess is a free of memory the allocator never issued.
void decodePayloadInto(
  NNOutput& out,
  const DecodedEntryHeader& h,
  const uint8_t* payload,
  const std::string& path
) {
  if(h.hasOwnerMap != (out.whiteOwnerMap != NULL))
    throw StringError(
      "NNEvalContainer: " + path + ": entry " + h.key.toString() +
      (h.hasOwnerMap ? " carries an ownership map and no storage was supplied for it."
                     : " carries no ownership map and storage was supplied for one.")
    );

  const int64_t area = (int64_t)h.nnXLen * (int64_t)h.nnYLen;
  out.nnHash = h.key;
  const uint8_t* v = payload;
  out.whiteWinProb = getF32(v + 0);
  out.whiteLossProb = getF32(v + 4);
  out.whiteNoResultProb = getF32(v + 8);
  out.whiteScoreMean = getF32(v + 12);
  out.whiteScoreMeanSq = getF32(v + 16);
  out.whiteLead = getF32(v + 20);
  out.varTimeLeft = getF32(v + 24);
  out.shorttermWinlossError = getF32(v + 28);
  out.shorttermScoreError = getF32(v + 32);
  out.policyOptimismUsed = getF32(v + 36);
  v += NUM_SCALARS * 4;

  out.nnXLen = h.nnXLen;
  out.nnYLen = h.nnYLen;
  // The slots past the board are ZEROED rather than left as whatever the allocation held.
  // They are not a fact the file carries, and an uninitialised slot is exactly the kind of
  // value that reads as plausible later.
  for(int i = 0; i < NNPos::MAX_NN_POLICY_SIZE; i++)
    out.policyProbs[i] = 0.0f;
  for(int64_t i = 0; i <= area; i++)
    out.policyProbs[i] = getF32(v + (size_t)i * 4);
  v += (size_t)(area + 1) * 4;

  if(h.hasOwnerMap) {
    for(int64_t i = 0; i < area; i++)
      out.whiteOwnerMap[i] = getF32(v + (size_t)i * 4);
  }
  // A file cannot carry search-time noise, so a decoded entry never has any.
  out.noisedPolicyProbs = NULL;
}

// The ordinary read path's decode: a heap NNOutput with a new[] ownership map that
// ~NNOutput owns from the moment it is attached, so a throw between here and the caller
// frees it.
std::unique_ptr<NNOutput> decodeEntryToHeap(
  const DecodedEntryHeader& h,
  const uint8_t* payload,
  const std::string& path
) {
  std::unique_ptr<NNOutput> out(new NNOutput());
  if(h.hasOwnerMap)
    out->whiteOwnerMap = new float[(size_t)h.nnXLen * (size_t)h.nnYLen];
  decodePayloadInto(*out, h, payload, path);
  return out;
}

//-------------------------------------------------------------------------------------
// The merge
//-------------------------------------------------------------------------------------

typedef std::map<std::pair<uint64_t,uint64_t>, size_t> KeyIndex;

// What the merge rule says to do with an incoming entry.
enum class MergeAction { Append, ReplaceExisting, Drop };

// LAST-WINS PER KEY, EXCEPT THAT AN ENTRY WITHOUT AN OWNERSHIP MAP NEVER SUPERSEDES ONE
// WITH. The exception is the store-side face of the live supersession rule: an entry lacking
// a requested ownership map costs a full re-evaluation on every hit, so letting a later
// ownermap-less re-evaluation overwrite an ownermap-carrying one would turn a restored level
// 0 into a bleed-out.
//
// THIS IS THE ONLY STATEMENT OF THE RULE. The full read, the key-set scan and compaction all
// route through it, so a live set assembled by one of them is the live set the others would
// assemble (ADR-0012 P1) -- which matters more here than tidiness, because the loader
// selects from the key-set scan's live set and then reads payloads the full read's live set
// would have to agree with.
MergeAction mergeActionFor(bool keyIsNew, bool existingHasOwnerMap, bool incomingHasOwnerMap) {
  if(keyIsNew)
    return MergeAction::Append;
  if(existingHasOwnerMap && !incomingHasOwnerMap)
    return MergeAction::Drop;  // the fuller entry stands
  return MergeAction::ReplaceExisting;
}

// Applies the rule to a vector of anything, given how to read an element's key and whether
// it has an ownership map. Returns the index the incoming element belongs at, or nullopt to
// drop it.
std::optional<size_t> mergeSlotFor(KeyIndex& indexOfKey, Hash128 key, bool incomingHasOwnerMap,
                                   size_t currentSize, bool existingHasOwnerMap) {
  const std::pair<uint64_t,uint64_t> mapKey(key.hash0, key.hash1);
  const KeyIndex::iterator it = indexOfKey.find(mapKey);
  const bool keyIsNew = it == indexOfKey.end();
  const MergeAction action = mergeActionFor(keyIsNew, existingHasOwnerMap, incomingHasOwnerMap);
  if(action == MergeAction::Drop)
    return std::nullopt;
  if(action == MergeAction::Append) {
    indexOfKey[mapKey] = currentSize;
    return currentSize;
  }
  return it->second;
}

void applyEntry(
  std::vector<std::unique_ptr<NNOutput>>& entries,
  KeyIndex& indexOfKey,
  std::unique_ptr<NNOutput> incoming
) {
  const std::pair<uint64_t,uint64_t> mapKey(incoming->nnHash.hash0, incoming->nnHash.hash1);
  const KeyIndex::const_iterator it = indexOfKey.find(mapKey);
  const bool existingHasOwnerMap =
    it != indexOfKey.end() && entries[it->second]->whiteOwnerMap != NULL;
  const std::optional<size_t> slot = mergeSlotFor(
    indexOfKey, incoming->nnHash, incoming->whiteOwnerMap != NULL, entries.size(), existingHasOwnerMap);
  if(!slot.has_value())
    return;
  if(slot.value() == entries.size())
    entries.push_back(std::move(incoming));
  else
    entries[slot.value()] = std::move(incoming);
}

void applyLocation(
  std::vector<NNEvalContainerEntryLocation>& locations,
  KeyIndex& indexOfKey,
  const NNEvalContainerEntryLocation& incoming
) {
  const std::pair<uint64_t,uint64_t> mapKey(incoming.key.hash0, incoming.key.hash1);
  const KeyIndex::const_iterator it = indexOfKey.find(mapKey);
  const bool existingHasOwnerMap =
    it != indexOfKey.end() && locations[it->second].hasOwnerMap;
  const std::optional<size_t> slot = mergeSlotFor(
    indexOfKey, incoming.key, incoming.hasOwnerMap, locations.size(), existingHasOwnerMap);
  if(!slot.has_value())
    return;
  if(slot.value() == locations.size())
    locations.push_back(incoming);
  else
    locations[slot.value()] = incoming;
}

//-------------------------------------------------------------------------------------
// The scan
//-------------------------------------------------------------------------------------

// One pass over a container. Both load() and appendBlock() go through this, so there is
// exactly one implementation of "where does the intact part of this file end" (ADR-0012 P1).
struct ScanResult {
  int64_t blocksApplied = 0;
  int64_t entriesApplied = 0;
  // Byte offset one past the last intact block. Equal to the file size when the tail is
  // intact.
  int64_t intactEndOffset = 0;
  int64_t tornTailBytes = 0;
  bool fileExists = false;
};

// Walks a block's gathered header array in order, decoding each 32-byte header and checking
// the two facts only the walk can check: that each entry's payload fits in what is left of
// the block's declared payload region, and that the offset the header STATES for its payload
// is the running sum of the preceding payload sizes. That sum has one authority -- this walk
// -- so the stored offset is recomputed and refused on disagreement rather than trusted
// (ADR-0012 P1).
//
// `onEntry(i, header, decodedHeader, payloadOffset)` is called per entry, in file order.
template<typename OnEntry>
void walkBlockHeaderArray(
  const uint8_t* headerArray,
  uint32_t entryCount,
  uint64_t totalPayloadBytes,
  const std::string& path,
  OnEntry onEntry
) {
  int64_t payloadOffset = 0;
  for(uint32_t i = 0; i < entryCount; i++) {
    const uint8_t* entryHeader = headerArray + (size_t)i * ENTRY_HEADER_BYTES;
    const DecodedEntryHeader h = decodeEntryHeader(entryHeader, path);
    if(h.payloadBytes > (int64_t)totalPayloadBytes - payloadOffset)
      throw StringError(
        "NNEvalContainer: " + path + ": an entry declares " + Global::int64ToString(h.payloadBytes) +
        " payload bytes that run past the payload region its block declares."
      );
    if(h.payloadOffset != payloadOffset)
      throw StringError(
        "NNEvalContainer: " + path + ": entry " + h.key.toString() + " places its payload at offset " +
        Global::uint64ToString((uint64_t)h.payloadOffset) + " and the entries before it end at " +
        Global::int64ToString(payloadOffset) + "."
      );
    onEntry(i, h, payloadOffset);
    payloadOffset += h.payloadBytes;
  }
  if(payloadOffset != (int64_t)totalPayloadBytes)
    throw StringError(
      "NNEvalContainer: " + path + ": a block's payloads occupy " + Global::int64ToString(payloadOffset) +
      " bytes and its header declares a payload region of " + Global::uint64ToString(totalPayloadBytes) + "."
    );
}

// How much of a region this pass is willing to leave resident behind itself before it starts
// telling the kernel to forget it. See checksumRegion for why the drop happens INSIDE the
// stream rather than once per block.
const int64_t DROP_BEHIND_CHUNK_BYTES = 8 << 20;

// Streams `regionBytes` bytes from `f`'s current position -- which the caller states as
// `regionStartOffset`, since a drop-behind needs absolute offsets -- through a bounded
// buffer, and returns their checksum. Nothing is retained: this is how a 69 MB block is
// verified without being held.
//
// THE DROP-BEHIND IS INSIDE THIS LOOP AND NOT MERELY ONCE PER BLOCK, and the difference is
// the difference between a bound and a wish. A scan that dropped a block's pages only after
// finishing the block would keep ONE BLOCK resident in flight -- fine for the ~59 MB blocks
// an interval dump writes, and useless for the file compaction produces, which is a header
// plus exactly ONE block. On a compacted 10-20 GB card "bounded by one block" is bounded by
// the whole card, which is no bound at all. Dropping every DROP_BEHIND_CHUNK_BYTES bounds it
// by a figure that does not depend on how the file happened to be written.
//
// Returns false if the file was short, and sets `readFailed` if the shortness was an I/O
// ERROR rather than an end of file. The caller must tell those apart: one is a torn tail and
// the other is a disk that could not be read, and treating the second as the first would
// have the write path truncate a perfectly good card down to wherever the read happened to
// fail (ADR-0002).
bool checksumRegion(
  NNCacheFileHandle& f,
  int64_t regionStartOffset,
  int64_t regionBytes,
  uint64_t seed,
  uint64_t& ret,
  bool& readFailed
) {
  NNCacheFileChecksum sum(seed);
  std::vector<uint8_t> buf(STREAM_BUFFER_BYTES);
  int64_t left = regionBytes;
  int64_t consumedFrom = regionStartOffset;
  int64_t consumedTo = regionStartOffset;
  while(left > 0) {
    const size_t want = (size_t)std::min<int64_t>(left, (int64_t)buf.size());
    if(std::fread(buf.data(), 1, want, f.get()) != want) {
      readFailed = std::ferror(f.get()) != 0;
      return false;
    }
    sum.update(buf.data(), want);
    left -= (int64_t)want;
    consumedTo += (int64_t)want;
    if(consumedTo - consumedFrom >= DROP_BEHIND_CHUNK_BYTES) {
      f.dropCachedRange(consumedFrom, consumedTo - consumedFrom);
      consumedFrom = consumedTo;
    }
  }
  ret = sum.finish();
  return true;
}

// One pass over a container's framing. What is MADE of each verified block is the visitor's
// business; where the intact part of the file ends is this function's, and is stated once
// (ADR-0012 P1). `onBlock(f, headerArray, entryCount, headerArrayFileOffset,
// payloadRegionFileOffset, totalPayloadBytes)` is called with the file positioned at the
// start of the payload region and may seek freely; the scan repositions itself afterwards.
template<typename OnBlock>
ScanResult scanContainer(
  const std::string& path,
  const std::string& context,
  uint64_t contextHash,
  const std::string& modelInternalName,
  int modelVersion,
  OnBlock onBlock
) {
  ScanResult result;

  NNCacheFileHandle f(path, "rb");
  if(!f.isOpen())
    return result;  // Absent is a normal answer: no dump has happened here yet.
  result.fileExists = true;
  // THE READ POSTURE FOR THIS WHOLE FUNCTION, set once before the first read. Every byte
  // below is read exactly once, front to back, and never re-read, so the two things this
  // asks for are the two things that are true: read in large strides, and do not keep what
  // has been read. See NNCacheFileHandle::useSequentialStreamBuffer for the measurement that
  // sized the buffer. The return is deliberately not checked -- a stream that kept its
  // default buffer reads the same bytes more slowly, which is not a fact any caller of this
  // function can act on.
  (void)f.useSequentialStreamBuffer();

  const int64_t fileSize = sizeOfOpenFile(f.get(), path);
  if(!seekTo(f.get(), 0))
    throw StringError("NNEvalContainer: could not rewind " + path + ".");

  // THE MAGIC IS CHECKED AS SOON AS THERE ARE EIGHT BYTES TO CHECK IT WITH, and before any
  // question of length. That ordering is the whole refusal-versus-torn-tail distinction at
  // its most delicate point: a short file could be a crash during the very first dump (a
  // torn tail, kept quiet) or an operator naming a file that is not a container at all (a
  // refusal, by name), and only the magic can tell the two apart. Deciding on length first
  // would let a foreign file that happens to be shorter than this container's header pass
  // as a crash artifact.
  if(fileSize >= 8) {
    uint8_t magic[8];
    if(std::fread(magic, 1, 8, f.get()) != 8)
      throw StringError("NNEvalContainer: could not read the first bytes of " + path + ".");
    if(std::memcmp(magic, FILE_MAGIC, 8) != 0)
      throw StringError("NNEvalContainer: " + path + " does not begin with the evaluation-container magic; it is not an nnevals container.");
    if(!seekTo(f.get(), 0))
      throw StringError("NNEvalContainer: could not rewind " + path + ".");
  }

  // A file too short to hold even the FIXED part of the header is a torn tail at offset
  // zero: there is nothing to verify and nothing to keep.
  if(fileSize < (int64_t)FIXED_FILE_HEADER_BYTES) {
    result.tornTailBytes = fileSize;
    result.intactEndOffset = 0;
    return result;
  }

  std::vector<uint8_t> fixedHeader(FIXED_FILE_HEADER_BYTES);
  if(std::fread(fixedHeader.data(), 1, FIXED_FILE_HEADER_BYTES, f.get()) != FIXED_FILE_HEADER_BYTES)
    throw StringError("NNEvalContainer: could not read the header of " + path + ".");
  // THE HEADER'S OWN DECLARED LENGTH governs, not the caller's expectation of it. The header
  // is self-describing precisely so that a reader never has to guess how long it is from
  // what it hoped to find there.
  const int64_t headerBytes = (int64_t)get32(fixedHeader.data() + 12);
  if(fileSize < headerBytes) {
    // The fixed part is whole and says the name follows, and the name does not. That is a
    // crash between the two writes, not a foreign file: the magic already cleared that.
    result.tornTailBytes = fileSize;
    result.intactEndOffset = 0;
    return result;
  }

  std::vector<uint8_t> fileHeader((size_t)headerBytes);
  std::memcpy(fileHeader.data(), fixedHeader.data(), FIXED_FILE_HEADER_BYTES);
  const size_t nameBytes = (size_t)headerBytes - FIXED_FILE_HEADER_BYTES;
  if(nameBytes > 0 &&
     std::fread(fileHeader.data() + FIXED_FILE_HEADER_BYTES, 1, nameBytes, f.get()) != nameBytes)
    throw StringError("NNEvalContainer: could not read the model name in the header of " + path + ".");
  verifyFileHeader(fileHeader, path, context, contextHash, modelInternalName, modelVersion);

  int64_t offset = headerBytes;
  result.intactEndOffset = offset;

  std::vector<uint8_t> headerArray;

  while(true) {
    const int64_t remaining = fileSize - offset;
    if(remaining <= 0)
      break;
    if(remaining < (int64_t)BLOCK_HEADER_BYTES)
      break;  // torn: not even a whole block header left

    uint8_t blockHeader[BLOCK_HEADER_BYTES];
    if(std::fread(blockHeader, 1, BLOCK_HEADER_BYTES, f.get()) != BLOCK_HEADER_BYTES) {
      // AN I/O ERROR IS NOT A TORN TAIL, and the two must not be conflated here. A short
      // read at end of file is a crash artifact and is kept quiet; a read that FAILED is a
      // device or filesystem that could not answer, and the file it could not answer about
      // is not evidence of anything. The write path acts on this verdict by TRUNCATING, in
      // place and irreversibly, so mistaking the second for the first would shorten a
      // perfectly good card to wherever the disk happened to hiccup (ADR-0002).
      if(std::ferror(f.get()) != 0)
        throw StringError(
          "NNEvalContainer: could not read a block header of " + path + " at offset " +
          Global::int64ToString(offset) + ". This is a read failure, not a torn tail, and the "
          "file is left exactly as it was found."
        );
      break;  // torn: the file shrank under us
    }

    // The header checksums ITSELF before its lengths are believed. Everything after this
    // point trusts numbers that came out of a file a crash may have half-written, so this is
    // the check that has to come first.
    if(get64(blockHeader + 24) != NNCacheFileChecksum::of(blockHeader, BLOCK_HEADER_CHECKED_BYTES, contextHash))
      break;
    if(get32(blockHeader + 0) != BLOCK_MAGIC)
      break;

    const uint32_t entryCount = get32(blockHeader + 4);
    const uint64_t totalPayloadBytes = get64(blockHeader + 8);
    const int64_t remainingAfterHeader = remaining - (int64_t)BLOCK_HEADER_BYTES;
    // The second bound on the lengths, against the bytes that actually exist. A header can
    // checksum correctly and still describe a block whose entries never reached the device.
    // The GATHERED HEADER ARRAY is bounded here too, and separately: it is a length a crash
    // can lie about and an array this reader is about to allocate, so the file must actually
    // hold 32 bytes per claimed entry before one of them is believed. Each term is bounded
    // before it is added, so no sum can wrap on the way to the comparison.
    const int64_t headersBytes = (int64_t)entryCount * (int64_t)ENTRY_HEADER_BYTES;
    if(headersBytes > remainingAfterHeader)
      break;
    if(totalPayloadBytes > (uint64_t)(remainingAfterHeader - headersBytes))
      break;
    const int64_t regionBytes = (int64_t)totalPayloadBytes + headersBytes;

    const int64_t regionStart = offset + (int64_t)BLOCK_HEADER_BYTES;
    uint64_t actualChecksum = 0;
    bool readFailed = false;
    if(!checksumRegion(f, regionStart, regionBytes, contextHash, actualChecksum, readFailed)) {
      if(readFailed)
        throw StringError(
          "NNEvalContainer: could not read the entries of a block of " + path + " at offset " +
          Global::int64ToString(regionStart) + ". This is a read failure, not a torn tail, and "
          "the file is left exactly as it was found."
        );
      break;
    }
    if(get64(blockHeader + 16) != actualChecksum)
      break;

    // The block is whole. Only now are its entries decoded -- and only now can a malformed
    // entry be a REFUSAL rather than a torn tail, because the checksum has proven the bytes
    // are the bytes the writer wrote.
    if(!seekTo(f.get(), regionStart))
      throw StringError("NNEvalContainer: could not seek within " + path + ".");
    // The gathered header array, read in one go. Its size was bounded against the bytes that
    // actually exist before this line, so it is 32 bytes per entry the file really holds --
    // 1.46 MB at the operator's median card, 9.3 MB at his largest -- and never a length a
    // crash chose.
    headerArray.resize((size_t)headersBytes);
    if(headersBytes > 0 &&
       std::fread(headerArray.data(), 1, (size_t)headersBytes, f.get()) != (size_t)headersBytes)
      throw StringError("NNEvalContainer: could not re-read a verified block key index of " + path + ".");

    onBlock(f.get(), headerArray.data(), entryCount, regionStart, regionStart + headersBytes, totalPayloadBytes);

    result.blocksApplied += 1;
    result.entriesApplied += (int64_t)entryCount;
    const int64_t blockStart = offset;
    offset += (int64_t)BLOCK_HEADER_BYTES + regionBytes;
    result.intactEndOffset = offset;
    // DROP-BEHIND, one block at a time. The block has been checksummed, its header array
    // read and its visitor run, so nothing below will look at these bytes again -- and at
    // 10-20 GB per card against a page cache that holds a fraction of one, keeping them
    // would evict the machine's working set to hold bytes nobody will read. Per block rather
    // than once at the end, so the resident footprint stays bounded by one block DURING the
    // load and not merely after it.
    f.dropCachedRange(blockStart, offset - blockStart);
    // The visitor was free to seek; the next block header is read from where the framing
    // says it is, never from wherever the visitor left the handle.
    if(!seekTo(f.get(), offset))
      throw StringError("NNEvalContainer: could not seek within " + path + ".");
  }

  result.tornTailBytes = fileSize - result.intactEndOffset;
  // The two ranges the per-block drop cannot reach: the file header, which is read before
  // the first block, and a torn trailing region, which checksumRegion streamed through page
  // cache before the loop gave up on it. The second is the one that matters -- a torn tail
  // is a partial block and a block here is tens of megabytes.
  f.dropCachedRange(0, headerBytes);
  if(result.tornTailBytes > 0)
    f.dropCachedRange(result.intactEndOffset, result.tornTailBytes);
  return result;
}

// The FULL read: every verified block's payloads decoded and merged into one live set.
struct ScannedEntries {
  std::vector<std::unique_ptr<NNOutput>> entries;
  ScanResult scan;
};

ScannedEntries scanEntries(
  const std::string& path,
  const std::string& context,
  uint64_t contextHash,
  const std::string& modelInternalName,
  int modelVersion
) {
  ScannedEntries out;
  KeyIndex indexOfKey;
  std::vector<uint8_t> entryPayload;
  out.scan = scanContainer(
    path, context, contextHash, modelInternalName, modelVersion,
    [&](std::FILE* f, const uint8_t* headerArray, uint32_t entryCount,
        int64_t /*headerArrayFileOffset*/, int64_t /*payloadRegionFileOffset*/, uint64_t totalPayloadBytes) {
      walkBlockHeaderArray(
        headerArray, entryCount, totalPayloadBytes, path,
        [&](uint32_t /*i*/, const DecodedEntryHeader& h, int64_t /*payloadOffset*/) {
          entryPayload.resize((size_t)h.payloadBytes);
          if(h.payloadBytes > 0 &&
             std::fread(entryPayload.data(), 1, (size_t)h.payloadBytes, f) != (size_t)h.payloadBytes)
            throw StringError("NNEvalContainer: could not re-read a verified entry payload of " + path + ".");
          applyEntry(out.entries, indexOfKey, decodeEntryToHeap(h, entryPayload.data(), path));
        }
      );
    }
  );
  return out;
}

// The FRAMING-ONLY read: where the intact part of the file ends, and nothing else.
//
// It is the cheapest of the three scans and it is what the TORN-TAIL REPAIR needs, which is
// the only question the write path asks before it appends. It decodes no payload and builds
// no entry, holding nothing per entry beyond the header array the scan already read -- and
// it still WALKS every block's header array, so every refusal a malformed file can earn is
// still earned here, at the same moment it was before.
//
// IT STILL READS AND CHECKSUMS EVERY BYTE OF THE FILE, AND A CHEAPER SOUND ANSWER DOES
// EXIST. Saying otherwise would be the comfortable version of this comment, so: every block
// header self-checksums and states its region's exact length, so the framing could be walked
// header to header in O(blocks) thirty-two-byte reads, checksumming only the LAST block's
// region -- and that is SOUND for the question the write path asks, by induction on fsync:
// every append fsyncs the block it wrote, so an interior block was durable before the append
// that followed it, and the only block a crash can leave partial is the last one.
//
// IT WAS NOT TAKEN, AND THE REASON IS A TRADE RATHER THAN AN IMPOSSIBILITY. That walk
// answers "did a crash tear the tail" and stops answering "is any block in this file
// damaged". Today an interior block that lost bytes to bit rot is found by the writer, which
// repairs by discarding from that block onward; under a header-only walk the writer would
// append past it and only readers would notice, silently losing every block after the rotten
// one with nothing recording that it happened. Trading a corruption detector for read
// bandwidth is a decision with its own charter and its own evidence, and it is not this
// one's to make quietly. The cost of not making it is stated plainly so it can be weighed:
// at 10-20 GB per card and a dump every fifteen minutes, this scan reads the whole card each
// time.
//
// WHAT THIS REPLACED, because the difference is the whole point: appendBlock used to ask
// scanEntries, which decodes every payload in the file into a heap NNOutput and merges them
// into the live set, purely to hand that live set to a rewrite. At the deployment's 10-20 GB
// per card that is the entire card materialised in memory, on every dump, to decide whether
// the last few kilobytes were torn -- and on a machine smaller than the card it is not slow
// but impossible.
ScanResult scanFraming(
  const std::string& path,
  const std::string& context,
  uint64_t contextHash,
  const std::string& modelInternalName,
  int modelVersion
) {
  return scanContainer(
    path, context, contextHash, modelInternalName, modelVersion,
    [&](std::FILE* /*f*/, const uint8_t* headerArray, uint32_t entryCount,
        int64_t /*headerArrayFileOffset*/, int64_t /*payloadRegionFileOffset*/,
        uint64_t totalPayloadBytes) {
      walkBlockHeaderArray(
        headerArray, entryCount, totalPayloadBytes, path,
        [](uint32_t /*i*/, const DecodedEntryHeader& /*h*/, int64_t /*payloadOffset*/) {}
      );
    }
  );
}

// The KEY-SET read: 32 bytes per entry, no payload decoded and none held.
struct ScannedLocations {
  std::vector<NNEvalContainerEntryLocation> locations;
  ScanResult scan;
};

ScannedLocations scanLocations(
  const std::string& path,
  const std::string& context,
  uint64_t contextHash,
  const std::string& modelInternalName,
  int modelVersion
) {
  ScannedLocations out;
  KeyIndex indexOfKey;
  out.scan = scanContainer(
    path, context, contextHash, modelInternalName, modelVersion,
    [&](std::FILE* /*f*/, const uint8_t* headerArray, uint32_t entryCount,
        int64_t headerArrayFileOffset, int64_t payloadRegionFileOffset, uint64_t totalPayloadBytes) {
      walkBlockHeaderArray(
        headerArray, entryCount, totalPayloadBytes, path,
        [&](uint32_t i, const DecodedEntryHeader& h, int64_t payloadOffset) {
          NNEvalContainerEntryLocation loc;
          loc.key = h.key;
          loc.nnXLen = h.nnXLen;
          loc.nnYLen = h.nnYLen;
          loc.hasOwnerMap = h.hasOwnerMap;
          loc.headerFileOffset = headerArrayFileOffset + (int64_t)i * (int64_t)ENTRY_HEADER_BYTES;
          loc.payloadFileOffset = payloadRegionFileOffset + payloadOffset;
          loc.payloadBytes = h.payloadBytes;
          applyLocation(out.locations, indexOfKey, loc);
        }
      );
    }
  );
  return out;
}

//-------------------------------------------------------------------------------------
// Writing
//-------------------------------------------------------------------------------------

// A block, prepared but for its payload bytes: the gathered header array in full, and the
// three figures the block header needs.
//
// The header array IS held in memory, and that is the deliberate asymmetry. It is 32 bytes
// per entry -- 1.46 MB at the operator's median card, 9.3 MB at his largest -- against the
// ~69-135 MB of payload it indexes, and it has to exist as a contiguous run in the file
// before any payload does. The payloads are the thing that must never be held, and they are
// not: they are encoded one at a time into a reused buffer, twice (once to checksum, once to
// write), which costs a second encode pass and saves two orders of magnitude of memory.
struct PreparedBlock {
  std::vector<uint8_t> headerArray;
  uint64_t totalPayloadBytes = 0;
  uint64_t entryChecksum = 0;
  int64_t regionBytes = 0;
};

// The one shape a block-writing pass takes. Both callers -- the appending path and the
// rewrite path -- iterate their own containers of entries, so the entry access is a callback
// rather than a container type, and the framing is stated once.
//
// EVERY REFUSAL AN ENTRY CAN EARN IS EARNED HERE, before the file is opened.
template<typename EntryAt>
PreparedBlock prepareBlock(size_t numEntries, EntryAt entryAt, uint64_t contextHash, const std::string& path) {
  if(numEntries > 0xFFFFFFFFull)
    throw StringError("NNEvalContainer: " + path + ": a dump of more than 2^32-1 entries cannot be framed.");
  PreparedBlock block;
  block.headerArray.assign(numEntries * ENTRY_HEADER_BYTES, 0);
  int64_t payloadOffset = 0;
  for(size_t i = 0; i < numEntries; i++) {
    const NNOutput& out = entryAt(i);
    const int64_t payloadBytes = verifiedPayloadBytesOf(out, path);
    encodeEntryHeader(block.headerArray.data() + i * ENTRY_HEADER_BYTES, out, payloadBytes, payloadOffset);
    payloadOffset += payloadBytes;
  }
  block.totalPayloadBytes = (uint64_t)payloadOffset;
  block.regionBytes = (int64_t)block.headerArray.size() + payloadOffset;

  // The checksum covers the block's entry bytes: the header array first, then the payloads,
  // in exactly the order they are written.
  NNCacheFileChecksum sum(contextHash);
  sum.update(block.headerArray.data(), block.headerArray.size());
  std::vector<uint8_t> buf;
  for(size_t i = 0; i < numEntries; i++) {
    const NNOutput& out = entryAt(i);
    encodePayload(buf, out, payloadBytesOf(out.nnXLen, out.nnYLen, out.whiteOwnerMap != NULL));
    sum.update(buf.data(), buf.size());
  }
  block.entryChecksum = sum.finish();
  return block;
}

template<typename EntryAt>
void writeBlock(std::FILE* f, size_t numEntries, EntryAt entryAt, const PreparedBlock& block, uint64_t contextHash, const std::string& path) {
  uint8_t blockHeader[BLOCK_HEADER_BYTES];
  encodeBlockHeader(blockHeader, (uint32_t)numEntries, block.totalPayloadBytes, block.entryChecksum, contextHash);
  if(std::fwrite(blockHeader, 1, BLOCK_HEADER_BYTES, f) != BLOCK_HEADER_BYTES)
    throw StringError("NNEvalContainer: could not write a block header to " + path + ".");
  if(block.headerArray.size() > 0 &&
     std::fwrite(block.headerArray.data(), 1, block.headerArray.size(), f) != block.headerArray.size())
    throw StringError("NNEvalContainer: could not write a block key index to " + path + ".");
  std::vector<uint8_t> buf;
  for(size_t i = 0; i < numEntries; i++) {
    const NNOutput& out = entryAt(i);
    encodePayload(buf, out, payloadBytesOf(out.nnXLen, out.nnYLen, out.whiteOwnerMap != NULL));
    if(std::fwrite(buf.data(), 1, buf.size(), f) != buf.size())
      throw StringError("NNEvalContainer: could not write an entry payload to " + path + ".");
  }
}

// Writes header + one block to a temp sibling, fsyncs it, renames it over `path`, and
// fsyncs the directory.
//
// THIS IS THE CRASH STORY FOR COMPACTION AND FOR TORN-TAIL REPAIR, and it is why neither
// ever writes in place. rename(2) is atomic: at every instant `path` names either the whole
// old file or the whole new one. A crash before the rename leaves the old file untouched and
// a stale temp that the next rewrite opens with "wb" and overwrites; nothing ever reads the
// temp. A crash after the rename but before the directory fsync is the case the directory
// fsync exists for.
void rewriteAsOneBlock(
  const std::string& path,
  const std::vector<std::unique_ptr<NNOutput>>& entries,
  uint64_t contextHash,
  int modelVersion,
  const std::string& modelInternalName
) {
  const std::string tempPath = path + ".compacting";
  const auto entryAt = [&entries](size_t i) -> const NNOutput& { return *entries[i]; };
  const PreparedBlock block = prepareBlock(entries.size(), entryAt, contextHash, path);

  {
    NNCacheFileHandle f(tempPath, "wb");
    if(!f.isOpen())
      throw StringError("NNEvalContainer: could not open " + tempPath + " for writing.");
    const std::vector<uint8_t> fileHeader = encodeFileHeader(contextHash, modelVersion, modelInternalName);
    if(std::fwrite(fileHeader.data(), 1, fileHeader.size(), f.get()) != fileHeader.size())
      throw StringError("NNEvalContainer: could not write the header of " + tempPath + ".");
    writeBlock(f.get(), entries.size(), entryAt, block, contextHash, tempPath);
    if(!f.flushAndSync())
      throw StringError("NNEvalContainer: could not fsync " + tempPath + ".");
    if(!f.closeNow())
      throw StringError("NNEvalContainer: could not close " + tempPath + ".");
  }

  FileUtils::rename(tempPath, path);
  NNCacheFileSync::directoryOf(path);
}

}  // namespace

//-------------------------------------------------------------------------------------
// NNEvalContainerContents
//-------------------------------------------------------------------------------------

NNEvalContainerContents::NNEvalContainerContents(
  std::vector<std::unique_ptr<NNOutput>> entries,
  NNEvalContainerTail tail,
  int64_t discardedTailBytes,
  int64_t blocksApplied,
  int64_t entriesApplied
)
  :entries_(std::move(entries)),
   tail_(tail),
   discardedTailBytes_(discardedTailBytes),
   blocksApplied_(blocksApplied),
   entriesApplied_(entriesApplied)
{}

NNEvalContainerContents NNEvalContainerContents::of(
  std::vector<std::unique_ptr<NNOutput>> entries,
  NNEvalContainerTail tail,
  int64_t discardedTailBytes,
  int64_t blocksApplied,
  int64_t entriesApplied
) {
  // The disposition and the byte count are coupled at construction, so "Truncated with
  // nothing discarded" is not a value a reader has to defend against (ADR-0012 P11: the
  // typed reason is TIED to the absence by a check, not merely written down beside it).
  const bool truncated = tail == NNEvalContainerTail::Truncated;
  if(truncated != (discardedTailBytes > 0))
    throw StringError(
      "NNEvalContainerContents: tail disposition and discarded byte count disagree -- " +
      std::string(truncated ? "Truncated" : "Intact") + " with " +
      Global::int64ToString(discardedTailBytes) + " discarded bytes."
    );
  if(discardedTailBytes < 0 || blocksApplied < 0 || entriesApplied < 0)
    throw StringError("NNEvalContainerContents: a count was negative.");
  return NNEvalContainerContents(
    std::move(entries), tail, discardedTailBytes, blocksApplied, entriesApplied);
}

std::vector<std::unique_ptr<NNOutput>> NNEvalContainerContents::takeEntries() {
  return std::move(entries_);
}

//-------------------------------------------------------------------------------------
// NNEvalContainerIndex and its sink
//-------------------------------------------------------------------------------------

NNEvalContainerEntrySink::~NNEvalContainerEntrySink() {}

NNEvalContainerIndex::NNEvalContainerIndex(
  std::vector<NNEvalContainerEntryLocation> entries,
  NNEvalContainerTail tail,
  int64_t discardedTailBytes,
  int64_t blocksApplied,
  int64_t entriesApplied
)
  :entries_(std::move(entries)),
   tail_(tail),
   discardedTailBytes_(discardedTailBytes),
   blocksApplied_(blocksApplied),
   entriesApplied_(entriesApplied)
{}

NNEvalContainerIndex NNEvalContainerIndex::of(
  std::vector<NNEvalContainerEntryLocation> entries,
  NNEvalContainerTail tail,
  int64_t discardedTailBytes,
  int64_t blocksApplied,
  int64_t entriesApplied
) {
  const bool truncated = tail == NNEvalContainerTail::Truncated;
  if(truncated != (discardedTailBytes > 0))
    throw StringError(
      "NNEvalContainerIndex: tail disposition and discarded byte count disagree -- " +
      std::string(truncated ? "Truncated" : "Intact") + " with " +
      Global::int64ToString(discardedTailBytes) + " discarded bytes."
    );
  if(discardedTailBytes < 0 || blocksApplied < 0 || entriesApplied < 0)
    throw StringError("NNEvalContainerIndex: a count was negative.");
  return NNEvalContainerIndex(
    std::move(entries), tail, discardedTailBytes, blocksApplied, entriesApplied);
}

int64_t NNEvalContainerIndex::totalPayloadBytes() const {
  int64_t bytes = 0;
  for(size_t i = 0; i < entries_.size(); i++)
    bytes += entries_[i].payloadBytes;
  return bytes;
}

//-------------------------------------------------------------------------------------
// NNEvalContainer
//-------------------------------------------------------------------------------------

NNEvalContainer::NNEvalContainer(
  std::string path,
  std::string directory,
  std::string context,
  std::string modelInternalName,
  int modelVersion,
  uint64_t contextHash
)
  :path_(std::move(path)),
   directory_(std::move(directory)),
   context_(std::move(context)),
   modelInternalName_(std::move(modelInternalName)),
   modelVersion_(modelVersion),
   contextHash_(contextHash)
{}

NNEvalContainer NNEvalContainer::forContextAndModel(
  const std::string& directory,
  const std::string& context,
  const std::string& modelInternalName,
  int modelVersion
) {
  NNCacheFileName::verify(context, "NNEvalContainer", "context name");
  NNCacheFileName::verify(modelInternalName, "NNEvalContainer", "model name");
  if(modelVersion < 0)
    throw StringError(
      "NNEvalContainer: model version " + Global::intToString(modelVersion) + " has no reading."
    );
  if(!FileUtils::isDirectory(directory))
    throw StringError("NNEvalContainer: '" + directory + "' is not an existing directory.");
  const std::string path = directory + "/" + context + "." + modelInternalName + ".nnevals";
  return NNEvalContainer(
    path, directory, context, modelInternalName, modelVersion, NNCacheFileName::hashOf(context)
  );
}

uint32_t NNEvalContainer::formatVersion() { return FORMAT_VERSION; }
size_t NNEvalContainer::fixedFileHeaderBytes() { return FIXED_FILE_HEADER_BYTES; }
size_t NNEvalContainer::blockHeaderBytes() { return BLOCK_HEADER_BYTES; }
size_t NNEvalContainer::entryHeaderBytes() { return ENTRY_HEADER_BYTES; }
int NNEvalContainer::defaultCompactionMultiple() { return DEFAULT_COMPACTION_MULTIPLE; }

int64_t NNEvalContainer::fileHeaderBytesFor(const std::string& modelInternalName) {
  NNCacheFileName::verify(modelInternalName, "NNEvalContainer", "model name");
  return (int64_t)FIXED_FILE_HEADER_BYTES + (int64_t)modelInternalName.size();
}

int64_t NNEvalContainer::payloadBytesFor(int nnXLen, int nnYLen, bool hasOwnerMap) {
  verifyShape(nnXLen, nnYLen, "a requested entry size");
  return payloadBytesOf(nnXLen, nnYLen, hasOwnerMap);
}

int64_t NNEvalContainer::bytesForEntry(int nnXLen, int nnYLen, bool hasOwnerMap) {
  return (int64_t)ENTRY_HEADER_BYTES + payloadBytesFor(nnXLen, nnYLen, hasOwnerMap);
}

NNEvalContainerContents NNEvalContainer::load() const {
  // SHARED, because this is a read: it admits every other reader and excludes every writer.
  // Without it a load concurrent with an append sees the half-written block as a torn tail and
  // silently discards it AND every block after it -- the writer's loss, suffered by the reader.
  const NNCacheFileLock lock =
    NNCacheFileLock::overContext(directory_, context_, NNCacheFileLockMode::Shared);
  (void)lock;
  ScannedEntries scan = scanEntries(path_, context_, contextHash_, modelInternalName_, modelVersion_);
  return NNEvalContainerContents::of(
    std::move(scan.entries),
    scan.scan.tornTailBytes > 0 ? NNEvalContainerTail::Truncated : NNEvalContainerTail::Intact,
    scan.scan.tornTailBytes,
    scan.scan.blocksApplied,
    scan.scan.entriesApplied
  );
}

NNEvalContainerIndex NNEvalContainer::loadIndex() const {
  const NNCacheFileLock lock =
    NNCacheFileLock::overContext(directory_, context_, NNCacheFileLockMode::Shared);
  (void)lock;
  ScannedLocations scan = scanLocations(path_, context_, contextHash_, modelInternalName_, modelVersion_);
  return NNEvalContainerIndex::of(
    std::move(scan.locations),
    scan.scan.tornTailBytes > 0 ? NNEvalContainerTail::Truncated : NNEvalContainerTail::Intact,
    scan.scan.tornTailBytes,
    scan.scan.blocksApplied,
    scan.scan.entriesApplied
  );
}

void NNEvalContainer::readEntriesInto(
  const std::vector<NNEvalContainerEntryLocation>& locations,
  NNEvalContainerEntrySink& sink
) const {
  if(locations.empty())
    return;

  // SHARED, and it matters more here than anywhere else on the read side: these locations were
  // read by an EARLIER call, so this is the window in which a writer could move the bytes they
  // point at. The header re-check below still refuses a file that changed between the two calls
  // -- the lock does not replace it -- but the lock is what keeps a compaction from landing in
  // the middle of this pass.
  const NNCacheFileLock lock =
    NNCacheFileLock::overContext(directory_, context_, NNCacheFileLockMode::Shared);
  (void)lock;

  NNCacheFileHandle f(path_, "rb");
  if(!f.isOpen())
    throw StringError(
      "NNEvalContainer: " + path_ + " could not be opened to read the entries a caller selected "
      "from it. It existed when its key set was read."
    );
  const int64_t fileSize = sizeOfOpenFile(f.get(), path_);
  if(!seekTo(f.get(), 0))
    throw StringError("NNEvalContainer: could not rewind " + path_ + ".");

  // THE SAME BOUNDARY load() APPLIES, re-applied here rather than assumed from the earlier
  // scan: this is a second open of a file another process may have replaced, and the model
  // check in particular is the one that keeps one net's evaluations from being read as
  // another's (ADR-0012 P2).
  if(fileSize < (int64_t)FIXED_FILE_HEADER_BYTES)
    throw StringError("NNEvalContainer: " + path_ + " is now too short to hold its own header.");
  std::vector<uint8_t> fixedHeader(FIXED_FILE_HEADER_BYTES);
  if(std::fread(fixedHeader.data(), 1, FIXED_FILE_HEADER_BYTES, f.get()) != FIXED_FILE_HEADER_BYTES)
    throw StringError("NNEvalContainer: could not read the header of " + path_ + ".");
  const int64_t headerBytes = (int64_t)get32(fixedHeader.data() + 12);
  if(fileSize < headerBytes)
    throw StringError("NNEvalContainer: " + path_ + " is now shorter than the header it declares.");
  std::vector<uint8_t> fileHeader((size_t)headerBytes);
  std::memcpy(fileHeader.data(), fixedHeader.data(), FIXED_FILE_HEADER_BYTES);
  const size_t nameBytes = (size_t)headerBytes - FIXED_FILE_HEADER_BYTES;
  if(nameBytes > 0 &&
     std::fread(fileHeader.data() + FIXED_FILE_HEADER_BYTES, 1, nameBytes, f.get()) != nameBytes)
    throw StringError("NNEvalContainer: could not read the model name in the header of " + path_ + ".");
  verifyFileHeader(fileHeader, path_, context_, contextHash_, modelInternalName_, modelVersion_);

  // ASCENDING FILE ORDER, whatever order the caller asked in. The loader's order is
  // descending popularity, which is scattered across the file; reading in that order would
  // turn one forward pass into tens of thousands of backward seeks.
  std::vector<size_t> order(locations.size());
  for(size_t i = 0; i < order.size(); i++)
    order[i] = i;
  std::sort(order.begin(), order.end(), [&locations](size_t x, size_t y) {
    if(locations[x].payloadFileOffset != locations[y].payloadFileOffset)
      return locations[x].payloadFileOffset < locations[y].payloadFileOffset;
    return x < y;
  });

  uint8_t entryHeader[ENTRY_HEADER_BYTES];
  std::vector<uint8_t> payload;
  for(size_t oi = 0; oi < order.size(); oi++) {
    const size_t i = order[oi];
    const NNEvalContainerEntryLocation& loc = locations[i];
    if(loc.payloadBytes < 0 || loc.headerFileOffset < 0 || loc.payloadFileOffset < 0 ||
       loc.payloadFileOffset + loc.payloadBytes > fileSize ||
       loc.headerFileOffset + (int64_t)ENTRY_HEADER_BYTES > fileSize)
      throw StringError(
        "NNEvalContainer: " + path_ + ": entry " + loc.key.toString() +
        " is located outside the file. The container changed under the caller."
      );

    if(!seekTo(f.get(), loc.headerFileOffset))
      throw StringError("NNEvalContainer: could not seek within " + path_ + ".");
    if(std::fread(entryHeader, 1, ENTRY_HEADER_BYTES, f.get()) != ENTRY_HEADER_BYTES)
      throw StringError("NNEvalContainer: could not read an entry header of " + path_ + ".");
    // THE KEY IS COMPARED FIRST, on the raw bytes, BEFORE the header is decoded at all.
    // That ordering is the whole point: if the file was rewritten between the key-set scan
    // and this read, the caller's offset now lands in the middle of some other entry's
    // payload, and decoding those bytes as a header produces a nonsense shape whose refusal
    // would blame the format for what is really a stale offset. The honest first question is
    // "is this still the entry I was told about", and only then "is this entry well formed"
    // (ADR-0021 Rule 1: the refusal observes the thing being claimed).
    const Hash128 keyOnDisk(get64(entryHeader + 0), get64(entryHeader + 8));
    if(!(keyOnDisk == loc.key))
      throw StringError(
        "NNEvalContainer: " + path_ + ": the entry at offset " +
        Global::int64ToString(loc.headerFileOffset) + " is now " + keyOnDisk.toString() +
        " and the caller's key set says it is " + loc.key.toString() +
        ". The container changed between reading its key set and reading its entries."
      );
    const DecodedEntryHeader h = decodeEntryHeader(entryHeader, path_);
    // And the rest of what the location says, for the same reason: a rewrite that happened
    // to leave this key's header at this offset can still have changed its shape.
    if(h.nnXLen != loc.nnXLen || h.nnYLen != loc.nnYLen ||
       h.hasOwnerMap != loc.hasOwnerMap || h.payloadBytes != loc.payloadBytes)
      throw StringError(
        "NNEvalContainer: " + path_ + ": entry " + h.key.toString() + " at offset " +
        Global::int64ToString(loc.headerFileOffset) +
        " no longer has the shape the caller's key set records for it. "
        "The container changed between reading its key set and reading its entries."
      );

    if(!seekTo(f.get(), loc.payloadFileOffset))
      throw StringError("NNEvalContainer: could not seek within " + path_ + ".");
    payload.resize((size_t)h.payloadBytes);
    if(h.payloadBytes > 0 &&
       std::fread(payload.data(), 1, (size_t)h.payloadBytes, f.get()) != (size_t)h.payloadBytes)
      throw StringError("NNEvalContainer: could not read an entry payload of " + path_ + ".");

    NNOutput& out = sink.outputFor(i);
    out.whiteOwnerMap =
      h.hasOwnerMap ? sink.ownerMapFor(i, (size_t)h.nnXLen * (size_t)h.nnYLen) : NULL;
    decodePayloadInto(out, h, payload.data(), path_);

    // This entry is now in the caller's memory and its bytes will never be read again, so
    // the pages holding it are dropped as the pass goes.
    //
    // ONLY THE RANGES THIS PASS ACTUALLY READ, and never the whole file. posix_fadvise
    // invalidates clean pages of the INODE, globally -- not of this descriptor and not of
    // this process -- and the deployment deliberately puts several engine processes on one
    // cache directory. A whole-file drop here would discard a peer's freshly-read pages
    // along with this pass's own, sending it back to the device for bytes it had just
    // fetched. The ranges this pass did not touch are not its to forget.
    f.dropCachedRange(loc.headerFileOffset, (int64_t)ENTRY_HEADER_BYTES);
    f.dropCachedRange(loc.payloadFileOffset, h.payloadBytes);
  }

  // NO LARGE SEQUENTIAL BUFFER IN THIS FUNCTION, unlike every scan: it reads a selection at
  // scattered offsets in ascending order, and a megabyte of readahead per two-kilobyte entry
  // would fetch far more than it used. The buffer is a decision about a streaming read and
  // this is not one.
}

NNEvalContainerContents NNEvalContainer::compact() const {
  // EXCLUSIVE. Compaction is the operation the dedicated lock file exists for: it writes a temp
  // sibling and renames it over path_, so a lock held on path_'s own inode would survive the
  // rename attached to an inode that no longer has the name, and the next process to open the
  // path would lock a different inode entirely. See NNCacheFileLock.
  const NNCacheFileLock lock =
    NNCacheFileLock::overContext(directory_, context_, NNCacheFileLockMode::Exclusive);
  (void)lock;
  return compactUnlocked();
}

NNEvalContainerContents NNEvalContainer::compactUnlocked() const {
  ScannedEntries scan = scanEntries(path_, context_, contextHash_, modelInternalName_, modelVersion_);
  rewriteAsOneBlock(path_, scan.entries, contextHash_, modelVersion_, modelInternalName_);
  const int64_t liveSet = (int64_t)scan.entries.size();
  return NNEvalContainerContents::of(
    std::move(scan.entries),
    NNEvalContainerTail::Intact,
    0,
    1,
    liveSet
  );
}

NNCacheFileMaintenance NNEvalContainer::compactIfNeeded(int liveSetMultiple) const {
  if(liveSetMultiple < 1)
    throw StringError(
      "NNEvalContainer: a compaction multiple of " + Global::intToString(liveSetMultiple) +
      " has no reading; it must be at least 1."
    );
  // EXCLUSIVE, and taken BEFORE the deciding scan rather than around the rewrite alone. The
  // decision and the rewrite it authorises must see the same file: a writer admitted between
  // them would have its block rewritten away by a compaction that never read it, or, if it
  // appended after the scan decided not to compact, would be judged against a file that no
  // longer exists.
  const NNCacheFileLock lock =
    NNCacheFileLock::overContext(directory_, context_, NNCacheFileLockMode::Exclusive);
  (void)lock;
  // The key-set scan is enough to decide this: the trigger reads the live-set size and the
  // physical entry count, and neither is a payload fact. It reads 32 bytes per entry rather
  // than the whole ~69 MB card the full read would decode and discard.
  const ScannedLocations scan = scanLocations(path_, context_, contextHash_, modelInternalName_, modelVersion_);
  const int64_t liveSet = (int64_t)scan.locations.size();
  // A torn tail is repaired whether or not the size trigger fires: leaving it would make the
  // next append land at an offset no loader reaches.
  const bool torn = scan.scan.tornTailBytes > 0;
  const bool overMultiple = liveSet > 0 && scan.scan.entriesApplied > (int64_t)liveSetMultiple * liveSet;
  if(!torn && !overMultiple)
    return NNCacheFileMaintenance::Nothing;
  // A TORN TAIL ALONE IS REPAIRED BY TRUNCATION, NEVER BY COMPACTION. The two are separate
  // decisions and only one of them was made here: the size trigger did not fire, so nothing
  // authorises rewriting a card the operator did not ask to have rewritten. Compaction, when
  // it IS authorised below, subsumes the repair -- it writes a fresh file from the intact
  // part and the torn tail is gone with the old inode.
  if(torn && !overMultiple) {
    NNCacheFileTruncate::toLength(path_, scan.scan.intactEndOffset);
    return NNCacheFileMaintenance::TruncatedTornTail;
  }
  // DELEGATES rather than rewriting here, so "rewrite this container" has exactly one home
  // and a change to it cannot leave this path saying the old thing. The cost is one extra
  // scan on the compacting path, on an explicitly-invoked dump. It delegates to the UNLOCKED
  // form because the exclusive lock is already held above; compact() would ask for it a second
  // time and wait out its own deadline against itself.
  const NNEvalContainerContents written = compactUnlocked();
  (void)written;
  return NNCacheFileMaintenance::Compacted;
}

NNEvalContainerAppendResult NNEvalContainer::appendBlock(
  const std::vector<std::shared_ptr<const NNOutput>>& entries
) const {
  for(size_t i = 0; i < entries.size(); i++) {
    if(entries[i] == nullptr)
      throw StringError(
        "NNEvalContainer: " + path_ + ": entry " + Global::uint64ToString(i) +
        " of this dump is null. A missing evaluation is not a storable fact."
      );
  }

  // EXCLUSIVE, and it is what makes a dump atomic against another process's dump. A block is
  // written through a reused per-entry buffer -- many small writes, not one -- so two unlocked
  // appends interleave IN THE MIDDLE of the file, and a reader stops at the first block that
  // fails its checksum and discards the entire remainder. One interleaved append therefore
  // costs every block written after it, including blocks that were themselves perfectly good.
  // Taken after the null check so a caller's own malformed dump is refused without waiting on
  // anybody else's lock.
  const NNCacheFileLock lock =
    NNCacheFileLock::overContext(directory_, context_, NNCacheFileLockMode::Exclusive);
  (void)lock;

  NNEvalContainerAppendResult result;
  result.bytesAppended = 0;
  result.tornTailBytesDiscarded = 0;
  result.tailRepair = NNCacheFileTailRepair::NotNeeded;

  // Scan first. A torn tail must be repaired BEFORE anything is appended: an append past a
  // torn tail lands at an offset no loader ever reaches, so every subsequent dump would be
  // silently lost while every call reported success.
  //
  // THE FRAMING SCAN AND NOT THE FULL READ, because the only thing this needs from the file
  // is where its intact part ends. The full read decoded every payload in the file into the
  // heap to hand the live set to a rewrite; there is no rewrite here to hand it to.
  const ScanResult scan = scanFraming(path_, context_, contextHash_, modelInternalName_, modelVersion_);
  if(scan.tornTailBytes > 0) {
    // TRUNCATION, NOT A REWRITE. The bytes that survive are identical either way -- the
    // rewrite re-encoded the live set, which is a subset of this same intact prefix, and
    // neither keeps a byte of the torn tail -- so the whole difference is what reaches the
    // device: a metadata operation here, and the entire 10-20 GB card there. Under a
    // lifecycle where engines are arbitrarily SIGKILLed and dump every fifteen minutes, that
    // was a full card rewritten to flash for every kill that landed inside a dump. See
    // NNCacheFileTruncate for the crash story and for why this does not compact.
    NNCacheFileTruncate::toLength(path_, scan.intactEndOffset);
    result.tornTailBytesDiscarded = scan.tornTailBytes;
    // The file was SHORTENED, not rewritten, and the disposition says so outright rather
    // than leaving an operator to infer it from the byte count beside it.
    result.tailRepair = NNCacheFileTailRepair::Truncated;
  }

  const auto entryAt = [&entries](size_t i) -> const NNOutput& { return *entries[i]; };
  // Every refusal an entry can earn is earned HERE, before the file is opened, so a rejected
  // dump never leaves a half-written block behind.
  const PreparedBlock block = prepareBlock(entries.size(), entryAt, contextHash_, path_);

  // A FILE WHOSE INTACT PART IS EMPTY NEEDS A HEADER, exactly as an absent one does, and the
  // question is asked of the SCAN rather than of the filesystem for that reason. Two states
  // reach here holding a file that exists and carries nothing this build can read: a crash
  // during the very first dump, which the truncation above has just shortened to zero bytes,
  // and a zero-byte file left by anything else. Opening either with "ab" would append a
  // block to a file with no file header, which no loader would ever accept -- the rewrite
  // this replaced wrote a fresh header in that case, and this is where that duty now lives.
  const bool needsFileHeader = !scan.fileExists || scan.intactEndOffset == 0;
  NNCacheFileHandle f(path_, needsFileHeader ? "wb" : "ab");
  if(!f.isOpen())
    throw StringError("NNEvalContainer: could not open " + path_ + " for appending.");
  if(needsFileHeader) {
    const std::vector<uint8_t> fileHeader = encodeFileHeader(contextHash_, modelVersion_, modelInternalName_);
    if(std::fwrite(fileHeader.data(), 1, fileHeader.size(), f.get()) != fileHeader.size())
      throw StringError("NNEvalContainer: could not write the header of " + path_ + ".");
    result.bytesAppended += (int64_t)fileHeader.size();
  }
  writeBlock(f.get(), entries.size(), entryAt, block, contextHash_, path_);
  result.bytesAppended += (int64_t)BLOCK_HEADER_BYTES + block.regionBytes;
  if(!f.flushAndSync())
    throw StringError("NNEvalContainer: could not fsync " + path_ + ".");
  // A newly created file needs its directory entry forced too; an append to an existing one
  // does not, because the entry is already durable.
  if(!scan.fileExists)
    NNCacheFileSync::directoryOf(path_);
  return result;
}
