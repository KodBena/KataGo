#include "../neuralnet/nnevalcontainer.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
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

// Decodes one entry from its already-checksum-verified bytes.
//
// Every refusal below is judged only AFTER the block's checksum has proven these are the
// bytes the writer wrote, so a refusal here means the file is genuinely not one this build
// can read -- never that a crash happened.
std::unique_ptr<NNOutput> decodeEntry(
  const uint8_t* header,
  const uint8_t* payload,
  int64_t payloadBytes,
  int64_t expectedPayloadOffset,
  const std::string& path
) {
  const Hash128 key(get64(header + 0), get64(header + 8));
  const uint16_t flags = (uint16_t)((uint16_t)header[16] | ((uint16_t)header[17] << 8));
  if((flags & (uint16_t)~(uint16_t)FLAG_HAS_OWNERMAP) != 0)
    throw StringError(
      "NNEvalContainer: " + path + ": entry " + key.toString() + " sets flag bits this build "
      "does not define. It was written by a later version of this format."
    );
  // The stored payload offset is a DERIVED quantity -- the running sum of the preceding
  // payload sizes -- so it is recomputed by the caller and checked here rather than trusted.
  // A second statement of a fact is only safe while something refuses to let the two
  // disagree (ADR-0012 P1).
  const uint64_t storedOffset = get64(header + 24);
  if(storedOffset != (uint64_t)expectedPayloadOffset)
    throw StringError(
      "NNEvalContainer: " + path + ": entry " + key.toString() + " places its payload at offset " +
      Global::uint64ToString(storedOffset) + " and the entries before it end at " +
      Global::int64ToString(expectedPayloadOffset) + "."
    );
  const int nnXLen = (int)header[18];
  const int nnYLen = (int)header[19];
  verifyShape(nnXLen, nnYLen, path + ": entry " + key.toString());

  const bool hasOwnerMap = (flags & FLAG_HAS_OWNERMAP) != 0;
  const int64_t area = (int64_t)nnXLen * (int64_t)nnYLen;
  const int64_t expected = payloadBytesOf(nnXLen, nnYLen, hasOwnerMap);
  if(payloadBytes != expected)
    throw StringError(
      "NNEvalContainer: " + path + ": entry " + key.toString() + " declares " +
      Global::int64ToString(payloadBytes) + " payload bytes but its " +
      Global::intToString(nnXLen) + "x" + Global::intToString(nnYLen) + " shape" +
      (hasOwnerMap ? " with an ownership map" : " without an ownership map") + " is " +
      Global::int64ToString(expected) + " bytes."
    );

  std::unique_ptr<NNOutput> out(new NNOutput());
  out->nnHash = key;
  const uint8_t* v = payload;
  out->whiteWinProb = getF32(v + 0);
  out->whiteLossProb = getF32(v + 4);
  out->whiteNoResultProb = getF32(v + 8);
  out->whiteScoreMean = getF32(v + 12);
  out->whiteScoreMeanSq = getF32(v + 16);
  out->whiteLead = getF32(v + 20);
  out->varTimeLeft = getF32(v + 24);
  out->shorttermWinlossError = getF32(v + 28);
  out->shorttermScoreError = getF32(v + 32);
  out->policyOptimismUsed = getF32(v + 36);
  v += NUM_SCALARS * 4;

  out->nnXLen = nnXLen;
  out->nnYLen = nnYLen;
  // The slots past the board are ZEROED rather than left as whatever the allocation held.
  // They are not a fact the file carries, and an uninitialised slot is exactly the kind of
  // value that reads as plausible later.
  for(int i = 0; i < NNPos::MAX_NN_POLICY_SIZE; i++)
    out->policyProbs[i] = 0.0f;
  for(int64_t i = 0; i <= area; i++)
    out->policyProbs[i] = getF32(v + (size_t)i * 4);
  v += (size_t)(area + 1) * 4;

  if(hasOwnerMap) {
    // ~NNOutput delete[]s this, so it is allocated with new[] and with nothing else.
    out->whiteOwnerMap = new float[(size_t)area];
    for(int64_t i = 0; i < area; i++)
      out->whiteOwnerMap[i] = getF32(v + (size_t)i * 4);
  }
  out->noisedPolicyProbs = NULL;
  return out;
}

//-------------------------------------------------------------------------------------
// The merge
//-------------------------------------------------------------------------------------

typedef std::map<std::pair<uint64_t,uint64_t>, size_t> KeyIndex;

// LAST-WINS PER KEY, EXCEPT THAT AN ENTRY WITHOUT AN OWNERSHIP MAP NEVER SUPERSEDES ONE
// WITH. The exception is the store-side face of the live supersession rule: an entry lacking
// a requested ownership map costs a full re-evaluation on every hit, so letting a later
// ownermap-less re-evaluation overwrite an ownermap-carrying one would turn a restored level
// 0 into a bleed-out. There is exactly one implementation of this rule and both the reader
// and compaction go through it (ADR-0012 P1).
void applyEntry(
  std::vector<std::unique_ptr<NNOutput>>& entries,
  KeyIndex& indexOfKey,
  std::unique_ptr<NNOutput> incoming
) {
  const std::pair<uint64_t,uint64_t> mapKey(incoming->nnHash.hash0, incoming->nnHash.hash1);
  const KeyIndex::iterator it = indexOfKey.find(mapKey);
  if(it == indexOfKey.end()) {
    indexOfKey[mapKey] = entries.size();
    entries.push_back(std::move(incoming));
    return;
  }
  const NNOutput& existing = *entries[it->second];
  if(existing.whiteOwnerMap != NULL && incoming->whiteOwnerMap == NULL)
    return;  // the fuller entry stands
  entries[it->second] = std::move(incoming);
}

//-------------------------------------------------------------------------------------
// The scan
//-------------------------------------------------------------------------------------

// One pass over a container. Both load() and appendBlock() go through this, so there is
// exactly one implementation of "where does the intact part of this file end" (ADR-0012 P1).
struct ScanResult {
  std::vector<std::unique_ptr<NNOutput>> entries;
  int64_t blocksApplied = 0;
  int64_t entriesApplied = 0;
  // Byte offset one past the last intact block. Equal to the file size when the tail is
  // intact.
  int64_t intactEndOffset = 0;
  int64_t tornTailBytes = 0;
  bool fileExists = false;
};

// Streams `regionBytes` bytes from the current position through a bounded buffer and returns
// their checksum, or false if the file was short. Nothing is retained: this is how a 69 MB
// block is verified without being held.
bool checksumRegion(std::FILE* f, int64_t regionBytes, uint64_t seed, uint64_t& ret) {
  NNCacheFileChecksum sum(seed);
  std::vector<uint8_t> buf(STREAM_BUFFER_BYTES);
  int64_t left = regionBytes;
  while(left > 0) {
    const size_t want = (size_t)std::min<int64_t>(left, (int64_t)buf.size());
    if(std::fread(buf.data(), 1, want, f) != want)
      return false;
    sum.update(buf.data(), want);
    left -= (int64_t)want;
  }
  ret = sum.finish();
  return true;
}

ScanResult scanContainer(
  const std::string& path,
  const std::string& context,
  uint64_t contextHash,
  const std::string& modelInternalName,
  int modelVersion
) {
  ScanResult result;

  NNCacheFileHandle f(path, "rb");
  if(!f.isOpen())
    return result;  // Absent is a normal answer: no dump has happened here yet.
  result.fileExists = true;

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

  KeyIndex indexOfKey;
  std::vector<uint8_t> headerArray;
  std::vector<uint8_t> entryPayload;

  while(true) {
    const int64_t remaining = fileSize - offset;
    if(remaining <= 0)
      break;
    if(remaining < (int64_t)BLOCK_HEADER_BYTES)
      break;  // torn: not even a whole block header left

    uint8_t blockHeader[BLOCK_HEADER_BYTES];
    if(std::fread(blockHeader, 1, BLOCK_HEADER_BYTES, f.get()) != BLOCK_HEADER_BYTES)
      break;  // torn: the file shrank under us, or the read failed

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
    if(!checksumRegion(f.get(), regionBytes, contextHash, actualChecksum))
      break;
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

    int64_t payloadOffset = 0;
    for(uint32_t i = 0; i < entryCount; i++) {
      const uint8_t* entryHeader = headerArray.data() + (size_t)i * ENTRY_HEADER_BYTES;
      const int64_t payloadBytes = (int64_t)get32(entryHeader + 20);
      if(payloadBytes > (int64_t)totalPayloadBytes - payloadOffset)
        throw StringError(
          "NNEvalContainer: " + path + ": an entry declares " + Global::int64ToString(payloadBytes) +
          " payload bytes that run past the payload region its block declares."
        );
      entryPayload.resize((size_t)payloadBytes);
      if(payloadBytes > 0 && std::fread(entryPayload.data(), 1, (size_t)payloadBytes, f.get()) != (size_t)payloadBytes)
        throw StringError("NNEvalContainer: could not re-read a verified entry payload of " + path + ".");
      applyEntry(result.entries, indexOfKey,
                 decodeEntry(entryHeader, entryPayload.data(), payloadBytes, payloadOffset, path));
      payloadOffset += payloadBytes;
    }
    if(payloadOffset != (int64_t)totalPayloadBytes)
      throw StringError(
        "NNEvalContainer: " + path + ": a block's payloads occupy " + Global::int64ToString(payloadOffset) +
        " bytes and its header declares a payload region of " + Global::uint64ToString(totalPayloadBytes) + "."
      );

    result.blocksApplied += 1;
    result.entriesApplied += (int64_t)entryCount;
    offset += (int64_t)BLOCK_HEADER_BYTES + regionBytes;
    result.intactEndOffset = offset;
  }

  result.tornTailBytes = fileSize - result.intactEndOffset;
  return result;
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
// NNEvalContainer
//-------------------------------------------------------------------------------------

NNEvalContainer::NNEvalContainer(
  std::string path,
  std::string context,
  std::string modelInternalName,
  int modelVersion,
  uint64_t contextHash
)
  :path_(std::move(path)),
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
  return NNEvalContainer(path, context, modelInternalName, modelVersion, NNCacheFileName::hashOf(context));
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
  ScanResult scan = scanContainer(path_, context_, contextHash_, modelInternalName_, modelVersion_);
  return NNEvalContainerContents::of(
    std::move(scan.entries),
    scan.tornTailBytes > 0 ? NNEvalContainerTail::Truncated : NNEvalContainerTail::Intact,
    scan.tornTailBytes,
    scan.blocksApplied,
    scan.entriesApplied
  );
}

NNEvalContainerContents NNEvalContainer::compact() const {
  ScanResult scan = scanContainer(path_, context_, contextHash_, modelInternalName_, modelVersion_);
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

bool NNEvalContainer::compactIfNeeded(int liveSetMultiple) const {
  if(liveSetMultiple < 1)
    throw StringError(
      "NNEvalContainer: a compaction multiple of " + Global::intToString(liveSetMultiple) +
      " has no reading; it must be at least 1."
    );
  const ScanResult scan = scanContainer(path_, context_, contextHash_, modelInternalName_, modelVersion_);
  const int64_t liveSet = (int64_t)scan.entries.size();
  // A torn tail is repaired whether or not the size trigger fires: leaving it would make the
  // next append land at an offset no loader reaches.
  const bool torn = scan.tornTailBytes > 0;
  const bool overMultiple = liveSet > 0 && scan.entriesApplied > (int64_t)liveSetMultiple * liveSet;
  if(!torn && !overMultiple)
    return false;
  // DELEGATES rather than rewriting here, so "rewrite this container" has exactly one home
  // and a change to it cannot leave this path saying the old thing. The cost is one extra
  // scan on the compacting path, on an explicitly-invoked dump.
  const NNEvalContainerContents written = compact();
  (void)written;
  return true;
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

  NNEvalContainerAppendResult result;
  result.bytesAppended = 0;
  result.tornTailBytesDiscarded = 0;
  result.rewroteTheFile = false;

  // Scan first. A torn tail must be repaired BEFORE anything is appended: an append past a
  // torn tail lands at an offset no loader ever reaches, so every subsequent dump would be
  // silently lost while every call reported success.
  const ScanResult scan = scanContainer(path_, context_, contextHash_, modelInternalName_, modelVersion_);
  if(scan.tornTailBytes > 0) {
    rewriteAsOneBlock(path_, scan.entries, contextHash_, modelVersion_, modelInternalName_);
    result.tornTailBytesDiscarded = scan.tornTailBytes;
    result.rewroteTheFile = true;
  }

  const auto entryAt = [&entries](size_t i) -> const NNOutput& { return *entries[i]; };
  // Every refusal an entry can earn is earned HERE, before the file is opened, so a rejected
  // dump never leaves a half-written block behind.
  const PreparedBlock block = prepareBlock(entries.size(), entryAt, contextHash_, path_);

  const bool fileExists = FileUtils::exists(path_);
  NNCacheFileHandle f(path_, fileExists ? "ab" : "wb");
  if(!f.isOpen())
    throw StringError("NNEvalContainer: could not open " + path_ + " for appending.");
  if(!fileExists) {
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
  if(!fileExists)
    NNCacheFileSync::directoryOf(path_);
  return result;
}
