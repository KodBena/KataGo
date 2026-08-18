#include "../neuralnet/nncachefileformat.h"

#include <cstring>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#include <fcntl.h>
#endif

#include "../core/global.h"

// The byte layer shared by the count log and the evaluation container. See
// nncachefileformat.h for why each of the three pieces has one home rather than two.

namespace {
  const size_t MAX_NAME_LEN = 128;
}

//-------------------------------------------------------------------------------------
// Little-endian packing
//-------------------------------------------------------------------------------------

void NNCacheFileBytes::put32(uint8_t* p, uint32_t v) {
  p[0] = (uint8_t)(v      ); p[1] = (uint8_t)(v >>  8);
  p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}

void NNCacheFileBytes::put64(uint8_t* p, uint64_t v) {
  for(int i = 0; i < 8; i++)
    p[i] = (uint8_t)(v >> (8 * i));
}

uint32_t NNCacheFileBytes::get32(const uint8_t* p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

uint64_t NNCacheFileBytes::get64(const uint8_t* p) {
  uint64_t v = 0;
  for(int i = 0; i < 8; i++)
    v |= ((uint64_t)p[i]) << (8 * i);
  return v;
}

void NNCacheFileBytes::putF32(uint8_t* p, float v) {
  static_assert(sizeof(float) == 4, "this format stores a float as its 4 IEEE-754 bytes");
  uint32_t bits;
  std::memcpy(&bits, &v, 4);
  put32(p, bits);
}

float NNCacheFileBytes::getF32(const uint8_t* p) {
  const uint32_t bits = get32(p);
  float v;
  std::memcpy(&v, &bits, 4);
  return v;
}

//-------------------------------------------------------------------------------------
// The checksum
//-------------------------------------------------------------------------------------

NNCacheFileChecksum::NNCacheFileChecksum(uint64_t seed)
  :state_(0xcbf29ce484222325ULL ^ seed)
{}

void NNCacheFileChecksum::update(const uint8_t* data, size_t len) {
  uint64_t h = state_;
  for(size_t i = 0; i < len; i++) {
    h ^= (uint64_t)data[i];
    h *= 0x100000001b3ULL;
  }
  state_ = h;
}

uint64_t NNCacheFileChecksum::finish() const {
  uint64_t h = state_;
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdULL;
  h ^= h >> 33;
  h *= 0xc4ceb9fe1a85ec53ULL;
  h ^= h >> 33;
  return h;
}

uint64_t NNCacheFileChecksum::of(const uint8_t* data, size_t len, uint64_t seed) {
  NNCacheFileChecksum sum(seed);
  sum.update(data, len);
  return sum.finish();
}

//-------------------------------------------------------------------------------------
// The path-component name boundary
//-------------------------------------------------------------------------------------

size_t NNCacheFileName::maxLength() { return MAX_NAME_LEN; }

void NNCacheFileName::verify(const std::string& name, const char* whoIsAsking, const char* whatItIs) {
  const std::string who = std::string(whoIsAsking) + ": the " + whatItIs;
  if(name.empty())
    throw StringError(who + " is empty.");
  if(name.size() > MAX_NAME_LEN)
    throw StringError(
      who + " is " + Global::uint64ToString(name.size()) +
      " characters; at most " + Global::uint64ToString(MAX_NAME_LEN) + " are allowed."
    );
  if(name == "." || name == "..")
    throw StringError(who + " is '" + name + "', which is not a usable name.");
  for(size_t i = 0; i < name.size(); i++) {
    const char c = name[i];
    const bool ok =
      (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
      c == '.' || c == '_' || c == '-';
    if(!ok)
      throw StringError(
        who + " contains '" + std::string(1, c) +
        "' at position " + Global::uint64ToString(i) +
        "; it may hold only ASCII letters, digits, '.', '_' and '-'."
      );
  }
}

uint64_t NNCacheFileName::hashOf(const std::string& name) {
  return NNCacheFileChecksum::of((const uint8_t*)name.data(), name.size(), 0);
}

//-------------------------------------------------------------------------------------
// The durability primitives
//-------------------------------------------------------------------------------------

NNCacheFileHandle::NNCacheFileHandle(const std::string& path, const char* mode)
  :file_(std::fopen(path.c_str(), mode))
{}

NNCacheFileHandle::~NNCacheFileHandle() {
  if(file_ != nullptr)
    std::fclose(file_);
}

bool NNCacheFileHandle::flushAndSync() {
  if(file_ == nullptr)
    return false;
  if(std::fflush(file_) != 0)
    return false;
#ifdef _WIN32
  return _commit(_fileno(file_)) == 0;
#else
  return ::fsync(::fileno(file_)) == 0;
#endif
}

bool NNCacheFileHandle::closeNow() {
  if(file_ == nullptr)
    return true;
  const bool ok = std::fclose(file_) == 0;
  file_ = nullptr;
  return ok;
}

void NNCacheFileSync::directoryOf(const std::string& path) {
#ifndef _WIN32
  const size_t slash = path.find_last_of('/');
  const std::string dir = slash == std::string::npos ? std::string(".") : path.substr(0, slash);
  const int fd = ::open(dir.c_str(), O_RDONLY);
  if(fd < 0)
    return;
  (void)::fsync(fd);
  (void)::close(fd);
#else
  (void)path;
#endif
}
