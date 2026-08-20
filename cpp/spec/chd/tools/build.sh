#!/bin/sh
# Build the conformance-vector generator.
#
# This program includes a header from /home/bork/kata_fork_archived, the archived prototype.
# It is the instrument that produced cpp/spec/chd/vectors/ and it is NOT needed to check an
# implementation against those vectors. Anyone working under the clean-room constraint
# should never build or run this file.
#
# Flags are KataGo's own Release flags for gcc on Linux, from cpp/CMakeLists.txt:
# CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG", "-mavx2 -mfma", "-fsigned-char".
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
CPP=$(cd "$HERE/../../.." && pwd)
OUT=${1:-$HERE/gen_vectors}

g++ -std=c++17 -O3 -DNDEBUG -mavx2 -mfma -fsigned-char \
    -I "$CPP" \
    -o "$OUT" \
    "$HERE/gen_vectors.cpp" \
    "$CPP/core/hash.cpp" \
    "$CPP/core/global.cpp" \
    "$CPP/core/sha2.cpp"

echo "built: $OUT"
