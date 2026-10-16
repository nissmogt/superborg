#!/bin/sh
# Build the borg engine as a WebAssembly module (wasm32-wasi reactor, SIMD128, one thread).
#
#   wasm/build.sh [out.wasm]       default: build/borg.wasm
#
# Needs clang with the wasm32-wasi target and a WASI sysroot (e.g. the wasi-libc or
# wasi-sdk packages). The module imports a handful of WASI functions; wasm/borg-runtime.js
# provides them, so it runs in browsers and Node without a WASI runtime.
set -eu
OUT=${1:-}
case $OUT in ''|/*) ;; *) OUT="$PWD/$OUT" ;; esac  # a relative path is the caller's
cd "$(dirname "$0")/.."
OUT=${OUT:-build/borg.wasm}
CC=${CC:-clang}
mkdir -p "$(dirname "$OUT")"
SYSROOT_FLAG=""
if [ -n "${WASI_SYSROOT:-}" ]; then SYSROOT_FLAG="--sysroot=$WASI_SYSROOT"; fi
$CC --target=wasm32-wasi $SYSROOT_FLAG -O3 -msimd128 -mexec-model=reactor -std=c11 -fvisibility=hidden \
    -D_GNU_SOURCE -D_DEFAULT_SOURCE -DBORG_NO_THREADS -DBORG_NO_MMAP -D_WASI_EMULATED_MMAN -Iengine \
    engine/json.c engine/model.c engine/kernels.c engine/pool.c engine/tokenizer.c engine/forward.c \
    engine/sample.c engine/generate.c engine/json_mode.c \
    -o "$OUT" -lm -lwasi-emulated-mman \
    -Wl,--export-dynamic -Wl,--export=malloc -Wl,--export=free -Wl,--max-memory=4294967296
echo "wrote $OUT ($(wc -c < "$OUT") bytes)"
