#!/bin/sh
# Side-by-side throughput of borg and llama.cpp on the same model and quantization.
#
#   bench/compare_llamacpp.sh MODEL.borg MODEL.gguf LLAMA_BENCH [THREADS] [PROMPT] [GEN] [REPS]
#
# Runs each engine REPS times (alternating) and prints every measurement, so background
# load shows up as spread instead of silently skewing one side.
#
# BORG=path overrides the borg binary (default build/borg). The llama.cpp lines name the build,
# the CPU features its ggml reports (AMX / AVX-512 VNNI / llamafile) and its system_info line, so
# the comparison says which llama.cpp kernel path it measured.
set -e
BORG_MODEL=$1; GGUF=$2; LLAMA_BENCH=$3
T=${4:-4}; P=${5:-128}; G=${6:-64}; R=${7:-3}
BORG=${BORG:-$(dirname "$0")/../build/borg}
LIBDIR=$(dirname "$LLAMA_BENCH")
python3 - "$LIBDIR" <<'PY' || true
import ctypes, glob, os, sys
d = sys.argv[1]
try:
    for lib in ("libggml-base.so", "libggml-cpu.so"):
        p = os.path.join(d, lib)
        if os.path.exists(p):
            l = ctypes.CDLL(p, ctypes.RTLD_GLOBAL)
    flags = {f: getattr(l, "ggml_cpu_has_" + f)() for f in ("amx_int8", "avx512_vnni", "avx_vnni", "avx512", "avx2", "llamafile")}
    print("llama.cpp ggml-cpu features: " + " ".join(f"{k}={v}" for k, v in flags.items()))
except Exception as e:  # static build or different library names: the flags are then unknown
    print("llama.cpp ggml-cpu features: unknown (%s)" % e)
try:  # the system_info line llama-cli prints at startup (llama-bench prints none)
    for lib in ("libggml.so", "libllama.so"):
        l = ctypes.CDLL(os.path.join(d, lib), ctypes.RTLD_GLOBAL)
    l.llama_backend_init()
    l.llama_print_system_info.restype = ctypes.c_char_p
    print("llama.cpp system_info: " + l.llama_print_system_info().decode())
except Exception as e:
    print("llama.cpp system_info: unknown (%s)" % e)
PY
"$LLAMA_BENCH" -m "$GGUF" -p 8 -n 0 -t "$T" -r 1 -v 2>&1 | grep -E "build:|cannot be used with preferred buffer type|buffer size" | sed 's/^/llama.cpp /' | head -5
i=1
while [ $i -le $R ]; do
    "$BORG" -m "$BORG_MODEL" --bench "$P" "$G" -t "$T" | awk -v r=$i '/prefill|decode/ {print "borg      run " r ": " $0}'
    "$LLAMA_BENCH" -m "$GGUF" -p "$P" -n "$G" -t "$T" -r 1 -o csv 2>/dev/null | awk -F, -v r=$i '
        NR == 1 { for (k = 1; k <= NF; k++) { gsub(/"/, "", $k); col[$k] = k } next }
        { gsub(/"/, ""); kind = ($col["n_prompt"] > 0) ? "prefill" : "decode ";
          n = ($col["n_prompt"] > 0) ? $col["n_prompt"] : $col["n_gen"];
          printf "llama.cpp run %d: %s %d tokens = %.1f tok/s\n", r, kind, n, $col["avg_ts"] }'
    i=$((i + 1))
done
