#!/bin/sh
# Side-by-side throughput of borg and llama.cpp on the same model and quantization.
#
#   bench/compare_llamacpp.sh MODEL.borg MODEL.gguf LLAMA_BENCH [THREADS] [PROMPT] [GEN] [REPS]
#
# Runs each engine REPS times (alternating) and prints every measurement, so background
# load shows up as spread instead of silently skewing one side.
set -e
BORG_MODEL=$1; GGUF=$2; LLAMA_BENCH=$3
T=${4:-4}; P=${5:-128}; G=${6:-64}; R=${7:-3}
BORG=$(dirname "$0")/../build/borg
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
