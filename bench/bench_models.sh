#!/bin/sh
# Prefill and decode throughput of the CLI over models and thread counts, best of REPS runs
# (the benchmark machine is shared with other work, so the minimum time is reported).
#
#   bench/bench_models.sh MODELS_DIR "model1 model2 ..." "1 2 4" [PROMPT] [GEN] [REPS] [BORG]
#
# Prints one line per model and thread count: prefill and decode tok/s of the best run and the
# spread over the runs. BORG_NO_REPACK=1 in the environment measures the plain-layout kernels.
set -e
DIR=$1; MODELS=$2; THREADS=${3:-"1 2 4"}; P=${4:-128}; G=${5:-32}; R=${6:-3}
BORG=${7:-$(dirname "$0")/../build/borg}
echo "# $(date -u +%Y-%m-%dT%H:%MZ) prompt=$P gen=$G reps=$R (min reported) load=$(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null) BORG_NO_REPACK=${BORG_NO_REPACK:-}"
for m in $MODELS; do
    for t in $THREADS; do
        i=1; best_p=0; best_d=0; worst_p=0; worst_d=0
        while [ $i -le $R ]; do
            out=$("$BORG" -m "$DIR/$m.borg" --bench "$P" "$G" -t "$t")
            p=$(echo "$out" | awk '/^prefill/ {print $(NF-1)}')
            d=$(echo "$out" | awk '/^decode/ {print $(NF-1)}')
            best_p=$(echo "$p $best_p" | awk '{print ($1 > $2) ? $1 : $2}')
            best_d=$(echo "$d $best_d" | awk '{print ($1 > $2) ? $1 : $2}')
            worst_p=$(echo "$p $worst_p" | awk '{print ($2 == 0 || $1 < $2) ? $1 : $2}')
            worst_d=$(echo "$d $worst_d" | awk '{print ($2 == 0 || $1 < $2) ? $1 : $2}')
            i=$((i + 1))
        done
        echo "$m threads=$t prefill=$best_p decode=$best_d (worst run: prefill=$worst_p decode=$worst_d)"
    done
done
