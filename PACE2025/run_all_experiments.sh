#!/bin/bash

TEST_DIR="$HOME/dissertation_ashwin/PACE2025-instances/private/hs/exact"
FEATURES="$HOME/dissertation_ashwin/hit/data/vertex_features_test.csv"
TIMEOUT=1800
WORKERS=20

JOBS="all_jobs.txt"
QUEUE_LOCK="all_jobs.lock"

CONFIGS=(
    "D2:/tmp/uzl_hs_D2_K32:1"
    "D5:/tmp/uzl_hs_D5_K32:1"
    "D10:/tmp/uzl_hs_D10_K32:1"
    "D100:/tmp/uzl_hs_D100_K32:1"
    "baseline:/tmp/uzl_hs_baseline:0"
)

rm -f "$JOBS" "$QUEUE_LOCK"

for entry in "${CONFIGS[@]}"; do
    config="${entry%%:*}"
    rest="${entry#*:}"
    binary="${rest%%:*}"
    ml="${rest##*:}"

    for instance in $(cat cadical_test_instances.txt); do
        echo "$config|$binary|$ml|$instance" >> "$JOBS"
    done
done

for entry in "${CONFIGS[@]}"; do
    config="${entry%%:*}"

    echo "instance,configuration,solved,runtime,decisions,conflicts,propagations" \
        > "test_results_${config}.csv"

    rm -f "test_results_${config}.csv.lock"
done

echo "Jobs: $(wc -l < "$JOBS")"
echo "Workers: $WORKERS"
echo

worker() {

    while true; do

        job=$(
            (
                flock 9

                if [ ! -s "$JOBS" ]; then
                    exit 1
                fi

                head -1 "$JOBS"
                sed -i '1d' "$JOBS"

            ) 9>"$QUEUE_LOCK"
        )

        [ -z "$job" ] && break

        config=$(echo "$job" | cut -d'|' -f1)
        binary=$(echo "$job" | cut -d'|' -f2)
        ml=$(echo "$job" | cut -d'|' -f3)
        instance=$(echo "$job" | cut -d'|' -f4)

        file="$TEST_DIR/private_${instance}.hgr"
        results="test_results_${config}.csv"

        echo "[$config] START $instance"

        start=$(date +%s.%N)

        output=$(timeout "$TIMEOUT"s env \
            INSTANCE="$instance" \
            VERTEX_FEATURES="$FEATURES" \
            ML_BRANCH="$ml" \
            "$binary" \
            < "$file" 2>&1)

        status=$?

        end=$(date +%s.%N)

        runtime=$(awk "BEGIN {printf \"%.3f\", $end-$start}")

        stats=$(echo "$output" | grep '\[STATS\]' | tail -1)

        decisions=$(echo "$stats" |
            sed -n 's/.*decisions=\([0-9]*\).*/\1/p')

        conflicts=$(echo "$stats" |
            sed -n 's/.*conflicts=\([0-9]*\).*/\1/p')

        propagations=$(echo "$stats" |
            sed -n 's/.*propagations=\([0-9]*\).*/\1/p')

        [ -z "$decisions" ] && decisions=0
        [ -z "$conflicts" ] && conflicts=0
        [ -z "$propagations" ] && propagations=0

        if [ "$status" -eq 0 ]; then
            solved=1
        else
            solved=0
        fi

        (
            flock 9
            echo "$instance,$config,$solved,$runtime,$decisions,$conflicts,$propagations"
        ) 9>>"${results}.lock" >> "$results"

        echo "[$config] FINISHED $instance runtime=${runtime}s solved=$solved"

    done
}

pids=()

for i in $(seq 1 "$WORKERS"); do
    worker &
    pids+=("$!")
done

for pid in "${pids[@]}"; do
    wait "$pid"
done

rm -f "$JOBS" "$QUEUE_LOCK"

echo
echo "========================================"
echo "ALL 500 JOBS FINISHED"
echo "========================================"
