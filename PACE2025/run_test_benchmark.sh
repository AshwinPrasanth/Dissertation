#!/bin/bash

TEST_DIR="$HOME/dissertation_ashwin/PACE2025-instances/private/hs/exact"
FEATURES="$HOME/dissertation_ashwin/hit/data/vertex_features_test.csv"
RESULTS="test_results.csv"
TIMEOUT=30

echo "instance,configuration,solved,runtime,decisions,conflicts,propagations" > "$RESULTS"

run_config() {
    local config="$1"
    local ml="$2"

    echo "========================================"
    echo "CONFIG: $config"
    echo "========================================"

    while IFS= read -r instance; do

        file="$TEST_DIR/private_${instance}.hgr"

        echo "[$config] $instance"

        start=$(date +%s.%N)

        output=$(timeout "$TIMEOUT"s env \
            INSTANCE="$instance" \
            VERTEX_FEATURES="$FEATURES" \
            ML_BRANCH="$ml" \
            cargo run --release --bin uzl_hs \
            < "$file" 2>&1)

        status=$?

        end=$(date +%s.%N)

        runtime=$(awk "BEGIN {printf \"%.3f\", $end - $start}")

        decisions=$(echo "$output" |
            grep '\[STATS\]' |
            tail -1 |
            sed -n 's/.*decisions=\([0-9]*\).*/\1/p')

        conflicts=$(echo "$output" |
            grep '\[STATS\]' |
            tail -1 |
            sed -n 's/.*conflicts=\([0-9]*\).*/\1/p')

        propagations=$(echo "$output" |
            grep '\[STATS\]' |
            tail -1 |
            sed -n 's/.*propagations=\([0-9]*\).*/\1/p')

        if [ "$status" -eq 0 ]; then
            solved=1
        else
            solved=0
        fi

        [ -z "$decisions" ] && decisions=0
        [ -z "$conflicts" ] && conflicts=0
        [ -z "$propagations" ] && propagations=0

        echo "$instance,$config,$solved,$runtime,$decisions,$conflicts,$propagations" \
            >> "$RESULTS"

    done < cadical_test_instances.txt
}

run_config "baseline" 0
run_config "ML_D2_K32" 1
run_config "ML_D5_K32" 1
run_config "ML_D10_K32" 1
run_config "ML_D100_K32" 1

echo
echo "========================================"
echo "DONE"
echo "========================================"
echo "Results: $RESULTS"
