#!/bin/bash

BINARY="$1"
CONFIG="$2"
CPUS="$3"

TEST_DIR="$HOME/dissertation_ashwin/PACE2025-instances/private/hs/exact"
FEATURES="$HOME/dissertation_ashwin/hit/data/vertex_features_test.csv"
RESULTS="test_results_${CONFIG}.csv"
TIMEOUT=1800

echo "instance,configuration,solved,runtime,decisions,conflicts,propagations" > "$RESULTS"

run_instance() {
    local instance="$1"
    local cpu="$2"

    local file="$TEST_DIR/private_${instance}.hgr"
    local start
    local end
    local output
    local status
    local runtime
    local stats
    local decisions
    local conflicts
    local propagations
    local ml

    echo "[$CONFIG] START $instance -> CPU $cpu"

    start=$(date +%s.%N)

    if [ "$CONFIG" = "baseline" ]; then
        ml=0
    else
        ml=1
    fi

    output=$(taskset -c "$cpu" timeout "$TIMEOUT"s env \
        INSTANCE="$instance" \
        VERTEX_FEATURES="$FEATURES" \
        ML_BRANCH="$ml" \
        "$BINARY" \
        < "$file" 2>&1)

    status=$?

    end=$(date +%s.%N)

    runtime=$(awk "BEGIN {printf \"%.3f\", $end - $start}")

    stats=$(echo "$output" | grep '\[STATS\]' | tail -1)

    decisions=$(echo "$stats" |
        sed -n 's/.*decisions=\([0-9]*\).*/\1/p')

    conflicts=$(echo "$stats" |
        sed -n 's/.*conflicts=\([0-9]*\).*/\1/p')

    propagations=$(echo "$stats" |
        sed -n 's/.*propagations=\([0-9]*\).*/\1/p')

    if [ "$status" -eq 0 ]; then
        solved=1
    else
        solved=0
    fi

    [ -z "$decisions" ] && decisions=0
    [ -z "$conflicts" ] && conflicts=0
    [ -z "$propagations" ] && propagations=0

    (
        flock 9
        echo "$instance,$CONFIG,$solved,$runtime,$decisions,$conflicts,$propagations"
    ) 9>>"${RESULTS}.lock" >> "$RESULTS"

    echo "[$CONFIG] FINISHED $instance runtime=${runtime}s solved=$solved"

    return 0
}

export -f run_instance
export BINARY CONFIG TEST_DIR FEATURES RESULTS TIMEOUT

mapfile -t CPULIST < <(
    echo "$CPUS" |
    tr ',' '\n' |
    tr '-' '\n' |
    awk '
    {
        if (NR % 2 == 1)
            start=$1;
        else
            for (i=start; i<=$1; i++)
                print i;
    }'
)

mapfile -t INSTANCES < cadical_test_instances.txt

total=${#INSTANCES[@]}

next_instance=0
active=0

declare -A PID_CPU
declare -A PID_INSTANCE

while [ "$next_instance" -lt "$total" ] || [ "$active" -gt 0 ]; do

    for cpu in "${CPULIST[@]}"; do

        if [ "$next_instance" -ge "$total" ]; then
            break
        fi

        if [ "$active" -ge "${#CPULIST[@]}" ]; then
            break
        fi

        instance="${INSTANCES[$next_instance]}"

        run_instance "$instance" "$cpu" &

        pid=$!

        PID_CPU[$pid]="$cpu"
        PID_INSTANCE[$pid]="$instance"

        next_instance=$((next_instance + 1))
        active=$((active + 1))

    done

    while true; do

        finished_pid=""

        for pid in "${!PID_CPU[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                finished_pid="$pid"
                break
            fi
        done

        if [ -n "$finished_pid" ]; then

            unset 'PID_CPU[$finished_pid]'
            unset 'PID_INSTANCE[$finished_pid]'

            active=$((active - 1))

            break
        fi

        sleep 0.2
    done

done

echo "[$CONFIG] ALL ${total} INSTANCES FINISHED"
