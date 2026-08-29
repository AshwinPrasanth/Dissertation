import json
import time
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

from solver_xgb import solve_xgb_hitting_set


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

INSTANCE_DIR = (
    PROJECT_ROOT
    / "PACE2025-test-reduced"
    / "exact"
)

TIME_LIMIT = 1800

N_WORKERS = 6


parser = argparse.ArgumentParser()

parser.add_argument(
    "--depth",
    type=int,
    default=5
)

args = parser.parse_args()

XGB_DEPTH = args.depth


OUTPUT_DIR = (
    PROJECT_ROOT
    / "hit"
    / "results"
    / "anytime_comparison"
    / f"xgb_hs_depth_{XGB_DEPTH}"
)

OUTPUT = (
    OUTPUT_DIR
    / "results.json"
)


def run_instance(instance):

    print(
        f"START: {instance}",
        flush=True
    )

    start = time.perf_counter()

    result = solve_xgb_hitting_set(
        instance,
        time_limit=TIME_LIMIT,
        xgb_max_depth=XGB_DEPTH
    )

    result["method"] = "SCIP+XGB"

    result["xgb_depth"] = XGB_DEPTH

    result["instance"] = str(
        instance
    )

    result["experiment_time"] = (
        time.perf_counter()
        -
        start
    )

    if "incumbent_history" in result:

        result["incumbent_history"] = [
            list(x)
            for x in result[
                "incumbent_history"
            ]
        ]

    print(
        f"FINISHED: {instance}",
        flush=True
    )

    return result


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    instances = sorted(
        INSTANCE_DIR.rglob("*.hgr")
    )

    if not instances:

        raise RuntimeError(
            f"No .hgr files found in {INSTANCE_DIR}"
        )

    print(
        f"Found {len(instances)} Hitting Set instances"
    )

    print(
        f"Running {N_WORKERS} instances in parallel"
    )

    print(
        f"Time limit per instance: {TIME_LIMIT}s"
    )

    print(
        f"XGB depth: {XGB_DEPTH}"
    )

    results = []

    with ProcessPoolExecutor(
        max_workers=N_WORKERS
    ) as executor:

        futures = {
            executor.submit(
                run_instance,
                instance
            ): instance
            for instance in instances
        }

        completed = 0

        for future in as_completed(
            futures
        ):

            instance = futures[
                future
            ]

            try:

                result = future.result()

                results.append(
                    result
                )

                completed += 1

                print()
                print(
                    "=" * 80
                )

                print(
                    f"COMPLETED "
                    f"{completed}/{len(instances)}"
                )

                print(
                    "INSTANCE:",
                    instance
                )

                print(
                    "Status:",
                    result.get("status")
                )

                print(
                    "Objective:",
                    result.get("objective")
                )

                print(
                    "Nodes:",
                    result.get("nodes")
                )

                print(
                    "Branches:",
                    result.get("branches")
                )

                print(
                    "Solve time:",
                    result.get("solve_time")
                )

                print(
                    "Total time:",
                    result.get("total_time")
                )

                print(
                    "Experiment time:",
                    result.get(
                        "experiment_time"
                    )
                )

                print(
                    "=" * 80
                )

                with open(
                    OUTPUT,
                    "w"
                ) as f:

                    json.dump(
                        results,
                        f,
                        indent=4
                    )

            except Exception as e:

                print()
                print(
                    f"FAILED: {instance}"
                )

                print(
                    repr(e)
                )

    results.sort(
        key=lambda x: x["instance"]
    )

    with open(
        OUTPUT,
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    print()
    print(
        "=" * 80
    )

    print(
        "DONE"
    )

    print(
        "=" * 80
    )

    print(
        "Results:",
        OUTPUT
    )


if __name__ == "__main__":

    main()