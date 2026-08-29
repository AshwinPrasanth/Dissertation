import json
import time
from pathlib import Path

from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)

from solver_scip_default import (
    solve_scip_default
)


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


OUTPUT_DIR = (
    PROJECT_ROOT
    / "hit"
    / "results"
    / "anytime_comparison"
    / "scip_hs_default"
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


    result = solve_scip_default(
        instance,
        time_limit=TIME_LIMIT
    )


    result["method"] = "SCIP"


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
            f"No .hgr files found in "
            f"{INSTANCE_DIR}"
        )


    print(
        f"Found {len(instances)} "
        f"Hitting Set instances"
    )


    print(
        f"Running {N_WORKERS} "
        f"instances in parallel"
    )


    print(
        f"Time limit per instance: "
        f"{TIME_LIMIT}s"
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
                    f"{completed}/"
                    f"{len(instances)}"
                )


                print(
                    "INSTANCE:",
                    instance
                )


                print(
                    "Status:",
                    result.get(
                        "status"
                    )
                )


                print(
                    "Objective:",
                    result.get(
                        "objective"
                    )
                )


                print(
                    "Nodes:",
                    result.get(
                        "nodes"
                    )
                )


                print(
                    "Solve time:",
                    result.get(
                        "solve_time"
                    )
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
    print("=" * 80)
    print("DONE")
    print("=" * 80)


    print(
        "Results:",
        OUTPUT
    )


if __name__ == "__main__":

    main()