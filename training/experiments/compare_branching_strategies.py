import os
import sys
import csv


current_dir = os.path.dirname(
    os.path.abspath(__file__)
)

anytime_dir = os.path.dirname(
    current_dir
)

if anytime_dir not in sys.path:
    sys.path.insert(
        0,
        anytime_dir,
    )


from solver_runner import solve_instance


GRAPH = os.path.join(
    anytime_dir,
    "graphs",
    "ca_family",
    "web-Google.txt",
)


RESULT_DIR = os.path.join(
    anytime_dir,
    "results",
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True,
)


OUTPUT_CSV = os.path.join(
    RESULT_DIR,
    "spine_comparison.csv",
)


strategies = [
    (
        "Degree K1",
        "residual_degree",
        False,
        1,
    ),
    (
        "Hybrid K1",
        "residual_degree",
        True,
        1,
    ),
]


results = []


for (
    strategy_name,
    selector,
    use_mwua_direction,
    max_spine_length,
) in strategies:

    print()

    print(
        "==================================="
    )

    print(
        strategy_name
    )

    print(
        "==================================="
    )

    result = solve_instance(
        graph_path=GRAPH,
        use_mwua=True,
        selector=selector,
        use_mwua_direction=use_mwua_direction,
        max_spine_length=max_spine_length,
        certainty_threshold=0.0,
    )

    result["strategy"] = (
        strategy_name
    )

    results.append(
        result
    )
    
baseline_nodes = None


for result in results:

    if result["strategy"] == "SCIP":

        baseline_nodes = result.get(
            "nodes"
        )

        break


if baseline_nodes is not None:

    for result in results:

        nodes = result.get(
            "nodes"
        )

        if nodes is None:

            result[
                "search_reduction"
            ] = None

        else:

            result[
                "search_reduction"
            ] = (
                (
                    baseline_nodes
                    - nodes
                )
                / baseline_nodes
            ) * 100.0

else:

    for result in results:

        result[
            "search_reduction"
        ] = None


fieldnames = [
    "strategy",
    "runtime",
    "nodes",
    "objective",
    "selector",
    "use_mwua_direction",
    "max_spine_length",
    "spine_length",
    "length_stops",
    "uncertainty_stops",
]


with open(
    OUTPUT_CSV,
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    for result in results:

        row = {}

        for field in fieldnames:

            row[field] = result.get(
                field
            )

        writer.writerow(
            row
        )


print()

print(
    "==================================="
)

print(
    "SUMMARY"
)

print(
    "==================================="
)


for result in results:

    print()

    print(
        result["strategy"]
    )

    print(
        "Runtime:",
        result.get("runtime"),
    )

    print(
        "Nodes:",
        result.get("nodes"),
    )

    print(
        "Objective:",
        result.get("objective"),
    )

    print(
        "Max spine length:",
        result.get(
            "max_spine_length"
        ),
    )

    print(
        "Actual spine length:",
        result.get(
            "spine_length"
        ),
    )

    print(
        "Search reduction:",
        result.get(
            "search_reduction"
        ),
    )


print()

print(
    "Saved to:",
    OUTPUT_CSV,
)