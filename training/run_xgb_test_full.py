import json
import time
from pathlib import Path
import argparse

from solver_scip_default import solve_scip_default
from solver_xgb_full import solve_xgb_dimacs, solve_xgb_snap


GRAPH_DIR = (
    Path("graphs")
    / "test_full"
)



TIME_LIMIT = 12000
parser = argparse.ArgumentParser()

parser.add_argument(
    "--depth",
    type=int,
    default=5
)

args = parser.parse_args()


XGB_DEPTH = args.depth

OUTPUT_DIR = (
    Path("results")
    / "anytime_comparison"
    / f"depth_frb30_FULL{XGB_DEPTH}"
)

OUTPUT = (
    OUTPUT_DIR
    / "results.json"
)



def convert_result(result, method):

    result["method"] = method
    result["xgb_depth"] = XGB_DEPTH

    if "incumbent_history" in result:

        result["incumbent_history"] = [
            list(x)
            for x in result["incumbent_history"]
        ]

    return result



def main():

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    graphs = sorted(
    GRAPH_DIR.rglob("*.mtx")
)


    if not graphs:

        raise RuntimeError(
            f"No .mtx files found in {GRAPH_DIR}"
        )


    print(
        f"Found {len(graphs)} graphs"
    )


    results = []


    for idx, graph in enumerate(graphs):

        print(
            f"[{idx+1}/{len(graphs)}] completed",
            flush=True
        )

        print()
        print("=" * 80)
        print(
            "GRAPH:",
            graph
        )
        print("=" * 80)


        '''print()
        print("Running SCIP default")

        start = time.perf_counter()

        scip_result = solve_scip_default(
            graph,
            time_limit=TIME_LIMIT
        )


        scip_result = convert_result(
            scip_result,
            "SCIP"
        )


        scip_result["experiment_time"] = (
            time.perf_counter()
            -
            start
        )


        results.append(
            scip_result
        )
        
        with open(
            OUTPUT,
            "w"
        ) as f:

            json.dump(
                results,
                f,
                indent=4
            )'''


        print()
        print("Running SCIP + XGB")


        start = time.perf_counter()


        xgb_result = solve_xgb_dimacs(
            graph,
            time_limit=TIME_LIMIT, xgb_max_depth=XGB_DEPTH,
        )


        xgb_result = convert_result(
            xgb_result,
            "SCIP+XGB"
        )


        xgb_result["experiment_time"] = (
            time.perf_counter()
            -
            start
        )


        results.append(
            xgb_result
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
    print("="*80)
    print("DONE")
    print("="*80)

    print(
        OUTPUT
    )



if __name__ == "__main__":
    main()







'''from pathlib import Path

from solver_xgb import solve_xgb_dimacs


GRAPH = (
    Path("graphs")
    / "Dimacs"
    / "brock200-1"
    / "brock200-1.mtx"
)


if __name__ == "__main__":

    result = solve_xgb_dimacs(
        GRAPH,
        time_limit=300,
    )

    print()
    print("FINAL RESULT")
    print(result)
'''   
    
'''from pathlib import Path

from solver_scip_default import solve_scip_default


GRAPH = (
    Path("graphs")
    / "Dimacs"
    / "brock200-1"
    / "brock200-1.mtx"
)


if __name__ == "__main__":

    result = solve_scip_default(
        GRAPH,
        time_limit=300,
    )

    print()
    print("FINAL RESULT")
    print(result)'''