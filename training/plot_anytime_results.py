import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


RESULT_FILE = (
    Path("results")
    / "anytime_comparison"
    / "results.json"
)


OUTPUT_DIR = (
    Path("results")
    / "anytime_comparison"
    / "plots"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def load_results():

    with open(RESULT_FILE, "r") as f:
        return json.load(f)



def prepare(results):

    data = defaultdict(dict)

    for r in results:

        graph = r["graph"]
        method = r["method"]

        data[graph][method] = r

    return data



def plot_node_reduction(data):

    graphs = []
    reductions = []

    for graph, values in data.items():

        if "SCIP" not in values:
            continue

        if "SCIP+XGB" not in values:
            continue


        scip_nodes = values["SCIP"]["nodes"]
        xgb_nodes = values["SCIP+XGB"]["nodes"]


        if scip_nodes == 0:
            continue


        reduction = (
            (scip_nodes - xgb_nodes)
            /
            scip_nodes
            *
            100
        )

        graphs.append(graph)
        reductions.append(reduction)


    plt.figure(figsize=(12,5))

    plt.bar(
        graphs,
        reductions
    )

    plt.xticks(
        rotation=90
    )

    plt.ylabel(
        "Node reduction (%)"
    )

    plt.title(
        "Search tree reduction: SCIP+XGB vs SCIP"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "node_reduction.png",
        dpi=300
    )

    plt.close()



def plot_objective_difference(data):

    graphs = []
    differences = []


    for graph, values in data.items():

        if "SCIP" not in values:
            continue

        if "SCIP+XGB" not in values:
            continue


        diff = (
            values["SCIP"]["objective"]
            -
            values["SCIP+XGB"]["objective"]
        )


        graphs.append(graph)
        differences.append(diff)


    plt.figure(figsize=(12,5))


    plt.bar(
        graphs,
        differences
    )


    plt.xticks(
        rotation=90
    )

    plt.ylabel(
        "Objective difference"
    )


    plt.title(
        "Solution quality difference"
    )


    plt.tight_layout()


    plt.savefig(
        OUTPUT_DIR / "objective_difference.png",
        dpi=300
    )


    plt.close()



def plot_anytime_curve(data, selected):


    for graph in selected:


        if graph not in data:
            continue


        plt.figure(figsize=(7,5))


        for method in [
            "SCIP",
            "SCIP+XGB"
        ]:

            if method not in data[graph]:
                continue


            history = (
                data[graph][method]
                ["incumbent_history"]
            )


            history = sorted(
                history,
                key=lambda x:x[0]
            )


            times = [
                x[0]
                for x in history
            ]


            objs = [
                x[1]
                for x in history
            ]


            plt.step(
                times,
                objs,
                where="post",
                label=method
            )


        plt.xlabel(
            "Time (seconds)"
        )

        plt.ylabel(
            "Incumbent objective"
        )


        plt.title(
            f"Anytime curve: {graph}"
        )


        plt.legend()

        plt.grid()

        plt.tight_layout()


        plt.savefig(
            OUTPUT_DIR / f"{graph}_anytime.png",
            dpi=300
        )

        plt.close()



def main():

    results = load_results()

    data = prepare(results)


    plot_node_reduction(
        data
    )


    plot_objective_difference(
        data
    )


    selected = [
        "frb35-17-3",
        "frb40-19-1",
        "frb50-23-1",
    ]


    plot_anytime_curve(
        data,
        selected
    )


    print(
        "Saved plots to:",
        OUTPUT_DIR
    )



if __name__ == "__main__":
    main()