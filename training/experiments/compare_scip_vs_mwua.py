import csv

from solver_runner import solve_instance


GRAPH = "graphs/bhoslib/frb30-15-1.clq"


results = []

print("Running Default SCIP...")

results.append(

    solve_instance(

        GRAPH,

        use_mwua=False,

    )

)

print("Running MWUA...")

results.append(

    solve_instance(

        GRAPH,

        use_mwua=True,

        depth_limit=-1,

    )

)


baseline_nodes = results[0]["nodes"]

with open(
    "results/scip_vs_mwua.csv",
    "w",
    newline="",
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "strategy",
        "runtime",
        "nodes",
        "search_reduction",
        "objective",
        "solutions",
    ])

    for r in results:

        reduction = (
            (baseline_nodes - r["nodes"])
            / baseline_nodes
        ) * 100

        writer.writerow([
            r["strategy"],
            r["runtime"],
            r["nodes"],
            reduction,
            r["objective"],
            r["solutions"],
        ])

        print(r)