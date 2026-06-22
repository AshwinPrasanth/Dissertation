from dataclasses import dataclass

from pyscipopt import (
    Model,
    quicksum,
)

from problem import build_mis_problem


@dataclass
class SCIPBenchmarkResult:

    objective: float

    nodes: int

    runtime: float


def run_scip_benchmark(
    G,
):

    problem = build_mis_problem(
        G
    )

    model = Model(
        "MIS"
    )

    x = []

    for i in range(
        problem.num_variables
    ):

        x.append(
            model.addVar(
                vtype="B",
                name=f"x_{i}",
            )
        )

    model.setObjective(
        quicksum(
            x[i]
            for i in range(
                problem.num_variables
            )
        ),
        "maximize",
    )

    for row in problem.A_ub:

        vars_in_row = [
            idx
            for idx, coeff in enumerate(row)
            if coeff != 0
        ]

        model.addCons(
            quicksum(
                x[idx]
                for idx in vars_in_row
            )
            <= 1
        )

    model.optimize()

    return SCIPBenchmarkResult(

        objective=model.getObjVal(),

        nodes=model.getNNodes(),

        runtime=model.getSolvingTime(),
    )