import time
from pathlib import Path

from pyscipopt import (
    Model,
    Eventhdlr,
    SCIP_EVENTTYPE,
)


class IncumbentEventHandler(Eventhdlr):

    def __init__(self):

        self.history = []


    def eventinit(self):

        self.model.catchEvent(
            SCIP_EVENTTYPE.BESTSOLFOUND,
            self
        )


    def eventexit(self):

        self.model.dropEvent(
            SCIP_EVENTTYPE.BESTSOLFOUND,
            self
        )


    def eventexec(self, event):

        sol = self.model.getBestSol()

        if sol is not None:

            obj = self.model.getSolObjVal(
                sol
            )

            self.history.append(
                (
                    self.model.getSolvingTime(),
                    obj
                )
            )


def load_hitting_set(path):

    hyperedges = []

    num_variables = None
    num_edges = None

    with open(
        path,
        "r"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            if line.startswith("c"):
                continue

            parts = line.split()

            if parts[0] == "p":

                num_variables = int(
                    parts[2]
                )

                num_edges = int(
                    parts[3]
                )

                continue

            edge = [
                int(v) - 1
                for v in parts
            ]

            if edge:

                hyperedges.append(
                    edge
                )


    if num_variables is None:

        if not hyperedges:

            raise ValueError(
                f"Could not parse instance: {path}"
            )

        num_variables = max(
            max(edge)
            for edge in hyperedges
        ) + 1


    if num_edges is None:

        num_edges = len(
            hyperedges
        )


    return (
        num_variables,
        hyperedges
    )


def build_hitting_set_model(
    num_variables,
    hyperedges
):

    model = Model(
        "Hitting Set"
    )

    x = {}

    for v in range(
        num_variables
    ):

        x[v] = model.addVar(
            name=f"x_{v}",
            vtype="B"
        )


    for i, edge in enumerate(
        hyperedges
    ):

        model.addCons(
            sum(
                x[v]
                for v in edge
            ) >= 1,
            name=f"edge_{i}"
        )


    model.setObjective(
        sum(
            x[v]
            for v in range(
                num_variables
            )
        ),
        "minimize"
    )


    return model


def solve_scip_default(
    instance_path,
    time_limit=1800
):

    start = time.perf_counter()

    print()
    print("=" * 70)
    print("SCIP HITTING SET SOLVER")
    print("=" * 70)


    num_variables, hyperedges = (
        load_hitting_set(
            instance_path
        )
    )


    print(
        "Variables:",
        num_variables
    )

    print(
        "Hyperedges:",
        len(hyperedges)
    )


    model = build_hitting_set_model(
        num_variables,
        hyperedges
    )


    inc_handler = (
        IncumbentEventHandler()
    )


    model.includeEventhdlr(
        inc_handler,
        "IncumbentLogger",
        "records incumbent solutions"
    )


    model.setIntParam(
        "display/verblevel",
        4
    )


    model.setBoolParam(
        "display/lpinfo",
        False
    )


    model.setRealParam(
        "limits/time",
        time_limit
    )


    solve_start = time.perf_counter()

    model.optimize()


    solve_time = (
        time.perf_counter()
        -
        solve_start
    )


    status = str(
        model.getStatus()
    )


    objective = None

    if model.getNSols():

        objective = model.getObjVal()


    dual_bound = model.getDualbound()

    gap = None

    if objective is not None:

        if objective != 0:

            gap = (
                abs(
                    objective - dual_bound
                )
                /
                abs(objective)
                *
                100
            )

        else:

            gap = 0.0


    print()
    print("INCUMBENT HISTORY")

    for t, obj in inc_handler.history:

        print(
            f"time={t:.3f}, objective={obj}"
        )


    result = {

        "status":
            status,

        "variables":
            num_variables,

        "hyperedges":
            len(hyperedges),

        "objective":
            objective,

        "dual_bound":
            dual_bound,

        "gap":
            gap,

        "nodes":
            model.getNNodes(),

        "solve_time":
            solve_time,

        "total_time":
            time.perf_counter()
            -
            start,

        "incumbent_history":
            inc_handler.history
    }


    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    print(
        "Status:",
        result["status"]
    )

    print(
        "Objective:",
        result["objective"]
    )

    print(
        "Dual bound:",
        result["dual_bound"]
    )

    print(
        "Gap:",
        result["gap"]
    )

    print(
        "Nodes:",
        result["nodes"]
    )

    print(
        "Solve time:",
        result["solve_time"]
    )

    print(
        "Total time:",
        result["total_time"]
    )


    return result