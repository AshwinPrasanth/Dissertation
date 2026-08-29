from pyscipopt import Model, quicksum


class SCIPHittingSetSolver:

    def solve(
        self,
        problem,
    ):

        model = Model(
            "HittingSet"
        )

        x = []

        for i in range(
            problem.num_variables
        ):

            x.append(

                model.addVar(

                    name=f"x_{i}",

                    vtype="B",

                )

            )

        model.setObjective(

            quicksum(

                x[i]

                for i in range(
                    problem.num_variables
                )

            ),

            "minimize",

        )

        for hyperedge in problem.hyperedges:

            model.addCons(

                quicksum(

                    x[v]

                    for v in hyperedge

                ) >= 1

            )

        model.optimize()

        solution = model.getBestSol()

        chosen = []

        if solution is not None:

            for i in range(
                problem.num_variables
            ):

                if model.getSolVal(
                    solution,
                    x[i],
                ) > 0.5:

                    chosen.append(i)

        return chosen