from typing import Dict, Optional

from scipy.optimize import linprog

from problem import MILPProblem


def solve_lp_relaxation(
    problem: MILPProblem,
    fixings: Optional[Dict[int, int]] = None,
):
    """
    Solve the LP relaxation of a binary optimization problem.

        min c^T x

        subject to

            A_ub x <= b_ub
            A_eq x = b_eq
            0 <= x <= 1

    Optionally, variables can be fixed to 0 or 1 using the
    'fixings' dictionary.
    """

    if fixings is None:
        fixings = {}

    if problem.A_ub is None or problem.b_ub is None:
        raise ValueError(
            "MILPProblem must define A_ub and b_ub."
        )

    bounds = []

    for i in range(problem.num_variables):

        if i in fixings:

            value = float(
                fixings[i]
            )

            bounds.append(
                (value, value)
            )

        else:

            bounds.append(
                (0.0, 1.0)
            )

    result = linprog(

        c=problem.c,

        A_ub=problem.A_ub,

        b_ub=problem.b_ub,

        A_eq=(
            problem.A_eq
            if problem.A_eq is not None
            else None
        ),

        b_eq=(
            problem.b_eq
            if problem.b_eq is not None
            else None
        ),

        bounds=bounds,

        method="highs",

    )

    if not result.success:

        raise RuntimeError(
            result.message
        )

    return result