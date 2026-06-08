from dataclasses import dataclass

import numpy as np

from problem import MILPProblem


@dataclass
class MWUAResult:

    x_avg: np.ndarray
    certainty: np.ndarray
    weights: np.ndarray


class MWUAFeatureExtractor:

    def __init__(
        self,
        rounds: int = 50,
        eta: float = 0.1,
    ):
        self.rounds = rounds
        self.eta = eta

    def compute(
        self,
        problem: MILPProblem,
    ) -> MWUAResult:

        m = len(problem.A_ub)
        n = problem.num_variables

        weights = np.ones(m)

        x_sum = np.zeros(n)

        for _ in range(self.rounds):

            # variable pressure
            pressure = (-problem.A_ub).T @ weights

            if pressure.max() > 0:
                x = pressure / pressure.max()
            else:
                x = np.zeros(n)

            x = np.clip(x, 0.0, 1.0)

            x_sum += x

            # constraint violations

            lhs = problem.A_ub @ x

            violations = np.maximum(
                lhs - problem.b_ub,
                0.0,
            )

            weights *= np.exp(
                self.eta * violations
            )

        x_avg = x_sum / self.rounds

        certainty = np.abs(
            x_avg - 0.5
        )

        return MWUAResult(
            x_avg=x_avg,
            certainty=certainty,
            weights=weights,
        )