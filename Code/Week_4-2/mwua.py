from dataclasses import dataclass

import numpy as np

from problem import MILPProblem


@dataclass
class MWUAResult:

    x_avg: np.ndarray

    certainty: np.ndarray

    final_weights: np.ndarray

    weight_min: np.ndarray
    weight_max: np.ndarray
    weight_avg: np.ndarray


class MWUAFeatureExtractor:

    def __init__(
        self,
        rounds: int = 100,
        eps: float = 0.1,
    ):
        self.rounds = rounds
        self.eps = eps

    def greedy_fractional_solution(
        self,
        scores: np.ndarray,
    ) -> np.ndarray:
        """
        Direct translation of
        greedySolveCombined().
        """

        n = len(scores)

        x = np.zeros(n)

        order = np.argsort(-scores)

        need = 1.0
        cumulative = 0.0

        for v in order:

            if need <= 1e-12:
                break

            if scores[v] <= 1e-12:
                continue

            can_give = min(
                1.0,
                need / scores[v]
            )

            x[v] = can_give

            cumulative += (
                scores[v]
                * can_give
            )

            need = 1.0 - cumulative

        return x

    def compute(
        self,
        problem: MILPProblem,
    ) -> MWUAResult:

        m = len(problem.A_ub)
        n = problem.num_variables

        weights = np.ones(m)

        x_avg = np.zeros(n)

        weight_history = []

        for t in range(1, self.rounds + 1):

            # ---------------------------------
            # Compute D(v)
            # ---------------------------------

            normalized_weights = (
                weights
                / weights.sum()
            )

            scores = (
            np.abs(problem.A_ub).T
             @ normalized_weights
        )

            # ---------------------------------
            # Greedy fractional oracle
            # ---------------------------------

            x_t = self.greedy_fractional_solution(
                scores
            )

            # ---------------------------------
            # Running average
            # ---------------------------------

            alpha = 1.0 / t

            x_avg += (
                x_t - x_avg
            ) * alpha

            weight_history.append(
                weights.copy()
            )

            # ---------------------------------
            # Update weights
            # ---------------------------------

            for edge_idx in range(m):

                row = problem.A_ub[edge_idx]

                vars_in_edge = np.where(
                np.abs(row) > 0
                )[0]

                cover = np.sum(
                    x_t[vars_in_edge]
                )

                if problem.problem_type == "mvc":

                    violation = max(0.0,1.0 - cover)

                elif problem.problem_type == "mis":

                    violation = max(0.0,cover - 1.0)
                
                else:
                    raise ValueError("Unknown problem type")

                weights[edge_idx] *= (
                    1.0
                    + self.eps
                    * violation
                )

        weight_history = np.array(
            weight_history
        )

        certainty = np.abs(
            x_avg - 0.5
        )

        return MWUAResult(
            x_avg=x_avg,
            certainty=certainty,
            final_weights=weights,
            weight_min=weight_history.min(
                axis=0
            ),
            weight_max=weight_history.max(
                axis=0
            ),
            weight_avg=weight_history.mean(
                axis=0
            ),
        )
