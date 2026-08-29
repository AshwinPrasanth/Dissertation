from dataclasses import dataclass
import time

import numpy as np

from problem import MILPProblem


@dataclass
class MWUAResult:

    x_avg: np.ndarray

    certainty: np.ndarray

    final_weights: np.ndarray


class MWUAFeatureExtractor:

    def __init__(
        self,
        rounds: int = 50000,
        eps: float = 0.25,
        delta: float = 1e-6,
        time_limit: float = 90.0,
    ):

        self.rounds = rounds

        self.eps = eps

        self.delta = delta

        self.time_limit = time_limit

    def greedy_fractional_solution(
        self,
        scores: np.ndarray,
    ) -> np.ndarray:

        n = len(scores)

        x = np.zeros(
            n,
            dtype=np.float64,
        )

        order = np.argsort(
            -scores
        )

        need = 1.0

        cumulative = 0.0

        for v in order:

            if need <= 1e-15:
                break

            if scores[v] <= 1e-15:
                break

            can_give = min(
                1.0,
                need / scores[v],
            )

            x[v] = can_give

            cumulative += (
                scores[v]
                * can_give
            )

            need = (
                1.0
                - cumulative
            )

        return x

    def max_violation(
        self,
        x: np.ndarray,
        problem: MILPProblem,
    ) -> float:

        worst = 0.0

        for u, v in problem.edges:

            cover = (
                x[u]
                + x[v]
            )

            violation = max(
                0.0,
                1.0 - cover,
            )

            worst = max(
                worst,
                violation,
            )

        return worst

    def min_feasibility_ratio(
        self,
        x: np.ndarray,
        problem: MILPProblem,
    ) -> float:

        eps = 1e-15

        alpha = 0.0

        for u, v in problem.edges:

            cover = (
                x[u]
                + x[v]
            )

            if cover <= eps:

                x[u] += 0.5

                x[v] += 0.5

                cover = 1.0

            ratio = (
                1.0 / cover
            )

            alpha = max(
                alpha,
                ratio,
            )

        return alpha

    def compute(
        self,
        problem: MILPProblem,
    ) -> MWUAResult:

        m = len(
            problem.edges
        )

        n = (
            problem.num_variables
        )

        weights = np.ones(
            m,
            dtype=np.float64,
        )

        x_avg = np.zeros(
            n,
            dtype=np.float64,
        )

        scores = np.zeros(
            n,
            dtype=np.float64,
        )

        start_time = time.time()

        completed_rounds = 0

        for t in range(
            1,
            self.rounds + 1,
        ):

            if (
                time.time()
                - start_time
                >= self.time_limit
            ):

                print(
                    "[MWUA] Time limit reached:",
                    self.time_limit,
                    "seconds",
                )

                break

            weight_sum = (
                weights.sum()
            )

            if (
                not np.isfinite(
                    weight_sum
                )
                or weight_sum <= 0.0
            ):

                raise FloatingPointError(
                    "MWUA weights became non-finite"
                )

            normalized_weights = (
                weights
                / weight_sum
            )

            scores.fill(
                0.0
            )

            for edge_idx, (
                u,
                v,
            ) in enumerate(
                problem.edges
            ):

                w = normalized_weights[
                    edge_idx
                ]

                scores[u] += w

                scores[v] += w

            x_t = (
                self
                .greedy_fractional_solution(
                    scores
                )
            )

            alpha = (
                1.0 / t
            )

            x_avg += (
                x_t
                - x_avg
            ) * alpha

            for edge_idx, (
                u,
                v,
            ) in enumerate(
                problem.edges
            ):

                cover = (
                    x_t[u]
                    + x_t[v]
                )

                violation = max(
                    0.0,
                    1.0 - cover,
                )

                weights[
                    edge_idx
                ] *= (
                    1.0
                    + self.eps
                    * violation
                )

            max_weight = (
                weights.max()
            )

            if (
                not np.isfinite(
                    max_weight
                )
            ):

                raise FloatingPointError(
                    "MWUA weight overflow"
                )

            if max_weight > 1e100:

                weights /= (
                    max_weight
                )

            completed_rounds = t

            if t % 10 == 0:

                violation = (
                    self.max_violation(
                        x_avg,
                        problem,
                    )
                )

                if (
                    violation
                    <= self.delta
                ):

                    print(
                        "[MWUA] Converged at round",
                        t,
                    )

                    break

        print(
            "[MWUA] Completed rounds:",
            completed_rounds,
        )

        print(
            "[MWUA] Final violation:",
            self.max_violation(
                x_avg,
                problem,
            ),
        )

        scale_factor = (
            self.min_feasibility_ratio(
                x_avg,
                problem,
            )
        )

        if scale_factor < 1.0:

            print(
                "[INFO] Scaling solution by",
                scale_factor,
            )

            x_avg *= (
                scale_factor
            )

        final_weight_sum = (
            weights.sum()
        )

        if (
            not np.isfinite(
                final_weight_sum
            )
            or final_weight_sum <= 0.0
        ):

            raise FloatingPointError(
                "Invalid final MWUA weights"
            )

        final_weights = (
            weights
            / final_weight_sum
        )

        certainty = np.abs(
            x_avg
            - 0.5
        )

        return MWUAResult(
            x_avg=x_avg,
            certainty=certainty,
            final_weights=final_weights,
        )