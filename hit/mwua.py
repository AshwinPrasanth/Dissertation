from dataclasses import dataclass
import time

import numpy as np

from problem import MILPProblem


@dataclass
class MWUAResult:

    x_avg: np.ndarray

    certainty: np.ndarray

    final_weights: np.ndarray

    history_iterations: list

    history_weights: list

    history_x_avg: list

    convergence_iteration: int | None

    completed_rounds: int

    converged: bool

    final_violation: float

    runtime: float


class MWUAFeatureExtractor:
    
    def hyperedge_cover(
        self,
        x: np.ndarray,
        hyperedge,
    ) -> float:
    
        return x[hyperedge].sum()

    def __init__(
        self,
        rounds: int = 50000,
        eps: float = 0.25,
        delta: float = 1e-6,
        time_limit: float = 90.0,
        verbose: bool = True,
    ):

        self.rounds = rounds

        self.eps = eps

        self.delta = delta

        self.time_limit = time_limit

        self.verbose = verbose
        

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

        for hyperedge in problem.hyperedges:

            cover = self.hyperedge_cover(
                x,
                hyperedge,
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

        for hyperedge in problem.hyperedges:

            cover = self.hyperedge_cover(
                x,
                hyperedge,
            )

            if cover <= eps:

                increment = 1.0 / len(hyperedge)

                for v in hyperedge:
                    x[v] += increment

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
            problem.hyperedges
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
        convergence_iteration = None
        
        history_iterations = []

        history_weights = []
        
        history_x_avg = []

        for t in range(
            1,
            self.rounds + 1,
        ):

            if (
                time.time()
                - start_time
                >= self.time_limit
            ):

                if self.verbose:
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

            for edge_idx, hyperedge in enumerate(
                problem.hyperedges
            ):

                w = normalized_weights[
                    edge_idx
                ]

                for v in hyperedge:
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

            for edge_idx, hyperedge in enumerate(
                problem.hyperedges
            ):

                cover = self.hyperedge_cover(
                    x_t,
                    hyperedge,
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

                history_iterations.append(t)

                history_weights.append(
                    normalized_weights.copy()
                )

                history_x_avg.append(
                    x_avg.copy()
                )

                violation = self.max_violation(
                    x_avg,
                    problem,
                )

                # Progress every 10k iterations
                if t % 10000 == 0:
                    if self.verbose:
                        print(
                        f"[MWUA] Iteration {t:,} | Violation = {violation:.8e}"
                    )

                if violation <= self.delta:
                    
                    if self.verbose:

                        print(
                        "[MWUA] Converged at round",
                        t,
                    )

                    convergence_iteration = t

                    break

        if self.verbose:
            print(
            "[MWUA] Completed rounds:",
            completed_rounds,
        )

        final_violation = self.max_violation(
            x_avg,
            problem,
        )

        runtime = (
            time.time()
            - start_time
        )
        if self.verbose:
            print(
            "[MWUA] Final violation:",
            final_violation,
        )
                
        if convergence_iteration is None:
            if self.verbose:
                print("[MWUA] Status: DID NOT CONVERGE")
        else:
            if self.verbose:
                print(
                "[MWUA] Status: CONVERGED at iteration",
                convergence_iteration,
            )

        scale_factor = (
            self.min_feasibility_ratio(
                x_avg,
                problem,
            )
        )

        if scale_factor < 1.0:
            if self.verbose:
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

    history_iterations=history_iterations,

    history_weights=history_weights,

    history_x_avg=history_x_avg,

    convergence_iteration=convergence_iteration,

    completed_rounds=completed_rounds,

    converged=(
        convergence_iteration
        is not None
    ),

    final_violation=final_violation,

    runtime=runtime,

)