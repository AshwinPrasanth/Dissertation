from dataclasses import dataclass
import numpy as np
from problem import MILPProblem

'''
The Greedy MWUA 

    for t:

    compute scores

    greedy oracle

    update x_avg

    save weight history

    update weights

    if t % 10 == 0 and max_violation(x_avg) <= delta:
        break'''

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
        delta: float = 1e-3,
    ):
        self.rounds = rounds
        self.eps = eps
        self.delta = delta

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

    def max_violation(self, x: np.ndarray,problem: MILPProblem,) -> float:
        """
        Ryan-style convergence check.
        """

        max_v = 0.0

        '''for row in problem.A_ub:

            vars_in_constraint = np.where(
                np.abs(row) > 0
            )[0]

            lhs = np.sum(
                x[vars_in_constraint]
            )

            if problem.problem_type == "mvc":

                violation = max(
                    0.0,
                    1.0 - lhs,
                )

            elif problem.problem_type == "mis":

                violation = max(
                    0.0,
                    lhs - 1.0,
                )

            else:

                raise ValueError(
                    "Unknown problem type"
                )

            max_v = max(
                max_v,
                violation,
            )'''
        
        for u, v in problem.edges:

            lhs = x[u] + x[v]

            violation = max(0.0,1.0 - lhs,)

            max_v = max(max_v,violation)

        return max_v
    
    def min_feasibility_ratio(
        self,
        x,
        problem,
    ):
        EPS = 1e-15


        alpha = 0.0

        for u, v in problem.edges:

            cover = x[u] + x[v]

            if cover <= EPS:

                x[u] += 0.5
                x[v] += 0.5

                cover = 1.0

            ratio = 1.0 / cover

            alpha = max(alpha, ratio)

        return alpha



    def compute(
        self,
        problem: MILPProblem,
    ) -> MWUAResult:

        # m = len(problem.A_ub)
        m = len(problem.edges)
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

            # scores = (np.abs(problem.A_ub).T@ normalized_weights)
            scores = np.zeros(n)

            for edge_idx, (u, v) in enumerate(problem.edges):

                w = normalized_weights[edge_idx]

                scores[u] += w
                scores[v] += w
            if t in [1, 10, 50, 100]:
                print(
                    f"Round {t}: "
                    f"score min={scores.min():.6f}, "
                    f"score max={scores.max():.6f}"
                    )

            # ---------------------------------
            # Greedy fractional oracle
            # ---------------------------------

            x_t = self.greedy_fractional_solution(
                scores
            )
            
            if t in [1, 10, 50, 100]:
                print(
                    f"Round {t}: "
                    f"x_t nonzero={np.count_nonzero(x_t)}"
                )

                print(
                    f"Round {t}: "
                    f"x_t max={x_t.max():.6f}"
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

                '''row = problem.A_ub[edge_idx]

                vars_in_edge = np.where(
                np.abs(row) > 0
                )[0]

                cover = np.sum(
                    x_t[vars_in_edge]
                )'''
                
                u, v = problem.edges[edge_idx]

                cover = (x_t[u] + x_t[v])

                violation = max(0.0,1.0 - cover)

                
                if (
                    t == 1
                    and edge_idx < 10
                ):
                    '''print("edge:",edge_idx,)
                    print("vars:",vars_in_edge,)
                    print("x_t:",x_t[vars_in_edge],)
                    print("cover:",cover,)
                    print("violation:",violation,)
                    print("-" * 30)'''

                weights[edge_idx] *= (
                    1.0
                    + self.eps
                    * violation
                )
            if t <= 3:
                print(
                    f"Round {t}: "
                    f"max weight = {weights.max():.6f}, "
                    f"min weight = {weights.min():.6f}"
                )
            if t % 10 == 0:

                violation = self.max_violation(
                    x_avg,
                    problem,
                )

                print(
                    "Round",
                    t,
                    "violation =",
                    violation,
                )

                if violation <= self.delta:
                    break
            '''if t % 10 == 0:
                print(
                    "Round",
                    t,
                    "violation =",
                    self.max_violation(
                        x_avg,
                        problem
                    )
                )
            if ( t % 10 == 0 and self.max_violation(x_avg, problem,) <= self.delta):
                break'''

        weight_history = np.array(
            weight_history
        )
        print( "Final violation:",self.max_violation(x_avg,problem,))
        # ---------------------------------
        # Ryan feasibility scaling
        # ---------------------------------

        scale_factor = (
            self.min_feasibility_ratio(
                x_avg,
                problem,
            )
        )
        
        print("Scale factor:",scale_factor)

        if scale_factor < 1.0:

            print("[INFO] Scaling solution by",scale_factor,)

            x_avg *= scale_factor
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