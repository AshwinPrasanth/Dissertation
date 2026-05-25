"""branching.py — Pluggable branching strategies for the B&B solver.

The main branching score is intentionally small and hypothesis-driven:

    score(v) = alpha * LP_certainty(v)
             + beta  * MWUA_certainty(v)
             + gamma * residualDegree(v)
             + delta * pseudoCost(v)

Plus a small set of comparison baselines:
    - RandomBranching
    - MostFractionalBranching  (classical uncertainty-first, the "opposite")
    - DegreeBranching          (degree centrality heuristic)
    - PseudoCostBranching      (pure pseudo-cost, no structural signal)

All strategies share a common BranchingStrategy interface so the solver
can swap them without any other changes.

The direction-selection policy is simple LP-guided rounding with MWUA as a
lightweight tie-break when available.
"""

from __future__ import annotations

import abc
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from core import GraphState, VertexFeatures


# ---------------------------------------------------------------------------
# Pseudo-cost tracker
# ---------------------------------------------------------------------------

@dataclass
class PseudoCostEntry:
    """Empirical cost of branching on variable i in direction d ∈ {0, 1}."""
    sum_improvement: float = 0.0
    count: int = 0

    @property
    def estimate(self) -> float:
        return self.sum_improvement / self.count if self.count > 0 else 0.0


class PseudoCostTracker:
    """Track per-variable pseudo-costs, updated after each LP solve.

    Pseudo-cost for variable i, direction d:
        PC(i, d) = mean over past branches of (obj_after - obj_before) / |fraction|

    Combined pseudo-cost:
        PC(i) = PC(i, 0) * (1 - f_i) + PC(i, 1) * f_i
    where f_i = LP fractional value.

    Reliability: use pseudo-cost only after min_obs observations per direction;
    fall back to a structural prior otherwise.
    """

    def __init__(self, n_vars: int, min_obs: int = 3) -> None:
        self.n_vars = n_vars
        self.min_obs = min_obs
        self._costs: list[list[PseudoCostEntry]] = [
            [PseudoCostEntry(), PseudoCostEntry()] for _ in range(n_vars)
        ]

    def update(self, var: int, direction: int, obj_before: float, obj_after: float, fraction: float) -> None:
        """Record an observed branching outcome."""
        if obj_after >= 1e30 or obj_before >= 1e30:
            return
        denom = abs(fraction - direction)
        if denom < 1e-9:
            return
        improvement = max(obj_after - obj_before, 0.0) / denom
        entry = self._costs[var][direction]
        entry.sum_improvement += improvement
        entry.count += 1

    def score(self, var: int, frac_val: float) -> float:
        """Combined pseudo-cost estimate; 0 if insufficient observations."""
        f = frac_val
        pc0 = self._costs[var][0]
        pc1 = self._costs[var][1]
        if pc0.count < self.min_obs or pc1.count < self.min_obs:
            return 0.0
        return pc0.estimate * (1.0 - f) + pc1.estimate * f

    def is_reliable(self, var: int) -> bool:
        return (
            self._costs[var][0].count >= self.min_obs
            and self._costs[var][1].count >= self.min_obs
        )


# ---------------------------------------------------------------------------
# BranchingStrategy interface
# ---------------------------------------------------------------------------

class BranchingStrategy(abc.ABC):
    """Abstract branching strategy."""

    @abc.abstractmethod
    def select(
        self,
        lp_x: np.ndarray,
        features: VertexFeatures | None,
        graph_state: GraphState | None,
        pseudo_costs: PseudoCostTracker | None,
        tol: float,
        depth: int,
    ) -> Optional[int]:
        """Return variable index to branch on, or None if all integral."""
        ...

    def preferred_direction(self, var: int, lp_x: np.ndarray, features: VertexFeatures | None) -> int:
        """Return preferred branch direction (0 or 1) for the chosen variable."""
        # Default: round toward LP value; tie-break toward MWUA if available.
        if features is not None and features.mwua_x_avg is not None:
            combined = 0.6 * lp_x[var] + 0.4 * features.mwua_x_avg[var]
            return 1 if combined >= 0.5 else 0
        return 1 if lp_x[var] >= 0.5 else 0

    def _fractional_mask(self, lp_x: np.ndarray, tol: float) -> np.ndarray:
        return np.flatnonzero(
            (lp_x > tol) & (lp_x < 1.0 - tol) & (np.abs(lp_x - np.rint(lp_x)) > tol)
        )


# ---------------------------------------------------------------------------
# 1. CertaintyFirstBranching  (the core research contribution)
# ---------------------------------------------------------------------------

@dataclass
class CertaintyFirstConfig:
    """Weights for the composite branching score."""
    alpha: float = 0.40       # LP certainty  |x_v - 0.5|
    beta: float = 0.30        # MWUA certainty
    gamma: float = 0.20       # residual degree (normalised)
    delta: float = 0.10       # pseudo-cost


class CertaintyFirstBranching(BranchingStrategy):
    """Branch on the variable with highest composite certainty score.

    This is the "certainty-first" hypothesis: a lightweight composite score
    can be strong enough to guide DFS diving without expensive per-node
    feature recomputation.
    """

    def __init__(self, config: CertaintyFirstConfig | None = None) -> None:
        self.cfg = config or CertaintyFirstConfig()

    def select(
        self,
        lp_x: np.ndarray,
        features: VertexFeatures | None,
        graph_state: GraphState | None,
        pseudo_costs: PseudoCostTracker | None,
        tol: float,
        depth: int,
    ) -> Optional[int]:
        frac = self._fractional_mask(lp_x, tol)
        if frac.size == 0:
            return None

        scores = np.zeros(len(lp_x), dtype=float)

        # LP certainty
        lp_cert = np.abs(lp_x - 0.5)
        scores += self.cfg.alpha * lp_cert / 0.5   # normalise to [0, 1]

        # MWUA certainty
        if features is not None:
            scores += self.cfg.beta * features.mwua_certainty

        # Residual degree ratio
        if graph_state is not None:
            rd_ratio = graph_state.residual_degree_ratio
            scores += self.cfg.gamma * rd_ratio

        # Pseudo-cost
        if pseudo_costs is not None:
            for i in frac:
                if pseudo_costs.is_reliable(i):
                    # Normalise: max plausible pseudo-cost ≈ 10
                    scores[i] += self.cfg.delta * min(pseudo_costs.score(i, lp_x[i]) / 10.0, 1.0)

        return int(frac[np.argmax(scores[frac])])

    def preferred_direction(self, var: int, lp_x: np.ndarray, features: VertexFeatures | None) -> int:
        if features is not None:
            combined = 0.6 * lp_x[var] + 0.4 * features.mwua_x_avg[var]
            return 1 if combined >= 0.5 else 0
        return 1 if lp_x[var] >= 0.5 else 0


# ---------------------------------------------------------------------------
# 2. MostFractionalBranching  (classical uncertainty-first, main comparison)
# ---------------------------------------------------------------------------

class MostFractionalBranching(BranchingStrategy):
    """Branch on the variable closest to 0.5 (maximum uncertainty).

    This is the classical baseline. Selecting this strategy in ablation
    experiments directly tests the certainty-first hypothesis.
    """

    def select(self, lp_x, features, graph_state, pseudo_costs, tol, depth) -> Optional[int]:
        frac = self._fractional_mask(lp_x, tol)
        if frac.size == 0:
            return None
        # Closest to 0.5 = most fractional = most uncertain
        return int(frac[np.argmin(np.abs(lp_x[frac] - 0.5))])

    def preferred_direction(self, var: int, lp_x, features) -> int:
        return 1 if lp_x[var] >= 0.5 else 0


# ---------------------------------------------------------------------------
# 3. RandomBranching
# ---------------------------------------------------------------------------

class RandomBranching(BranchingStrategy):
    """Uniform random branching — weakest baseline."""

    def select(self, lp_x, features, graph_state, pseudo_costs, tol, depth) -> Optional[int]:
        frac = self._fractional_mask(lp_x, tol)
        if frac.size == 0:
            return None
        return int(np.random.choice(frac))

    def preferred_direction(self, var: int, lp_x, features) -> int:
        return random.randint(0, 1)


# ---------------------------------------------------------------------------
# 4. DegreeBranching  (graph-structure baseline — no MWUA, no pseudo-costs)
# ---------------------------------------------------------------------------

class DegreeBranching(BranchingStrategy):
    """Branch on the fractional variable with highest (residual) degree.

    Intermediate baseline: uses graph structure but no LP/MWUA information.
    """

    def select(self, lp_x, features, graph_state, pseudo_costs, tol, depth) -> Optional[int]:
        frac = self._fractional_mask(lp_x, tol)
        if frac.size == 0:
            return None
        if graph_state is not None:
            rd = graph_state.residual_degree
            return int(frac[np.argmax(rd[frac])])
        if features is not None:
            return int(frac[np.argmax(features.degree_rank[frac])])
        return int(frac[0])

    def preferred_direction(self, var: int, lp_x, features) -> int:
        return 1 if lp_x[var] >= 0.5 else 0


# ---------------------------------------------------------------------------
# 5. PseudoCostBranching  (pure empirical, no structural priors)
# ---------------------------------------------------------------------------

class PseudoCostBranching(BranchingStrategy):
    """Branch purely on pseudo-cost; fall back to LP certainty when unreliable.

    Tests whether structural features add value beyond pure empirical cost.
    """

    def select(self, lp_x, features, graph_state, pseudo_costs, tol, depth) -> Optional[int]:
        frac = self._fractional_mask(lp_x, tol)
        if frac.size == 0:
            return None

        if pseudo_costs is None:
            return int(frac[np.argmax(np.abs(lp_x[frac] - 0.5))])

        scores = np.zeros(len(lp_x), dtype=float)
        for i in frac:
            if pseudo_costs.is_reliable(i):
                scores[i] = pseudo_costs.score(i, lp_x[i])
            else:
                # Fall back to LP certainty as prior
                scores[i] = abs(lp_x[i] - 0.5)

        return int(frac[np.argmax(scores[frac])])

    def preferred_direction(self, var: int, lp_x, features) -> int:
        return 1 if lp_x[var] >= 0.5 else 0


# ---------------------------------------------------------------------------
# 6. MWUAOnlyBranching  (ablation: MWUA signal without LP certainty)
# ---------------------------------------------------------------------------

class MWUAOnlyBranching(BranchingStrategy):
    """Ablation: branch on MWUA certainty only, ignoring LP values for ranking."""

    def select(self, lp_x, features, graph_state, pseudo_costs, tol, depth) -> Optional[int]:
        frac = self._fractional_mask(lp_x, tol)
        if frac.size == 0:
            return None
        if features is None:
            return int(frac[np.argmax(np.abs(lp_x[frac] - 0.5))])
        return int(frac[np.argmax(features.mwua_certainty[frac])])

    def preferred_direction(self, var: int, lp_x, features) -> int:
        if features is not None:
            return 1 if features.mwua_x_avg[var] >= 0.5 else 0
        return 1 if lp_x[var] >= 0.5 else 0


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

STRATEGIES: dict[str, type[BranchingStrategy]] = {
    "certainty_first": CertaintyFirstBranching,
    "most_fractional": MostFractionalBranching,
    "random": RandomBranching,
    "degree": DegreeBranching,
    "pseudo_cost": PseudoCostBranching,
    "mwua_only": MWUAOnlyBranching,
}


def make_strategy(name: str, **kwargs) -> BranchingStrategy:
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{name}'. Available: {list(STRATEGIES)}")
    cls = STRATEGIES[name]
    if name == "certainty_first" and kwargs:
        return cls(config=CertaintyFirstConfig(**kwargs))
    return cls()


__all__ = [
    "BranchingStrategy",
    "CertaintyFirstBranching",
    "CertaintyFirstConfig",
    "MostFractionalBranching",
    "RandomBranching",
    "DegreeBranching",
    "PseudoCostBranching",
    "MWUAOnlyBranching",
    "PseudoCostTracker",
    "STRATEGIES",
    "make_strategy",
]
