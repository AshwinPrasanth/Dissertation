"""core.py — Problem definitions, a minimal MWUA snapshot, and graph state.

The week-1 proposal keeps the global snapshot intentionally small:

1. PrincipledMWUA: root-level multiplicative weights update producing a
    global fractional estimate and per-variable MWUA certainty.

2. StructuralFeatureEngine: computes the root snapshot once and exposes only
    the MWUA signals plus the root LP certainty snapshot used for analysis.

3. GraphState: a living, incrementally-updatable view of the residual graph
    that provides the cheap local signal used during branching.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
from scipy.optimize import linprog


ArrayLike = Sequence[float] | np.ndarray


# ---------------------------------------------------------------------------
# Problem definition
# ---------------------------------------------------------------------------

@dataclass
class MILPProblem:
    """Binary MILP — minimise c·x subject to A_ub·x ≤ b_ub, x ∈ {0,1}^n."""

    c: np.ndarray
    A_ub: np.ndarray | None = None
    b_ub: np.ndarray | None = None
    A_eq: np.ndarray | None = None
    b_eq: np.ndarray | None = None
    bounds: list[tuple[float, float]] | None = None
    # Optional: the underlying graph for structure-aware features.
    graph: nx.Graph | None = None

    def __post_init__(self) -> None:
        self.c = np.asarray(self.c, dtype=float)
        n = self.c.size
        self.A_ub = _coerce_matrix(self.A_ub, n)
        self.b_ub = _coerce_vector(self.b_ub, self.A_ub.shape[0] if self.A_ub is not None else 0)
        self.A_eq = _coerce_matrix(self.A_eq, n)
        self.b_eq = _coerce_vector(self.b_eq, self.A_eq.shape[0] if self.A_eq is not None else 0)
        if self.bounds is None:
            self.bounds = [(0.0, 1.0)] * n
        else:
            if len(self.bounds) != n:
                raise ValueError("bounds length mismatch")
            self.bounds = [(float(lo), float(hi)) for lo, hi in self.bounds]

    @property
    def n_vars(self) -> int:
        return int(self.c.size)


def _coerce_matrix(m: ArrayLike | None, n: int) -> np.ndarray | None:
    if m is None:
        return None
    a = np.asarray(m, dtype=float)
    if a.ndim != 2 or a.shape[1] != n:
        raise ValueError(f"Matrix shape {a.shape} incompatible with n_vars={n}")
    return a


def _coerce_vector(v: ArrayLike | None, size: int) -> np.ndarray | None:
    if v is None:
        return None if size == 0 else np.zeros(size, dtype=float)
    a = np.asarray(v, dtype=float)
    if a.ndim != 1:
        raise ValueError("Vector must be 1-D")
    return a


def build_mvc_problem(graph: nx.Graph) -> MILPProblem:
    """Standard minimum vertex cover ILP from a graph."""
    nodes = sorted(graph.nodes())
    node_idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    edges = list(graph.edges())
    m = len(edges)
    c = np.ones(n, dtype=float)
    A_ub = np.zeros((m, n), dtype=float)
    b_ub = -np.ones(m, dtype=float)
    for k, (u, v) in enumerate(edges):
        A_ub[k, node_idx[u]] = -1.0
        A_ub[k, node_idx[v]] = -1.0
    return MILPProblem(c=c, A_ub=A_ub, b_ub=b_ub, graph=graph)


def build_problem(
    c: ArrayLike,
    A_ub: ArrayLike | None = None,
    b_ub: ArrayLike | None = None,
    A_eq: ArrayLike | None = None,
    b_eq: ArrayLike | None = None,
    bounds: Iterable[tuple[float, float]] | None = None,
    graph: nx.Graph | None = None,
) -> MILPProblem:
    return MILPProblem(
        c=np.asarray(c, dtype=float),
        A_ub=None if A_ub is None else np.asarray(A_ub, dtype=float),
        b_ub=None if b_ub is None else np.asarray(b_ub, dtype=float),
        A_eq=None if A_eq is None else np.asarray(A_eq, dtype=float),
        b_eq=None if b_eq is None else np.asarray(b_eq, dtype=float),
        bounds=None if bounds is None else list(bounds),
        graph=graph,
    )


# ---------------------------------------------------------------------------
# LP relaxation
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class LPResult:
    status: int
    obj: float
    x: np.ndarray | None
    message: str

    @property
    def feasible(self) -> bool:
        return self.status == 0 and self.x is not None


def solve_lp(problem: MILPProblem, bounds: list[tuple[float, float]]) -> LPResult:
    res = linprog(
        c=problem.c,
        A_ub=problem.A_ub,
        b_ub=problem.b_ub,
        A_eq=problem.A_eq,
        b_eq=problem.b_eq,
        bounds=bounds,
        method="highs",
    )
    obj = float(res.fun) if res.fun is not None else np.inf
    x = np.asarray(res.x, dtype=float) if res.x is not None else None
    return LPResult(status=int(res.status), obj=obj, x=x, message=res.message)


# ---------------------------------------------------------------------------
# Principled MWUA  (Algorithm 1 from the CPAIOR paper, faithfully implemented)
# ---------------------------------------------------------------------------

@dataclass
class MWUAResult:
    """Full output of the MWUA run."""
    # Per-variable approximate fractional solution (average across iterations).
    x_avg: np.ndarray
    # Per-constraint final weights.
    constraint_weights: np.ndarray
    # Per-variable: average, min, max of incident constraint weights.
    avg_incident_weight: np.ndarray
    min_incident_weight: np.ndarray
    max_incident_weight: np.ndarray
    # LP-surrogate certainty: distance of x_avg from 0.5 ∈ [0, 0.5].
    certainty: np.ndarray
    iterations_run: int
    converged: bool


class PrincipledMWUA:
    """Multiplicative Weights Update for packing/covering LP surrogates.

    Follows Algorithm 1 in the paper exactly.  Supports both MIS (edge
    constraints) and generic inequality constraints from MILPProblem.

    Convergence criterion: mean fractional change in x_avg < tol.
    A hard time cap ensures predictable runtime.
    """

    def __init__(
        self,
        eta: float = 0.1,
        max_iter: int = 200,
        time_cap_sec: float = 10.0,
        convergence_tol: float = 1e-4,
    ) -> None:
        self.eta = eta                      # multiplicative weight step
        self.max_iter = max_iter
        self.time_cap_sec = time_cap_sec
        self.convergence_tol = convergence_tol

    def run(self, problem: MILPProblem) -> MWUAResult:
        """Run MWUA on problem.A_ub · x ≤ b_ub constraints."""
        n = problem.n_vars
        A = problem.A_ub  # shape (m, n), entries are −1 for covering constraints
        b = problem.b_ub

        if A is None or b is None or A.shape[0] == 0:
            # No constraints: uniform certainty
            ones = np.ones(n, dtype=float)
            return MWUAResult(
                x_avg=0.5 * ones, constraint_weights=np.array([]),
                avg_incident_weight=ones, min_incident_weight=ones,
                max_incident_weight=ones, certainty=np.zeros(n),
                iterations_run=0, converged=True,
            )

        m = A.shape[0]
        # For covering constraints: −x_u − x_v ≤ −1 (MVC).
        # We interpret violation of constraint k as: A[k]·x − b[k] > 0.
        weights = np.ones(m, dtype=float)          # constraint weights
        x_history: list[np.ndarray] = []
        t0 = time.perf_counter()
        converged = False
        prev_x_avg = None

        for it in range(self.max_iter):
            if time.perf_counter() - t0 > self.time_cap_sec:
                break

            # Weighted score per variable: how much each variable is "needed"
            # by violated constraints.
            # For covering (A has negative entries), flip sign so high weight
            # = "this variable is needed by heavy constraints".
            w_score = (-A).T @ weights   # shape (n,), non-negative for MVC
            w_score = np.maximum(w_score, 0.0)

            # Normalise to [0, 1]
            denom = w_score.max()
            if denom > 0:
                w_score = w_score / denom

            # Fractional assignment: greedily assign based on score,
            # clipped to [0, 1] variable bounds (active bounds from problem).
            x = np.clip(w_score, 0.0, 1.0)
            x_history.append(x.copy())

            # Convergence check
            if len(x_history) >= 5:
                x_avg_now = np.mean(x_history, axis=0)
                if prev_x_avg is not None:
                    delta = np.mean(np.abs(x_avg_now - prev_x_avg))
                    if delta < self.convergence_tol:
                        converged = True
                        break
                prev_x_avg = x_avg_now

            # Constraint violation and weight update
            # violation[k] > 0  ⟺  constraint k is violated by x
            violation = A @ x - b   # shape (m,)
            # Multiplicative update: upweight violated constraints
            weights *= np.exp(self.eta * np.maximum(violation, 0.0))
            # Normalise weights to prevent overflow
            w_sum = weights.sum()
            if w_sum > 0:
                weights = weights / w_sum * m   # keep mean ≈ 1

        x_avg = np.mean(x_history, axis=0) if x_history else 0.5 * np.ones(n)

        # Per-variable incident weight statistics
        # incident[i] = indices of constraints involving variable i
        abs_A = np.abs(A)   # (m, n)
        involved = abs_A > 1e-9  # boolean mask

        avg_inc = np.zeros(n)
        min_inc = np.zeros(n)
        max_inc = np.zeros(n)
        for i in range(n):
            mask = involved[:, i]
            if mask.any():
                inc_w = weights[mask]
                avg_inc[i] = inc_w.mean()
                min_inc[i] = inc_w.min()
                max_inc[i] = inc_w.max()
            else:
                avg_inc[i] = min_inc[i] = max_inc[i] = 0.0

        # Normalise all weight arrays to [0, 1]
        for arr in [avg_inc, min_inc, max_inc]:
            mx = arr.max()
            if mx > 0:
                arr /= mx

        certainty = np.abs(x_avg - 0.5)  # ∈ [0, 0.5], 0.5 = totally certain

        return MWUAResult(
            x_avg=x_avg,
            constraint_weights=weights,
            avg_incident_weight=avg_inc,
            min_incident_weight=min_inc,
            max_incident_weight=max_inc,
            certainty=certainty,
            iterations_run=len(x_history),
            converged=converged,
        )


# ---------------------------------------------------------------------------
# Structural Feature Engine
# ---------------------------------------------------------------------------

@dataclass
class VertexFeatures:
    """Minimal root-level snapshot used by branching and analysis."""
    n: int

    # MWUA-derived (static global, computed once at root)
    mwua_x_avg: np.ndarray            # approximate LP value
    mwua_certainty: np.ndarray        # |x - 0.5| ∈ [0, 0.5]

    # Root LP certainty snapshot, kept explicit for analysis.
    root_lp_certainty: np.ndarray


class StructuralFeatureEngine:
    """Compute the minimal root-level snapshot once at the B&B root."""

    def __init__(
        self,
        mwua_eta: float = 0.1,
        mwua_max_iter: int = 200,
        mwua_time_cap: float = 10.0,
    ) -> None:
        self.mwua = PrincipledMWUA(eta=mwua_eta, max_iter=mwua_max_iter, time_cap_sec=mwua_time_cap)

    def compute(self, problem: MILPProblem, lp_root: LPResult | None = None) -> VertexFeatures:
        mwua_result = self.mwua.run(problem)

        n = problem.n_vars
        if lp_root is not None and lp_root.x is not None:
            lp_certainty = np.abs(lp_root.x - 0.5) / 0.5
        else:
            lp_certainty = np.zeros(n, dtype=float)

        return VertexFeatures(
            n=n,
            mwua_x_avg=mwua_result.x_avg,
            mwua_certainty=mwua_result.certainty / 0.5,
            root_lp_certainty=lp_certainty,
        )


# ---------------------------------------------------------------------------
# Incremental graph state: tracks residual degree and active neighbours
# ---------------------------------------------------------------------------

class GraphState:
    """Lightweight, incrementally updatable view of the residual graph.

    Rather than recomputing degree or neighbourhood statistics from scratch at
    every B&B node, GraphState maintains a mutable copy that can be pushed/
    popped as we enter/leave branches.

    Snapshot stacks allow O(degree) push/pop instead of O(n+m) recomputation.
    """

    def __init__(self, graph: nx.Graph | None, n_vars: int) -> None:
        self.n = n_vars
        # active[i] = True iff variable i is still free (not fixed)
        self.active = np.ones(n_vars, dtype=bool)
        # Residual degree: number of active neighbours for each variable.
        if graph is not None and graph.number_of_nodes() == n_vars:
            nodes = sorted(graph.nodes())
            self._node_to_idx = {v: i for i, v in enumerate(nodes)}
            self._adj: list[list[int]] = [[] for _ in range(n_vars)]
            for u, v in graph.edges():
                i, j = self._node_to_idx[u], self._node_to_idx[v]
                self._adj[i].append(j)
                self._adj[j].append(i)
            self.residual_degree = np.array(
                [len(self._adj[i]) for i in range(n_vars)], dtype=float
            )
        else:
            self._adj = [[] for _ in range(n_vars)]
            self.residual_degree = np.zeros(n_vars, dtype=float)

        # Stack for undo: each entry = (var_fixed, affected_neighbours)
        self._undo_stack: list[tuple[int, list[int]]] = []

    @property
    def residual_degree_ratio(self) -> np.ndarray:
        """Residual degree / original degree — 1.0 at root, 0 when isolated."""
        orig = np.array([len(a) for a in self._adj], dtype=float)
        orig[orig == 0] = 1.0
        return self.residual_degree / orig

    def fix_variable(self, var: int) -> None:
        """Mark variable as fixed; update residual degrees of its neighbours."""
        affected = []
        for nbr in self._adj[var]:
            if self.active[nbr]:
                self.residual_degree[nbr] -= 1.0
                affected.append(nbr)
        self.active[var] = False
        self.residual_degree[var] = 0.0
        self._undo_stack.append((var, affected))

    def undo_fix(self) -> None:
        """Restore the last fix_variable call."""
        if not self._undo_stack:
            return
        var, affected = self._undo_stack.pop()
        self.active[var] = True
        self.residual_degree[var] = float(len([n for n in self._adj[var] if self.active[n]]))
        for nbr in affected:
            self.residual_degree[nbr] += 1.0

    def local_density(self, var: int) -> float:
        """Edge density within the 1-hop neighbourhood of var."""
        nbrs = [n for n in self._adj[var] if self.active[n]]
        k = len(nbrs)
        if k < 2:
            return 0.0
        nbr_set = set(nbrs)
        edges = sum(1 for u in nbrs for v in self._adj[u] if v in nbr_set) // 2
        return edges / (k * (k - 1) / 2)


# ---------------------------------------------------------------------------
# LP forced assignments (Nemhauser–Trotter reductions)
# ---------------------------------------------------------------------------

def lp_forced_assignments(lp_x: np.ndarray, tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Identify vertices LP-forced to 0 or 1.

    For MVC: LP values exactly 0 ⟹ safely excluded from cover (set 0).
             LP values exactly 1 ⟹ safely included in cover (set 1).
    Returns (forced_zero_indices, forced_one_indices).
    """
    forced_zero = np.flatnonzero(lp_x < tol)
    forced_one = np.flatnonzero(lp_x > 1.0 - tol)
    return forced_zero, forced_one


__all__ = [
    "MILPProblem",
    "LPResult",
    "MWUAResult",
    "VertexFeatures",
    "GraphState",
    "PrincipledMWUA",
    "StructuralFeatureEngine",
    "build_mvc_problem",
    "build_problem",
    "solve_lp",
    "lp_forced_assignments",
]
