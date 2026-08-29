from abc import ABC, abstractmethod

import networkx as nx
import numpy as np

from pyscipopt import (
    Branchrule,
    SCIP_RESULT,
)

from research.search_statistics import SearchStatistics


class BranchingStrategy(ABC):

    @abstractmethod
    def select(
        self,
        lp_solution: np.ndarray,
    ) -> int:
        pass


class MostFractionalBranching(
    BranchingStrategy
):

    def select(
        self,
        lp_solution: np.ndarray,
    ) -> int:

        fractional_vars = [
            i
            for i, value
            in enumerate(lp_solution)
            if 1e-6 < value < 1 - 1e-6
        ]

        if not fractional_vars:
            return -1

        return min(
            fractional_vars,
            key=lambda i:
            abs(lp_solution[i] - 0.5),
        )


class MWUABranching(
    BranchingStrategy
):

    def __init__(
        self,
        mwua_certainty,
    ):

        self.mwua_certainty = (
            mwua_certainty
        )

    def select(
        self,
        lp_solution,
    ):

        fractional = [
            i
            for i, x
            in enumerate(lp_solution)
            if 1e-6 < x < 1 - 1e-6
        ]

        if not fractional:
            return -1

        return max(
            fractional,
            key=lambda i:
            self.mwua_certainty[i],
        )


class DegreeBranching(
    BranchingStrategy
):

    def __init__(
        self,
        degrees,
    ):

        self.degrees = degrees

    def select(
        self,
        lp_solution,
    ):

        fractional = [
            i
            for i, x
            in enumerate(lp_solution)
            if 1e-6 < x < 1 - 1e-6
        ]

        if not fractional:
            return -1

        return max(
            fractional,
            key=lambda i:
            self.degrees[i],
        )


class SCIPMWUABranchRule(
    Branchrule
):

    def __init__(
        self,
        mwua_certainty,
        mwua_prediction,
        graph,
        selector="mwua",
        use_mwua_direction=True,
        max_spine_length=1,
        certainty_threshold=0.0,
    ):

        self.mwua_certainty = np.asarray(
            mwua_certainty,
            dtype=float,
        )

        self.mwua_prediction = np.asarray(
            mwua_prediction,
            dtype=np.int8,
        )

        self.graph = graph

        self.selector = str(
            selector
        )

        self.use_mwua_direction = bool(
            use_mwua_direction
        )

        self.max_spine_length = int(
            max_spine_length
        )

        self.certainty_threshold = float(
            certainty_threshold
        )

        self.call_count = 0

        self.branch_count = 0

        self.predicted_one_count = 0

        self.predicted_zero_count = 0

        self.uncertainty_stop_count = 0

        self.length_stop_count = 0

        self.off_spine_count = 0

        self.spine_length = 0

        self.spine_active = True

        self.preferred_node_number = None

        self.spine_vertices = []

        self.spine_certainties = []

        self.spine_predictions = []

        self.spine_residual_degrees = []

        self.stats = SearchStatistics()

    @staticmethod
    def _variable_index(
        var,
    ):

        name = var.name

        if name.startswith(
            "t_x_"
        ):

            return int(
                name.replace(
                    "t_x_",
                    "",
                )
            )

        if name.startswith(
            "x_"
        ):

            return int(
                name.replace(
                    "x_",
                    "",
                )
            )

        return None

    def _is_spine_node(
        self,
    ):

        current_node = (
            self.model.getCurrentNode()
        )

        if current_node is None:
            return False

        current_number = (
            current_node.getNumber()
        )

        if self.preferred_node_number is None:

            return (
                self.model.getDepth() == 0
            )

        return (
            current_number
            == self.preferred_node_number
        )

    def _build_variable_map(
        self,
    ):

        variable_map = {}

        for var in self.model.getVars():

            idx = self._variable_index(
                var
            )

            if idx is not None:

                variable_map[idx] = var

        return variable_map

    def _is_residual_vertex(
    self,
    idx,
    variable_map,
    ):

        var = variable_map.get(
            idx
        )

        if var is None:
            return False

        lb = var.getLbLocal()

        ub = var.getUbLocal()

        return (
            lb < 0.5
            and ub > 0.5
        )

    def _residual_degree(
        self,
        idx,
        variable_map,
    ):

        if idx not in self.graph:
            return 0

        degree = 0

        for neighbour in self.graph.neighbors(
            idx
        ):

            if self._is_residual_vertex(
                neighbour,
                variable_map,
            ):

                degree += 1

        return degree

    def _select_candidate(
        self,
        cands,
        npriocands,
    ):

        variable_map = (
            self._build_variable_map()
        )

        candidates = []

        for i in range(
            npriocands
        ):

            var = cands[i]

            idx = self._variable_index(
                var
            )

            if idx is None:
                continue

            certainty = float(
                self.mwua_certainty[
                    idx
                ]
            )

            residual_degree = (
                self._residual_degree(
                    idx,
                    variable_map,
                )
            )

            candidates.append(
                (
                    var,
                    idx,
                    certainty,
                    residual_degree,
                )
            )

        if not candidates:

            return None

        if self.selector == "mwua":

            return max(
                candidates,
                key=lambda item: (
                    item[2],
                    item[3],
                    -item[1],
                ),
            )

        if self.selector == "residual_degree":

            return max(
                candidates,
                key=lambda item: (
                    item[3],
                    item[2],
                    -item[1],
                ),
            )

        raise ValueError(
            "Unknown selector: "
            + self.selector
        )

    def branchexeclp(
        self,
        allowaddcons,
    ):

        self.call_count += 1

        depth = (
            self.model.getDepth()
        )

        self.stats.record_node(
            depth
        )

        if not self.spine_active:

            return {
                "result":
                    SCIP_RESULT.DIDNOTRUN
            }

        if not self._is_spine_node():

            self.off_spine_count += 1

            return {
                "result":
                    SCIP_RESULT.DIDNOTRUN
            }

        if (
            self.spine_length
            >= self.max_spine_length
        ):

            self.length_stop_count += 1

            self.spine_active = False

            print()

            print(
                "[SPINE LENGTH STOP]",
                "depth=",
                depth,
                "length=",
                self.spine_length,
                "max_length=",
                self.max_spine_length,
            )

            return {
                "result":
                    SCIP_RESULT.DIDNOTRUN
            }

        (
            cands,
            candssol,
            candsscore,
            ncands,
            npriocands,
            nfracimplvars,
        ) = (
            self.model
            .getLPBranchCands()
        )

        if ncands == 0:

            self.spine_active = False

            return {
                "result":
                    SCIP_RESULT.DIDNOTRUN
            }

        selected = (
            self._select_candidate(
                cands,
                npriocands,
            )
        )

        if selected is None:

            self.spine_active = False

            return {
                "result":
                    SCIP_RESULT.DIDNOTRUN
            }

        (
            best_var,
            best_idx,
            best_certainty,
            best_residual_degree,
        ) = selected

        if (
            self.selector == "mwua"
            and
            best_certainty
            < self.certainty_threshold
        ):

            self.uncertainty_stop_count += 1

            self.spine_active = False

            print()

            print(
                "[SPINE UNCERTAINTY STOP]",
                "depth=",
                depth,
                "certainty=",
                f"{best_certainty:.6f}",
                "threshold=",
                f"{self.certainty_threshold:.6f}",
            )

            return {
                "result":
                    SCIP_RESULT.DIDNOTRUN
            }

        if self.use_mwua_direction:

            prediction = int(
                self.mwua_prediction[
                    best_idx
                ]
            )

        else:

            lp_value = float(
                self.model.getSolVal(
                    None,
                    best_var,
                )
            )

            prediction = int(
                lp_value >= 0.5
            )

        preferred_value = float(
            prediction
        )

        deferred_value = float(
            1 - prediction
        )

        preferred_estimate = (
            self.model.calcChildEstimate(
                best_var,
                preferred_value,
            )
        )

        deferred_estimate = (
            self.model.calcChildEstimate(
                best_var,
                deferred_value,
            )
        )

        preferred_child = (
            self.model.createChild(
                1000000,
                preferred_estimate,
            )
        )

        deferred_child = (
            self.model.createChild(
                -1000000,
                deferred_estimate,
            )
        )

        if prediction == 1:

            self.model.chgVarLbNode(
                preferred_child,
                best_var,
                1.0,
            )

            self.model.chgVarUbNode(
                deferred_child,
                best_var,
                0.0,
            )

            self.predicted_one_count += 1

        else:

            self.model.chgVarUbNode(
                preferred_child,
                best_var,
                0.0,
            )

            self.model.chgVarLbNode(
                deferred_child,
                best_var,
                1.0,
            )

            self.predicted_zero_count += 1

        self.preferred_node_number = (
            preferred_child.getNumber()
        )

        self.branch_count += 1

        self.spine_length += 1

        self.spine_vertices.append(
            best_idx
        )

        self.spine_certainties.append(
            best_certainty
        )

        self.spine_predictions.append(
            prediction
        )

        self.spine_residual_degrees.append(
            best_residual_degree
        )

        self.stats.record_branch()

        print(
            "[SPINE]",
            "selector=",
            self.selector,
            "step=",
            self.spine_length,
            "depth=",
            depth,
            "vertex=",
            best_idx,
            "certainty=",
            f"{best_certainty:.6f}",
            "residual_degree=",
            best_residual_degree,
            "prediction=",
            prediction,
            "mwua_direction=",
            self.use_mwua_direction,
            "preferred_node=",
            self.preferred_node_number,
        )

        return {
            "result":
                SCIP_RESULT.BRANCHED
        }

from branch_sample import BranchSample


class SCIPSBDataCollector(Branchrule):

    FEATURE_NAMES = [
        "degree_rank",
        "nbr_min_rank",
        "nbr_max_rank",
        "nbr_avg_rank",
        "pagerank",
        "core_number",
        "clustering",
        "degree_centrality",
        "mwua_xavg",
        "mwua_weight_min",
        "mwua_weight_max",
        "mwua_weight_avg",
        "lp_value",
        "lp_certainty",
        "luby_frequency",
    ]

    def __init__(
        self,
        static_features,
        graph,
        graph_name,
        writer,
        max_sb_nodes=10,
        candidate_limit=None,
        strongbranch_itlim=100,
    ):
        super().__init__()

        self.static_features = static_features
        self.graph = graph
        self.graph_name = str(graph_name)
        self.writer = writer
        self.max_sb_nodes = int(max_sb_nodes)
        self.candidate_limit = (
            None
            if candidate_limit is None
            else int(candidate_limit)
        )
        self.strongbranch_itlim = int(strongbranch_itlim)

        self.call_count = 0
        self.sb_nodes = 0
        self.sb_candidates = 0
        self.branch_count = 0

    @staticmethod
    def _variable_index(var):
        name = var.name

        if name.startswith("t_x_"):
            return int(name.replace("t_x_", ""))

        if name.startswith("x_"):
            return int(name.replace("x_", ""))

        return None

    def _build_variable_map(self):
        variable_map = {}

        for var in self.model.getVars():
            idx = self._variable_index(var)

            if idx is not None:
                variable_map[idx] = var

        return variable_map

    @staticmethod
    def _is_residual_variable(var):
        return (
            var.getLbLocal() < 0.5
            and var.getUbLocal() > 0.5
        )

    def _build_residual_graph(self, variable_map):
        residual_vertices = [
            idx
            for idx, var in variable_map.items()
            if self._is_residual_variable(var)
        ]

        return self.graph.subgraph(residual_vertices).copy()

    @staticmethod
    def _dynamic_structural_features(residual_graph):
        nodes = list(residual_graph.nodes())

        if not nodes:
            return {}

        degrees = np.asarray(
            [residual_graph.degree(v) for v in nodes],
            dtype=float,
        )

        order = np.argsort(degrees, kind="stable")
        ranks = np.empty(len(nodes), dtype=float)
        ranks[order] = np.arange(len(nodes), dtype=float)

        if len(nodes) > 1:
            ranks /= len(nodes) - 1

        rank_by_vertex = {
            v: float(ranks[i])
            for i, v in enumerate(nodes)
        }

        if residual_graph.number_of_edges() > 0:
            core_number = nx.core_number(residual_graph)
        else:
            core_number = {v: 0 for v in nodes}

        clustering = nx.clustering(residual_graph)
        degree_centrality = nx.degree_centrality(residual_graph)

        features = {}

        for v in nodes:
            neighbors = list(residual_graph.neighbors(v))

            if neighbors:
                neighbor_ranks = np.asarray(
                    [rank_by_vertex[u] for u in neighbors],
                    dtype=float,
                )

                nbr_min_rank = float(neighbor_ranks.min())
                nbr_max_rank = float(neighbor_ranks.max())
                nbr_avg_rank = float(neighbor_ranks.mean())
            else:
                nbr_min_rank = 0.0
                nbr_max_rank = 0.0
                nbr_avg_rank = 0.0

            features[v] = {
                "degree_rank": rank_by_vertex[v],
                "nbr_min_rank": nbr_min_rank,
                "nbr_max_rank": nbr_max_rank,
                "nbr_avg_rank": nbr_avg_rank,
                "core_number": float(core_number[v]),
                "clustering": float(clustering[v]),
                "degree_centrality": float(degree_centrality[v]),
            }

        return features

    def _selected_candidates(
        self,
        cands,
        candssol,
        npriocands,
    ):
        candidates = []

        for i in range(npriocands):
            var = cands[i]
            idx = self._variable_index(var)

            if idx is None:
                continue

            candidates.append(
                (
                    var,
                    idx,
                    float(candssol[i]),
                )
            )

        if self.candidate_limit is not None:
            candidates = candidates[:self.candidate_limit]

        return candidates

    def branchexeclp(self, allowaddcons):
        (
            cands,
            candssol,
            candsscore,
            ncands,
            npriocands,
            nfracimplvars,
        ) = self.model.getLPBranchCands()

        if ncands == 0:
            return {"result": SCIP_RESULT.DIDNOTRUN}

        candidates = self._selected_candidates(
            cands,
            candssol,
            npriocands,
        )

        if not candidates:
            return {"result": SCIP_RESULT.DIDNOTRUN}

        if self.sb_nodes >= self.max_sb_nodes:
            self.model.interruptSolve()

            return {
                "result":
                    SCIP_RESULT.DIDNOTRUN
            }

        depth = self.model.getDepth()
        current_node = self.model.getCurrentNode()

        node_number = (
            current_node.getNumber()
            if current_node is not None
            else -1
        )

        parent_lp_obj = float(self.model.getLPObjVal())

        variable_map = self._build_variable_map()
        residual_graph = self._build_residual_graph(variable_map)

        dynamic_features = self._dynamic_structural_features(
            residual_graph
        )

        candidate_ids = []
        candidate_features = []
        lp_values = []
        sb_down_bounds = []
        sb_up_bounds = []
        sb_down_gains = []
        sb_up_gains = []
        sb_scores = []
        sb_down_valid = []
        sb_up_valid = []
        sb_down_infeasible = []
        sb_up_infeasible = []
        candidate_vars = []

        best_score = -self.model.infinity()
        best_var = None

        self.model.startStrongbranch()

        try:
            for var, idx, lp_value in candidates:
                if idx not in dynamic_features:
                    continue

                (
                    down,
                    up,
                    downvalid,
                    upvalid,
                    downinf,
                    upinf,
                    downconflict,
                    upconflict,
                    lperror,
                ) = self.model.getVarStrongbranch(
                    var,
                    self.strongbranch_itlim,
                    idempotent=False,
                )

                if lperror:
                    continue

                if downinf and upinf:
                    return {"result": SCIP_RESULT.CUTOFF}

                if not downinf and downvalid:
                    down_gain = max(
                        float(down) - parent_lp_obj,
                        0.0,
                    )
                else:
                    down_gain = 0.0

                if not upinf and upvalid:
                    up_gain = max(
                        float(up) - parent_lp_obj,
                        0.0,
                    )
                else:
                    up_gain = 0.0

                score = float(
                    self.model.getBranchScoreMultiple(
                        var,
                        [
                            down_gain,
                            up_gain,
                        ],
                    )
                )

                feature_values = {}
                feature_values.update(dynamic_features[idx])
                feature_values.update(self.static_features[idx])
                feature_values["lp_value"] = lp_value
                feature_values["lp_certainty"] = abs(lp_value - 0.5)

                candidate_ids.append(idx)

                candidate_features.append(
                    [
                        float(feature_values[name])
                        for name in self.FEATURE_NAMES
                    ]
                )

                lp_values.append(lp_value)
                sb_down_bounds.append(float(down))
                sb_up_bounds.append(float(up))
                sb_down_gains.append(down_gain)
                sb_up_gains.append(up_gain)
                sb_scores.append(score)
                sb_down_valid.append(int(downvalid))
                sb_up_valid.append(int(upvalid))
                sb_down_infeasible.append(int(downinf))
                sb_up_infeasible.append(int(upinf))
                candidate_vars.append(var)

                if score > best_score:
                    best_score = score
                    best_var = var

        finally:
            self.model.endStrongbranch()

        if best_var is None:
            return {"result": SCIP_RESULT.DIDNOTRUN}

        chosen_variable = self._variable_index(best_var)

        sample = BranchSample(
            graph_name=self.graph_name,
            node_number=node_number,
            depth=depth,
            residual_n=residual_graph.number_of_nodes(),
            residual_m=residual_graph.number_of_edges(),
            parent_lp_obj=parent_lp_obj,
            feature_names=np.asarray(
                self.FEATURE_NAMES,
                dtype=object,
            ),
            candidate_ids=np.asarray(
                candidate_ids,
                dtype=np.int32,
            ),
            candidate_features=np.asarray(
                candidate_features,
                dtype=float,
            ),
            lp_values=np.asarray(
                lp_values,
                dtype=float,
            ),
            sb_down_bounds=np.asarray(
                sb_down_bounds,
                dtype=float,
            ),
            sb_up_bounds=np.asarray(
                sb_up_bounds,
                dtype=float,
            ),
            sb_down_gains=np.asarray(
                sb_down_gains,
                dtype=float,
            ),
            sb_up_gains=np.asarray(
                sb_up_gains,
                dtype=float,
            ),
            sb_scores=np.asarray(
                sb_scores,
                dtype=float,
            ),
            sb_down_valid=np.asarray(
                sb_down_valid,
                dtype=np.int8,
            ),
            sb_up_valid=np.asarray(
                sb_up_valid,
                dtype=np.int8,
            ),
            sb_down_infeasible=np.asarray(
                sb_down_infeasible,
                dtype=np.int8,
            ),
            sb_up_infeasible=np.asarray(
                sb_up_infeasible,
                dtype=np.int8,
            ),
            chosen_variable=chosen_variable,
            best_sb_score=float(best_score),
        )

        self.writer.save(sample)

        self.call_count += 1
        self.sb_nodes += 1
        self.sb_candidates += len(candidate_ids)

        print(
            "[SB DATA]",
            f"graph={self.graph_name}",
            f"node={node_number}",
            f"depth={depth}",
            f"residual_n={residual_graph.number_of_nodes()}",
            f"candidates={len(candidate_ids)}",
            f"best={chosen_variable}",
            f"best_score={best_score:.8g}",
            f"sb_nodes={self.sb_nodes}/{self.max_sb_nodes}",
        )

        if self.sb_nodes >= self.max_sb_nodes:

            print()

            print(
                "[SB COLLECTION COMPLETE]",
                f"graph={self.graph_name}",
                f"sb_nodes={self.sb_nodes}",
            )

            self.model.interruptSolve()

            return {
                "result":
                    SCIP_RESULT.DIDNOTRUN
            }

        self.model.branchVar(
            best_var
        )

        self.branch_count += 1

        return {
            "result":
                SCIP_RESULT.BRANCHED
        }
