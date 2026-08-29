import numpy as np

from xgboost import XGBRanker

from pyscipopt import (
    Branchrule,
    SCIP_RESULT,
)


class SCIPXGBBranchRule(
    Branchrule
):

    FEATURE_NAMES = [

        "mwua_xavg",
        "mwua_weight_min",
        "mwua_weight_max",
        "mwua_weight_avg",

        "bipartite_core_number",
        "bipartite_pagerank",
        "frequency_rank",
        "min_set_size",
        "max_set_size",
        "pair_count",

        "residual_frequency_rank",
        "residual_min_set_size",
        "residual_max_set_size",
        "residual_pair_count",
        "residual_constraint_mass",
        "residual_cooccurrence_degree",
        "residual_coverage_ratio",

        "lp_value",
        "lp_certainty",
    ]

    def __init__(
        self,
        static_features,
        hyperedges,
        model_path,
        max_depth=-1,
    ):

        super().__init__()

        self.static_features = (
            static_features
        )

        self.hyperedges = hyperedges

        self.xgb_model = XGBRanker()

        self.xgb_model.load_model(
            model_path
        )

        self.max_depth = (
            max_depth
        )

        self.branch_count = 0

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

    def _selected_candidates(
        self,
        cands,
        candssol,
        npriocands,
    ):

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

            candidates.append(
                (
                    var,
                    idx,
                    float(
                        candssol[i]
                    ),
                )
            )

        return candidates

    def _compute_residual_features(
        self,
    ):

        n = len(
            self.static_features
        )

        frequency = np.zeros(
            n,
            dtype=float,
        )

        pair_count = np.zeros(
            n,
            dtype=float,
        )

        min_size = np.full(
            n,
            np.inf,
            dtype=float,
        )

        max_size = np.zeros(
            n,
            dtype=float,
        )

        constraint_mass = np.zeros(
            n,
            dtype=float,
        )

        variable_map = (
            self._build_variable_map()
        )

        active_edges = 0

        residual_vertices = np.zeros(
            n,
            dtype=bool,
        )

        residual_edges = []

        for hyperedge in self.hyperedges:

            residual_edge = []

            satisfied = False

            for v in hyperedge:

                var = variable_map.get(
                    int(v)
                )

                if var is None:

                    continue

                lb = var.getLbLocal()
                ub = var.getUbLocal()

                if lb >= 0.5:

                    satisfied = True

                    break

                if ub > 0.5:

                    residual_edge.append(
                        int(v)
                    )

            if satisfied:

                continue

            if not residual_edge:

                continue

            active_edges += 1

            size = len(
                residual_edge
            )

            inv_size = (
                1.0 / size
            )

            residual_edges.append(
                residual_edge
            )

            for v in residual_edge:

                residual_vertices[v] = True

                frequency[v] += 1

                min_size[v] = min(
                    min_size[v],
                    size,
                )

                max_size[v] = max(
                    max_size[v],
                    size,
                )

                constraint_mass[v] += (
                    inv_size
                )

                if size == 2:

                    pair_count[v] += 1

        min_size[
            frequency == 0
        ] = 0.0

        order = np.argsort(
            frequency,
            kind="stable",
        )

        frequency_rank = np.empty(
            n,
            dtype=float,
        )

        frequency_rank[
            order
        ] = np.arange(
            n,
            dtype=float,
        )

        if n > 1:

            frequency_rank /= (
                n - 1
            )

        coverage_ratio = np.zeros(
            n,
            dtype=float,
        )

        if active_edges > 0:

            mask = (
                frequency > 0
            )

            coverage_ratio[mask] = (
                frequency[mask]
                /
                active_edges
            )

        cooccurrence_degree = np.zeros(
            n,
            dtype=float,
        )

        for residual_edge in residual_edges:

            for v in residual_edge:

                total = 0.0

                for u in residual_edge:

                    if u != v:

                        total += frequency[u]

                cooccurrence_degree[v] += (
                    total
                )

        return {

            "residual_frequency_rank":
                frequency_rank,

            "residual_min_set_size":
                min_size,

            "residual_max_set_size":
                max_size,

            "residual_pair_count":
                pair_count,

            "residual_constraint_mass":
                constraint_mass,

            "residual_cooccurrence_degree":
                cooccurrence_degree,

            "residual_coverage_ratio":
                coverage_ratio,

            "residual_n":
                int(
                    np.sum(
                        residual_vertices
                    )
                ),

            "residual_m":
                int(
                    active_edges
                ),
        }

    def branchexeclp(
        self,
        allowaddcons,
    ):

        if (
            self.max_depth >= 0
            and
            self.model.getDepth()
            > self.max_depth
        ):

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
        ) = self.model.getLPBranchCands()

        if ncands == 0:

            return {
                "result":
                    SCIP_RESULT.DIDNOTRUN
            }

        candidates = (
            self._selected_candidates(
                cands,
                candssol,
                npriocands,
            )
        )

        if not candidates:

            return {
                "result":
                    SCIP_RESULT.DIDNOTRUN
            }

        dynamic_features = (
            self._compute_residual_features()
        )

        feature_rows = []

        valid_candidates = []

        for (
            var,
            idx,
            lp_value,
        ) in candidates:

            if idx not in (
                self.static_features
            ):

                continue

            values = dict(
                self.static_features[
                    idx
                ]
            )

            values[
                "residual_frequency_rank"
            ] = (
                dynamic_features[
                    "residual_frequency_rank"
                ][idx]
            )

            values[
                "residual_min_set_size"
            ] = (
                dynamic_features[
                    "residual_min_set_size"
                ][idx]
            )

            values[
                "residual_max_set_size"
            ] = (
                dynamic_features[
                    "residual_max_set_size"
                ][idx]
            )

            values[
                "residual_pair_count"
            ] = (
                dynamic_features[
                    "residual_pair_count"
                ][idx]
            )

            values[
                "residual_constraint_mass"
            ] = (
                dynamic_features[
                    "residual_constraint_mass"
                ][idx]
            )

            values[
                "residual_cooccurrence_degree"
            ] = (
                dynamic_features[
                    "residual_cooccurrence_degree"
                ][idx]
            )

            values[
                "residual_coverage_ratio"
            ] = (
                dynamic_features[
                    "residual_coverage_ratio"
                ][idx]
            )

            values[
                "lp_value"
            ] = lp_value

            values[
                "lp_certainty"
            ] = abs(
                lp_value - 0.5
            )

            feature_rows.append(
                [
                    float(
                        values[name]
                    )

                    for name in (
                        self.FEATURE_NAMES
                    )
                ]
            )

            valid_candidates.append(
                (
                    var,
                    idx,
                )
            )

        if not feature_rows:

            return {
                "result":
                    SCIP_RESULT.DIDNOTRUN
            }

        X = np.asarray(
            feature_rows,
            dtype=np.float32,
        )

        assert (
            X.shape[1] == 19
        )

        predictions = (
            self.xgb_model.predict(
                X
            )
        )

        best_position = int(
            np.argmax(
                predictions
            )
        )

        best_var = (
            valid_candidates[
                best_position
            ][0]
        )

        best_idx = (
            valid_candidates[
                best_position
            ][1]
        )

        best_prediction = float(
            predictions[
                best_position
            ]
        )

        self.model.branchVar(
            best_var
        )

        self.branch_count += 1

        if (
            self.branch_count == 1
            or
            self.branch_count % 100 == 0
        ):

            print(
                "[XGB]",
                "depth=",
                self.model.getDepth(),
                "candidates=",
                len(valid_candidates),
                "selected=",
                best_idx,
                "prediction=",
                f"{best_prediction:.6f}",
            )

        return {
            "result":
                SCIP_RESULT.BRANCHED
        }