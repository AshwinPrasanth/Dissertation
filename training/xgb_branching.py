import numpy as np
import networkx as nx

from xgboost import XGBRanker

from pyscipopt import (
    Branchrule,
    SCIP_RESULT,
)


class SCIPXGBBranchRule(Branchrule):

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
        model_path,
        max_depth=5
    ):

        super().__init__()

        self.static_features = static_features

        self.graph = graph

        self.rank_model = XGBRanker()

        self.rank_model.load_model(
            model_path
        )
        self.max_depth = max_depth

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



    @staticmethod
    def _is_residual_variable(
        var,
    ):

        return (
            var.getLbLocal() < 0.5
            and
            var.getUbLocal() > 0.5
        )



    def _build_residual_graph(
        self,
        variable_map,
    ):

        vertices = [
            idx
            for idx,var in variable_map.items()
            if self._is_residual_variable(var)
        ]

        return self.graph.subgraph(
            vertices
        ).copy()



    @staticmethod
    def _dynamic_structural_features(
        residual_graph,
    ):

        nodes = list(
            residual_graph.nodes()
        )

        if not nodes:

            return {}


        degrees = np.asarray(
            [
                residual_graph.degree(v)
                for v in nodes
            ],
            dtype=float,
        )


        order = np.argsort(
            degrees,
            kind="stable",
        )


        ranks = np.empty(
            len(nodes),
            dtype=float,
        )


        ranks[order] = np.arange(
            len(nodes),
            dtype=float,
        )


        if len(nodes) > 1:

            ranks /= (
                len(nodes)-1
            )


        rank_by_vertex = {

            v: float(ranks[i])

            for i,v in enumerate(nodes)

        }


        if residual_graph.number_of_edges() > 0:

            core_number = nx.core_number(
                residual_graph
            )

        else:

            core_number = {
                v:0
                for v in nodes
            }


        clustering = nx.clustering(
            residual_graph
        )


        degree_centrality = nx.degree_centrality(
            residual_graph
        )


        features = {}


        for v in nodes:

            neighbours = list(
                residual_graph.neighbors(v)
            )


            if neighbours:

                neighbour_ranks = np.asarray(
                    [
                        rank_by_vertex[u]
                        for u in neighbours
                    ],
                    dtype=float,
                )


                nbr_min_rank = float(
                    neighbour_ranks.min()
                )

                nbr_max_rank = float(
                    neighbour_ranks.max()
                )

                nbr_avg_rank = float(
                    neighbour_ranks.mean()
                )

            else:

                nbr_min_rank = 0.0
                nbr_max_rank = 0.0
                nbr_avg_rank = 0.0



            features[v] = {

                "degree_rank":
                    rank_by_vertex[v],

                "nbr_min_rank":
                    nbr_min_rank,

                "nbr_max_rank":
                    nbr_max_rank,

                "nbr_avg_rank":
                    nbr_avg_rank,

                "core_number":
                    float(core_number[v]),

                "clustering":
                    float(clustering[v]),

                "degree_centrality":
                    float(degree_centrality[v]),

            }


        return features



    def _get_candidates(
        self,
    ):

        (
            cands,
            candssol,
            candsscore,
            ncands,
            npriocands,
            nfracimplvars,
        ) = self.model.getLPBranchCands()


        candidates = []


        for i in range(npriocands):

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
                    float(candssol[i]),
                )
            )


        return candidates



    def branchexeclp(
        self,
        allowaddcons,
    ):
        if self.model.getDepth() > self.max_depth:

            return {
                "result":
                    SCIP_RESULT.DIDNOTRUN
            }
            
            


        candidates = self._get_candidates()


        if not candidates:

            return {
                "result":
                    SCIP_RESULT.DIDNOTRUN
            }


        variable_map = self._build_variable_map()


        residual_graph = self._build_residual_graph(
            variable_map
        )


        dynamic_features = (
            self._dynamic_structural_features(
                residual_graph
            )
        )


        feature_rows = []

        valid_candidates = []


        for (
            var,
            idx,
            lp_value,
        ) in candidates:


            if idx not in dynamic_features:

                continue


            values = {}


            values.update(
                self.static_features[idx]
            )


            values.update(
                dynamic_features[idx]
            )


            values["lp_value"] = (
                lp_value
            )


            values["lp_certainty"] = abs(
                lp_value - 0.5
            )


            feature_rows.append(
                [
                    float(
                        values[name]
                    )

                    for name in self.FEATURE_NAMES
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

        assert len(feature_rows[0]) == 15

        scores = self.rank_model.predict(
            np.asarray(
                feature_rows,
                dtype=float,
            )
        )


        best_position = int(
            np.argmax(scores)
        )


        best_var, best_idx = (
            valid_candidates[
                best_position
            ]
        )


        self.model.branchVar(
            best_var
        )


        self.branch_count += 1
        
        #print(feature_rows[0])

        print(
            "[XGB]",
            "depth=",
            self.model.getDepth(),
            "vertex=",
            best_idx,
            "score=",
            f"{scores[best_position]:.6f}",
        )



        return {
            "result":
                SCIP_RESULT.BRANCHED
        }