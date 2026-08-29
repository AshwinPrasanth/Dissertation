
import numpy as np
import networkx as nx

from features import (
    CentralityFeatureExtractor,
    LubyFeatureExtractor,
    MWUAVertexFeatureExtractor,
)
from problem import build_vertex_cover_problem


class MLFeatureBuilder:

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

    def __init__(self, graph):
        self.graph = graph
        self.static = self._build_static()
        self.dynamic = {}

    def variable_index(self, var):

        name = var.name

        if name.startswith("x_"):
            return int(name.replace("x_", ""))

        if name.startswith("t_x_"):
            return int(name.replace("t_x_", ""))

        return None

    def _build_static(self):

        centrality = (
            CentralityFeatureExtractor()
            .compute(self.graph)
        )

        luby = (
            LubyFeatureExtractor()
            .compute(self.graph)
        )

        mwua = (
            MWUAVertexFeatureExtractor()
            .compute(
                build_vertex_cover_problem(
                    self.graph
                )
            )
        )

        out = {}

        for v in self.graph.nodes():

            out[v] = {
                "pagerank":
                    float(centrality.pagerank[v]),

                "mwua_xavg":
                    float(mwua.x_avg[v]),

                "mwua_weight_min":
                    float(mwua.weight_min[v]),

                "mwua_weight_max":
                    float(mwua.weight_max[v]),

                "mwua_weight_avg":
                    float(mwua.weight_avg[v]),

                "luby_frequency":
                    float(luby.frequency[v]),
            }

        return out

    def _dynamic(self, graph):

        nodes = list(graph.nodes())

        degrees = np.asarray(
            [
                graph.degree(v)
                for v in nodes
            ],
            dtype=float,
        )

        order = np.argsort(
            degrees,
            kind="stable"
        )

        ranks = np.empty(
            len(nodes),
            dtype=float
        )

        ranks[order] = np.arange(
            len(nodes),
            dtype=float
        )

        if len(nodes) > 1:
            ranks /= len(nodes)-1

        degree_rank = {
            v: float(ranks[i])
            for i, v in enumerate(nodes)
        }

        core = nx.core_number(
            self.graph
        )

        out = {}

        for v in nodes:

            nbr = list(
                graph.neighbors(v)
            )

            nbr_rank = [
                degree_rank[u]
                for u in nbr
            ]

            if nbr_rank:
                mn = min(nbr_rank)
                mx = max(nbr_rank)
                avg = sum(nbr_rank)/len(nbr_rank)
            else:
                mn = mx = avg = 0.0

            out[v] = {
                "degree_rank":
                    degree_rank[v],

                "nbr_min_rank":
                    mn,

                "nbr_max_rank":
                    mx,

                "nbr_avg_rank":
                    avg,

                "core_number":
                    float(core[v]),

                "clustering":
                    float(
                        nx.clustering(
                            self.graph,
                            v
                        )
                    ),

                "degree_centrality":
                    float(
                        graph.degree(v)
                        /
                        max(len(nodes)-1,1)
                    ),
            }

        return out
    
    def update_dynamic(self, graph):

        self.dynamic = self._dynamic(
            graph
        )

    def features(
        self,
        vertex,
        lp_value,
    ):


        f = {}

        f.update(
            self.dynamic[vertex]
        )

        f.update(
            self.static[vertex]
        )

        f["lp_value"] = float(lp_value)

        f["lp_certainty"] = abs(
            float(lp_value)-0.5
        )

        return [
            f[name]
            for name in self.FEATURE_NAMES
        ]
