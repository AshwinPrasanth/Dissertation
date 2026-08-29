from dataclasses import dataclass

import networkx as nx

import numpy as np

from lp import solve_lp_relaxation

from mwua import MWUAFeatureExtractor


@dataclass
class DegreeFeatures:

    degree_rank: np.ndarray

    nbr_min_rank: np.ndarray

    nbr_max_rank: np.ndarray

    nbr_avg_rank: np.ndarray


class DegreeFeatureExtractor:

    def compute(
        self,
        G,
    ) -> DegreeFeatures:

        n = len(G)

        degrees = np.array(
            [
                G.degree(v)
                for v in range(n)
            ],
            dtype=float,
        )

        order = np.argsort(
            degrees
        )

        ranks = np.empty(
            n,
            dtype=float,
        )

        ranks[order] = np.arange(
            n,
            dtype=float,
        )

        if n > 1:

            ranks /= (
                n - 1
            )

        nbr_min_rank = np.zeros(
            n
        )

        nbr_max_rank = np.zeros(
            n
        )

        nbr_avg_rank = np.zeros(
            n
        )

        for v in range(n):

            neighbors = list(
                G.neighbors(v)
            )

            if not neighbors:
                continue

            neighbor_ranks = ranks[
                neighbors
            ]

            nbr_min_rank[v] = (
                neighbor_ranks.min()
            )

            nbr_max_rank[v] = (
                neighbor_ranks.max()
            )

            nbr_avg_rank[v] = (
                neighbor_ranks.mean()
            )

        return DegreeFeatures(
            degree_rank=ranks,
            nbr_min_rank=nbr_min_rank,
            nbr_max_rank=nbr_max_rank,
            nbr_avg_rank=nbr_avg_rank,
        )


@dataclass
class CentralityFeatures:

    pagerank: np.ndarray

    core_number: np.ndarray

    clustering: np.ndarray

    degree_centrality: np.ndarray


class CentralityFeatureExtractor:

    def compute(
        self,
        G,
    ) -> CentralityFeatures:

        n = len(G)

        pagerank_dict = (
            nx.pagerank(G)
        )

        core_dict = (
            nx.core_number(G)
        )

        clustering_dict = (
            nx.clustering(G)
        )

        degree_centrality_dict = (
            nx.degree_centrality(G)
        )

        pagerank = np.zeros(
            n
        )

        core_number = np.zeros(
            n
        )

        clustering = np.zeros(
            n
        )

        degree_centrality = np.zeros(
            n
        )

        for v in G.nodes():

            pagerank[v] = (
                pagerank_dict[v]
            )

            core_number[v] = (
                core_dict[v]
            )

            clustering[v] = (
                clustering_dict[v]
            )

            degree_centrality[v] = (
                degree_centrality_dict[v]
            )

        return CentralityFeatures(
            pagerank=pagerank,
            core_number=core_number,
            clustering=clustering,
            degree_centrality=degree_centrality,
        )


@dataclass
class MWUAVertexFeatures:

    x_avg: np.ndarray

    weight_min: np.ndarray

    weight_max: np.ndarray

    weight_avg: np.ndarray


class MWUAVertexFeatureExtractor:

    def compute(
        self,
        problem,
    ) -> MWUAVertexFeatures:

        mwua = (
            MWUAFeatureExtractor()
        )

        result = mwua.compute(
            problem
        )

        G = problem.graph

        n = len(G)

        final_weights = (
            result.final_weights.copy()
        )

        w_min = np.min(
            final_weights
        )

        w_max = np.max(
            final_weights
        )

        if (
            w_max - w_min
            <= 0.0
        ):

            final_weights = (
                np.zeros_like(
                    final_weights
                )
            )

        else:

            final_weights = (

                final_weights
                - w_min

            ) / (

                w_max
                - w_min

            )

        weight_sum = np.zeros(
            n
        )

        weight_count = np.zeros(
            n
        )

        weight_min = np.full(
            n,
            np.inf,
        )

        weight_max = np.full(
            n,
            -np.inf,
        )

        for edge_idx, (
            u,
            v,
        ) in enumerate(
            G.edges()
        ):

            w = final_weights[
                edge_idx
            ]

            weight_sum[u] += w

            weight_count[u] += 1

            weight_min[u] = min(
                weight_min[u],
                w,
            )

            weight_max[u] = max(
                weight_max[u],
                w,
            )

            weight_sum[v] += w

            weight_count[v] += 1

            weight_min[v] = min(
                weight_min[v],
                w,
            )

            weight_max[v] = max(
                weight_max[v],
                w,
            )

        weight_avg = np.zeros(
            n
        )

        has_edges = (
            weight_count > 0
        )

        weight_avg[
            has_edges
        ] = (

            weight_sum[
                has_edges
            ]

            / weight_count[
                has_edges
            ]

        )

        weight_min[
            ~has_edges
        ] = 0.0

        weight_max[
            ~has_edges
        ] = 0.0

        cover_xavg = (
            result.x_avg
        )

        mis_xavg = (
            1.0
            - cover_xavg
        )

        return MWUAVertexFeatures(
            x_avg=mis_xavg,
            weight_min=weight_min,
            weight_max=weight_max,
            weight_avg=weight_avg,
        )

    @staticmethod
    def compute_mwua_score(
        dataset,
    ):

        names = (
            dataset.feature_names
        )

        xavg_idx = names.index(
            "mwua_xavg"
        )

        mis_xavg = dataset.X[
            :,
            xavg_idx
        ]

        certainty = np.abs(
            mis_xavg
            - 0.5
        )

        return certainty

    @staticmethod
    def compute_mwua_prediction(
        dataset,
    ):

        names = (
            dataset.feature_names
        )

        xavg_idx = names.index(
            "mwua_xavg"
        )

        mis_xavg = dataset.X[
            :,
            xavg_idx
        ]

        prediction = (
            mis_xavg
            > 0.5
        ).astype(
            np.int8
        )

        return prediction


@dataclass
class LPFeatures:

    lp_value: np.ndarray

    lp_certainty: np.ndarray


class LPFeatureExtractor:

    def compute(
        self,
        problem,
    ) -> LPFeatures:

        result = (
            solve_lp_relaxation(
                problem
            )
        )

        lp_value = (
            result.x
        )

        lp_certainty = np.abs(
            lp_value
            - 0.5
        )

        return LPFeatures(
            lp_value=lp_value,
            lp_certainty=lp_certainty,
        )


@dataclass
class LubyFeatures:

    frequency: np.ndarray


class LubyFeatureExtractor:

    def __init__(
        self,
        runs=100,
    ):

        self.runs = runs

    def compute(
        self,
        G,
    ) -> LubyFeatures:

        n = len(G)

        counts = np.zeros(
            n
        )

        rng = (
            np.random.default_rng(
                0
            )
        )

        for _ in range(
            self.runs
        ):

            active = set(
                G.nodes()
            )

            selected = []

            while active:

                priorities = {
                    v: rng.random()
                    for v in active
                }

                chosen = []

                for v in active:

                    if all(

                        priorities[v]
                        > priorities[u]

                        for u in G.neighbors(v)

                        if u in active

                    ):

                        chosen.append(
                            v
                        )

                if not chosen:

                    break

                selected.extend(
                    chosen
                )

                remove = set(
                    chosen
                )

                for v in chosen:

                    remove.update(
                        G.neighbors(v)
                    )

                active -= remove

            counts[
                selected
            ] += 1

        frequency = (
            counts
            / self.runs
        )

        return LubyFeatures(
            frequency=frequency
        )