from dataclasses import dataclass

import networkx as nx

import numpy as np

from lp import solve_lp_relaxation

from mwua import MWUAFeatureExtractor

@dataclass
@dataclass
class HypergraphFeatures:

    frequency_rank: np.ndarray

    min_set_size: np.ndarray

    max_set_size: np.ndarray

    pair_count: np.ndarray

    bipartite_core_number: np.ndarray

    bipartite_pagerank: np.ndarray


class HypergraphFeatureExtractor:

    def compute(
        self,
        problem,
    ):

        n = problem.num_variables

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
        )

        max_size = np.zeros(
            n,
            dtype=float,
        )

        for hyperedge in problem.hyperedges:

            size = len(
                hyperedge
            )

            for v in hyperedge:

                frequency[v] += 1

                min_size[v] = min(
                    min_size[v],
                    size,
                )

                max_size[v] = max(
                    max_size[v],
                    size,
                )

                if size == 2:

                    pair_count[v] += 1

        mask = (
            frequency > 0
        )

        min_size[
            ~mask
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

        incidence = nx.Graph()

        variable_nodes = [
            ("v", v)
            for v in range(n)
        ]

        hyperedge_nodes = [
            ("e", i)
            for i in range(
                len(problem.hyperedges)
            )
        ]

        incidence.add_nodes_from(
            variable_nodes,
            bipartite=0,
        )

        incidence.add_nodes_from(
            hyperedge_nodes,
            bipartite=1,
        )

        for edge_idx, hyperedge in enumerate(
            problem.hyperedges
        ):

            edge_node = (
                "e",
                edge_idx,
            )

            for v in hyperedge:

                incidence.add_edge(
                    ("v", v),
                    edge_node,
                )

        core_numbers = nx.core_number(
            incidence
        )

        pagerank = nx.pagerank(
            incidence,
            alpha=0.85,
            max_iter=200,
        )

        bipartite_core_number = np.zeros(
            n,
            dtype=float,
        )

        bipartite_pagerank = np.zeros(
            n,
            dtype=float,
        )

        for v in range(n):

            node = (
                "v",
                v,
            )

            bipartite_core_number[v] = (
                float(
                    core_numbers.get(
                        node,
                        0,
                    )
                )
            )

            bipartite_pagerank[v] = (
                float(
                    pagerank.get(
                        node,
                        0.0,
                    )
                )
            )

        return HypergraphFeatures(

            frequency_rank=frequency_rank,

            min_set_size=min_size,

            max_set_size=max_size,

            pair_count=pair_count,

            bipartite_core_number=(
                bipartite_core_number
            ),

            bipartite_pagerank=(
                bipartite_pagerank
            ),
        )
        

@dataclass
class MWUAElementFeatures:

    x_avg: np.ndarray

    weight_min: np.ndarray

    weight_max: np.ndarray

    weight_avg: np.ndarray


class MWUAElementFeatureExtractor:

    def compute(
        self,
        problem,
    ) -> MWUAElementFeatures:

        mwua = MWUAFeatureExtractor()

        result = mwua.compute(problem)

        n = problem.num_variables

        final_weights = result.final_weights.copy()

        w_min = np.min(final_weights)

        w_max = np.max(final_weights)

        if w_max - w_min <= 0.0:

            final_weights = np.zeros_like(
                final_weights
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
            n,
            dtype=float,
        )

        weight_count = np.zeros(
            n,
            dtype=float,
        )

        weight_min = np.full(
            n,
            np.inf,
        )

        weight_max = np.full(
            n,
            -np.inf,
        )

        for edge_idx, hyperedge in enumerate(
            problem.hyperedges
        ):

            w = final_weights[
                edge_idx
            ]

            for v in hyperedge:

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
            n,
            dtype=float,
        )

        has_incidence = (
            weight_count > 0
        )

        weight_avg[
            has_incidence
        ] = (

            weight_sum[
                has_incidence
            ]

            /

            weight_count[
                has_incidence
            ]

        )

        weight_min[
            ~has_incidence
        ] = 0.0

        weight_max[
            ~has_incidence
        ] = 0.0

        return MWUAElementFeatures(

            x_avg=result.x_avg,

            weight_min=weight_min,

            weight_max=weight_max,

            weight_avg=weight_avg,

        )

    @staticmethod
    def compute_mwua_score(
        dataset,
    ):

        names = dataset.feature_names

        xavg_idx = names.index(
            "mwua_xavg"
        )

        x_avg = dataset.X[
            :,
            xavg_idx,
        ]

        certainty = np.abs(
            x_avg - 0.5
        )

        return certainty

    @staticmethod
    def compute_mwua_prediction(
        dataset,
    ):

        names = dataset.feature_names

        xavg_idx = names.index(
            "mwua_xavg"
        )

        x_avg = dataset.X[
            :,
            xavg_idx,
        ]

        prediction = (
            x_avg >= 0.5
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

