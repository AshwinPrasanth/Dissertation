from dataclasses import dataclass

import numpy as np
import time
from features import (
    HypergraphFeatureExtractor,
    MWUAElementFeatureExtractor,
    LPFeatureExtractor,
)

@dataclass
class ElementFeatureDataset:

    X: np.ndarray

    feature_names: list

    variable_names: list


class DatasetBuilder:

    def __init__(self):

        self.hypergraph = (
            HypergraphFeatureExtractor()
        )

        self.mwua = (
            MWUAElementFeatureExtractor()
        )

        self.lp = (
            LPFeatureExtractor()
        )

    def build(
        self,
        problem,
    ) -> ElementFeatureDataset:

        # ---------------------------------
        # Compute all feature groups
        # ---------------------------------

        t = time.time()

        hyper = self.hypergraph.compute(
            problem
        )

        print(
            "hypergraph:",
            time.time() - t,
        )

        t = time.time()

        lp = self.lp.compute(
            problem
        )

        print(
            "lp:",
            time.time() - t,
        )

        t = time.time()

        mwua = self.mwua.compute(
            problem
        )

        print(
            "mwua:",
            time.time() - t,
        )

        # ---------------------------------
        # Stack into feature matrix
        # ---------------------------------

        X = np.column_stack([

            hyper.frequency,

            hyper.frequency_rank,

            hyper.coverage_ratio,

            hyper.avg_set_size,

            hyper.inverse_avg_set_size,

            hyper.min_set_size,

            hyper.max_set_size,

            hyper.set_size_variance,

            hyper.singleton_count,

            hyper.pair_count,

            mwua.x_avg,

            np.abs(
                mwua.x_avg - 0.5
            ),

            mwua.weight_min,

            mwua.weight_max,

            mwua.weight_avg,

            lp.lp_value,

            lp.lp_certainty,

        ])

        feature_names = [

            "frequency",

            "frequency_rank",

            "coverage_ratio",

            "avg_set_size",

            "inverse_avg_set_size",

            "min_set_size",

            "max_set_size",

            "set_size_variance",

            "singleton_count",

            "pair_count",

            "mwua_xavg",

            "mwua_certainty",

            "mwua_weight_min",

            "mwua_weight_max",

            "mwua_weight_avg",

            "lp_value",

            "lp_certainty",

        ]

        return ElementFeatureDataset(

            X=X,

            feature_names=feature_names,

            variable_names=(
                problem.variable_names
            ),

        )