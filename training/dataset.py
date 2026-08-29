from dataclasses import dataclass

import numpy as np
import time
from features import (
    DegreeFeatureExtractor,
    CentralityFeatureExtractor,
    MWUAVertexFeatureExtractor,
    LPFeatureExtractor,
    LubyFeatureExtractor,
)


@dataclass
class VertexFeatureDataset:

    X: np.ndarray

    feature_names: list

    variable_names: list


class DatasetBuilder:

    def __init__(self):

        self.degree = (
            DegreeFeatureExtractor()
        )

        self.centrality = (
            CentralityFeatureExtractor()
        )

        self.mwua = (
            MWUAVertexFeatureExtractor()
        )

        self.lp = (
            LPFeatureExtractor()
        )

        self.luby = (
            LubyFeatureExtractor()
        )

    def build(
        self,
        problem,
    ) -> VertexFeatureDataset:

        # ---------------------------------
        # Compute all feature groups
        # ---------------------------------

        t=time.time()
        degree = self.degree.compute(
            problem.graph
        )
        print("degree:", time.time()-t)

        t=time.time()
        centrality = (
            self.centrality.compute(
                problem.graph
            )
        )
        print("centrality:", time.time()-t)
        
        t=time.time()
        lp = self.lp.compute(
            problem
        )
        print("lp:", time.time()-t)
        
        t=time.time()
        luby = self.luby.compute(problem.graph)
        print("luby:", time.time()-t)

        t=time.time()
        mwua = self.mwua.compute(
            problem
        )
        print("mwua:", time.time()-t)
        

        # ---------------------------------
        # Stack into feature matrix
        # ---------------------------------

        X = np.column_stack([

            # Degree features

            degree.degree_rank,
            degree.nbr_min_rank,
            degree.nbr_max_rank,
            degree.nbr_avg_rank,

            # Centrality features

            centrality.pagerank,
            centrality.core_number,
            centrality.clustering,
            centrality.degree_centrality,

            # MWUA features

            mwua.x_avg,
            mwua.weight_min,
            mwua.weight_max,
            mwua.weight_avg,

            # LP features

            lp.lp_value,
            lp.lp_certainty,

            # Luby feature

            luby.frequency,

        ])

        feature_names = [

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

        return VertexFeatureDataset(

            X=X,

            feature_names=feature_names,

            variable_names=(
                problem.variable_names
            ),

        )