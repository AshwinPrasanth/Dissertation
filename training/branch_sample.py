from dataclasses import dataclass

import numpy as np


@dataclass
class BranchSample:

    graph_name: str

    node_number: int

    depth: int

    residual_n: int

    residual_m: int

    parent_lp_obj: float

    feature_names: np.ndarray

    candidate_ids: np.ndarray

    candidate_features: np.ndarray

    lp_values: np.ndarray

    sb_down_bounds: np.ndarray

    sb_up_bounds: np.ndarray

    sb_down_gains: np.ndarray

    sb_up_gains: np.ndarray

    sb_scores: np.ndarray

    sb_down_valid: np.ndarray

    sb_up_valid: np.ndarray

    sb_down_infeasible: np.ndarray

    sb_up_infeasible: np.ndarray

    chosen_variable: int

    best_sb_score: float
