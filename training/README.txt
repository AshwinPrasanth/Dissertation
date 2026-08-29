LTB DATA COLLECTOR: 6 STATIC GLOBAL + 9 DYNAMIC LOCAL = 15 FEATURES

STATIC GLOBAL, computed once before SCIP:
1. pagerank
2. mwua_xavg
3. mwua_weight_min
4. mwua_weight_max
5. mwua_weight_avg
6. luby_frequency

DYNAMIC LOCAL, recomputed at each selected SCIP B&B node:
1. degree_rank
2. nbr_min_rank
3. nbr_max_rank
4. nbr_avg_rank
5. core_number
6. clustering
7. degree_centrality
8. lp_value
9. lp_certainty

The seven structural dynamic features are computed on the current residual graph induced by locally unfixed SCIP variables.
The two LP features come from the current SCIP LP branching candidate values.
Strong branching is used only for labels and branch choice at the first MAX_SB_NODES eligible nodes.

IMPORTANT:
Run a one-graph pilot first:
GRAPH_SIZES = [50]
TARGET_DEGREES = [10]
SEEDS = [42]
MAX_SB_NODES = 10
CANDIDATE_LIMIT = None

Copy branching.py over the existing anytime/branching.py.
Place solver_training.py and generate_ltb_training_data.py in anytime/.
