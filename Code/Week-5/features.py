# features.py- a feature extractor module
'''features.py does the following:
1. Defines various feature extractors for vertices in the graph,
which can be used for branching decisions in the branch-and-bound solver.
2. The features include degree-based features, centrality-based features, 
MWUA-based features, LP relaxation-based features, and Luby's algorithm-based features.
3. Influences building the dataset.py'''


from dataclasses import dataclass
from unittest import result
import networkx as nx
import numpy as np
from mwua import MWUAFeatureExtractor # for MWUA-based features
from lp import solve_lp_relaxation

####### degree features: degree rank, neighbor degree ranks (min/max/avg) #######
''' The main purpose of the degree features is to capture the local structure of the graph around each vertex.
The degree rank gives a normalized measure of how connected a vertex is compared to others, 
which can be a strong indicator of its importance in the independent set. 
The neighbor degree ranks (min/max/avg) provide additional context about the local neighborhood of the vertex, 
identify vertices that are in dense regions of the graph (high neighbor degree ranks) vs those in sparser regions (low neighbor degree ranks). 
'''

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
        # Compute degree-based features for each vertex in the graph G.

        n = len(G)

        degrees = np.array([
            G.degree(v)
            for v in G.nodes()
        ])

        unique_deg = np.unique(
            degrees
        )

        rank_map = {}

        if len(unique_deg) == 1:

            rank_map[
                unique_deg[0]
            ] = 1.0

        else:

            for i, d in enumerate( 
                sorted(unique_deg)
            ): # assign a rank in [0,1] based on degree, with higher degree getting higher rank

                rank_map[d] = (
                    i
                    /
                    (
                        len(unique_deg)
                        - 1
                    )
                )

        degree_rank = np.array([
            rank_map[d]
            for d in degrees
        ])
        
        nbr_min_rank = np.zeros(n)
        nbr_max_rank = np.zeros(n)
        nbr_avg_rank = np.zeros(n)

        for v in G.nodes(): # for each vertex, compute min/max/avg degree rank of its neighbors

            nbrs = list(
                G.neighbors(v)
            )

            if len(nbrs) == 0:

                nbr_min_rank[v] = 0
                nbr_max_rank[v] = 0
                nbr_avg_rank[v] = 0

                continue

            nbr_ranks = degree_rank[
                nbrs
            ]

            nbr_min_rank[v] = np.min(
                nbr_ranks
            )

            nbr_max_rank[v] = np.max(
                nbr_ranks
            )

            nbr_avg_rank[v] = np.mean(
                nbr_ranks
            )

        return DegreeFeatures(
            degree_rank=degree_rank,
            nbr_min_rank=nbr_min_rank,
            nbr_max_rank=nbr_max_rank,
            nbr_avg_rank=nbr_avg_rank,
        ) 


######## centrality features: pagerank, core number, clustering coefficient, degree centrality #######
''' The centrality features aim to capture more global structural properties of the graph that
may not be fully reflected in the degree features.'''


@dataclass
class CentralityFeatures:
    # Centrality-based features for each vertex in the graph.
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

        pagerank_dict = nx.pagerank(G) # compute PageRank for each vertex

        core_dict = nx.core_number(G) # compute core number for each vertex

        clustering_dict = nx.clustering(G) # compute clustering coefficient for each vertex

        degree_centrality_dict = (
            nx.degree_centrality(G)
        ) # compute degree centrality for each vertex

        pagerank = np.zeros(n)

        core_number = np.zeros(n)

        clustering = np.zeros(n)

        degree_centrality = np.zeros(n)

        for v in G.nodes(): # populate the feature arrays based on the computed centrality measures

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

######## MWUA-based features: min/max/avg final edge weight for edges incident to each vertex, avg x value from greedy fractional solution #######
''' The MWUA-based features are designed to capture information from the MWUA algorithm, 
which is a powerful heuristic for approximating the maximum independent set.'''


@dataclass
class MWUAVertexFeatures:
    # MWUA-based features for each vertex, derived from the final weights on edges after running MWUA.
    x_avg: np.ndarray

    weight_min: np.ndarray

    weight_max: np.ndarray

    weight_avg: np.ndarray

class MWUAVertexFeatureExtractor:

    def compute(
        self,
        problem,
    ) -> MWUAVertexFeatures:

        mwua = MWUAFeatureExtractor()

        result = mwua.compute(
            problem
        ) # run MWUA on the problem to get final edge weights and average x values

        G = problem.graph # get the graph from the problem instance

        n = len(G)

        weight_min = np.zeros(n)

        weight_max = np.zeros(n)

        weight_avg = np.zeros(n)

        final_weights = (result.final_weights)
        
        w_min = np.min(result.final_weights)
        w_max = np.max(result.final_weights)

        if ( w_max - w_min < 1e-12): # if all weights are the same, then we can't normalize, so just set them to 0.5
            final_weights = np.zeros_like(result.final_weights)

        else:
            final_weights = (result.final_weights - w_min) / (w_max - w_min) # normalize final weights to [0,1]
        
        for v in G.nodes():

            incident_weights = []
            for edge_idx in range(len(problem.A_ub)): # for each edge, check if it's incident to vertex v and if so, add its final weight to the list of incident weights

                row = problem.A_ub[edge_idx]
                vars_in_edge = np.where(np.abs(row) > 0)[0]
                if v in vars_in_edge:
                    incident_weights.append(final_weights[edge_idx])

            if len(
                incident_weights
            ) == 0:

                continue

            incident_weights = np.array(
                incident_weights
            )

            weight_min[v] = np.min(
                incident_weights
            )

            weight_max[v] = np.max(
                incident_weights
            )

            weight_avg[v] = np.mean(
                incident_weights
            )

        return MWUAVertexFeatures(
            x_avg=result.x_avg,
            weight_min=weight_min,
            weight_max=weight_max,
            weight_avg=weight_avg,
        )
    
    
####### Constraint programming / LP relaxation-based features: LP value for each vertex, certainty (distance from 0.5) #######     
''' The LP relaxation-based features are intended to capture information from 
the linear programming relaxation of the maximum independent set problem.
The features computed are: 1. LP value and 2. Certainty, which is the distance of the LP value from 0.5. 
The intuition behind these features is that the LP relaxation can provide a fractional solution that tells us 
how likely each vertex is to be in the independent set.'''   


@dataclass
class LPFeatures:
    # LP relaxation-based features for each vertex, derived from the LP solution of the problem.

    lp_value: np.ndarray

    lp_certainty: np.ndarray

class LPFeatureExtractor:

    def compute(
        self,
        problem,
    ) -> LPFeatures:

        lp = solve_lp_relaxation(
            problem
        )

        lp_value = lp.x.copy()

        lp_certainty = np.abs(
            lp_value - 0.5
        )

        return LPFeatures(
            lp_value=lp_value,
            lp_certainty=lp_certainty,
        )
        
        
####### Luby's algorithm-based features: frequency of being chosen in the independent set across multiple runs of Luby's algorithm #######
''' Luby's algorithm is used to find a maximal independent set in a graph. By running Luby's algorithm multiple times with randomization, 
we can compute the frequency with which each vertex is included in the independent set across these runs.'''

@dataclass
class LubyFeatures:
    # Luby's algorithm-based features for each vertex, derived from the frequency of being chosen in the independent set across multiple runs of Luby's algorithm.

    frequency: np.ndarray
    
class LubyFeatureExtractor:

    def __init__(
        self,
        runs: int = 100,
        seed: int = 42,
    ):
        self.runs = runs
        self.seed = seed

    def compute(
        self,
        G,
    ) -> LubyFeatures: 

        rng = np.random.default_rng(
            self.seed
        )

        n = len(G)

        counts = np.zeros(n)

        for _ in range(self.runs): 

            active = set(
                G.nodes()
            )

            indep_set = []

            while active:

                chosen = []

                for v in active: 

                    r_v = rng.random()

                    is_local_min = True

                    for nbr in G.neighbors(v):

                        if nbr not in active:
                            continue

                        r_nbr = rng.random()

                        if r_nbr < r_v:

                            is_local_min = False
                            break

                    if is_local_min:

                        chosen.append(v)

                indep_set.extend(
                    chosen
                )

                removed = set()

                for v in chosen:

                    removed.add(v)

                    removed.update(
                        G.neighbors(v)
                    )

                active -= removed

            for v in indep_set:

                counts[v] += 1

        frequency = (
            counts / self.runs
        )

        return LubyFeatures(
            frequency=frequency
        )