import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from problem import (build_vertex_cover_problem, build_mis_problem)
from solver import BranchAndBoundSolver
from mwua import MWUAFeatureExtractor
from lp import solve_lp_relaxation
from branching import (MostFractionalBranching,MWUABranching,DegreeBranching)
from features import DegreeFeatureExtractor, CentralityFeatureExtractor, MWUAVertexFeatureExtractor, LPFeatureExtractor, LubyFeatureExtractor
from dataset import DatasetBuilder


########## phase 1 early experiments ##########

def run_demo():

    G = nx.erdos_renyi_graph(
        n=30,
        p=0.2,
        seed=42,
    )

    problem = build_vertex_cover_problem(G)

    solver = BranchAndBoundSolver()

    result = solver.solve(problem)

    print("Objective:", result.objective)
    print("Nodes:", result.nodes_explored)
    print("LP Solves:", result.lp_solves)

def run_demo2():

    G = nx.erdos_renyi_graph(
        n=30,
        p=0.2,
        seed=42,
    )

    problem = build_vertex_cover_problem(G)

    mwua = MWUAFeatureExtractor()

    result = mwua.compute(problem)

    print("\nMWUA x_avg (first 10)")
    print(np.round(result.x_avg[:10], 3))

    print("\nMWUA certainty (first 10)")
    print(np.round(result.certainty[:10], 3))

    degrees = np.array(
        [G.degree(v) for v in G.nodes()]
    )

    top_degree = np.argsort(
        -degrees
    )[:10]

    top_mwua = np.argsort(
        -result.certainty
    )[:10]

    print("\nTop Degree")
    print(top_degree)

    print("\nTop MWUA")
    print(top_mwua)

    overlap = len(
        set(top_degree)
        &
        set(top_mwua)
    )

    print(
        f"\nTop-10 overlap: {overlap}/10"
    )
    print(
    "Unique x_avg values:",
    len(np.unique(
        np.round(result.x_avg, 3)
    ))
)

    #print(result.x_avg[:10])
    #print(result.certainty[:10])
    

# Sanity check: high degree vertices should have higher x_avg and certainty
'''G = nx.erdos_renyi_graph(
        n=30,
        p=0.2,
        seed=42,
    )
degrees = dict(G.degree())
mwua = MWUAFeatureExtractor()
problem = build_vertex_cover_problem(G)
result = mwua.compute(problem)
for i in range(10):
    print(
        f"v={i:2d}",
        f"deg={degrees[i]:2d}",
        f"x={result.x_avg[i]:.3f}",
        f"cert={result.certainty[i]:.3f}",
    )'''
    
def plot_mwua(G, result):
    
    degrees = [G.degree(v) for v in G.nodes()]
    cert = result.certainty
    
    # compute correlation
    corr = np.corrcoef(
    degrees,
    result.certainty)[0,1]
    print("Correlation:", corr)

    # visualize mwau certainty vs degree [branching]
    plt.scatter(degrees, cert)

    plt.xlabel("Degree")
    plt.ylabel("MWUA Certainty")

    plt.title("Degree vs MWUA Certainty")

    plt.show()
    

#### early phase experiments comparing LP branching vs MWUA branching on random graphs single and multi runs ####

def compare_lp_vs_mwua():
# Compare LP branching vs MWUA branching on one random graph(erdos_renyi_graph)
    G = nx.erdos_renyi_graph(
        n=30,
        p=0.2,
        seed=42,
    )

    problem = build_vertex_cover_problem(G)

    # -----------------------
    # MWUA snapshot
    # -----------------------

    mwua = MWUAFeatureExtractor()

    mwua_result = mwua.compute(problem)

    # -----------------------
    # LP branching
    # -----------------------

    lp_solver = BranchAndBoundSolver(
        MostFractionalBranching()
    )

    lp_result = lp_solver.solve(problem)

    # -----------------------
    # MWUA branching
    # -----------------------

    mwua_solver = BranchAndBoundSolver(
        MWUABranching(
            mwua_result.certainty
        )
    )

    mwua_result_solver = mwua_solver.solve(problem)

    print("\nLP Branching")
    print("Objective:", lp_result.objective)
    print("Nodes:", lp_result.nodes_explored)

    print("\nMWUA Branching")
    print("Objective:", mwua_result_solver.objective)
    print("Nodes:", mwua_result_solver.nodes_explored)
    

def compare_lp_vs_mwua_many():
# Compare multiple strategies branching on multiple random graphs(erdos_renyi_graph)
    lp_nodes = []
    mwua_nodes = []
    degree_nodes = []
    results = []
    n1=p1=0

    for seed in range(20):

        G = nx.erdos_renyi_graph(
            n=50,
            p=0.2,
            seed=seed,
        )
        n1,p1=len(G),nx.density(G)

        problem = build_mis_problem(G)
        
        lp = solve_lp_relaxation(problem)

        mwua = MWUAFeatureExtractor()
        mwua_result = mwua.compute(problem)
        
        #print("Unique MWUA:",len(np.unique(np.round(mwua_result.x_avg,4))))
        #print(np.round(mwua_result.x_avg[:10],4))
        

        lp_solver = BranchAndBoundSolver(
            MostFractionalBranching()
        )

        lp_result = lp_solver.solve(problem)

        mwua_solver = BranchAndBoundSolver(
            MWUABranching(
                mwua_result.certainty
            )
        )
        mwua_result_solver = mwua_solver.solve(problem)
        
        degrees = np.array([G.degree(v) for v in G.nodes()])
        degree_solver = BranchAndBoundSolver(
            DegreeBranching(degrees)
        )    
        degree_result = degree_solver.solve(problem)
        
        lp_nodes.append(
            lp_result.nodes_explored
        )

        mwua_nodes.append(
            mwua_result_solver.nodes_explored
        )
        
        degree_nodes.append(
    degree_result.nodes_explored)
        
        

    print(
        "LP Avg:",
        np.mean(lp_nodes)
    )

    print(
        "MWUA Avg:",
        np.mean(mwua_nodes)
    )
    
    print(
    "Degree Avg:",
    np.mean(degree_nodes)
    )
    
    print(
    "Unique LP values:",
    len(np.unique(
        np.round(lp.x, 4)
    ))
)

    print(
    "Unique MWUA values:",
    len(np.unique(
        np.round(
            mwua_result.x_avg,
            4
        )
    ))
)
    
    results.append([n1,p1,np.mean(lp_nodes),np.mean(mwua_nodes)]) 
    top_degree = np.argsort(-degrees)[:10]
    top_mwua = np.argsort(
    -mwua_result.certainty
)[:10]
    print(top_degree)
    print(top_mwua)
    
##### testing MIS solver  #####

def test_mis():

    G = nx.path_graph(4)

    problem = build_mis_problem(G)

    solver = BranchAndBoundSolver()

    result = solver.solve(problem)

    print(
        "objective:",
        result.objective
    )
    
    print(
        "solution:",
        result.solution
    )
    
############ feature tests (from features.py) ############
'''the following tests are for the feature extractors, which compute various features for each vertex based on degree, centrality, 
MWUA weights, LP relaxation, and Luby's algorithm. 
These features can be used for branching decisions in the branch-and-bound solver.
The tests are used as an analyse the feature extractors and understand the distribution of the features across vertices in random graphs.
Goal: To gain insights for building dataset.py for training ML models to predict good branching decisions.'''
    
def test_features():
    # Test degree-based features on a random graph- assigns scores based on degree and neighbor degrees

    G = nx.erdos_renyi_graph(
        20,
        0.2,
        seed=0,
    )

    features = (
        DegreeFeatureExtractor()
        .compute(G)
    )

    print(
        features.degree_rank[:10]
    )

    print(
        features.nbr_avg_rank[:10]
    )
    
def test_centrality():

    G = nx.erdos_renyi_graph(
        n=20,
        p=0.2,
        seed=0,
    )

    result = (
        CentralityFeatureExtractor()
        .compute(G)
    )

    print(
        "PageRank"
    )
    print(
        result.pagerank[:10]
    )

    print(
        "\nCore Number"
    )
    print(
        result.core_number[:10]
    )

    print(
        "\nClustering"
    )
    print(
        result.clustering[:10]
    )

    print(
        "\nDegree Centrality"
    )
    print(
        result.degree_centrality[:10]
    )
    
def test_dataset():

    G = nx.erdos_renyi_graph(
        n=30,
        p=0.2,
        seed=0,
    )

    df = (
        DatasetBuilder()
        .build_from_graph(G)
    )

    print(df.head())

    print()

    print(df["label"].value_counts())
    
def test_mwua_features():

    G = nx.erdos_renyi_graph(
        30,
        0.2,
        seed=0,
    )

    problem = build_mis_problem(G)

    result = (
        MWUAVertexFeatureExtractor()
        .compute(problem)
    )
    
    print("x_avg")
    print(result.x_avg[:10])

    print("\nweight_min")
    print(result.weight_min[:10])

    print("\nweight_max")
    print(result.weight_max[:10])

    print("\nweight_avg")
    print(result.weight_avg[:10])
    
    mwua = MWUAFeatureExtractor() 
    raw_result = mwua.compute(problem)
    print(
    "Unique final weights:"
)
    print(
        np.unique(
            np.round(
                raw_result.final_weights,
                4,
            )
        )
    )

    print(
        "Max final weight:",
        raw_result.final_weights.max()
    )

    print(
        "Argmax edge:",
        np.argmax(
            raw_result.final_weights
        )
    )
    

    
def test_lp_features():

    G = nx.erdos_renyi_graph(
        n=50,
        p=0.2,
        seed=0,
    )

    problem = build_mis_problem(G)

    result = (
        LPFeatureExtractor()
        .compute(problem)
    )

    print(
        "LP value"
    )
    print(
        result.lp_value[:10]
    )

    print(
        "\nLP certainty"
    )
    print(
        result.lp_certainty[:10]
    )

    print(
        "\nUnique LP values:"
    )

    print(
        len(
            np.unique(
                np.round(
                    result.lp_value,
                    4
                )
            )
        )
    )

def test_luby():

    G = nx.erdos_renyi_graph(
        30,
        0.2,
        seed=0,
    )

    result = (
        LubyFeatureExtractor(
            runs=100
        )
        .compute(G)
    )

    print(
        result.frequency[:10]
    )

    print(
        "Unique:",
        len(
            np.unique(
                np.round(
                    result.frequency,
                    3
                )
            )
        )
    )
    
    
from dataset import DatasetBuilder

def test_dataset_builder():

    G = nx.erdos_renyi_graph(
        30,
        0.2,
        seed=0,
    )

    problem = build_mis_problem(
        G
    )

    builder = DatasetBuilder()

    dataset = builder.build(
        problem
    )

    print(
        "Feature matrix shape:"
    )
    print(
        dataset.X.shape
    )

    print(
        "\nFeature names:"
    )
    print(
        dataset.feature_names
    )

    print(
        "\nFirst vertex:"
    )
    print(
        dataset.X[0]
    )
    
if __name__ == "__main__":
    #run_demo()
    #run_demo2()
    #compare_lp_vs_mwua()
    #compare_lp_vs_mwua_many()
    #test_mis()
    #test_features()
    #test_dataset()
    #test_centrality()
    #test_mwua_features()
    #test_lp_features()
    #test_luby()
    test_dataset_builder()
    '''G = nx.watts_strogatz_graph(
        n=50,
        k=6,
        p=0.1
    )
    problem=build_mis_problem(G)
    lp=solve_lp_relaxation(problem)
    n_half = np.sum(
    np.abs(lp.x - 0.5) < 1e-6)

    print( "Half-integral:",n_half,"/",len(lp.x))
    print("Fraction:",n_half / len(lp.x))'''
    
    '''problem = build_vertex_cover_problem(G)
    mwua = MWUAFeatureExtractor()
    result = mwua.compute(problem)'''
    #plot_mwua(G, result)
