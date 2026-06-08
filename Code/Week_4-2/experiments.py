import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from problem import build_vertex_cover_problem
from solver import BranchAndBoundSolver
from mwua import MWUAFeatureExtractor
from lp import solve_lp_relaxation
from branching import (MostFractionalBranching,MWUABranching)

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
    mwua = MWUAFeatureExtractor()
    problem = build_vertex_cover_problem(G)
    result = mwua.compute(problem)
    lp = solve_lp_relaxation(problem)

    lp_certainty = np.abs(
        lp.x - 0.5
    )
    corr = np.corrcoef(
        lp_certainty,
        result.certainty
    )[0, 1]
    print(lp.x)
    print("LP-MWUA Correlation:", corr)
    for i in range(10):

        print(
            f"v={i:2d}",
            f"LP={lp_certainty[i]:.3f}",
            f"MWUA={result.certainty[i]:.3f}",
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

def compare_lp_vs_mwua():

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

    lp_nodes = []
    mwua_nodes = []
    results = []
    n1=p1=0

    for seed in range(20):

        G = nx.erdos_renyi_graph(
            n=30,
            p=0.2,
            seed=seed,
        )
        n1,p1=len(G),nx.density(G)

        problem = build_vertex_cover_problem(G)

        mwua = MWUAFeatureExtractor()
        mwua_result = mwua.compute(problem)

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

        lp_nodes.append(
            lp_result.nodes_explored
        )

        mwua_nodes.append(
            mwua_result_solver.nodes_explored
        )
        
        

    print(
        "LP Avg:",
        np.mean(lp_nodes)
    )

    print(
        "MWUA Avg:",
        np.mean(mwua_nodes)
    )
    
    results.append([n1,p1,np.mean(lp_nodes),np.mean(mwua_nodes)]) 
    print(results)


    
if __name__ == "__main__":
    #run_demo()
    #run_demo2()
    #compare_lp_vs_mwua()
    compare_lp_vs_mwua_many()
    '''G = nx.erdos_renyi_graph(
        n=30,
        p=0.2,
        seed=42,
    )
    problem = build_vertex_cover_problem(G)
    mwua = MWUAFeatureExtractor()
    result = mwua.compute(problem)'''
    #plot_mwua(G, result)
