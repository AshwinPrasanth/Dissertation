Zhang et al. (2023) provide a comprehensive survey of the integration of machine learning (ML) within Mixed Integer Programming (MIP), emphasizing the "mutual embrace" between traditional exact solvers and data-driven components. The authors categorize learning-based approaches into exact and heuristic algorithms, with a significant focus on enhancing the **Branch-and-Bound (B&B)** framework through improved Branching Variable Selection (BVS) and node selection. Critically for exact optimization, the paper outlines how ML can be utilized as a subroutine within state-of-the-art solvers to maintain mathematical guarantees while navigating discrete and non-convex feasible regions.

A key contribution of the survey is its analysis of the computational trade-offs in **variable selection heuristics**, noting that while Strong Branching (SB) minimizes tree size, its "enormous computational overhead" is often "unacceptable in practice". The authors discuss alternatives such as **Reinforcement Learning (RL) from scratch**, which can discover policies independent of traditional heuristics, and imitation learning models that use Graph Convolutional Networks (GCNs) to process bipartite graph representations of variables and constraints. These methodologies support the development of **lightweight models** that avoid the per-node bottleneck of expensive calculations by leveraging structural priors.

Regarding the challenge of **large-scale instances**, Zhang et al. advocate for features that are **independent of problem size**, suggesting the use of feature aggregation to create "global feature vectors". This is particularly relevant to approaches that utilize **initial global snapshots**, as the survey highlights hybrid architectures that apply complex models (like GNNs) only at the root node while employing faster, weaker predictors deeper in the search tree. By reducing the solution space through variable fixing and predicting high-quality initial solutions, the paper demonstrates how ML can scale to industrial-level problems with millions of variables.

The source paper by Zhang et al. (2023) discusses various heuristics used for **variable selection**, **node selection**, and **variable ordering** within the context of Branch-and-Bound (B&B) for Mixed-Integer Programming (MIP). As a survey, it details both traditional rules and modern machine-learning (ML) enhancements.

### 1. Branching Variable Selection (BVS)
Variable selection determines which fractional variables (candidates) should be used to partition a current node into child nodes. The paper identifies several key heuristics:

*   **Strong Branching (SB):** This rule tentatively computes the linear relaxation for candidate variables to identify which one provides the most significant **lower bound improvement**. While it keeps the search tree small, its **computational overhead** is often considered unacceptable in practice.
*   **Pseudo-cost Branching:** This heuristic tracks variables that have already been branched on and predicts their impact on the objective function based on **historical success**. It is highly efficient but lacks reliable history at the beginning of the search process.
*   **Hybrid Branching:** To balance efficiency and accuracy, this strategy performs **strong branching at the start** of the solution process and then switches to pseudo-cost branching once enough history is recorded.
*   **Reliability Pseudo-cost Branching (relpscost):** Used in solvers like SCIP, this combines SB and pseudo-cost branching while depending on **high-level branching factors**.
*   **Learning-Based BVS:** Modern approaches use **Imitation Learning** to create fast approximations of Strong Branching or **Reinforcement Learning** to discover entirely new branching policies from scratch.

### 2. Node Selection
Node selection determines the priority of unexplored subproblems in the search tree to guide the search toward promising areas and find excellent **incumbent solutions** quickly.

*   **Traditional Rules:** The paper describes two primary search strategies, though it notes their specific characteristics differently than standard computer science definitions:
    *   **Breath-first search (BFS):** Explores a node branch as far as possible before backtracking. 
    *   **Depth-first search (DFS):** Starts at the root and explores all nodes at a given depth before moving deeper; it typically has **lower memory requirements** and finds feasible solutions faster.
*   **Learning-Based Node Selection:** Researchers use ML to train policies that enforce **linear priorities** on current nodes or predict which child nodes should be selected to reach a solution faster rather than following a standard BFS approach.

### 3. Variable Ordering and Primal Heuristics
While "variable ordering" is often a result of the scoring in BVS, the paper details specific **primal heuristics** that determine the order in which variables are fixed or rounded to find approximate solutions:

*   **Diving Heuristics:** These simulate a **depth-first search** by fixing integer variables step-by-step through continuous branching and LP solving until constraints are satisfied.
    *   **Fractional Diving:** A specific strategy where the variable **closest to an integer** is bounded first.
*   **Rounding:** This basic component rounds fractional solutions from LP relaxations to integer points while attempting to maintain **feasibility**.
*   **Feasibility Pump (FP):** An improvement heuristic that iteratively rounds continuous solutions and projects them back into the relaxed feasible region until a feasible integer solution is found.
*   **Large Neighborhood Search (LNS):** This method defines a large neighborhood around an incumbent solution—often by **fixing a subset of variables**—and then searches that sub-problem for a better solution.

Finally, the authors highlight that modern solvers often use **bipartite or tripartite graph representations** to extract correlations between variables and constraints, which allows ML models to "rank" or order variables for branching more effectively than simple handcrafted rules.
