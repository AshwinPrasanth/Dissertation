# Annotated Bibliography — Learning-to-Branch for MIS and Hitting Set

A detailed reading guide to the 19 sources in your literature review plan, organized by the four reading phases. Each entry covers: what the paper actually does (mechanism-level), why it matters technically, and how it plugs into your dissertation's argument.

---

## Phase A — Building the Foundation

### 1. Khalil, Le Bodic, Song, Nemhauser & Dilkina (2016) — *Learning to Branch in Mixed Integer Programming*

**What it does.** This is the paper that turned "which variable should branch-and-bound branch on?" into a supervised machine-learning problem. Classical MIP solvers use hand-tuned rules (most-fractional, pseudocost branching, reliability branching) to pick the branching variable at each node. Strong Branching (SB) — actually solving the LP relaxation for both child nodes of *every* candidate variable and picking whichever produces the best bound improvement — gives excellent per-node decisions but is prohibitively slow, because it means solving 2×(number of candidates) extra LPs at every single node.

Khalil et al.'s idea: run full Strong Branching only for the first few hundred nodes of the tree to collect (features, SB-score) pairs, train a fast model offline (per-instance, at the start of solving), and then switch to using that model's predictions for branching decisions for the remainder of the tree. This gives you SB-quality decisions without SB-level per-node cost after the initial "burn-in."

**Key mechanisms worth citing:**
- **Ranking, not regression.** Because branching only needs the *relative order* of candidate variables (you just need to know which is best, not the exact SB score), they frame learning as a ranking problem (rank-SVM style pairwise loss) rather than regressing the raw SB score. This is a design choice you should explicitly connect to modern ranking objectives in gradient-boosted trees (e.g., XGBoost's `rank:pairwise` / `rank:ndcg` objectives), since it's the direct ancestor of that idea.
- **Candidate-variable features.** Static features (objective coefficients, constraint degree, bound ranges) combined with dynamic features computed from the current LP relaxation (fractionality, pseudocost history, reduced costs). This feature-engineering template — static + dynamic/state-dependent features — is the pattern your own MIS/Hitting-Set feature sets should be benchmarked against.
- **Teacher-quality vs. inference-cost tradeoff.** The paper is explicit that SB is "too slow to use everywhere," which is the earliest clean statement of the tradeoff that defines the entire Learning-to-Branch literature: an expensive, high-quality *teacher* signal is used sparingly to train a cheap *student* that gets deployed at scale.

**Relevance to you.** This is your template for "imitate an expensive oracle with a cheap model," and the ranking framing anticipates why you might frame MIS/Hitting-Set vertex or literal selection as ranking rather than classification. It's also your natural point of contrast for teacher choice: your MIS chapter uses MWUA-derived global information rather than SB as the label source, and your Hitting-Set chapter uses CaDiCaL's own decision trace rather than SB at all — both deviations should be justified against this paper's SB-centric framing.

---

### 2. Bengio, Lodi & Prouvost (2021) — *Machine Learning for Combinatorial Optimization: A Methodological Tour d'Horizon*

**What it does.** A survey/position paper (European Journal of Operational Research) that doesn't propose a new method but instead builds the conceptual scaffolding for the entire field. It organizes ML-for-CO approaches along several axes:

- **End-to-end vs. algorithmic (hybrid) integration.** End-to-end: the ML model *is* the solver (or produces the solution directly, e.g., pointer networks for TSP). Algorithmic: ML augments one component of a classical, otherwise-exact or otherwise-guaranteed algorithm (a branching rule, a node-selection policy, a cut-selection heuristic), while the surrounding algorithm retains its correctness guarantees.
- **Frequency of ML invocation.** Some approaches query the model once per instance (e.g., predicting a good initial configuration or a warm-start). Others query it repeatedly, once per decision point inside a search procedure (e.g., once per branching decision) — which changes the cost-benefit calculus entirely, because now *inference latency accumulates across thousands of calls*.
- **Learning paradigms mapped onto CO structure**: supervised learning against an oracle (imitation), reinforcement learning against solver performance directly, and unsupervised/self-supervised approaches exploiting problem structure (e.g., relaxation-based losses).
- **Exactness vs. learned guidance.** A repeated theme: when ML sits *inside* an otherwise-exact algorithm (branch-and-bound, branch-and-reduce), the algorithm's correctness doesn't depend on the ML component being right — a bad prediction costs time, not correctness. This is fundamentally different from ML replacing the solver outright, where a bad prediction costs solution quality or feasibility.

**Relevance to you.** Use this paper to explicitly *locate* your dissertation on its axes in your literature review's opening framing: you are doing **algorithmic (hybrid) integration**, with **repeated, per-decision invocation** (not once-per-instance), preserving **exactness** in the surrounding branch-and-reduce / CDCL search. This framing sentence essentially writes itself once you've read this paper carefully, and reviewers/examiners will expect you to use exactly this vocabulary.

---

### 3. Gasse, Chételat, Ferroni, Charlin & Lodi (2019) — *Exact Combinatorial Optimization with Graph Convolutional Neural Networks*

**What it does.** The paper that made GNN-based Learning-to-Branch mainstream (NeurIPS 2019). Key contribution: a **bipartite graph representation** of a MIP node — one node type for variables, one for constraints, edges for nonzero coefficients, with edge/node features encoding coefficient values, variable bounds, objective coefficients, and LP-relaxation state (fractional value, reduced cost, etc.). A graph convolutional network processes this bipartite graph and predicts, for each variable, a score used to imitate Strong Branching decisions — trained via behavioral cloning (cross-entropy against the SB-selected variable) rather than pairwise ranking.

They plug the trained GCN into SCIP as a custom branching rule and evaluate on four synthetic MIP families introduced in that paper (which have since become the *de facto* benchmark suite for the whole subfield): **set covering, combinatorial auctions, capacitated facility location, and maximum independent set (as an MIP)**. This is worth noting explicitly — MIS as an *MIP* benchmark (not as a graph algorithm) originates here, and many follow-up Learning-to-Branch papers (branching, node selection, cut selection) reuse these exact instance generators.

**Key results relevant to you:**
- The GCN imitates SB decisions with high fidelity and reduces the number of branch-and-bound nodes explored relative to classical rules.
- However — and this is the critical point for your argument — the wall-clock speedup is much smaller than the node-count reduction would suggest, because GPU/neural-network inference at every single node adds latency that eats into the savings from a smaller tree. Depending on hardware and instance size, the net effect on *time* can be modest or even negative compared to reliability branching, even when the *tree* is dramatically smaller.

**Relevance to you.** This is your strongest citation for the central design argument of your dissertation: a heavy, high-capacity model (GNN) can imitate an expensive oracle extremely well in terms of decision quality, but the per-node inference overhead is a real, load-bearing cost that a lightweight tabular model (XGBoost, near-instant per-call) avoids almost entirely. Cite the tree-size-vs-wall-clock gap specifically — it's the empirical anchor for "quality of imitation is not the same as end-to-end usefulness," which is presumably one of your dissertation's core claims.

---

## Phase B — Building the MIS Contribution

### 4. O'Connor, Coleman, Strash, Ray & Ajwani (CPAIOR 2026 proceedings, presented 2025) — *A Scalable Learning Approach for Efficient Computation of Independent Set and Cover Variants*

**What it does, precisely.** This paper proposes a hybrid ML–algorithmic framework following the **Learning to Prune (LTP)** paradigm: a classifier predicts, per vertex, whether that vertex should be *fixed* (forced into or out of the solution) — i.e., pruned from the search space — after which the simplified instance is handed to a state-of-the-art exact/heuristic solver (**ReduMIS** for MIS). The central technical contribution is *how* they generate features cheaply:

- LP-relaxation-derived features are known to be highly informative for LTP-style pruning (a variable's LP-relaxation value is a strong signal for its integral value), but solving the LP relaxation of a MIS-as-ILP formulation at real-world scale is far too slow — LP solvers don't scale to graphs with millions of vertices in reasonable time, and can also be too coarse/uninformative on the highly symmetric MIS polytope.
- Their fix: adapt the **Multiplicative Weights Update Algorithm (MWUA)** from the theoretical CS literature as a fast surrogate. MWUA iteratively reweights vertices based on how "constrained" or "conflicted" they are (roughly: vertices in many violated/tight constraints get down-weighted, analogous to how MWUA reweights poorly-performing experts), producing a continuous per-vertex score that approximates the *structural signal* an LP relaxation would give, without needing to actually solve an LP. This is computed once, globally, over the (reduced) graph, and used as a feature for the pruning classifier.
- They also demonstrate the technique generalizes beyond MIS by applying the same MWUA-surrogate-feature idea to the **3-path vertex cover problem (VCP₃)**, showing it isn't MIS-specific machinery.

**Reported results (worth citing precisely):**
- Training on roughly **100 graph instances** labeled with ReduMIS solutions is sufficient for the learned pruning classifier to produce solutions within **10% of ReduMIS's own solution quality**, while running in **roughly half the time** — with the gain most pronounced on **dense graphs**.
- On VCP₃, using the top-ranked heuristic solver from the **PACE 2025 challenge** as the downstream solver, the aggressive pruning enabled by their classifier yields solutions **5% better than the PACE 2025 heuristic baseline** in **half the runtime** on large instances.

**Relevance to you.** This is your closest methodological sibling, and you should treat it almost as a direct predecessor to your MIS chapter's design: (1) it establishes that a cheap MWUA-based proxy for LP information is both theoretically motivated and empirically effective, which is exactly the justification you need for your own "global MWUA representation"; (2) it establishes the "reduce/prune first, then hand to an exact solver" pipeline shape that your dissertation's architecture mirrors; (3) its quantitative results (10% quality gap at ~2× speedup; 5% improvement at 2× speedup on VCP₃) give you concrete numbers to benchmark your own MIS results against, or to explain how/why your Learning-to-Branch approach (which intervenes *inside* the branching process rather than *before* it as a one-shot pruning step) differs in mechanism even if the motivating idea is shared. Note also the acknowledgement in the paper crediting a specific MWUA implementation from a collaborator (Ratul) — if your dissertation or advisor's group has any code-sharing lineage with this paper, that's worth flagging explicitly for provenance.

---

### 5. Arora, Hazan & Kale (2012) — *The Multiplicative Weights Update Method: A Meta-Algorithm and Applications*

**What it does.** A unifying survey (Theory of Computing) showing that an enormous range of algorithms — AdaBoost, Winnow, online learning with expert advice, approximate solvers for LPs/SDPs, game-theoretic equilibrium computation, and more — are all instances of one simple template:

1. Maintain a weight $w_i$ for each of $n$ "experts" (or elements, or constraints), initialized uniformly.
2. At each round, observe how well each expert performed (a loss or gain), and multiplicatively update its weight: $w_i \leftarrow w_i \cdot (1 \pm \eta \cdot \text{payoff}_i)$ for a learning rate $\eta$.
3. Decisions are made using the current weight distribution (e.g., weighted majority vote, or as a probability distribution for randomized decisions).

**Why this matters mechanically for MWUA-as-optimization.** The part you actually need (per your advisor's note — don't over-invest in the whole survey) is the application to **packing and covering LPs**. The intuition: for an LP like $\max c^Tx$ s.t. $Ax \le b$, you can treat each constraint as an "expert" whose weight reflects how *violated* or *tight* it currently is under your current candidate solution. Iteratively re-solving a much easier "weighted" version of the problem (e.g., a simple greedy pass weighted by constraint tightness) and updating constraint weights multiplicatively converges, after $O(\log m / \epsilon^2)$ rounds (for $m$ constraints, accuracy $\epsilon$), to a solution that is provably close to the LP optimum — without ever calling a general-purpose LP solver. This is exactly the mechanism that lets O'Connor et al. (paper #4) get "LP-like" structural signal at a fraction of the cost of actually solving an LP.

**Relevance to you.** You don't need the general regret-bound theory in depth. What you need to be able to state and cite correctly: (a) the multiplicative update rule itself and why it's cheap (no matrix factorization, no simplex iterations — just elementwise multiplication and renormalization per round); (b) the informal justification for why iterative reweighting on a packing/covering-structured problem (which MIS's LP relaxation is — MIS is a packing problem) approximates the LP optimum; (c) the convergence-rate intuition (logarithmic in problem size, polynomial in $1/\epsilon$), which is your argument for *why* MWUA is fast enough to compute globally even on very large graphs where an actual LP solve would not be.

---

### 6. Akiba & Iwata (2016) — *Branch-and-Reduce Exponential/FPT Algorithms in Practice: A Case Study of Vertex Cover*

**What it does.** An algorithm-engineering paper (Theoretical Computer Science) that systematically studies **kernelization** — polynomial-time-computable reduction rules that shrink a problem instance while preserving the ability to reconstruct an optimal solution — applied to Vertex Cover (equivalently, via complementation, to Independent Set). It catalogs and empirically evaluates a battery of reduction rules:

- **Degree-based rules**: isolated vertices, degree-0/1 vertices, and more subtly, **vertex folding** for degree-2 vertices (replacing a degree-2 vertex and its two neighbors with a single merged vertex when the neighbors aren't adjacent).
- **LP-relaxation-based rules**: using the fact that the LP relaxation of Vertex Cover is half-integral (Nemhauser-Trotter theorem) to fix variables at 0 or 1 whenever the LP relaxation already assigns them an integral value.
- **Domination and crown-decomposition rules**, and other more advanced structural reductions.

The empirical headline finding: applying these rules exhaustively and *repeatedly* (since applying one rule can expose opportunities for another) reduces huge real-world graphs down to a small **"irreducible kernel"** — often orders of magnitude smaller than the input — such that essentially all the exponential-time branching work happens on this tiny residual instance, not on the original graph.

**Relevance to you.** This paper is your justification for a specific, easy-to-miss methodological decision: **compute your MWUA-based global signal *after* reduction, on the actual residual/kernel graph the solver will branch on — not on the raw input graph.** The argument, straight from this paper: the raw input graph's structure is mostly irrelevant to the actual search, because reduction rules will strip most of it away before any branching decision is made; computing expensive-ish global features on the pre-reduction graph wastes compute on vertices/edges that won't exist in the instance the solver actually struggles with, and — more importantly — could give a distorted signal, since the *reduced* graph has a different degree distribution, density, and structure than the original. Cite specific reduction rules by name (isolated vertex removal, degree-1/2 folding, LP-based fixing) when describing your own preprocessing pipeline, since precision here signals to examiners that you understand the mechanism, not just the headline claim.

---

### 7. Langedal, Hespe & Sanders (SEA 2024) — *Targeted Branching for the Maximum Independent Set Problem Using Graph Neural Networks*

**What it does.** This is the direct GNN-based sequel to Hespe, Lamm & Schorr's earlier (non-learned) 2021 "Targeted Branching" paper, and it's your closest direct comparison point in the MIS Learning-to-Branch space specifically (as opposed to MIP Learning-to-Branch more generally, which is Khalil/Gasse).

**Background — what "targeted branching" means.** In branch-and-reduce MIS solvers, once you've exhausted cheap polynomial-time reductions, you're forced to *branch*: pick a vertex, and recursively solve two subproblems (include it / exclude it). The classical default is to branch on the maximum-degree vertex, on the (weak) intuition that removing a high-degree vertex triggers the most follow-on reduction opportunities. "Targeted" branching strategies instead choose branching vertices with a more specific structural goal in mind — e.g., (1) **disconnection-targeting**: pick a vertex whose removal splits the graph into multiple connected components, since components can then be solved *independently* (and independent MIS problems are easier in aggregate than one large connected one), or (2) **reduction-targeting**: pick a vertex specifically because removing it will trigger additional reduction rules to fire (e.g., unlocking a folding opportunity), producing a smaller residual graph than degree-based branching would.

**What the GNN adds.** Rather than using hand-crafted heuristics to decide *which* of these targeted strategies (or which specific vertex within a strategy) to use at each branching point, this paper trains a GNN to predict, from local/global graph structure, which branching choice is likely to be most effective — i.e., learning the branching *strategy selection*, not just scoring individual vertices for direct inclusion/exclusion the way Khalil/Gasse-style SB-imitation does.

**What to measure carefully when reading it.** Per your advisor's guidance, the paper's own evaluation distinguishes between two outcomes that can diverge: reduction in **search-tree size** (number of branch-and-reduce nodes explored) and reduction in **actual wall-clock runtime**. As with Gasse et al. (paper #3), GNN inference cost at each branching decision is nontrivial, so a strategy that shrinks the tree substantially can still fail to improve — or can even worsen — real running time if the model has to be queried at every node.

**Relevance to you.** This is the paper to compare your MIS Learning-to-Branch results against directly, both in terms of problem setting (MIS, branch-and-reduce, GNN vs. your lightweight-feature approach) and in terms of the *specific evaluation pitfall* it illustrates: report both tree-size and wall-clock numbers for your own method, and use this paper's tree-size/wall-clock divergence as precedent for why wall-clock is the metric that actually matters for your dissertation's practical claims.

---

### 8. Ahn, Seo & Shin (ICML 2020) — *Learning What to Defer for Maximum Independent Sets*

**What it does.** A reinforcement-learning approach to constructing MIS solutions sequentially: at each step, the current partial solution and remaining ("live") graph define a state, and a policy decides which vertex to add to the independent set next (removing it and its neighbors from consideration). The paper's distinctive contribution is the **defer mechanism**: rather than forcing the policy to commit to a decision at every single step, it can choose to **defer** — effectively passing the decision to a later point in the process, once more information (from further graph reduction, or from the evolving state) is available, rather than committing early based on incomplete information.

The motivation is explicitly framed around **decision difficulty heterogeneity**: some decisions in MIS construction are "easy" (locally obvious — e.g., a vertex with very few remaining neighbors, or one that dominates its neighborhood) and can be handled by simple/greedy logic, while others are "hard" (genuinely ambiguous, requiring more global information or more search) and benefit from the learned policy's attention. Rather than spending equal learned-model effort on every decision, the defer mechanism lets the system triage: cheap logic for easy decisions, the (more expensive) learned policy concentrated on hard ones.

**Relevance to you.** This paper is explicitly flagged by your advisor as connecting to your dissertation's *conclusion*, and the connection is precise: your overall argument (through both the MIS and Hitting Set chapters) is that learned/expensive intervention should be **selective and bounded** — used only where it earns its computational cost — rather than applied uniformly to every decision in the search (which is what a "replace-every-branching-decision-with-a-GNN-call" approach like Gasse or Langedal effectively does). Ahn et al. gives you a second, independent line of evidence (from a construction-based RL setting rather than a branch-and-bound setting) for the same underlying principle: *not every decision should be handled by the learned/expensive component.* Cite it in your conclusion as convergent evidence from a different algorithmic paradigm.

---

## Phase C — Building the Hitting Set Literature

### 9. The PACE 2025 Challenge Paper — *Dominating Set and Hitting Set*

**What it is.** The official problem/challenge description for the 10th PACE (Parameterized Algorithms and Computational Experiments) Challenge, run in conjunction with IPEC 2025. PACE is an annual competitive benchmark where teams submit solvers for a chosen NP-hard problem, evaluated on curated instance sets under strict time/memory limits, split into **exact** and **heuristic** tracks.

**Problem formulations, precisely:**
- **Dominating Set**: given a graph $G=(V,E)$, find a minimum-size subset $S \subseteq V$ such that every vertex in $V$ either is in $S$ or has a neighbor in $S$.
- **Hitting Set**: given a hypergraph $H=(V,\mathcal{E})$ (a generalization — $\mathcal{E}$ is a set of hyperedges, i.e., arbitrary subsets of $V$, not just pairs), find a minimum-size subset $S \subseteq V$ such that $S \cap e \neq \emptyset$ for every hyperedge $e \in \mathcal{E}$.
- The two are closely related: Dominating Set reduces naturally to Hitting Set (each vertex's closed neighborhood becomes a hyperedge), which is why most PACE 2025 submissions (per the solver-description papers below) handle both problems through a shared Hitting Set-based pipeline.

**Tracks and constraints.** Both an **exact track** (provably optimal solutions required, with a correctness sketch, five-minute-ish time limits and ~16GB memory in the reported configurations) and a **heuristic track** (best solution found within the time limit, no optimality proof required, but with anytime behavior expected — solvers must be able to output a valid solution if terminated early via SIGTERM). Submissions may use non-commercial SAT/MaxSAT/ILP solvers as backends, which is why nearly every submission (see items #10–12 below) is built around **MaxSAT reformulation**.

**Benchmark characteristics.** Instances are drawn from a mix of synthetic generators and real-world graph/hypergraph sources, deliberately including hard, large-scale cases meant to stress both reduction-rule effectiveness and backend solver scalability — this is the evaluation ecosystem your Hitting Set chapter's experiments should be positioned within.

**Relevance to you.** Read this before any of the solver-description papers, because they all assume the reader knows: what counts as valid input/output format, what the track rules are (which shapes what "exact" vs. "anytime heuristic" behavior means for a submission), and what instance characteristics make these problems hard in practice (large hyperedges, high vertex-to-hyperedge ratios, real-world scale-free structure). Your experimental chapter's benchmark section should explicitly ground itself in this challenge's instance families and evaluation protocol if you're using PACE 2025 data.

---

### 10. UzL Solver Description (Bannach, Chudigiewitsch & Wienöbst, IPEC 2025) — *PACE Solver Description: UzL Solver for Dominating Set and Hitting Set*

**What it does, in detail.** This is the solver architecture your final Hitting Set chapter is explicitly modeled on, so it's worth understanding precisely:

1. **Reduction/preprocessing pipeline**: the raw Dominating Set / Hitting Set instance is first passed through hitting-set-specific reduction rules (e.g., dominated hyperedges — if hyperedge $e_1 \subseteq e_2$, then $e_2$ is redundant since hitting $e_1$ automatically hits $e_2$; forced/mandatory vertices — if some vertex is the *only* member of some hyperedge, it must be in every hitting set; and similar polynomial-time-computable simplifications) to shrink the instance before any expensive solving happens — directly analogous to the Akiba & Iwata kernelization philosophy from paper #6, but specialized to hypergraphs.
2. **MaxSAT transformation**: the reduced instance is encoded as a MaxSAT problem — one **hard clause per hyperedge** (encoding "at least one vertex in this hyperedge must be selected"), and **one soft clause per vertex** (encoding the objective — minimize the number of selected vertices, via unit soft clauses with weight 1 that are satisfied when a vertex is *not* selected). This is a very direct, almost mechanical encoding, which is part of why "straightforward MaxSAT formulation" is the paper's own description of the core approach.
3. **CDCL backend**: the resulting MaxSAT instance is solved via a MaxSAT solving procedure built on a CDCL SAT solver as its core engine (see items #11 and #12) — meaning the actual combinatorial search happening under the hood is CDCL-style: decisions, unit propagation, conflict-driven clause learning.
4. **Special-case fallback**: if, after reduction, the instance's structure collapses to something recognizable as a (small) **Vertex Cover** instance, the solver switches to a dedicated clique/vertex-cover solver instead of continuing with the general MaxSAT pipeline, since specialized VC solvers can be much faster on that specific structure than general MaxSAT solving.
5. **Lower-bound matching**: the description also mentions attempting to match certain lower bounds (a standard exact-algorithm technique — if a fast-to-compute lower bound on the solution size matches a feasible solution already found, you can terminate early with a proof of optimality without further search).

**Relevance to you.** This is the *architectural blueprint* your Hitting Set chapter's hybrid system should be described against: reduce → encode as MaxSAT → solve via CDCL, with your ML intervention point living **inside step 3** (guiding the CDCL solver's decision heuristic) rather than replacing steps 1–2. When you write up your own pipeline, walk through it stage-by-stage against this paper's stages, explicitly stating where your learned component sits and why you chose that insertion point (e.g., "unlike UzL's default CDCL decision heuristic, we replace/augment the branching-literal selection with a model trained to imitate/improve on CaDiCaL's native VSIDS-based choices").

---

### 11. CaDiCaL Solver (Biere and collaborators; system descriptions from SAT Competition entries, e.g., Froleyks, Heule, Iser, Järvisalo & Suda)

**What it does.** CaDiCaL is a modern, widely used, simplified/clean-codebase **CDCL (Conflict-Driven Clause Learning)** SAT solver, commonly used both as a standalone solver and as the underlying SAT engine inside MaxSAT tools (including EvalMaxSAT — see #12 — and, per the PACE 2025 solver-description papers, several teams' MaxSAT pipelines explicitly build on CaDiCaL).

**The CDCL loop, mechanically — this is what you need in depth, since your model learns from it directly:**

1. **Decision**: when unit propagation stalls (no forced assignments remain), the solver picks an unassigned variable and a polarity (true/false) to try next. Classical heuristics here are activity-based — **VSIDS** (Variable State Independent Decaying Sum) tracks a score per variable that increases when the variable appears in a recently-learned conflict clause, and periodically decays all scores, so the heuristic dynamically focuses on variables that have recently been "relevant" to conflicts (a proxy for problem difficulty in that region of the search).
2. **Unit propagation (Boolean Constraint Propagation)**: after each decision, repeatedly apply the rule "if a clause has all-but-one literals falsified, the remaining literal must be true" until no more forced assignments follow. This is the workhorse of CDCL and is implemented extremely efficiently via **watched literals** (each clause tracks only two "watched" literals rather than scanning the whole clause on every assignment change).
3. **Conflict detection and analysis**: if propagation derives a clause that is entirely falsified (a conflict), the solver doesn't just backtrack chronologically — it analyzes the *implication graph* of assignments that led to the conflict and derives a **learned clause** (via the First-UIP scheme) that captures the root cause, preventing the solver from repeating the same mistake in a different part of the search.
4. **Backjumping**: rather than backtracking one decision level at a time, the solver jumps directly to the decision level indicated by the learned clause (typically much earlier than the most recent decision), which is what makes CDCL dramatically faster than plain DPLL backtracking search.
5. **Restarts and clause deletion**: periodically the solver discards its current decision trail and starts over (keeping learned clauses, within limits), and periodically prunes low-activity learned clauses to bound memory — both are engineering-level heuristics tuned for empirical performance.

**Why this matters for your framing.** Note the qualitative difference from Strong-Branching-style teachers (Khalil, Gasse): SB scores are derived from LP relaxations and are fundamentally about *continuous bound tightness*. CDCL's decision heuristic (VSIDS) is derived from *conflict frequency* — a completely different signal, rooted in propositional proof search rather than linear-programming duality. This is your explicit justification for why your Hitting Set model's teacher/imitation target (CaDiCaL's own decision behavior) is not just "SAT's version of Strong Branching" — it's a structurally different kind of signal, and that distinction should be stated explicitly in your methodology chapter.

**Relevance to you.** Your Hitting Set model learns from CaDiCaL's decision behavior, not from Strong Branching. Make sure your write-up correctly names the decision heuristic you're imitating or augmenting (VSIDS or CaDiCaL's specific variant of it), and be precise about which point in the loop (step 1, decision selection) your ML component intervenes on versus which points it leaves untouched (propagation, conflict analysis, backjumping remain native CDCL machinery).

---

### 12. EvalMaxSAT Solver (Avellaneda)

**What it does.** EvalMaxSAT is a MaxSAT solver — meaning it solves the **optimization** version of SAT: given hard clauses (must be satisfied) and soft clauses (each with a weight; satisfy as many/as much total weight as possible), find an assignment satisfying all hard clauses while maximizing satisfied soft-clause weight (equivalently, minimizing the total weight of *un*satisfied soft clauses). Per the PACE 2025 solver-description papers your reading list draws from, EvalMaxSAT is the backend of choice for **most** PACE 2025 Hitting Set/Dominating Set exact-track submissions (UzL, Shadoks, OBLX, and others all reference it), typically running on top of CaDiCaL as the underlying SAT engine.

**Core architecture (the part worth knowing):**
- **Core-guided search**: rather than directly searching over all possible numbers of satisfied soft clauses, core-guided MaxSAT solvers repeatedly call the underlying SAT solver with **assumptions** (temporarily-forced literal values) on the soft clauses, and when the SAT solver returns UNSAT, it also returns an **unsatisfiable core** — a minimal subset of the assumed soft clauses that cannot all be simultaneously satisfied. The MaxSAT solver uses this core to add a cardinality constraint (e.g., "at most $k{-}1$ of these clauses can be satisfied") and relaxes the encoding, iterating until SAT is returned, at which point the current assignment is provably optimal (or a valid bound has been established).
- **Interaction with the SAT backend**: each iteration of core-guided search is fundamentally "call the SAT solver, interpret its (UN)SAT verdict and, if UNSAT, its returned core" — so the actual combinatorial work happens inside repeated CDCL solver calls (item #11), and EvalMaxSAT's job is to manage the sequence of SAT queries and cardinality-constraint bookkeeping around them.

**Relevance to you — scoped narrowly, per your advisor's guidance.** You don't need EvalMaxSAT's full internals (e.g., specific cardinality-encoding schemes, or comparisons against linear-search MaxSAT strategies) unless your implementation directly extends or modifies it. What you need clearly stated: EvalMaxSAT sits **between** the MaxSAT encoding (produced by the reduction/transformation stage, per item #10) and the CDCL solving (item #11) — it's the orchestration layer that turns repeated SAT calls into a provably optimal MaxSAT solution, and your ML intervention (if it touches the SAT layer's decision heuristic) is therefore happening *underneath* EvalMaxSAT's core-guided loop, not inside it.

---

## Phase D — Supporting Methodology

### 13. Chen & Guestrin (2016) — *XGBoost: A Scalable Tree Boosting System*

**What it does.** The systems paper behind one of the most widely deployed gradient-boosted decision tree (GBDT) implementations. Beyond the standard gradient boosting idea (iteratively fit new trees to the negative gradient of the loss with respect to current predictions), the paper's specific engineering contributions are:

- **Sparsity-aware split finding**: real-world tabular data (and especially engineered graph/solver features, which are often sparse — many zero or missing entries) is handled via a default direction learned for missing values at each split, avoiding the need for explicit imputation and making the algorithm efficient on sparse feature matrices.
- **Weighted quantile sketch**: an approximate algorithm for finding good split candidates on weighted data (relevant if your ranking/weighted objectives assign different importance to different training examples) without needing to sort the full dataset at every tree level.
- **Cache-aware access patterns and out-of-core computation**: engineering-level optimizations (block-based data layout, compression, sharding) that make training and inference fast even on very large datasets — relevant background if your feature sets are large-scale (many vertices/hyperedges × many features).
- **Regularized objective**: XGBoost's loss function explicitly includes a regularization term over tree complexity (number of leaves, leaf weight magnitudes), which is part of why it tends to generalize well without extensive manual tuning compared to plain gradient boosting.
- **Ranking objectives**: XGBoost supports pairwise (`rank:pairwise`) and listwise (`rank:ndcg`) ranking losses natively, which is the direct technical link back to Khalil et al.'s (paper #1) framing of branching as a ranking problem — if your model uses a ranking objective for vertex/literal selection, this is where that objective is implemented.

**Relevance to you.** This should be a supporting citation, not a centerpiece: cite it for (a) why gradient-boosted trees handle heterogeneous, structured tabular features (degree, local density, MWUA scores, reduction-rule trigger indicators, VSIDS-style activity, etc.) well without extensive feature scaling or architecture design, unlike neural approaches; (b) why inference is fast (a handful of tree traversals, no matrix multiplications or GPU dependency) — directly supporting your dissertation's argument that a lightweight model avoids the per-node inference cost problem documented in Gasse et al. (#3) and Langedal et al. (#7); (c) if relevant, the native ranking-objective support connecting back to Khalil's ranking framing.

---

### 14. Mandi & Guns (2020) — *Interior Point Solving for LP-Based Predict-and-Optimise* (Optional)

**What it does.** This sits in the **decision-focused learning** / "predict-and-optimize" line of work, which is conceptually distinct from Learning-to-Branch. The setup: you have a prediction task (e.g., predict some LP's cost coefficients from features) followed by a downstream optimization problem (solve the LP with those predicted coefficients), and instead of training the predictive model with a standard loss (e.g., mean-squared error against the true coefficients), you train it **end-to-end against the downstream decision quality** — i.e., backpropagate through the optimization step itself, so the model learns to predict coefficients that lead to *good decisions*, even if the coefficient predictions themselves aren't individually accurate.

The technical challenge: standard LP solvers (simplex-based) are not differentiable — the optimal solution as a function of the input coefficients is piecewise-constant/non-smooth almost everywhere, breaking gradient-based training. Mandi & Guns' contribution is to use **interior-point methods** instead of simplex: interior-point solvers traverse the interior of the feasible region and are, under mild conditions, differentiable at the optimum (via implicit differentiation of the KKT/optimality conditions), which allows gradients to flow from the downstream decision quality back through the LP solve and into the upstream predictive model's parameters.

**Relevance to you — and why it's skippable.** Your dissertation doesn't differentiate *through* an optimization solver anywhere — you're doing imitation/ranking-based learning over discrete branching/literal-selection decisions (a classification/ranking task with a fixed, pre-computed label), not end-to-end differentiable optimization. Per your advisor's note, only pull this in if you want a short paragraph situating your dissertation against the broader "ML-optimization interaction" landscape (distinguishing decision-focused/predict-and-optimize approaches from the imitation-learning approach your work actually uses) — otherwise it can be omitted entirely without weakening your argument.

---

### 15. Zhang, Liu, Wang, Lu, Song, Zhu, Shi & Chi, or similar authorship — *A Survey for Solving Mixed Integer Programming via Machine Learning* (2023)

**What it does.** A broad survey mapping the ML-for-MIP landscape by *where in the MIP-solving pipeline* the learned component is inserted, roughly:

- **Branching**: the Learning-to-Branch literature (Khalil, Gasse, and dozens of follow-ups) — predicting or ranking candidate branching variables.
- **Node selection**: which open node in the branch-and-bound tree to explore next (best-first vs. depth-first vs. learned policies balancing bound tightness against exploration).
- **Cutting-plane / cut selection**: which valid inequalities to add to tighten the LP relaxation, and in what order, since adding too many cuts can slow down each LP solve even as it tightens bounds.
- **Primal heuristics**: predicting good initial feasible solutions or partial assignments (e.g., "Neural Diving"-style approaches) to give the solver a strong incumbent early, improving the primal-dual gap quickly.
- **Other components**: presolve reduction selection, solver/parameter configuration (algorithm selection and hyperparameter tuning via ML), and cross-cutting representation-learning work (e.g., GNN embeddings of MIP instances reused across multiple downstream tasks).

**Relevance to you.** Use this as an **organizing citation map**, not a technical pillar: in your literature review's structural overview (probably early in the MIP/MIS-adjacent background section), you can say something like "branching has been studied extensively [Khalil 2016, Gasse 2019, ...]; node selection has a separate but related literature [cite via survey]; cutting-plane selection is a further separate thread [cite via survey]" — using this single survey as the anchor citation for the sub-areas you are *not* covering in depth, which lets you scope your literature review precisely without needing to individually source every adjacent sub-area yourself.

---

## Optional / Supplementary

### ML4CO-Bench-101 (Ma, Pan, Li & Yan, NeurIPS 2025 Datasets & Benchmarks Track)

**What it does, confirmed.** A large-scale reproducibility-focused benchmark for **neural combinatorial optimization (NCO)** solvers — note this is a different subfield from Learning-to-Branch: NCO typically means end-to-end neural models that construct or predict solutions directly (autoregressive construction, diffusion-based generation, GNN-based direct prediction), not ML embedded inside a classical branch-and-bound/branch-and-reduce solver. The paper:

- Organizes NCO methods by a **"paradigm–model–learning" taxonomy**: *paradigm* (how the solution is produced — global prediction, local/sequential construction, or adaptive expansion), *model* (architecture family — e.g., GNN, diffusion, consistency models), and *learning strategy* (supervised, reinforcement, self-/unsupervised).
- Consolidates **65 datasets across 7 mainstream graph-structured CO problems** — TSP, ATSP, CVRP (edge-oriented/routing problems) and **Maximum Independent Set, Maximum Clique, Minimum Vertex Cover, Maximum Cut** (node-oriented problems, directly overlapping your MIS work) — spanning a range of instance scales.
- A key empirical finding worth citing: much of the reported performance gap between NCO methods in prior literature is attributable to **inconsistent pre-/post-inference heuristic tricks** (e.g., aggressive local search wrapped around the neural component) rather than the raw learned model's own contribution — the paper explicitly argues such tricks shouldn't be allowed to "compensate for sub-par capability of the data-driven counterparts," and builds a unified, controlled evaluation harness to strip this confound out.

**Relevance to you.** Two uses: (1) as a source of standardized, scale-varied MIS datasets/instance generators if you want external validation instances beyond what O'Connor et al. or the classical Gasse-style generators provide; (2) as a citation for the broader methodological point — echoed across your reading list — that fair, confound-free evaluation is a known open problem in ML4CO, which supports being explicit and careful about your own experimental protocol (e.g., clearly separating gains from your learned component vs. gains from surrounding reduction/heuristic machinery).

### ConRep4CO, Symb4CO, and BackPaS

I was not able to locate these three under these names across CPAIOR, NeurIPS, arXiv, or PACE-related sources — they don't appear in any of the reference lists or proceedings I checked. Possibilities: very recent/unindexed preprints, an informal or internal shorthand your advisor is using for papers with different official titles, or a slight name variation (my best guesses, unverified: something in the contrastive-representation-learning-for-CO space for "ConRep4CO," and symbolic-regression-for-CO for "Symb4CO," but I want to flag clearly that these are guesses, not confirmed identifications). I'd recommend asking your advisor directly for the exact titles/authors/venues, or searching your institution's reference manager / shared Zotero library if these were added there — I don't want to characterize their content when I can't confirm what they actually are.

---

## Quick-reference comparison table

| # | Paper | Core mechanism | Teacher/signal | Where it sits in your pipeline |
|---|---|---|---|---|
| 1 | Khalil et al. 2016 | SB imitation, ranking loss | Strong Branching | Template for imitation framing |
| 2 | Bengio et al. 2021 | Taxonomy | — | Positions your work (hybrid, repeated, exact) |
| 3 | Gasse et al. 2019 | GCN on bipartite MIP graph | Strong Branching | Heavy-model baseline; inference-cost argument |
| 4 | O'Connor et al. 2025/26 | LTP pruning + MWUA features | ReduMIS solutions | Direct predecessor to MIS chapter |
| 5 | Arora, Hazan & Kale 2012 | Multiplicative weights | — | Theory behind MWUA feature |
| 6 | Akiba & Iwata 2016 | Kernelization | — | Justifies reduce-then-featurize order |
| 7 | Langedal et al. 2024 | GNN-selected branching strategy | — | Direct MIS L2B comparison |
| 8 | Ahn et al. 2020 | Defer mechanism in RL construction | — | Supports "selective intervention" thesis |
| 9 | PACE 2025 challenge | Benchmark/track definitions | — | Experimental context for HS chapter |
| 10 | UzL solver 2025 | Reduce → MaxSAT → CDCL | — | Architectural blueprint for HS chapter |
| 11 | CaDiCaL | CDCL (VSIDS, propagation, learning) | — | Your HS model's imitation target |
| 12 | EvalMaxSAT | Core-guided MaxSAT over SAT calls | — | Orchestration layer above CDCL |
| 13 | Chen & Guestrin 2016 | GBDT engineering | — | Justifies lightweight model choice |
| 14 | Mandi & Guns 2020 | Differentiable LP via interior point | — | Optional; contrast with your approach |
| 15 | Zhang et al. 2023 | Survey | — | Organizing citation for adjacent areas |
