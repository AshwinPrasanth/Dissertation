I think this is the right time to step back and enumerate the design space systematically. Below is how I currently see the landscape.

| Strategy                                     | Teacher                       | Target Label                        | Feature Type                                     | ML Task                   | Online Cost         | MIS/MVC General?          | Engineering Effort              | Scientific Value | My Assessment                                                            |
| -------------------------------------------- | ----------------------------- | ----------------------------------- | ------------------------------------------------ | ------------------------- | ------------------- | ------------------------- | ------------------------------- | ---------------- | ------------------------------------------------------------------------ |
| **A. CPAIOR Reproduction**                   | ReduMIS / PACE solver         | Final membership (0/1)              | Static handcrafted (MWUA + PageRank + LP + Luby) | Binary classification     | Very low            | ❌ Mostly MIS/LTP-specific | High (need ReduMIS integration) | Medium           | Useful only as historical baseline.                                      |
| **B. Exact MILP Membership**                 | Gurobi/CPLEX exact solve      | Final membership (0/1)              | Static handcrafted                               | Binary classification     | Very low            | ⚠️ Problem-specific       | Low                             | Low              | Solves wrong problem (LTP, not LtB).                                     |
| **C. Full Strong Branching**                 | SCIP/Gurobi Strong Branching  | Branching score for every candidate | Static + optional local                          | Regression                | Low (student model) | ✅                         | Very High                       | Very High        | Gold standard, but difficult to implement and collect.                   |
| **D. Strong Branching Top-1**                | SCIP/Gurobi Strong Branching  | Best branching variable only        | Static + optional local                          | Multiclass classification | Very low            | ✅                         | Moderate                        | Medium           | Symb4CO style. Brittle due to information loss.                          |
| **E. Pairwise Ranking (Khalil)**             | Strong Branching              | Positive/negative variable pairs    | Static + optional local                          | Pairwise ranking          | Low                 | ✅                         | High                            | High             | Proven approach, but dataset generation is complex.                      |
| **F. Continuous Branching Score Regression** | SCIP/Gurobi teacher heuristic | Real-valued branching score         | Static + cheap local                             | Regression                | Low                 | ✅                         | Moderate                        | **Very High**    | My current favorite. Preserves information and avoids top-1 brittleness. |
| **G. Behavioral Cloning of SCIP**            | SCIP default branching        | Variable chosen by SCIP             | Static + cheap local                             | Classification            | Very low            | ✅                         | Moderate                        | High             | Attractive if Strong Branching scores are inaccessible.                  |
| **H. Pseudocost Regression**                 | SCIP pseudocost heuristic     | Pseudocost value                    | Static + cheap local                             | Regression                | Low                 | ✅                         | Low                             | Medium           | Easier than Strong Branching; weaker teacher.                            |
| **I. Symbolic Branching (Symb4CO style)**    | Strong Branching              | Top-1 or score                      | Static + dynamic                                 | Symbolic regression       | Extremely low       | ✅                         | Very High                       | High             | Interesting, but symbolic search itself is a research project.           |
| **J. RL (Learning What to Defer style)**     | Self-supervised / PPO         | Reward from solution quality        | Static + dynamic                                 | Reinforcement learning    | Medium              | ⚠️                        | Very High                       | Medium           | Heavy training cost; not aligned with proposal.                          |
| **K. MWUA Snapshot Lifetime Study**          | SCIP/Gurobi teacher           | Branching score + node depth        | Static root features + tiny local updates        | Regression/analysis       | Low                 | ✅                         | Moderate                        | **Very High**    | Directly matches your proposal.                                          |

---

# Comparison of Target Labels

| Label Type                 | Example                    | Advantages                                   | Drawbacks                                    |
| -------------------------- | -------------------------- | -------------------------------------------- | -------------------------------------------- |
| Solution Membership        | (y_v \in {0,1})            | Easy to understand; CPAIOR-compatible        | Wrong objective for branching                |
| Top-1 Branch Variable      | (y=\arg\max \text{score})  | Simple training                              | Very noisy; loses ranking information        |
| Pairwise Preference        | (v_i \succ v_j)            | Robust; Khalil validated                     | Large datasets; pair construction complexity |
| Continuous Branching Score | (y_v \in \mathbb{R})       | Rich information; naturally supports ranking | Need access to teacher scores                |
| Pseudocost                 | (y_v=\text{pseudocost}(v)) | Easy to obtain                               | Teacher weaker than Strong Branching         |
| SCIP Decision              | Chosen variable only       | Practical if internals limited               | Inherits top-1 limitations                   |

---

# Comparison of Teacher Solvers

| Teacher       | Access to Branching Internals | MIS                  | MVC | Open Research Use            | Difficulty                     |
| ------------- | ----------------------------- | -------------------- | --- | ---------------------------- | ------------------------------ |
| ReduMIS       | ❌ No                          | ✅                    | ❌   | Moderate                     | High integration effort        |
| KaMIS         | ⚠️ Potentially                | ✅                    | ❌   | High                         | Unknown instrumentation effort |
| SCIP          | ✅ Excellent                   | ✅ (MILP formulation) | ✅   | **Excellent**                | Moderate                       |
| Gurobi        | ⚠️ Some callbacks             | ✅                    | ✅   | Excellent (academic license) | Moderate                       |
| CPLEX         | ⚠️ Some callbacks             | ✅                    | ✅   | Excellent                    | Moderate                       |
| Homegrown B&B | ✅ Full control                | ✅                    | ✅   | Unlimited                    | Extremely high effort          |

---

# Comparison of Feature Strategies

| Feature Strategy                  | Recomputed Every Node? | Cost      | Matches Proposal? |
| --------------------------------- | ---------------------- | --------- | ----------------- |
| Full dynamic GNN (Gasse et al.)   | Yes                    | Very high | ❌                 |
| Dynamic handcrafted features      | Yes                    | Moderate  | ⚠️                |
| CPAIOR static features only       | No                     | Very low  | ✅                 |
| Static + cheap local updates      | Minimal                | Low       | **✅ Exactly**     |
| Symbolic feature subset (Symb4CO) | Partial                | Very low  | ✅                 |

---

# Mapping to Your Dissertation

| Proposal Statement                                            | Matching Strategy                                     |
| ------------------------------------------------------------- | ----------------------------------------------------- |
| Compute expensive global information once                     | Static MWUA features                                  |
| Avoid repeated feature recomputation                          | Static + cheap local updates                          |
| Lightweight online model                                      | RF / XGBoost / small MLP / symbolic expression        |
| Study usefulness of root snapshot down the tree               | Node-depth-aware regression analysis                  |
| Switch to fallback heuristic when static information degrades | Hybrid deployment                                     |
| Support MIS and MVC                                           | Generic MILP teacher (SCIP/Gurobi/CPLEX), not ReduMIS |

---

# My Current Ranking

| Rank     | Strategy                                                            | Reasoning                                                                    |
| -------- | ------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| 🥇 **1** | **Continuous branching score regression using SCIP/Gurobi teacher** | Preserves information, generalizes to MIS/MVC, aligns with proposal.         |
| 🥈 **2** | **MWUA snapshot lifetime study (score + depth)**                    | This is arguably the actual scientific contribution of your dissertation.    |
| 🥉 **3** | **Behavioral cloning of SCIP default branching**                    | Practical fallback if true branching scores are inaccessible.                |
| 4        | Pairwise ranking (Khalil)                                           | Proven, but heavier engineering.                                             |
| 5        | Strong Branching top-1 (Symb4CO style)                              | Simpler, but information-poor and potentially noisy.                         |
| 6        | CPAIOR reproduction with ReduMIS labels                             | Useful as background, but not aligned with LtB.                              |
| 7        | RL / PPO approaches                                                 | Interesting academically, but too far from your proposal and infrastructure. |

---

# What I would pursue today

If I had to commit to one roadmap, it would be:

```text
Graph (MIS or MVC)
        ↓
Compute 15 static MWUA + structural features once
        ↓
Run SCIP / Gurobi / CPLEX as offline teacher
        ↓
Collect branching utility score (or best available proxy)
        ↓
Train lightweight regression model
        ↓
Deploy using only static features + a few cheap local updates
        ↓
Measure how predictive the root MWUA snapshot remains as depth increases
        ↓
Switch to fallback heuristic beyond the useful horizon
```

I think this is the strategy that is simultaneously:

1. **closest to your written proposal,**
2. **general enough for both MIS and MVC,**
3. **less brittle than top-1 imitation,**
4. **more original than reproducing CPAIOR or Symb4CO directly**, and
5. **practically feasible without implementing your own Strong Branching engine.**
