# Week Plan — Graph Dataset Strategy for ML4CO (MIS / MVC)

## Objective

Establish a robust dataset pipeline for training and evaluating an ML-guided branching policy for Maximum Independent Set (MIS) / Vertex Cover (MVC).

The focus this week is identifying **hard graph distributions** where Branch-and-Bound (B&B) produces meaningful branching behavior for supervised learning.

---

## Why Not Use SNAP for Training?

Large real-world graph repositories such as:

- :contentReference[oaicite:0]{index=0} (SNAP)
- :contentReference[oaicite:1]{index=1}

are not suitable for primary training because:

- Graphs are often too large for Full Strong Branching label generation.
- Many instances are solved during pre-solving with minimal branching.
- Limited useful branching supervision for learning.

These will instead be used later for **generalization testing**.

---

## Dataset Strategy

### Tier 1 — Synthetic Training Graphs (Primary Focus)

Generate training graphs using NetworkX.

Families:

- Erdős-Rényi (G(n,p))
- Barabási-Albert
- Random Regular Graphs
- Random Geometric Graphs

Scale:

- 50 – 300 nodes

Purpose:

- Generate supervised branching labels
- Train Random Forest branching policy

---

### Tier 2 — Hard Benchmark Evaluation

Benchmark on established optimization datasets.

Sources:

- DIMACS benchmark suite  
- BHOSLIB benchmark suite

Examples:

- p_hat graphs  
- brock graphs  
- frb instances  

Purpose:

- Evaluate solver performance on known hard combinatorial instances

---

### Tier 3 — Real-World Generalization

Use sampled subgraphs only.

Sources:

- :contentReference[oaicite:2]{index=2}
- :contentReference[oaicite:3]{index=3}

Examples:

- ca-GrQc  
- wiki-Vote  

Purpose:

- Test out-of-distribution generalization

---

## SCIP Baseline Experiments

### 1. Phase Transition Search

Find graph parameters where SCIP tree size spikes.

Test:

**Erdős-Rényi**

- n = 100, 150
- p = 0.05 → 0.40

Target:

- Identify hardness region

Expected:

- p ≈ 0.12 – 0.22

---

### 2. Random Regular Graphs

Test:

- n = 100
- d = 3, 4, 5

Reason:

- Degree heuristics become ineffective
- Forces branching decisions on structural features

---

## SCIP Configurations

### Baseline A — Default SCIP

Purpose:

- Standard solver baseline

---

### Baseline B — Full Strong Branching

Setting:

- branching/fullstrong/priority = 999999

Purpose:

- Oracle / teacher labels

---

### Baseline C — No Presolving

Settings:

- presolving/maxrounds = 0
- propagating/maxrounds = 0

Purpose:

- Preserve raw graph structure

---

## Weekly Tasks

- [ ] Generate synthetic graph families with NetworkX  
- [ ] Build SCIP experiment pipeline  
- [ ] Run Erdős-Rényi parameter sweep  
- [ ] Identify phase transition hardness region  
- [ ] Test Random Regular Graph baselines  
- [ ] Download DIMACS benchmark instances  
- [ ] Download BHOSLIB benchmark instances  
- [ ] Create separate SNAP test suite for later evaluation  

---

## Success Criterion

A useful training graph should produce:

- >100 branch-and-bound nodes under default SCIP
- Significant difference between Default SCIP and Full Strong Branching
- Non-trivial branching decisions for supervised learning
