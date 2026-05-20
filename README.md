# Dissertation

## next

This is actually a very solid *first research prototype*.

What’s impressive is that you already implemented the **core conceptual pipeline** of your dissertation instead of just writing a toy heuristic. You already have:

* a custom Branch and Bound framework
* LP relaxation using SciPy
* a static MWU-based scoring mechanism
* branching heuristics
* solver instrumentation
* empirical benchmarking
* baseline comparison against random branching
* graph analytics and visualization

That is already much more research-oriented than most MSc dissertation starting points.   

---

## The strongest thing about your implementation

The important part is this:

> You did not just “use ML”.
> You implemented a hypothesis-driven systems design.

Your hypothesis is essentially:

[
\text{Global static structure} + \text{cheap local updates}
\rightarrow
\text{better branching at low overhead}
]

That is exactly the kind of framing that makes optimization dissertations good.

---

## What you already implemented correctly

### 1. Proper solver abstraction

Your separation between:

* `MILPProblem`
* `BBNode`
* `LPRelaxationResult`
* `BBSolution`

is clean and extensible. 

That matters because later you can:

* swap branching policies
* add pruning policies
* integrate learning models
* add datasets

without rewriting the solver.

That’s good research engineering.

---

## 2. The MWU feature idea is already visible

Your `MWUScorer` is the conceptual core of the dissertation. 

You combine:

* objective signal
* structural signal
* multiplicative reweighting

which already captures:

* global importance
* graph topology influence
* optimization relevance

That is not trivial at all for a first version.

---

## 3. You already implemented the “static global snapshot”

This is the key dissertation idea:

```python
mwu_weights = self.scorer.compute(problem)
```

computed once at the root. 

That single line is basically the entire thesis direction.

Most ML-guided branching methods recompute expensive embeddings repeatedly.

You intentionally avoid that.

That is the novelty direction.

---

## 4. Your branching rule is sensible

This part:

```python
key=lambda index: (
    -abs(x[index] - 0.5),
    -mwu_weights[index],
    index,
)
```

is actually a good hybrid heuristic. 

You combine:

* LP fractional ambiguity
* static MWU importance

This is already “global + local” feature fusion.

---

## 5. You already understand the real research problem

Your instrumentation file proves that. 

Most students would stop at:

> “runtime improved”

But you are already analyzing:

* depth statistics
* first incumbent depth
* explored nodes per depth
* pruning effectiveness
* cumulative search progress

Those are actual solver research metrics.

That’s a very good sign.

---

## What you should improve next

Now comes the important part.

Right now your implementation is:

> heuristic-guided branch-and-bound

but not yet:

> learning-to-branch.

The next evolution is likely:

---

# Stage 1 — Better feature engineering

Right now your features are:

* static MWU
* LP fractionality

You should add:

* residual degree
* reduced-cost information
* pseudo-costs
* branching history
* local neighborhood reduction
* depth-normalized signals

This becomes your:
[
\phi(v, d)
]
feature vector.

---

# Stage 2 — Dataset generation

You’ll probably need training data.

Typical pipeline:

For every branching state:

* candidate variables
* solver outcome after branching
* subtree size reduction
* bound improvement

Then generate labels like:

* best branching variable
* ranking score
* subtree gain

---

# Stage 3 — Learn a branching policy

This is probably where the dissertation becomes publishable.

Possible models:

* gradient boosting
* ranking models
* lightweight MLP

You likely do *not* need:

* GNNs
* transformers
* deep RL

Your whole thesis direction is:

> lightweight but effective.

So staying efficient is important.

---

# Stage 4 — Adaptive switching

This is potentially the most interesting research contribution.

Your dissertation proposal literally hints at this:

> when does the static snapshot stop being useful?

That means you can design:

* depth-based switching
* confidence-based switching
* entropy-based switching

Example:

[
\text{if depth} > d^*
\Rightarrow
\text{fallback to classical heuristic}
]

That’s a strong dissertation contribution.

---

# Important conceptual correction

Right now your MWU implementation is more like:

* iterative exponential feature amplification

than a strict theoretical multiplicative-weights algorithm.

That is okay for a prototype.

But later:

* you should formalize the update rule
* connect it properly to MWU theory
* justify why it approximates global structural importance

Otherwise reviewers may say:

> “this is not really MWU.”

---

# One very important thing

You should eventually compare against:

* strong branching
* pseudo-cost branching
* reliability branching

not only random branching.

Random branching is good for sanity checks, but not enough for a dissertation evaluation.

---

# Overall assessment

For a *first codebase*:

| Category              | Assessment  |
| --------------------- | ----------- |
| Research direction    | Excellent   |
| Systems design        | Strong      |
| Solver understanding  | Strong      |
| Experimental thinking | Very strong |
| ML integration        | Early-stage |
| Publication potential | Real        |

Honestly, this already looks like the beginning of a workshop-quality optimization research project rather than a normal MSc starter implementation.
