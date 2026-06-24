from abc import ABC, abstractmethod
from pyscipopt import ( Branchrule,SCIP_RESULT,) # importing the base class for branching rules and the result enum
import numpy as np


class BranchingStrategy(ABC):
    """
    Base interface for branching rules.
    """

    @abstractmethod
    def select(self, lp_solution: np.ndarray) -> int:
        pass


class MostFractionalBranching(BranchingStrategy):
    """
    Classical branching rule.

    Choose the variable closest to 0.5.
    """

    def select(self, lp_solution: np.ndarray) -> int:

        fractional_vars = []

        for i, value in enumerate(lp_solution):

            if 1e-6 < value < 1 - 1e-6:
                fractional_vars.append(i)

        if not fractional_vars:
            return -1

        return min(
            fractional_vars,
            key=lambda i: abs(lp_solution[i] - 0.5)
        )
        
class MWUABranching(BranchingStrategy):

    def __init__(self, mwua_certainty):
        self.mwua_certainty = mwua_certainty

    def select(self, lp_solution):

        fractional = [
            i
            for i, x in enumerate(lp_solution)
            if 1e-6 < x < 1 - 1e-6
        ]

        if not fractional:
            return -1

        return max(
            fractional,
            key=lambda i:
            self.mwua_certainty[i]
        )

class DegreeBranching(BranchingStrategy):
# a simple heuristic: branch on the variable with highest degree in the constraint graph
    def __init__(self, degrees):
        self.degrees = degrees

    def select(self, lp_solution):

        fractional = [
            i
            for i, x in enumerate(lp_solution)
            if 1e-6 < x < 1 - 1e-6
        ]

        if not fractional:
            return -1

        return max(
            fractional,
            key=lambda i: self.degrees[i]
        )

class SCIPMWUABranchRule(Branchrule):

    def __init__(self, mwua_scores):
        self.mwua_scores = mwua_scores
        self.call_count = 0

        self.logfile = open(
        "results/mwua_branch_log.csv",
        "w"
    )

        self.logfile.write(
        "call,node_depth,var,score,num_candidates\n"
    )

    def branchexeclp(self, allowaddcons):

        (
            cands,
            candssol,
            candsscore,
            ncands,
            npriocands,
            nfracimplvars,
        ) = self.model.getLPBranchCands()
        self.call_count += 1

        print(f"\nCandidates: {ncands}")

        if ncands == 0:
            return {
                "result": SCIP_RESULT.DIDNOTRUN
            }

        best_var = None
        best_score = -1e20

        for var in cands:

            # SCIP transforms names:
            # x_7 -> t_x_7

            name = var.name

            if name.startswith("t_x_"):
                idx = int(name.replace("t_x_", ""))

            elif name.startswith("x_"):
                idx = int(name.replace("x_", ""))

            else:
                continue

            score = self.mwua_scores[idx]
            if idx < 5:
                print( f"idx={idx}, " f"score={score}")

            if score > best_score:
                best_score = score
                best_var = var

        #print(f"Branching on {best_var.name} "f"(MWUA={best_score:.4f})")
        depth = self.model.getDepth()
        DEPTH_LIMIT = 0

        if depth > DEPTH_LIMIT:
            return {
        "result": SCIP_RESULT.DIDNOTRUN
    }

        self.logfile.write(
    f"{self.call_count},"
    f"{depth},"
    f"{best_var.name},"
    f"{best_score},"
    f"{ncands}\n"
)

        self.logfile.flush()
        self.model.branchVar(best_var)

        return {
            "result": SCIP_RESULT.BRANCHED
        }
        
'''class SCIPMWUABranchRule(Branchrule):

    def __init__(self, mwua_scores):
        self.mwua_scores = mwua_scores

    def branchexeclp(self, allowaddcons):

        print("\n===== MWUA CALLBACK HIT =====")

        result = self.model.getLPBranchCands()

        print("Type:", type(result))

        try:
            print("Length:", len(result))
        except:
            print("No length")

        print("Result:")
        print(result)

        return {
            "result": SCIP_RESULT.DIDNOTRUN
        }

'''
      
'''import numpy as np
from branching import MostFractionalBranching
x = np.array([1.0,0.3,0.52,0.9,0.48,])
brancher = MostFractionalBranching()
print(brancher.select(x))'''
