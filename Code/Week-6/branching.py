from abc import ABC, abstractmethod
from pyscipopt import ( Branchrule,SCIP_RESULT,) # importing the base class for branching rules and the result enum
import numpy as np
from research.branch_sample import BranchSample
from research.dataset_writer import DatasetWriter


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

    def __init__(self,dataset, mwua_scores):
        self.mwua_scores = mwua_scores
        self.dataset = dataset
        self.writer = DatasetWriter("dataset/frb30-15-1")
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
        candidate_ids = []
        candidate_features = []
        candidate_mwua = []
    

        for var in cands:

            # SCIP transforms names:
            # x_7 -> t_x_7

            name = var.name

            if name.startswith("t_x_"):
                idx = int(name.replace("t_x_", ""))

                candidate_ids.append(idx)
                candidate_features.append(self.dataset.X[idx])

                candidate_mwua.append(self.mwua_scores[idx])

            elif name.startswith("x_"):
                idx = int(name.replace("x_", ""))
     
                candidate_ids.append(idx)
                candidate_features.append(self.dataset.X[idx])

                candidate_mwua.append(self.mwua_scores[idx])

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
        sample = BranchSample( graph_name="frb30-15-1", depth=depth, candidate_ids=np.array(candidate_ids), candidate_features=np.array(candidate_features), 
                              mwua_scores=np.array(candidate_mwua),chosen_variable=int(best_var.name.replace("t_x_", "").replace("x_", "")),)

        self.writer.save(sample)
        
        self.model.branchVar(best_var)

        return {
            "result": SCIP_RESULT.BRANCHED
        }
        