'''import numpy as np

from pyscipopt import (
    Branchrule,
    SCIP_RESULT,
)

from research.branch_sample import BranchSample
from research.dataset_writer import DatasetWriter

def scip_branch_score(down, up):
    """
    Approximation of SCIP's default branching score.

    down, up are dual bound gains returned by strong branching.
    """

    if down > up:
        down, up = up, down

    return down + 0.1 * up

class SCIPDatasetCollector(Branchrule):

    def __init__(
        self,
        dataset,
        mwua_scores,
        graph_name,
    ):

        super().__init__()

        self.dataset = dataset
        self.mwua_scores = mwua_scores
        self.graph_name = graph_name

        self.writer = DatasetWriter(graph_name)

        self.call_count = 0

    def branchexeclp(self, allowaddcons):

        (
            cands,
            candssol,
            candsscore,
            ncands,
            npriocands,
            nfracimplvars,
        ) = self.model.getLPBranchCands()

        if ncands == 0:
            return {
                "result": SCIP_RESULT.DIDNOTRUN
            }

        depth = self.model.getDepth()

        candidate_ids = []
        candidate_features = []
        candidate_mwua = []
        candidate_lp = []
        candidate_vars = []

        for var in cands:

            name = var.name

            if name.startswith("t_x_"):

                idx = int(
                    name.replace("t_x_", "")
                )

            elif name.startswith("x_"):

                idx = int(
                    name.replace("x_", "")
                )

            else:
                continue

            candidate_ids.append(idx)

            candidate_features.append(
                self.dataset.X[idx]
            )

            candidate_mwua.append(
                self.mwua_scores[idx]
            )

            candidate_lp.append(
                var.getLPSol()
            )

            candidate_vars.append(
                var
            )

        self.model.startStrongbranch()

        sb_scores = []

        best_score = -1e100
        best_var = None

        for i, var in enumerate(candidate_vars):

            (
                down,
                up,
                downvalid,
                upvalid,
                downinf,
                upinf,
                downconflict,
                upconflict,
                lperror,
            ) = self.model.getVarStrongbranch(
                var,
                1000,
            )



            if lperror:
                sb_scores.append(-1e20)
                continue

            if not downvalid:
                down = -1e20

            if not upvalid:
                up = -1e20

            score = scip_branch_score(down, up)


            sb_scores.append(score)

            if score > best_score:

                best_score = score
                best_var = var

        self.model.endStrongbranch()
           
        

        if best_var is None:

            return {
                "result": SCIP_RESULT.DIDNOTRUN
            }

        chosen = int(
            best_var.name
            .replace("t_x_", "")
            .replace("x_", "")
        )
        
        best_sb_score = best_score
        
        #chosen_position = candidate_ids.index(chosen)

        sample = BranchSample(

            graph_name=self.graph_name,

            node_number=self.call_count,

            depth=depth,

            candidate_ids=np.array(
                candidate_ids
            ),

            candidate_features=np.array(
                candidate_features
            ),

            mwua_scores=np.array(
                candidate_mwua
            ),

            lp_values=np.array(
                candidate_lp
            ),

            sb_scores=np.array(
                sb_scores
            ),

            chosen_variable=chosen,
            
            best_sb_score=best_sb_score,
            
            #chosen_position=chosen_position,
        )

        self.writer.save(sample)

        self.call_count += 1

        self.model.branchVar(best_var)

        return {
            "result": SCIP_RESULT.BRANCHED
        }'''
        
import numpy as np

from pyscipopt import (
    Branchrule,
    SCIP_RESULT,
)

from research.branch_sample import BranchSample
from research.dataset_writer import DatasetWriter


ITLIM = 100


def scip_branch_score(down, up):
    if down > up:
        down, up = up, down

    return down + 0.1 * up


class SCIPDatasetCollector(Branchrule):

    def __init__(
        self,
        dataset,
        mwua_scores,
        graph_name,
    ):

        super().__init__()

        self.dataset = dataset
        self.mwua_scores = mwua_scores
        self.graph_name = graph_name

        self.writer = DatasetWriter(graph_name)

        self.call_count = 0

    def branchexeclp(self, allowaddcons):

        (
            cands,
            candssol,
            candsscore,
            ncands,
            npriocands,
            nfracimplvars,
        ) = self.model.getLPBranchCands()

        if ncands == 0:
            return {
                "result": SCIP_RESULT.DIDNOTRUN
            }

        if self.call_count == 0:
            print(f"ncands = {ncands}")
            print(f"npriocands = {npriocands}")

        depth = self.model.getDepth()

        candidate_ids = []
        candidate_features = []
        candidate_mwua = []
        candidate_lp = []
        candidate_vars = []

        for var in cands:

            name = var.name

            if name.startswith("t_x_"):
                idx = int(name.replace("t_x_", ""))

            elif name.startswith("x_"):
                idx = int(name.replace("x_", ""))

            else:
                continue

            candidate_ids.append(idx)

            candidate_features.append(
                self.dataset.X[idx]
            )

            candidate_mwua.append(
                self.mwua_scores[idx]
            )

            candidate_lp.append(
                var.getLPSol()
            )

            candidate_vars.append(var)

        self.model.startStrongbranch()

        sb_scores = np.full(
            len(candidate_vars),
            -1e20,
        )

        best_score = -1e20
        best_var = None

        for i, var in enumerate(candidate_vars[:npriocands]):

            (
                down,
                up,
                downvalid,
                upvalid,
                downinf,
                upinf,
                downconflict,
                upconflict,
                lperror,
            ) = self.model.getVarStrongbranch(
                var,
                ITLIM,
            )

            if lperror:
                continue

            if not downvalid:
                down = -1e20

            if not upvalid:
                up = -1e20

            score = scip_branch_score(
                down,
                up,
            )

            sb_scores[i] = score

            if score > best_score:

                best_score = score
                best_var = var

        self.model.endStrongbranch()

        if best_var is None:

            return {
                "result": SCIP_RESULT.DIDNOTRUN
            }

        chosen = int(
            best_var.name
            .replace("t_x_", "")
            .replace("x_", "")
        )

        sample = BranchSample(

            graph_name=self.graph_name,

            node_number=self.call_count,

            depth=depth,

            candidate_ids=np.array(
                candidate_ids
            ),

            candidate_features=np.array(
                candidate_features
            ),

            mwua_scores=np.array(
                candidate_mwua
            ),

            lp_values=np.array(
                candidate_lp
            ),

            sb_scores=sb_scores,

            chosen_variable=chosen,

            best_sb_score=best_score,
        )

        self.writer.save(sample)

        self.call_count += 1

        self.model.branchVar(best_var)

        return {
            "result": SCIP_RESULT.BRANCHED
        }