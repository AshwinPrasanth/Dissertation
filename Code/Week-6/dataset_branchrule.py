import numpy as np

from pyscipopt import (
    Branchrule,
    SCIP_RESULT,
)

from research.branch_sample import BranchSample
from research.dataset_writer import DatasetWriter


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

            candidate_features.append(self.dataset.X[idx])

            candidate_mwua.append(self.mwua_scores[idx])
            candidate_lp.append(var.getLPSol())
            candidate_vars.append(var)
            
        #best_var = cands[0]
        self.model.startStrongbranch()
        sb_scores = []
        best_score = -1e100
        best_var = None
        
        for var in candidate_vars:

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
                continue
            
            if downinf:
                down = self.model.getObjVal() + 1e20

            if upinf:
                up = self.model.getObjVal() + 1e20

            score = down + up

            sb_scores.append(score)

            if score > best_score:

                best_score = score

                best_var = var

        self.model.endStrongbranch()
        if best_var is None:
            self.model.endStrongbranch()
            return {
                "result": SCIP_RESULT.DIDNOTRUN
            }
        chosen = best_var.name

        chosen = (
            chosen
            .replace("t_x_", "")
            .replace("x_", "")
        )

        sample = BranchSample(

            graph_name=self.graph_name,

            depth=depth,

            candidate_ids=np.array(candidate_ids),

            candidate_features=np.array(
                candidate_features
            ),

            mwua_scores=np.array(
                candidate_mwua
            ),

            node_number=self.call_count,

            lp_values=np.array(candidate_lp),

            sb_scores=np.array(sb_scores),

            chosen_variable=int(chosen),
                    )

        self.writer.save(sample)
        self.call_count += 1
        self.model.branchVar(best_var)

        return {
            "result": SCIP_RESULT.BRANCHED
        }