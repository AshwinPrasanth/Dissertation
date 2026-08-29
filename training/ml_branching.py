from pyscipopt import Branchrule, SCIP_RESULT
import numpy as np
import torch


class MLBranchRule(Branchrule):

    def __init__(
        self,
        ml_model,
        feature_builder,
        device
    ):
        super().__init__()

        self.ml_model = ml_model
        self.feature_builder = feature_builder
        self.device = device

        self.max_ml_depth = 5
        self.calls = 0


    def branchexeclp(
        self,
        allowaddcons
    ):

        depth = self.model.getDepth()

        if depth > self.max_ml_depth:
            return {
                "result": SCIP_RESULT.DIDNOTRUN
            }


        cands, candssol, _, ncands, npriocands, _ = (
            self.model.getLPBranchCands()
        )


        if ncands == 0:
            return {
                "result": SCIP_RESULT.DIDNOTRUN
            }


        self.calls += 1

        if self.calls % 100 == 0:
            print(
                "ML branch calls:",
                self.calls,
                "depth:",
                depth
            )


        current_graph = self.get_current_graph()

        self.feature_builder.update_dynamic(
            current_graph
        )


        features = []
        vars_ = []


        for i in range(npriocands):

            var = cands[i]

            idx = (
                self.feature_builder
                .variable_index(var)
            )

            if idx is None:
                continue


            features.append(
                self.feature_builder.features(
                    idx,
                    float(candssol[i])
                )
            )

            vars_.append(var)


        if not vars_:
            return {
                "result": SCIP_RESULT.DIDNOTRUN
            }


        x = torch.tensor(
            np.asarray(features),
            dtype=torch.float32,
            device=self.device
        )


        with torch.no_grad():

            scores = (
                self.ml_model(x)
                .cpu()
                .numpy()
            )


        best = int(
            np.argmax(scores)
        )


        self.model.branchVar(
            vars_[best]
        )


        return {
            "result": SCIP_RESULT.BRANCHED
        }



    def get_current_graph(self):

        import networkx as nx

        G = nx.Graph()

        active = []


        for var in self.model.getVars():

            idx = (
                self.feature_builder
                .variable_index(var)
            )

            if idx is not None:
                active.append(idx)


        G.add_nodes_from(active)


        for u, v in self.feature_builder.graph.edges():

            if u in active and v in active:

                G.add_edge(
                    u,
                    v
                )


        return G