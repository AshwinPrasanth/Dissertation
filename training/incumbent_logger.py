from pyscipopt import Eventhdlr


class IncumbentLogger(Eventhdlr):

    def __init__(self):
        self.history = []

    def eventinit(self):

        self.model.catchEvent(
            "BESTSOLFOUND",
            self
        )


    def eventexit(self):

        self.model.dropEvent(
            "BESTSOLFOUND",
            self
        )


    def eventexec(self, event):

        self.history.append(
            (
                self.model.getSolvingTime(),
                self.model.getObjVal()
            )
        )