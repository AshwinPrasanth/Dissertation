from dataclasses import dataclass, field
import time


@dataclass
class SolverStatistics:

    start_time: float = field(default_factory=time.time)

    nodes_processed: int = 0

    branches: int = 0

    max_depth: int = 0

    first_incumbent_time: float | None = None

    best_objective: float = float("-inf")

    incumbent_updates: list = field(default_factory=list)

    runtime: float = 0.0

    def node(self, depth):

        self.nodes_processed += 1

        self.max_depth = max(
            self.max_depth,
            depth,
        )

    def branch(self):

        self.branches += 1

    def incumbent(self, obj):

        now = time.time() - self.start_time

        self.incumbent_updates.append(
            (now, obj)
        )

        if self.first_incumbent_time is None:

            self.first_incumbent_time = now

        if obj > self.best_objective:

            self.best_objective = obj

    def finish(self):

        self.runtime = (
            time.time()
            - self.start_time
        )