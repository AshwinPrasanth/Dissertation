import time


class SearchStatistics:

    def __init__(self):

        self.start_time = time.time()

        self.nodes_processed = 0
        self.branches = 0
        self.max_depth = 0

        self.depth_counts = {}

        self.runtime = 0.0

    def record_node(self, depth):

        self.nodes_processed += 1

        self.max_depth = max(
            self.max_depth,
            depth,
        )

        if depth not in self.depth_counts:
            self.depth_counts[depth] = 0

        self.depth_counts[depth] += 1

    def record_branch(self):

        self.branches += 1

    def finish(self):

        self.runtime = (
            time.time()
            - self.start_time
        )

    def summary(self):

        return {
            "nodes_processed":
                self.nodes_processed,

            "branches":
                self.branches,

            "max_depth":
                self.max_depth,

            "runtime":
                self.runtime,

            "depth_counts":
                self.depth_counts,
        }