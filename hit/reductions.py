'''from dataclasses import dataclass
import networkx as nx
import time


@dataclass
class ReductionStats:

    original_vertices: int
    original_edges: int

    kernel_vertices: int
    kernel_edges: int

    degree0_removed: int
    degree1_removed: int

    runtime: float


class ReductionEngine:

    def reduce(self, G):

        start = time.time()

        G = G.copy()
        original_n = G.number_of_nodes()
        original_m = G.number_of_edges()

        degree0_removed = 0
        degree1_removed = 0

        changed = True

        while changed:

            changed = False

            c = self.degree_zero(G)

            if c > 0:

                degree0_removed += c
                changed = True

            c = self.degree_one(G)

            if c > 0:

                degree1_removed += c
                changed = True

        stats = ReductionStats(

            original_vertices = original_n,
            original_edges = original_m,

            kernel_vertices=G.number_of_nodes(),
            kernel_edges=G.number_of_edges(),

            degree0_removed=degree0_removed,
            degree1_removed=degree1_removed,

            runtime=time.time() - start,
        )

        return G, stats
    
    def degree_zero(self, G):

        isolated = [ v for v in G.nodes() if G.degree(v) == 0]

        if isolated:
            G.remove_nodes_from(isolated)

        return len(isolated)
    
    def degree_one(self, G):

        removed = 0

        while True:

            leaves = [ v for v in G.nodes() if G.degree(v) == 1]

            if not leaves:
                break

            leaf = leaves[0]

            if leaf not in G:
                continue

            nbr = next(iter(G.neighbors(leaf)))

            G.remove_node(leaf)

            if nbr in G:
                G.remove_node(nbr)
            removed += 2
        return removed'''
    

from dataclasses import dataclass
import networkx as nx
import time


@dataclass
class ReductionStats:

    original_vertices: int
    original_edges: int

    kernel_vertices: int
    kernel_edges: int

    degree0_removed: int
    degree1_removed: int
    forced_vertices: int

    runtime: float


class ReductionEngine:

    def __init__(self):

        self.degree0_removed = 0
        self.degree1_removed = 0
        self.forced_cover = []

    def degree_zero(self, G):

        isolated = [
            v
            for v in list(G.nodes())
            if G.degree(v) == 0
        ]

        if isolated:

            G.remove_nodes_from(isolated)

        self.degree0_removed += len(isolated)

        return len(isolated)
    
    def degree_one(self, G):

        changed = 0

        while True:

            leaves = [ v for v in G.nodes() if G.degree(v) == 1]

            if not leaves:

                break

            leaf = leaves[0]

            if leaf not in G:
                continue

            neighbour = next(iter(G.neighbors(leaf)))

            #
            # MVC Rule:
            #
            # neighbour MUST belong
            # to every minimum cover
            #

            self.forced_cover.append(neighbour)

            #
            # Remove both
            #

            if neighbour in G:

                G.remove_node(neighbour)

            if leaf in G:

                G.remove_node(leaf)

            self.degree1_removed += 2

            changed += 2

        return changed

    def reduce(self, G):

        start = time.time()

        G = G.copy()

        original_vertices = G.number_of_nodes()
        original_edges = G.number_of_edges()

        changed = True

        while changed:

            changed = False

            if self.degree_zero(G) > 0:
                changed = True
            if self.degree_one(G) > 0:
                changed = True
        
        #
        # Relabel vertices to 0...n-1
        #

        mapping = {old: new for new, old in enumerate(G.nodes())}
        self.mapping = mapping
        self.reverse_mapping = { new: old for old, new in mapping.items()}

        G = nx.relabel_nodes(G,mapping,)
        
        stats = ReductionStats(

            original_vertices=original_vertices,
            original_edges=original_edges,

            kernel_vertices=G.number_of_nodes(),
            kernel_edges=G.number_of_edges(),

            degree0_removed=self.degree0_removed,
            degree1_removed=self.degree1_removed,
            forced_vertices=len(self.forced_cover),
            runtime=time.time() - start,
        )

        return G, stats