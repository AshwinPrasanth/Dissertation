@dataclass
class ReductionStats:

    original_vertices: int
    original_edges: int

    kernel_vertices: int
    kernel_edges: int

    degree0_removed: int
    degree1_removed: int
    twin_removed: int
    domination_removed: int
    folded: int

    runtime: float
    
