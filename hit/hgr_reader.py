from typing import List


class HittingSetInstance:

    def __init__(
        self,
        num_vertices: int,
        hyperedges: List[List[int]],
    ):

        self.num_vertices = num_vertices
        self.hyperedges = hyperedges

        self.num_hyperedges = len(
            hyperedges
        )


def read_hgr(
    filename: str,
) -> HittingSetInstance:

    num_vertices = None
    num_hyperedges = None

    hyperedges = []

    with open(
        filename,
        "r",
    ) as f:

        for line in f:

            line = line.strip()

            if (
                not line
                or line.startswith("c")
            ):
                continue

            if line.startswith("p"):

                parts = line.split()

                if (
                    len(parts) != 4
                    or parts[1] != "hs"
                ):
                    raise ValueError(
                        "Invalid HGR header."
                    )

                num_vertices = int(
                    parts[2]
                )

                num_hyperedges = int(
                    parts[3]
                )

                continue

            edge = [
                int(v) - 1
                for v in line.split()
            ]

            hyperedges.append(edge)

    if num_vertices is None:
        raise ValueError(
            "Missing HGR header."
        )

    if (
        len(hyperedges)
        != num_hyperedges
    ):
        raise ValueError(
            "Incorrect number of hyperedges."
        )

    return HittingSetInstance(
        num_vertices,
        hyperedges,
    )
    
if __name__ == "__main__":

    instance = read_hgr(
        "sample.hgr"
    )

    print(instance.num_vertices)

    print(instance.num_hyperedges)

    print(instance.hyperedges[:5])