import networkx as nx


def generate_er_graph(
    n: int,
    p: float,
    seed: int = 0,
):
    return nx.erdos_renyi_graph(
        n=n,
        p=p,
        seed=seed,
    )


def generate_ba_graph(
    n: int,
    m: int,
    seed: int = 0,
):
    return nx.barabasi_albert_graph(
        n=n,
        m=m,
        seed=seed,
    )


def generate_regular_graph(
    n: int,
    d: int,
    seed: int = 0,
):
    return nx.random_regular_graph(
        d=d,
        n=n,
        seed=seed,
    )


def generate_geometric_graph(
    n: int,
    radius: float,
    seed: int = 0,
):
    return nx.random_geometric_graph(
        n=n,
        radius=radius,
        seed=seed,
    )


def generate_small_world_graph(
    n: int,
    k: int,
    p: float,
    seed: int = 0,
):
    return nx.watts_strogatz_graph(
        n=n,
        k=k,
        p=p,
        seed=seed,
    )