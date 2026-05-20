from __future__ import annotations

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

try:
    from Lit.bnb_solver import BranchAndBoundSolver, build_problem
except ModuleNotFoundError:
    from bnb_solver import BranchAndBoundSolver, build_problem


def build_mvc_problem(graph: nx.Graph):
    n_vars = graph.number_of_nodes()
    n_edges = graph.number_of_edges()

    c = np.ones(n_vars, dtype=float)
    A_ub = np.zeros((n_edges, n_vars), dtype=float)
    b_ub = -np.ones(n_edges, dtype=float)

    for edge_idx, (u, v) in enumerate(graph.edges()):
        A_ub[edge_idx, u] = -1.0
        A_ub[edge_idx, v] = -1.0

    return build_problem(c=c, A_ub=A_ub, b_ub=b_ub)


def save_primary_graph_visuals(graph: nx.Graph, out_dir: Path) -> None:
    pos_spring = nx.spring_layout(graph, seed=42)
    pos_kamada = nx.kamada_kawai_layout(graph)
    pos_circular = nx.circular_layout(graph)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, title, pos in [
        (axes[0], "Spring Layout", pos_spring),
        (axes[1], "Kamada-Kawai Layout", pos_kamada),
        (axes[2], "Circular Layout", pos_circular),
    ]:
        nx.draw_networkx(
            graph,
            pos=pos,
            ax=ax,
            with_labels=True,
            node_color="#ffd166",
            edge_color="#3a506b",
            node_size=700,
            font_size=9,
        )
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("Minimum Vertex Cover Test Graph: Multiple Visual Styles", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "graph_layouts_comparison.png", dpi=220)
    plt.close(fig)

    degrees = [deg for _, deg in graph.degree()]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(degrees, bins=np.arange(min(degrees), max(degrees) + 2) - 0.5, color="#ef476f", edgecolor="white")
    ax.set_title("Degree Distribution")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(out_dir / "degree_distribution.png", dpi=220)
    plt.close(fig)

    adjacency = nx.to_numpy_array(graph, nodelist=sorted(graph.nodes()))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(adjacency, cmap="Blues", interpolation="nearest")
    ax.set_title("Adjacency Matrix Heatmap")
    ax.set_xlabel("Node Index")
    ax.set_ylabel("Node Index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "adjacency_heatmap.png", dpi=220)
    plt.close(fig)

    centrality = nx.degree_centrality(graph)
    nodes_sorted = sorted(centrality.keys(), key=lambda n: centrality[n], reverse=True)
    top_nodes = nodes_sorted[:10]
    top_values = [centrality[n] for n in top_nodes]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([str(n) for n in top_nodes], top_values, color="#118ab2")
    ax.set_title("Top Degree Centrality Nodes")
    ax.set_xlabel("Node")
    ax.set_ylabel("Degree Centrality")
    fig.tight_layout()
    fig.savefig(out_dir / "centrality_bar.png", dpi=220)
    plt.close(fig)


def run_scaling_benchmark(out_dir: Path):
    sizes = [10, 15, 20, 25, 30, 40, 50, 60, 80, 100]
    p = 0.4
    seed_base = 42

    rows = []
    for i, n in enumerate(sizes):
        graph = nx.gnp_random_graph(n=n, p=p, seed=seed_base + i)
        problem = build_mvc_problem(graph)
        solver = BranchAndBoundSolver(
            tolerance=1e-5,
            mwu_learning_rate=0.35,
            mwu_iterations=4,
            max_nodes=300000,
        )

        t0 = perf_counter()
        solution = solver.solve(problem)
        dt = perf_counter() - t0

        rows.append(
            {
                "n": n,
                "edges": graph.number_of_edges(),
                "density": nx.density(graph),
                "status": solution.status,
                "objective": solution.objective_value,
                "explored": solution.explored_nodes,
                "pruned": solution.pruned_nodes,
                "prune_ratio": (solution.pruned_nodes / solution.explored_nodes) if solution.explored_nodes else 0.0,
                "time_sec": dt,
            }
        )

    csv_path = out_dir / "scaling_benchmark.csv"
    header = ["n", "edges", "density", "status", "objective", "explored", "pruned", "prune_ratio", "time_sec"]
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(
                f"{row['n']},{row['edges']},{row['density']:.6f},{row['status']},{row['objective']},"
                f"{row['explored']},{row['pruned']},{row['prune_ratio']:.6f},{row['time_sec']:.6f}\n"
            )

    return rows


def save_scaling_plots(rows, out_dir: Path) -> None:
    n = [row["n"] for row in rows]
    edges = [row["edges"] for row in rows]
    explored = [row["explored"] for row in rows]
    pruned = [row["pruned"] for row in rows]
    prune_ratio = [row["prune_ratio"] for row in rows]
    time_sec = [row["time_sec"] for row in rows]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].plot(n, time_sec, marker="o", color="#06d6a0")
    axes[0, 0].set_title("Runtime vs Nodes")
    axes[0, 0].set_xlabel("Number of Graph Nodes")
    axes[0, 0].set_ylabel("Runtime (s)")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(n, explored, marker="o", color="#ef476f", label="Explored")
    axes[0, 1].plot(n, pruned, marker="s", color="#118ab2", label="Pruned")
    axes[0, 1].set_title("Search Tree Size")
    axes[0, 1].set_xlabel("Number of Graph Nodes")
    axes[0, 1].set_ylabel("Node Count")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(n, prune_ratio, marker="o", color="#8338ec")
    axes[1, 0].set_title("Pruning Ratio")
    axes[1, 0].set_xlabel("Number of Graph Nodes")
    axes[1, 0].set_ylabel("Pruned / Explored")
    axes[1, 0].set_ylim(0.0, 1.0)
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].scatter(edges, time_sec, c=n, cmap="viridis", s=70)
    axes[1, 1].set_title("Runtime vs Edge Count")
    axes[1, 1].set_xlabel("Number of Edges")
    axes[1, 1].set_ylabel("Runtime (s)")
    axes[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "scaling_dashboard.png", dpi=220)
    plt.close(fig)


def run_density_sweep(out_dir: Path) -> None:
    n = 60
    densities = [0.10, 0.20, 0.30, 0.40, 0.50]
    seed = 7
    runtimes = []
    explored_counts = []

    for i, p in enumerate(densities):
        graph = nx.gnp_random_graph(n=n, p=p, seed=seed + i)
        problem = build_mvc_problem(graph)
        solver = BranchAndBoundSolver(
            tolerance=1e-5,
            mwu_learning_rate=0.35,
            mwu_iterations=4,
            max_nodes=250000,
        )
        t0 = perf_counter()
        solution = solver.solve(problem)
        dt = perf_counter() - t0
        runtimes.append(dt)
        explored_counts.append(solution.explored_nodes)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(densities, runtimes, marker="o", color="#ff7f11", label="Runtime (s)")
    ax1.set_xlabel("Graph Density p")
    ax1.set_ylabel("Runtime (s)", color="#ff7f11")
    ax1.tick_params(axis="y", labelcolor="#ff7f11")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(densities, explored_counts, marker="s", color="#004e89", label="Explored Nodes")
    ax2.set_ylabel("Explored B&B Nodes", color="#004e89")
    ax2.tick_params(axis="y", labelcolor="#004e89")

    fig.suptitle("Difficulty vs Graph Density (n=60)")
    fig.tight_layout()
    fig.savefig(out_dir / "density_sweep.png", dpi=220)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent
    analysis_dir = root / "analysis_output"
    plot_dir = analysis_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    graphml_path = analysis_dir / "mvc_graph.graphml"
    if not graphml_path.exists():
        raise FileNotFoundError(
            f"Expected graph file at {graphml_path}. Run bnb_solver.py first to generate it."
        )

    graph = nx.read_graphml(graphml_path)
    mapping = {node: int(node) if str(node).isdigit() else node for node in graph.nodes()}
    graph = nx.relabel_nodes(graph, mapping)

    save_primary_graph_visuals(graph, plot_dir)
    rows = run_scaling_benchmark(plot_dir)
    save_scaling_plots(rows, plot_dir)
    run_density_sweep(plot_dir)

    report_path = plot_dir / "analysis_report.txt"
    with report_path.open("w", encoding="utf-8") as report:
        report.write("Generated analysis artifacts\n")
        report.write("- graph_layouts_comparison.png\n")
        report.write("- degree_distribution.png\n")
        report.write("- adjacency_heatmap.png\n")
        report.write("- centrality_bar.png\n")
        report.write("- scaling_dashboard.png\n")
        report.write("- density_sweep.png\n")
        report.write("- scaling_benchmark.csv\n")

    print(f"Saved plots to: {plot_dir}")
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()