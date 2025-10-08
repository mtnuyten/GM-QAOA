"""
Grover Mixer QAOA Variance Experiments

This module provides functions for constructing QAOA circuits with both
standard and Grover mixers. For computing empirical and theoretical
variance of the loss function corresponding the the MaxCut problem across different graphs, system sizes,
and depths. Includes examples at the bottom with seeding for reproducibility.

Author: Matthew Nuyten
Date: 2025-09-24
"""


import matplotlib.pyplot as plt
import networkx as nx
import pennylane as qml
import numpy as np

import time
import itertools as it
from typing import List, Tuple, Dict

import os


### Variational Quantum Algorithms ###

# ================================================================
# Hamiltonians
# ================================================================

def maxcut_hamiltonian(n_wires: int, edges: List[Tuple[int, int]]) -> qml.Hamiltonian:
    """Build the MaxCut Hamiltonian H_P = sum_{(i,j) in E} (I - Z_i Z_j)."""
    coeffs, ops = [], []
    for (i, j) in edges:
        coeffs.append(1)
        ops.append(qml.Identity(wires=0))
        coeffs.append(-1)
        ops.append(qml.PauliZ(i) @ qml.PauliZ(j))
    H_p = qml.Hamiltonian(coeffs, ops)
    # Rescaled observable 
    return (1/len(edges)) * H_p

def x_mixer_hamiltonian(n_wires: int) -> qml.Hamiltonian:
    """Return the standard QAOA mixer Hamiltonian H = sum_i X_i."""
    coeffs = [1.0] * n_wires
    ops = [qml.PauliX(i) for i in range(n_wires)]
    return qml.Hamiltonian(coeffs, ops)

# ================================================================
# Grover Mixer
# ================================================================

def phase_on_zero_state(beta, wires: List[int]):
    """Apply a relative phase e^{-i beta} to |0^n> using multi-controlled gates."""
    for w in wires:
        qml.PauliX(wires=w)
    qml.MultiControlledX(wires=wires, control_values = wires[:-1])
    qml.RZ(2* beta, wires=wires[-1])
    qml.MultiControlledX(wires=wires, control_values = wires[:-1])
    for w in wires:
        qml.PauliX(wires=w)

def grover_mixer_unitary(beta, wires: List[int]):
    """Apply Grover mixer U_M(beta) = H^{⊗n} * exp(-i beta |0^n><0^n|) * H^{⊗n}."""
    for w in wires:
        qml.Hadamard(wires=w)
    phase_on_zero_state(beta, wires)
    for w in wires:
        qml.Hadamard(wires=w)

# ================================================================
# QAOA Circuits
# ================================================================

def qaoa_layer(params, cost_h: qml.Hamiltonian, mixer: str, n_wires: int):
    """Single QAOA layer: cost evolution then mixer evolution."""
    gamma, beta = params
    qml.ApproxTimeEvolution(cost_h, gamma, 1)
    if mixer == "standard":
        qml.ApproxTimeEvolution(x_mixer_hamiltonian(n_wires), beta, 1)
    elif mixer == "grover":
        grover_mixer_unitary(beta, wires=list(range(n_wires)))
    else:
        raise ValueError(f"Unknown mixer: {mixer}")

def qaoa_circuit(params, p: int, cost_h: qml.Hamiltonian, mixer: str, n_wires: int):
    """Full QAOA circuit with p layers."""
    for w in range(n_wires):
        qml.Hadamard(wires=w)
    for i in range(p):
        qaoa_layer(params[i], cost_h, mixer, n_wires)


### Data ###

# ================================================================
# Graph Utilities for Discrete Optimization
# ================================================================
def random_connected_graph(n_wires: int, rng: np.random.Generator) -> List[Tuple[int, int]]:
    """Generate a random connected graph with n_wires nodes and return list of edges."""
    while True:
        G = nx.gnp_random_graph(n_wires, seed=int(rng.integers(0, 1000)), p=0.5)
        if nx.is_connected(G):
            return list(G.edges())
        
# ================================================================
# Theoretical Loss Function for MaxCut
# ================================================================
# add comments

def maxcut_objective_values(n_wires: int, edges: List[Tuple[int, int]]):
    """Compute distinct, nonzero MaxCut objective function values for a graph."""
    values = []
    for bits in it.product([0, 1], repeat=n_wires):
        cut_value = sum(bits[u] != bits[v] for u, v in edges)
        if cut_value != 0:
            values.append(cut_value)
    return sorted(set(values))

# Need to check for correctness.
def theoretical_variance(n_wires: int, edges: List[Tuple[int, int]]):
    """Compute theoretical variance from distinct MaxCut values."""
    lambdas = maxcut_objective_values(n_wires, edges)
    d = len(lambdas)
    total = sum((lambdas[j] - lambdas[i]) ** 2 for i in range(d) for j in range(i + 1, d))
    return total / (d**2 * (d + 1)), lambdas

def min_max_theoretical_variance(n: int, n_graphs: int = 5, rng: np.random.Generator = None):
    """Return min/max theoretical variance over n_graphs random graphs with n nodes."""
    if rng is None:
        rng = np.random.default_rng()
    vars_ = []
    for _ in range(n_graphs):
        edges = random_connected_graph(n, rng)
        var, _ = theoretical_variance(n, edges)
        vars_.append(var)
    return (min(vars_), max(vars_)) if vars_ else (None, None)

def variance_range_vs_n(n_min: int = 4, n_max: int = 16, n_graphs: int = 5, rng: np.random.Generator = None):
    """Compute min/max variance for n_min ≤ n ≤ n_max."""
    if rng is None:   
        rng = np.random.default_rng()
    results = {}
    for n in range(n_min, n_max + 1, 2):
        vmin, vmax = min_max_theoretical_variance(n, n_graphs=n_graphs, rng=rng)
        if vmin is not None:
            results[n] = {"min": vmin, "max": vmax}
    return results


# ================================================================
# Empirical Variance
# ================================================================

def empirical_variance(
    n_wires: int,
    edges: List[Tuple[int, int]],
    max_depth: int = 30,
    mixer: str = "grover",
    n_param_samples: int = 100,
    rng: np.random.Generator = None,
    device_name: str = "default.qubit",
    gamma_range: Tuple[float, float] = (0.0, np.pi),
    beta_range: Tuple[float, float] = (0.0, 2*np.pi),
):
    """
    Compute variance of the loss function for MaxCut <H_P> progressively for depths 1..max_depth,
    reusing the same random graphs and parameter samples.

    Returns:
        dict mapping depth -> {
            "variance": float,
            "mean": float,
            "samples": np.ndarray of shape (n_param_samples,)
        }
    """
    if rng is None:
        rng = np.random.default_rng()

    dev = qml.device(device_name, wires=n_wires)
    cost_h = maxcut_hamiltonian(n_wires, edges)

    @qml.qnode(dev)
    def expval_circuit(params, depth):
        """Run QAOA up to depth layers and return <H_P>."""
        # initial state |+>^n
        for w in range(n_wires):
            qml.Hadamard(wires=w)
        # apply first `depth` layers
        for i in range(depth):
            gamma, beta = params[i]
            qaoa_layer((gamma, beta), cost_h, mixer, n_wires)
        return qml.expval(cost_h)

    # Sample all parameters once for max_depth
    gammas = rng.uniform(gamma_range[0], gamma_range[1],
                         size=(n_param_samples, max_depth))
    betas = rng.uniform(beta_range[0], beta_range[1],
                        size=(n_param_samples, max_depth))
    params_all = np.stack([gammas, betas], axis=2)  # shape (n_samples, max_depth, 2)

    results = {}
    for depth in range(1, max_depth+1):
        vals = []
        t0 = time.time()
        for s in range(n_param_samples):
            vals.append(expval_circuit(params_all[s], depth))
        t1 = time.time()

        vals = np.array(vals)
        results[depth] = {
            "variance": float(np.var(vals, ddof=0)),
            "mean": float(np.mean(vals)),
            "samples": vals,
            "time": (t1 - t0)
        }
    return results

def variance_vs_depth(
    n_wires: int,
    n_graphs: int = 5,
    max_depth: int = 10,
    mixer: str = "grover",
    n_param_samples: int = 100,
    rng: np.random.Generator = None,
    gamma_range: Tuple[float, float] = (0.0, np.pi),
    beta_range: Tuple[float, float] = (0.0, 2 * np.pi),
):
    """
    Compute variance vs depth for multiple random graphs, reusing computation across depths.

    Returns:
        dict: results[p][g] = {"variance": float, "time": float}
    """
    if rng is None:
        rng = np.random.default_rng()

    results = {p: {} for p in range(1, max_depth + 1)}

    for g in range(n_graphs):
        edges = random_connected_graph(n_wires, rng=rng)

        print(f"\nGraph {g}: edges={edges}")

        res = empirical_variance(
            n_wires=n_wires,
            edges=edges,
            max_depth=max_depth,
            mixer=mixer,
            n_param_samples=n_param_samples,
            rng=rng,
            gamma_range=gamma_range,
            beta_range=beta_range,
        )

        for p in range(1, max_depth + 1):
            var = res[p]["variance"]
            t = res[p]["time"]
            results[p][g] = {"variance": var, "time": t}
            print(f"  depth={p:2d} | variance={var:.6f} | time={t:.2f}s")

    return results

    
### Plotting ###

# ================================================================
# Plotting Utilities
# ================================================================

def ensure_dir(path: str):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def save_plot(fig, fig_path: str, filename: str, show: bool = True):
    """Save matplotlib figure to file and optionally display it."""
    ensure_dir(fig_path)
    filepath = os.path.join(fig_path, filename)
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {filepath}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_variance_data(results, max_depth: int, outdir: str = "./figures", show: bool = True):
    """
    Plot per-graph variance vs depth curves.
    Saves as PNG and PDF.
    """
    ensure_dir(outdir)
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    for g in range(len(next(iter(results.values())))):
        depths = sorted(results.keys())
        variances = [results[p][g]["variance"] for p in depths]
        ax.plot(depths, variances, marker="o", label=f"Graph {g}")

    ax.set_xlabel("QAOA depth p")
    ax.set_ylabel("Variance of <H_P>")
    ax.set_yscale("log")
    ax.set_title("Variance vs Depth per Graph")
    ax.legend(fontsize=7)
    ax.grid(True)

    save_plot(fig, outdir, "variance_per_graph.png", show=show)
    save_plot(fig, outdir, "variance_per_graph.pdf", show=show)


def plot_variance_mean_std(results, max_depth: int, outdir: str = "./figures", show: bool = True):
    """
    Plot mean ± std variance across graphs vs depth.
    Saves as PNG and PDF.
    """
    ensure_dir(outdir)
    depths = sorted(results.keys())
    means, stds = [], []

    for p in depths:
        variances = [results[p][g]["variance"] for g in results[p].keys()]
        means.append(np.mean(variances))
        stds.append(np.std(variances))

    means, stds = np.array(means), np.array(stds)

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.plot(depths, means, marker="o", color="b", label="Mean variance")
    ax.fill_between(depths, means-stds, means+stds,
                    color="skyblue", alpha=0.4, label="±1 std")

    ax.set_xlabel("QAOA depth p")
    ax.set_ylabel("Variance of <H_P>")
    ax.set_yscale("log")
    ax.set_title("Mean Variance vs Depth")
    ax.legend(fontsize=7)
    ax.grid(True)

    save_plot(fig, outdir, "variance_mean_std.png", show=show)
    save_plot(fig, outdir, "variance_mean_std.pdf", show=show)

