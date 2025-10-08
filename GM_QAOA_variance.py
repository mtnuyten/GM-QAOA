"""
Grover Mixer QAOA Variance Experiments

This module provides functions for constructing QAOA circuits with both
standard and Grover mixers. For computing empirical and theoretical
variance of the loss function corresponding the the MaxCut problem across different graphs, system sizes,
and depths. Includes examples at the bottom with seeding for reproducibility.

Author: Matthew Nuyten
Date: 2025-09-21
"""

import itertools
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import networkx as nx
import pennylane as qml
from pennylane import numpy as np

# ================================================================
# Hamiltonians
# ================================================================

def maxcut_hamiltonian(n_wires: int, edges: List[Tuple[int, int]]) -> qml.Hamiltonian:
    """Build the MaxCut Hamiltonian H_P = sum_{(i,j) in E} 0.5*(I - Z_i Z_j)."""
    coeffs, ops = [], []
    for (i, j) in edges:
        coeffs.append(0.5)
        ops.append(qml.Identity(wires=0))
        coeffs.append(-0.5)
        ops.append(qml.PauliZ(i) @ qml.PauliZ(j))
    return qml.Hamiltonian(coeffs, ops)

def x_mixer_hamiltonian(n_wires: int) -> qml.Hamiltonian:
    """Return the standard QAOA mixer Hamiltonian H = sum_i X_i."""
    coeffs = [1.0] * n_wires
    ops = [qml.PauliX(i) for i in range(n_wires)]
    return qml.Hamiltonian(coeffs, ops)

# ================================================================
# Grover Mixer
# ================================================================

def phase_on_zero_state(beta, wires: List[int]):
    """Apply a relative phase e^{i beta} to |0^n> using multi-controlled gates."""
    for w in wires:
        qml.PauliX(wires=w)
    qml.MultiControlledX(wires=wires)
    qml.RZ(2 * beta, wires=wires[-1])
    qml.MultiControlledX(wires=wires)
    for w in wires:
        qml.PauliX(wires=w)

def grover_mixer_unitary(beta, wires: List[int]):
    """Apply Grover mixer U_M(beta) = H^{⊗n} * exp(i beta |0^n><0^n|) * H^{⊗n}."""
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

# ================================================================
# Random Graph Utilities
# ================================================================

def random_connected_graph(n_wires: int, rng: np.random.Generator) -> List[Tuple[int, int]]:
    """Generate a random connected graph with n_wires nodes."""
    nx_seed = int(rng.integers(0, 1000))
    while True:
        G = nx.gnp_random_graph(n_wires, p=0.5, seed=nx_seed)
        if nx.is_connected(G):
            return list(G.edges())

# ================================================================
# Theoretical Variance
# ================================================================

def maxcut_objective_values(n_wires: int, edges: List[Tuple[int, int]]):
    """Compute distinct nonzero MaxCut objective values for a graph."""
    values = []
    for bits in itertools.product([0, 1], repeat=n_wires):
        cut_value = sum(bits[u] != bits[v] for u, v in edges)
        if cut_value != 0:
            values.append(cut_value)
    return sorted(set(values))

def theoretical_variance(n_wires: int, edges: List[Tuple[int, int]]):
    """Compute theoretical variance from distinct MaxCut objective values."""
    lambdas = maxcut_objective_values(n_wires, edges)
    d = len(lambdas)
    total = sum((lambdas[j] - lambdas[i]) ** 2 for i in range(d) for j in range(i + 1, d))
    return total / (d**2 * (d + 1)), lambdas

def min_max_theoretical_variance(n: int, n_graphs: int = 30, seed: int = None,):
    """Return min/max theoretical variance over random graphs with n nodes."""
    
    rng = np.random.default_rng(seed)
    vars_ = []
    for _ in range(n_graphs):
        edges = random_connected_graph(n, rng)
        var, _ = theoretical_variance(n, edges)
        vars_.append(var)
    return (min(vars_), max(vars_)) if vars_ else (None, None)

def variance_range_vs_n(n_min=4, n_max=16, n_graphs=5, seed = None,):
    """Compute min/max variance for n_min ≤ n ≤ n_max."""
    rng = np.random.default_rng(seed)
    results = {}
    for n in range(n_min, n_max + 1, 2):
        vmin, vmax = min_max_theoretical_variance(n, n_graphs=n_graphs, rng=rng)
        if vmin is not None:
            results[n] = {"min": vmin, "max": vmax}
    return results

def plot_variance_range(results: Dict[int, Dict[str, float]]):
    """Plot shaded region between min and max theoretical variance."""
    ns = sorted(results.keys())
    mins = [results[n]["min"] for n in ns]
    maxs = [results[n]["max"] for n in ns]
    plt.figure(figsize=(7, 5))
    plt.fill_between(ns, mins, maxs, color="skyblue", alpha=0.4, label="Variance range")
    plt.plot(ns, mins, "b--", label="Min variance")
    plt.plot(ns, maxs, "b-", label="Max variance")
    plt.xlabel("Number of qubits n")
    plt.ylabel("Theoretical variance of MaxCut")
    plt.yscale("log")
    plt.title("Range of theoretical variances over random graphs")
    plt.legend()
    plt.grid(True)
    plt.show()

# ================================================================
# Empirical Variance
# ================================================================

def empirical_variance_over_params(
    n_wires: int,
    edges: List[Tuple[int, int]],
    p: int = 1,
    mixer: str = "grover",
    n_param_samples: int = 200,
    rng: np.random.Generator = None,
    device_name: str = "default.qubit",
    gamma_range: Tuple[float, float] = (0.0, np.pi),
    beta_range: Tuple[float, float] = (0.0, 2* np.pi),
):
    """
    Estimate the variance of the cost Hamiltonian expectation <H_C>
    over random QAOA parameters (gamma, beta) sampled uniformly.

    Args:
        n_wires (int): number of qubits
        edges (list): problem edges (e.g. for MaxCut)
        p (int): QAOA depth
        mixer (str): 'grover' or 'x'
        n_param_samples (int): number of random parameter samples
        rng : numpy.random.Generator, optional. Random number generator. If None, a new one is created.
        device_name (str): PennyLane device
        gamma_range (tuple): range for gamma parameters (default [0, π])
        beta_range (tuple): range for beta parameters (default [0, 2π])

    Returns:
        dict with keys:
            - "variance": float
            - "mean": float
            - "samples": np.ndarray of sampled <H_C> values
            - "time": float runtime in seconds
    """
    if rng is None:
        rng = np.random.default_rng()

    dev = qml.device(device_name, wires=n_wires)
    cost_h = maxcut_hamiltonian(n_wires, edges)

    @qml.qnode(dev)
    def expval_circuit(params):
        qaoa_circuit(params, p, cost_h, mixer=mixer, n_wires=n_wires)
        return qml.expval(cost_h)

    vals = np.zeros(n_param_samples, dtype=float)

    import time
    t0 = time.time()
    for k in range(n_param_samples):
        gammas = rng.uniform(gamma_range[0], gamma_range[1], size=p)
        betas = rng.uniform(beta_range[0], beta_range[1], size=p)
        params = np.stack([gammas, betas], axis=1)  # shape (p, 2)
        vals[k] = expval_circuit(params)
    t1 = time.time()

    return {
        "variance": float(np.var(vals, ddof=0)),
        "mean": float(np.mean(vals)),
        "samples": vals,
        "time": (t1 - t0),
    }

def empirical_variance_vs_n(
    n_min: int = 4,
    n_max: int = 16,
    depths: list = [1, 5, 20],
    n_graphs: int = 5,
    n_param_samples: int = 100,
    rng: np.random.Generator = None,
):
    """
    Compute empirical variance scaling with system size n for Grover mixer QAOA.

    Args:
        n_min, n_max: range of qubits
        depths: list of depths p to test
        n_graphs: number of random connected graphs per n
        n_param_samples: number of random parameter samples for empirical variance
        rng: numpy.random.Generator, optional.

    Returns:
        results: dict {p: {n: mean variance}}
    """
    if rng is None:
        rng = np.random.default_rng()

    results = {p: {} for p in depths}

    for n in range(n_min, n_max + 1, 2):
        for p in depths:
            variances = []
            for k in range(n_graphs):
                edges = random_connected_graph(n, rng=rng)
                res = empirical_variance_over_params(
                    n_wires=n,
                    edges=edges,
                    p=p,
                    mixer="grover",
                    n_param_samples=n_param_samples,
                    rng=rng,
                )
                variances.append(res["variance"])
            results[p][n] = np.mean(variances)
    return results

def plot_empirical_variance_vs_n(results, depths):
    plt.figure(figsize=(7,5))
    for p in depths:
        ns = sorted(results[p].keys())
        vars_ = [results[p][n] for n in ns]
        plt.plot(ns, vars_, marker="o", label=f"p={p}")
    plt.yscale("log")
    plt.xlabel("Number of qubits n")
    plt.ylabel("Empirical variance of <H_C>")
    plt.title("Scaling of empirical variance with system size")
    plt.legend()
    plt.grid(True, which="both")
    plt.show()


def variance_vs_depth(
    n_wires: int,
    edges: List[Tuple[int, int]],
    depths: List[int],
    mixer: str = "grover",
    n_param_samples: int = 100,
    seed: int = None,
    gamma_range: Tuple[float, float] = (0.0, np.pi),
    beta_range: Tuple[float, float] = (0.0, 2*np.pi),
):
    """
    Sweep QAOA depth and compute variance statistics.

    Args:
        n_wires (int): number of qubits
        edges (list): problem edges
        depths (list): list of QAOA depths p
        mixer (str): 'grover' or 'x'
        n_param_samples (int): number of random parameter samples per depth
        seed (int): for reproducibility
        gamma_range (tuple): range for gamma params
        beta_range (tuple): range for beta params

    Returns:
        dict mapping depth -> results dict
    """
    rng = np.random.default_rng(seed)

    results = {}
    for p in depths:
        res = empirical_variance_over_params(
            n_wires,
            edges,
            p=p,
            mixer=mixer,
            n_param_samples=n_param_samples,
            rng=rng,
            gamma_range=gamma_range,
            beta_range=beta_range,
            )
        results[p] = res
    return results