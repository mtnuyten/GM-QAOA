"""
Title: Variance of the GM-QAOA loss function for MaxCut

This module provides functions for constructing QAOA circuits with Grover mixers (GM-QAOA). The code was inspired by 
"Provable avoidance of barren plateaus for the Quantum Approximate Optimization Algorithm with Grover mixers."
arXiv:2509.10424
Author: Matthew Nuyten
Last Update: 1/19/2026
"""

import networkx as nx
import pennylane as qml
import numpy as np
import pandas as pd

import time
import itertools as it
from typing import List, Tuple


# ================================================================
# Hamiltonians
# ================================================================

def maxcut_hamiltonian(edges: List[Tuple[int, int]], normalize: bool = True) -> qml.Hamiltonian:
    """
    The problem Hamiltonian we want to maximize
    H_P = sum_{(i,j) in E} Z_i Z_j

    Leave normalize=True.
    """
    m = len(edges)
    coeffs = []
    ops = []
    scale = 1.0 / m if (normalize and m > 0) else 1.0

    for (i, j) in edges:
        coeffs.append(1.0 * scale)  # 1.0 * Z_i Z_j
        ops.append(qml.PauliZ(i) @ qml.PauliZ(j))

    return qml.Hamiltonian(coeffs, ops)

def x_mixer_hamiltonian(n_wires: int) -> qml.Hamiltonian:
    """Return the standard QAOA mixer Hamiltonian:
    Transverse field mixer, X-mixer, H = sum_i X_i."""
    
    coeffs = [1.0] * n_wires
    ops = [qml.PauliX(i) for i in range(n_wires)]
    return qml.Hamiltonian(coeffs, ops)

# ================================================================
# Grover Mixer
# ================================================================


def phase_on_zero_state(beta, wires: List[int]):
    controls = wires[:-1]
    target = wires[-1]
    for w in wires:
        # apply flip   
        qml.PauliX(wires=w)
    # Apply a multi-controlled PhaseShift of e^{-i*beta} on the last qubit
    qml.ctrl(qml.PhaseShift, control=controls)(-beta, target)
    for w in wires:
        # Undo flip
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
# Graph Utilities
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

def maxcut_objective_values(n_wires: int, edges: List[Tuple[int, int]]):
    """Compute distinct, nonzero MaxCut objective function values for a graph."""
    values = []
    for bits in it.product([0, 1], repeat=n_wires):
        cut_value = sum(bits[u] != bits[v] for u, v in edges)
        if cut_value != 0:
            values.append(cut_value)
    return sorted(set(values))

def theoretical_variance(edges: List[Tuple[int, int]], coeff: float = 1.0, n_wires: int = 8):
    """Compute theoretical variance from distinct MaxCut values."""
    lambdas = maxcut_objective_values(n_wires, edges)
    d = len(lambdas)
    total = sum((lambdas[i] - lambdas[j]) ** 2 for i in range(d) for j in range(i + 1, d))
    result = coeff * total / (d**2 * (d + 1))
    return result, lambdas

# ================================================================
# Empirical Variance
# ================================================================

def empirical_variance(
    n_wires: int,
    edges: List[Tuple[int, int]],
    max_depth: int = 30,
    mixer: str = "grover",
    n_param_samples: int = 50,
    rng: np.random.Generator = None,
    device_name: str = "lightning.qubit",
    gamma_range: Tuple[float, float] = (0.0, 2*np.pi),
    beta_range: Tuple[float, float] = (0.0, np.pi),
):
    """
    Compute expectation values and variances for depth 1...max_depth
    with the same set of random parameters and graphs.

    Args:
        n_wires (int): number of qubits
        edges (list[tuple]): graph edges
        max_depth (int): maximum QAOA depth p
        mixer (str): "standard" or "grover"
        n_param_samples (int): number of random parameter samples
        rng (np.random.Generator): random generator
        device_name (str): PennyLane's "default.qubit" or "lightning.qubit"
        gamma_range (tuple): sampling range for gammas
        beta_range (tuple): sampling range for betas

    Returns:
        dict: res[p] = {
            "variance": float,
            "expectations": np.array,
            "mean": float,
            "time": float
        }
    """
    if rng is None:
        rng = np.random.default_rng()

    # build problem Hamiltonian H_P
    cost_h = maxcut_hamiltonian(edges, normalize=True)
    
    dev = qml.device(device_name, wires=n_wires)

    @qml.qnode(dev)
    def expval_circuit(params, depth):
        """Run QAOA up to depth layers and return expectation value of the loss function."""
        qaoa_circuit(params[:depth], depth, cost_h, mixer=mixer, n_wires=n_wires)
        return qml.expval(cost_h)
        
    # Pre-sample parameters for all depths
    gammas = rng.uniform(gamma_range[0], gamma_range[1],
                         size=(n_param_samples, max_depth))
    betas = rng.uniform(beta_range[0], beta_range[1],
                        size=(n_param_samples, max_depth))
    params = np.stack([gammas, betas], axis=-1)  # shape (n_samples, max_depth, 2)
    
    results = {}
    for p in range(1, max_depth + 1):
        t0 = time.time()
        vals = [expval_circuit(params[i, :p], p) for i in range(n_param_samples)]
        t1 = time.time()

        results[p] = {
            "variance": float(np.var(vals, ddof=0)),
            "expectations": np.array(vals),
            "mean": float(np.mean(vals)),
            "time": t1 - t0,
        }

    return results

def stats_vs_depth(
    n_wires: int,
    n_graphs: int = 5,
    max_depth: int = 30,
    mixer: str = "grover",
    n_param_samples: int = 100,
    rng: np.random.Generator = None,
    gamma_range: Tuple[float, float] = (0.0, 2*np.pi),
    beta_range: Tuple[float, float] = (0.0, np.pi),
):
    """
    Compute variance and expectation vs depth for multiple random graphs, reusing computation across depths.

    Returns:
        dict: results[p][g] = {
            "variance": float,
            "expectation": float,
            "time": float
        }
    """
    if rng is None:
        rng = np.random.default_rng()

    results = {p: {} for p in range(1, max_depth + 1)}

    for g in range(n_graphs):
        edges = random_connected_graph(n_wires, rng=rng)
        print(f"\nGraph {int(g+1)}: edges={edges}")

        # progressive empirical calculation
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
            mean = res[p]["mean"]     
            t = res[p]["time"]

            results[p][g] = {"variance": var, "expectation": mean, "time": t}
            print(f"  depth={p:2d} | var={var:.6f} | ⟨H_P⟩={mean:.6f} | time={t:.2f}s")

    return results


def empirical_variance_range_vs_n(
    n_min: int = 4,
    n_max: int = 10,
    n_graphs: int = 5,
    max_depth: int = 30,
    n_param_samples: int = 100,
    rng: np.random.Generator = None,
):
    """
    Estimate the lower and upper bound of the variance as a function of n.
    For each n, sample multiple random graphs, compute variances across depths,
    record the minimum (maximum) variance per graph, and then take the minimum (maximum) across graphs.
    
    Returns:
        dict[n] = {
            "min_over_graphs": float,
            "mins_per_graph": list,
            "max_over_graphs": float,
            "maxs_per_graph": list,
        }
    """
    if rng is None:
        rng = np.random.default_rng()
    
    results = {}
    for n in range(n_min, n_max + 1):
        mins_per_graph = []
        maxs_per_graph = []
        for g in range(n_graphs):
            edges = random_connected_graph(n, rng=rng)
            res = empirical_variance(
                n_wires=n,
                edges=edges,
                max_depth=max_depth,
                rng=rng,
                n_param_samples=n_param_samples,
            )
            # take min/max variance across depths for this graph
            min_var_graph = min(res[p]["variance"] for p in res)
            mins_per_graph.append(min_var_graph)
            max_var_graph = max(res[p]["variance"] for p in res)
            maxs_per_graph.append(max_var_graph)
        
        results[n] = {
            "min_over_graphs": float(np.min(mins_per_graph)),
            "mins_per_graph": mins_per_graph,
            "max_over_graphs": float(np.max(maxs_per_graph)),
            "maxs_per_graph": maxs_per_graph,
        }
        print(f"n={n}: min variance across {n_graphs} graphs = {results[n]['min_over_graphs']:.3e}")
        print(f"n={n}: max variance across {n_graphs} graphs = {results[n]['max_over_graphs']:.3e}")
    
    return results

### Run this ###

if __name__ == "__main__":

    # Parameters
    n_wires = 10
    n_graphs = 5
    small_depth = 15
    max_depth = 40
    n_param_samples=50
    rng = np.random.default_rng(42)
    
    # Run variance vs depth
    #my_data = stats_vs_depth(
    #    n_wires=n_wires,
    #    n_graphs=n_graphs,
    #    max_depth=max_depth,
    #    mixer="grover",
    #    n_param_samples=n_param_samples,
    #    rng=rng,
    #)
    #df = pd.DataFrame(my_data)
    #df.to_csv("variance_vs_depth.csv", index=False)
    

    #Compute variance bounds across n
    variance_bounds = empirical_variance_range_vs_n(n_min=4, n_max=24, max_depth=small_depth, n_param_samples=n_param_samples, rng=rng)
    
    df = pd.DataFrame(variance_bounds)
    df.to_csv("varrange_vs_nqubits.csv", index=False)
    