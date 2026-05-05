"""
Title: Benchmarking GM-QAOA for MaxCut

This module provides functions for constructing QAOA circuits with Grover mixers (GM-QAOA). The code was inspired by 
"Provable avoidance of barren plateaus for the Quantum Approximate Optimization Algorithm with Grover mixers."
arXiv:2509.10424
Author: Matthew Nuyten
Last Update: 1/16/2026
"""

import matplotlib.pyplot as plt
import networkx as nx
import pennylane as qml
import numpy as np
from pennylane import numpy as pnp
import pandas as pd
import time
import itertools as it
from typing import Dict, List, Tuple
from matplotlib.ticker import LogLocator, MaxNLocator, AutoLocator

import os


# ================================================================
# Hamiltonians
# ================================================================

def maxcut_hamiltonian(edges: List[Tuple[int, int]]) -> qml.Hamiltonian:
    """
    The problem Hamiltonian we want to maximize
    H_P = sum_{(i,j) in E}  Z_i Z_j 
    """
    coeffs = []
    ops = []
    for (i, j) in edges:
        coeffs.append(1.0)  # 1.0 * Z_i Z_j
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

def phase_on_zero_state(beta, wires):
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

def grover_mixer_unitary(beta, wires):
    """Apply Grover mixer U_M(beta) = H^{⊗n} * exp(-i beta |0^n><0^n|) * H^{⊗n}."""
    for w in wires:
        qml.Hadamard(wires=w)
    phase_on_zero_state(beta, wires)
    for w in wires:
        qml.Hadamard(wires=w)

# ================================================================
# QAOA Circuit
# ================================================================

def qaoa_layer(gamma, beta, cost_h, mixer, n_wires):
    qml.ApproxTimeEvolution(cost_h, gamma, 1)

    if mixer == "standard":
        qml.ApproxTimeEvolution(
            x_mixer_hamiltonian(n_wires), beta, 1
        )
    elif mixer == "grover":
        grover_mixer_unitary(beta, wires=list(range(n_wires)))
    else:
        raise ValueError(f"Unknown mixer: {mixer}")

def qaoa_circuit(
    gammas,
    betas,
    cost_h,
    mixer,
    n_wires,
):
    
    for w in range(n_wires):
        qml.Hadamard(wires=w)

    for gamma, beta in zip(gammas, betas):
        qaoa_layer(gamma, beta, cost_h, mixer, n_wires)

# ================================================================
# MaxCut approximation ratio
# ================================================================

def cut_value(
    bitstring: Tuple[int, ...],
    edges: List[Tuple[int, int]],
    weights: Dict[Tuple[int, int], float] = None,
):
    """Compute the cut value for a given bitstring for weighted or unweighted graphs."""
    val = 0.0
    for i, j in edges:
        if bitstring[i] != bitstring[j]:
            w = 1.0 if weights is None else weights[(i, j)]
            val += w
    return val

def max_cut_value(
    n: int,
    edges: List[Tuple[int, int]],
    weights: Dict[Tuple[int, int], float] = None,
):
    """
    Compute the maximum cut value and all optimal bitstrings.

    Returns:
        max_value: float
        optimal_bitstrings: list of tuples
    """
    max_value = float("-inf")
    optimal_bitstrings = []

    for z in it.product([0, 1], repeat=n):
        val = cut_value(z, edges, weights)

        if val > max_value:
            max_value = val
            optimal_bitstrings = [z]
        elif val == max_value:
            optimal_bitstrings.append(z)

    return max_value, optimal_bitstrings

def qaoa_approximation_ratio(
    n_wires,
    edges,
    max_depth=5,
    mixer="grover",
    optimizer_name="adam",
    steps=200,
    stepsize=0.05,
    device_name="default.qubit",
    gamma_range=(0.0, 2*np.pi),
    beta_range=(0.0, np.pi),
    seed=None,
):
    rng = np.random.default_rng(seed)

    # Exact Max-Cut value
    C_max, _ = max_cut_value(n_wires, edges)

    cost_h = maxcut_hamiltonian(edges)
    dev = qml.device(device_name, wires=n_wires)

    @qml.qnode(dev)
    def cost_circuit(gammas, betas):
        qaoa_circuit(gammas, betas, cost_h, mixer, n_wires)
        return qml.expval(cost_h)

    results = {}

    for p in range(1, max_depth + 1):

        # optimizer per depth
        if optimizer_name.lower() == "adam":
            opt = qml.AdamOptimizer(stepsize)
        elif optimizer_name.lower() == "gd":
            opt = qml.GradientDescentOptimizer(stepsize)
        else:
            raise ValueError("Unknown optimizer")

        # initialize parameters
        init_gammas = rng.uniform(*gamma_range, size=p)
        init_betas  = rng.uniform(*beta_range, size=p)

        gammas = pnp.array(init_gammas, requires_grad=True)
        betas  = pnp.array(init_betas,  requires_grad=True)

        t0 = time.time()

        def cost_fn(g, b):
            return cost_circuit(g, b)

        for _ in range(steps):
            gammas, betas = opt.step(cost_fn, gammas, betas)

        opt_value = - float(cost_circuit(gammas, betas))
        approx_ratio = opt_value / C_max

        t1 = time.time()

        results[p] = {
            "approx_ratio": approx_ratio,
            "opt_value": opt_value,
            "time": t1 - t0,
        }

    return results

### Graph Utilities
def sample_random_graph(n_wires: int, seed: int | None = None) -> List[Tuple[int, int]]:
    """Generate a random connected graph with n_wires nodes and return list of edges."""
    rng = np.random.default_rng(seed)
    while True:
        G = nx.gnp_random_graph(n_wires, seed=int(rng.integers(0, 1000)), p=0.5)
        if nx.is_connected(G):
            return list(G.edges())

### Experiment
def random_graph_experiment(
    n_min: int,
    n_max: int,
    max_depth: int,
    seed: int,
    **qaoa_kwargs,
):
    """
    For n = n_min,...,n_max:
        - sample random graph
        - compute QAOA approximation ratio vs depth

    Returns:
        results[n][p] = approx_ratio
    """
    rng = np.random.default_rng(seed)
    results = {}

    for n in range(n_min, n_max + 1):
        edges = sample_random_graph(n, seed=rng.integers(1e9))

        # Compute exact Max-Cut value and bitstring
        C_max, _ = max_cut_value(n, edges)

        print(f"Running QAOA for n={n}, |E|={len(edges)}, C_max={C_max}")

        qaoa_res = qaoa_approximation_ratio(
            n,
            edges,
            max_depth=max_depth,
            seed=rng.integers(1e9),
            **qaoa_kwargs,
        )
        optimal_expval = max([qaoa_res[p]["opt_value"] for p in qaoa_res])
        print("expectation value <H_C>", optimal_expval)

        # Extract approximation ratios by depth
        results[n] = {
            "C_max": C_max,
            "expval": optimal_expval,
            "edges": edges,
            "ratios": {
                p: qaoa_res[p]["approx_ratio"]
                for p in qaoa_res
            },
        }
        
    return results

### RUN THIS ###
if __name__ == "__main__":

    results = random_graph_experiment(
        n_min=9,
        n_max=15,
        max_depth=40,
        mixer="grover",
        optimizer_name="adam",
        steps=200,
        stepsize=0.05,
        seed=1234,
    )

    df = pd.DataFrame(results)
    df.to_csv("benchmark_depth40_out.csv", index=False)
