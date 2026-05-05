import numpy as np
import networkx as nx
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.primitives import StatevectorEstimator
from scipy.optimize import minimize
import time

# --- 1. Problem Setup ---
seed = np.random.seed(42)
G = nx.random_regular_graph(3, 10, seed)
n_qubits = G.number_of_nodes()
edges = G.edges()
num_edges = len(edges)

# Calculate Max Cut via Brute Force for the Ratio denominator
def get_max_cut(n, edges):
    max_c = 0
    for i in range(2**n):
        b = format(i, f'0{n}b')
        cut = sum(1 for u, v in edges if b[u] != b[v])
        if cut > max_c: max_c = cut
    return max_c

MAX_CUT_VAL = get_max_cut(n_qubits, edges)

# Define Cost Hamiltonian H_C = sum(ZiZj)
pauli_list = []
for i, j in edges:
    p_str = ["I"] * n_qubits
    p_str[n_qubits - 1 - i] = "Z"
    p_str[n_qubits - 1 - j] = "Z"
    pauli_list.append("".join(p_str))
cost_h = SparsePauliOp(pauli_list, coeffs=[1.0] * num_edges)

# --- 2. Adaptive Reflection Logic ---
def add_reflection(qc, state_vec, beta):
    n = qc.num_qubits
    v_gate = QuantumCircuit(n)
    v_gate.prepare_state(state_vec)
    
    # Unitary: V * (I - (1-e^-ib)|0><0|) * V_adj
    qc.append(v_gate.inverse(), range(n))
    qc.x(range(n))
    qc.mcp(-beta, list(range(n-1)), n-1)
    qc.x(range(n))
    qc.append(v_gate, range(n))

# --- 3. Iterative Optimization Loop ---
p_total = 10  # Number of iterative layers
estimator = StatevectorEstimator()
current_state = Statevector.from_label('+' * n_qubits)
ratios = []

print(f"{'Layer':<6} | {'Energy':<10} | {'Exp. Cut':<10} | {'Ratio (%)':<10}")
print("-" * 45)
start_time = time.perf_counter()
for p in range(1, p_total + 1):
    
    def objective(params):
        gamma, beta = params
        qc = QuantumCircuit(n_qubits)
        qc.prepare_state(current_state, normalize=True)
        
        # Phase Separation (Cost)
        for i, j in edges:
            qc.rzz(2 * gamma, i, j)
            
        # Adaptive Mixing (Projector Mixer)
        add_reflection(qc, current_state, beta)
        
        pub = (qc, cost_h)
        return estimator.run([pub]).result()[0].data.evs

    # Layer-wise optimization
    res = minimize(objective, [0.71, 0.39], method='COBYQA')
    
    # --- Calculate Ratio ---
    # Expected Cut = 0.5 * (num_edges - <H_C>)
    expected_cut = 0.5 * (num_edges - res.fun)
    ratio = expected_cut / MAX_CUT_VAL
    ratios.append(ratio)
    
    # Update state for next layer
    final_qc = QuantumCircuit(n_qubits)
    final_qc.prepare_state(current_state)
    for i, j in edges:
        final_qc.rzz(2 * res.x[0], i, j)
    add_reflection(final_qc, current_state, res.x[1])
    current_state = Statevector.from_instruction(final_qc)
    
    print(f"{p:<6} | {res.fun:<10.4f} | {expected_cut:<10.4f} | {ratio*100:<10.2f}%")
print("-" * 45)
print(f"Final Approximation Ratio: {ratios[-1]:.4f}")
end_time = time.perf_counter()
print(f"Final runtime: {(end_time - start_time)/60}min")