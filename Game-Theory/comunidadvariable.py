import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# Parámetros generales
T = 100
mutation_prob = 0.01
initial_coop_prob = 0.5
n_runs = 10
b_values = np.linspace(0.5, 2.0, 20)

# ======== FUNCIÓN PARA CREAR RED CON COMUNIDADES ============
def crear_red_comunidades(num_communities=4, community_size=50, p_in=0.3, p_out=0.01):
    sizes = [community_size] * num_communities
    n = num_communities * community_size
    p_matrix = np.full((num_communities, num_communities), p_out)
    np.fill_diagonal(p_matrix, p_in)
    G = nx.stochastic_block_model(sizes, p_matrix)
    return G

# ======== DINÁMICA EVOLUTIVA ================================
def compute_payoff(node, G, states, R, S, T, P):
    payoff = 0
    for neighbor in G.neighbors(node):
        if states[node] == 'C' and states[neighbor] == 'C':
            payoff += R
        elif states[node] == 'C' and states[neighbor] == 'D':
            payoff += S
        elif states[node] == 'D' and states[neighbor] == 'C':
            payoff += T
        elif states[node] == 'D' and states[neighbor] == 'D':
            payoff += P
    return payoff

def run_simulation_community(b, G):
    R, S, T_payoff, P = 1.0, 0.0, b, 0.0

    states = {n: 'C' if np.random.rand() < initial_coop_prob else 'D' for n in G.nodes}
    
    for t in range(T):
        payoffs = {n: compute_payoff(n, G, states, R, S, T_payoff, P) for n in G.nodes}
        new_states = states.copy()
        for node in G.nodes:
            neighbors = list(G.neighbors(node))
            if not neighbors:
                continue
            best_neighbor = max(neighbors, key=lambda n: payoffs[n])
            if payoffs[best_neighbor] > payoffs[node]:
                new_states[node] = states[best_neighbor]
            elif np.random.rand() < mutation_prob:
                new_states[node] = 'C' if states[node] == 'D' else 'D'
        states = new_states

    coop_fraction = sum(1 for s in states.values() if s == 'C') / len(G.nodes)
    return coop_fraction

# ======== ESTUDIO COMPARATIVO ============================
def analizar_estructuras(num_communities, community_size, p_out):
    resultados = []

    for b in b_values:
        coop_list = []
        for _ in range(n_runs):
            G = crear_red_comunidades(num_communities, community_size, p_in=0.3, p_out=p_out)
            pC = run_simulation_community(b, G)
            coop_list.append(pC)
        resultados.append({
            'b': b,
            'mean_pc': np.mean(coop_list),
            'std_pc': np.std(coop_list)
        })

    df = pd.DataFrame(resultados)
    
    # Cálculo de b_c
    dp_db = np.gradient(df['mean_pc'].values, df['b'].values)
    b_c = df['b'].values[np.argmin(dp_db)]
    print(f"Para {num_communities} comunidades, p_out = {p_out} → b_c ≈ {b_c:.2f}")

    # Gráfico
    plt.errorbar(df['b'], df['mean_pc'], yerr=df['std_pc'], label=f'N_comm={num_communities}, p_out={p_out}', capsize=4)
    plt.xlabel("Tentación b")
    plt.ylabel("⟨p_C⟩ ± σ")
    plt.title("Fracción de cooperadores vs b")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

# ======== EXPERIMENTOS ============================
plt.figure(figsize=(10,6))

# Prueba con distintos p_out y número de comunidades
analizar_estructuras(num_communities=5, community_size=50, p_out=0.01)
analizar_estructuras(num_communities=5, community_size=50, p_out=0.05)
analizar_estructuras(num_communities=5, community_size=50, p_out=0.10)
analizar_estructuras(num_communities=10, community_size=50, p_out=0.01)
analizar_estructuras(num_communities=10, community_size=50, p_out=0.05)
analizar_estructuras(num_communities=10, community_size=50, p_out=0.10)
analizar_estructuras(num_communities=15, community_size=50, p_out=0.01)
analizar_estructuras(num_communities=15, community_size=50, p_out=0.05)
analizar_estructuras(num_communities=15, community_size=50, p_out=0.10)
analizar_estructuras(num_communities=20, community_size=50, p_out=0.01)
analizar_estructuras(num_communities=20, community_size=50, p_out=0.05)
analizar_estructuras(num_communities=20, community_size=50, p_out=0.10)

plt.show()
