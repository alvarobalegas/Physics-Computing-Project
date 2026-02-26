import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# Parámetros globales
N = 400
T = 100
mutation_prob = 0.01
initial_coop_prob = 0.5
n_runs = 30
b_values = np.linspace(0.5, 2.0, 20)

# Parámetros red comunidades
n_communities = 8
nodes_per_community = N // n_communities
p_intra = 0.3
p_inter = 0.01

def compute_payoff(node, G, states, R, S, T_payoff, P):
    payoff = 0
    for neighbor in G.neighbors(node):
        if states[node] == 'C' and states[neighbor] == 'C':
            payoff += R
        elif states[node] == 'C' and states[neighbor] == 'D':
            payoff += S
        elif states[node] == 'D' and states[neighbor] == 'C':
            payoff += T_payoff
        elif states[node] == 'D' and states[neighbor] == 'D':
            payoff += P
    return payoff

def simulate_game(G, b):
    R, S, T_payoff, P = 1.0, 0.0, b, 0.0

    states = {n: ('C' if np.random.rand() < initial_coop_prob else 'D') for n in G.nodes}

    for _ in range(T):
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

def generate_community_graph():
    sizes = [nodes_per_community] * n_communities
    p_matrix = np.full((n_communities, n_communities), p_inter)
    np.fill_diagonal(p_matrix, p_intra)
    G = nx.stochastic_block_model(sizes, p_matrix)
    return G

def run_for_network_type(net_type, b_values):
    results = []
    for b in b_values:
        coop_runs = []
        for _ in range(n_runs):
            if net_type == 'ER':
                p_ER = 0.05  # Ajustar para grado promedio similar a comunidades
                G = nx.erdos_renyi_graph(N, p_ER)
            elif net_type == 'BA':
                m = 3  # promedio enlaces por nodo
                G = nx.barabasi_albert_graph(N, m)
            elif net_type == 'Community':
                G = generate_community_graph()
            else:
                raise ValueError("Tipo de red no reconocido")
            coop_fraction = simulate_game(G, b)
            coop_runs.append(coop_fraction)

        mean_coop = np.mean(coop_runs)
        std_coop = np.std(coop_runs)

        # Para comunidades calculamos modularidad (usando la partición conocida)
        if net_type == 'Community':
            sizes = [nodes_per_community] * n_communities
            partition = []
            for i, size in enumerate(sizes):
                partition += [i] * size
            modularity = nx.algorithms.community.quality.modularity(G, 
                [set(np.where(np.array(partition)==i)[0]) for i in range(n_communities)])
        else:
            modularity = np.nan

        results.append({
            'network': net_type,
            'b': b,
            'mean_pc': mean_coop,
            'std_pc': std_coop,
            'modularity': modularity
        })
        print(f"{net_type} | b={b:.2f} | pC={mean_coop:.3f}±{std_coop:.3f} | Q={modularity if modularity==modularity else 'NA'}")
    return pd.DataFrame(results)

# Ejecutar simulaciones para cada red
df_er = run_for_network_type('ER', b_values)
df_ba = run_for_network_type('BA', b_values)
df_comm = run_for_network_type('Community', b_values)

# Concatenar resultados
df_all = pd.concat([df_er, df_ba, df_comm])

# Graficar comparación
plt.figure(figsize=(10,6))
for net_type, df_net in df_all.groupby('network'):
    plt.errorbar(df_net['b'], df_net['mean_pc'], yerr=df_net['std_pc'], label=net_type, marker='o', capsize=4)
plt.xlabel("Tentación b")
plt.ylabel("Fracción de cooperadores ⟨p_C⟩")
plt.title("Comparación cooperación en ER, BA y Comunidades")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Graficar modularidad solo para comunidades
plt.figure(figsize=(6,4))
plt.plot(df_comm['b'], df_comm['modularity'], marker='o')
plt.xlabel("Tentación b")
plt.ylabel("Modularidad Q")
plt.title("Modularidad en red con comunidades")
plt.grid(True)
plt.tight_layout()
plt.show()
