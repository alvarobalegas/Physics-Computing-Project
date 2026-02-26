import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Parámetros globales
N0 = 400      # nodos iniciales
T = 100       # iteraciones evolutivas
mutation_prob = 0.01
initial_coop_prob = 0.5
n_runs = 50
b_values = np.linspace(0.5, 2.0, 20)
n_communities = 4
nodes_per_community = 100
p_intra = 0.5   # probabilidad de conexión dentro de comunidades
p_inter = 0.1  # probabilidad de conexión entre comunidades

summary_results = []

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

def run_simulation_communities(b):
    R, S, T_payoff, P = 1.0, 0.0, b, 0.0
    sizes = [nodes_per_community] * n_communities
    p_matrix = np.full((n_communities, n_communities), p_inter)
    np.fill_diagonal(p_matrix, p_intra)
    G = nx.stochastic_block_model(sizes, p_matrix)

    states = {
        n: 'C' if np.random.rand() < initial_coop_prob else 'D'
        for n in G.nodes
    }

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

    final_coop_fraction = sum(1 for s in states.values() if s == 'C') / len(G.nodes)
    var_k = np.var([deg for _, deg in G.degree()])
    clustering = nx.average_clustering(G)

    # MOD: calcular modularidad usando la partición que ya conocemos
    communities = [list(range(i*nodes_per_community, (i+1)*nodes_per_community)) for i in range(n_communities)]
    modularity = nx.algorithms.community.quality.modularity(G, communities)

    return final_coop_fraction, var_k, clustering, modularity

# Ejecutar simulaciones
for b in b_values:
    coop_list, var_k_list, clust_list, mod_list = [], [], [], []
    for _ in range(n_runs):
        pC, var_k, cl, mod = run_simulation_communities(b)
        coop_list.append(pC)
        var_k_list.append(var_k)
        clust_list.append(cl)
        mod_list.append(mod)
    summary_results.append({
        'network': 'Community',
        'b': b,
        'mean_pc': np.mean(coop_list),
        'std_pc': np.std(coop_list),
        'mean_var_k': np.mean(var_k_list),
        'std_var_k': np.std(var_k_list),
        'mean_clust': np.mean(clust_list),
        'std_clust': np.std(clust_list),
        'mean_mod': np.mean(mod_list),          # MOD
        'std_mod': np.std(mod_list)             # MOD
    })
    print(f"COMM | b={b:.2f} | pC={np.mean(coop_list):.3f}±{np.std(coop_list):.3f} | var(k)={np.mean(var_k_list):.2f} | C={np.mean(clust_list):.2f} | Q={np.mean(mod_list):.3f}")

# Análisis umbral b_c
df = pd.DataFrame(summary_results)
subset = df[df['network'] == 'Community']
b_vals = subset['b'].values
p_c_vals = subset['mean_pc'].values
dp_db = np.gradient(p_c_vals, b_vals)
b_c = b_vals[np.argmin(dp_db)]
print(f"Umbral crítico para red con comunidades: b_c ≈ {b_c:.2f}")

# Gráficas
def plot_with_error(x, y, yerr, title, ylabel):
    plt.figure(figsize=(10,6))
    plt.errorbar(df[x], df[y], yerr=df[yerr], label='Community', marker='o', capsize=4)
    plt.xlabel("Tentación b")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_with_error('b', 'mean_pc', 'std_pc', "Fracción de cooperadores en redes con comunidades", "⟨p_C⟩ ± σ")
plot_with_error('b', 'mean_var_k', 'std_var_k', "Varianza del grado en redes con comunidades", "⟨Var(k)⟩ ± σ")
plot_with_error('b', 'mean_clust', 'std_clust', "Coeficiente de clustering en redes con comunidades", "⟨C⟩ ± σ")
plot_with_error('b', 'mean_mod', 'std_mod', "Modularidad en redes con comunidades", "⟨Q⟩ ± σ")  # MOD
