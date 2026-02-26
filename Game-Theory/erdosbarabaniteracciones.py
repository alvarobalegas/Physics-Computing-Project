import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# Parámetros
N0 = 800
T = 300
p = 0.02
m = 4
mutation_prob = 0.001
min_coop_fraction = 0.05
initial_coop_prob = 0.5
n_runs = 200

b_values = np.linspace(0.5, 2.0, 10)
network_types = ['ER', 'BA']

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

def _preferential_attachment_targets(G, m):
    degrees = np.array([deg for _, deg in G.degree()])
    nodes = np.array(list(G.nodes()))
    probs = degrees / degrees.sum()
    targets = np.random.choice(nodes, size=m, replace=False, p=probs)
    return targets

def run_simulation(network_type, b):
    R, S, T_payoff, P = 1.0, 0.0, b, 0.0

    if network_type == 'ER':
        G = nx.erdos_renyi_graph(N0, p)
    elif network_type == 'BA':
        G = nx.barabasi_albert_graph(N0, m)
    else:
        raise ValueError("Tipo de red no soportado")

    states = {
        n: 'C' if np.random.rand() < initial_coop_prob else 'D'
        for n in G.nodes
    }

    for t in range(T):
        new_node = len(G.nodes)
        G.add_node(new_node)
        states[new_node] = np.random.choice(['C', 'D'])

        if network_type == 'ER':
            for existing in list(G.nodes)[:-1]:
                if np.random.rand() < p:
                    G.add_edge(new_node, existing)
        elif network_type == 'BA':
            targets = _preferential_attachment_targets(G, m)
            for target in targets:
                G.add_edge(new_node, target)

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
        """
        num_coop = sum(1 for s in new_states.values() if s == 'C')
        if num_coop < min_coop_fraction * len(G.nodes):
            defectors = [n for n, s in new_states.items() if s == 'D']
            if len(defectors) > 0:
                to_convert = np.random.choice(defectors, size=int(min_coop_fraction * len(G.nodes)), replace=False)
                for n in to_convert:
                    new_states[n] = 'C'
        """
        states = new_states

    final_coop_fraction = sum(1 for s in states.values() if s == 'C') / len(G.nodes)
    var_k = np.var([deg for _, deg in G.degree()])
    clustering = nx.average_clustering(G)

    return final_coop_fraction, var_k, clustering

# Ejecutar simulaciones
for net_type in network_types:
    for b in b_values:
        coop_list, var_k_list, clust_list = [], [], []
        for _ in range(n_runs):
            pC, var_k, cl = run_simulation(net_type, b)
            coop_list.append(pC)
            var_k_list.append(var_k)
            clust_list.append(cl)
        summary_results.append({
            'network': net_type,
            'b': b,
            'mean_pc': np.mean(coop_list),
            'std_pc': np.std(coop_list),
            'mean_var_k': np.mean(var_k_list),
            'std_var_k': np.std(var_k_list),
            'mean_clust': np.mean(clust_list),
            'std_clust': np.std(clust_list)
        })
        print(f"{net_type} | b={b:.2f} | pC={np.mean(coop_list):.3f}±{np.std(coop_list):.3f} | var(k)={np.mean(var_k_list):.2f} | C={np.mean(clust_list):.2f}")

# DataFrame
df = pd.DataFrame(summary_results)

df_sorted = df.sort_values(['network', 'b'])
for net in df_sorted['network'].unique():
    subset = df_sorted[df_sorted['network'] == net]
    b_vals = subset['b'].values
    p_c_vals = subset['mean_pc'].values  # Aquí estaba el error
    dp_db = np.gradient(p_c_vals, b_vals)
    b_c = b_vals[np.argmin(dp_db)]
    print(f"Umbral crítico para red {net}: b_c ≈ {b_c:.2f}")


# Definir valores b cercanos al umbral crítico para cada red
b_focus = {
    'ER': [1.20, 1.25, 1.30, 1.33, 1.36, 1.40],
    'BA': [1.00, 1.10, 1.15, 1.17, 1.20, 1.25]
}


# Guardar histogramas de pC para ver bimodalidad
import matplotlib.pyplot as plt

for net in ['ER', 'BA']:
    for b in b_focus[net]:
        coop_samples = []
        for _ in range(n_runs):
            pC, _, _ = run_simulation(net, b)
            coop_samples.append(pC)
        plt.figure(figsize=(7,5))
        plt.hist(coop_samples, bins=20, alpha=0.7, color='tab:blue', edgecolor='black')
        plt.title(f"Distribución de fracción cooperadores (pC) para {net}, b={b:.2f}")
        plt.xlabel("Fracción cooperadores (pC)")
        plt.ylabel("Frecuencia")
        plt.grid(True)
        plt.show()




# === GRÁFICAS ===
import seaborn as sns

def plot_with_error(x, y, yerr, hue, title, ylabel):
    plt.figure(figsize=(10,6))
    for net in df[hue].unique():
        data = df[df[hue] == net]
        plt.errorbar(data[x], data[y], yerr=data[yerr], label=net, marker='o', capsize=4)
    plt.xlabel("Tentación b")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Gráfico 1: Fracción de cooperadores
plot_with_error('b', 'mean_pc', 'std_pc', 'network', "Fracción de cooperadores vs b", "⟨p_C⟩ ± σ")

# Gráfico 2: Varianza del grado
plot_with_error('b', 'mean_var_k', 'std_var_k', 'network', "Varianza del grado vs b", "⟨Var(k)⟩ ± σ")

# Gráfico 3: Clustering
plot_with_error('b', 'mean_clust', 'std_clust', 'network', "Coeficiente de clustering vs b", "⟨C⟩ ± σ")


