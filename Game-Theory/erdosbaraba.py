
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Parámetros de la simulación
N0 = 100  # nodos iniciales
T = 900    # pasos temporales por simulación (para acortar el cálculo)
p = 0.1   # probabilidad de conexión para ER
m = 3     # parámetro de crecimiento para BA (número de enlaces por nodo)
mutation_prob = 0.01
min_coop_fraction = 0.05
initial_coop_prob = 0.5

# Rango de b para barrer
b_values = np.linspace(0.5, 2.0, 10)

# Redes a analizar
network_types = ['ER', 'BA']

# Guardar resultados
results = {
    'network': [],
    'b': [],
    'final_var_degree': [],
    'final_clustering': [],
    'final_coop_fraction': []
}

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

def run_simulation(network_type, b):

    R, S, T_payoff, P = 1.0, 0.0, b, 0.0

    # Crear red inicial
    if network_type == 'ER':
        G = nx.erdos_renyi_graph(N0, p)
    elif network_type == 'BA':
        G = nx.barabasi_albert_graph(N0, m)
    else:
        raise ValueError("Network type no soportado")

    # Estados iniciales
    states = {
        n: 'C' if np.random.rand() < initial_coop_prob else 'D'
        for n in G.nodes
    }

    for t in range(T):
        # Añadir un nodo nuevo
        nuevonodo = len(G.nodes)
        G.add_node(nuevonodo)
        states[nuevonodo] = np.random.choice(['C', 'D'])

        # Conectar nuevo nodo
        if network_type == 'ER':
            for nodoexistente in list(G.nodes)[:-1]:
                if np.random.rand() < p:
                    G.add_edge(nuevonodo, nodoexistente)
        elif network_type == 'BA':
            # En BA, nuevo nodo conecta a m nodos con preferencia grado
            targets = _preferential_attachment_targets(G, m)
            for target in targets:
                G.add_edge(nuevonodo, target)

        # Calcular payoffs
        payoffs = {n: compute_payoff(n, G, states, R, S, T_payoff, P) for n in G.nodes}

        # Actualizar estrategias (UI)
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

        # Mantener mínimo cooperadores
        num_coop = sum(1 for s in new_states.values() if s == 'C')
        if num_coop < min_coop_fraction * len(G.nodes):
            defectors = [n for n, s in new_states.items() if s == 'D']
            if len(defectors) > 0:
                to_convert = np.random.choice(defectors, size=int(min_coop_fraction * len(G.nodes)), replace=False)
                for n in to_convert:
                    new_states[n] = 'C'

        states = new_states

    # Métricas finales
    grados = np.array([deg for _, deg in G.degree()])
    var_degree = np.var(grados)
    clustering = nx.average_clustering(G)
    coop_fraction = sum(1 for s in states.values() if s == 'C') / len(G.nodes)

    return var_degree, clustering, coop_fraction

def _preferential_attachment_targets(G, m):
    # Selección preferencial por grado para BA
    degrees = np.array([deg for _, deg in G.degree()])
    nodes = np.array(list(G.nodes()))
    probs = degrees / degrees.sum()
    targets = np.random.choice(nodes, size=m, replace=False, p=probs)
    return targets

# Ejecutar barrido
for net_type in network_types:
    print(f"Simulando para red {net_type}")
    for b in b_values:
        var_k, c, p_c = run_simulation(net_type, b)
        results['network'].append(net_type)
        results['b'].append(b)
        results['final_var_degree'].append(var_k)
        results['final_clustering'].append(c)
        results['final_coop_fraction'].append(p_c)
        print(f"b={b:.2f} var(k)={var_k:.3f} C={c:.3f} p_C={p_c:.3f}")

# Resultados: ejemplo de análisis gráfico
import pandas as pd
df = pd.DataFrame(results)

import seaborn as sns

plt.figure(figsize=(12,6))
sns.lineplot(data=df, x='b', y='Fraccion de cooperadores', hue='network', marker='o')
plt.title("Fracción final de cooperadores vs b")
plt.grid(True)
plt.show()

plt.figure(figsize=(12,6))
sns.scatterplot(data=df, x='Varianza del grado', y='Fracción de cooperadores', hue='network', style='network', s=100)
plt.title("Fracción final de cooperadores vs varianza del grado")
plt.grid(True)
plt.show()

plt.figure(figsize=(12,6))
sns.scatterplot(data=df, x='Coeficiente de agrupamiento', y='Fracción de cooperadores', hue='network', style='network', s=100)
plt.title("Fracción final de cooperadores vs coeficiente de clustering")
plt.grid(True)
plt.show()
