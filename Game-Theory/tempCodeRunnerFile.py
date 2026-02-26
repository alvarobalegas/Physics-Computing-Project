import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.animation as animation

# Parámetros del dilema del prisionero
b = 0.3 #Para controlar las ganancias
R, S, T, P = 1.0, 0.0, b, 0.0 

# Datos iniciales
N0 = 100  # Nodos iniciales
p = 0.1
T = 100  # Número de pasos temporales
gif_filename = 'evolucion_red.gif'
initial_coop_prob = 0.9  # Porcentaje inicial de cooperadores (entre 0 y 1)


# Inicializamos arrays para resultados
nnodos = np.zeros(T)
vark = np.zeros(T)
C = np.zeros(T)
pc = np.zeros(T)  # Fracción de cooperadores

# Red inicial
G = nx.erdos_renyi_graph(N0, p)
graphs = [G.copy()]

# Estado inicial aleatorio de los nodos
states = {
    n: 'C' if np.random.rand() < initial_coop_prob else 'D'
    for n in G.nodes
}


# Función de payoff
def compute_payoff(node, G, states):
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

# Evolución temporal
for t in range(T):
    nuevonodo = len(G.nodes)
    G.add_node(nuevonodo)
    states[nuevonodo] = np.random.choice(['C', 'D'])  # Estado aleatorio del nuevo nodo

    for nodoexistente in list(G.nodes)[:-1]:
        if np.random.rand() < p:
            G.add_edge(nuevonodo, nodoexistente)

    # Calcular payoffs y actualizar estrategias (UI)
    payoffs = {n: compute_payoff(n, G, states) for n in G.nodes}
    new_states = states.copy()
    mutation_prob = 0.01  # pequeña probabilidad de mutación

    for node in G.nodes:
        neighbors = list(G.neighbors(node))
        if not neighbors:
            continue
        best_neighbor = max(neighbors, key=lambda n: payoffs[n])
        if payoffs[best_neighbor] > payoffs[node]:
            new_states[node] = states[best_neighbor]
        elif np.random.rand() < mutation_prob:
            new_states[node] = 'C' if states[node] == 'D' else 'D'  # mutación
    # Asegurar una fracción mínima de cooperadores
    min_coop_fraction = 0.05
    num_cooperators = sum(1 for s in new_states.values() if s == 'C')
    if num_cooperators < min_coop_fraction * len(G.nodes):
        defectors = [n for n, s in new_states.items() if s == 'D']
        if len(defectors) > 0:
            to_convert = np.random.choice(defectors, size=int(min_coop_fraction * len(G.nodes)), replace=False)
            for n in to_convert:
                new_states[n] = 'C'


    # Calcular métricas
    grados = np.array([deg for _, deg in G.degree()])
    vark[t] = np.var(grados)
    C[t] = nx.average_clustering(G)
    nnodos[t] = len(G.nodes)
    pc[t] = sum(1 for s in states.values() if s == 'C') / len(G.nodes)

    graphs.append(G.copy())

# Layout fijo para visualización
pos = nx.spring_layout(graphs[-1], seed=42)
fig, ax = plt.subplots(figsize=(6, 6))


def update(i):
    ax.clear()
    G_step = graphs[i]
    nx.draw(
        G_step, pos, ax=ax,
        node_size=50, node_color='skyblue',
        edge_color='gray', with_labels=False
    )
    ax.set_title(f"Paso {i}", fontsize=14)
    ax.set_axis_off()


# Animación
ani = animation.FuncAnimation(fig, update, frames=len(graphs), interval=100)
ani.save(gif_filename, writer='pillow')
plt.close()
print(f"GIF generado exitosamente: {gif_filename}")


# Gráficas
plt.figure(figsize=(15,5))


plt.plot(nnodos, vark, label='Varianza del grado')
plt.xlabel("Número de nodos")
plt.ylabel("Var(k)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.plot(nnodos, C, label='Coeficiente de agrupamiento', color='orange')
plt.xlabel("Número de nodos")
plt.ylabel("C")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


plt.plot(nnodos, pc, label='Fracción de cooperadores', color='green')
plt.xlabel("Número de nodos")
plt.ylabel("p_C(t)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

"""
# Matriz de adyacencia final
A = nx.to_numpy_array(G)
plt.figure(figsize=(6,6))
plt.matshow(A, cmap='Greys')
plt.title("Matriz de adyacencia final de la red")
plt.xlabel("Nodo")
plt.ylabel("Nodo")
plt.colorbar(label="Conexión")
plt.show()
"""