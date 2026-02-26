import numpy as np
import matplotlib.pyplot as plt

# Función auxiliar para anular y renormalizar zona específica
def proyectar(phi, j_min, j_max):
    phi[j_min:j_max + 1] = 0
    norma = np.sum(np.abs(phi)**2)
    if norma > 0:
        phi /= np.sqrt(norma)
    return phi

# Cálculo de K teórico
def K_teorico(lambdaa, n_ciclos):
    if lambdaa < 1:
        num = 4 * (1 - lambdaa)
        denom = 4 * (1 - lambdaa) + lambdaa**2 * np.sin((2 * np.pi / 5) * n_ciclos * np.sqrt(1 - lambdaa))**2
        return num / denom
    elif lambdaa > 1:
        num = 4 * (1 - lambdaa)
        denom = 4 * (lambdaa - 1) + lambdaa**2 * np.sinh((2 * np.pi / 5) * n_ciclos * np.sqrt(lambdaa - 1))**2
        return num / denom
    else:
        return 0.0

# Parámetros fijos
m = 1000       # Número de simulaciones por punto       
dt_factor = 0.25  # Factor para el paso temporal    # Número de ciclos fijo en la gaussiana

Ns = [500, 1000, 2000]                     
lambdas = [5, 10]        
K_results = np.zeros((len(lambdas), len(Ns)))  

for li, lambdaa in enumerate(lambdas):
    for ni, N in enumerate(Ns):
        h = 1
        n_ciclos = N/4
        k_0 = 2 * np.pi * n_ciclos / N
        k0_barra = k_0 * h
        s_barra = 0.25 / (k0_barra**2)
        s = dt_factor * h**2

        # Barrera de potencial centrada 
        V_j = np.zeros(N + 1)
        j_inicial = int((2*N)/5)
        j_final = int ((3*N)/5)
        V_j[j_inicial:j_final + 1] = lambdaa * k0_barra**2

        # Detectores
        lim_izq = int(N/5)
        lim_der = int((N/5) + 1)

        # Estimar tiempo de viaje del paquete a la barrera
        x0_j = int(0.15 * N)  # Centro del paquete en índices
        j_ini = int(2 * N / 5)  # Inicio de la barrera en índices

        # Velocidad de grupo en índices
        k_0 = 2 * np.pi * n_ciclos / N
        v_g = 2 * np.sin(k_0 / 2)  # Ya sin factor h, porque j es adimensional

        # Tiempo de vuelo en pasos de malla
        t_viaje = (j_ini - x0_j) / v_g
        n_D = max(int(t_viaje / s_barra), 10)

        m_T = 0  # Contador de transmisiones

        for sim in range(m):

            # Función de onda inicial físicamente coherente
            sigma_j = N / 16
            j_centro = N / 4
            j = np.arange(N+1)
            phi = np.exp(1j * k0_barra * j) * np.exp(-((j - j_centro)**2) / (2 * sigma_j**2))
            phi[0] = 0
            phi[N] = 0
            phi /= np.sqrt(np.sum(np.abs(phi)**2))

            terminado = False
            while not terminado:

                for t in range(n_D):
                    A_p = 1
                    A_0 = -2 + (2j / s_barra) - V_j
                    A_m = 1

                    b_jn = np.zeros(N, dtype=complex)
                    b_jn[1:N] = (4j / s_barra) * phi[1:N]

                    alpha = np.zeros(N, dtype=complex)
                    beta = np.zeros(N, dtype=complex)
                    chi = np.zeros(N + 1, dtype=complex)

                    alpha[N - 1] = 0
                    beta[N - 1] = 0

                    for jj in range(N - 2, -1, -1):
                        gamma = 1 / (A_0[jj + 1] + A_p * alpha[jj + 1])
                        alpha[jj] = -gamma * A_m
                        beta[jj] = gamma * (b_jn[jj + 1] - A_p * beta[jj + 1])

                    for w in range(1, N - 1):
                        chi[w] = alpha[w - 1] * chi[w - 1] + beta[w - 1]

                    phi = chi - phi
                    phi[0] = 0
                    phi[N] = 0

                    norm = np.sum(np.abs(phi)**2)
                    if norm > 0:
                        phi /= np.sqrt(norm)

                P_D = np.sum(np.abs(phi[lim_der:])**2)

                if np.random.rand() < P_D:
                    m_T += 1
                    terminado = True
                    continue

                phi = proyectar(phi, lim_izq, N)

                P_I = np.sum(np.abs(phi[:lim_izq + 1])**2)

                if np.random.rand() < P_I:
                    terminado = True
                    continue

                phi = proyectar(phi, 0, lim_der)

        K = m_T / m
        K_results[li, ni] = K

        K_teo = K_teorico(lambdaa, n_ciclos)
        print(f"λ = {lambdaa:.2f}, N = {N} → K_sim = {K:.4f}, K_teórico = {K_teo:.4f}")

plt.figure(figsize=(8, 6))
for li, lambdaa in enumerate(lambdas):
    plt.plot(Ns, K_results[li], 'o-', label=f"Simulación λ = {lambdaa:.2f}")
    K_teo = K_teorico(lambdaa, n_ciclos)
    plt.hlines(K_teo, Ns[0], Ns[-1], colors='gray', linestyles='--', label=f"Teórico λ = {lambdaa:.2f}")

plt.xlabel("Tamaño de la malla N")
plt.ylabel("Coeficiente de transmisión K")
plt.title("Coeficiente de transmisión K según N para distintos λ")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('KfrenteaN_teorico.pdf')
plt.show()
