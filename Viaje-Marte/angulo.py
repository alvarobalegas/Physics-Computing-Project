import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Datos necesarios
G = 6.67e-11              # Constante de gravitación
M_T = 5.9736e24           # Masa de la Tierra
M_M = 0.642e24            # Masa de Marte
M_S = 1.9885e30           # Masa del Sol
OMEGA_T = 1.99e-7         # Velocidad angular de la Tierra [rad/s]
OMEGA_M = 1.06e-7         # Velocidad angular de Marte [rad/s]
R_T = 6.37816e6           # Radio de la Tierra [m]
R_M = 3.396e6             # Radio de Marte [m]
D_T = 146.9e9             # Distancia Tierra-Sol [m]
D_M = 228e9               # Distancia Marte-Sol [m]

# Constantes derivadas
delta   = (G * M_S) / (D_T**3)
mu_T    = M_T / M_S
mu_M    = M_M / M_S
lambdaa = D_M / D_T      # Radio orbital de Marte (en unidades de D_T)
r_t0    = 2e6            # Altura inicial sobre la Tierra [m]

# Parámetros de integración
dt = 60                   # Paso temporal [s]
N  = 280_000              # Número de pasos (~195 días)

# Función que define todos los arrays necesarios para la simulación
def definir_arrays():
    x_T   = np.zeros(N)
    y_T   = np.zeros(N)
    x_M   = np.zeros(N)
    y_M   = np.zeros(N)
    x_N   = np.zeros(N)
    y_N   = np.zeros(N)
    r     = np.zeros(N)
    p_r   = np.zeros(N)
    phi   = np.zeros(N)
    p_phi = np.zeros(N)
    H     = np.zeros(N)
    return x_T, y_T, x_M, y_M, x_N, y_N, r, p_r, phi, p_phi, H

# Distancias a Tierra y Marte en coordenadas reescaladas
def fdist_t(r, phi, t):
    return np.sqrt(1 + r**2 - 2*r*np.cos(phi - OMEGA_T*t))

def fdist_m(r, phi, t):
    return np.sqrt(r**2 + lambdaa**2 - 2*r*lambdaa*np.cos(phi - OMEGA_M*t))

# Derivadas del sistema Hamiltoniano

def dr_dt(pr):
    return pr

def dphi_dt(pphi, r):
    return pphi / r**2

def dpr_dt(r, pphi, phi, t):
    rT = fdist_t(r, phi, t)
    rM = fdist_m(r, phi, t)
    return (pphi**2 / r**3) - delta * (1/r**2 + mu_T*(r - np.cos(phi - OMEGA_T*t))/rT**3
                                       + mu_M*(r - lambdaa*np.cos(phi - OMEGA_M*t))/rM**3)

def dpphi_dt(r, phi, t):
    rT = fdist_t(r, phi, t)
    rM = fdist_m(r, phi, t)
    return -delta * r * (mu_T*np.sin(phi - OMEGA_T*t)/rT**3
                         + mu_M*lambdaa*np.sin(phi - OMEGA_M*t)/rM**3)

# Estima el tiempo y ángulo al afelio (r = lambdaa) con RK4 completo
def estimar_t_aphelio(r0, p_r0, p_phi0, tol=1e-3, max_steps=N):
    r_val, pr_val, phi_val, pphi_val = r0, p_r0, 0.0, p_phi0
    for i in range(max_steps):
        t = i * dt
        # RK4
        k1_r    = dt * dr_dt(pr_val)
        k1_phi  = dt * dphi_dt(pphi_val, r_val)
        k1_pr   = dt * dpr_dt(r_val, pphi_val, phi_val, t)
        k1_pphi = dt * dpphi_dt(r_val, phi_val, t)

        k2_r    = dt * dr_dt(pr_val + 0.5*k1_pr)
        k2_phi  = dt * dphi_dt(pphi_val + 0.5*k1_pphi, r_val + 0.5*k1_r)
        k2_pr   = dt * dpr_dt(r_val + 0.5*k1_r, pphi_val + 0.5*k1_pphi, phi_val + 0.5*k1_phi, t + 0.5*dt)
        k2_pphi = dt * dpphi_dt(r_val + 0.5*k1_r, phi_val + 0.5*k1_phi, t + 0.5*dt)

        k3_r    = dt * dr_dt(pr_val + 0.5*k2_pr)
        k3_phi  = dt * dphi_dt(pphi_val + 0.5*k2_pphi, r_val + 0.5*k2_r)
        k3_pr   = dt * dpr_dt(r_val + 0.5*k2_r, pphi_val + 0.5*k2_pphi, phi_val + 0.5*k2_phi, t + 0.5*dt)
        k3_pphi = dt * dpphi_dt(r_val + 0.5*k2_r, phi_val + 0.5*k2_phi, t + 0.5*dt)

        k4_r    = dt * dr_dt(pr_val + k3_pr)
        k4_phi  = dt * dphi_dt(pphi_val + k3_pphi, r_val + k3_r)
        k4_pr   = dt * dpr_dt(r_val + k3_r, pphi_val + k3_pphi, phi_val + k3_phi, t + dt)
        k4_pphi = dt * dpphi_dt(r_val + k3_r, phi_val + k3_phi, t + dt)

        r_val   += (k1_r + 2*k2_r + 2*k3_r + k4_r)/6
        phi_val += (k1_phi + 2*k2_phi + 2*k3_phi + k4_phi)/6
        pr_val  += (k1_pr + 2*k2_pr + 2*k3_pr + k4_pr)/6
        pphi_val+= (k1_pphi + 2*k2_pphi + 2*k3_pphi + k4_pphi)/6

        if abs(r_val - lambdaa) < tol:
            return t, phi_val
    return None, None

# Calcula el ángulo inicial de Marte para encuentro en afelio
def calcular_phi0M(t_transf, phi_nave, omega_m=OMEGA_M):
    phi0 = phi_nave - omega_m * t_transf
    return phi0 % (2 * np.pi)

# ------- Bloque principal -------
# Crear arrays
x_T, y_T, x_M, y_M, x_N, y_N, r, p_r, phi, p_phi, H = definir_arrays()

# Condiciones iniciales
dist_rel  = (r_t0 + R_T) / D_T
r[0]      = 1 + dist_rel
phi[0]    = 0.0
# Impulsos iniciales
a         = 0.5 * (r[0] + lambdaa)
p_r[0]    = 1.1675 * np.sqrt(2 * delta * mu_T / dist_rel)
p_phi[0]  = r[0] * np.sqrt(2 * delta * ((1/r[0]) - (1/(2*a))))

# Estimamos tiempo y ángulo al afelio
t_aphelio, phi_aphelio = estimar_t_aphelio(r[0], p_r[0], p_phi[0])
if t_aphelio is None:
    raise RuntimeError("No se alcanzó la distancia de Marte en la simulación de transferencia")

# Calculamos phi0_M
phi0_M = calcular_phi0M(t_aphelio, phi_aphelio)
print(f"Tiempo de transferencia ~ {t_aphelio/3600:.1f} h, phi_\u2070_M={phi0_M:.3f} rad")

# Posiciones iniciales
x_T[0], y_T[0] = 1, 0
x_M[0], y_M[0] = lambdaa * np.cos(phi0_M), lambdaa * np.sin(phi0_M)
x_N[0], y_N[0] = r[0], 0

# (Aquí continúa tu bucle RK4 completo y la visualización final)

# Ejemplo de gráfico inicial
plt.figure(figsize=(6,6))
plt.plot(x_N[0], y_N[0], 'bo', label='Nave')
plt.plot(x_T[0], y_T[0], 'go', label='Tierra')
plt.plot(x_M[0], y_M[0], 'ro', label='Marte')
plt.plot(0,0,'yo',label='Sol')
plt.axis('equal'); plt.legend(); plt.grid(True)
plt.title('Configuración inicial con phi0_M')
