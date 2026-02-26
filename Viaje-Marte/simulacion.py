#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#Datos necesarios 
G = 6.67e-11 #Constante de gravitacion
M_T = 5.9736e24 #Masa de la tierra
M_M = 0.642e24# Masa de Marte
M_S = 1.9885e30 #Masa del Sol
OMEGA_T = 1.99e-7 #Velocidad con la que la Tierra alrededor del Sol
OMEGA_M = 1.06e-7 #Velocidad con la que Marte gira alrededor del Sol
R_T = 6.37816e6 #Radio de la tierra
R_M = 3.396e6 #Radio de marte
D_M = 228e9 #Distancia de marte al Sol
D_T = 146.9e9 #Distancia tierra al Sol
#Nuevas constantes
delta = (G*M_S) / (D_T**3)
mu_T = M_T / M_S
mu_M = M_M / M_S
lambdaa = D_M / D_T
r_t0 = 2e6 #Distancia inicial del cohete respecto a la Tierra

#Definimos los arrays que nos van a hacer falta
h = 60 #Paso temporal, en teoria tiene que ser de un minuto
N = 300000  #Tiempo total 
x_T = np.zeros(N)
y_T = np.zeros(N)
x_M = np.zeros(N)
y_M = np.zeros(N)
x_N = np.zeros(N)
y_N = np.zeros(N)
r = np.zeros(N)
p_r = np.zeros(N)
phi = np.zeros(N)
p_phi = np.zeros(N)
H = np.zeros(N)


#Definimos las funciones para calcular la distancias entre la nave, sol y marte

def fdistanciat(r, phi, t): #Distancia entre la nave y la Tierra
    r_T = np.sqrt(1+r**2-2*r*np.cos(phi-OMEGA_T*t))
    return r_T

def fdistanciam(r, phi, t):
    r_M = np.sqrt(r**2+lambdaa**2-2*r*lambdaa*np.cos(phi-OMEGA_M*t))
    return r_M 

R_t0 = (r_t0+R_T) / D_T #Distancia al centro de la Tierra respecto de la nave
#Definimos las condiciones iniciales 
r[0] = 1+R_t0
a = 0.5*(r[0]+lambdaa) #Semieje mayor de la orbita de transferencia
phi[0] = 0
p_r[0] = 1.1675*np.sqrt(2*delta*mu_T/R_t0)   #Impulso radial para salir del pozo de potencial terrestre
p_phi[0] = r[0]*np.sqrt(2*delta*((1/r[0])-(1/(2*a)))) #p_phi para escapar de la Tierra y entrar en orbita de transferencia

#Posiciones iniciales 

phi0_M = 0.3639185350402585#Angulo inicial que tiene marte
x_T[0] = 1
y_T[0] = 0
x_M[0] = lambdaa*np.cos(phi0_M)
y_M[0] = lambdaa*np.sin(phi0_M)
x_N[0] = r[0]
y_N[0] = 0

"""
Definir energías
"""


#Creamos las funciones que nos hacen falta

def f0(pr): 
    return pr

def f1(pphi, r):
    phi_punto = pphi / r**2
    return phi_punto

def f2(pphi, r, rT, rM, phi, t): 
    pr_punto = (pphi**2 / r**3) - delta * ((1 / r**2) + (mu_T / rT**3)*(r-np.cos(phi-OMEGA_T*t))+(mu_M / rM**3)*(r-lambdaa*np.cos(phi-OMEGA_M*t)))
    return pr_punto

def f3(r, rT, rM, phi, t):
    pphi_punto = - delta*r*((mu_T / rT**3)*np.sin(phi-OMEGA_T*t)+(mu_M*lambdaa/rM**3)*np.sin(phi-OMEGA_M*t))
    return pphi_punto

plt.plot(x_N[0], y_N[0], 'bo', label='Nave')
plt.plot(x_T[0], y_T[0], 'go', label='Tierra')
plt.plot(x_M[0], y_M[0], 'ro', label='Marte')
plt.plot(0, 0, 'yo', label='Sol')
plt.axis("equal")
plt.legend()
plt.grid()
plt.title("Posiciones iniciales")
plt.show()



for t in range(1, N):
    tiempo = (t-1) * h
    rT = fdistanciat(r[t-1], phi[t-1], tiempo)
    rM = fdistanciam(r[t-1], phi[t-1], tiempo)

    # k1
    k1_0 = h * f0(p_r[t-1])
    k1_1 = h * f1(p_phi[t-1], r[t-1])
    k1_2 = h * f2(p_phi[t-1], r[t-1], rT, rM, phi[t-1], tiempo)
    k1_3 = h * f3(r[t-1], rT, rM, phi[t-1], tiempo)

    # rT y rM para k2
    rT2 = fdistanciat(r[t-1] + k1_0 / 2, phi[t-1] + k1_1 / 2, tiempo + h/2)
    rM2 = fdistanciam(r[t-1] + k1_0 / 2, phi[t-1] + k1_1 / 2, tiempo + h/2)
    
    # k2
    k2_0 = h * f0(p_r[t-1] + k1_2 / 2)
    k2_1 = h * f1(p_phi[t-1] + k1_3 / 2, r[t-1] + k1_0 / 2)
    k2_2 = h * f2(p_phi[t-1] + k1_3 / 2, r[t-1] + k1_0 / 2, rT2, rM2, phi[t-1] + k1_1 / 2, tiempo + h/2)
    k2_3 = h * f3(r[t-1] + k1_0 / 2, rT2, rM2, phi[t-1] + k1_1 / 2, tiempo + h/2)

    # rT y rM para k3
    rT3 = fdistanciat(r[t-1] + k2_0 / 2, phi[t-1] + k2_1 / 2, tiempo + h/2)
    rM3 = fdistanciam(r[t-1] + k2_0 / 2, phi[t-1] + k2_1 / 2, tiempo + h/2)

    # k3
    k3_0 = h * f0(p_r[t-1] + k2_2 / 2)
    k3_1 = h * f1(p_phi[t-1] + k2_3 / 2, r[t-1] + k2_0 / 2)
    k3_2 = h * f2(p_phi[t-1] + k2_3 / 2, r[t-1] + k2_0 / 2, rT3, rM3, phi[t-1] + k2_1 / 2, tiempo + h/2)
    k3_3 = h * f3(r[t-1] + k2_0 / 2, rT3, rM3, phi[t-1] + k2_1 / 2, tiempo + h/2)

    # rprima para k4
    rT4 = fdistanciat(r[t-1] + k3_0, phi[t-1] + k3_1, tiempo + h)
    rM4 = fdistanciam(r[t-1] + k3_0, phi[t-1] + k3_1, tiempo + h)
    
    # k4
    k4_0 = h * f0(p_r[t-1] + k3_2)
    k4_1 = h * f1(p_phi[t-1] + k3_3, r[t-1] + k3_0)
    k4_2 = h * f2(p_phi[t-1] + k3_3, r[t-1] + k3_0, rT4, rM4, phi[t-1] + k3_1, tiempo + h)
    k4_3 = h * f3(r[t-1] + k3_0, rT4, rM4, phi[t-1] + k3_1, tiempo + h)

    # Actualizamos las variables
    r[t] = r[t-1] + (k1_0 + 2*k2_0 + 2*k3_0 + k4_0) / 6
    phi[t] = phi[t-1] + (k1_1 + 2*k2_1 + 2*k3_1 + k4_1) / 6
    p_r[t] = p_r[t-1] + (k1_2 + 2*k2_2 + 2*k3_2 + k4_2) / 6
    p_phi[t] = p_phi[t-1] + (k1_3 + 2*k2_3 + 2*k3_3 + k4_3) / 6

    # Calculamos coordenadas cartesianas del cohete, la Tierra y Marte
    tiempo2 = t*h 
    x_N[t] = r[t] * np.cos(phi[t]) 
    y_N[t] = r[t] * np.sin(phi[t]) 
    x_T[t] = np.cos(OMEGA_T*tiempo2)
    y_T[t] = np.sin(OMEGA_T*tiempo2)
    x_M[t] = lambdaa*np.cos(phi0_M + OMEGA_M*tiempo2)
    y_M[t] = lambdaa*np.sin(phi0_M + OMEGA_M*tiempo2)
    

# Graficamos la trayectoria del cohete, marte y la Tierra
plt.figure(figsize=(8, 8))
plt.plot(x_N, y_N, label="Cohete", color='blue')
plt.plot(x_T, y_T, label="Tierra", color='green')
plt.plot(x_M, y_M, label = "Marte", color = 'red')
plt.plot(0, 0, 'o', label="Sol", color='yellow')  # Sol en el origen

plt.xlabel("x [AU] reescalado")
plt.ylabel("y [AU] reescalado")
plt.title("Trayectoria del cohete, la tierra y Marte")
plt.legend()
plt.axis("equal")
plt.grid(True)
plt.tight_layout()
plt.show()

