#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors

#Datos necesarios 
valores_T = np.linspace(0, 3, 100) #Array de temperaturas
#Defino el número de neuronas
N = 400

#Conjunto de P memorias
P = 1
valores_m = []
xi = np.zeros((1, N), dtype=int)
xi[0, :N // 4] = 1

a = np.sum(xi) / (N*P) #ACTIVACIÓN MEDIA DE LOS PATRONES

#Calculo la matriz de pesos sinapticos
w = np.zeros((N, N))

sumatorio1 = 0
for mu1 in range(P):
    for i in range(N):
        for j in range(N):
            w[i, j] += (xi[mu1, i]-a)*(xi[mu1, j]-a)

w = np.copy(w)/(a*(1-a)*N) #MATRIZ DE PESOS SINÁPTICOS

#Calculamos el vector de umbrales
theta = np.zeros(N) 

theta = np.sum(w, axis = 1)

theta = 0.5*np.copy(theta) #Vector de umbrales

for T in valores_T:
    
    #Condicion inicial del caso 1 b
    s = np.copy(xi[0]) #Copiamos la fila cero con tamaño N
    #Creamos un array con el tamaño de s con numeros aleatorios entre 0 y 1 y esta es nuestra probabilidad
    deformado = np.random.rand(N) < 0.1 #Condicion para que la probabilidad sea menor que 0.1
    s[deformado] = 1 - s[deformado] #Donde se cumpla cambio los cero por los 1 o viceversa

    
    if T == 0:
        for _ in range(50*N): 
            j = np.random.randint(0, N)
            #Calculamos el campo local
            h_j = np.sum(w[j, :]*s)
            #Calculamos la probabilidad de que la neurona se dispare
            P_j = 1 if h_j>theta[j] else 0 
            u = np.random.rand()
            s[j] = 1 if u < P_j else 0
            
        m = np.sum((xi[0] - a) * (s - a)) / (a * (1 - a) * N)
        valores_m.append(m)

    else: 
        beta = 1/T
        for _ in range(50*N):
        #Elegimos una neurona j al azar
            j = np.random.randint(0, N)
            #Calculamos el campo local
            h_j = np.sum(w[j, :]*s)
            #Calculamos la probabilidad de que la neurona se dispare
            P_j = 0.5*(1+np.tanh(beta*(h_j-theta[j])))
            u = np.random.rand()
            s[j] = 1 if P_j > u else 0
            
        m = np.sum((xi[0] - a) * (s - a)) / (a * (1 - a) * N)
        valores_m.append(m)

plt.plot(valores_T, valores_m, marker='o')
plt.axhline(0.75, color='r', linestyle='--', label='Umbral de recuperación')
plt.xlabel('Temperatura')
plt.ylabel('Solapamiento con patrón 0')
plt.title('Solapamiento final vs temperatura')
plt.legend()
plt.grid()
plt.show()


# %%
