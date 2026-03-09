#%%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors

#Datos necesarios 
T = 0

#Defino el número de neuronas
N = 400

#Conjunto de P memorias
valores_P = np.arange(10, 110, 10)
valores_m = [] #Lista vacia para calcular los valores de m 
recuperado = [] #Lista vacia para calcular los patrones que se pueden almacenar

for P in valores_P:

    tamaño = N // P
    xi = np.zeros((P, N), dtype = int)

    for mu in range(P):
        inicio = mu * tamaño
        fin = (mu + 1) * tamaño
        xi[mu, inicio:fin] = 1

    a = np.sum(xi)/ (N*P) #ACTIVACIÓN MEDIA DE LOS PATRONES

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

    #Condicion inicial del caso 1 b

    s = np.copy(xi[0]) #Copiamos la fila cero con tamaño N
    #Creamos un array con el tamaño de s con numeros aleatorios entre 0 y 1 y esta es nuestra probabilidad
    deformado = np.random.rand(N) < 0.1 #Condicion para que la probabilidad sea menor que 0.1
    s[deformado] = 1 - s[deformado] #Donde se cumpla cambio los cero por los 1 o viceversa


    for _ in range(100*N): 
        j = np.random.randint(0, N)
        #Calculamos el campo local
        h_j = np.sum(w[j, :]*s)
        #Calculamos la probabilidad de que la neurona se dispare
        P_j = 1 if h_j>theta[j] else 0 
        u = np.random.rand()
        s[j] = 1 if u < P_j else 0
            
    m = np.sum((xi[0] - a) * (s - a)) / (a * (1 - a) * N)
    valores_m.append(m)
    
    recuperado.append(int(abs(m)>0.75))

valores_m = np.array(valores_m)
recuperado = np.array(recuperado)

# Encontrar el mayor P tal que todos los anteriores se recuperaron
fallos = np.where(recuperado == 0)[0]
if len(fallos) > 0:
    idx_max = fallos[0]
else:
    idx_max = len(recuperado)

P_max_recuperado = valores_P[idx_max - 1] if idx_max > 0 else 0
fraccion_recuperado = P_max_recuperado / N

plt.plot(valores_P, valores_m, marker='o')
plt.axhline(0.75, color='r', linestyle='--', label='Umbral de recuperación')
plt.xlabel('Número de patrones P')
plt.ylabel('Solapamiento con patrón 0')
plt.title('Solapamiento final vs número de patrones')
plt.legend()
plt.grid()
plt.savefig('figura_hopfield2.png')
plt.show()

# Gráfica de recuperación (1 si se recuperó, 0 si no)
plt.figure(figsize=(8, 4))
plt.plot(valores_P, recuperado, 'go--', label='Patrón recuperado')
plt.xlabel('Número de patrones P')
plt.ylabel('¿Recuperado? (1=SÍ, 0=NO)')
plt.title('Recuperación de patrón vs número de patrones')
plt.ylim(-0.1, 1.1)
plt.grid()
plt.tight_layout()
plt.show()

print(f"Máximo P recuperado correctamente: {P_max_recuperado}")
print(f"Fracción máxima de patrones que la red puede almacenar: {fraccion_recuperado:.2f}")
# %%
