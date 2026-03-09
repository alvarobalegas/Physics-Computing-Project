#%%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors

#Datos necesarios 
T = 0.1
beta = 1/T 

#Defino el número de neuronas
N = 100

#Conjunto de P memorias
P = 4
tamaño = N // P #tamaño de cada bloque
xi = np.zeros((P, N), dtype = int)

for mu in range(P):
    inicio = mu * tamaño
    fin = (mu + 1) * tamaño
    xi[mu, inicio:fin] = 1

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


#BUCLE DE EVOLUCIÓN

#Creo la figura para la animación 
s = np.copy(xi[0]) 
fig, ax = plt.subplots()
img = ax.imshow(s.reshape(10, 10), vmin = 0, vmax = 1, cmap = 'gray') #Creamos la imagen
ax.axis('off') #Para no mostrar los valores en los ejes x e y 
cbar = fig.colorbar(img, ax=ax, orientation= 'vertical') #Muestro la barra de colores que crea la figura de forma vertical

solapamiento = np.zeros((10, P))

def animation(frame):
    for _ in range(10):#10 actualizaciones por frame
        #Elegimos una neurona j al azar
        j = np.random.randint(0, N)
        #Calculamos el campo local
        h_j = np.sum(w[j, :]*s)
        #Calculamos la probabilidad de que la neurona se dispare
        P_j = 0.5*(1+np.tanh(beta*(h_j-theta[j])))
        u = np.random.rand()
        s[j] = 1 if P_j > u else 0
        
    for mu in range (P): 
        solapamiento[frame, mu] = np.sum((xi[mu] - a) * (s - a)) / (a * (1 - a) * N)
    
    print(f"Paso {frame*10}, Solapamiento: {solapamiento[frame]}")
    img.set_data(s.reshape(10, 10))
    ax.set_title(f"Paso {frame*10}")
    return [img]

anim = FuncAnimation(fig, animation, frames=10, interval=300, repeat = False) #Añadimos blit = true si queremos optimizar el gif, el numero de frames determina las iteraciones
plt.show()

for mu in range(P):
    plt.plot(solapamiento[:, mu], label=f'μ = {mu}')
plt.xlabel("Frame")
plt.ylabel("Solapamiento")
plt.title("Evolución del Solapamiento por patrón")
plt.legend()
plt.grid()
plt.show()
# %%
