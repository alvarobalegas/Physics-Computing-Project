#%%
import numpy as np
import matplotlib.pyplot as plt   
import matplotlib.animation as animation
#Constantes para reescalar las unidades
G = 6.67e-11
M_s = 1.99e30
c = 1.496e11

#Constantes reales de los planetas
#Creo un array de zeros para r0, m0, theta0 y omega y los relleno para cada planeta
m = np.zeros(9)
r = np.zeros(9)
theta0 = np.zeros(9)
v = np.zeros(9) #Velocidad de traslacion respecto al Sol

#Relleno el array con los datos iniciales de los planetas
"""Sol"""
m[0] = 1.99e30
r[0] = 0
theta0[0] = 0
v[0] = 0

"""Mercurio"""
m[1] = 0.33e24
r[1] = 57.9e9
theta0[1] = 7.0
v[1] = 47.4e3

"""Venus"""
m[2] = 4.87e24
r[2] = 108.2e9
theta0[2] = 3.4
v[2] = 35e3

"""La Tierra"""
m[3] = 5.97e24
r[3] = 149.6e9
theta0[3] = 0.0
v[3] = 29.8e3

"""Marte"""
m[4] = 0.642e24
r[4] = 228.0e9
theta0[4] = 1.8
v[4] = 24.1e3

"""Jupiter"""
m[5] = 1898e24
r[5] = 778.5e9
theta0[5] = 1.3
v[5] = 13.1e3

"""Saturno"""
m[6] = 568.0e24
r[6] = 1432.0e9
theta0[6] = 2.5
v[6] = 9.7e3

"""Urano"""
m[7] = 86.8e24
r[7] = 2867.0e9
theta0[7] = 0.8
v[7] = 6.8e3

"""Neptuno"""
m[8] = 102e24
r[8] = 4515.0e9
theta0[8] = 1.8
v[8] = 5.4e3

t = 5.5e9 #s Periodo del planeta neptuno aproximadamente

#Valores a utilizar y condiciones iniciales
r_reescalado = np.copy(r)/c
m_reescalado = np.copy(m)/M_s
v_reescalado = np.copy(v)*np.sqrt(c / (G*M_s))
theta_rads = np.radians(theta0)
h_r = 0.01 #Paso a utilizar en el algoritmo de Verlet
t_reescalado = np.arange(0, np.sqrt((G*M_s) / c**3)*t, h_r)

#Condiciones iniciales
#Tomamos al principio que estamos en theta = 0 para comprobar que nos funciona bien el programa
x_r = np.copy(r_reescalado)
y_r = np.zeros_like(x_r)
vx_r = np.zeros_like(x_r)
vy_r = np.copy(v_reescalado)
x_r[0], y_r[0], vx_r[0], vy_r[0] = 0, 0, 0, 0


#Realizamos una funcion para calcular la aceleracion de los planetas
def AceleracionPlanetas(m, x ,y):
    N = len(m)
    ax = np.zeros(N)
    ay = np.zeros(N)
    
    #Interaccion gravitatoria entre los planetas
    for i in range(0, N):  
        for j in range(0, N):
            if j != i:
                # Distancia entre planetas i y j
                dx = x[j] - x[i]
                dy = y[j] - y[i]
                r_ij = np.sqrt(dx**2 + dy**2)  #Distancia entre planetas
                if r_ij > 1e-10:  # Para evitar divisiones entre cero
                    factor = m[j] / r_ij**3
                    ax[i] += dx * factor
                    ay[i] += dy * factor

    return ax, ay

#Funcion para guardar la trayectoria de cada planeta
def TrayectoriaPlaneta(m, x, y, vx, vy, h, t):

    N = len(m) #Numero de planetas
    num_pasos = len(t)

    trayectorias_x = np.zeros((N, num_pasos))
    trayectorias_y = np.zeros((N, num_pasos))
    trayectorias_x[:, 0] = x #Rellenamos la primera columna del array con las posiciones iniciales
    trayectorias_y[:, 0] = y
    
    """Creo dos arrays uno para almacenar la energia de cada planeta en cada instante
    y otro para calcular la energia total del sistema en cada instante"""
    energia_cinetica = np.zeros((N, num_pasos)) 
    energia_potencial = np.zeros((N, num_pasos))
    energia = np.zeros((N, num_pasos))
    energia_total = np.zeros(num_pasos)
    #Calculamos la energia inicial de los planetas
    energia_cinetica[:, 0] = 0.5*m[:]*(vx**2+vy**2)
    for i in range(N):
        for j in range(N):
            if j!=i:
                energia_potencial[i, 0] += -m[i]*m[j] / np.sqrt((x[i]-x[j])**2+(y[i]-y[j])**2)
    energia[:, 0] = energia_cinetica[:, 0] + energia_potencial[:, 0]
    energia_total[0] = np.sum(energia[:, 0])

    #Realizo el algoritmo de Verlet para calcular las posiciones
    ax, ay = AceleracionPlanetas(m, x, y)
    
    for i in range(1, len(t)):

        # Actualización de posiciones con Verlet
        x_new = x + vx * h + 0.5 * ax * h**2
        y_new = y + vy * h + 0.5 * ay * h**2

        # Calculamos la nueva aceleración en la posición actualizada
        ax_new, ay_new = AceleracionPlanetas(m, x_new, y_new)

        # Actualización de velocidades con Verlet en velocidad
        vx_new = vx + 0.5 * (ax + ax_new) * h
        vy_new = vy + 0.5 * (ay + ay_new) * h

        # Actualizamos las posiciones, velocidades y aceleraciones para el siguiente paso
        x, y = x_new, y_new
        ax, ay = ax_new, ay_new
        vx, vy = vx_new, vy_new

        # Guardamos en la trayectoria
        trayectorias_x[:, i] = x
        trayectorias_y[:, i] = y
        trayectorias_x[0, i] = 0 #Obligo a que el sol este en el cero siempre
        trayectorias_y[0, i] = 0
        
        energia[0, i] = 0
        for j in range(N):
            energia_cinetica[j, i] = 0.5*m[j]*(vx[j]**2+vy[j]**2)
            for k in range(N):
                if k != j:
                    energia_potencial[j,i] += -m[j]*m[k] / np.sqrt((x[j]-x[k])**2+(y[j]-y[k])**2)
            energia[j, i] = energia_cinetica[j, i] + energia_potencial[j, i]
        energia_total[i] = np.sum(energia[:,i])
        
        
    return trayectorias_x, trayectorias_y, energia, energia_total

def PeriodoPlanetas(m, x, y, t):
    N = len(m)  # Número de planetas
    periodos = np.zeros(N)  # Array para almacenar los períodos
    cruces = [[] for _ in range(N)]  # Lista para almacenar tiempos de cruce [[], [], ...N]

    # Recorremos en el tiempo y detectamos cruces por el eje x con vy > 0
    for i in range(1, len(t) - 1):
        for j in range(1, N): 
            if np.sign(y[j, i]) != np.sign(y[j, i - 1]): #Añado el tiempo cuando el eje y cambia de signo
                if (y[j,i]-y[j,i-1])>0:
                    cruces[j].append(t[i])

    # Calcular el período como la diferencia entre dos cruces consecutivos
    for j in range(1, N):
        if len(cruces[j]) > 1:
            periodos[j] = np.mean(np.diff(cruces[j])) # Promedio del tiempo entre cruces en el eje x

    return periodos * (np.sqrt(c**3 / (G*M_s))) /86400


def crear_animacion(trayectorias_x, trayectorias_y, t, N):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal', adjustable='box')
    
    #Dibujo el sol en el cero
    ax.scatter(0, 0, label = 'Sol', color = 'yellow', s = 100)

    puntos, = ax.plot([], [], 'o', markersize=5, label='Planetas')  # Planetas
    trayectorias = [ax.plot([], [], '-', markersize=2)[0] for _ in range(N-1)]  #Trayectorias

    # Función de animación
    def animate(i):
        puntos.set_data(trayectorias_x[1:, i], trayectorias_y[1:, i])  

        for j in range(N-1):
            trayectorias[j].set_data(trayectorias_x[j+1, :i], trayectorias_y[j+1, :i])

        return [puntos] + trayectorias

    ani = animation.FuncAnimation(fig, animate, frames=len(t), interval=50, blit=True)
    
    plt.show()
    
#Funcion principal donde aplico las funciones creadas con los datos que voy a utilizar anteriormente definidos
def main():
    
    trayectorias_x, trayectorias_y, energia, energia_total = TrayectoriaPlaneta(m_reescalado, x_r, y_r, vx_r, vy_r, h_r, t_reescalado)
    
    periodos = PeriodoPlanetas(m_reescalado, trayectorias_x, trayectorias_y, t_reescalado)
    
    crear_animacion(trayectorias_x, trayectorias_y, t_reescalado, len(m_reescalado))
    
    N = len(m_reescalado)
    
    for i in range(1, N):
        plt.plot(trayectorias_x[i], trayectorias_y[i], label = f'Planeta {i}')
    plt.scatter(0, 0, color = 'red', label = 'Sol', s = 1)
    plt.legend()
    plt.xlabel("x (c)")
    plt.ylabel("y (c)")
    plt.title("Órbitas planetarias (unidades reescaladas)")
    plt.grid()
    plt.show()
    
    plt.plot(t_reescalado, energia_total)
    plt.xlabel("t'")
    plt.ylabel("E")
    plt.title("Conservacion de la energia del sistema")
    plt.grid()
    plt.show()
    
    
    for j in range(1, N):
        plt.plot(t_reescalado, energia[j], label= f"Planeta{j}")
    plt.legend()
    plt.xlabel("t'")
    plt.ylabel("Energía")
    plt.title("Energia de cada planeta")
    plt.grid()
    plt.show()
    
    
    for i in range(1, N):
       print(f"Planeta {i} - Periodo: {periodos[i]:.2f} días")
    
    
main()

#Funcion del gif

# %%
