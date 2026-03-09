#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image

# Parámetros
valores_T = np.linspace(0, 3, 100)
N = 400  # 20x20 imagen
P = 1
valores_m = []
lado = int(np.sqrt(N))

# --- CARGAR Y DEFINIR xi[0] DESDE IMAGEN ---
img = Image.open('yingyang.png').convert('L').resize((lado, lado))
img_array = np.array(img)

# Mostrar la imagen redimensionada
plt.imshow(img_array, cmap='gray')
plt.title('Imagen original (20x20)')
plt.axis('off')
plt.show()

# Inicializar xi
xi = np.zeros((1, N), dtype=int)
xi[0] = (img_array > 128).astype(int).flatten()  # Binarización

# Mostrar patrón binarizado
plt.imshow(xi[0].reshape(lado, lado), cmap='gray')
plt.title('Imagen binarizada como patrón xi[0]')
plt.axis('off')
plt.show()

# --- DINÁMICA DE RED DE HOPFIELD ---
a = np.sum(xi) / (N*P)  # Activación media
w = np.zeros((N, N))
for mu1 in range(P):
    for i in range(N):
        for j in range(N):
            w[i, j] += (xi[mu1, i] - a) * (xi[mu1, j] - a)
w = w / (a * (1 - a) * N)
theta = 0.5 * np.sum(w, axis=1)

# Inicialización de la figura para animación
fig, ax = plt.subplots()
img_disp = ax.imshow(np.zeros((lado, lado)), cmap='gray', vmin=0, vmax=1)
title = ax.set_title("")
plt.axis('off')

# Función de actualización
def actualizar(T):
    global s  # mantener entre frames
    beta = 1 / T if T != 0 else None

    s = np.copy(xi[0])
    deformado = np.random.rand(N) < 0.1
    s[deformado] = 1 - s[deformado]

    for _ in range(50 * N):
        j = np.random.randint(0, N)
        h_j = np.sum(w[j, :] * s)
        P_j = 1 if T == 0 and h_j > theta[j] else 0.5 * (1 + np.tanh(beta * (h_j - theta[j]))) if T != 0 else 0
        u = np.random.rand()
        s[j] = 1 if u < P_j else 0

    m = np.sum((xi[0] - a) * (s - a)) / (a * (1 - a) * N)
    valores_m.append(m)

    img_disp.set_data(s.reshape((lado, lado)))
    title.set_text(f"T = {T:.2f}, m = {m:.2f}")
    return img_disp, title

# Crear y guardar animación
ani = FuncAnimation(fig, actualizar, frames=valores_T, interval=800, blit=False)
ani.save("evolucion_patron.gif", writer=PillowWriter(fps=5))
plt.close(fig)

valores_m = valores_m[:len(valores_T)]
# Gráfico final del solapamiento
plt.plot(valores_T, valores_m, marker='o')
plt.axhline(0.75, color='r', linestyle='--', label='Umbral de recuperación')
plt.xlabel('Temperatura')
plt.ylabel('Solapamiento con patrón 0')
plt.title('Solapamiento final vs temperatura')
plt.legend()
plt.grid()
plt.show()
