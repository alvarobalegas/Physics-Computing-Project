# Ecuación de Schrödinger: Estudio del Coeficiente de Transmisión

[cite_start]Este repositorio contiene el desarrollo y análisis de una simulación numérica para estudiar el coeficiente de transmisión de una partícula cuántica en un pozo de potencial con un obstáculo central. [cite_start]El proyecto utiliza el método de Crank-Nicolson para resolver la Ecuación de Schrödinger Dependiente del Tiempo (TDSE)[cite: 56].

## 📖 Fundamento Teórico

[cite_start]El proyecto analiza la probabilidad de que una partícula cuántica atraviese una barrera de potencial ($V_0$)[cite: 41]. [cite_start]A diferencia de la mecánica clásica, donde el comportamiento es determinista según la energía ($E$), en la mecánica cuántica se presentan fenómenos como la reflexión con $E > V_0$ y el **efecto túnel** cuando $E < V_0$[cite: 33, 35].



### Algoritmo de Evolución
[cite_start]Para garantizar la estabilidad y la **unitariedad** (conservación de la probabilidad), se emplea la **aproximación de Cayley** para el operador de evolución temporal[cite: 56, 58]:

[cite_start]$$\phi_{j,n+1} = \chi_{j,n} - \phi_{j,n}$$ [cite: 58]

Donde $\chi_{j,n}$ se obtiene resolviendo la relación:
[cite_start]$$\chi_{j,n} = \frac{2}{1 + is\frac{H_D}{2}} \phi_{j,n}$$ [cite: 58]

## 🛠️ Metodología de la Simulación

### 1. Configuración del Sistema
* [cite_start]**Malla:** Se utilizan tamaños de malla de $N = 500, 1000, 2000$ puntos[cite: 121].
* [cite_start]**Paquete Inicial:** Una función gaussiana normalizada que representa la partícula[cite: 36, 39].
* [cite_start]**Barrera de Potencial:** Definida en el intervalo central $j \in [2N/5, 3N/5]$ con una altura proporcional a $\lambda$[cite: 66, 121].



### 2. Método de Detección Estocástica
[cite_start]Para calcular el coeficiente de transmisión ($K$), se implementan detectores en los extremos del pozo[cite: 79]:
* [cite_start]**Detectores:** Situados en $j \in [0, N/5]$ (izquierda) y $j \in [4N/5, N]$ (derecha)[cite: 81].
* [cite_start]**Intervalo de Detección ($n_D$):** Tiempo calculado mediante la velocidad de grupo $v_g = 2\sin(k_0/2)$ para evitar reflexiones en las paredes[cite: 93, 96, 99].
* **Colapso de la Función de Onda:** Si se detecta la partícula, se detiene la simulación; si no, se proyecta la función de onda anulando la probabilidad en la zona del detector y se normaliza de nuevo[cite: 87, 88, 89].

## 📊 Resultados Clave

El coeficiente de transmisión se estudia en función del parámetro $\lambda$[cite: 121]:

| Régimen | Observación en la Simulación [cite: 177, 178, 179] |
| :--- | :--- |
| **$\lambda < 1$** | Excelente concordancia con la teoría (errores < 1% en mallas finas)[cite: 256]. |
| **$\lambda = 1$** | Punto crítico ($E = V_0$). Transmisión observada $\approx 0.5$ debido a que el paquete no es monocromático[cite: 191, 194]. |
| **$\lambda > 1$** | Manifestación clara del **Efecto Túnel** cuántico, inexistente en física clásica[cite: 279, 291]. |

## 📂 Estructura del Repositorio
* `Simulación.py`: Código fuente con el algoritmo de Cayley y lógica de detección.
* `Schrödinger.pdf`: Informe completo con el desarrollo matemático y tablas de datos[cite: 20].
* `video_onda.mp4`: Animación de la densidad de probabilidad $|\psi(x,t)|^2$ interactuando con la barrera.

## 📋 Requisitos
* Python 3.x
* NumPy, Matplotlib, SciPy

---
[cite_start]**Autor:** Álvaro Manuel Balegas López [cite: 17]
[cite_start]**Fecha:** 30 de junio de 2025 [cite: 18]
[cite_start]**Institución:** Universidad de Granada, Facultad de Ciencias [cite: 12, 14]
