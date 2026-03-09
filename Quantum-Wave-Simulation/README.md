# Ecuación de Schrödinger: Estudio del Coeficiente de Transmisión

Este repositorio contiene el desarrollo y análisis de una simulación numérica para estudiar el coeficiente de transmisión de una partícula cuántica en un pozo de potencial con un obstáculo central. El proyecto utiliza el método de Crank-Nicolson para resolver la Ecuación de Schrödinger Dependiente del Tiempo (TDSE).

## 📖 Fundamento Teórico

El proyecto analiza la probabilidad de que una partícula cuántica atraviese una barrera de potencial ($V_0$). A diferencia de la mecánica clásica, donde el comportamiento es determinista según la energía ($E$), en la mecánica cuántica se presentan fenómenos como la reflexión con $E > V_0$ y el **efecto túnel** cuando $E < V_0$.



### Algoritmo de Evolución
Para garantizar la estabilidad y la **unitariedad** (conservación de la probabilidad), se emplea la **aproximación de Cayley** para el operador de evolución temporal:

$$\phi_{j,n+1} = \chi_{j,n} - \phi_{j,n}$$ 

Donde $\chi_{j,n}$ se obtiene resolviendo la relación:
$$\chi_{j,n} = \frac{2}{1 + is\frac{H_D}{2}} \phi_{j,n}$$ 

## 🛠️ Metodología de la Simulación

### 1. Configuración del Sistema
* **Malla:** Se utilizan tamaños de malla de $N = 500, 1000, 2000$ puntos.
* **Paquete Inicial:** Una función gaussiana normalizada que representa la partícula.
* **Barrera de Potencial:** Definida en el intervalo central $j \in [2N/5, 3N/5]$ con una altura proporcional a $\lambda$.



### 2. Método de Detección Estocástica
Para calcular el coeficiente de transmisión ($K$), se implementan detectores en los extremos del pozo:
* **Detectores:** Situados en $j \in [0, N/5]$ (izquierda) y $j \in [4N/5, N]$ (derecha).
* **Intervalo de Detección ($n_D$):** Tiempo calculado mediante la velocidad de grupo $v_g = 2\sin(k_0/2)$ para evitar reflexiones en las paredes.
* **Colapso de la Función de Onda:** Si se detecta la partícula, se detiene la simulación; si no, se proyecta la función de onda anulando la probabilidad en la zona del detector y se normaliza de nuevo.

## 📊 Resultados Clave

El coeficiente de transmisión se estudia en función del parámetro $\lambda$:

| Régimen | Observación en la Simulación  |
| :--- | :--- |
| **$\lambda < 1$** | Excelente concordancia con la teoría (errores < 1% en mallas finas). |
| **$\lambda = 1$** | Punto crítico ($E = V_0$). Transmisión observada $\approx 0.5$ debido a que el paquete no es monocromático. |
| **$\lambda > 1$** | Manifestación clara del **Efecto Túnel** cuántico, inexistente en física clásica. |

## 📂 Estructura del Repositorio
* `Transmision.py`: Código fuente con el algoritmo de Cayley y lógica de detección.
* `Schrödinger.pdf`: Informe completo con el desarrollo matemático y tablas de datos.

## 📋 Requisitos
* Python 3.x
* NumPy, Matplotlib, SciPy

---
**Autor:** Álvaro Manuel Balegas López 
**Fecha:** 30 de junio de 2025 
**Institución:** Universidad de Granada, Facultad de Ciencias 
