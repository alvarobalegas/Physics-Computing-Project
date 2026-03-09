
# Estudio de la Emergencia de Cooperación en Redes Espaciales y Complejas 🎮🕸️

Este proyecto analiza la dinámica evolutiva de la cooperación en poblaciones de agentes utilizando el **Dilema del Prisionero** y el **Juego de los Bienes Públicos (PGG)**. El objetivo principal es identificar cómo la arquitectura de la red y los parámetros de recompensa influyen en la supervivencia de las estrategias cooperativas frente a la deserción.

## 📖 Resumen del Proyecto
El trabajo se basa en la aplicación de la teoría de juegos a sistemas complejos, explorando la transición de fase entre regímenes de cooperación total y deflexión total. Se investiga el impacto del parámetro de tentación $b$, la existencia de caos espacial y la influencia de la estructura mesoscópica (comunidades) en la red.

## 🔬 Fundamento Teórico

### Ecuación del Replicador
La evolución temporal de las estrategias se rige por la ecuación del replicador, que relaciona la tasa de cambio de una estrategia con su éxito relativo frente a la media:

$$\dot{x}_{i}=x_{i}((Wx)_{i}-x^{T}Wx)$$

Donde $W$ es la matriz de ganancias, que para el dilema del prisionero simplificado depende del parámetro de tentación $b$:

$$
W = \begin{pmatrix} 1 & 0 \\ b & 0 \end{pmatrix}
$$

[Image of the replicator equation dynamics plot]

### Caracterización del Régimen Caótico
Se ha determinado que para valores de $1.8 \le b \le 2$, el sistema entra en un régimen caótico. Para medir la sensibilidad a las condiciones iniciales, se utiliza la **distancia de Hamming normalizada** ($d_{12}$):

$$d_{12}(t)=\frac{1}{N^{2}}\sum_{i,j=1}^{N}(1-\delta_{s_{1}(i,j),s_{2}(i,j)})$$

## 🕸️ Topología de Redes Complejas
El estudio compara la evolución de la cooperación en tres tipos de arquitecturas:
* **Erdös-Rényi (ER):** Redes aleatorias homogéneas con un umbral crítico de cooperación en $b_c \approx 1.33$.
* **Barabási-Albert (BA):** Redes heterogéneas tipo "scale-free" donde la presencia de *hubs* facilita la propagación de la traición, reduciendo el umbral a $b_c \approx 1.17$.
* **Redes de Comunidades:** Estructuras divididas en subredes densamente conectadas que actúan como refugios para los cooperadores, manteniendo una modularidad $Q \approx 0.70$.

[Image of network modularity and clustering coefficient comparison]

## 🏛️ Juego de Bienes Públicos (PGG) con Castigadores
Se incluye una extensión del juego de bienes públicos que introduce una tercera estrategia: el **agente castigador (P)**. 
* **Mecánica:** Los castigadores penalizan a los deflectores basándose en su reputación e historial de traición.
* **Sinergia:** Se demuestra que la interacción entre cooperadores y castigadores crea una simbiosis que permite la extinción de los deflectores en ciertos umbrales de movilidad y castigo.

## 🛠️ Requisitos
Para ejecutar las simulaciones,
