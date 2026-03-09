# Estudio de la emergencia de cooperación en juegos entre agentes en redes espaciales 🎮🕸️

Este proyecto investiga la dinámica de la cooperación en redes de jugadores que participan en un juego de **dilema del prisionero evolutivo**. A través de simulaciones numéricas, se analiza cómo la arquitectura de la red (redes aleatorias, complejas y de comunidades) y el parámetro de tentación $b$ influyen en la supervivencia de los cooperadores.

## 👥 Autores
* **Pablo Gaitán Ruz** - Universidad de Granada.
* **Álvaro Manuel Balegas López** - Universidad de Granada.

**(Fecha: 26 de febrero de 2026)**.

---

## 📂 Estructura del Repositorio
* **`Simulacion_Replicador.py`**: Scripts dedicados al estudio de la dinámica del replicador y el comportamiento del sistema en retículos regulares.
* **`erdosbarabaniteracciones.py`**: Implementación de las simulaciones en redes de Erdös-Rényi (ER) y Barabási-Albert (BA).
* **`comunidadvariable.py`**: Análisis de la cooperación en estructuras mesoscópicas y cálculo de la modularidad $Q$.
* **`comunidadva.py`**: Simulación Monte Carlo del juego de bienes públicos (PGG) con agentes castigadores y sistemas de reputación.
* **`TeoriaJuego_SistemasComplejos.pdf`**: Informe técnico completo con el fundamento físico-matemático y la discusión de resultados.

---

## 🔬 Fundamento Teórico

### Matriz de Ganancias ($W$)
El juego se rige por la siguiente matriz de beneficios, donde las filas representan la estrategia del jugador y $b$ es el parámetro de tentación de desertar:

| | Cooperador (C) | Deflector (D) |
| :--- | :---: | :---: |
| **Cooperador (C)** | 1 | 0 |
| **Deflector (D)** | $b$ | 0 |



### Dinámica del Replicador
Para modelar la evolución de las estrategias, se utiliza la **ecuación del replicado**, la cual describe cómo cambia la probabilidad $x_i$ de utilizar una estrategia $i$ en función de la ganancia obtenida:
$$\dot{x}_{i}=x_{i}((Wx)_{i}-x^{T}Wx)$$

---

## 🌀 Análisis de Resultados

### Régimen Caótico
Se ha detectado un comportamiento caótico en el rango $1.8 \le b \le 2$. Para medir la sensibilidad a las condiciones iniciales, se emplea la **distancia de Hamming normalizada** ($d_{12}$) entre dos redes idénticas con una pequeña perturbación inicial:
$$d_{12}(t)=\frac{1}{N^{2}}\sum_{i,j=1}^{N}(1-\delta_{s_{1}(i,j),s_{2}(i,j)})$$



### Topología de Red y Umbrales Críticos ($b_c$)
El colapso de la cooperación depende drásticamente de la estructura de conexiones. Los umbrales críticos de tentación identificados son:
* **Red Erdös-Rényi (ER):** $b_c \approx 1.33$. Es más resistente a la traición debido a su homogeneidad.
* **Red Barabási-Albert (BA):** $b_c \approx 1.17$. La presencia de nodos muy conectados (*hubs*) acelera la difusión de la deflexión.
* **Redes de Comunidades:** Actúan como refugios locales que retrasan el colapso de los cooperadores siempre que la modularidad sea alta ($Q \approx 0.70$).

---

## 🏛️ Promoción de la Cooperación (PGG)
Se estudia el Juego de los Bienes Públicos con agentes castigadores (**P**), quienes penalizan a los deflectores basándose en su reputación.
* [cite_start]**Simbiosis C-P:** La acción de los castigadores favorece la supervivencia de los cooperadores.
* **Efecto de la Movilidad ($m_C$):** Se demuestra que si los cooperadores abandonan con demasiada facilidad sus comunidades para buscar protección de un castigador, se rompen los núcleos que los mantienen vivos, promoviendo paradójicamente la deflexión a largo plazo.



---

## 🛠️ Requisitos
* Python 3.x 
* NumPy
* Matplotlib
* NetworkX 
* SciPy
* Pandas
