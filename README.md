# Physics Computing Projects ⚛️💻

Este repositorio contiene una colección de los proyectos más significativos que he desarrollado durante mi formación en el grado de Física en la **Universidad de Granada**. En ellos se aplican métodos numéricos avanzados e inteligencia artificial para resolver problemas que abarcan desde la física atmosférica y la mecánica celeste hasta la física estadística y la mecánica cuántica.

---

## 📂 Contenido del Repositorio

### 1. Trabajo de Fin de Grado: Emulación de la transferencia radiativa atmosférica mediante Deep Learning 🌍🤖
Desarrollo y optimización de un emulador estadístico para acelerar el cálculo del transporte radiativo en modelos climáticos globales utilizando la base de datos ClimART.
* **Descripción:** Implementación de una arquitectura de Redes Neuronales Convolucionales 1D (CNN) en Python para sustituir las costosas parametrizaciones físicas tradicionales. El modelo evalúa con alta precisión los flujos radiativos tanto en condiciones ideales (Cielo Prístino) como bajo la perturbación óptica de los aerosoles (Cielo Despejado), aplicando estrategias de muestreo multianual al 10% y regularización estricta para garantizar la solidez física y generalización del emulador.

### 2. Estudio de la emergencia de cooperación en redes espaciales 🎮🕸️
Investigación de la dinámica de cooperación en redes de jugadores mediante el **dilema del prisionero evolutivo**.
* **Descripción:** Análisis de cómo influyen la arquitectura de la red (redes aleatorias, complejas y de comunidades) y el parámetro de tentación $b$ en la supervivencia de los cooperadores.
* **Autores:** Pablo Gaitán Ruz y Álvaro Manuel Balegas López.

### 3. Modelo de Hopfield de Red Neuronal 🧠
Implementación y análisis de una red neuronal artificial recurrente con propiedad de memoria asociativa.
* **Descripción:** Inspirado en los modelos de espines de la física estadística (como el modelo de Ising), el proyecto explora cómo estas redes almacenan y recuperan patrones mediante la minimización de un **Hamiltoniano** de energía.

### 4. Ecuación de Schrödinger: Estudio del Coeficiente de Transmisión 🌊
Simulación numérica de una partícula cuántica en un pozo de potencial con un obstáculo central.
* **Descripción:** Resolución de la **Ecuación de Schrödinger Dependiente del Tiempo (TDSE)** utilizando el método de **Crank-Nicolson** para calcular el coeficiente de transmisión.

### 5. Simulación Dinámica del Sistema Solar (N-Cuerpos) ☀️🪐
Modelado de las trayectorias planetarias bajo la influencia de la interacción gravitatoria mutua.
* **Descripción:** Uso de datos reales de masa, distancia y velocidad para simular la dinámica del Sistema Solar integrando las ecuaciones de movimiento de Newton.

### 6. Viaje a Marte: Simulación de Órbita de Transferencia de Hohmann 🚀
Simulación de una misión espacial desde la Tierra hasta Marte desarrollada para la asignatura de **Física Computacional**.
* **Descripción:** Uso de **mecánica hamiltoniana** y el integrador numérico **Runge-Kutta de 4.º orden (RK4)** para modelar la trayectoria de una nave bajo la influencia gravitatoria del Sol, la Tierra y Marte.

---

## 🛠️ Tecnologías y Métodos Numéricos

Para el desarrollo de estos proyectos se han empleado herramientas estándar de la computación científica e Inteligencia Artificial:

* **Lenguaje:** Python 3.x
* **Librerías y Frameworks:** TensorFlow / Keras (Deep Learning), NumPy, SciPy, Matplotlib, Pandas, NetworkX.
* **Métodos Numéricos y Algoritmos:** * Redes Neuronales Convolucionales (CNN 1D) y regularización estocástica (*Dropout*).
  * Integración de EDOs (Runge-Kutta 4).
  * Resolución de EDPs (Crank-Nicolson).
  * Dinámica de redes y sistemas complejos (Teoría de Juegos Evolutiva).
  * Algoritmos de minimización de energía en sistemas de espines.
  * Análisis de densidad de datos masivos (*Hexbin plots* en escala logarítmica).

---

## 👨‍🔬 Autor

**Álvaro Manuel Balegas López** *Grado en Física - Universidad de Granada (UGR)*

---
