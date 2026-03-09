# Quantum Wave Simulation 🌊

Este repositorio contiene una implementación numérica para simular la evolución temporal de una función de onda cuántica en una dimensión, resolviendo la **Ecuación de Schrödinger Dependiente del Tiempo (TDSE)**. 

El proyecto fue desarrollado para la asignatura de **Computación Física**, centrándose en el análisis del comportamiento de un paquete de ondas frente a diferentes potenciales.

## 📂 Contenido del Repositorio

* **`Simulación.py`**: Script principal en Python que contiene la lógica del solver numérico, la definición de los potenciales y el motor de animación.
* **`Project_Report.pdf`**: Informe detallado que describe el fundamento físico, el desarrollo matemático del método numérico, el análisis de errores y la interpretación de los resultados obtenidos.
* **`video_onda.mp4`**: Un ejemplo de salida visual que muestra la evolución de la densidad de probabilidad $|\Psi(x,t)|^2$ interactuando con una barrera de potencial.

## 🔬 Fundamento Físico y Numérico

El núcleo del proyecto es la resolución de la ecuación de Schrödinger:
$$i\hbar \frac{\partial}{\partial t} \Psi(x,t) = \hat{H} \Psi(x,t)$$

### Implementación Numérica
Como se detalla en el `Project_Report.pdf`, se ha utilizado el **método de Crank-Nicolson**. Este esquema es implícito y de segundo orden, lo que garantiza una propiedad crítica en física cuántica: la **unitariedad**. Esto asegura que la probabilidad total (la norma de la función de onda) se conserve constante durante toda la simulación.

El sistema se resuelve mediante una matriz tridiagonal que relaciona el estado de la onda en el instante $n$ con el instante $n+1$.

## 🛠️ Requisitos

Para ejecutar el código de `Simulación.py`, necesitarás las siguientes librerías de Python:

```bash
pip install numpy matplotlib scipy
