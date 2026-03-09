# Modelo de Hopfield de Red Neuronal – Física Computacional

**Autor:** Álvaro Manuel Balegas López

Este repositorio contiene la implementación y el análisis del **Modelo de Hopfield**, una red neuronal artificial recurrente con propiedad de memoria asociativa. El proyecto explora cómo estas redes, inspiradas en los modelos de espines de la física estadística (como el modelo de Ising), pueden almacenar y recuperar patrones mediante la minimización de una función de energía o Hamiltoniano.

## 🧠 Fundamentos Teóricos

La red está compuesta por neuronas binarias ($s_i = 0, 1$) donde el estado 1 indica que la neurona está disparando y 0 que está en reposo.

### El Hamiltoniano (Energía)
La dinámica del sistema busca minimizar el Hamiltoniano, que define la configuración de menor energía de la red:

$$H(s) = -\frac{1}{2}\sum_{i,j} w_{ij}s_i s_j + \sum_{i} \theta_i s_i$$

Donde:
* **$w_{ij}$**: Pesos sinápticos entre neuronas (matriz simétrica y sin autoconexiones).
* **$\theta_i$**: Umbrales de disparo, calculados como $\theta_i = \frac{1}{2}\sum_{j} w_{ij}$.
* **$h_i$**: Campo local que siente la neurona $i$, dado por $h_i = \sum_{j} w_{ij} s_j$.

### Regla de Aprendizaje Hebbiano
Para que un conjunto de $P$ patrones ($\xi^{\mu}$) se conviertan en atractores (mínimos de energía), los pesos se calculan mediante la regla de Hebb:

$$w_{ij} = \frac{1}{a(1-a)N} \sum_{\mu=1}^{P} (\xi_i^{\mu} - a)(\xi_j^{\mu} - a)$$

Donde **$a$** es la activación media de los patrones almacenados.

### Dinámica de Monte Carlo
La probabilidad de que una neurona se active ($s_i = 1$) en el siguiente paso de tiempo sigue una función de transferencia estocástica dependiente de la temperatura ($T$):

$$P(s_i = 1) = \frac{1}{2} \left[ 1 + \tanh\left( \beta(h_i - \theta_i) \right) \right]$$

Con $\beta = 1/T$. 
* Para **$T=0$**, el proceso es determinista: la neurona se dispara si el campo local supera el umbral.
* Para **$T=\infty$**, el sistema es puramente aleatorio y no hay memoria.

---

## 💻 Contenido del Repositorio

El repositorio se divide en varios scripts de Python que abordan los objetivos del estudio:

### 1. Evolución y Condiciones Iniciales
Los scripts `Hopfield1a.py`, `Hopfield1b.py` y `Hopfield1c.py` estudian la recuperación para $N=100$ y $P=4$ bajo tres escenarios:
* **1a:** Estado inicial igual al primer patrón (estabilidad del atractor).
* **1b:** Patrón con deformación (ruido del 10%) para probar la capacidad asociativa.
* **1c:** Condición inicial aleatoria para observar a qué atractor converge el sistema.

### 2. Capacidad de Almacenamiento (`Hopfield2.py`)
Estudia el límite de saturación de la red para $N=400$ a $T=0$.
* Analiza cuántas memorias puede retener la red antes de que aparezcan estados espurios. Se considera que una memoria se "recuerda" si el solapamiento final $|m| > 0.75$.

### 3. Efecto de la Temperatura (`Hopfield3.py`)
Analiza la transición de fase al variar la temperatura $T \in [0, 3]$.
* Muestra el paso de la **fase de memoria** (retrieval) a la **fase paramagnética** (desorden total) o de **vidrio de espines** (spin glass).

### 4. Simulación con Imágenes 2D (`Hopfield4.py`)
Implementación práctica utilizando imágenes binarizadas (como el símbolo Ying-Yang) como patrones de memoria.
* Incluye la generación de la evolución visual guardada en `evolucion_patron.gif`.

---

## 🚀 Instrucciones de Uso

1.  **Requisitos:** Se requiere Python 3 y las siguientes librerías:
    ```bash
    pip install numpy matplotlib pillow
    ```
2.  **Ejecución:** Ejecuta cualquier script para visualizar la evolución del solapamiento y el estado de las neuronas:
    ```bash
    python Hopfield4.py
    ```

## 📊 Parámetro de Orden: Solapamiento
El éxito de la recuperación se mide mediante el solapamiento ($m^{\mu}$), que cuantifica la correlación entre el estado actual de la red $s$ y el patrón almacenado $\xi^{\mu}$:

$$m^{\mu}(s) = \frac{1}{a(1-a)N} \sum_{i} (\xi_i^{\mu} - a)(s_i - a)$$

---
*Este proyecto ha sido desarrollado como parte de la asignatura de **Física Computacional**, aplicando métodos de la mecánica estadística al campo de la computación neuronal.*
