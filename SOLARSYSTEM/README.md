# Simulación Dinámica del Sistema Solar (N-Cuerpos) ☀️🪐

Este proyecto es una simulación numérica de la dinámica del Sistema Solar. Utiliza datos reales de masa, distancia y velocidad para modelar las trayectorias de los planetas bajo la influencia de la interacción gravitatoria mutua.

## 📖 Fundamento Teórico

### Hipótesis Nebular y Dinámica de Laplace
El modelo se basa en la hipótesis nebular de Laplace, que sostiene que el Sistema Solar evolucionó a partir del colapso gravitacional de una nube molecular giratoria. La conservación del momento angular durante la contracción de la nube dio lugar a la formación de un disco protoplanetario y, finalmente, a los cuerpos celestes actuales.



### Interacción Gravitatoria de N-Cuerpos
El sistema se compone de $N$ cuerpos caracterizados por su masa ($m_i$) y radio ($R_i$). La evolución de cada cuerpo $i$ se rige por la suma de las fuerzas gravitatorias ejercidas por todos los demás cuerpos $j$ del sistema:

$$m_{i}\frac{d^{2}r_{i}}{dt^{2}} = -Gm_{i}\sum_{j\ne i}\frac{m_{j}(r_{i}-r_{j})}{|r_{i}-r_{j}|^{3}}$$

## 🛠️ Implementación Numérica

### Algoritmo de Verlet en Velocidad
Para integrar las ecuaciones de movimiento, se emplea el **algoritmo de Verlet en velocidad**, un método de un paso basado en desarrollos de Taylor. Este algoritmo es reversible y conserva en promedio la energía y el momento angular, lo que lo hace ideal para simulaciones de larga duración.



### Reescalado de Unidades
Para evitar errores de redondeo derivados de magnitudes físicas extremas, las ecuaciones se reescalan utilizando las siguientes constantes:
* **Distancia ($r'$):** Se escala respecto a $c = 1.496 \times 10^{11}$ m (distancia Tierra-Sol).
* **Masa ($m'$):** Se escala respecto a la masa solar $M_s = 1.99 \times 10^{30}$ kg.
* **Tiempo ($t'$):** Se utiliza el factor $[\frac{GM_s}{c^3}]^{1/2}$.

## 📂 Estructura del Repositorio
* **`ProgramaFinal.py`**: Script principal que contiene el motor de integración Verlet, el cálculo de aceleraciones gravitatorias y el sistema de animación.
* **`Problema1-planetas-SLIDES.pdf`**: Presentación teórica sobre la formación del Sistema Solar y la introducción a la dinámica molecular.

## 📊 Resultados Obtenidos
* **Validación Cinemática**: El programa calcula los períodos de rotación de cada planeta, permitiendo verificar la precisión de la simulación frente a datos reales.
* **Estabilidad Energética**: El algoritmo mantiene la conservación de la energía total del sistema, factor crítico para validar el sentido físico de los resultados a largo plazo.
* **Visualización**: Generación de órbitas relativas y animaciones dinámicas que muestran el baile orbital de los ocho planetas principales.

## 💻 Requisitos
* Python 3.x
* NumPy
* Matplotlib

---
**Autor:** Álvaro Manuel Balegas López   
**Asignatura:** Física Computacional   
**Institución:** Universidad de Granada, Facultad de Ciencias 
