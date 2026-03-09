# Estudio del Coeficiente de Transmisión en un Pozo de Potencial ⚛️

[cite_start]Este proyecto realiza un estudio numérico de la **Ecuación de Schrödinger Dependiente del Tiempo (TDSE)** para una partícula cuántica en un pozo de potencial infinito con un obstáculo central[cite: 15, 282]. [cite_start]El objetivo principal es analizar el coeficiente de transmisión ($K$) y observar fenómenos como el **efecto túnel**[cite: 35, 291].

## 📖 Fundamento Teórico

[cite_start]La simulación modela una partícula representada por una **función de onda gaussiana**[cite: 36]. [cite_start]A diferencia de la mecánica clásica, donde la transmisión es binaria ($K=1$ si $E > V_0$, $K=0$ si $E < V_0$), en mecánica cuántica existe una probabilidad de reflexión incluso cuando la energía supera la barrera, y una probabilidad de transmisión (efecto túnel) cuando es inferior[cite: 29, 33, 35].

### Algoritmo Numérico
[cite_start]Para la evolución temporal se utiliza la **aproximación de Cayley** del operador de evolución, lo que garantiza que el operador sea unitario y se conserve la norma de la función de onda[cite: 56, 58]:
- [cite_start]**Discretización espacial:** El Hamiltoniano se discretiza en una malla de $N$ puntos[cite: 42, 52].
- [cite_start]**Condiciones de contorno:** Se aplican condiciones de pozo infinito ($\phi = 0$ en los extremos)[cite: 42, 70].
- [cite_start]**Esquema de detección:** Se utilizan detectores finitos de ancho $N/5$ a ambos lados de la barrera para medir la probabilidad de detección y proyectar la función de onda tras cada intervalo de tiempo $n_D$[cite: 79, 81, 93].

## 🛠️ Contenido del Repositorio

- [cite_start]`Simulación.py`: Implementación en Python del algoritmo de Cayley y el sistema de detección estocástica[cite: 67, 111].
- [cite_start]`Schrödinger.pdf`: Informe detallado con el desarrollo matemático, tablas de datos y conclusiones[cite: 15].
- `video_onda.mp4`: Animación de la evolución del paquete de ondas y su interacción con la barrera.

## 🚀 Parámetros de la Simulación

El sistema permite ajustar:
- [cite_start]**Tamaño del sistema ($N$):** Probado para 500, 1000 y 2000 puntos[cite: 121, 132].
- [cite_start]**Altura de la barrera ($\lambda$):** Parámetro adimensional que relaciona la energía del paquete con el potencial ($V_j = \lambda k_0^2$)[cite: 66, 132].
- [cite_start]**Detección:** Intervalos de tiempo calculados mediante la velocidad de grupo ($v_g$) para evitar reflexiones espurias en las paredes[cite: 94, 96].

## 📊 Resultados Destacados

[cite_start]Según el informe incluido[cite: 121, 283]:
- [cite_start]**$\lambda < 1$:** Excelente concordancia entre los valores simulados ($K_{sim}$) y teóricos ($K_{teo}$), con errores relativos inferiores al 1% en mallas finas[cite: 255, 258].
- [cite_start]**$\lambda = 1$:** Se observa una transmisión del $\approx 50\%$ debido a que el paquete no es monocromático (contiene componentes con $E > V_0$)[cite: 194, 284].
- [cite_start]**$\lambda > 1$:** La simulación captura con éxito el **efecto túnel**, mostrando coeficientes de transmisión mayores a cero en regímenes donde la física clásica predeciría reflexión total[cite: 279, 291].

## 📋 Requisitos

Se requiere Python 3 con las siguientes librerías:
```bash
pip install numpy matplotlib scipy
