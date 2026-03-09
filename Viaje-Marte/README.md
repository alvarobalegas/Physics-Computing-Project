# Viaje a Marte: Simulación de Órbita de Transferencia de Hohmann 🚀

Este repositorio contiene una simulación numérica de una misión espacial desde la Tierra hasta Marte, desarrollada para la asignatura de **Física Computacional**. El proyecto utiliza mecánica hamiltoniana y el integrador numérico Runge-Kutta de 4.º orden para modelar la trayectoria de una nave bajo la influencia gravitatoria del Sol, la Tierra y Marte.

## 📖 Fundamento Teórico

### Hamiltoniano del Sistema
La dinámica de la nave se describe mediante un Hamiltoniano que considera la interacción con el Sol ($M_S$), la Tierra ($M_T$) y Marte ($M_M$). En coordenadas polares, el sistema se define como:

$$H=\frac{P_{r}^{2}}{2m}+\frac{P_{\phi}^{2}}{2mr^{2}}-G\frac{mM_{S}}{r}-G\frac{mM_{T}}{r_{T}(r,\phi,t)}-G\frac{mM_{M}}{r_{M}(r,\phi,t)}$$

Para optimizar el cálculo computacional y evitar errores de redondeo, se realiza un reescalado de variables donde las distancias se expresan en Unidades Astronómicas (UA) y las masas se normalizan respecto a la del Sol.

### Órbita de Transferencia de Hohmann
El viaje sigue una trayectoria elíptica tangente a la órbita de salida (Tierra) y a la de llegada (Marte). Este plan requiere dos impulsos estratégicos:
1. **Primer impulso:** Realizado para escapar de la influencia terrestre y entrar en la elipse de transferencia hacia el afelio.
2. **Segundo impulso:** Ejecutado al llegar a la órbita de Marte para igualar velocidades y permitir la captura gravitatoria.



## 🛠️ Implementación Numérica

* **Integrador:** Se emplea el método de **Runge-Kutta de 4.º orden (RK4)** para resolver las ecuaciones de movimiento de Hamilton.
* **Condiciones Iniciales:** La nave comienza en una órbita terrestre baja (**LEO**) a 2000 km de la superficie.
* **Criterio de Escape:** Se utiliza el **radio de Hill** ($\sim 0.01$ UA) para determinar el momento exacto en el que la gravedad solar predomina sobre la terrestre, activando el primer impulso tangencial.

## 📊 Resultados de la Simulación

Según los datos obtenidos en el informe técnico `VIAJEMARTE.pdf`:
* **Cronología del Viaje:**
    * **Primer impulso:** Realizado a los **55.52 días** de simulación.
    * **Segundo impulso:** Aplicado a los **307.60 días** al alcanzar el afelio.
* **Conclusión:** Aunque la fase de transferencia fue exitosa siguiendo la trayectoria teórica, la nave no logró quedar orbitando Marte. La discrepancia final de **0.024 UA** en el punto de encuentro fue demasiado grande para que la gravedad de Marte superara la atracción solar, resultando en un viaje fallido pero físicamente coherente con las limitaciones del modelo.



## 📂 Estructura del Proyecto
* `simulacion.py`: Script principal con el motor de cálculo RK4 y la lógica de los impulsos.
* `VIAJEMARTE.pdf`: Informe detallado con el desarrollo matemático, gráficas de trayectoria y conclusiones.

## 📋 Requisitos
* Python 3.x
* NumPy
* Matplotlib (para la visualización de órbitas)

---
**Autor:** Álvaro Manuel Balegas López  
**Fecha:** 9 de julio de 2025  
**Institución:** Universidad de Granada, Facultad de Ciencias
