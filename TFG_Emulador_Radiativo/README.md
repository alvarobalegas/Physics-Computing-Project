# Emulación de la Transferencia Radiativa Atmosférica mediante Deep Learning 🌍🤖

Este subdirectorio alberga el ecosistema completo de desarrollo, optimización y validación del **Trabajo de Fin de Grado (TFG)** titulado *"Emulación de la transferencia radiativa atmosférica en modelos meteorológicos mediante Deep Learning"*, presentado en la **Facultad de Ciencias de la Universidad de Granada (UGR)**.

El objetivo principal de la investigación es diseñar un emulador estadístico altamente eficiente basado en redes neuronales profundas para sustituir las costosas parametrizaciones físicas tradicionales encargadas de resolver la Ecuación de Transferencia Radiativa (RTE) en los Modelos de Circulación General (GCM).

---

## 📊 Resumen Ejecutivo del Trabajo

Los cálculos de transferencia radiativa atmosférica son el cuello de botella computacional de las simulaciones climáticas globales. Utilizando la base de datos de referencia **ClimART** (derivada del Modelo Canadiense del Sistema Terrestre, CanESM5), este trabajo aborda el problema aplicando técnicas avanzadas de Inteligencia Artificial. La investigación se estructuró de forma incremental en tres grandes hitos metodológicos:

1. **Sesgo Inductivo y Topología Espacial (Fase 1):** Se demostró empíricamente que los modelos densos tradicionales (Perceptrón Multicapa, MLP) adolecen de limitaciones estructurales insalvables para el transporte radiativo. Al "aplanar" las variables verticales, el MLP destruye la jerarquía geométrica de la atmósfera, provocando un **sobreajuste estadístico estructural masivo** e incapacidad para generalizar entre diferentes años climáticos. Las **Redes Neuronales Convolucionales 1D (CNN)** resuelven este problema al actuar como operadores espaciales que preservan la continuidad física vertical y los gradientes de la columna atmosférica.
2. **Restricciones Termodinámicas y Regularización (Fase 2):** Para evitar que las redes neuronales operasen como "cajas negras" acríticas, se impusieron **condiciones de contorno físicas** explícitas. Se implementó una activación asimétrica tipo **ReLU** en la capa de salida de Onda Corta para garantizar matemáticamente el límite termodinámico de no-negatividad de los flujos solares. Asimismo, se optimizó el espacio latente incrementando las restricciones estocásticas mediante técnicas de regularización severa (**Dropout** al 30% y **Early Stopping**), forzando a la red a extraer correlaciones redundantes y robustas en lugar de memorizar el ruido meteorológico local.
3. **Eficiencia Bi-objetivo y Muestreo Multianual (Fase 3):** Enfrentando el desafío computacional de procesar un *dataset* masivo de más de 1.5 TB, se descubrió que la diversidad climática en el entrenamiento es infinitamente más crítica que el volumen bruto de datos. Diseñando una estrategia de **muestreo multianual estratégico distribuido uniformemente (10% de la información)**, el emulador final logró estabilizar su precisión con un comportamiento asintótico óptimo. El modelo definitivo alcanzó un **RMSE de 8.12 W/m² en Onda Corta (Pristine Sky)** sobre el conjunto de test independiente (2007-2014), batiendo holgadamente a los MLP ($14.1 \text{ W/m}^2$) y GraphNet ($12.3 \text{ W/m}^2$) de la literatura, quedándose a solo $2 \text{ W/m}^2$ de la CNN oficial entrenada con el 100% de los datos, reduciendo el coste de memoria RAM y computación en un orden de magnitud.

---

## 📂 Organización y Distribución del Código (`src/`)

El código fuente se encuentra clasificado rigurosamente de forma cronológica y conceptual para reflejar el proceso de experimentación científica:

### 📁 `1_baselines_MLP/` (El colapso de las arquitecturas densas)
Contiene los scripts de entrenamiento y evaluación de los Perceptrones Multicapa iniciales. Documenta empíricamente cómo el modelo ignora la estructura geométrica vertical.
* `train_shortwave_MLP.py`: Script de entrenamiento base para los flujos de Onda Corta (SW).
* `train_MLP_base_LW.py`: Entrenamiento del MLP base para Onda Larga (LW) térmica, evidenciando subestimaciones severas en flujos altos.
* `train_MLP_corregido_LW.py`: Primeros intentos de mitigar el sobreajuste mediante regularizaciones densas tradicionales.
* `evaluar_test_2007.py`: Script de evaluación sobre el conjunto de test del año 2007 para Onda Corta.
* `evaluar_test_LW.py`: Análisis residual y dispersión de la Onda Larga en el MLP.

### 📁 `2_redes_CNN_simples/` (El salto al procesamiento convolucional local)
Implementación de los primeros operadores convolucionales unidimensionales (`Conv1D`) para evaluar el impacto de la preservación de la topología espacial de la columna de aire en ambas regiones del espectro.
* `train_shortwave_CNN.py`: Arquitectura convolucional inicial aplicada a la radiación solar incidente y reflejada (SW).
* `train_CNN_LW.py`: Aplicación de núcleos convolucionales para mapear perfiles térmicos y de humedad hacia flujos infrarrojos (LW).
* `evaluar_test_2007_CNN.py`: Validación de la CNN en el dominio espectral solar del año 2007.
* `evaluar_test_CNN_LW.py`: Script encargado de verificar la corrección de las ramas de subestimación térmica observadas en el MLP.

### 📁 `3_modelo_definitivo/` (Optimización e integridad física avanzada)
Aloja los scripts de producción definitivos que incorporan la combinación de todas las estrategias de éxito: regularización avanzada, restricciones termodinámicas y eficiencia en datos.
* `train_CNN_multianual.py`: **El código estrella del TFG.** Implementa el cargador dinámico y eficiente de memoria HDF5, aplica el enmascaramiento estocástico de variables funcionales (*Dropout* del 30%), y entrena sobre el muestreo estratégico multianual distribuido (10% de los datos) interconectando los escenarios ópticos (*Pristine Sky* y *Clear Sky*) y las interacciones moleculares (Rayleigh y Mie).
* `train_cnn_sw_rapido.py`: Configuración optimizada de bajo tiempo de cómputo para análisis paramétricos rápidos de canales convolucionales y tasas de aprendizaje.

---

## 🛠️ Tecnologías Utilizadas

* **Framework principal:** TensorFlow 2.x / Keras (Diseño, regularización y optimización de redes profundas).
* **Computación Científica:** NumPy y SciPy (Procesamiento matricial de variables de niveles y capas atmosféricas).
* **Gestión de Datos Masivos:** h5py y Pandas (Lectura secuencial y eficiente de perfiles vectoriales encapsulados en formatos HDF5).
* **Visualización Científica:** Matplotlib y Seaborn (Análisis de residuos y generación de mapas de densidad hexagonal o *Hexbin plots* en escalas logarítmicas de recuento).

---

## 👨‍🔬 Autor
**Álvaro Manuel Balegas López** *Grado en Física — Universidad de Granada (UGR)* Especialización orientada a la computación científica, Big Data e Inteligencia Artificial aplicada.
