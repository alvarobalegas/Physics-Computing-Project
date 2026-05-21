import h5py
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN DE RUTAS ---
DATA_DIR = "ClimART_DATA"
STATS_PATH = os.path.join(DATA_DIR, "statistics.npz")

def cargar_y_procesar_datos(año, stats_path):
    """
    Carga los datos H5 de un año específico, filtra para cielo prístino,
    aplana las dimensiones espaciales y normaliza las entradas.
    """
    print(f"Cargando datos del año {año}...")
    
    # 1. Cargar estadísticas para normalización
    stats = np.load(stats_path)
    
    # 2. Leer las variables de entrada (Inputs)
    ruta_inputs = os.path.join(DATA_DIR, f"inputs/{año}.h5")
    with h5py.File(ruta_inputs, 'r') as f:
        X_globals = np.array(f['globals'])
        X_levels = np.array(f['levels'])
        # Para pristine-sky, solo usamos las primeras 14 características de las capas
        X_layers = np.array(f['layers'])[:, :, :14] 
        
    # 3. Leer las variables objetivo (Outputs: Onda corta)
    ruta_outputs = os.path.join(DATA_DIR, f"outputs_pristine/{año}.h5")
    with h5py.File(ruta_outputs, 'r') as f:
        Y_rsdc = np.array(f['rsdc']) # Flujo descendente (N, 50)
        Y_rsuc = np.array(f['rsuc']) # Flujo ascendente (N, 50)
        
    # Concatenamos los dos flujos para predecirlos a la vez (Multi-task básico)
    # Forma final de Y: (N, 100)
    Y = np.concatenate([Y_rsdc, Y_rsuc], axis=1)

    # 4. Normalización (Z-score scaling)
    # Se suma 1e-8 a la desviación estándar para evitar divisiones por cero
    X_globals = (X_globals - stats['globals_mean']) / (stats['globals_std'] + 1e-8)
    X_levels = (X_levels - stats['levels_mean']) / (stats['levels_std'] + 1e-8)
    # Cuidado: stats['layers_mean'] tiene 45 canales, tomamos solo los primeros 14
    X_layers = (X_layers - stats['layers_mean'][:14]) / (stats['layers_std'][:14] + 1e-8)

    # 5. Aplanado (Flattening) para el MLP
    # El MLP no entiende matrices 3D, necesita un vector 1D por cada muestra
    N = X_globals.shape[0]
    X_levels_flat = X_levels.reshape(N, -1) # De (N, 50, 4) a (N, 200)
    X_layers_flat = X_layers.reshape(N, -1) # De (N, 49, 14) a (N, 686)
    
    # Concatenamos todo en un único vector de entrada: 82 + 200 + 686 = 968 características
    X_final = np.concatenate([X_globals, X_levels_flat, X_layers_flat], axis=1)
    
    return X_final, Y

# --- FASE DE CARGA ---
# Cargamos los años que descargaste en el script anterior
X_train, Y_train = cargar_y_procesar_datos(1990, STATS_PATH)
X_val, Y_val = cargar_y_procesar_datos(2005, STATS_PATH)

print(f"Forma de X_train: {X_train.shape}") # Debería ser (N, 968)
print(f"Forma de Y_train: {Y_train.shape}") # Debería ser (N, 100)

# --- CREACIÓN DEL tf.data.Dataset ---
# Esto optimiza el uso de memoria RAM y CPU/GPU enviando los datos en lotes (batches)
BATCH_SIZE = 128

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
train_dataset = train_dataset.shuffle(buffer_size=10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# --- DISEÑO DE LA RED NEURONAL (MLP) ---
# Basado estrictamente en la arquitectura descrita en el paper de ClimART
def construir_modelo_mlp(input_shape):
    modelo = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_shape,)),
        
        # Capa Oculta 1
        tf.keras.layers.Dense(512, activation=tf.nn.gelu),
        tf.keras.layers.LayerNormalization(), # El paper menciona LayerNorm para el MLP
        
        # Capa Oculta 2
        tf.keras.layers.Dense(256, activation=tf.nn.gelu),
        tf.keras.layers.LayerNormalization(),
        
        # Capa Oculta 3
        tf.keras.layers.Dense(256, activation=tf.nn.gelu),
        tf.keras.layers.LayerNormalization(),
        
        # Capa de Salida (100 neuronas lineales para regresión: 50 para rsdc + 50 para rsuc)
        tf.keras.layers.Dense(100, activation='linear') 
    ])
    return modelo

modelo = construir_modelo_mlp(X_train.shape[1])

# --- COMPILACIÓN Y ENTRENAMIENTO ---
# Usamos Adam con un learning rate base, y MSE como función de pérdida
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4)

modelo.compile(
    optimizer=optimizer,
    loss='mse',
    metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')]
)

print("\nResumen del modelo:")
modelo.summary()

print("\nIniciando el entrenamiento...")
# Entrenaremos por unas pocas épocas para validar que el código funciona en tu portátil
history = modelo.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10, # Sube este valor a 50 o 100 cuando tengas tiempo de dejar el PC trabajando
    verbose=1
)

# 1. Guardar el modelo entrenado para no tener que repetir el proceso
modelo.save("modelo_climart_mlp.keras")
print("¡Entrenamiento completado y modelo guardado!")

# 2. Extraer el historial numérico del entrenamiento
historia = history.history

# Guardar los números exactos en un archivo de texto
with open('resultados_MLP_numeros.txt', 'w') as f:
    f.write("Epoca\tLoss_Train\tRMSE_Train\tLoss_Val\tRMSE_Val\n")
    for i in range(len(historia['loss'])):
        f.write(f"{i+1}\t{historia['loss'][i]:.4f}\t{historia['rmse'][i]:.4f}\t{historia['val_loss'][i]:.4f}\t{historia['val_rmse'][i]:.4f}\n")
print("¡Archivo de texto 'resultados_MLP_numeros.txt' generado!")

# 3. Crear y guardar la gráfica del RMSE
plt.figure(figsize=(10, 6))
plt.plot(historia['rmse'], label='RMSE Entrenamiento (1990)', color='blue', linewidth=2)
plt.plot(historia['val_rmse'], label='RMSE Validación (2005)', color='orange', linewidth=2)
plt.title('Evolución del Error (RMSE) del MLP - Cielo Prístino')
plt.xlabel('Épocas')
plt.ylabel('RMSE (W/m²)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('curva_rmse_mlp.png', dpi=300, bbox_inches='tight') # dpi=300 da calidad de impresión
plt.close() # Cierra la figura para liberar memoria

# 4. Crear y guardar la gráfica de la Loss (MSE)
plt.figure(figsize=(10, 6))
plt.plot(historia['loss'], label='Loss Entrenamiento (MSE)', color='blue', linewidth=2)
plt.plot(historia['val_loss'], label='Loss Validación (MSE)', color='orange', linewidth=2)
plt.title('Evolución de la Función de Pérdida del MLP')
plt.xlabel('Épocas')
plt.ylabel('Pérdida (MSE)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('curva_loss_mlp.png', dpi=300, bbox_inches='tight')
plt.close()

print("¡Gráficas guardadas con éxito (curva_rmse_mlp.png y curva_loss_mlp.png)!\n")