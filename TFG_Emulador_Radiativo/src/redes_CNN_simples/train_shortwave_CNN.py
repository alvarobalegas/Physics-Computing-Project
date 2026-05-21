import h5py
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN DE RUTAS ---
DATA_DIR = "ClimART_DATA"
STATS_PATH = os.path.join(DATA_DIR, "statistics.npz")

def cargar_y_procesar_datos(año, stats_path):
    print(f"Cargando datos del año {año}...")
    stats = np.load(stats_path)
    
    ruta_inputs = os.path.join(DATA_DIR, f"inputs/{año}.h5")
    with h5py.File(ruta_inputs, 'r') as f:
        X_globals = np.array(f['globals'])
        X_levels = np.array(f['levels'])
        X_layers = np.array(f['layers'])[:, :, :14]
        
    ruta_outputs = os.path.join(DATA_DIR, f"outputs_pristine/{año}.h5")
    with h5py.File(ruta_outputs, 'r') as f:
        Y_rsdc = np.array(f['rsdc']) 
        Y_rsuc = np.array(f['rsuc']) 
        
    Y = np.concatenate([Y_rsdc, Y_rsuc], axis=1)

    # Normalización Z-score
    X_globals = (X_globals - stats['globals_mean']) / (stats['globals_std'] + 1e-8)
    X_levels = (X_levels - stats['levels_mean']) / (stats['levels_std'] + 1e-8)
    X_layers = (X_layers - stats['layers_mean'][:14]) / (stats['layers_std'][:14] + 1e-8)

    # Aplanado para formar una secuencia 1D continua
    N = X_globals.shape[0]
    X_levels_flat = X_levels.reshape(N, -1) 
    X_layers_flat = X_layers.reshape(N, -1) 
    X_final = np.concatenate([X_globals, X_levels_flat, X_layers_flat], axis=1)
    
    return X_final, Y

# --- FASE DE PIPELINE DE DATOS ---
X_train, Y_train = cargar_y_procesar_datos(1990, STATS_PATH)
X_val, Y_val = cargar_y_procesar_datos(2005, STATS_PATH)

BATCH_SIZE = 128
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
train_dataset = train_dataset.shuffle(buffer_size=10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# --- DISEÑO DE LA RED NEURONAL CONVOLUCIONAL 1D (CNN) ---
def construir_modelo_cnn(input_shape):
    modelo = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_shape,)),
        
        # 1. Expandir dimensiones: La CNN 1D espera forma (muestras, pasos_espaciales, canales)
        tf.keras.layers.Reshape((input_shape, 1)),
        
        # 2. Bloque Convolucional 1 (Extracción de características a nivel local)
        tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation=tf.nn.gelu, padding='same'),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(0.2),
        
        # 3. Bloque Convolucional 2 (Extracción de patrones espaciales más amplios)
        tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation=tf.nn.gelu, padding='same'),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(0.2),
        
        # 4. Transición a Red Densa
        tf.keras.layers.Flatten(),
        
        # 5. Capa Oculta final para cruzar toda la información
        tf.keras.layers.Dense(256, activation=tf.nn.gelu),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        # 6. Capa de Salida
        tf.keras.layers.Dense(100, activation='linear') 
    ])
    return modelo

modelo = construir_modelo_cnn(X_train.shape[1])

# --- COMPILACIÓN Y ENTRENAMIENTO ---
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
    loss='mse',
    metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')]
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',       
    patience=5,               
    restore_best_weights=True,
    verbose=1
)

print("\nIniciando entrenamiento de la CNN 1D... (Tardará más que el MLP)")
history = modelo.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50, 
    callbacks=[early_stopping],
    verbose=1
)

# --- AUTOMATIZACIÓN DE RESULTADOS ---
print("\nGenerando gráficas y guardando el modelo CNN...")

modelo.save("modelo_climart_cnn.keras")

historia = history.history
with open('resultados_CNN_numeros.txt', 'w') as f:
    f.write("Epoca\tLoss_Train\tRMSE_Train\tLoss_Val\tRMSE_Val\n")
    for i in range(len(historia['loss'])):
        f.write(f"{i+1}\t{historia['loss'][i]:.4f}\t{historia['rmse'][i]:.4f}\t{historia['val_loss'][i]:.4f}\t{historia['val_rmse'][i]:.4f}\n")

# Gráfica comparativa
plt.figure(figsize=(10, 6))
plt.plot(historia['rmse'], label='Entrenamiento CNN (1990)', color='purple', linewidth=2)
plt.plot(historia['val_rmse'], label='Validación CNN (2005)', color='green', linewidth=2)
plt.title('Evolución del Error (RMSE) - Red Neuronal Convolucional 1D', fontsize=14)
plt.xlabel('Épocas', fontsize=12)
plt.ylabel('RMSE (W/m²)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('curva_rmse_cnn.png', dpi=300, bbox_inches='tight')
plt.close()

print("¡Entrenamiento CNN completado con éxito! Revisa la nueva curva guardada.")