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

    X_globals = (X_globals - stats['globals_mean']) / (stats['globals_std'] + 1e-8)
    X_levels = (X_levels - stats['levels_mean']) / (stats['levels_std'] + 1e-8)
    X_layers = (X_layers - stats['layers_mean'][:14]) / (stats['layers_std'][:14] + 1e-8)

    N = X_globals.shape[0]
    X_levels_flat = X_levels.reshape(N, -1) 
    X_layers_flat = X_layers.reshape(N, -1) 
    X_final = np.concatenate([X_globals, X_levels_flat, X_layers_flat], axis=1)
    
    return X_final, Y

# --- PIPELINE DE DATOS ---
X_train, Y_train = cargar_y_procesar_datos(1990, STATS_PATH)
X_val, Y_val = cargar_y_procesar_datos(2005, STATS_PATH)

BATCH_SIZE = 128
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
train_dataset = train_dataset.shuffle(buffer_size=10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# --- MODELO BASE (SIN REGULARIZACIÓN PARA PROVOCAR SOBREAJUSTE) ---
def construir_modelo_mlp_base(input_shape):
    modelo = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_shape,)),
        
        # Capas sin Dropout
        tf.keras.layers.Dense(512, activation=tf.nn.gelu),
        tf.keras.layers.LayerNormalization(),
        
        tf.keras.layers.Dense(256, activation=tf.nn.gelu),
        tf.keras.layers.LayerNormalization(),
        
        tf.keras.layers.Dense(256, activation=tf.nn.gelu),
        tf.keras.layers.LayerNormalization(),
        
        tf.keras.layers.Dense(100, activation='linear') 
    ])
    return modelo

modelo = construir_modelo_mlp_base(X_train.shape[1])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
    loss='mse',
    metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')]
)

print("\nIniciando entrenamiento BASE (50 épocas forzadas, sin Early Stopping)...")
# Al no poner Early Stopping, la red se verá obligada a hacer las 50 pasadas
history = modelo.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50, 
    verbose=1
)

# --- GENERACIÓN DE LA GRÁFICA DEL DESASTRE ---
print("\nGenerando gráfica de sobreajuste...")

historia = history.history

plt.figure(figsize=(10, 6))
plt.plot(historia['rmse'], label='Entrenamiento (1990)', color='#1f77b4', linewidth=2)
plt.plot(historia['val_rmse'], label='Validación (2005)', color='#d62728', linewidth=2) # En rojo para destacar el error

# Título claro para tu TFG
plt.title('Evolución del RMSE: Efecto de Sobreajuste (Overfitting) en el MLP Base', fontsize=14)
plt.xlabel('Épocas', fontsize=12)
plt.ylabel('RMSE (W/m²)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.7)

# Guardar la imagen con el nombre que indica el problema
plt.savefig('rmse_mlp_base_50epocas_sobreajuste.png', dpi=300, bbox_inches='tight')
plt.close()

print("¡Gráfica guardada como 'rmse_mlp_base_50epocas_sobreajuste.png'!")