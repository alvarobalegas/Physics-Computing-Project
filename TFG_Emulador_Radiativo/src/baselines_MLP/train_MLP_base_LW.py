import h5py
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN DE RUTAS ---
DATA_DIR = "ClimART_DATA" # Ajusta esto si tus datos están en la carpeta anterior (ej: "../ClimART_DATA")
STATS_PATH = os.path.join(DATA_DIR, "statistics.npz")

def cargar_y_procesar_datos_lw(año, stats_path):
    print(f"Cargando datos de ONDA LARGA del año {año}...")
    stats = np.load(stats_path)
    
    ruta_inputs = os.path.join(DATA_DIR, f"inputs/{año}.h5")
    with h5py.File(ruta_inputs, 'r') as f:
        X_globals = np.array(f['globals'])
        X_levels = np.array(f['levels'])
        X_layers = np.array(f['layers'])[:, :, :14]
        
    # --- EL CAMBIO FÍSICO: ONDA LARGA (Longwave) ---
    ruta_outputs = os.path.join(DATA_DIR, f"outputs_pristine/{año}.h5")
    with h5py.File(ruta_outputs, 'r') as f:
        Y_rldc = np.array(f['rldc']) # Longwave Downwelling
        Y_rluc = np.array(f['rluc']) # Longwave Upwelling
        
    Y = np.concatenate([Y_rldc, Y_rluc], axis=1)

    X_globals = (X_globals - stats['globals_mean']) / (stats['globals_std'] + 1e-8)
    X_levels = (X_levels - stats['levels_mean']) / (stats['levels_std'] + 1e-8)
    X_layers = (X_layers - stats['layers_mean'][:14]) / (stats['layers_std'][:14] + 1e-8)

    N = X_globals.shape[0]
    X_levels_flat = X_levels.reshape(N, -1) 
    X_layers_flat = X_layers.reshape(N, -1) 
    X_final = np.concatenate([X_globals, X_levels_flat, X_layers_flat], axis=1)
    
    return X_final, Y

# --- FASE DE PIPELINE DE DATOS ---
X_train, Y_train = cargar_y_procesar_datos_lw(1990, STATS_PATH)
X_val, Y_val = cargar_y_procesar_datos_lw(2005, STATS_PATH)

BATCH_SIZE = 128
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# --- DISEÑO DEL MLP BASE (Sin Regularización) ---
modelo_base = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(512, activation=tf.nn.gelu),
    tf.keras.layers.Dense(256, activation=tf.nn.gelu),
    tf.keras.layers.Dense(256, activation=tf.nn.gelu),
    tf.keras.layers.Dense(100, activation='linear')
])

modelo_base.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
    loss='mse',
    metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')]
)

print("\nIniciando entrenamiento MLP BASE (Onda Larga)...")
# Entrenamos 50 épocas sin Early Stopping para forzar el sobreajuste
history_base = modelo_base.fit(train_dataset, validation_data=val_dataset, epochs=50, verbose=1)

# --- GUARDADO ---
modelo_base.save("modelo_climart_mlp_base_LW.keras")
historia = history_base.history

plt.figure(figsize=(10, 6))
plt.plot(historia['rmse'], label='Entrenamiento (1990)', color='blue', linewidth=2)
plt.plot(historia['val_rmse'], label='Validación (2005)', color='red', linewidth=2)
plt.title('MLP Base Onda Larga: Evolución del RMSE (Sobreajuste)', fontsize=14)
plt.xlabel('Épocas', fontsize=12)
plt.ylabel('RMSE (W/m²)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('curva_rmse_mlp_base_LW.png', dpi=300, bbox_inches='tight')
plt.close()
print("Entrenamiento Base finalizado. Revisa 'curva_rmse_mlp_base_LW.png'.")