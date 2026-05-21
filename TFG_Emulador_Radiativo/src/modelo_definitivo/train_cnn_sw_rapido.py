import h5py
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN DE RUTAS ---
DATA_DIR = "ClimART_DATA" 
STATS_PATH = os.path.join(DATA_DIR, "statistics.npz")

def cargar_y_procesar_datos_sw(año, stats_path, fraccion=1.0):
    print(f"Cargando datos de ONDA CORTA del año {año} (Fracción: {fraccion*100}%)...")
    stats = np.load(stats_path)
    
    ruta_inputs = os.path.join(DATA_DIR, f"inputs/{año}.h5")
    with h5py.File(ruta_inputs, 'r') as f:
        X_globals = np.array(f['globals'])
        X_levels = np.array(f['levels'])
        X_layers = np.array(f['layers'])[:, :, :14]
        
    ruta_outputs = os.path.join(DATA_DIR, f"outputs_pristine/{año}.h5")
    with h5py.File(ruta_outputs, 'r') as f:
        Y_rsdc = np.array(f['rsdc']) # Onda Corta Descendente
        Y_rsuc = np.array(f['rsuc']) # Onda Corta Ascendente
        
    Y = np.concatenate([Y_rsdc, Y_rsuc], axis=1)

    X_globals = (X_globals - stats['globals_mean']) / (stats['globals_std'] + 1e-8)
    X_levels = (X_levels - stats['levels_mean']) / (stats['levels_std'] + 1e-8)
    X_layers = (X_layers - stats['layers_mean'][:14]) / (stats['layers_std'][:14] + 1e-8)

    N = X_globals.shape[0]
    X_levels_flat = X_levels.reshape(N, -1) 
    X_layers_flat = X_layers.reshape(N, -1) 
    X_final = np.concatenate([X_globals, X_levels_flat, X_layers_flat], axis=1)
    
    # --- REDUCCIÓN ALEATORIA DE DATOS ---
    if fraccion < 1.0:
        limite = int(N * fraccion)
        np.random.seed(42) # Semilla fija para reproducibilidad
        indices = np.random.permutation(N)
        X_final = X_final[indices][:limite]
        Y = Y[indices][:limite]
    
    return X_final, Y

# --- PIPELINE DE DATOS ---
FRACCION_USO = 0.1  # Usamos el 10% para hiper-optimizar rápido

X_train, Y_train = cargar_y_procesar_datos_sw(1990, STATS_PATH, fraccion=FRACCION_USO)
X_val, Y_val = cargar_y_procesar_datos_sw(2005, STATS_PATH, fraccion=FRACCION_USO)

BATCH_SIZE = 128
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# --- ARQUITECTURA CNN 1D (Para optimizar hiperparámetros) ---
def construir_modelo_cnn(input_shape):
    modelo = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_shape,)),
        tf.keras.layers.Reshape((input_shape, 1)),
        
        # Bloque 1: Vamos a probar si 64 filtros son suficientes
        tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation=tf.nn.gelu, padding='same'),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(0.3), # Puedes probar a subir esto a 0.3 o 0.4
        
        # Bloque 2
        tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation=tf.nn.gelu, padding='same'),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Flatten(),
        
        tf.keras.layers.Dense(256, activation=tf.nn.gelu),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(100, activation='relu') # Volvemos a 'relu' porque en Onda Corta no hay flujos negativos
    ])
    return modelo

modelo_cnn = construir_modelo_cnn(X_train.shape[1])

modelo_cnn.compile(
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

print("\nIniciando entrenamiento RÁPIDO CNN (Onda Corta)...")
history_cnn = modelo_cnn.fit(
    train_dataset, 
    validation_data=val_dataset, 
    epochs=50, 
    callbacks=[early_stopping], 
    verbose=1
)