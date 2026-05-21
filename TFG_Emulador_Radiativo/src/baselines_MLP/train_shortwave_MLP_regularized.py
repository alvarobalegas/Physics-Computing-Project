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
    Carga los datos H5, filtra para cielo prístino, aplana y normaliza.
    """
    print(f"Cargando datos del año {año}...")
    stats = np.load(stats_path)
    
    # Leer Inputs
    ruta_inputs = os.path.join(DATA_DIR, f"inputs/{año}.h5")
    with h5py.File(ruta_inputs, 'r') as f:
        X_globals = np.array(f['globals'])
        X_levels = np.array(f['levels'])
        X_layers = np.array(f['layers'])[:, :, :14] # Solo 14 canales para pristine-sky
        
    # Leer Outputs (Flujos solares descendente y ascendente)
    ruta_outputs = os.path.join(DATA_DIR, f"outputs_pristine/{año}.h5")
    with h5py.File(ruta_outputs, 'r') as f:
        Y_rsdc = np.array(f['rsdc']) 
        Y_rsuc = np.array(f['rsuc']) 
        
    Y = np.concatenate([Y_rsdc, Y_rsuc], axis=1) # Forma (N, 100)

    # Normalización Z-score (Usando las claves correctas descubiertas)
    X_globals = (X_globals - stats['globals_mean']) / (stats['globals_std'] + 1e-8)
    X_levels = (X_levels - stats['levels_mean']) / (stats['levels_std'] + 1e-8)
    X_layers = (X_layers - stats['layers_mean'][:14]) / (stats['layers_std'][:14] + 1e-8)

    # Aplanado para el MLP
    N = X_globals.shape[0]
    X_levels_flat = X_levels.reshape(N, -1) 
    X_layers_flat = X_layers.reshape(N, -1) 
    X_final = np.concatenate([X_globals, X_levels_flat, X_layers_flat], axis=1)
    
    return X_final, Y

# --- FASE DE CARGA Y PIPELINE ---
X_train, Y_train = cargar_y_procesar_datos(1990, STATS_PATH)
X_val, Y_val = cargar_y_procesar_datos(2005, STATS_PATH)

BATCH_SIZE = 128
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
train_dataset = train_dataset.shuffle(buffer_size=10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# --- DISEÑO DE LA RED NEURONAL REGULARIZADA ---
def construir_modelo_mlp_regularizado(input_shape):
    modelo = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_shape,)),
        
        # Capa Oculta 1 + Dropout
        tf.keras.layers.Dense(512, activation=tf.nn.gelu),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        # Capa Oculta 2 + Dropout
        tf.keras.layers.Dense(256, activation=tf.nn.gelu),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        # Capa Oculta 3 + Dropout
        tf.keras.layers.Dense(256, activation=tf.nn.gelu),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        # Capa de Salida
        tf.keras.layers.Dense(100, activation='linear') 
    ])
    return modelo

modelo = construir_modelo_mlp_regularizado(X_train.shape[1])

# --- COMPILACIÓN Y ENTRENAMIENTO ---
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
    loss='mse',
    metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')]
)

# Configuración de la Parada Temprana (Early Stopping)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',       
    patience=5,               
    restore_best_weights=True,
    verbose=1
)

print("\nIniciando entrenamiento con Dropout y Early Stopping...")
history = modelo.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50, 
    callbacks=[early_stopping],
    verbose=1
)

# --- AUTOMATIZACIÓN DE EXTRACCIÓN DE RESULTADOS PARA LA MEMORIA ---
print("\nGenerando gráficas y reportes...")

# 1. Guardar modelo (Formato moderno Keras)
modelo.save("modelo_climart_mlp_regularizado.keras")

# 2. Guardar historial numérico
historia = history.history
with open('resultados_MLP_regularizado_numeros.txt', 'w') as f:
    f.write("Epoca\tLoss_Train\tRMSE_Train\tLoss_Val\tRMSE_Val\n")
    for i in range(len(historia['loss'])):
        f.write(f"{i+1}\t{historia['loss'][i]:.4f}\t{historia['rmse'][i]:.4f}\t{historia['val_loss'][i]:.4f}\t{historia['val_rmse'][i]:.4f}\n")

# 3. Gráfica de RMSE
plt.figure(figsize=(10, 6))
plt.plot(historia['rmse'], label='Entrenamiento (1990)', color='#1f77b4', linewidth=2)
plt.plot(historia['val_rmse'], label='Validación (2005)', color='#ff7f0e', linewidth=2)
plt.title('Evolución del Error (RMSE) con Regularización', fontsize=14)
plt.xlabel('Épocas', fontsize=12)
plt.ylabel('RMSE (W/m²)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('curva_rmse_mlp_regularizado.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Gráfica de Loss
plt.figure(figsize=(10, 6))
plt.plot(historia['loss'], label='Entrenamiento (MSE)', color='#1f77b4', linewidth=2)
plt.plot(historia['val_loss'], label='Validación (MSE)', color='#ff7f0e', linewidth=2)
plt.title('Evolución de la Función de Pérdida con Regularización', fontsize=14)
plt.xlabel('Épocas', fontsize=12)
plt.ylabel('Pérdida (MSE)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('curva_loss_mlp_regularizado.png', dpi=300, bbox_inches='tight')
plt.close()

print("¡Todo listo! Revisa tu carpeta para ver las gráficas y el archivo de texto.")