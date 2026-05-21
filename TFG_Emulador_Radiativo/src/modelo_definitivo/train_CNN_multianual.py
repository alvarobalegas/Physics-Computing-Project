import h5py
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN DE RUTAS ---
DATA_DIR = "ClimART_DATA" 
STATS_PATH = os.path.join(DATA_DIR, "statistics.npz")

# ¡NUEVO! Variables globales arriba del todo para que las funciones las detecten bien
TIPO_ONDA = "Shortwave (SW)" # Cambia a "Longwave (LW)" para radiación térmica
TIPO_CIELO = "All Sky"  # Cambia a "Clear Sky" o "All Sky"

# 1. CARGA INDIVIDUAL DE UN AÑO (AHORA 100% DINÁMICA)
def cargar_y_procesar_datos(año, stats_path, fraccion=1.0):
    print(f"  -> Leyendo año {año}...")
    stats = np.load(stats_path)
    
    ruta_inputs = os.path.join(DATA_DIR, f"inputs/{año}.h5")
    with h5py.File(ruta_inputs, 'r') as f:
        X_globals = np.array(f['globals'])
        X_levels = np.array(f['levels'])
        X_layers = np.array(f['layers'])[:, :, :14]
        
    # --- AUTO-SELECCIÓN DE CARPETA SEGÚN EL TIPO DE CIELO ---
    if TIPO_CIELO == "Pristine Sky":
        carpeta = "outputs_pristine"
    elif TIPO_CIELO == "Clear Sky":
        carpeta = "outputs_clear_sky"
    else:
        carpeta = "outputs_all_sky" 
        
    ruta_outputs = os.path.join(DATA_DIR, f"{carpeta}/{año}.h5")
    
    with h5py.File(ruta_outputs, 'r') as f:
        # --- AUTO-SELECCIÓN DE VARIABLES (SW vs LW) ---
        if "SW" in TIPO_ONDA:
            Y_down = np.array(f['rsdc']) 
            Y_up = np.array(f['rsuc'])
        else:
            Y_down = np.array(f['rldc']) 
            Y_up = np.array(f['rluc']) 
            
    Y = np.concatenate([Y_down, Y_up], axis=1)

    X_globals = (X_globals - stats['globals_mean']) / (stats['globals_std'] + 1e-8)
    X_levels = (X_levels - stats['levels_mean']) / (stats['levels_std'] + 1e-8)
    X_layers = (X_layers - stats['layers_mean'][:14]) / (stats['layers_std'][:14] + 1e-8)

    N = X_globals.shape[0]
    X_levels_flat = X_levels.reshape(N, -1) 
    X_layers_flat = X_layers.reshape(N, -1) 
    X_final = np.concatenate([X_globals, X_levels_flat, X_layers_flat], axis=1)
    
    if fraccion < 1.0:
        limite = int(N * fraccion)
        np.random.seed(42) 
        indices = np.random.permutation(N)
        X_final = X_final[indices][:limite]
        Y = Y[indices][:limite]
    
    return X_final, Y

# 2. MEZCLADORA MULTIANUAL
def cargar_datos_multianual(años, stats_path, fraccion=1.0):
    X_total = []
    Y_total = []
    for año in años:
        X_temp, Y_temp = cargar_y_procesar_datos(año, stats_path, fraccion)
        X_total.append(X_temp)
        Y_total.append(Y_temp)
        
    X_final = np.vstack(X_total)
    Y_final = np.vstack(Y_total)
    
    N = X_final.shape[0]
    np.random.seed(42)
    indices = np.random.permutation(N)
    
    return X_final[indices], Y_final[indices]

# --- CONFIGURACIÓN DEL EXPERIMENTO ---
AÑOS_TRAIN = [1985, 1990, 2000] 
AÑOS_VAL = [2005]               
FRACCION_USO = 0.10

print(f"\n--- PREPARANDO DATOS: {TIPO_CIELO} | {TIPO_ONDA} (Fracción: {FRACCION_USO*100}%) ---")
print("Entrenamiento:")
X_train, Y_train = cargar_datos_multianual(AÑOS_TRAIN, STATS_PATH, fraccion=FRACCION_USO)
print("Validación:")
X_val, Y_val = cargar_datos_multianual(AÑOS_VAL, STATS_PATH, fraccion=FRACCION_USO)

BATCH_SIZE = 128
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# --- ARQUITECTURA GANADORA ---
def construir_modelo_optimo(input_shape):
    modelo = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(input_shape,)),
        tf.keras.layers.Reshape((input_shape, 1)),
        tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation=tf.nn.gelu, padding='same'),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(0.3), 
        tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation=tf.nn.gelu, padding='same'),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(0.3), 
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation=tf.nn.gelu),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(100, activation='relu') 
    ])
    return modelo

modelo_cnn = construir_modelo_optimo(X_train.shape[1])

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

print(f"\n--- INICIANDO ENTRENAMIENTO MULTIANUAL: {TIPO_CIELO} - {TIPO_ONDA} ---")
history_cnn = modelo_cnn.fit(
    train_dataset, 
    validation_data=val_dataset, 
    epochs=50, 
    callbacks=[early_stopping], 
    verbose=1 
)

# ==============================================================================
# --- EXAMEN FINAL Y GENERACIÓN DE GRÁFICAS ---
# ==============================================================================
print(f"\n--- INICIANDO EVALUACIÓN DE TEST Pura (Año 2007) ---")
X_test, Y_test = cargar_y_procesar_datos(2007, STATS_PATH, fraccion=FRACCION_USO)

loss_test, rmse_test = modelo_cnn.evaluate(X_test, Y_test, verbose=0)
print("==================================================")
print(f" RESULTADO FINAL TEST RMSE: {rmse_test:.4f} W/m² ")
print("==================================================")

# --- GUARDADO AUTOMÁTICO DEL MODELO ---
nombre_cielo_limpio = TIPO_CIELO.replace(" ", "_")
nombre_onda_limpia = "SW" if "SW" in TIPO_ONDA else "LW"
nombre_modelo = f"Modelo_CNN_{nombre_cielo_limpio}_{nombre_onda_limpia}_{int(FRACCION_USO*100)}pct.keras"
modelo_cnn.save(nombre_modelo)
print(f"¡Modelo guardado con éxito como '{nombre_modelo}'!")

# --- GENERANDO GRÁFICAS DE ANÁLISIS FÍSICO AUTOMATIZADAS ---
print("Generando gráficas...")
Y_pred = modelo_cnn.predict(X_test, batch_size=1024)
errores = Y_pred - Y_test

fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# --- GRÁFICA 1: Hexbin Plot (Densidad logarítmica) ---
hb = axs[0].hexbin(Y_test.flatten(), Y_pred.flatten(), gridsize=100, cmap='Blues', bins='log', mincnt=1)
cb = fig.colorbar(hb, ax=axs[0], label='Densidad de puntos (Nº en Escala Logarítmica)')
min_val = np.min([Y_test.min(), Y_pred.min()])
max_val = np.max([Y_test.max(), Y_pred.max()])
axs[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Predicción Perfecta (y=x)')

axs[0].set_title(f'Dispersión Hexagonal: {TIPO_CIELO} | {TIPO_ONDA}')
axs[0].set_xlabel('Valor Real Físico (W/m²)')
axs[0].set_ylabel('Predicción de la CNN (W/m²)')
axs[0].legend(loc='upper left')
axs[0].grid(True, linestyle='--', alpha=0.6)

# --- GRÁFICA 2: Histograma de Errores ---
axs[1].hist(errores.flatten(), bins=100, color='coral', edgecolor='black', alpha=0.7)
axs[1].axvline(x=0, color='red', linestyle='--', lw=2, label='Error Cero')

texto_rmse = f"RMSE Test:\n{rmse_test:.2f} W/m²"
axs[1].text(0.95, 0.95, texto_rmse, transform=axs[1].transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.9))

axs[1].set_title(f'Error (Pred - Real): {TIPO_CIELO} | {TIPO_ONDA}\nDatos: {int(FRACCION_USO*100)}% Multianual')
axs[1].set_xlabel('Error (W/m²)')
axs[1].set_ylabel('Frecuencia (Nº de puntos)')
axs[1].legend(loc='upper left')
axs[1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()

# --- GUARDADO AUTOMÁTICO DE LA GRÁFICA ---
nombre_grafica = f"Test_Hexbin_{nombre_cielo_limpio}_{nombre_onda_limpia}_{int(FRACCION_USO*100)}pct.png"
plt.savefig(nombre_grafica, dpi=300)
print(f"¡Gráficas guardadas automáticamente como '{nombre_grafica}'!")
plt.show()