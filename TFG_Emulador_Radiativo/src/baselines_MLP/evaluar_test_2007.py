import h5py
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN DE RUTAS ---
DATA_DIR = "ClimART_DATA"
STATS_PATH = os.path.join(DATA_DIR, "statistics.npz")
MODELO_PATH = "modelo_climart_mlp_regularizado.keras"

def cargar_y_procesar_datos(año, stats_path):
    print(f"Cargando datos de TEST del año {año}...")
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

# --- EVALUACIÓN Y PREDICCIÓN ---

print("\nCargando el modelo y los datos...")
modelo = tf.keras.models.load_model(MODELO_PATH)
X_test, Y_test = cargar_y_procesar_datos(2007, STATS_PATH)

print("\nRealizando predicciones sobre el año 2007...")
Y_pred = modelo.predict(X_test, batch_size=256)

# Para las gráficas, aplanamos todo para ver el error global en todos los niveles
Y_test_flat = Y_test.flatten()
Y_pred_flat = Y_pred.flatten()

# --- GENERACIÓN DE GRÁFICAS PROFESIONALES (VERSIÓN MEJORADA) ---
print("\nGenerando gráficas de evaluación final...")

# 1. Gráfica de Dispersión (Real vs Predicho)
plt.figure(figsize=(8, 8))
# AÑADIDO CLAVE: bins='log' para que el punto masivo de la noche (0,0) no oculte los datos del día
plt.hexbin(Y_test_flat, Y_pred_flat, gridsize=50, cmap='Blues', mincnt=1, bins='log')
plt.colorbar(label='Densidad de puntos (Escala Logarítmica)')

lims = [0, max(Y_test_flat.max(), Y_pred_flat.max())]
plt.plot(lims, lims, 'r--', alpha=0.75, zorder=3, label="Ideal (1:1)")

plt.title('Comparación: Valores Reales vs. Predicciones (Test 2007)', fontsize=14)
plt.xlabel('Flujo Radiativo Real (W/m²)', fontsize=12)
plt.ylabel('Flujo Radiativo Predicho (W/m²)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('test_scatter_real_vs_pred_log.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Histograma de Residuos (Errores)
errores = Y_pred_flat - Y_test_flat
rmse_final = np.sqrt(np.mean(errores**2))

plt.figure(figsize=(10, 6))
plt.hist(errores, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(0, color='red', linestyle='dashed', linewidth=2)

# AÑADIDO CLAVE: Escala logarítmica en el eje Y para ver las "colas" de los errores
plt.yscale('log')

plt.title('Distribución de Errores (Residuos) - Test 2007', fontsize=14)
plt.xlabel('Error (Predicho - Real) [W/m²]', fontsize=12)
plt.ylabel('Frecuencia (Escala Logarítmica)', fontsize=12)
plt.grid(True, alpha=0.3)

plt.text(0.95, 0.95, f'RMSE Final: {rmse_final:.2f} W/m²', 
         transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.savefig('test_histograma_errores_log.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\n¡Gráficas mejoradas guardadas con éxito!")
print(f"Métrica definitiva RMSE: {rmse_final:.4f} W/m²")