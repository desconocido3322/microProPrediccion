import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import tensorflow as tf
import sys
import json

json_file = 'datos.csv'

with open(json_file, 'r') as file:
    data = json.load(file)

# Verificar si los datos son una lista o un diccionario
if isinstance(data, list):
    dataFrame = pd.DataFrame(data)
else:
    dataFrame = pd.DataFrame([data])

# Guardar los datos en un archivo CSV
csv_file = 'datos_convertidos.csv'
dataFrame.to_csv(csv_file, index=False)

print(f"Archivo CSV creado: {csv_file}")

# Configuración de semillas para reproducibilidad
seed = 12122008
np.random.seed(seed)
tf.random.set_seed(seed)

# Convertir el argumento a una lista de valores
try:
    datos = list(eval(sys.argv[1]))  # Convierte el argumento en una lista
    if not isinstance(datos, list) or len(datos) != 3:
        raise ValueError
except:
    print("Error: El argumento debe ser una lista de 3 valores. Ejemplo: '[13.2, 13.3, 13.4]'")
    sys.exit(1)

# Cargar el modelo entrenado
try:
    model = load_model("model_lstm.h5")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    sys.exit(1)

# Normalizar los datos de entrada
scaler = MinMaxScaler()
datos_scaled = scaler.fit_transform([datos])  # Asegúrate de normalizar los datos antes de la predicción

# Redimensionar los datos para que sean compatibles con LSTM
datos_scaled = datos_scaled.reshape((1, 1, len(datos)))  # [1 muestra, 1 timestep, 3 características]

# Realizar la predicción
prediccion = model.predict(datos_scaled, verbose=0)[0][0]
print(f"Predicción de la temperatura en 1 hora: {prediccion:.2f}")

# Ejemplo de creación de un gráfico (ajusta según tu necesidad)
plt.scatter([1, 2, 3], [12, 13, 14], c='blue', label='Datos')  # Datos de ejemplo

# Guardar el gráfico como una imagen PNG
plt.savefig('clusters_plot.png')
print("Gráfico guardado en 'clusters_plot.png'")
