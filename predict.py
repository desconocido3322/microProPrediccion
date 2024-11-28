import numpy as np
import pandas as pd
import json
import sys
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Configuración de semillas para reproducibilidad
seed = 12122008
np.random.seed(seed)
import tensorflow as tf
tf.random.set_seed(seed)

# 1. Cargar archivo JSON y convertirlo en un DataFrame
json_file = 'datos.csv'

with open(json_file, 'r') as file:
    data = json.load(file)

if isinstance(data, list):
    dataFrame = pd.DataFrame(data)
else:
    dataFrame = pd.DataFrame([data])

# Guardar como CSV
csv_file = 'datos_convertidos.csv'
dataFrame.to_csv(csv_file, index=False)

print(f"Archivo CSV creado: {csv_file}")

# Leer el archivo CSV y procesar los datos
df = pd.read_csv(csv_file)

# Convertir timestamp a datetime y redondear a la hora
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.floor('H')

# Eliminar columnas no necesarias
df = df.drop(columns=['_id', '__v', 'timestamp'])

# Agrupar datos por hora y calcular la media
grouped = df.groupby('hour').mean()

print(grouped)

# Seleccionar las columnas relevantes
filtered_dataFrame = grouped[['temperature', 'humidity', 'light']]
print(filtered_dataFrame)

# 2. Preparar los datos para predicción
# Escalar los datos (usando MinMaxScaler, asegurándote de usar el mismo rango que en el entrenamiento)
scaler = MinMaxScaler()
scaler.fit(filtered_dataFrame[['humidity', 'light']])  # Ajusta el escalador con las columnas de entrada

# Pedir al usuario que ingrese valores de humedad y luz
try:
    user_input = input("Ingresa valores de humedad y luz (formato: '[50, 700]'): ")
    datos = list(eval(user_input))  # Convertir el input del usuario en una lista
    if not isinstance(datos, list) or len(datos) != 2:
        raise ValueError
except Exception as e:
    print("Error: El argumento debe ser una lista de 2 valores. Ejemplo: '[50, 700]'")
    sys.exit(1)

# Normalizar los datos de entrada
datos_scaled = scaler.transform([datos])  # Asegúrate de normalizar los datos antes de la predicción
print(f"Datos originales: {datos}")
print(f"Datos escalados: {datos_scaled}")

# 3. Cargar el modelo y realizar predicción
model_path = "model.h5"

if not os.path.exists(model_path):
    print(f"No se encontró el modelo entrenado en {model_path}. Asegúrate de entrenar el modelo primero.")
    sys.exit(1)

model = load_model(model_path)

# Realizar la predicción
prediccion = model.predict(datos_scaled, verbose=0)[0][0]
print(f"Predicción de la temperatura: {prediccion:.2f}°C")

# 4. Guardar la predicción en un archivo de texto
with open('predicho.txt', 'w') as f:
    f.write(f'{prediccion:.2f}\n')

print("Predicción guardada en 'predicho.txt'")
