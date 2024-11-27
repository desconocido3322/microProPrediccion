# Librerías necesarias
import pandas as pd
import json
from datetime import datetime
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.losses import MeanSquaredError

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt
import os

# Ruta del archivo JSON
json_file = 'datos.csv'

# Cargar y convertir el archivo JSON a DataFrame
with open(json_file, 'r') as file:
    data = json.load(file)  

if isinstance(data, list):
    dataFrame = pd.DataFrame(data)
else:
    dataFrame = pd.DataFrame([data])  

# Guardar el DataFrame en un archivo CSV
csv_file = 'datos_convertidos.csv'
dataFrame.to_csv(csv_file, index=False)
print(f"Archivo CSV creado: {csv_file}")

# Leer el archivo CSV
df = pd.read_csv(csv_file)

# Convertir la columna 'timestamp' a formato datetime y agrupar datos por hora
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.floor('H')
df = df.drop(columns=['_id', '__v', 'timestamp'])

# Agrupación por hora y cálculo de la media
grouped = df.groupby('hour').mean()
print("Datos agrupados por hora:")
print(grouped)

# Selección de columnas relevantes
filtered_dataFrame = grouped[['temperature', 'humidity', 'light']]
print("Datos filtrados:")
print(filtered_dataFrame)

# Preparar datos para el modelo
tempData = filtered_dataFrame['temperature'].values
humidityData = filtered_dataFrame['humidity'].values
lightData = filtered_dataFrame['light'].values

X = np.column_stack((humidityData, lightData))  # Variables de entrada (X)
y = tempData  # Variable objetivo (y)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ruta del modelo
model_path = "model.h5"

# Cargar modelo si existe, o crear uno nuevo
if os.path.exists(model_path):
    print(f"Cargando modelo existente desde {model_path}...")
    model = load_model(model_path)
else:
    print("No se encontró un modelo existente. Creando uno nuevo...")
    model = Sequential([
        Input(shape=(2,)),  
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')  
    ])
    model.compile(
        optimizer='adam',
        loss=MeanSquaredError(),
        metrics=['mae']
    )
    model.summary()

# Entrenamiento del modelo
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=4,
    validation_data=(X_test, y_test),
    verbose=1
)

# Guardar el modelo entrenado
model.save(model_path)
print(f"Modelo guardado en {model_path}.")

# Evaluar el modelo
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f'Validation Loss: {loss:.4f}')
print(f'Mean Absolute Error (MAE): {mae:.4f}')

# Realizar predicciones
predictions = model.predict(X_test)

# Métricas adicionales
r2 = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
print(f'R^2 Score: {r2:.4f}')
print(f'Mean Squared Error (MSE): {mse:.4f}')

# Mostrar valores reales y predicciones
print("\nValores Reales:")
print(y_test)
print("\nPredicciones:")
print(predictions.flatten())

# Graficar resultados
plt.figure(figsize=(10, 6))
plt.plot(y_test, label="Valores Reales", marker='o')
plt.plot(predictions.flatten(), label="Predicciones", marker='x')
plt.title("Valores Reales vs Predicciones")
plt.xlabel("Índice")
plt.ylabel("Temperatura")
plt.legend()
plt.grid(True)

# Guardar el gráfico
plot_path = "clusters_plot.png"
plt.savefig(plot_path)
print(f"Gráfico guardado en: {plot_path}")
plt.show()
