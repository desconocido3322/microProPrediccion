import pandas as pd
import json
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os
from tensorflow.keras.losses import MeanSquaredError

# Cargar archivo JSON y convertirlo en un DataFrame
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

# Preparar datos para el modelo
tempData = filtered_dataFrame['temperature'].values
humidityData = filtered_dataFrame['humidity'].values
lightData = filtered_dataFrame['light'].values

# Crear datos en formato secuencial para LSTM
time_steps = 3  # Ventana de tiempo para secuencias
features = 2    # Número de características (humidity, light)

def create_sequences(data, labels, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(labels[i + time_steps])
    return np.array(X), np.array(y)

# Combinar las características en un solo array
data = np.column_stack((humidityData, lightData))

# Crear las secuencias para LSTM
X, y = create_sequences(data, tempData, time_steps)

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ruta del modelo LSTM
model_path = "model.h5"

# Crear o cargar el modelo LSTM
if os.path.exists(model_path):
    print(f"Cargando modelo LSTM existente desde {model_path}...")
    model = load_model(model_path, custom_objects={'mse': MeanSquaredError()})
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
else:
    print("No se encontró un modelo LSTM existente. Creando uno nuevo...")
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(time_steps, features), return_sequences=True),
        LSTM(32, activation='relu', return_sequences=False),
        Dense(1, activation='linear')  # Salida de temperatura
    ])
    model.compile(
        optimizer='adam',
        loss=MeanSquaredError(),
        metrics=['mae']
    )
    model.summary()

# Entrenar el modelo
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=4,
    validation_data=(X_test, y_test),
    verbose=1
)

# Guardar el modelo
model.save(model_path)
print(f"Modelo LSTM guardado en {model_path}.")

# Evaluar el modelo
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f'Validation Loss: {loss:.4f}')
print(f'Mean Absolute Error (MAE): {mae:.4f}')

# Hacer predicciones
predictions = model.predict(X_test)

# Calcular métricas de desempeño
r2 = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

print(f'R^2 Score: {r2:.4f}')
print(f'Mean Squared Error (MSE): {mse:.4f}')

# Mostrar resultados
print("\nValores Reales:")
print(y_test)
print("\nPredicciones:")
print(predictions.flatten())
