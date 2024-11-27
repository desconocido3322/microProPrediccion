import pandas as pd
import json
from datetime import datetime
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import LSTM, Input, Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.losses import MeanSquaredError

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import os

# Cargar y procesar los datos
json_file = 'datos.csv'

with open(json_file, 'r') as file:
    data = json.load(file)  

if isinstance(data, list):
    dataFrame = pd.DataFrame(data)
else:
    dataFrame = pd.DataFrame([data])  

csv_file = 'datos_convertidos.csv'
dataFrame.to_csv(csv_file, index=False)

df = pd.read_csv(csv_file)

# Convertir la columna 'timestamp' a datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Agrupar los datos por hora
df['hour'] = df['timestamp'].dt.floor('H')
df = df.drop(columns=['_id', '__v', 'timestamp'])

# Agrupar por hora y calcular la media
grouped = df.groupby('hour').mean()

# Seleccionar las características de interés
filtered_dataFrame = grouped[['temperature', 'humidity', 'light']]

# Convertir a arrays
tempData = filtered_dataFrame['temperature'].values
humidityData = filtered_dataFrame['humidity'].values
lightData = filtered_dataFrame['light'].values

# Normalizar los datos
scaler = MinMaxScaler()
X = np.column_stack((humidityData, lightData))  
y = tempData

X_scaled = scaler.fit_transform(X)

# Redimensionar X para que sea compatible con LSTM
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))  # [muestras, timesteps, características]

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Crear el modelo LSTM
model = Sequential([
    LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),  # 50 neuronas LSTM
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.summary()

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=50, batch_size=4, validation_data=(X_test, y_test), verbose=1)

# Guardar el modelo
model.save('model_lstm.h5')
print("Modelo guardado en 'model_lstm.h5'.")

# Evaluar el modelo
loss, mae = model.evaluate(X_test, y_test)
print(f"Validation Loss: {loss:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")

# Realizar predicciones
predictions = model.predict(X_test)

r2 = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

print(f'R^2 Score: {r2:.4f}')
print(f'Mean Squared Error (MSE): {mse:.4f}')
