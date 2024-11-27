import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Paso 1: Leer el archivo JSON (datos.csv) y convertirlo en un DataFrame
json_file = 'datos.csv'  # Aquí está el archivo JSON de entrada
with open(json_file, 'r') as file:
    data = json.load(file)

# Convertir los datos a un DataFrame de pandas
if isinstance(data, list):
    dataFrame = pd.DataFrame(data)
else:
    dataFrame = pd.DataFrame([data])

# Guardar el DataFrame en un archivo CSV
csv_file = 'datos_convertidos.csv'  # Archivo CSV de salida
dataFrame.to_csv(csv_file, index=False)
print(f"Archivo CSV creado: {csv_file}")

# Paso 2: Procesar el CSV para entrenamiento
df = pd.read_csv(csv_file)

# Asegurarse de que la columna 'timestamp' sea de tipo datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Crear una nueva columna 'hour' que contiene la hora redondeada
df['hour'] = df['timestamp'].dt.floor('H')

# Eliminar columnas no necesarias
df = df.drop(columns=['_id', '__v', 'timestamp'])

# Agrupar por hora y calcular la media
grouped = df.groupby('hour').mean()

# Mostrar el dataframe agrupado
print(grouped)

# Paso 3: Usar las columnas 'humidity' y 'light' como entradas (X)
# Y 'temperature' como salida (y)
humidityData = grouped['humidity'].values
lightData = grouped['light'].values
tempData = grouped['temperature'].values

# Normalizar los datos de entrada (X)
scaler = MinMaxScaler()
X = np.column_stack((humidityData, lightData))  # Combinación de características
X_scaled = scaler.fit_transform(X)  # Normalizar las características

# Redimensionar los datos para el modelo LSTM
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, 2))  # (samples, time_steps, features)

# La variable objetivo (temperatura)
y = tempData

# Paso 4: Crear y entrenar el modelo LSTM
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_scaled.shape[1], X_scaled.shape[2])))
model.add(Dense(1))  # Salida para la predicción de la temperatura
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
model.fit(X_scaled, y, epochs=50, batch_size=4)

# Guardar el modelo entrenado
model.save('model_lstm.h5')
