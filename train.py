import pandas as pd
import json
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import os
from tensorflow.keras.losses import MeanSquaredError

json_file = 'datos.csv'

with open(json_file, 'r') as file:
    data = json.load(file)  

if isinstance(data, list):
    dataFrame = pd.DataFrame(data)
else:
    dataFrame = pd.DataFrame([data])  

csv_file = 'datos_convertidos.csv'
dataFrame.to_csv(csv_file, index=False)

print(f"Archivo CSV creado: {csv_file}")

df = pd.read_csv(csv_file)

df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.floor('H')
df = df.drop(columns=['_id', '__v', 'timestamp'])

grouped = df.groupby('hour').mean()
filtered_dataFrame = grouped[['temperature', 'humidity', 'light']]

model_path = "model.h5"
tempData = filtered_dataFrame['temperature'].values
humidityData = filtered_dataFrame['humidity'].values
lightData = filtered_dataFrame['light'].values

# Reestructurar los datos en forma de secuencia
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length + 1):  # Ajuste aquí para asegurarse de que las secuencias sean del tamaño adecuado
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)

seq_length = 2  # Longitud de la secuencia (puedes ajustarlo)
X = np.column_stack((humidityData, lightData))
X_seq = create_sequences(X, seq_length)  # Convertir X en secuencias
y = tempData[seq_length - 1:]  # Ajustar el tamaño de y para que coincida con X

# Verificar las formas de X_train y X_test antes del reshape
X_train, X_test, y_train, y_test = train_test_split(X_seq, y, test_size=0.2, random_state=42)

print(f"Forma de X_train antes del reshape: {X_train.shape}")
print(f"Forma de X_test antes del reshape: {X_test.shape}")

# Verificar que las formas sean correctas
if X_train.shape[0] % seq_length != 0:
    print(f"Ajustando el número de muestras en X_train ({X_train.shape[0]}) para que sea divisible por {seq_length}.")
    X_train = X_train[:-(X_train.shape[0] % seq_length)]  # Recortar para que sea divisible por seq_length

if X_test.shape[0] % seq_length != 0:
    print(f"Ajustando el número de muestras en X_test ({X_test.shape[0]}) para que sea divisible por {seq_length}.")
    X_test = X_test[:-(X_test.shape[0] % seq_length)]  # Recortar para que sea divisible por seq_length

# Redimensionar los datos para que sean compatibles con LSTM
X_train = X_train.reshape((X_train.shape[0], seq_length, 2))  # 2 es el número de características (humedad y luz)
X_test = X_test.reshape((X_test.shape[0], seq_length, 2))  # Hacer lo mismo con X_test

print(f"Forma de X_train después del reshape: {X_train.shape}")
print(f"Forma de X_test después del reshape: {X_test.shape}")

# Normalización de datos
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, 2))
X_test = scaler.transform(X_test.reshape(-1, 2))

X_train = X_train.reshape((X_train.shape[0], seq_length, 2))
X_test = X_test.reshape((X_test.shape[0], seq_length, 2))

# Cargar el modelo existente o crear uno nuevo
if os.path.exists(model_path):
    print(f"Cargando modelo existente desde {model_path}...")
    model = load_model(model_path, custom_objects={'mse': MeanSquaredError()})
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
else:
    print("No se encontró un modelo existente. Creando uno nuevo...")
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(seq_length, 2), return_sequences=False),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss=MeanSquaredError(), metrics=['mae'])
    model.summary()

# Entrenamiento
history = model.fit(X_train, y_train, epochs=50, batch_size=4, validation_data=(X_test, y_test), verbose=1)

# Guardar el modelo entrenado
model.save(model_path)
print(f"Modelo guardado en {model_path}.")

# Evaluación del modelo
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f'Validation Loss: {loss:.4f}')
print(f'Mean Absolute Error (MAE): {mae:.4f}')

# Predicciones y métricas
predictions = model.predict(X_test)
r2 = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

print(f'R^2 Score: {r2:.4f}')
print(f'Mean Squared Error (MSE): {mse:.4f}')

print("\nValores Reales:")
print(y_test)
print("\nPredicciones:")
print(predictions.flatten())
