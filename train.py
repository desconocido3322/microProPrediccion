#libs
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

import random as rd
import matplotlib.pyplot as plt
import os

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

csv_file = 'datos_convertidos.csv'
df = pd.read_csv(csv_file)

df['timestamp'] = pd.to_datetime(df['timestamp'])

df['hour'] = df['timestamp'].dt.floor('H')

df = df.drop(columns=['_id', '__v', 'timestamp'])

grouped = df.groupby('hour').mean()

print(grouped)

filtered_dataFrame = grouped[['temperature', 'humidity', 'light']]
print(filtered_dataFrame)

model_path = "model.h5"

tempData = filtered_dataFrame['temperature'].values
humidityData = filtered_dataFrame['humidity'].values
lightData = filtered_dataFrame['light'].values

print(tempData)
print(humidityData)
print(lightData)


X = np.column_stack((humidityData, lightData))  
y = tempData 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_path = "model.h5"

if os.path.exists(model_path):
    print(f"Cargando modelo existente desde {model_path}...")
    model = load_model(model_path, custom_objects={'mse': MeanSquaredError()})
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
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

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=4,
    validation_data=(X_test, y_test),
    verbose=1
)

model.save(model_path)
print(f"Modelo guardado en {model_path}.")

loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f'Validation Loss: {loss:.4f}')
print(f'Mean Absolute Error (MAE): {mae:.4f}')

predictions = model.predict(X_test)

r2 = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

print(f'R^2 Score: {r2:.4f}')
print(f'Mean Squared Error (MSE): {mse:.4f}')

print("\nValores Reales:")
print(y_test)
print("\nPredicciones:")
print(predictions.flatten())
