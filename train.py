import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Leer los datos desde el archivo CSV
csv_file = 'datos_convertidos.csv'
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

# Usar las columnas 'humidity' y 'light' como entradas (X)
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

# Crear el modelo LSTM
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_scaled.shape[1], X_scaled.shape[2])))
model.add(Dense(1))  # Salida para la predicción de la temperatura
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
model.fit(X_scaled, y, epochs=50, batch_size=4)

# Guardar el modelo entrenado
model.save('model_lstm.h5')
