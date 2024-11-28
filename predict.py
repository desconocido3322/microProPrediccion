import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Cargar los datos procesados (por ejemplo, 'datos_convertidos.csv')
csv_file = 'datos_convertidos.csv'
df = pd.read_csv(csv_file)

# Asegurarse de que los datos estén procesados como se espera
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.floor('H')
df = df.drop(columns=['_id', '__v', 'timestamp'])
grouped = df.groupby('hour').mean()

# Filtrar las columnas relevantes (humedad y luz)
filtered_dataFrame = grouped[['temperature', 'humidity', 'presion']]

# Tomar los últimos 3 valores de humedad y luz
humedad_input = filtered_dataFrame['humidity'].iloc[-10:].values  # Últimos 10 valores de humedad
temperature_input = filtered_dataFrame['temperature'].iloc[-10:].values  # Últimos 10 valores de humedad
presion_input = filtered_dataFrame['presion'].iloc[-10:].values  # Últimos 10 valores de luz

# Concatenar los datos de humedad y luz en un solo array de 3 timesteps y 3 características
datos_input = np.array([humedad_input, presion_input,temperature_input]).T.reshape(1, 10, 3)  # Forma (1, 3, 3)

# Mostrar los valores de entrada que estamos utilizando
print(f"Últimos 10 valores de humedad: {humedad_input}")
print(f"Últimos 10 valores de presion: {presion_input}")
print(f"Últimos 10 valores de temperature: {temperature_input}")
print(f"Datos de entrada para el modelo: {datos_input}")

# Normalizar los datos de entrada con el scaler (se supone que ya tienes un scaler ajustado)
scaler = MinMaxScaler()
datos_input_scaled = scaler.fit_transform(datos_input.reshape(-1, 3))  # Normalización de 3 características (humedad, presion, temperatura)

# Redimensionar para el modelo LSTM
datos_input_scaled = datos_input_scaled.reshape(1, 10, 3)  # [1 muestra, 10 timesteps, 3 características]

# Cargar el modelo entrenado
try:
    model = load_model("model.h5")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit(333)

# Realizar la predicción
prediccion = model.predict(datos_input_scaled, verbose=0)
print("aaaaaaaaaaaaaaaaaaaaaaaaaa",prediccion)
print(f"Predicción de la temperatura: {prediccion:.2f}")

# Guardar la predicción en un archivo de texto
with open('predicho.txt', 'w') as f:
    f.write(f'{prediccion:.2f}\n')

print("Predicción guardada en 'predicho.txt'")
