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
filtered_dataFrame = grouped[['temperature', 'humidity', 'light']]

# Tomar los últimos valores de humedad y luz
humedad_input = filtered_dataFrame['humidity'].iloc[-1]  # Último valor de humedad
luz_input = filtered_dataFrame['light'].iloc[-1]  # Último valor de luz

# Mostrar los valores que estamos usando
print(f"Última humedad: {humedad_input}")
print(f"Última luz: {luz_input}")

# Preparar los datos para la predicción
datos_input = np.array([humedad_input, luz_input]).reshape(1, 1, 2)  # Redimensionar para el modelo

# Normalizar los datos de entrada con el scaler (se supone que ya tienes un scaler ajustado)
scaler = MinMaxScaler()
datos_input_scaled = scaler.fit_transform(datos_input.reshape(-1, 2))  # Normalización de 2 características (humedad, luz)

# Redimensionar para el modelo LSTM
datos_input_scaled = datos_input_scaled.reshape(1, 1, 2)  # [1 muestra, 1 timestep, 2 características]

# Cargar el modelo entrenado
try:
    model = load_model("model_lstm.h5")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit(1)

# Realizar la predicción
prediccion = model.predict(datos_input_scaled, verbose=0)[0][0]
print(f"Predicción de la temperatura: {prediccion:.2f}")

# Guardar la predicción en un archivo de texto
with open('predicho.txt', 'w') as f:
    f.write(f'{prediccion:.2f}\n')

print("Predicción guardada en 'predicho.txt'")
