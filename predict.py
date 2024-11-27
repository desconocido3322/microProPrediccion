import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import sys

# Configuración de semillas para reproducibilidad
seed = 12122008
np.random.seed(seed)
tf.random.set_seed(seed)

# Convertir el argumento a una lista de valores
try:
    datos = list(eval(sys.argv[1]))  # Convierte el argumento en una lista
    if not isinstance(datos, list) or len(datos) != 3:
        raise ValueError
except:
    print("Error: El argumento debe ser una lista de 3 valores. Ejemplo: '[13.2, 13.3, 13.4]'")
    sys.exit(1)

# Cargar el modelo entrenado
try:
    model = load_model("model_lstm.h5")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    sys.exit(1)

# Normalizar los datos de entrada
scaler = MinMaxScaler()
datos_scaled = scaler.fit_transform([datos])  # Asegúrate de normalizar los datos antes de la predicción

# Redimensionar los datos para que sean compatibles con LSTM
datos_scaled = datos_scaled.reshape((1, 1, len(datos)))  # [1 muestra, 1 timestep, 3 características]

# Realizar la predicción
prediccion = model.predict(datos_scaled, verbose=0)[0][0]
print(f"Predicción de la temperatura en 1 hora: {prediccion:.2f}")
