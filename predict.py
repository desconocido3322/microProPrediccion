import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Últimos valores de humedad, luz y temperatura (puedes reemplazar con los datos actuales)
ultimo_humedad = 70  # Último valor de humedad
ultimo_luz = 550  # Último valor de luz

# Preparar los datos de entrada para la predicción
entrada = np.array([[ultimo_humedad, ultimo_luz]])  # 1 muestra, 2 características

# Normalizar los datos de entrada (de acuerdo con la normalización utilizada en el entrenamiento)
scaler = MinMaxScaler()
entrada_scaled = scaler.fit_transform(entrada)  # Normalizar para que tenga el mismo rango que los datos de entrenamiento

# Redimensionar la entrada para el modelo LSTM: (samples, time_steps, features)
entrada_scaled = entrada_scaled.reshape((1, 1, 2))  # 1 muestra, 1 paso de tiempo, 2 características

# Cargar el modelo entrenado (asegúrate de tener el modelo guardado como "model_lstm.h5")
model = load_model("model_lstm.h5")

# Realizar la predicción
prediccion = model.predict(entrada_scaled, verbose=0)[0][0]
print(f"Predicción de la temperatura en 1 hora: {prediccion:.2f}")
