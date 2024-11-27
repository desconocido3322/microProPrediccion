import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras import metrics
import tensorflow as tf
import sys

# Configuración de semillas para reproducibilidad
seed = 12122008
np.random.seed(seed)
tf.random.set_seed(seed)

# Convertir el argumento a una lista de valores
try:
    datos = list(eval(sys.argv[1]))  # Convierte el argumento en una lista
    if not isinstance(datos, list) or len(datos) != 2:  # Esperamos solo 2 valores: [humedad, luz]
        raise ValueError
except:
    print("Error: El argumento debe ser una lista de 2 valores. Ejemplo: '[13.2, 13.3]'")
    sys.exit(1)

# Cargar el modelo entrenado
try:
    model = load_model("model.h5", custom_objects={'mse': metrics.MeanSquaredError()})
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    sys.exit(1)

# Convertir la lista de datos a un formato compatible con el modelo
# El modelo espera dos entradas, no tres, por lo que se ajusta la forma de los datos
datos = np.array(datos).reshape((1, 2))  # 1 muestra con 2 características

# Generar predicciones para completar una hora (12 predicciones si son intervalos de 5 minutos)
ultimo_valor = None
for _ in range(12):  # 12 predicciones = 1 hora si los intervalos son de 5 minutos
    # Realizar la predicción
    prediccion = model.predict(datos, verbose=0)[0][0]
    ultimo_valor = prediccion  # Guardar el último valor predicho
    
    # Actualizar la lista de entrada: los nuevos datos consisten en los dos valores actuales y la predicción
    datos = np.array([datos[0][1], prediccion]).reshape((1, 2))  # La nueva entrada es [luz, predicción]

# Guardar el último valor predicho en un archivo de texto
try:
    with open("predicho.txt", "w") as file:
        file.write(f"{ultimo_valor:.2f}")
    print(f"Último valor predicho después de 1 hora: {ultimo_valor:.2f}")
    print("El valor ha sido guardado en 'predicho.txt'.")
except Exception as e:
    print(f"Error al guardar el archivo: {e}")
