import os
from tensorflow.keras.models import load_model
import numpy as np

# Ruta relativa al archivo .keras
model_dir = os.path.dirname(__file__)  # Obtiene el directorio actual del archivo .py
model_path = os.path.join(model_dir, '..', 'model', 'melanoma-diagnosis.keras')  # Apunta al archivo dentro de la carpeta 'model'

# Verifica si el archivo existe y carga el modelo
if os.path.exists(model_path):
    model = load_model(model_path)
    print(f"Model successfully loaded from {model_path}")
else:
    raise FileNotFoundError(f"Model file not found at {model_path}")

def predict_melanoma(processed_image):
    """ Make the prediction using the loaded model """
    # La imagen debe tener la forma (224, 224, 3)
    image_batch = np.expand_dims(processed_image, axis=0)  # Expande las dimensiones de la imagen
    prediction = model.predict(image_batch)[0]  # Realiza la predicción y toma el primer valor del lote
    
    # Procesar la predicción
    melanoma_probability = prediction[0]  # Supongamos que el modelo tiene una sola salida
    
    # Definir el umbral para el diagnóstico
    threshold = 0.5
    if melanoma_probability >= threshold:
        diagnosis = "Positive for Melanoma"
    else:
        diagnosis = "Negative for Melanoma"
    
    # Calcular la certeza
    certainty = melanoma_probability * 100 if melanoma_probability >= threshold else (1 - melanoma_probability) * 100

    # Retornar el diagnóstico y la certeza
    return diagnosis, certainty