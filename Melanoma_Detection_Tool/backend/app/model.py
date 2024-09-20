from tensorflow.keras.models import load_model
import numpy as np

# Cargar el modelo entrenado desde la carpeta `model/`
import zipfile
from tensorflow.keras.models import load_model
import os

# Ruta al archivo comprimido .keras (si es un archivo zip)
zip_path = "../model/melanoma-diagnosis.h5"

# Ruta donde se descomprimirá
extract_path = "../model/"

# Verificar si el archivo es un zip y descomprimirlo
if zipfile.is_zipfile(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    # Una vez descomprimido, cargar el modelo
    model = load_model(os.path.join(extract_path, 'melanoma-diagnosis.h5'))
else:
    # Si no es un zip, intentar cargar el archivo directamente
    model = load_model(zip_path)

def predict_melanoma(processed_image):
    # La imagen debe tener la forma (224, 224, 3)
    # Expande las dimensiones de la imagen a (1, 224, 224, 3)
    image_batch = np.expand_dims(processed_image, axis=0)

    # Realiza la predicción con el modelo
    prediction = model.predict(image_batch)[0]  # Toma el primer (y único) valor del lote
    
    # La predicción será un valor entre 0 y 1 que representa la probabilidad de melanoma
    melanoma_probability = prediction[0]  # Si el modelo tiene una sola salida

    # Define el umbral para el diagnóstico
    threshold = 0.5
    if melanoma_probability >= threshold:
        diagnosis = "Positivo para melanoma"
    else:
        diagnosis = "Negativo para melanoma"
    
    # Calcula el porcentaje de certeza basado en la predicción
    certainty = melanoma_probability * 100 if melanoma_probability >= threshold else (1 - melanoma_probability) * 100

    # Retorna el diagnóstico y el porcentaje de certeza
    return diagnosis, certainty