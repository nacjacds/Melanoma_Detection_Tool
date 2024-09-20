from tensorflow.keras.models import load_model
import os

# Ruta al archivo del modelo .h5
model_path = "../model/melanoma-diagnosis.h5"

# Verificar si el archivo existe
if os.path.exists(model_path):
    print(f"Archivo de modelo encontrado en {model_path}. Cargando el modelo...")
    model = load_model(model_path)
else:
    raise FileNotFoundError(f"No se encontró el archivo de modelo en {model_path}")

# Función de predicción
def predict_melanoma(image):
    # Aquí va la lógica para preprocesar la imagen y realizar predicciones
    return model.predict(image)
    
    # Calcula el porcentaje de certeza basado en la predicción
    certainty = melanoma_probability * 100 if melanoma_probability >= threshold else (1 - melanoma_probability) * 100

    # Retorna el diagnóstico y el porcentaje de certeza
    return diagnosis, certainty