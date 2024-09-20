from utils import save_image, preprocess_image
from model import predict_melanoma
from fastapi import FastAPI, File, UploadFile
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Guarda la imagen subida
    image_path = save_image(file)

    # Preprocesa la imagen para el modelo
    processed_image = preprocess_image(image_path)

    # Realiza la predicción con el modelo
    result = predict_melanoma(processed_image)

    # Devuelve el resultado de la predicción
    return {"diagnosis": result}
