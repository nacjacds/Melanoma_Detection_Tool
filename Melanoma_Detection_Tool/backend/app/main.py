from .utils import save_image, preprocess_image
from fastapi import FastAPI, File, UploadFile
from .model import predict_melanoma

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
