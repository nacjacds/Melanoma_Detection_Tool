from PIL import Image
import numpy as np
import os

def save_image(uploaded_file):
    # Crea una carpeta para guardar las imágenes subidas si no existe
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    
    # Usa el atributo `name` en lugar de `filename`
    image_path = os.path.join("uploads", uploaded_file.name)
    
    # Guarda la imagen con el método adecuado para Streamlit
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return image_path

def preprocess_image(image_path):
    # Abre la imagen usando PIL y convierte a formato RGB (si está en otro formato)
    image = Image.open(image_path).convert("RGB")

    # Redimensiona la imagen al tamaño que espera el modelo (224x224 píxeles)
    image = image.resize((224, 224))

    # Convierte la imagen a un array NumPy con tres canales (RGB)
    image_array = np.array(image)

    # Normaliza los valores de los píxeles entre 0 y 1
    image_array = image_array / 255.0

    return image_array