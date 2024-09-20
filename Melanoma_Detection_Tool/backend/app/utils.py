from PIL import Image
import numpy as np
import os

def save_image(uploaded_file):
    # Crea una carpeta para guardar las imágenes subidas si no existe
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    
    # Define la ruta donde se guardará la imagen
    image_path = os.path.join("uploads", uploaded_file.filename)
    
    # Guarda la imagen
    with open(image_path, "wb") as f:
        f.write(uploaded_file.file.read())
    
    return image_path

def preprocess_image(image_path):
    # Abre la imagen usando PIL y convierte a formato RGB (si está en otro formato)
    image = Image.open(image_path).convert("RGB")

    # Redimensiona la imagen al tamaño que espera el modelo (224x224 píxeles)
    image = image.resize((224, 224))

    # Convierte la imagen a un array NumPy con tres canales (RGB)
    image_array = np.array(image)

    # Asegúrate de que la imagen tenga 3 canales (por si acaso fuera una imagen en escala de grises)
    if len(image_array.shape) == 2:  # Si es una imagen en escala de grises
        image_array = np.stack([image_array]*3, axis=-1)

    # Normaliza los valores de los píxeles (opcional, dependiendo del modelo)
    image_array = image_array / 255.0

    return image_array
