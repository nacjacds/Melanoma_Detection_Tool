import streamlit as st
from backend.app.utils import save_image, preprocess_image
from backend.app.model import predict_melanoma
from PIL import Image

# Título de la aplicación
st.title("Melanoma Detection Tool")

# Subir la imagen
uploaded_file = st.file_uploader("Sube una imagen para predecir si tiene melanoma", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Mostrar la imagen subida
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen subida", use_column_width=True)

    # Guardar la imagen en el directorio 'uploads'
    image_path = save_image(uploaded_file)

    # Preprocesar la imagen
    processed_image = preprocess_image(image_path)

    # Realizar la predicción
    diagnosis, certainty = predict_melanoma(processed_image)

    # Mostrar los resultados de la predicción
    st.write(f"Diagnóstico: {diagnosis}")
    st.write(f"Porcentaje de certeza: {certainty:.2f}%")
