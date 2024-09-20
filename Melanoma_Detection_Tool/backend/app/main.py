import streamlit as st
from utils import save_image, preprocess_image
from model import predict_melanoma
from PIL import Image

# Título de la aplicación
st.title("Detector de Melanoma")

# Subir imagen usando Streamlit
uploaded_file = st.file_uploader("Sube una imagen lo más nítida posible de un lunar o lesión cutánea para obtener un diagnóstico del riesgo de melanoma: ", type=["jpg", "jpeg", "png"])

# Si el usuario sube una imagen
if uploaded_file is not None:
    # Guardar la imagen subida
    image_path = save_image(uploaded_file)

    # Preprocesar la imagen para hacer la predicción
    processed_image = preprocess_image(image_path)

    # Realizar la predicción con el modelo
    diagnosis, certainty = predict_melanoma(processed_image)

    # Mostrar el resultado del diagnóstico y la certeza
    st.write(f"**Diagnóstico:** {diagnosis}")
    st.write(f"**Certeza:** {certainty:.2f}%")

    # Mostrar la imagen cargada por el usuario
    st.image(uploaded_file, caption="Imagen cargada", use_column_width=True)
