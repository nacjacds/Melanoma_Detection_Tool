import streamlit as st
from utils import save_image, preprocess_image
from model import predict_melanoma

# Título de la aplicación
st.title("Melanoma Detection Tool")

# Cargar imagen usando Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Guardar la imagen
    image_path = save_image(uploaded_file)

    # Preprocesar la imagen
    processed_image = preprocess_image(image_path)

    # Realizar la predicción
    diagnosis, certainty = predict_melanoma(processed_image)

    # Mostrar el resultado
    st.write(f"Diagnosis: {diagnosis}")
    st.write(f"Certainty: {certainty:.2f}%")

    # Mostrar la imagen cargada
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
