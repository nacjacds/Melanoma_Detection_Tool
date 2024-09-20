import streamlit as st
from utils import save_image, preprocess_image
from model import predict_melanoma
from PIL import Image

# Título de la aplicación
st.title("Melanoma Detection Tool")

# Subir imagen usando Streamlit
uploaded_file = st.file_uploader("""This system uses artificial intelligence to help detect potential melanomas. The model has been trained to differentiate between melanomas and other skin lesions. However, it is important to note that some lesions, which may also be cancerous, could be classified as 'Non-Melanoma'.

Melanoma is the most aggressive type of skin cancer, and early detection is crucial to increase the chances of successful treatment. There are other lesions, such as Lentigo Maligna, which, although not melanomas in their initial stage, have the potential to develop into melanomas if left untreated.

This system does not replace a professional medical evaluation. Any suspicious lesion should be examined by a dermatologist for proper diagnosis and treatment. This tool in no way replaces a professional medical assessment.""", type=["jpg", "jpeg", "png"])

# Añadir un texto informativo debajo del área de carga de imagen
st.write("""
Please ensure that the image is clear and well-lit. The quality of the image can significantly impact the accuracy of the diagnosis.
""")

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
