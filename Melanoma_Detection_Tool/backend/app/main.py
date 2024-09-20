import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Cargar el modelo preentrenado (ajusta la ruta según tu archivo guardado)
model = load_model('models/melanoma-diagnosis_3.keras')

# Título de la aplicación
st.title("Melanoma Detection Tool")

# Texto explicativo debajo del título
st.markdown("""This system uses artificial intelligence to help detect potential melanomas. The model has been trained to differentiate between melanomas and other skin lesions. However, it is important to note that some lesions, which may also be cancerous, could be classified as 'Non-Melanoma'.

Melanoma is the most aggressive type of skin cancer, and early detection is crucial to increase the chances of successful treatment. There are other lesions, such as Lentigo Maligna, which, although not melanomas in their initial stage, have the potential to develop into melanomas if left untreated.

This system does not replace a professional medical evaluation. Any suspicious lesion should be examined by a dermatologist for proper diagnosis and treatment. This tool in no way replaces a professional medical assessment.""")

# Función para preprocesar la imagen antes de hacer la predicción
def preprocess_image(image):
    img = image.resize((224, 224))  # Redimensionar a 224x224
    img = np.array(img) / 255.0     # Normalizar entre 0 y 1
    img = np.expand_dims(img, axis=0)  # Añadir dimensión para batch (1, 224, 224, 3)
    return img

# Subir una imagen
uploaded_file = st.file_uploader("Upload an image of the mole", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Mostrar la imagen cargada
    image = Image.open(uploaded_file)
    st.image(image, caption='Image uploaded.', use_column_width=True)

    # Botón para hacer la predicción
    if st.button('Diagnose'):
        # Preprocesar la imagen
        processed_image = preprocess_image(image)

        # Realizar la predicción con el modelo cargado
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        certainty = np.max(prediction) * 100  # Convertir certeza a porcentaje

        # Mapear la predicción a etiquetas (0 = Melanoma, 1 = NotMelanoma)
        if predicted_class == 0:
            diagnosis_label = "Melanoma"
        elif predicted_class == 1:
            diagnosis_label = "NotMelanoma"

        # Mostrar el diagnóstico y la certeza
        if predicted_class == 0:
            st.warning(f"The analysis suggests that the mole could be *{diagnosis_label}*. We recommend a visit to the dermatologist.")
        else:
            st.success(f"The analysis indicates that the mole is *{diagnosis_label}*. However, remain vigilant for any changes.")
        
        st.write(f"We are {certainty:.2f}% sure of this result.")
