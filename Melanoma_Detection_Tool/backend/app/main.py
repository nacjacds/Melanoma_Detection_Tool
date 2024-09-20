import streamlit as st
from utils import save_image, preprocess_image
from model import predict_melanoma
from PIL import Image

# Título de la aplicación
st.title("Melanoma Detection Tool")

# Subir imagen usando Streamlit
uploaded_file = st.file_uploader("""Upload the clearest possible image of a mole or skin lesion to receive a melanoma risk diagnosis. please be aware that the algorithm's effectiveness may be influenced by factors such as the image's characteristics, angle, lighting, contrast, and overall quality.""", type=["jpg", "jpeg", "png"])



# Si el usuario sube una imagen
if uploaded_file is not None:
    # Guardar la imagen subida
    image_path = save_image(uploaded_file)

    # Preprocesar la imagen para hacer la predicción
    processed_image = preprocess_image(image_path)

    # Realizar la predicción con el modelo
    diagnosis, certainty = predict_melanoma(processed_image)

    # Mostrar el resultado del diagnóstico y la certeza
    st.write(f"**Diagnosis:** {diagnosis}")
    st.write(f"**Certainty:** {certainty:.2f}%")

    # Mostrar la imagen cargada por el usuario
    st.image(uploaded_file, caption="Image loaded", use_column_width=True)

    # Añadir un texto informativo debajo del área de carga de imagen
st.write("""
This system uses artificial intelligence to help detect potential melanomas. The model has been trained on thousands of images of benign moles, melanomas, and other types of skin lesions. It is a binary classification model, meaning it will only provide a diagnosis of Melanoma or NotMelanoma.
The focus on melanoma is due to it being the most aggressive and dangerous form of skin cancer. However, the NotMelanoma category may include benign moles, seborrheic keratosis, and less aggressive forms of skin cancer that may still require medical attention. 

There are also other lesions, such as Lentigo Maligna, which, while not melanomas in their initial stage, have the potential to develop into melanomas if left untreated.

Melanoma is the most aggressive type of skin cancer, and early detection is crucial to improve the chances of successful treatment.
Although the model has been trained to differentiate between melanomas and other skin lesions, it is important to note that some cancerous lesions could be classified as NotMelanoma. Therefore, this system should not be considered a substitute for a professional medical evaluation. Any suspicious lesion should be examined by a dermatologist for proper diagnosis and treatment.

My aim is to make the model as accurate as possible, but please be aware that the algorithm's effectiveness may be influenced by factors such as the image's characteristics, angle, lighting, contrast, and overall quality.

Disclaimer: This is an anonymous and free application, and it is not intended to replace a doctor's diagnosis. If you have any concerns about a mole or skin lesion, you should consult a healthcare professional.
This project is part of my training as a Data Scientist, aimed at demonstrating the skills I have developed and showcasing them in my portfolio.
""")
