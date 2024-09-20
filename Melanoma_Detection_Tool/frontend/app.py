import streamlit as st
import requests

st.title("Melanoma Detection Tool")

# Texto explicativo debajo del título
st.markdown("""This system uses artificial intelligence to help detect potential melanomas. The model has been trained to differentiate between melanomas and other skin lesions. However, it is important to note that some lesions, which may also be cancerous, could be classified as 'Non-Melanoma'.

Melanoma is the most aggressive type of skin cancer, and early detection is crucial to increase the chances of successful treatment. There are other lesions, such as Lentigo Maligna, which, although not melanomas in their initial stage, have the potential to develop into melanomas if left untreated.

This system does not replace a professional medical evaluation. Any suspicious lesion should be examined by a dermatologist for proper diagnosis and treatment. This tool in no way replaces a professional medical assessment.""")

# Subir una imagen
uploaded_file = st.file_uploader("Upload an image of the mole", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Mostrar la imagen cargada
    st.image(uploaded_file, caption='Image uploaded.', use_column_width=True)

    # Botón para enviar la imagen
    if st.button('Diagnose'):
        files = {"file": uploaded_file.getvalue()}
        url = "http://backend:8000/predict/"
        
        # Enviar la imagen al backend
        response = requests.post(url, files={"file": uploaded_file})
        
        # Verificar la respuesta
        if response.status_code == 200:
            result = response.json()
            diagnosis = result.get("diagnosis", ["Unknown", None])
            
            # Formatear el diagnóstico y la certeza
            if diagnosis[0] == "Negativo para melanoma":
                st.success(f"The analysis indicates that the mole is *{diagnosis[0]}*. However, remain vigilant for any changes.")
            else:
                st.warning(f"The analysis suggests that the mole could be *{diagnosis[0]}*. We recommend a visit to the dermatologist. ")
            
            if diagnosis[1] is not None:
                st.write(f"We are {diagnosis[1]:.2f}% sure of this result.")
            else:
                st.write("We couldn't calculate the certainty level at this time.")
        else:
            st.error(f"Server response error. Status code: {response.status_code}")