# Melanoma Detection Tool

This project is based on **Convolutional Neural Networks (CNNs)**, which are used in computer vision for tasks such as image classification and object detection. It is a **binary classification model** using deep learning techniques. The aim of the project was to create a system capable of distinguishing between benign and malignant skin lesions, including moles and other types of skin conditions. To achieve this, I trained a convolutional neural network on a dataset of **37,000 images** of various skin lesions.

## Model Overview

- **Parameters**: 7.5 million
- **Accuracy**: 96% after fine-tuning
- **Dataset**: 37,000 images of benign and malignant skin lesions

The developed model enables accurate predictions about the presence or absence of melanoma. After training, I achieved a **96% accuracy rate** in diagnosis, following fine-tuning to improve performance and minimise overfitting.

## Key Techniques Used

The project involved continuous iteration and experimentation with different configurations to optimise the model. Key techniques used include:

- **L1 and L2 regularisation**
- **Dropout** to reduce overfitting
- **Data augmentation** for better generalisation

These techniques helped improve the model's ability to generalise to new data and avoid overfitting on the training set.

## Deployment

Once the model was ready, I created a **Docker container** to ensure portability and ease of deployment across different environments. Additionally, I used **FastAPI** to develop an API interface, which allowed for local testing and refinement of the system until I achieved satisfactory results.

The application was then deployed on **Streamlit**, making it accessible to anyone. Users can upload an image of a mole or skin lesion and receive a quick diagnosis on the risk of melanoma, providing an easy-to-use tool for early detection of potential malignant conditions.

## Test the Model

You can test the model here:
[https://melanomadetectiontool-nacjacds.streamlit.app/](https://melanomadetectiontool-nacjacds.streamlit.app/)

## How to Run the Project Locally

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/melanoma-detection-tool.git
cd melanoma-detection-tool
Step 2: Build the Docker Container
bash
Copy code
docker build -t melanoma-detection .
Step 3: Run the Application
bash
Copy code
streamlit run app.py
Step 4: Upload an Image
Once the application is running, you can upload an image of a mole or lesion to receive a melanoma risk diagnosis.

Technologies Used
Python
TensorFlow/Keras
Docker
FastAPI
Streamlit
License
This project is licensed under the MIT License - see the LICENSE file for details.

less
Copy code

### ¿Qué incluye este README.md?
- **Sección "Test the Model"**: Ahora tienes un enlace que dirige a los usuarios a tu aplicación en Streamlit donde pueden probar el modelo en tiempo real.
- El resto del **README.md** incluye instrucciones claras sobre cómo clonar y ejecutar el proyecto localmente, junto con una explicación detallada de los aspectos técnicos del proyecto.

Este README proporcionará a los usuarios una forma rápida de entender tu proyecto y probar el modelo sin necesidad de configuraciones complejas.
