from fastapi.testclient import TestClient
from app.melanoma_detection_app import app  # Asegúrate de importar tu app correctamente

client = TestClient(app)

def test_upload_image():
    with open("tests/test_images/test_image.jpg", "rb") as img:
        response = client.post("/predict/", files={"file": img})
        assert response.status_code == 200
        assert "prediction" in response.json()

def test_upload_invalid_image_size():
    with open("tests/test_images/test_image_invalid_size.jpg", "rb") as img:
        response = client.post("/predict/", files={"file": img})
        assert response.status_code == 200  # La API debe devolver 200 OK
        assert "prediction" in response.json()  # Verifica que la predicción esté presente en la respuesta

def test_upload_unsupported_file_type():
    with open("tests/test_images/test_image.txt", "rb") as txt:
        response = client.post("/predict/", files={"file": ("test_image.txt", txt, "text/plain")})
    assert response.status_code == 415
    assert response.json() == {"detail": "Unsupported file type. Please upload a JPEG or PNG image."}

# Nota: Crea un archivo test_image_invalid_size.jpg con un tamaño diferente a 224x224 para esta prueba.
