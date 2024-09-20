# Imagen base de Python
FROM python:3.9

# Crear directorio de trabajo
WORKDIR /app

# Copiar el archivo de requisitos
COPY requirements.txt .

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el código de la aplicación
COPY . .

# Exponer el puerto de FastAPI (8000)
EXPOSE 8000

# Exponer el puerto de Streamlit (8501)
EXPOSE 8501

# Comando para ejecutar FastAPI y Streamlit en segundo plano
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 & streamlit run frontend/app.py --server.port 8501"]