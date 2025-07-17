FROM python:3.9-buster

# Evitar que Python escriba archivos .pyc y forzar salida en consola
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Agregar la ruta raíz del proyecto al PYTHONPATH
ENV PYTHONPATH=/app

WORKDIR /app

# Instalar dependencias del sistema que requiere OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# Copiar el fichero de requerimientos e instalar dependencias de Python
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copiar el resto del código del proyecto
COPY . .

# Comando para iniciar la aplicación Streamlit
CMD ["streamlit", "run", "app/main.py", "--server.port", "8501"]
