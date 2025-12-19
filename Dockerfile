# Base image con CUDA 12.8 support para RTX 5090 (Blackwell architecture)
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

# Evitar prompts interactivos durante instalación
ENV DEBIAN_FRONTEND=noninteractive

# Evitar que Python escriba archivos .pyc y forzar salida en consola
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Agregar la ruta raíz del proyecto al PYTHONPATH
ENV PYTHONPATH=/app

WORKDIR /app

# Instalar Python 3.12 y dependencias del sistema
# Nota: git es necesario para instalar Depth Anything V2 desde source
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Configurar python3.12 como python3 por defecto
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Instalar pip para Python 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Actualizar pip, setuptools y wheel
RUN python3 -m pip install --upgrade pip setuptools wheel

# Copiar el fichero de requerimientos
COPY requirements.txt .

# FIX 1: Forzar instalación de blinker ignorando versión del sistema (evita error distutils)
RUN pip install --ignore-installed blinker

# Instalar PyTorch 2.9.0 con CUDA 12.8 support
RUN pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128

# Instalar resto de dependencias desde requirements.txt
RUN pip install -r requirements.txt

RUN git clone https://github.com/DepthAnything/Depth-Anything-V2.git

ENV PYTHONPATH="${PYTHONPATH}:/app/Depth-Anything-V2"

# Crear directorio para checkpoints
RUN mkdir -p /app/checkpoints

# Copiar el resto del código
COPY . .

# Comando de inicio
CMD ["streamlit", "run", "app/main.py", "--server.port", "8501"]