# ============================================
# Depth Service - Depth-Anything-3
# ============================================
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# Metadata
LABEL maintainer="Synthetic Dataset Generator"
LABEL description="Depth estimation service using Depth-Anything-3"
LABEL version="1.0"

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH="/app:/app/Depth-Anything-3"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3

# Install pip for Python 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

WORKDIR /app

# Clone Depth-Anything-3
RUN git clone --depth 1 https://github.com/ByteDance-Seed/Depth-Anything-3.git

RUN python -m pip install --ignore-installed blinker

# Install PyTorch with CUDA 12.8 support
# Instalamos numpy antes para asegurar compatibilidad
RUN python -m pip install --no-cache-dir numpy

RUN python -m pip install --no-cache-dir \
    torch>=2.0 \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu128

# Install xformers
RUN python -m pip install --no-cache-dir --no-build-isolation xformers

RUN python -m pip install --no-cache-dir hatchling editables pathspec

# Install Depth-Anything-3
RUN cd Depth-Anything-3 && \
    python -m pip install -e . --no-build-isolation

# Copy requirements and install dependencies
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Create directories for shared volumes
RUN mkdir -p /shared/images /shared/depth /app/checkpoints

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run the service
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]