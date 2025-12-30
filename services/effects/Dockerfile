# ============================================
# Effects Service - Realism Pipeline
# ============================================
FROM python:3.12-slim

# Metadata
LABEL maintainer="Synthetic Dataset Generator"
LABEL description="Realism effects service for synthetic image generation"
LABEL version="1.0"

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (libgl1 replaces libgl1-mesa-glx in newer Debian)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY effects/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY effects/app/ ./app/

# Create directories for shared volumes
RUN mkdir -p /shared/images /shared/effects /shared/output /app/cache

# Expose port
EXPOSE 8003

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

# Run the service
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003"]
