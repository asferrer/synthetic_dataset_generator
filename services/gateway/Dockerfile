# ============================================
# Gateway Service - API Orchestrator
# ============================================
FROM python:3.12-slim

# Metadata
LABEL maintainer="Synthetic Dataset Generator"
LABEL description="API Gateway for orchestrating synthetic data generation services"
LABEL version="1.0"

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY gateway/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared module and application code
COPY shared/ ./shared/
COPY gateway/app/ ./app/

# Create directories for shared volumes
RUN mkdir -p /shared/images /shared/depth /shared/output

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the service
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
