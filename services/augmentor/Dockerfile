# Augmentor Service Dockerfile
# GPU-enabled for Quality Validation (LPIPS)
# Multi-stage build with test stage

# =============================================================================
# Stage: base - Common dependencies
# =============================================================================
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS base

WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install pip packages
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# =============================================================================
# Stage: test - Run tests during build (fails build if tests fail)
# =============================================================================
FROM base AS test

# Install test dependencies
COPY requirements-test.txt .
RUN pip3 install --no-cache-dir -r requirements-test.txt

# Copy application and tests
COPY app/ ./app/
COPY tests/ ./tests/

# Run tests - build fails if tests fail
RUN python -m pytest tests/ -v --tb=short -x

# Create marker file to prove tests passed
RUN echo "tests passed" > /app/.tests_passed

# =============================================================================
# Stage: production - Final production image
# =============================================================================
FROM base AS production

# Copy marker from test stage - this creates a dependency, ensuring tests run
COPY --from=test /app/.tests_passed /app/.tests_passed

# Copy application code
COPY app/ ./app/

# Expose port
EXPOSE 8004

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8004/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8004"]
