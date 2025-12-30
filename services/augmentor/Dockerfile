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
COPY services/augmentor/requirements.txt .
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
COPY services/augmentor/requirements-test.txt .
RUN pip3 install --no-cache-dir -r requirements-test.txt

# Copy shared module first (needed for imports)
COPY services/shared/ ./shared/

# Copy configuration files
COPY config/ ./config/

# Copy application and tests
COPY services/augmentor/app/ ./app/
COPY services/augmentor/tests/ ./tests/

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

# Copy shared module and application code
COPY services/shared/ ./shared/
COPY services/augmentor/app/ ./app/

# Copy configuration files
COPY config/ ./config/

# Expose port
EXPOSE 8004

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8004/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8004"]
