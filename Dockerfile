# Lightweight CPU-optimized Python base image
FROM python:3.11-slim

# Prevent Python from writing pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

WORKDIR /app

# Model artifacts are not baked into the image.
# The API will run in demo_mode=True if /app/models is empty.
RUN mkdir -p /app/models /app/artifacts

# Install minimal system dependencies for runtime and healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy minimal API dependencies first for layer caching
COPY requirements-api.txt .

# Install CPU-only Python dependencies for inference API
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy application source code and lightweight config/examples only.
# No raw datasets or large model artifacts are copied.
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY examples/ /app/examples/

# Expose HTTP port
EXPOSE 8000

# Healthcheck endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start Uvicorn HTTP server
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
