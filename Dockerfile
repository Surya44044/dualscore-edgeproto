# Lightweight CPU-optimized Python base image
FROM python:3.11-slim

# Prevent Python from writing pyc files to disk and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

WORKDIR /app

# Model artifacts are not baked into the image.
# The API will run in demo_mode=True if /app/models is empty.
RUN mkdir -p /app/models /app/artifacts

# Install system dependencies needed for runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency requirements first for efficient layer caching
COPY requirements.txt .

# Install CPU-only PyTorch and python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and models (no raw/processed datasets)
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY examples/ /app/examples/

# Expose default HTTP port
EXPOSE 8000

# Healthcheck endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Start Uvicorn HTTP server listening on 0.0.0.0 and dynamic PORT
CMD ["sh", "-c", "uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
