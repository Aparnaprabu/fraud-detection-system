# Production Dockerfile for Real-Time Fraud Detection API
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies (OpenMP runtime required by XGBoost and LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code, dashboard, and serialized model artifacts
COPY src/ /app/src/
COPY dashboard/ /app/dashboard/
COPY models/ /app/models/

# Expose port for FastAPI
EXPOSE 8000

# Health check to ensure service is ready
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI inference service with Uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
