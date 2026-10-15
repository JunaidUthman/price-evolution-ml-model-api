# ==============================================================================
# Multi-Stage Dockerfile for FastAPI ML Inference Service (product-price-api)
# Python 3.12 (Debian Bookworm Stable) + XGBoost + Uvicorn
# ==============================================================================

# Stage 1: Install Python dependencies (all packages have binary wheels)
FROM python:3.12-slim-bookworm AS builder

WORKDIR /app

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python packages directly without compiling (fast pre-built wheels)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Minimal runtime image
FROM python:3.12-slim-bookworm AS runner

WORKDIR /app

# libgomp1 is strictly required by XGBoost (OpenMP runtime)
# curl is used for Docker / K8s health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-built virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000

# Create unprivileged system user
RUN useradd -u 1000 -m -s /bin/bash appuser

# Copy application code
COPY app/ ./app/
COPY main.py .

RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
