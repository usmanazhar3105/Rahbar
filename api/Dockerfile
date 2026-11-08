# Production-ready, security-hardened Dockerfile for RAHBAR API
FROM python:3.11-slim

# Prevent Python from writing bytecode and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    TMPDIR=/tmp

WORKDIR /app

# Create a dedicated non-root user and group with UID/GID 1000
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/false --no-create-home appuser

# Copy and install dependencies first for optimal layer caching
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY app/ ./app/

# Ensure application files are owned by the non-root user
RUN chown -R appuser:appgroup /app

# Non-root user compatibility for Kubernetes readOnlyRootFilesystem: true
USER 1000

EXPOSE 8000

# Exec form ensures signals (SIGTERM) reach uvicorn PID 1 for graceful shutdown
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

