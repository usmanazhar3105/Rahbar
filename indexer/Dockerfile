# Production-ready, security-hardened Dockerfile for RAHBAR Indexer Job
FROM python:3.11-slim
RUN pip install --upgrade pip setuptools==78.1.1

# Prevent Python from writing bytecode and enable unbuffered batch output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    TMPDIR=/tmp

WORKDIR /app

# System dependencies required for PDF processing and optical character recognition (OCR):
# - tesseract-ocr: Tesseract OCR engine used for image-only/scanned documents without embedded text layers
# - poppler-utils: pdftoppm utility used by pdf2image to convert PDF pages into raster images for OCR
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        poppler-utils && \
    rm -rf /var/lib/apt/lists/*

# Create a dedicated non-root user and group with UID/GID 1000
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/false --no-create-home appuser

# Copy and install dependencies first for optimal layer caching
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy indexer pipeline code
COPY pipeline/ ./pipeline/

# Ensure application files are owned by the non-root user
RUN chown -R appuser:appgroup /app

# Switch to non-root user for Kubernetes security context compliance
USER 1000

# Exec form ensures signals reach Python PID 1
CMD ["python", "-m", "pipeline.main"]

