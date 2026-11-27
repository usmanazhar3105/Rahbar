# Production-ready, security-hardened Dockerfile for RAHBAR Evaluation & RAGOps Runner
FROM python:3.11-slim
RUN pip install --upgrade pip setuptools==78.1.1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    TMPDIR=/tmp

WORKDIR /app

RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/false --no-create-home appuser

COPY eval/requirements.txt ./eval/requirements.txt
RUN pip install --no-cache-dir -r ./eval/requirements.txt

COPY eval/ ./eval/
COPY scripts/ ./scripts/
COPY corpus/ ./corpus/
COPY api/ ./api/

RUN chown -R appuser:appgroup /app

USER 1000

CMD ["python", "eval/run_eval.py", "--base-url", "http://api-service:8000/api", "--output", "/data/results/"]
