# Imagen de la API de predicción (Fases 3 y 4).
# Construir desde la raíz del repositorio:
#   docker build -f fase-3/Dockerfile -t uber-lyft-api:fase3-fase4 .

FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_VERSION=v1 \
    MODEL_DIR=/app/fase-2/models \
    TRAIN_DATA_PATH=/app/fase-2/data/rideshare_kaggle.csv \
    TRAIN_TARGET=price \
    TRAIN_SEED=42 \
    PREDICT_LOG_PATH=/app/fase-4/logs/

WORKDIR /app

# Las dependencias se instalan desde fase-3/requirements.txt, que a su vez
# incluye las de la Fase 2 (pandas, numpy, scikit-learn, joblib y pytest).
COPY fase-2/requirements.txt /app/fase-2/requirements.txt
COPY fase-3/requirements.txt /app/fase-3/requirements.txt
RUN pip install --no-cache-dir -r /app/fase-3/requirements.txt

COPY fase-2 /app/fase-2
COPY fase-3 /app/fase-3
COPY fase-4 /app/fase-4

# Directorios persistentes: datos de entrenamiento, modelos versionados y logs.
RUN mkdir -p /app/fase-2/data /app/fase-2/models /app/fase-4/logs \
    && useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app

USER appuser

WORKDIR /app/fase-3

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

# El servidor escucha en todas las interfaces para ser accesible desde el host.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
