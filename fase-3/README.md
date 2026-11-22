# Fase 3: API

API inicial del proyecto de predicción de tarifas Uber/Lyft. Incluye
`GET /health` y `POST /predict`; el endpoint de predicción carga el artefacto
serializado de Fase 2 bajo demanda y reutiliza su pipeline completo.

## Instalación y ejecución

Desde esta carpeta:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
TRAIN_DATA_PATH=../rideshare_kaggle.csv uvicorn main:app --reload
```

Consulta `http://127.0.0.1:8000/health` para comprobar el estado,
`http://127.0.0.1:8000/docs` para abrir la documentación interactiva.

Por defecto, `/predict` usa `fase-2/models/v1/modelo.joblib`. Se puede elegir
otra versión existente mediante `MODEL_VERSION` y cambiar la carpeta base con
`MODEL_DIR`:

```bash
MODEL_VERSION=v2 uvicorn main:app --reload
```

La API carga el modelo cuando recibe la primera predicción; por eso
`GET /health` puede responder aunque el artefacto no esté disponible. Si no
puede cargarlo, `POST /predict` devuelve HTTP `503`.

El cuerpo de `POST /predict` debe ser un objeto JSON con las 13 columnas
guardadas en `columnas_entrada`. Los valores numéricos aceptan `null`, que el
`SimpleImputer(strategy="median")` guardado en el pipeline transforma usando la
mediana aprendida durante el entrenamiento. Las columnas categóricas deben ser
texto no nulo; el artefacto no tiene una regla de imputación categórica. Los
campos ausentes, adicionales o de tipo incompatible producen HTTP `422` con el
nombre del campo.

Ejemplo:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "distance": 0.91,
    "surge_multiplier": 1.0,
    "latitude": 42.35,
    "longitude": -71.06,
    "temperature": 38.0,
    "apparentTemperature": 36.0,
    "humidity": 0.7,
    "windSpeed": 3.0,
    "windGust": 4.0,
    "visibility": 10.0,
    "pressure": 1010.0,
    "name": "Black SUV",
    "cab_type": "Uber"
  }'
```

La respuesta contiene `prediction` y `model_version`.

Cada `POST /predict`, incluso si FastAPI rechaza el cuerpo antes de ejecutar el
endpoint, agrega una fila a `fase-4/logs/predict_requests.csv`. El archivo se
crea junto con su encabezado en la primera solicitud y se abre en modo append
para conservar las anteriores. Las columnas son timestamp UTC, código HTTP,
versión del modelo (vacía si no se llegó a usar) y predicción (vacía si falló).
La ruta puede cambiarse con `PREDICT_LOG_PATH`.

Para persistir los registros al ejecutar en Docker, monta el directorio de logs
del host y deja `PREDICT_LOG_PATH` dentro del punto de montaje:

```bash
mkdir -p fase-4/logs
docker run --rm \
  -p 8000:8000 \
  -v "$(pwd)/fase-2/models:/app/fase-2/models" \
  -v "$(pwd)/fase-4/logs:/app/fase-4/logs" \
  -e PREDICT_LOG_PATH=/app/fase-4/logs/predict_requests.csv \
  <imagen-api> \
  uvicorn main:app --host 0.0.0.0 --port 8000
```

## Entrenamiento desde la API

`POST /train` reutiliza `train_model()` de `fase-2/train.py`; el CLI de Fase 2
continúa usando la misma función. La API toma la ruta del CSV de
`TRAIN_DATA_PATH`; `TRAIN_TARGET` y `TRAIN_SEED` son opcionales y por defecto
usan `price` y `42`. Si no se envía cuerpo, aplica esa configuración. También
se puede sobreescribir el objetivo o semilla con un JSON:

```bash
curl -X POST http://127.0.0.1:8000/train
curl -X POST http://127.0.0.1:8000/train \
  -H 'Content-Type: application/json' \
  -d '{"target": "price", "seed": 123}'
```

La API crea un identificador nuevo con prefijo `api-`, conserva las versiones
anteriores y guarda modelo, métricas y metadatos dentro de
`MODEL_DIR/<version>/` (por defecto, `fase-2/models/`). Responde HTTP `201` con
el identificador y las métricas. Si no está configurado el dataset, devuelve
HTTP `503`; si el CSV no cumple el esquema, devuelve HTTP `422`.

## Prueba

```bash
python3 -m pytest
```

Las pruebas validan el estado de salud, la predicción con artefacto real, los
campos faltantes, tipos incompatibles, imputación de un nulo numérico, rechazo
de nulo categórico, el error HTTP `503` cuando falta el modelo, el entrenamiento
con cuerpo vacío o configuración explícita y los registros de llamadas exitosas
y rechazadas a `/predict`.
