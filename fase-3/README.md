# Fase 3: API de predicción de tarifas

**Proyecto Integrador · Modelos y Simulación de Sistemas I · Universidad de Antioquia**

Integrantes: Alexandra Garavito Vargas, Juan Camilo Mona Luján y Dina Marcela
Reales Corredor.

## Propósito

Esta fase expone el modelo de la Fase 2 como un servicio HTTP con FastAPI. La API
carga un artefacto versionado (`modelo.joblib`), devuelve la tarifa estimada de
una cotización de Uber o Lyft, permite lanzar un reentrenamiento y registra cada
solicitud de predicción para que la Fase 4 calcule métricas de operación.

El modelo es el `Pipeline` de `HistGradientBoostingRegressor` con imputación por
mediana y One-Hot Encoding construido en la Fase 1 y empaquetado en la Fase 2. La
API no reimplementa el preprocesamiento: carga el pipeline completo del artefacto.

## Estructura

```text
fase-3/
├── main.py                 # aplicación FastAPI y lógica de la API
├── cliente.py              # cliente de prueba con httpx
├── ejemplo_payload.json    # cuerpo de ejemplo para POST /predict
├── conftest.py             # hace importable `main` al correr pytest desde la raíz
├── tests/
│   ├── test_api.py         # endpoints, validación y registro de solicitudes
│   └── test_cliente.py     # pruebas del cliente
├── requirements.txt
├── Dockerfile
└── README.md
```

El `conftest.py` proviene del PR #17 y permite ejecutar `python -m pytest` desde
la raíz del repositorio. No es necesario crear otro ni modificar ese archivo.

## Dependencias

`fase-3/requirements.txt` incluye las dependencias de la Fase 2 y las del
servicio:

```text
-r ../fase-2/requirements.txt
fastapi==0.115.12
httpx==0.28.1
pytest==8.4.2
uvicorn==0.34.0
```

La instalación local se hace desde `fase-3/`, porque la primera línea es una
referencia relativa:

```bash
cd fase-3
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Variables de entorno

| Variable | Valor por defecto | Uso |
|---|---|---|
| `MODEL_VERSION` | `v1` | Carpeta de la versión activa dentro de `MODEL_DIR`. Admite letras, números, puntos, guiones y guiones bajos. |
| `MODEL_DIR` | `<raíz>/fase-2/models`; `/app/fase-2/models` en la imagen | Directorio base. El artefacto se busca en `MODEL_DIR/<MODEL_VERSION>/modelo.joblib`. |
| `TRAIN_DATA_PATH` | Sin valor en el código; `/app/fase-2/data/rideshare_kaggle.csv` en la imagen | CSV etiquetado para `POST /train`. |
| `TRAIN_TARGET` | `price` | Columna objetivo cuando la solicitud de entrenamiento no envía `target`. |
| `TRAIN_SEED` | `42` | Semilla cuando la solicitud no envía `seed`. Entero no negativo. |
| `PREDICT_LOG_PATH` | `<raíz>/fase-4/logs/predict_requests.csv`; `/app/fase-4/logs/` en la imagen | Archivo o directorio del registro de `POST /predict`. Si es un directorio, escribe `predict_requests.csv` dentro. |

### Seleccionar o cambiar la versión del modelo

Las versiones disponibles en el repositorio son `v1` (semilla 42) y `v2`
(semilla 123). Para servir `v2`:

```bash
export MODEL_DIR="$(pwd)/fase-2/models"
export MODEL_VERSION=v2
```

El archivo `fase-2/models/v2/modelo.joblib` debe existir. Si no existe,
`GET /health` sigue respondiendo, pero `POST /predict` devuelve `503` porque el
modelo se carga bajo demanda.

En Docker el cambio se hace con `-e MODEL_VERSION=v2` al crear el contenedor. No
basta con reiniciar uno existente: las variables de entorno se fijan al crear el
proceso y hay que recrearlo.

| Versión | Semilla | R² | MAE |
|---|---|---|---|
| `v1` | 42 | 0,9617 | $1,25 |
| `v2` | 123 | 0,9622 | $1,25 |

## Endpoints

### `GET /health`

Comprueba que el proceso esté disponible. No carga el modelo.

```bash
curl http://127.0.0.1:8000/health
```

```json
{"status": "ok", "message": "API en ejecución"}
```

### `POST /predict`

Recibe exactamente las 13 columnas del esquema. El orden no importa; no se
admiten campos faltantes, adicionales ni tipos incompatibles.

| Columna | Tipo | ¿Admite `null`? |
|---|---|---|
| `distance` | numérico | Sí; se imputa con la mediana aprendida |
| `surge_multiplier` | numérico | Sí |
| `latitude` | numérico | Sí |
| `longitude` | numérico | Sí |
| `temperature` | numérico | Sí |
| `apparentTemperature` | numérico | Sí |
| `humidity` | numérico | Sí |
| `windSpeed` | numérico | Sí |
| `windGust` | numérico | Sí |
| `visibility` | numérico | Sí |
| `pressure` | numérico | Sí |
| `name` | texto | No |
| `cab_type` | texto | No |

Solicitud real con la primera fila de `fase-2/data/ejemplo_prediccion.csv`:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d @fase-3/ejemplo_payload.json
```

```json
{"prediction": 26.707956061611075, "model_version": "v1"}
```

`prediction` es la tarifa estimada en dólares y `model_version` confirma el
artefacto usado.

Errores esperados:

| Situación | Código | `detail` de ejemplo |
|---|---|---|
| Falta una columna | `422` | `Faltan campos requeridos: distance` |
| Campo adicional | `422` | `Campos no permitidos: foo` |
| Tipo incompatible | `422` | `El campo 'distance' debe ser numérico o null.` |
| `null` categórico | `422` | `El campo 'name' debe ser texto y no admite null.` |
| Artefacto inexistente | `503` | `El modelo configurado no está disponible: ...` |

### `POST /train`

Reutiliza `train_model()` de `fase-2/train.py`. Toma `TRAIN_DATA_PATH`,
`TRAIN_TARGET` y `TRAIN_SEED` del entorno. El cuerpo es opcional; si se envía,
`target` y `seed` sobrescriben esos valores.

```bash
curl -X POST http://127.0.0.1:8000/train

curl -X POST http://127.0.0.1:8000/train \
  -H 'Content-Type: application/json' \
  -d '{"target": "price", "seed": 7}'
```

Cada entrenamiento crea una versión `api-<fecha>-<sufijo>` y guarda
`modelo.joblib`, `metrics.json` y `metadata.json` en `MODEL_DIR/<version>/`. No
sobrescribe versiones anteriores.

Evidencia real dejada por Dina al probar `POST /train` con cuerpo vacío:

```text
HTTP 201 en 7,2 s
version: api-20261122T225624-0ee7e2f8
modelo:      R² 0,9621 | MAE $1,25 | RMSE $1,86
modelo base: R² -0,0907 | MAE $7,57
```

La versión nueva quedó guardada junto a `v1`, que no se tocó. El ensayo se hizo
con `MODEL_DIR` apuntando a un directorio aparte para no agregar artefactos al
repositorio.

Códigos de respuesta:

| Situación | Código |
|---|---|
| Entrenamiento correcto | `201` |
| Falta `TRAIN_DATA_PATH` o el archivo no existe | `503` |
| CSV inválido, sin columnas requeridas o sin filas útiles | `422` |
| `TRAIN_SEED` no es entero no negativo | `503` |
| Error al escribir los artefactos | `500` |

Para publicar la versión creada hay que recrear el servicio con
`MODEL_VERSION=api-20261122T225624-0ee7e2f8`. El procedimiento de promoción y
rollback está en [`fase-4/README.md`](../fase-4/README.md).

## Registro de solicitudes

Cada `POST /predict`, incluso si FastAPI rechaza el cuerpo antes de ejecutar el
endpoint, agrega una fila a `PREDICT_LOG_PATH`. El archivo se crea con encabezado
en la primera solicitud y se abre en modo append:

```text
timestamp,http_status,model_version,prediction
```

- `timestamp`: instante UTC en ISO 8601.
- `http_status`: código HTTP (`200`, `422`, `503`, …).
- `model_version`: versión usada; vacío si no se llegó a cargar el modelo.
- `prediction`: predicción; vacío si la solicitud falló.

`PREDICT_LOG_PATH` puede apuntar al archivo o al directorio. Si apunta a un
directorio, la API escribe `predict_requests.csv` dentro de él. La imagen usa
`/app/fase-4/logs/`, que corresponde a `fase-4/logs/` del host cuando se monta el
volumen.

## Cliente de prueba: `fase-3/cliente.py`

El cliente toma una fila de `fase-2/data/ejemplo_prediccion.csv`, conserva los
tipos del esquema (números como números, categorías como texto y celdas vacías
como `null`) y envía el JSON a `POST /predict`. Usa `httpx`, ya incluido en las
dependencias.

Con la API en ejecución:

```bash
cd fase-3

# Fila 0 del CSV de ejemplo
python3 cliente.py

# Otra fila
python3 cliente.py --fila 1

# Omite distance para comprobar el rechazo 422
python3 cliente.py --sin-campo distance

# Otra URL
python3 cliente.py --url http://localhost:8000
```

Una ejecución normal imprime el JSON enviado, el código HTTP y la respuesta:

```text
[*] HTTP 200
{
  "prediction": 26.707956061611075,
  "model_version": "v1"
}

[+] Precio estimado: $26.71 (modelo v1)
```

Con `--sin-campo distance` se obtiene el rechazo esperado:

```text
[*] HTTP 422
{
  "detail": "Faltan campos requeridos: distance"
}

[!] La API rechazó la solicitud: Faltan campos requeridos: distance
```

## Ejecución local

Desde la raíz del repositorio:

```bash
cd fase-3
export MODEL_DIR="$(cd ../fase-2/models && pwd)"
export MODEL_VERSION=v1
export PREDICT_LOG_PATH="$(cd .. && pwd)/fase-4/logs/"
export TRAIN_DATA_PATH="$(cd .. && pwd)/rideshare_kaggle.csv"

uvicorn main:app --host 0.0.0.0 --port 8000
```

La documentación interactiva queda en <http://127.0.0.1:8000/docs>.

## Ejecución con Docker

El `Dockerfile` está en `fase-3/`, pero el contexto de construcción es la raíz
del repositorio porque la imagen necesita `fase-2/` y `fase-4/`:

```bash
docker build -f fase-3/Dockerfile -t uber-lyft-api:fase3-fase4 .
```

Desde la raíz del repositorio, para levantar el contenedor publicando el puerto
y montando los volúmenes:

```bash
mkdir -p fase-2/data fase-2/models fase-4/logs

docker run --rm -p 8000:8000 \
  -v "$(pwd)/fase-2/data:/app/fase-2/data" \
  -v "$(pwd)/fase-2/models:/app/fase-2/models" \
  -v "$(pwd)/fase-4/logs:/app/fase-4/logs" \
  -e MODEL_DIR=/app/fase-2/models \
  -e MODEL_VERSION=v1 \
  -e PREDICT_LOG_PATH=/app/fase-4/logs/ \
  -e TRAIN_DATA_PATH=/app/fase-2/data/rideshare_kaggle.csv \
  -e TRAIN_TARGET=price \
  -e TRAIN_SEED=42 \
  uber-lyft-api:fase3-fase4
```

| Montaje | Ruta en el contenedor | Contenido |
|---|---|---|
| `fase-2/data/` | `/app/fase-2/data/` | Dataset de entrenamiento; no se incluye en la imagen |
| `fase-2/models/` | `/app/fase-2/models/` | Versiones `v1`, `v2` y las creadas por `/train` |
| `fase-4/logs/` | `/app/fase-4/logs/` | Registro persistente de `POST /predict` |

La imagen arranca con:

```text
uvicorn main:app --host 0.0.0.0 --port 8000
```

Si el dataset no está montado, `/predict` sigue funcionando con las versiones
incluidas y `/train` responde `503` hasta que `TRAIN_DATA_PATH` exista.

La imagen ejecuta el proceso como `appuser` (UID 1000). Si el UID del host es
distinto y no se puede escribir en `fase-4/logs/`, agrega
`--user "$(id -u):$(id -g)"`.

## Pruebas

Desde la raíz del repositorio, con las dependencias instaladas:

```bash
python -m pytest -q
```

Resultado actual: **42 pruebas superadas** (7 de Fase 2, 17 de Fase 3 y 18 de
Fase 4). El `conftest.py` de esta carpeta hace importable `main` cuando pytest se
ejecuta desde la raíz.

Ejecución individual por carpeta:

```bash
cd fase-2 && python -m pytest -q   # 7 pruebas
cd ../fase-3 && python -m pytest -q # 17 pruebas
cd ../fase-4 && python -m pytest -q # 18 pruebas
```

Dentro de la imagen, desde la raíz del proyecto:

```bash
docker run --rm -w /app uber-lyft-api:fase3-fase4 python -m pytest -q
```

Las pruebas de Fase 3 cubren:

- `GET /health` y su respuesta exacta.
- Predicción con el artefacto real y versión devuelta.
- Campo faltante, campo adicional y tipos incompatibles.
- Imputación de un `null` numérico y rechazo de un `null` categórico.
- `503` cuando la versión configurada no existe.
- Registro de solicitudes exitosas y rechazadas.
- Entrenamiento con cuerpo vacío y con `target`/`seed` explícitos.
- `503` cuando falta `TRAIN_DATA_PATH`.
- Cliente: tipos del esquema, celdas vacías, fila fuera de rango, `--sin-campo`
  válido e inválido y API inalcanzable.

## Limitaciones

- El modelo se carga bajo demanda y se cachea por ruta; cambiar `MODEL_VERSION`
  requiere recrear el proceso o el contenedor.
- `surge_multiplier` es una entrada del modelo. Si no se conoce al cotizar, debe
  reentrenarse sin esa columna; la decisión viene de la Fase 2.
- Las predicciones son cotizaciones, no precios pagados.
- La API no implementa autenticación ni límites de tasa; el alcance es exponer el
  modelo y registrar las solicitudes.
- El registro es un CSV con escritura en modo append y un bloqueo en memoria. Es
  suficiente para una instancia; varias réplicas requerirían un recolector
  centralizado.
