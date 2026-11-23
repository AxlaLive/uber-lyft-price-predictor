# Fase 4: monitoreo, reentrenamiento y rollback

**Proyecto Integrador · Modelos y Simulación de Sistemas I · Universidad de Antioquia**

Integrantes: Alexandra Garavito Vargas, Juan Camilo Mona Luján y Dina Marcela
Reales Corredor.

## Propósito

La Fase 3 dejó una API que predice tarifas y registra cada solicitud. Esta fase
cierra el ciclo operativo: calcula métricas de operación, documenta cuándo
reentrenar y explica cómo publicar una versión nueva o volver a la anterior.

## Estructura

```text
fase-4/
├── metricas.py              # reporte de solicitudes y porcentaje de fallidas
├── logs/
│   ├── muestra_ensayo.csv   # evidencia real del ensayo del PR #15
│   └── predict_requests.csv # generado por la API; ignorado por Git
├── tests/
│   └── test_metricas.py     # 18 pruebas del reporte
└── README.md
```

Las predicciones se registran desde `fase-3/main.py` mediante el middleware de
`POST /predict` y `_append_prediction_event()`. El registro tiene las columnas
`timestamp,http_status,model_version,prediction`. `fase-4/metricas.py` lee ese
archivo y calcula el total de solicitudes y el porcentaje de fallidas (HTTP
4xx o 5xx).

## Variables de entorno

| Variable | Valor por defecto | Uso |
|---|---|---|
| `PREDICT_LOG_PATH` | `fase-4/logs/predict_requests.csv` en el código; `/app/fase-4/logs/` en la imagen | Archivo o directorio del registro. Si es directorio, escribe `predict_requests.csv` dentro. |
| `MODEL_VERSION` | `v1` | Versión activa. Puede ser `v1`, `v2` o una versión `api-...`. |
| `MODEL_DIR` | `<raíz>/fase-2/models`; `/app/fase-2/models` en la imagen | Directorio base de las versiones. |
| `TRAIN_DATA_PATH` | Sin valor en el código; `/app/fase-2/data/rideshare_kaggle.csv` en la imagen | CSV para `POST /train`. |
| `TRAIN_TARGET` | `price` | Columna objetivo del reentrenamiento. |
| `TRAIN_SEED` | `42` | Semilla del reentrenamiento. |

La convención de la entrega es dejar los logs en `fase-4/logs/`. Con el volumen
montado, el archivo persistente del host es:

```text
fase-4/logs/predict_requests.csv
```

## Inicio de la API con registro habilitado

Construir la imagen desde la raíz del repositorio:

```bash
docker build -f fase-3/Dockerfile -t uber-lyft-api:fase3-fase4 .
```

Desde la raíz del repositorio, levantar el contenedor publicando el puerto y
habilitando el registro:

```bash
mkdir -p fase-2/data fase-2/models fase-4/logs

docker run -d --name uber-lyft-api -p 8000:8000 \
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

Comprobar el arranque:

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok","message":"API en ejecución"}
```

Cada `POST /predict` agrega una fila a
`/app/fase-4/logs/predict_requests.csv`, que corresponde a
`fase-4/logs/predict_requests.csv` en el host. Si el directorio no existe, la API
lo crea.

## Script de monitoreo: `fase-4/metricas.py`

Uso:

```bash
cd fase-4
python3 metricas.py
python3 metricas.py --registro logs/predict_requests.csv --detalle
```

Desde la raíz:

```bash
python3 fase-4/metricas.py \
  --registro fase-4/logs/predict_requests.csv --detalle
```

El reporte informa:

- **Total de solicitudes:** filas de datos del registro.
- **Solicitudes fallidas:** respuestas HTTP entre `400` y `599`.
- **Porcentaje de fallidas:** `fallidas / total * 100`, con un decimal.
- **`--detalle`:** desglose por código HTTP y por versión del modelo.

Características:

- Lee **CSV y JSONL**. El formato se deduce del contenido, no de la extensión.
- Con un registro vacío informa `Total de solicitudes: 0` y
  `Porcentaje de fallidas: sin solicitudes`, sin dividir por cero.
- Si el registro no existe o una línea JSONL es inválida, termina con código `1`
  y explica el error; no produce un reporte vacío aparentemente válido.

## Reporte

El ensayo fue ejecutado por Dina y su evidencia está en
[`fase-4/logs/muestra_ensayo.csv`](logs/muestra_ensayo.csv). Se enviaron dos
solicitudes válidas y una tercera sin la columna `distance`, con la API en
ejecución (`MODEL_VERSION=v1`) y un registro vacío.

Contenido del registro:

```text
timestamp,http_status,model_version,prediction
2026-11-23T03:55:40.580702+00:00,200,v1,26.707956061611075
2026-11-23T03:55:41.099661+00:00,200,v1,9.584435594221473
2026-11-23T03:55:41.585510+00:00,422,,
```

Salida del script:

```text
Total de solicitudes: 3
Solicitudes fallidas (HTTP 4xx o 5xx): 1
Porcentaje de fallidas: 33.3 %
```

Interpretación: la API atendió dos registros con `v1`, que estimaron $26,71 y
$9,58, y rechazó el tercero con HTTP 422 y el mensaje «Faltan campos requeridos:
distance». La tasa de fallos fue 1/3 (33,3 %). El rechazo era el resultado
esperado de la prueba de validación, no una falla del servicio.

Datos reales de `POST /train` reportados en el PR #15:

```text
HTTP 201 en 7,2 s
version: api-20261122T225624-0ee7e2f8
modelo:      R² 0,9621 | MAE $1,25 | RMSE $1,86
modelo base: R² -0,0907 | MAE $7,57
```

La versión `api-20261122T225624-0ee7e2f8` quedó guardada junto a `v1`, que
permaneció intacta. El ensayo se ejecutó con `MODEL_DIR` apuntando a un directorio
aparte para no agregar artefactos al repositorio.

`muestra_ensayo.csv` es evidencia versionada y no debe sobrescribirse ni
borrarse. Para nuevas pruebas de Docker conviene usar una ruta temporal distinta
en `PREDICT_LOG_PATH`.

## Versiones de referencia

| Versión | Origen | R² | MAE | RMSE |
|---|---|---|---|---|
| `v1` | Fase 2, semilla 42 | 0,9617 | $1,25 | $1,87 |
| `v2` | Fase 2, semilla 123 | 0,9622 | $1,25 | $1,85 |
| `api-20261122T225624-0ee7e2f8` | `POST /train` del PR #15, cuerpo vacío | 0,9621 | $1,25 | $1,86 |

El modelo base de la versión `api-20261122T225624-0ee7e2f8` obtuvo R² −0,0907 y
MAE $7,57. Las tres versiones viven en carpetas separadas y ninguna sobrescribe
a las anteriores.

## Política de reentrenamiento

El modelo se revisa cuando se acumulan datos reales etiquetados nuevos o cuando
el reporte de monitoreo muestra una degradación sostenida; `/train` lo ejecuta
quien esté a cargo del despliegue, usando `TRAIN_DATA_PATH`, `TRAIN_TARGET` y
`TRAIN_SEED`. La versión nueva solo se activa si, evaluada sobre datos reservados,
iguala o mejora la métrica principal de la Fase 2 (R²) y no empeora el MAE frente
a la versión activa. Si no cumple, se conserva la versión anterior y se documenta
el resultado.

## Rollback: volver a la versión anterior

1. **Identificar la versión anterior.** Las versiones están en
   `fase-2/models/`. Revisa sus métricas antes de cambiar:

   ```bash
   ls -1 fase-2/models
   cat fase-2/models/v1/metrics.json
   cat fase-2/models/v2/metrics.json
   ```

2. **Detener y eliminar el contenedor actual.** No basta con `docker restart`,
   porque las variables de entorno se fijan al crear el contenedor:

   ```bash
   docker stop uber-lyft-api
   docker rm uber-lyft-api
   ```

3. **Recrear el contenedor con la versión anterior.** Por ejemplo, para volver de
   `api-20261122T225624-0ee7e2f8` a `v1`:

   ```bash
   docker run -d --name uber-lyft-api -p 8000:8000 \
     -v "$(pwd)/fase-2/data:/app/fase-2/data" \
     -v "$(pwd)/fase-2/models:/app/fase-2/models" \
     -v "$(pwd)/fase-4/logs:/app/fase-4/logs" \
     -e MODEL_DIR=/app/fase-2/models \
     -e MODEL_VERSION=v1 \
     -e PREDICT_LOG_PATH=/app/fase-4/logs/ \
     uber-lyft-api:fase3-fase4
   ```

   Para volver a `v2`, cambia `MODEL_VERSION=v2`. Si usas Docker Compose, cambia
   la variable en el archivo y ejecuta `docker compose up -d`.

4. **Validar la salud y el modelo.** La respuesta de `POST /predict` debe incluir
   la versión esperada:

   ```bash
   curl http://127.0.0.1:8000/health
   curl -X POST http://127.0.0.1:8000/predict \
     -H 'Content-Type: application/json' \
     -d @fase-3/ejemplo_payload.json
   ```

   Con `v1`, la fila de ejemplo devuelve `prediction` aproximadamente `26.708` y
   `"model_version": "v1"`.

5. **Comprobar el registro y el reporte.** Las filas nuevas deben mostrar la
   versión restaurada:

   ```bash
   tail -n 5 fase-4/logs/predict_requests.csv
   python3 fase-4/metricas.py \
     --registro fase-4/logs/predict_requests.csv --detalle
   ```

   Conserva la versión fallida para el análisis posterior; no borres su carpeta.

## Pruebas

Desde la raíz del repositorio:

```bash
python -m pytest -q
```

Resultado actual: **42 pruebas superadas** (7 de Fase 2, 17 de Fase 3 y 18 de
Fase 4). El `conftest.py` de `fase-3/` permite que pytest importe `main` al
ejecutarse desde la raíz.

Ejecución individual:

```bash
cd fase-2 && python -m pytest -q   # 7 pruebas
cd ../fase-3 && python -m pytest -q # 17 pruebas
cd ../fase-4 && python -m pytest -q # 18 pruebas
```

Dentro de la imagen:

```bash
docker run --rm -w /app uber-lyft-api:fase3-fase4 python -m pytest -q
```

Las pruebas de Fase 4 cubren registro vacío, dos válidas y una inválida
(`33.3 %`), límites de los códigos HTTP, todas fallidas (`100.0 %`), `--detalle`,
registro inexistente, JSONL equivalente al CSV, JSONL vacío y JSONL con una línea
corrupta.

## Limitaciones

- El monitoreo es descriptivo: cuenta solicitudes y fallidas, pero no calcula
  MAE en producción porque el precio real no está disponible al cotizar.
- El porcentaje de fallidas mezcla errores del cliente (`422`) con errores del
  servicio (`5xx`); para decisiones operativas conviene revisar `--detalle`.
- El registro CSV con bloqueo en memoria sirve para una instancia; varias
  réplicas necesitarían un recolector centralizado.
- El dataset real y los logs de producción no se versionan.
