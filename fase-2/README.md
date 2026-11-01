# Fase 2: entrenamiento versionado

`train.py` entrena el pipeline de HistGradientBoosting con una lista explícita de
13 columnas de entrada y guarda el modelo, las métricas y los metadatos en una
carpeta por versión. No sobrescribe una versión que ya exista.

El dataset se puede colocar en `data/rideshare_kaggle.csv`. También se puede
reutilizar el CSV de la raíz del repositorio sin duplicar sus 350 MB.

Desde esta carpeta, ejecuta:

```bash
python3 train.py --data ../rideshare_kaggle.csv --target price --output-dir models --version v1 --seed 42
```

Para generar una segunda versión independiente:

```bash
python3 train.py --data ../rideshare_kaggle.csv --target price --output-dir models --version v2 --seed 123
```

Cada versión contiene `modelo.joblib`, `metrics.json` y `metadata.json`.

## Inferencia

El archivo `data/ejemplo_prediccion.csv` contiene tres filas sin la variable
objetivo. Desde esta carpeta, genera las predicciones con:

```bash
python3 predict.py --model models/v1/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv
```

El CSV de salida conserva las columnas y el orden de las filas de entrada y agrega
la columna `prediccion`. Si falta una columna requerida o no se puede leer el
modelo, la entrada o la salida, el script termina con código de error `1`.

## Pruebas automatizadas

Desde esta carpeta, ejecuta:

```bash
python3 -m pytest
```

## Contenedor Docker

Construye la imagen desde `fase-2/`:

```bash
docker build -t uber-lyft-mlops:v2 .
```

La imagen excluye los datasets, modelos y resultados locales. El CSV pequeño de
ejemplo sí se incluye. El build solo instala dependencias; no entrena el modelo.
Por defecto, al iniciar el contenedor se ejecutan las pruebas:

```bash
docker run --rm uber-lyft-mlops:v2
```

Para entrenar, coloca el dataset en `fase-2/data/rideshare_kaggle.csv` y monta los
directorios locales. `v1` y `v2` ya están creadas en este checkout, así que usa
una versión nueva, por ejemplo `v3`:

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  uber-lyft-mlops:v2 \
  python train.py --data data/rideshare_kaggle.csv --target price --output-dir models --version v3 --seed 42
```

Para ejecutar inferencia y conservar la salida en el host:

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/outputs:/app/outputs" \
  uber-lyft-mlops:v2 \
  python predict.py --model models/v1/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones_docker.csv
```
