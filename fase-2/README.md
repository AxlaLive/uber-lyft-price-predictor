# Fase 2: predicción de tarifas de Uber y Lyft como aplicación reutilizable

**Proyecto Integrador · Modelos y Simulación de Sistemas I · Universidad de Antioquia**

Integrantes: Alexandra Garavito Vargas, Juan Camilo Mona Luján y Dina Marcela
Reales Corredor.

## Problema y relación con la Fase 1

Se predice el precio de una cotización de viaje de Uber o Lyft en Boston en el
instante en que el usuario la solicita. Es un problema de regresión y la variable
objetivo es `price`.

La Fase 1 dejó el modelo construido en un notebook: un `Pipeline` de
`HistGradientBoostingRegressor` con imputación por mediana y One-Hot Encoding,
ajustado solo con los datos de entrenamiento. Esta fase convierte ese trabajo en
dos scripts ejecutables por línea de comandos, `train.py` y `predict.py`, con
versionamiento de los artefactos, pruebas automatizadas y ejecución en Docker. Se
conservan el problema, el conjunto de datos y el algoritmo.

`train.py` entrena el pipeline con una lista explícita de 13 columnas de entrada y
guarda el modelo, las métricas y los metadatos en una carpeta por versión. No
sobrescribe una versión que ya exista.

## Correcciones respecto a la Entrega 1

La calificación de la Fase 1 fue 87/100 y señaló varios puntos. Estos son los
cambios aplicados en esta entrega:

| Observación de la Fase 1 | Corrección |
|---|---|
| El código generaba 31 columnas de entrada y 24 numéricas, frente a 29 y 22 en el artefacto guardado | Se fijó una lista explícita de 13 variables, compartida por `train.py`, `predict.py` y el notebook |
| El artefacto guardaba `columnas_entrada` y la celda de carga esperaba `columnas_entrada_raw`, por lo que fallaba | Se unificó la clave en `columnas_entrada` y se regeneraron los artefactos en una sola ejecución |
| Faltaba el enlace de descarga y la ubicación del CSV | Documentados en este README y en el de `fase-1` |
| No se guardaban las métricas del modelo base | `metrics.json` incluye ahora el `DummyRegressor(median)` evaluado sobre la misma partición |
| Metadatos incompletos | `metadata.json` añade tipos de entrada, parámetros del modelo, fuente y tamaño de los datos, estrategia de separación, métrica, referencia a dependencias y hash del commit |
| Afirmación incorrecta de que se retiraron las redundancias | Corregida en la sección de limitaciones: `temperature`/`apparentTemperature` y `windSpeed`/`windGust` siguen en el modelo, a propósito |
| Pregunta sobre cómo obtendrá la API el `surge_multiplier` | Respondida en la sección de limitaciones |

Queda pendiente del lado del notebook separar los experimentos históricos, que
evalúan en test, del flujo definitivo por validación cruzada.

## Entorno y dependencias

Se usa **Python 3.9**, la misma versión de la imagen Docker (`python:3.9-slim`).
Desde la carpeta `fase-2`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

`requirements.txt` fija pandas 2.3.3, numpy 2.0.2, scikit-learn 1.6.1, joblib
1.5.3 y pytest 8.4.2. La versión de scikit-learn debe coincidir con la de los
artefactos guardados, que queda registrada en cada `metadata.json`.

Para consultar las opciones de cada script:

```bash
python3 train.py --help
python3 predict.py --help
```

## Origen y ubicación de los datos

El dataset es **Uber & Lyft Cab prices (Boston, MA)** de Kaggle:
<https://www.kaggle.com/datasets/brllrb/uber-and-lyft-dataset-boston-ma>

Pesa unos 350 MB descomprimido, así que no se almacena en Git. Para obtenerlo:

1. Descarga `uber-and-lyft-dataset-boston-ma.zip` desde el enlace anterior
   (requiere una cuenta de Kaggle).
2. Descomprímelo y deja `rideshare_kaggle.csv` en la raíz del repositorio, o en
   `fase-2/data/rideshare_kaggle.csv` si vas a entrenar dentro de Docker.

Los comandos de esta guía asumen la primera ubicación, `../rideshare_kaggle.csv`.
El archivo trae 693.071 filas y 57 columnas; el entrenamiento usa 13 de ellas y
descarta las 55.095 filas sin `price`, que corresponden a la categoría Taxi.

## Entrenamiento

Desde esta carpeta, ejecuta:

```bash
python3 train.py --data ../rideshare_kaggle.csv --target price --output-dir models --version v1 --seed 42
```

Para generar una segunda versión independiente:

```bash
python3 train.py --data ../rideshare_kaggle.csv --target price --output-dir models --version v2 --seed 123
```

Cada versión contiene `modelo.joblib`, `metrics.json` y `metadata.json`.

## Métricas y comparación con el modelo base

`metrics.json` guarda las métricas del modelo y del modelo base sobre la **misma
partición de prueba**, para conservar la comparación de la Fase 1. El modelo base
es `DummyRegressor(strategy="median")`, que predice siempre la mediana del
entrenamiento.

| Versión | Modelo R² | Modelo MAE | Base R² | Base MAE |
|---|---|---|---|---|
| v1 (semilla 42) | 0,9648 | $1,16 | −0,1062 | $7,37 |
| v2 (semilla 123) | 0,9648 | $1,16 | −0,1074 | $7,37 |

La métrica principal es R². El modelo base obtiene R² negativo, es decir, es peor
que predecir la media, mientras que el modelo reduce el MAE de $7,37 a $1,16.

`metadata.json` registra, por versión: identificador y fecha, tipo de problema,
variable objetivo, algoritmo, semilla, nombres y tipos de las variables de
entrada, parámetros del regresor, preprocesamiento, fuente y tamaño de los datos,
estrategia de separación, métrica de evaluación, versión de Python y de
scikit-learn, referencia a `requirements.txt` y el hash del commit que la generó.

## Limitaciones conocidas

- **`surge_multiplier` puede no estar disponible al cotizar.** Es una de las 13
  variables de entrada y correlaciona 0,24 con el precio. En el dataset viene
  junto a la cotización, pero si en producción ese factor se conoce solo al
  confirmar el viaje, pasaría a ser información posterior al evento y habría que
  retirarlo del esquema. Antes de exponer la API en la Fase 3 hay que decidir de
  dónde saldrá ese valor: si lo entrega el cliente de la API, el modelo sirve tal
  cual; si no, debe reentrenarse sin esa columna.
- **El esquema conserva pares de variables correlacionadas**,
  `temperature`/`apparentTemperature` y `windSpeed`/`windGust`. Se mantienen a
  propósito, porque un modelo de árboles no se ve afectado por la redundancia; no
  es cierto que se hayan retirado.
- **Las predicciones son cotizaciones, no precios pagados.** El dataset recoge
  estimaciones de la aplicación, no transacciones liquidadas.
- **El dataset no tiene valores faltantes en las variables predictoras**; los
  55.095 nulos están en `price`, la variable objetivo.

## Inferencia

El archivo `data/ejemplo_prediccion.csv` contiene tres filas sin la variable
objetivo: un Uber de recorrido corto, un Lyft de más de 5 millas y un viaje con
`surge_multiplier` de 1,25. Sus columnas son exactamente las 13 del esquema de
entrenamiento:

```
distance, surge_multiplier, latitude, longitude, temperature,
apparentTemperature, humidity, windSpeed, windGust, visibility, pressure,
name, cab_type
```

`predict.py` reordena las columnas según el esquema guardado, así que el orden del
CSV no importa. Las columnas adicionales se conservan en la salida sin usarse para
predecir, lo que permite incluir un identificador propio de cada fila. Las
categorías de `name` o `cab_type` que no se vieron en el entrenamiento se codifican
en ceros, por `handle_unknown="ignore"`, en lugar de producir un error. Una celda
vacía en una variable numérica se imputa con la mediana aprendida en el
entrenamiento; en cambio, la ausencia de una columna requerida termina con código
de error.

Desde esta carpeta, genera las predicciones con:

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
