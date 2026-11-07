# Fase 1: Modelo predictivo de tarifas de Uber y Lyft

**Proyecto Integrador · Modelos y Simulación de Sistemas I · Universidad de Antioquia**

Integrantes: Alexandra Garavito Vargas, Juan Camilo Mona Luján y Dina Marcela Reales Corredor.

Esta carpeta contiene la primera fase del proyecto de predicción del precio de viajes de rideshare.

## Problema y objetivo del modelo

El modelo estima el precio de una cotización de viaje de Uber o Lyft en Boston en el instante en que el usuario la solicita. Es un problema de **regresión** y la variable objetivo es `price`, en dólares.

Importante: el dataset recoge **cotizaciones mostradas por la aplicación**, no transacciones efectivamente pagadas. Por eso la predicción debe entenderse como una estimación de tarifa, no como el valor de un viaje liquidado.

La fuente es el dataset de Kaggle **Uber & Lyft Cab prices (Boston, MA)**: <https://www.kaggle.com/datasets/brllrb/uber-and-lyft-dataset-boston-ma>. Trae 693.071 registros y 57 columnas. Tras descartar las 55.095 filas sin `price`, que corresponden a la categoría Taxi, y eliminar duplicados exactos, quedan **582.066 observaciones**.

> **Sobre el conteo de observaciones.** Las fases aplican criterios de deduplicación distintos. Este notebook elimina duplicados exactos del conjunto de columnas que queda tras la auditoría de variables y conserva 582.066 filas. `train.py`, en cambio, considera duplicadas las filas que coinciden en las 13 variables de entrada del modelo y `price`, aunque difieran en otras columnas que no usa, y conserva 549.313. Por eso las cifras difieren; cada una corresponde al criterio documentado para esa fase.

## Contenido

- `fase_1_modelo.ipynb`: exploración, preparación de datos, entrenamiento y evaluación del modelo.
- `../modelos/modelo_hgb_v1.joblib`: pipeline entrenado y metadatos para realizar predicciones.
- `requirements.txt`: dependencias del proyecto. La versión de scikit-learn está fijada para mantener la compatibilidad con el modelo guardado.

El dataset utilizado es `../rideshare_kaggle.csv`, que no se almacena en Git por su
tamaño (unos 350 MB). Se descarga de Kaggle, **Uber & Lyft Cab prices (Boston,
MA)**: <https://www.kaggle.com/datasets/brllrb/uber-and-lyft-dataset-boston-ma>.
Descomprime `uber-and-lyft-dataset-boston-ma.zip` y deja `rideshare_kaggle.csv` en
la raíz del repositorio, al lado de las carpetas `fase-1` y `fase-2`.

## Qué incluye el notebook

1. Exploración y limpieza de los datos.
2. Separación de los datos en entrenamiento y prueba.
3. Comparación de modelos mediante validación cruzada.
4. Preprocesamiento y modelo encapsulados en un `Pipeline`.
5. Evaluación con RMSE, MAE y R².
6. Verificación para reducir el riesgo de fuga de información.
7. Guardado y carga del artefacto `../modelos/modelo_hgb_v1.joblib`.

## Instalación

Desde la carpeta `fase-1`:

```bash
python3 -m pip install -r requirements.txt
```

La versión de scikit-learn debe ser `1.4.2`, que es la versión con la que se generó `modelo.joblib`. Usar otra versión puede producir errores o resultados incompatibles al cargar el archivo.

## Ejecución

Abre `fase_1_modelo.ipynb` desde la carpeta `fase-1` y selecciona el entorno creado con las dependencias anteriores. El notebook utiliza la ruta relativa `../rideshare_kaggle.csv`.

El artefacto `../modelos/modelo_hgb_v1.joblib` se genera al ejecutar las celdas de entrenamiento y guardado. El reentrenamiento completo puede tardar debido al tamaño del dataset y a la validación de modelos.

## Uso del modelo guardado

El archivo contiene el pipeline entrenado junto con metadatos de las columnas de entrada (`columnas_entrada`) y las métricas de evaluación. Para cargarlo:

```python
import joblib

artefacto = joblib.load("../modelos/modelo_hgb_v1.joblib")
pipeline = artefacto["pipeline"]
predicciones = pipeline.predict(nuevos_datos)
```

## Resultados obtenidos

La métrica principal es **R²**, acompañada de MAE y RMSE para leer el error en dólares. La selección del modelo se hizo con **validación cruzada de 5 pliegues sobre el conjunto de entrenamiento**, sin consultar el conjunto de prueba:

| Modelo | R² | MAE | RMSE |
|---|---|---|---|
| Dummy (mediana) — modelo base | −0,0857 | $7,49 | $9,84 |
| Regresión lineal | 0,9254 | $1,82 | $2,58 |
| **HistGradientBoosting** (elegido) | **0,9627** | **$1,22** | **$1,82** |

Sobre el conjunto de prueba, usado una sola vez para evaluar el modelo ya elegido: **R² 0,9628, MAE $1,22 y RMSE $1,81**. El resultado coincide con la estimación de la validación cruzada, lo que indica que la métrica no está inflada.

El modelo base obtiene R² negativo, es decir, es peor que predecir la media. La mejora del modelo frente a él es de $6,27 en el MAE.

Como control adicional se reentrenó el mismo pipeline con separaciones más exigentes:

| Separación | R² | MAE |
|---|---|---|
| Aleatoria 80/20 (la utilizada) | 0,9628 | $1,22 |
| Temporal (pasado → futuro) | 0,9626 | $1,22 |
| Por rutas (72 rutas, las de prueba nunca vistas) | 0,9524 | $1,36 |

## Limitaciones conocidas

- **El poder predictivo viene del tarifario, no de la dinámica del precio.** Con las variables numéricas solas el R² queda alrededor de 0,18; el salto a 0,96 lo produce `name`, el tipo de servicio. El modelo aprende la tabla de tarifas de Uber y Lyft más que el comportamiento del precio.
- **`surge_multiplier` puede no estar disponible al momento de cotizar.** Si en producción ese factor se conoce solo al confirmar el viaje, sería información posterior al evento y habría que retirarlo. Queda por decidir antes de exponer la API.
- **El esquema conserva pares de variables correlacionadas**, `temperature`/`apparentTemperature` y `windSpeed`/`windGust`. Se mantienen a propósito, porque un modelo de árboles no se ve afectado por la redundancia; no es cierto que se hayan retirado.
- **No hay valores faltantes en las variables predictoras.** Los 55.095 nulos del dataset están en `price`, la variable objetivo, de modo que el conjunto de datos no cumple el requisito de 0,1 % a 2 % de faltantes en una predictora.
- **La cobertura temporal es corta**, del 26 de noviembre al 18 de diciembre de 2018, y geográfica, limitada a Boston con 72 rutas. No cubre estacionalidad anual ni otras ciudades.
