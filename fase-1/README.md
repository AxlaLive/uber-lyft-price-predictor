# Fase 1: Modelo predictivo

Esta carpeta contiene la primera fase del proyecto de predicción del precio de viajes de rideshare.

## Contenido

- `fase_1_modelo.ipynb`: exploración, preparación de datos, entrenamiento y evaluación del modelo.
- `modelo.joblib`: pipeline entrenado y guardado para realizar predicciones.
- `requirements.txt`: dependencias del proyecto. La versión de scikit-learn está fijada para mantener la compatibilidad con el modelo guardado.

El dataset utilizado es `../rideshare_kaggle.csv`.

## Qué incluye el notebook

1. Exploración y limpieza de los datos.
2. Separación de los datos en entrenamiento y prueba.
3. Comparación de modelos mediante validación cruzada.
4. Preprocesamiento y modelo encapsulados en un `Pipeline`.
5. Evaluación con RMSE, MAE y R².
6. Verificación para reducir el riesgo de fuga de información.
7. Guardado y carga del artefacto `modelo.joblib`.

## Instalación

Desde la carpeta `fase-1`:

```bash
python3 -m pip install -r requirements.txt
```

La versión de scikit-learn debe ser `1.4.2`, que es la versión con la que se generó `modelo.joblib`. Usar otra versión puede producir errores o resultados incompatibles al cargar el archivo.

## Ejecución

Abre `fase_1_modelo.ipynb` desde la carpeta `fase-1` y selecciona el entorno creado con las dependencias anteriores. El notebook utiliza la ruta relativa `../rideshare_kaggle.csv`.

El modelo ya está almacenado en `modelo.joblib`, por lo que no es necesario volver a entrenarlo para entregar o revisar el artefacto. El reentrenamiento completo puede tardar debido al tamaño del dataset y a la validación de modelos.

## Uso del modelo guardado

El archivo contiene el pipeline entrenado junto con metadatos de las columnas de entrada y las métricas de evaluación. Para cargarlo:

```python
import joblib

artefacto = joblib.load("modelo.joblib")
pipeline = artefacto["pipeline"]
predicciones = pipeline.predict(nuevos_datos)
```