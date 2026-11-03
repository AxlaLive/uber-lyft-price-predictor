"""
Script de entrenamiento (train.py) - Fase 2
Proyecto Integrador - Modelos y Simulación de Sistemas I
"""

import argparse
import json
import re
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


COLS_NUM = [
    "distance",
    "surge_multiplier",
    "latitude",
    "longitude",
    "temperature",
    "apparentTemperature",
    "humidity",
    "windSpeed",
    "windGust",
    "visibility",
    "pressure",
]
COLS_CAT = ["name", "cab_type"]
VERSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
FUENTE_DATOS = (
    "Kaggle: Uber & Lyft Cab prices, Boston MA "
    "(https://www.kaggle.com/datasets/brllrb/uber-and-lyft-dataset-boston-ma)"
)
METRICA_PRINCIPAL = "r2"


def calcular_metricas(y_true, y_pred) -> dict:
    """RMSE, MAE y R² sobre la misma partición de prueba."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def tipos_esperados() -> dict:
    """Tipo de dato que predict.py espera por columna de entrada."""
    tipos = {columna: "numerico" for columna in COLS_NUM}
    tipos.update({columna: "categorico" for columna in COLS_CAT})
    return tipos


def _git(*argumentos: str) -> str:
    """Ejecuta un comando de Git en la carpeta del script; cadena vacía si falla."""
    try:
        resultado = subprocess.run(
            ["git", *argumentos],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return resultado.stdout.strip()


def referencia_codigo() -> dict:
    """Identifica la versión del código que generó los artefactos.

    'commit' es el HEAD en el momento de ejecutar el entrenamiento. Como los
    artefactos se confirman después, ese hash corresponde al código que los
    produjo y no al commit que los contiene. 'estado_arbol' advierte si había
    cambios sin confirmar, caso en el que el hash no describe el código exacto.
    """
    commit = _git("rev-parse", "HEAD") or "desconocido"
    rama = _git("rev-parse", "--abbrev-ref", "HEAD") or "desconocida"
    estado = "con cambios sin confirmar" if _git("status", "--porcelain") else "limpio"
    return {
        "commit": commit,
        "rama": rama,
        "estado_arbol": estado,
        "nota": (
            "HEAD en el momento de entrenar. Los artefactos se confirman en un "
            "commit posterior, así que este hash identifica el código que los "
            "generó, no el commit que los almacena."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena HistGradientBoosting para predecir tarifas."
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Ruta al archivo CSV de datos etiquetados.",
    )
    parser.add_argument(
        "--target",
        default="price",
        help="Nombre de la variable objetivo (por defecto: price).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directorio base para las versiones del modelo (por defecto: models).",
    )
    parser.add_argument(
        "--version",
        default="v1",
        help="Identificador nuevo de la versión (por defecto: v1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria para reproducibilidad (por defecto: 42).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not VERSION_PATTERN.fullmatch(args.version) or args.version in {".", ".."}:
        raise ValueError(
            "La versión solo puede contener letras, números, puntos, guiones "
            "y guiones bajos, y debe comenzar con letra o número."
        )

    if not args.data.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {args.data}")

    print(f"[*] Cargando datos desde: {args.data}")
    df = pd.read_csv(args.data)

    if args.target not in df.columns:
        raise ValueError(
            f"La variable objetivo '{args.target}' no se encuentra en el dataset."
        )

    required_columns = COLS_NUM + COLS_CAT + [args.target]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Faltan las siguientes columnas requeridas en el dataset: "
            f"{missing_columns}"
        )

    n_original = len(df)
    df = df.dropna(subset=[args.target])
    n_con_objetivo = len(df)

    # La deduplicación se hace sobre las columnas que el modelo realmente usa más
    # la variable objetivo. Aplicarla sobre todas las columnas del CSV no elimina
    # ninguna fila, porque 'id', 'timestamp' y 'datetime' vuelven única cada una:
    # el resultado era que observaciones idénticas para el modelo cayeran a ambos
    # lados de la partición y la evaluación usara filas ya vistas al entrenar.
    df = df.drop_duplicates(subset=required_columns)
    n_duplicadas = n_con_objetivo - len(df)
    if df.empty:
        raise ValueError("No hay filas disponibles después de limpiar el dataset.")

    print(
        f"[*] Filas: {n_original} originales -> {n_con_objetivo} con '{args.target}' "
        f"-> {len(df)} tras descartar {n_duplicadas} duplicadas"
    )

    X = df[COLS_NUM + COLS_CAT].copy()
    y = df[args.target].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), COLS_NUM),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                COLS_CAT,
            ),
        ],
        remainder="drop",
    )
    model = Pipeline(
        steps=[
            ("preprocesamiento", preprocessor),
            (
                "regresor",
                HistGradientBoostingRegressor(random_state=args.seed),
            ),
        ]
    )

    print("[*] Entrenando HistGradientBoosting...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metricas_modelo = calcular_metricas(y_test, predictions)

    # Modelo base de la Fase 1: predice siempre la mediana de entrenamiento.
    # Se evalúa sobre la MISMA partición y con las mismas métricas para conservar
    # la comparación exigida por la especificación.
    print("[*] Entrenando el modelo base (mediana)...")
    baseline = DummyRegressor(strategy="median").fit(X_train, y_train)
    metricas_base = calcular_metricas(y_test, baseline.predict(X_test))

    metrics = {
        "metrica_principal": METRICA_PRINCIPAL,
        "modelo": metricas_modelo,
        "modelo_base": {"estrategia": "median", **metricas_base},
    }
    print(
        "[+] Modelo     -> "
        f"R²: {metricas_modelo['r2']:.4f} | "
        f"MAE: ${metricas_modelo['mae']:.2f} | "
        f"RMSE: ${metricas_modelo['rmse']:.2f}"
    )
    print(
        "[+] Modelo base -> "
        f"R²: {metricas_base['r2']:.4f} | "
        f"MAE: ${metricas_base['mae']:.2f} | "
        f"RMSE: ${metricas_base['rmse']:.2f}"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    version_dir = args.output_dir / args.version
    try:
        version_dir.mkdir()
    except FileExistsError as exc:
        raise FileExistsError(
            f"La versión '{args.version}' ya existe en '{version_dir}'. "
            "No se sobrescribirá."
        ) from exc

    artifact = {
        "pipeline": model,
        "columnas_entrada": COLS_NUM + COLS_CAT,
        "cols_num": COLS_NUM,
        "cols_cat": COLS_CAT,
        "version_sklearn": sklearn.__version__,
    }
    metadata = {
        "version": args.version,
        "fecha_creacion": datetime.now().astimezone().isoformat(),
        "tipo_problema": "Regresion",
        "variable_objetivo": args.target,
        "algoritmo": "HistGradientBoostingRegressor",
        "semilla": args.seed,
        "n_entrenamiento": len(X_train),
        "n_prueba": len(X_test),
        "python_version": platform.python_version(),
        "version_sklearn": sklearn.__version__,
        "columnas_entrada": COLS_NUM + COLS_CAT,
        "tipos_entrada": tipos_esperados(),
        "parametros_modelo": model.named_steps["regresor"].get_params(),
        "preprocesamiento": {
            "numericas": "SimpleImputer(strategy='median')",
            "categoricas": "OneHotEncoder(handle_unknown='ignore')",
            "ajustado_con": "solo X_train",
        },
        "datos": {
            "fuente": FUENTE_DATOS,
            "archivo": args.data.name,
            "n_observaciones_originales": int(n_original),
            "filas_sin_objetivo_descartadas": int(n_original - n_con_objetivo),
            "filas_duplicadas_descartadas": int(n_duplicadas),
            "deduplicado_por": required_columns,
            "n_observaciones_utilizadas": int(len(df)),
        },
        "separacion": {
            "metodo": "train_test_split aleatorio",
            "test_size": 0.2,
            "metrica_evaluacion": METRICA_PRINCIPAL,
            "modelo_base": "DummyRegressor(strategy='median')",
        },
        "dependencias": "requirements.txt",
        "codigo": referencia_codigo(),
    }

    joblib.dump(artifact, version_dir / "modelo.joblib")
    (version_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (version_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[+] Artefactos guardados en: {version_dir}/")


if __name__ == "__main__":
    main()
