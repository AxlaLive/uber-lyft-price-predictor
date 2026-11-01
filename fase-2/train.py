"""
Script de entrenamiento (train.py) - Fase 2
Proyecto Integrador - Modelos y Simulación de Sistemas I
"""

import argparse
import json
import re
import platform
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
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

    df = df.dropna(subset=[args.target]).drop_duplicates()
    if df.empty:
        raise ValueError("No hay filas disponibles después de limpiar el dataset.")

    required_columns = COLS_NUM + COLS_CAT + [args.target]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Faltan las siguientes columnas requeridas en el dataset: "
            f"{missing_columns}"
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
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }
    print(
        "[+] Métricas -> "
        f"R²: {metrics['r2']:.4f} | "
        f"MAE: ${metrics['mae']:.2f} | "
        f"RMSE: ${metrics['rmse']:.2f}"
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
