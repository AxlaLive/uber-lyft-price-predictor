"""
Script de predicción (predict.py) - Fase 2
Proyecto Integrador - Modelos y Simulación de Sistemas I
"""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera predicciones utilizando un modelo versionado."
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Ruta al archivo modelo.joblib de la versión.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Ruta al CSV con las variables de entrada.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Ruta al CSV de salida con las predicciones.",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"[ERROR] {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    args = parse_args()

    if not args.model.is_file():
        fail(f"El archivo de modelo no existe en la ruta: {args.model}")
    if not args.input.is_file():
        fail(f"El archivo de entrada no existe en la ruta: {args.input}")

    print(f"[*] Cargando artefacto del modelo desde: {args.model}")
    try:
        artifact = joblib.load(args.model)
        pipeline = artifact["pipeline"]
        expected_columns = artifact["columnas_entrada"]
        if not isinstance(expected_columns, list) or not expected_columns:
            raise ValueError("'columnas_entrada' debe ser una lista no vacía.")
    except (OSError, EOFError, KeyError, TypeError, ValueError, ImportError) as exc:
        fail(f"No se pudo cargar un artefacto válido del modelo: {exc}")

    print(f"[*] Leyendo datos de entrada desde: {args.input}")
    try:
        df_input = pd.read_csv(args.input)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        fail(f"No se pudo leer el archivo CSV de entrada: {exc}")

    missing_columns = [
        column for column in expected_columns if column not in df_input.columns
    ]
    if missing_columns:
        fail(
            "El archivo de entrada no contiene las columnas requeridas: "
            f"{missing_columns}"
        )

    X_new = df_input.loc[:, expected_columns]

    print("[*] Generando predicciones...")
    try:
        predictions = pipeline.predict(X_new)
    except (AttributeError, TypeError, ValueError, KeyError) as exc:
        fail(f"Falló la ejecución de la predicción: {exc}")

    if len(predictions) != len(df_input):
        fail(
            "El modelo devolvió una cantidad de predicciones distinta a las "
            "filas de entrada."
        )

    output = df_input.copy()
    output["prediccion"] = predictions

    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(args.output, index=False)
    except (OSError, ValueError) as exc:
        fail(f"No se pudo guardar el archivo de salida: {exc}")

    print(f"[+] Predicciones guardadas exitosamente en: {args.output}")


if __name__ == "__main__":
    main()
