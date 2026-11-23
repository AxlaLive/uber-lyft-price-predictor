"""Cliente de prueba de la API de Fase 3.

Toma un registro del CSV de ejemplo de la Fase 2, lo envía a POST /predict con
los tipos del esquema de entrenamiento y muestra la respuesta. Con --sin-campo
omite una variable obligatoria, para comprobar el rechazo y su registro.

Ejemplos:

    python3 cliente.py
    python3 cliente.py --fila 2
    python3 cliente.py --sin-campo distance
    python3 cliente.py --url http://localhost:8000
"""

import argparse
import json
import sys
from pathlib import Path

import httpx
import pandas as pd


EJEMPLO_POR_DEFECTO = (
    Path(__file__).resolve().parents[1] / "fase-2" / "data" / "ejemplo_prediccion.csv"
)


def construir_registro(csv_path: Path, fila: int) -> dict:
    """Convierte una fila del CSV en un objeto JSON con los tipos del esquema.

    Las columnas numéricas viajan como números y las categóricas como texto,
    tal como se entrenó el modelo. Una celda vacía viaja como null, que el
    pipeline imputa con la mediana aprendida en entrenamiento.
    """
    datos = pd.read_csv(csv_path)
    if not 0 <= fila < len(datos):
        raise SystemExit(
            f"[ERROR] La fila {fila} no existe en {csv_path} "
            f"(tiene {len(datos)} filas, de 0 a {len(datos) - 1})."
        )

    registro = {}
    for columna, valor in datos.iloc[fila].items():
        if pd.isna(valor):
            registro[columna] = None
        elif pd.api.types.is_number(valor):
            registro[columna] = float(valor)
        else:
            registro[columna] = str(valor)
    return registro


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Envía una predicción a la API de Fase 3."
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="URL base de la API (por defecto: http://localhost:8000).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=EJEMPLO_POR_DEFECTO,
        help="CSV de ejemplo del que se toma el registro.",
    )
    parser.add_argument(
        "--fila",
        type=int,
        default=0,
        help="Índice de la fila a enviar (por defecto: 0).",
    )
    parser.add_argument(
        "--sin-campo",
        metavar="COLUMNA",
        help="Omite esa columna, para comprobar el rechazo por campo faltante.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Segundos de espera de la respuesta (por defecto: 30).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.csv.is_file():
        print(f"[ERROR] No se encontró el CSV de ejemplo: {args.csv}", file=sys.stderr)
        return 1

    registro = construir_registro(args.csv, args.fila)
    if args.sin_campo:
        if args.sin_campo not in registro:
            print(
                f"[ERROR] La columna '{args.sin_campo}' no está en el CSV. "
                f"Columnas disponibles: {', '.join(registro)}",
                file=sys.stderr,
            )
            return 1
        registro.pop(args.sin_campo)
        print(f"[*] Se omite la columna '{args.sin_campo}' a propósito.")

    destino = f"{args.url.rstrip('/')}/predict"
    print(f"[*] POST {destino}")
    print(json.dumps(registro, indent=2, ensure_ascii=False))

    try:
        respuesta = httpx.post(destino, json=registro, timeout=args.timeout)
    except httpx.HTTPError as exc:
        print(f"[ERROR] No se pudo contactar la API en {destino}: {exc}", file=sys.stderr)
        return 1

    print(f"\n[*] HTTP {respuesta.status_code}")
    try:
        cuerpo = respuesta.json()
    except ValueError:
        print(respuesta.text)
        return 0 if respuesta.is_success else 1

    print(json.dumps(cuerpo, indent=2, ensure_ascii=False))

    if respuesta.is_success:
        print(
            f"\n[+] Precio estimado: ${cuerpo['prediction']:.2f} "
            f"(modelo {cuerpo['model_version']})"
        )
        return 0

    print(f"\n[!] La API rechazó la solicitud: {cuerpo.get('detail')}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
