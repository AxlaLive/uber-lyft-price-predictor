"""Métricas de operación de la API (Fase 4).

Lee el registro de solicitudes a POST /predict que escribe la API y muestra dos
indicadores: el total de solicitudes y el porcentaje de fallidas, contando como
fallida toda respuesta HTTP 4xx o 5xx. Si el registro está vacío, informa un
total de 0 y «sin solicitudes» en lugar de un porcentaje.

Ejemplos:

    python3 metricas.py
    python3 metricas.py --registro logs/predict_requests.csv
    python3 metricas.py --detalle
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


REGISTRO_POR_DEFECTO = Path(__file__).resolve().parent / "logs" / "predict_requests.csv"
CAMPO_ESTADO = "http_status"
CAMPO_VERSION = "model_version"


def es_fallida(estado: str) -> bool:
    """Una solicitud falla si su código HTTP está entre 400 y 599."""
    try:
        codigo = int(str(estado).strip())
    except (TypeError, ValueError):
        return False
    return 400 <= codigo <= 599


def leer_registro(ruta: Path) -> list[dict[str, str]]:
    """Devuelve las filas de datos del registro; lista vacía si no hay ninguna."""
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe el registro de solicitudes: {ruta}")

    with ruta.open(newline="", encoding="utf-8") as archivo:
        filas = [fila for fila in csv.DictReader(archivo) if any(fila.values())]

    if filas and CAMPO_ESTADO not in filas[0]:
        raise ValueError(
            f"El registro no tiene la columna '{CAMPO_ESTADO}'. "
            f"Columnas encontradas: {', '.join(filas[0])}"
        )
    return filas


def calcular_metricas(filas: list[dict[str, str]]) -> dict:
    """Total de solicitudes y porcentaje de fallidas."""
    total = len(filas)
    fallidas = sum(1 for fila in filas if es_fallida(fila.get(CAMPO_ESTADO, "")))
    return {
        "total": total,
        "fallidas": fallidas,
        # Sin solicitudes no hay porcentaje que calcular: se informa aparte.
        "porcentaje_fallidas": None if total == 0 else fallidas * 100 / total,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calcula el total de solicitudes y el porcentaje de fallidas."
    )
    parser.add_argument(
        "--registro",
        type=Path,
        default=REGISTRO_POR_DEFECTO,
        help="Ruta del registro de solicitudes (por defecto: logs/predict_requests.csv).",
    )
    parser.add_argument(
        "--detalle",
        action="store_true",
        help="Muestra además el desglose por código HTTP y por versión del modelo.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        filas = leer_registro(args.registro)
    except (FileNotFoundError, OSError, ValueError, csv.Error) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    metricas = calcular_metricas(filas)

    print(f"Registro: {args.registro}")
    print(f"Total de solicitudes: {metricas['total']}")
    if metricas["porcentaje_fallidas"] is None:
        print("Porcentaje de fallidas: sin solicitudes")
    else:
        print(
            f"Solicitudes fallidas (HTTP 4xx o 5xx): {metricas['fallidas']}"
        )
        print(f"Porcentaje de fallidas: {metricas['porcentaje_fallidas']:.1f} %")

    if args.detalle and filas:
        print("\nPor código HTTP:")
        for codigo, cuenta in sorted(
            Counter(fila.get(CAMPO_ESTADO, "") for fila in filas).items()
        ):
            print(f"  {codigo or 'sin código'}: {cuenta}")

        print("\nPor versión del modelo:")
        for version, cuenta in sorted(
            Counter(fila.get(CAMPO_VERSION, "") for fila in filas).items()
        ):
            print(f"  {version or 'sin versión (solicitud rechazada)'}: {cuenta}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
