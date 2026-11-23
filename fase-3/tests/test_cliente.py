"""Pruebas del cliente de prueba de la API (fase-3/cliente.py)."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cliente  # noqa: E402


PHASE_DIR = Path(__file__).resolve().parents[1]
CLIENTE = PHASE_DIR / "cliente.py"
EJEMPLO = PHASE_DIR.parent / "fase-2" / "data" / "ejemplo_prediccion.csv"


def test_construye_el_registro_con_los_tipos_del_esquema() -> None:
    """Los números viajan como números y las categorías como texto."""
    registro = cliente.construir_registro(EJEMPLO, 0)
    esperado = pd.read_csv(EJEMPLO).iloc[0]

    assert set(registro) == set(esperado.index)
    assert isinstance(registro["distance"], float)
    assert isinstance(registro["surge_multiplier"], float)
    assert isinstance(registro["name"], str)
    assert isinstance(registro["cab_type"], str)
    assert registro["distance"] == pytest.approx(float(esperado["distance"]))
    assert registro["name"] == esperado["name"]


def test_una_celda_vacia_viaja_como_null(tmp_path: Path) -> None:
    """Un dato faltante se envía como null, que el pipeline imputa."""
    datos = pd.read_csv(EJEMPLO)
    datos.loc[0, "visibility"] = None
    csv_path = tmp_path / "con_vacio.csv"
    datos.to_csv(csv_path, index=False)

    registro = cliente.construir_registro(csv_path, 0)

    assert registro["visibility"] is None
    assert registro["distance"] is not None


def test_fila_fuera_de_rango_termina_con_mensaje_claro() -> None:
    """Pedir una fila inexistente no debe producir una solicitud vacía."""
    total = len(pd.read_csv(EJEMPLO))

    with pytest.raises(SystemExit) as error:
        cliente.construir_registro(EJEMPLO, total)

    assert f"La fila {total} no existe" in str(error.value)


def ejecutar_cliente(*extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CLIENTE), *extra],
        capture_output=True,
        text=True,
        check=False,
    )


def test_sin_campo_omite_la_columna_indicada() -> None:
    """--sin-campo debe quitar la variable antes de enviar la solicitud.

    Se apunta a un puerto donde no hay API: basta con revisar el JSON impreso
    antes del envío, sin depender de un servidor en ejecución.
    """
    resultado = ejecutar_cliente(
        "--sin-campo", "distance", "--url", "http://127.0.0.1:9", "--timeout", "2"
    )

    assert "Se omite la columna 'distance'" in resultado.stdout
    assert '"distance"' not in resultado.stdout
    assert '"cab_type"' in resultado.stdout


def test_columna_inexistente_en_sin_campo_es_rechazada() -> None:
    resultado = ejecutar_cliente(
        "--sin-campo", "no_existe", "--url", "http://127.0.0.1:9", "--timeout", "2"
    )

    assert resultado.returncode == 1
    assert "no está en el CSV" in resultado.stderr


def test_api_inalcanzable_termina_con_codigo_de_error() -> None:
    """Sin API no debe simular un resultado: termina con código distinto de cero."""
    resultado = ejecutar_cliente("--url", "http://127.0.0.1:9", "--timeout", "2")

    assert resultado.returncode == 1
    assert "No se pudo contactar la API" in resultado.stderr
