"""Pruebas del script de métricas de operación (Fase 4)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metricas import calcular_metricas, es_fallida, leer_registro  # noqa: E402


PHASE_DIR = Path(__file__).resolve().parents[1]
SCRIPT = PHASE_DIR / "metricas.py"
ENCABEZADO = "timestamp,http_status,model_version,prediction\n"


def escribir_registro(path: Path, filas: str = "") -> Path:
    path.write_text(ENCABEZADO + filas, encoding="utf-8")
    return path


def ejecutar(registro: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--registro", str(registro), *extra],
        capture_output=True,
        text=True,
        check=False,
    )


def test_registro_vacio_informa_cero_y_sin_solicitudes(tmp_path: Path) -> None:
    """Un registro con encabezado pero sin filas no debe dividir por cero."""
    registro = escribir_registro(tmp_path / "vacio.csv")

    assert calcular_metricas(leer_registro(registro)) == {
        "total": 0,
        "fallidas": 0,
        "porcentaje_fallidas": None,
    }

    resultado = ejecutar(registro)
    assert resultado.returncode == 0, resultado.stderr
    assert "Total de solicitudes: 0" in resultado.stdout
    assert "Porcentaje de fallidas: sin solicitudes" in resultado.stdout


def test_dos_validas_y_una_invalida_reportan_33_por_ciento(tmp_path: Path) -> None:
    """El ensayo de la especificación: 3 solicitudes y 33,3 % de fallos."""
    registro = escribir_registro(
        tmp_path / "ensayo.csv",
        "2026-11-22T10:00:00,200,v1,18.7\n"
        "2026-11-22T10:01:00,200,v1,19.2\n"
        "2026-11-22T10:02:00,422,,\n",
    )

    metricas = calcular_metricas(leer_registro(registro))
    assert metricas["total"] == 3
    assert metricas["fallidas"] == 1
    assert metricas["porcentaje_fallidas"] == pytest.approx(100 / 3)

    resultado = ejecutar(registro)
    assert resultado.returncode == 0, resultado.stderr
    assert "Total de solicitudes: 3" in resultado.stdout
    assert "Porcentaje de fallidas: 33.3 %" in resultado.stdout


@pytest.mark.parametrize(
    "estado, esperado",
    [
        ("200", False),
        ("201", False),
        ("399", False),
        ("400", True),
        ("422", True),
        ("503", True),
        ("599", True),
        ("600", False),
        ("", False),
        ("no-es-un-codigo", False),
    ],
)
def test_clasificacion_de_codigos_http(estado: str, esperado: bool) -> None:
    """Fallan solo los códigos entre 400 y 599."""
    assert es_fallida(estado) is esperado


def test_todas_fallidas_reportan_cien_por_ciento(tmp_path: Path) -> None:
    registro = escribir_registro(
        tmp_path / "todas_fallan.csv",
        "2026-11-22T10:00:00,422,,\n2026-11-22T10:01:00,503,,\n",
    )

    resultado = ejecutar(registro)
    assert resultado.returncode == 0, resultado.stderr
    assert "Total de solicitudes: 2" in resultado.stdout
    assert "Porcentaje de fallidas: 100.0 %" in resultado.stdout


def test_detalle_desglosa_por_codigo_y_version(tmp_path: Path) -> None:
    registro = escribir_registro(
        tmp_path / "detalle.csv",
        "2026-11-22T10:00:00,200,v1,18.7\n"
        "2026-11-22T10:01:00,200,v2,19.2\n"
        "2026-11-22T10:02:00,422,,\n",
    )

    resultado = ejecutar(registro, "--detalle")
    assert resultado.returncode == 0, resultado.stderr
    assert "  200: 2" in resultado.stdout
    assert "  422: 1" in resultado.stdout
    assert "  v1: 1" in resultado.stdout
    assert "sin versión (solicitud rechazada): 1" in resultado.stdout


def test_registro_inexistente_termina_con_error(tmp_path: Path) -> None:
    """Un registro que no existe no debe producir un reporte aparentemente válido."""
    resultado = ejecutar(tmp_path / "no_existe.csv")

    assert resultado.returncode == 1
    assert "No existe el registro" in resultado.stderr
    assert "Total de solicitudes" not in resultado.stdout


def escribir_jsonl(path: Path, eventos: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(evento, ensure_ascii=False) + "\n" for evento in eventos),
        encoding="utf-8",
    )
    return path


def test_registro_jsonl_se_lee_igual_que_el_csv(tmp_path: Path) -> None:
    """La especificación admite CSV o JSONL: el reporte debe ser el mismo.

    Antes, un JSONL con solicitudes se informaba como «sin solicitudes», porque
    el lector solo entendía CSV y no avisaba del problema.
    """
    registro = escribir_jsonl(
        tmp_path / "eventos.jsonl",
        [
            {
                "timestamp": "2026-11-22T10:00:00",
                "http_status": 200,
                "model_version": "v1",
                "prediction": 18.7,
            },
            {
                "timestamp": "2026-11-22T10:01:00",
                "http_status": 200,
                "model_version": "v1",
                "prediction": 19.2,
            },
            {
                "timestamp": "2026-11-22T10:02:00",
                "http_status": 422,
                "model_version": None,
                "prediction": None,
            },
        ],
    )

    metricas = calcular_metricas(leer_registro(registro))
    assert metricas["total"] == 3
    assert metricas["fallidas"] == 1
    assert metricas["porcentaje_fallidas"] == pytest.approx(100 / 3)

    resultado = ejecutar(registro, "--detalle")
    assert resultado.returncode == 0, resultado.stderr
    assert "Total de solicitudes: 3" in resultado.stdout
    assert "Porcentaje de fallidas: 33.3 %" in resultado.stdout
    # Un valor nulo no debe romper el desglose ni su ordenamiento.
    assert "sin versión (solicitud rechazada): 1" in resultado.stdout
    assert "  v1: 2" in resultado.stdout


def test_jsonl_vacio_informa_sin_solicitudes(tmp_path: Path) -> None:
    registro = escribir_jsonl(tmp_path / "vacio.jsonl", [])

    resultado = ejecutar(registro)
    assert resultado.returncode == 0, resultado.stderr
    assert "Total de solicitudes: 0" in resultado.stdout
    assert "Porcentaje de fallidas: sin solicitudes" in resultado.stdout


def test_jsonl_con_linea_invalida_termina_con_error(tmp_path: Path) -> None:
    """Una línea corrupta no debe reportarse como si no hubiera solicitudes."""
    registro = tmp_path / "corrupto.jsonl"
    registro.write_text(
        '{"timestamp": "2026-11-22T10:00:00", "http_status": 200}\n'
        "{esto no es json}\n",
        encoding="utf-8",
    )

    resultado = ejecutar(registro)
    assert resultado.returncode == 1
    assert "línea 2" in resultado.stderr
    assert "Total de solicitudes" not in resultado.stdout
