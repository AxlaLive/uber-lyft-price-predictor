"""
Pruebas automatizadas con pytest - Fase 2
Proyecto Integrador - Modelos y Simulación de Sistemas I
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest


PHASE_DIR = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PHASE_DIR / "train.py"
PREDICT_SCRIPT = PHASE_DIR / "predict.py"
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
INPUT_COLUMNS = COLS_NUM + COLS_CAT


@pytest.fixture
def datos_sinteticos(tmp_path: Path) -> Path:
    """Escribe datos pequeños y deterministas con todas las columnas del modelo."""
    df = pd.DataFrame(
        {
            "distance": [0.5 + i * 0.1 for i in range(30)],
            "surge_multiplier": [1.0 + (i % 3) * 0.25 for i in range(30)],
            "latitude": [42.35 + (i % 5) * 0.01 for i in range(30)],
            "longitude": [-71.05 - (i % 7) * 0.01 for i in range(30)],
            "temperature": [38.0 + i % 9 for i in range(30)],
            "apparentTemperature": [36.0 + i % 9 for i in range(30)],
            "humidity": [0.5 + (i % 5) * 0.05 for i in range(30)],
            "windSpeed": [7.0 + i % 6 for i in range(30)],
            "windGust": [9.0 + i % 8 for i in range(30)],
            "visibility": [8.0 + (i % 4) * 0.5 for i in range(30)],
            "pressure": [1010.0 + i % 7 for i in range(30)],
            "name": [["Shared", "Lux", "Black"][i % 3] for i in range(30)],
            "cab_type": ["Lyft" if i % 2 else "Uber" for i in range(30)],
            "price": [8.0 + i * 0.7 + (i % 4) for i in range(30)],
        }
    )
    csv_path = tmp_path / "datos_prueba.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def train_model(
    data_path: Path,
    output_dir: Path,
    version: str = "v1",
    seed: int = 42,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--data",
            str(data_path),
            "--target",
            "price",
            "--output-dir",
            str(output_dir),
            "--version",
            version,
            "--seed",
            str(seed),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def predict(
    model_path: Path, input_path: Path, output_path: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(PREDICT_SCRIPT),
            "--model",
            str(model_path),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def modelo_entrenado(datos_sinteticos: Path, tmp_path: Path) -> Path:
    output_dir = tmp_path / "models"
    result = train_model(datos_sinteticos, output_dir)
    assert result.returncode == 0, result.stderr
    return output_dir / "v1" / "modelo.joblib"


def test_entrenamiento_y_almacenamiento(
    datos_sinteticos: Path, tmp_path: Path
) -> None:
    """El CLI produce un artefacto cargable, métricas y metadatos."""
    output_dir = tmp_path / "models"
    result = train_model(datos_sinteticos, output_dir)
    version_dir = output_dir / "v1"

    assert result.returncode == 0, result.stderr
    model_path = version_dir / "modelo.joblib"
    metrics_path = version_dir / "metrics.json"
    metadata_path = version_dir / "metadata.json"
    assert model_path.is_file()
    assert metrics_path.is_file()
    assert metadata_path.is_file()

    artifact = joblib.load(model_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert artifact["columnas_entrada"] == INPUT_COLUMNS
    assert set(metrics) == {"rmse", "mae", "r2"}
    assert all(isinstance(value, float) for value in metrics.values())
    assert metadata["version"] == "v1"
    assert metadata["n_entrenamiento"] + metadata["n_prueba"] == 30


def test_carga_y_prediccion_conserva_filas(
    datos_sinteticos: Path, modelo_entrenado: Path, tmp_path: Path
) -> None:
    """El artefacto se carga y predice una vez por cada fila en el mismo orden."""
    input_df = pd.read_csv(datos_sinteticos).iloc[[9, 2, 17, 0]].copy()
    input_df["fila_id"] = ["fila-9", "fila-2", "fila-17", "fila-0"]
    input_path = tmp_path / "entrada.csv"
    output_path = tmp_path / "outputs" / "predicciones.csv"
    input_df.to_csv(input_path, index=False)

    result = predict(modelo_entrenado, input_path, output_path)

    assert result.returncode == 0, result.stderr
    output_df = pd.read_csv(output_path)
    assert len(output_df) == len(input_df)
    assert output_df["fila_id"].tolist() == input_df["fila_id"].tolist()
    assert output_df["prediccion"].notna().all()
    pd.testing.assert_frame_equal(
        output_df.drop(columns="prediccion"),
        input_df.reset_index(drop=True),
        check_dtype=False,
    )


def test_valores_faltantes_en_variables_numericas(
    datos_sinteticos: Path, modelo_entrenado: Path, tmp_path: Path
) -> None:
    """El imputador del pipeline permite inferir con una característica numérica nula."""
    input_df = pd.read_csv(datos_sinteticos).iloc[:5].copy()
    input_df.loc[input_df.index[1], "distance"] = float("nan")
    input_path = tmp_path / "entrada_con_nulos.csv"
    output_path = tmp_path / "predicciones_con_nulos.csv"
    input_df.to_csv(input_path, index=False)

    result = predict(modelo_entrenado, input_path, output_path)

    assert result.returncode == 0, result.stderr
    output_df = pd.read_csv(output_path)
    assert len(output_df) == len(input_df)
    assert output_df["prediccion"].notna().all()


def test_entrada_invalida_falla_con_codigo_no_cero(
    datos_sinteticos: Path, modelo_entrenado: Path, tmp_path: Path
) -> None:
    """Una columna obligatoria ausente se rechaza sin producir un CSV de salida."""
    input_df = pd.read_csv(datos_sinteticos).drop(columns="distance")
    input_path = tmp_path / "entrada_invalida.csv"
    output_path = tmp_path / "no_debe_generarse.csv"
    input_df.to_csv(input_path, index=False)

    result = predict(modelo_entrenado, input_path, output_path)

    assert result.returncode == 1
    assert "distance" in result.stderr
    assert not output_path.exists()


def test_versionamiento_conserva_versiones_y_rechaza_sobrescritura(
    datos_sinteticos: Path, tmp_path: Path
) -> None:
    """Dos versiones coexisten y reusar una versión existente no la modifica."""
    output_dir = tmp_path / "models"
    first = train_model(datos_sinteticos, output_dir, version="v1", seed=42)
    assert first.returncode == 0, first.stderr

    v1_dir = output_dir / "v1"
    v1_files = {path.name for path in v1_dir.iterdir()}
    v1_hash_before = hashlib.sha256(
        (v1_dir / "modelo.joblib").read_bytes()
    ).hexdigest()

    second = train_model(datos_sinteticos, output_dir, version="v2", seed=123)
    assert second.returncode == 0, second.stderr
    v2_dir = output_dir / "v2"
    assert v1_files == {path.name for path in v2_dir.iterdir()}
    assert json.loads((v1_dir / "metadata.json").read_text())["version"] == "v1"
    assert json.loads((v2_dir / "metadata.json").read_text())["version"] == "v2"

    duplicate = train_model(datos_sinteticos, output_dir, version="v1", seed=999)

    assert duplicate.returncode != 0
    assert "No se sobrescribirá" in duplicate.stderr
    assert hashlib.sha256((v1_dir / "modelo.joblib").read_bytes()).hexdigest() == (
        v1_hash_before
    )
    assert v1_files == {path.name for path in v1_dir.iterdir()}
