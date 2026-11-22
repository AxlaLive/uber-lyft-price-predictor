import csv
import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from main import app


client = TestClient(app)
EXAMPLE_PATH = (
    Path(__file__).resolve().parents[2]
    / "fase-2"
    / "data"
    / "ejemplo_prediccion.csv"
)


def example_record() -> dict:
    return pd.read_csv(EXAMPLE_PATH).iloc[0].to_dict()


@pytest.fixture(autouse=True)
def configure_prediction_log(tmp_path, monkeypatch):
    log_path = tmp_path / "logs" / "prediction_events.csv"
    monkeypatch.setenv("PREDICT_LOG_PATH", str(log_path))
    return log_path


def read_prediction_events(log_path: Path) -> list[dict[str, str]]:
    with log_path.open(newline="", encoding="utf-8") as log_file:
        return list(csv.DictReader(log_file))


def create_training_csv(path: Path) -> Path:
    rows = []
    for index in range(30):
        rows.append(
            {
                "distance": 0.5 + index * 0.1,
                "surge_multiplier": 1.0 + (index % 3) * 0.25,
                "latitude": 42.35 + (index % 5) * 0.01,
                "longitude": -71.05 - (index % 7) * 0.01,
                "temperature": 38.0 + index % 9,
                "apparentTemperature": 36.0 + index % 9,
                "humidity": 0.5 + (index % 5) * 0.05,
                "windSpeed": 7.0 + index % 6,
                "windGust": 9.0 + index % 8,
                "visibility": 8.0 + (index % 4) * 0.5,
                "pressure": 1010.0 + index % 7,
                "name": ["Shared", "Lux", "Black"][index % 3],
                "cab_type": "Lyft" if index % 2 else "Uber",
                "price": 8.0 + index * 0.7 + index % 4,
                "fare": 8.0 + index * 0.7 + index % 4,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_health_returns_http_200_and_running_status() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "API en ejecución"}


def test_predict_uses_saved_pipeline_and_returns_version(
    configure_prediction_log: Path,
) -> None:
    response = client.post("/predict", json=example_record())

    assert response.status_code == 200
    assert response.json()["model_version"] == "v1"
    assert isinstance(response.json()["prediction"], float)
    event = read_prediction_events(configure_prediction_log)[0]
    assert event["http_status"] == "200"
    assert event["model_version"] == "v1"
    assert float(event["prediction"]) == response.json()["prediction"]
    assert event["timestamp"]


def test_predict_reports_missing_field_name(
    configure_prediction_log: Path,
) -> None:
    payload = example_record()
    del payload["distance"]

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert "distance" in response.json()["detail"]
    event = read_prediction_events(configure_prediction_log)[0]
    assert event["http_status"] == "422"
    assert event["model_version"] == ""
    assert event["prediction"] == ""


def test_predict_logs_fastapi_validation_rejection_and_appends_rows(
    configure_prediction_log: Path,
) -> None:
    response = client.post("/predict", content="{invalid json")
    valid_response = client.post("/predict", json=example_record())

    assert response.status_code == 422
    assert valid_response.status_code == 200
    with configure_prediction_log.open(newline="", encoding="utf-8") as log_file:
        lines = log_file.readlines()
    events = read_prediction_events(configure_prediction_log)

    assert len(lines) == 3
    assert lines[0].startswith("timestamp,http_status,model_version,prediction")
    assert len(events) == 2
    assert events[0]["http_status"] == "422"
    assert events[0]["model_version"] == ""
    assert events[0]["prediction"] == ""
    assert events[1]["http_status"] == "200"
    assert events[1]["model_version"] == "v1"
    assert events[1]["prediction"]


def test_predict_rejects_incompatible_numeric_type() -> None:
    payload = example_record()
    payload["distance"] = "long"

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert "distance" in response.json()["detail"]


def test_predict_rejects_incompatible_categorical_type() -> None:
    payload = example_record()
    payload["name"] = 123

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert "name" in response.json()["detail"]


def test_predict_imputes_null_numeric_value_with_saved_pipeline() -> None:
    payload = example_record()
    payload["distance"] = None

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert isinstance(response.json()["prediction"], float)


def test_predict_rejects_null_categorical_value() -> None:
    payload = example_record()
    payload["name"] = None

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert "name" in response.json()["detail"]


def test_predict_returns_503_when_configured_model_does_not_exist(
    monkeypatch, tmp_path, configure_prediction_log: Path
) -> None:
    from main import _load_model

    _load_model.cache_clear()
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("MODEL_VERSION", "missing")

    response = client.post("/predict", json=example_record())

    assert response.status_code == 503
    assert "no está disponible" in response.json()["detail"]
    event = read_prediction_events(configure_prediction_log)[0]
    assert event["http_status"] == "503"
    assert event["model_version"] == ""
    assert event["prediction"] == ""
    _load_model.cache_clear()


def test_train_accepts_empty_body_and_uses_configured_defaults(
    monkeypatch, tmp_path
) -> None:
    from main import _load_model

    data_path = create_training_csv(tmp_path / "training.csv")
    model_dir = tmp_path / "models"
    (model_dir / "v1").mkdir(parents=True)
    (model_dir / "v1" / "keep.txt").write_text("existing version")
    (model_dir / "v2").mkdir()

    monkeypatch.setenv("TRAIN_DATA_PATH", str(data_path))
    monkeypatch.setenv("TRAIN_TARGET", "price")
    monkeypatch.setenv("TRAIN_SEED", "42")
    monkeypatch.setenv("MODEL_DIR", str(model_dir))

    first_response = client.post("/train")
    assert first_response.status_code == 201, first_response.text
    first = first_response.json()
    assert first["version"].startswith("api-")
    assert set(first["metrics"]) == {"metrica_principal", "modelo", "modelo_base"}
    first_version_dir = model_dir / first["version"]
    assert (first_version_dir / "modelo.joblib").is_file()
    assert (first_version_dir / "metrics.json").is_file()
    assert (first_version_dir / "metadata.json").is_file()
    assert (model_dir / "v1" / "keep.txt").read_text() == "existing version"

    second_response = client.post("/train", json={"target": "fare", "seed": 7})
    assert second_response.status_code == 201, second_response.text
    second = second_response.json()
    assert second["version"] != first["version"]
    assert (model_dir / second["version"] / "modelo.joblib").is_file()
    second_metadata = json.loads(
        (model_dir / second["version"] / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert second_metadata["variable_objetivo"] == "fare"
    assert second_metadata["semilla"] == 7
    _load_model.cache_clear()


def test_train_returns_503_when_dataset_is_not_configured(monkeypatch) -> None:
    monkeypatch.delenv("TRAIN_DATA_PATH", raising=False)

    response = client.post("/train", json={})

    assert response.status_code == 503
    assert "TRAIN_DATA_PATH" in response.json()["detail"]
