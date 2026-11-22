"""API de predicción de tarifas para la Fase 3."""

import csv
import importlib.util
import logging
import math
import os
import pickle
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


app = FastAPI(
    title="Uber/Lyft Price Prediction API",
    description="API para el proyecto de predicción de tarifas.",
    version="0.1.0",
)
logger = logging.getLogger(__name__)
VERSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "fase-2" / "models"
DEFAULT_PREDICT_LOG_PATH = (
    Path(__file__).resolve().parents[1]
    / "fase-4"
    / "logs"
    / "predict_requests.csv"
)
PHASE2_TRAIN_PATH = Path(__file__).resolve().parents[1] / "fase-2" / "train.py"
_training_lock = threading.Lock()
_prediction_log_lock = threading.Lock()
PREDICT_LOG_FIELDS = (
    "timestamp",
    "http_status",
    "model_version",
    "prediction",
)


def _load_training_module() -> Any:
    spec = importlib.util.spec_from_file_location("phase2_train", PHASE2_TRAIN_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(
            f"No se pudo importar el entrenador de Fase 2: {PHASE2_TRAIN_PATH}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


training_module = _load_training_module()


class TrainRequest(BaseModel):
    target: Optional[str] = Field(default=None, min_length=1)
    seed: Optional[int] = Field(default=None, ge=0)


@dataclass(frozen=True)
class ModelBundle:
    pipeline: Any
    columns: tuple[str, ...]
    numeric_columns: frozenset[str]
    categorical_columns: frozenset[str]


def _model_path() -> tuple[str, Path]:
    version = os.getenv("MODEL_VERSION", "v1")
    if not VERSION_PATTERN.fullmatch(version) or version in {".", ".."}:
        raise ValueError("MODEL_VERSION tiene un formato inválido.")

    model_dir = Path(os.getenv("MODEL_DIR", str(DEFAULT_MODEL_DIR))).expanduser()
    return version, model_dir / version / "modelo.joblib"


@lru_cache(maxsize=4)
def _load_model(path: str) -> ModelBundle:
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"No existe el artefacto del modelo: {artifact_path}")

    artifact = joblib.load(artifact_path)
    if not isinstance(artifact, dict):
        raise ValueError("El artefacto del modelo debe ser un diccionario.")

    pipeline = artifact.get("pipeline")
    columns = artifact.get("columnas_entrada")
    numeric_columns = artifact.get("cols_num")
    categorical_columns = artifact.get("cols_cat")

    if not callable(getattr(pipeline, "predict", None)):
        raise ValueError("El artefacto no contiene un pipeline predictivo válido.")
    if (
        not isinstance(columns, list)
        or not columns
        or not all(isinstance(column, str) for column in columns)
        or len(set(columns)) != len(columns)
    ):
        raise ValueError(
            "'columnas_entrada' debe ser una lista válida y sin duplicados."
        )
    if not isinstance(numeric_columns, list) or not isinstance(categorical_columns, list):
        raise ValueError(
            "El artefacto no contiene las columnas numéricas y categóricas."
        )
    if set(numeric_columns) & set(categorical_columns):
        raise ValueError("El esquema del artefacto clasifica una columna dos veces.")
    if set(columns) != set(numeric_columns) | set(categorical_columns):
        raise ValueError("El esquema del artefacto es inconsistente.")

    return ModelBundle(
        pipeline=pipeline,
        columns=tuple(columns),
        numeric_columns=frozenset(numeric_columns),
        categorical_columns=frozenset(categorical_columns),
    )


def _get_model() -> tuple[str, ModelBundle]:
    try:
        version, path = _model_path()
        return version, _load_model(str(path.resolve()))
    except (
        AttributeError,
        EOFError,
        FileNotFoundError,
        ImportError,
        KeyError,
        OSError,
        pickle.UnpicklingError,
        TypeError,
        ValueError,
    ) as exc:
        logger.exception("No se pudo cargar la versión configurada del modelo.")
        raise HTTPException(
            status_code=503,
            detail=f"El modelo configurado no está disponible: {exc}",
        ) from exc


def _validate_features(payload: dict[str, Any], bundle: ModelBundle) -> dict[str, Any]:
    missing = [column for column in bundle.columns if column not in payload]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Faltan campos requeridos: {', '.join(missing)}",
        )

    unexpected = [column for column in payload if column not in bundle.columns]
    if unexpected:
        raise HTTPException(
            status_code=422,
            detail=f"Campos no permitidos: {', '.join(unexpected)}",
        )

    for column in bundle.columns:
        value = payload[column]
        if column in bundle.numeric_columns:
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise HTTPException(
                    status_code=422,
                    detail=f"El campo '{column}' debe ser numérico o null.",
                )
            try:
                is_finite = math.isfinite(float(value))
            except (OverflowError, ValueError):
                is_finite = False
            if not is_finite:
                raise HTTPException(
                    status_code=422,
                    detail=f"El campo '{column}' debe ser un número finito.",
                )
        elif value is None or not isinstance(value, str):
            raise HTTPException(
                status_code=422,
                detail=f"El campo '{column}' debe ser texto y no admite null.",
            )

    return {column: payload[column] for column in bundle.columns}


def _new_training_version(output_dir: Path) -> str:
    while True:
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        version = f"api-{timestamp}-{uuid4().hex[:8]}"
        if not (output_dir / version).exists():
            return version


def _append_prediction_event(
    timestamp: str,
    http_status: int,
    model_version: str,
    prediction: Optional[float],
) -> None:
    log_path = Path(
        os.getenv("PREDICT_LOG_PATH", str(DEFAULT_PREDICT_LOG_PATH))
    ).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with _prediction_log_lock:
        needs_header = not log_path.exists() or log_path.stat().st_size == 0
        with log_path.open("a", newline="", encoding="utf-8") as log_file:
            writer = csv.DictWriter(log_file, fieldnames=PREDICT_LOG_FIELDS)
            if needs_header:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp": timestamp,
                    "http_status": http_status,
                    "model_version": model_version,
                    "prediction": "" if prediction is None else prediction,
                }
            )
            log_file.flush()


@app.middleware("http")
async def log_predict_requests(request: Request, call_next):
    if request.method != "POST" or request.url.path != "/predict":
        return await call_next(request)

    request.state.model_version = ""
    request.state.prediction = None
    timestamp = datetime.now(timezone.utc).isoformat()

    try:
        response = await call_next(request)
    except Exception:
        try:
            _append_prediction_event(
                timestamp,
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                request.state.model_version,
                request.state.prediction,
            )
        except (OSError, csv.Error):
            logger.exception("No se pudo guardar el registro de POST /predict.")
        raise

    try:
        _append_prediction_event(
            timestamp,
            response.status_code,
            request.state.model_version,
            request.state.prediction,
        )
    except (OSError, csv.Error) as exc:
        logger.exception("No se pudo guardar el registro de POST /predict.")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "No se pudo registrar la solicitud de predicción."},
        )

    return response


@app.get("/health")
def health() -> dict[str, str]:
    """Indica que el proceso de la API está disponible."""
    return {"status": "ok", "message": "API en ejecución"}


@app.post("/predict")
def predict(payload: dict[str, Any], request: Request) -> dict[str, Any]:
    """Predice el precio de un registro usando el pipeline versionado."""
    version, bundle = _get_model()
    features = _validate_features(payload, bundle)

    try:
        frame = pd.DataFrame([features], columns=bundle.columns)
    except (TypeError, ValueError) as exc:
        logger.exception("No se pudo construir la fila de entrada.")
        raise HTTPException(
            status_code=422,
            detail="No fue posible preparar los datos de entrada.",
        ) from exc

    try:
        request.state.model_version = version
        prediction = float(bundle.pipeline.predict(frame)[0])
        request.state.prediction = prediction
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        logger.exception("Falló la predicción con el modelo cargado.")
        raise HTTPException(
            status_code=503,
            detail="No fue posible generar una predicción con el modelo configurado.",
        ) from exc

    return {"prediction": prediction, "model_version": version}


@app.post("/train", status_code=status.HTTP_201_CREATED)
def train(request: Optional[TrainRequest] = None) -> dict[str, Any]:
    """Entrena una versión nueva con la configuración del servicio."""
    data_path_value = os.getenv("TRAIN_DATA_PATH")
    if not data_path_value:
        raise HTTPException(
            status_code=503,
            detail="Falta configurar TRAIN_DATA_PATH con la ruta al CSV de entrenamiento.",
        )

    try:
        target = (
            request.target
            if request is not None and request.target is not None
            else os.getenv("TRAIN_TARGET", "price")
        )
        target = target.strip()
        seed_value = (
            request.seed
            if request is not None and request.seed is not None
            else int(os.getenv("TRAIN_SEED", "42"))
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=503,
            detail="La configuración TRAIN_SEED debe ser un entero no negativo.",
        ) from exc

    if not target:
        raise HTTPException(
            status_code=503,
            detail="La variable TRAIN_TARGET debe tener un nombre válido.",
        )

    data_path = Path(data_path_value).expanduser()
    model_dir = Path(os.getenv("MODEL_DIR", str(DEFAULT_MODEL_DIR))).expanduser()

    with _training_lock:
        version = _new_training_version(model_dir)
        try:
            result = training_module.train_model(
                data_path=data_path,
                target=target,
                output_dir=model_dir,
                version=version,
                seed=seed_value,
            )
        except FileNotFoundError as exc:
            logger.exception("No se encontró el dataset de entrenamiento.")
            raise HTTPException(
                status_code=503,
                detail=f"No está disponible el dataset configurado: {exc}",
            ) from exc
        except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            logger.exception("El dataset de entrenamiento no es válido.")
            raise HTTPException(
                status_code=422,
                detail=f"No se pudo entrenar con el dataset configurado: {exc}",
            ) from exc
        except FileExistsError as exc:
            logger.exception("La versión generada ya existe.")
            raise HTTPException(
                status_code=409,
                detail="No se pudo reservar un identificador de versión disponible.",
            ) from exc
        except (OSError, RuntimeError) as exc:
            logger.exception("Falló el entrenamiento del modelo.")
            raise HTTPException(
                status_code=500,
                detail="Ocurrió un error al entrenar o guardar la nueva versión.",
            ) from exc

    return {
        "version": result["version"],
        "metrics": result["metrics"],
    }
