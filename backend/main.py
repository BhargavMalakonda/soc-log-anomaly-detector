"""
FastAPI application for the AI Log Anomaly Detection Dashboard.

Routes:
    GET  /          → serves index.html (Jinja2 template)
    POST /upload    → validates + parses CSV → preview JSON
    POST /detect    → preprocess + detect → results JSON
    GET  /static/*  → serves CSS / JS static files

Validation rules applied to both /upload and /detect:
    1. File present and non-empty (0 bytes)       → HTTP 400
    2. File size > 50 MB                          → HTTP 413
    3. Extension is not .csv                      → HTTP 422
    4. CSV cannot be parsed                       → HTTP 422
    5. CSV has more than 500 000 rows             → HTTP 413
    6. contamination outside [0.01, 0.30]         → HTTP 422  (/detect only)
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from backend.model import CONTAMINATION, RANDOM_STATE, run_isolation_forest
from backend.preprocess import (
    CSVParseError,
    FeatureReportEntry,
    PreprocessingError,
    csv_bytes_to_dataframe,
    preprocess,
)
from backend.utils import assign_severities, error_response

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_FILE_SIZE_BYTES: int = 50 * 1024 * 1024   # 50 MB
MAX_CSV_ROWS: int = 500_000

CONTAMINATION_MIN: float = 0.01
CONTAMINATION_MAX: float = 0.30

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="AI Log Anomaly Detection Dashboard")

# CORS — credentials disabled; wildcard origin is safe without credentials.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
app.mount(
    "/static",
    StaticFiles(directory=str(PROJECT_ROOT / "frontend" / "static")),
    name="static",
)
templates = Jinja2Templates(
    directory=str(PROJECT_ROOT / "frontend" / "templates")
)

# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class UploadResponse(BaseModel):
    columns: list[str]
    preview_rows: list[dict]
    total_rows: int


class SummaryStats(BaseModel):
    total: int
    anomalies: int
    high: int
    medium: int
    low: int


class DetectResponse(BaseModel):
    columns: list[str]
    rows: list[dict]
    summary: SummaryStats
    feature_report: list[dict]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_python_native(value: Any) -> Any:
    """Convert numpy scalar types and NaN/Inf to JSON-serialisable Python types."""
    if value is None:
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        f = float(value)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    return value


def _sanitise_row(row: dict) -> dict:
    """Return a copy of *row* with all values converted to JSON-safe Python types."""
    return {k: _to_python_native(v) for k, v in row.items()}


class _ValidationError(Exception):
    """Internal sentinel used to short-circuit validation in route handlers."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


async def _read_and_validate_file(file: UploadFile) -> bytes:
    """Read the uploaded file and apply file-level validation rules.

    Returns raw bytes on success; raises _ValidationError on any failure.
    """
    # Rule 1 — file must be present and non-empty
    if file is None or file.filename == "":
        raise _ValidationError(status_code=400, detail="No file provided")

    content: bytes = await file.read()

    if len(content) == 0:
        raise _ValidationError(status_code=400, detail="Uploaded file is empty")

    # Rule 2 — file size must not exceed 50 MB
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise _ValidationError(
            status_code=413, detail="File exceeds the 50 MB size limit"
        )

    # Rule 3 — extension must be .csv
    filename: str = file.filename or ""
    if not filename.lower().endswith(".csv"):
        raise _ValidationError(
            status_code=422, detail="Only .csv files are accepted"
        )

    return content


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
async def index(request: Request):
    """Serve the Jinja2 index.html template."""
    return templates.TemplateResponse(request, "index.html")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Validate and preview an uploaded CSV file.

    Returns:
        UploadResponse — columns, up to 10 preview rows, total row count.

    HTTP errors:
        400 — file missing or empty
        413 — file exceeds 50 MB, or CSV has > 500 000 rows
        422 — not a .csv, or CSV cannot be parsed
    """
    try:
        content = await _read_and_validate_file(file)
    except _ValidationError as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response(exc.detail),
        )

    try:
        df = csv_bytes_to_dataframe(content)
    except CSVParseError:
        return JSONResponse(
            status_code=422,
            content=error_response("File could not be parsed as CSV"),
        )

    # Rule 5 — row count limit
    if len(df) > MAX_CSV_ROWS:
        return JSONResponse(
            status_code=413,
            content=error_response(
                f"CSV has {len(df):,} rows; the limit is {MAX_CSV_ROWS:,} rows"
            ),
        )

    columns: list[str] = list(df.columns)
    preview_rows: list[dict] = [
        _sanitise_row(row) for row in df.head(10).to_dict(orient="records")
    ]

    return UploadResponse(
        columns=columns,
        preview_rows=preview_rows,
        total_rows=len(df),
    )


@app.post("/detect")
async def detect_anomalies(
    file: UploadFile = File(...),
    contamination: float = Form(default=CONTAMINATION),
):
    """Run the full preprocessing + Isolation Forest pipeline on an uploaded CSV.

    Form fields:
        file          — the CSV file (required)
        contamination — expected anomaly fraction, float in [0.01, 0.30],
                        default 0.05

    Returns:
        DetectResponse — augmented rows, summary statistics, feature report.

    HTTP errors:
        400 — file missing or empty
        413 — file exceeds 50 MB, or CSV has > 500 000 rows
        422 — not a .csv, CSV cannot be parsed, insufficient features,
              or contamination outside [0.01, 0.30]
        500 — unexpected model error
    """
    # Rule 6 — validate contamination
    if not (CONTAMINATION_MIN <= contamination <= CONTAMINATION_MAX):
        return JSONResponse(
            status_code=422,
            content=error_response(
                f"contamination must be between {CONTAMINATION_MIN} and "
                f"{CONTAMINATION_MAX} (got {contamination})"
            ),
        )

    try:
        content = await _read_and_validate_file(file)
    except _ValidationError as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response(exc.detail),
        )

    try:
        df = csv_bytes_to_dataframe(content)
    except CSVParseError:
        return JSONResponse(
            status_code=422,
            content=error_response("File could not be parsed as CSV"),
        )

    # Rule 5 — row count limit
    if len(df) > MAX_CSV_ROWS:
        return JSONResponse(
            status_code=413,
            content=error_response(
                f"CSV has {len(df):,} rows; the limit is {MAX_CSV_ROWS:,} rows"
            ),
        )

    try:
        X, retained_columns, feature_report = preprocess(df)
    except PreprocessingError:
        return JSONResponse(
            status_code=422,
            content=error_response(
                "Insufficient usable feature columns (minimum 2 required)"
            ),
        )

    if len(retained_columns) < 2:
        return JSONResponse(
            status_code=422,
            content=error_response(
                "Insufficient usable feature columns (minimum 2 required)"
            ),
        )

    try:
        labels, scores = run_isolation_forest(
            X, contamination=contamination, random_state=RANDOM_STATE
        )
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(
            status_code=500,
            content=error_response(f"Model error: {exc}"),
        )

    # Compute relative severity from the flagged-rows score distribution.
    severities: list[str] = assign_severities(scores, labels)

    # Augment original rows
    original_columns: list[str] = list(df.columns)
    result_columns: list[str] = original_columns + [
        "anomaly_label",
        "anomaly_score",
        "severity",
    ]

    raw_rows: list[dict] = df.to_dict(orient="records")
    augmented_rows: list[dict] = []

    for i, raw_row in enumerate(raw_rows):
        row = _sanitise_row(raw_row)
        row["anomaly_label"] = int(labels[i])
        row["anomaly_score"] = round(float(scores[i]), 6)
        row["severity"] = severities[i]
        augmented_rows.append(row)

    # Summary statistics
    total: int = len(augmented_rows)
    anomalies: int = sum(1 for r in augmented_rows if r["anomaly_label"] == -1)
    high: int = sum(
        1 for r in augmented_rows
        if r["anomaly_label"] == -1 and r["severity"] == "HIGH"
    )
    medium: int = sum(
        1 for r in augmented_rows
        if r["anomaly_label"] == -1 and r["severity"] == "MEDIUM"
    )
    low: int = sum(
        1 for r in augmented_rows
        if r["anomaly_label"] == -1 and r["severity"] == "LOW"
    )

    return DetectResponse(
        columns=result_columns,
        rows=augmented_rows,
        summary=SummaryStats(
            total=total,
            anomalies=anomalies,
            high=high,
            medium=medium,
            low=low,
        ),
        feature_report=[dict(entry) for entry in feature_report],
    )


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a generic 500 to the client; log the real exception server-side."""
    logger.exception("Unhandled exception on %s %s", request.method, request.url)
    return JSONResponse(
        status_code=500,
        content=error_response("Internal server error"),
    )
