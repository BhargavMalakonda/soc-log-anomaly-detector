"""
FastAPI application for the AI Log Anomaly Detection Dashboard.

Routes:
    GET  /          → serves index.html (Jinja2 template)
    POST /upload    → validates + parses CSV → preview JSON
    POST /detect    → preprocess + detect → results JSON
    GET  /static/*  → serves CSS / JS static files

Validation rules (in order, applied to both /upload and /detect):
    1. File present and non-empty (0 bytes)  → HTTP 400
    2. File size > 50 MB                     → HTTP 413
    3. Extension is not .csv                 → HTTP 422
    4. CSV cannot be parsed                  → HTTP 422
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from backend.model import CONTAMINATION, RANDOM_STATE, run_isolation_forest
from backend.preprocess import CSVParseError, PreprocessingError, csv_bytes_to_dataframe, preprocess
from backend.utils import error_response, severity_label

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

# Resolve the project root: ai-log-anomaly-dashboard/
# __file__ is ai-log-anomaly-dashboard/backend/main.py
# .parent       → ai-log-anomaly-dashboard/backend/
# .parent.parent → ai-log-anomaly-dashboard/
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_FILE_SIZE_BYTES: int = 50 * 1024 * 1024  # 50 MB

# ---------------------------------------------------------------------------
# Task 7.1 — FastAPI app, static files, templates
# ---------------------------------------------------------------------------

app = FastAPI(title="AI Log Anomaly Detection Dashboard")

# CORS middleware — allow all origins for local development (Task 7.5)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files at /static → frontend/static/
app.mount(
    "/static",
    StaticFiles(directory=str(PROJECT_ROOT / "frontend" / "static")),
    name="static",
)

# Jinja2 templates → frontend/templates/
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "frontend" / "templates"))

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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_python_native(value: Any) -> Any:
    """Convert numpy scalar types and NaN/Inf to JSON-serialisable Python types."""
    if value is None:
        return None
    # Handle numpy integer types
    if isinstance(value, (np.integer,)):
        return int(value)
    # Handle numpy floating types
    if isinstance(value, (np.floating,)):
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    # Handle numpy bool
    if isinstance(value, np.bool_):
        return bool(value)
    # Handle Python float NaN / Inf
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def _sanitise_row(row: dict) -> dict:
    """Return a copy of *row* with all values converted to JSON-safe Python types."""
    return {k: _to_python_native(v) for k, v in row.items()}


async def _read_and_validate_file(file: UploadFile) -> bytes:
    """Read the uploaded file and apply all validation rules.

    Returns the raw file bytes on success.

    Raises:
        JSONResponse with the appropriate HTTP status on any validation failure.
    """
    # Rule 1 — file must be present and non-empty
    if file is None or file.filename == "":
        raise _ValidationError(
            status_code=400,
            detail="No file provided",
        )

    content: bytes = await file.read()

    if len(content) == 0:
        raise _ValidationError(
            status_code=400,
            detail="Uploaded file is empty",
        )

    # Rule 2 — file size must not exceed 50 MB
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise _ValidationError(
            status_code=413,
            detail="File exceeds the 50 MB size limit",
        )

    # Rule 3 — extension must be .csv
    filename: str = file.filename or ""
    if not filename.lower().endswith(".csv"):
        raise _ValidationError(
            status_code=422,
            detail="Only .csv files are accepted",
        )

    return content


class _ValidationError(Exception):
    """Internal sentinel used to short-circuit validation in route handlers."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


# ---------------------------------------------------------------------------
# Task 7.2 — GET / route
# ---------------------------------------------------------------------------


@app.get("/")
async def index(request: Request):
    """Serve the Jinja2 index.html template."""
    return templates.TemplateResponse(request, "index.html")


# ---------------------------------------------------------------------------
# Task 7.3 — POST /upload route
# ---------------------------------------------------------------------------


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Validate and preview an uploaded CSV file.

    Returns:
        UploadResponse with columns, up to 10 preview rows, and total row count.

    Raises:
        HTTP 400 — file missing or empty
        HTTP 413 — file exceeds 50 MB
        HTTP 422 — not a .csv file, or CSV cannot be parsed
    """
    # Validate and read file bytes
    try:
        content = await _read_and_validate_file(file)
    except _ValidationError as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response(exc.detail),
        )

    # Rule 4 — attempt CSV parsing
    try:
        df = csv_bytes_to_dataframe(content)
    except CSVParseError:
        return JSONResponse(
            status_code=422,
            content=error_response("File could not be parsed as CSV"),
        )

    columns: list[str] = list(df.columns)
    total_rows: int = len(df)

    # Build preview rows (up to 10), sanitising all values
    preview_rows: list[dict] = [
        _sanitise_row(row) for row in df.head(10).to_dict(orient="records")
    ]

    return UploadResponse(
        columns=columns,
        preview_rows=preview_rows,
        total_rows=total_rows,
    )


# ---------------------------------------------------------------------------
# Task 7.4 — POST /detect route
# ---------------------------------------------------------------------------


@app.post("/detect")
async def detect_anomalies(file: UploadFile = File(...)):
    """Run the full preprocessing + Isolation Forest pipeline on an uploaded CSV.

    Returns:
        DetectResponse with augmented rows and summary statistics.

    Raises:
        HTTP 400 — file missing or empty
        HTTP 413 — file exceeds 50 MB
        HTTP 422 — not a .csv file, CSV cannot be parsed, or insufficient features
        HTTP 500 — model error
    """
    # Re-validate the file (same rules as /upload)
    try:
        content = await _read_and_validate_file(file)
    except _ValidationError as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response(exc.detail),
        )

    # Parse CSV
    try:
        df = csv_bytes_to_dataframe(content)
    except CSVParseError:
        return JSONResponse(
            status_code=422,
            content=error_response("File could not be parsed as CSV"),
        )

    # Preprocess
    try:
        X, retained_columns = preprocess(df)
    except PreprocessingError:
        return JSONResponse(
            status_code=422,
            content=error_response(
                "Insufficient usable feature columns (minimum 2 required)"
            ),
        )

    # Guard: require at least 2 retained columns
    if len(retained_columns) < 2:
        return JSONResponse(
            status_code=422,
            content=error_response(
                "Insufficient usable feature columns (minimum 2 required)"
            ),
        )

    # Run Isolation Forest
    try:
        labels, scores = run_isolation_forest(X)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(
            status_code=500,
            content=error_response(f"Model error: {exc}"),
        )

    # Augment original rows with detection results
    original_columns: list[str] = list(df.columns)
    result_columns: list[str] = original_columns + [
        "anomaly_label",
        "anomaly_score",
        "severity",
    ]

    raw_rows: list[dict] = df.to_dict(orient="records")
    augmented_rows: list[dict] = []

    for i, raw_row in enumerate(raw_rows):
        label: int = int(labels[i])
        score: float = round(float(scores[i]), 6)
        sev: str = severity_label(score)

        row = _sanitise_row(raw_row)
        row["anomaly_label"] = label
        row["anomaly_score"] = score
        row["severity"] = sev
        augmented_rows.append(row)

    # Build summary statistics
    total: int = len(augmented_rows)
    anomalies: int = sum(1 for r in augmented_rows if r["anomaly_label"] == -1)
    high: int = sum(
        1 for r in augmented_rows if r["anomaly_label"] == -1 and r["severity"] == "HIGH"
    )
    medium: int = sum(
        1 for r in augmented_rows if r["anomaly_label"] == -1 and r["severity"] == "MEDIUM"
    )
    low: int = sum(
        1 for r in augmented_rows if r["anomaly_label"] == -1 and r["severity"] == "LOW"
    )

    summary = SummaryStats(
        total=total,
        anomalies=anomalies,
        high=high,
        medium=medium,
        low=low,
    )

    return DetectResponse(
        columns=result_columns,
        rows=augmented_rows,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Task 7.5 — Global exception handler
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch any unhandled exception and return HTTP 500 with a JSON error body."""
    return JSONResponse(
        status_code=500,
        content=error_response(str(exc)),
    )
