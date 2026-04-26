"""
Shared utilities for the AI Log Anomaly Detection Dashboard.

Exposes:
    severity_label(score)   → str  ("HIGH", "MEDIUM", or "LOW")
    error_response(message) → dict ({"error": message})

Note: csv_bytes_to_dataframe and CSVParseError live in preprocess.py.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Severity threshold constants (mirrors design.md)
# ---------------------------------------------------------------------------

SEVERITY_HIGH_THRESHOLD: float = -0.2   # score < this → HIGH
SEVERITY_MEDIUM_THRESHOLD: float = -0.1  # score < this (and >= HIGH) → MEDIUM
                                          # score >= this → LOW


def severity_label(score: float) -> str:
    """Map an anomaly score to a severity string.

    Thresholds
    ----------
    score < -0.2          → "HIGH"
    -0.2 <= score < -0.1  → "MEDIUM"
    score >= -0.1         → "LOW"

    Parameters
    ----------
    score : float
        The decision_function output from IsolationForest.
        More negative values indicate stronger anomalies.

    Returns
    -------
    str
        One of ``"HIGH"``, ``"MEDIUM"``, or ``"LOW"``.
    """
    if score < SEVERITY_HIGH_THRESHOLD:
        return "HIGH"
    if score < SEVERITY_MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def error_response(message: str) -> dict:
    """Return a standardised JSON-serialisable error body.

    Parameters
    ----------
    message : str
        Human-readable description of the error.

    Returns
    -------
    dict
        ``{"error": message}``
    """
    return {"error": message}
