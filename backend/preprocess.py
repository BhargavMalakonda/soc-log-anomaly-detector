"""
Preprocessing pipeline for the AI Log Anomaly Detection Dashboard.

Exposes:
    preprocess(df)              → (np.ndarray, list[str], list[FeatureReportEntry])
    csv_bytes_to_dataframe(b)   → pd.DataFrame
    PreprocessingError          — raised when the pipeline cannot produce usable features
    CSVParseError               — raised when raw CSV bytes cannot be parsed
"""

from __future__ import annotations

import io
import ipaddress
import logging
from typing import TypedDict

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

# Fraction of sampled values that must parse successfully for a column to be
# classified as IP-like or datetime-like.
_PARSE_THRESHOLD: float = 0.80

# Maximum number of non-null values sampled when probing a column.
_SAMPLE_SIZE: int = 50

# A column whose unique-value ratio exceeds this is treated as high-cardinality
# free text and dropped (unless it was already classified as IP or datetime).
_HIGH_CARDINALITY_RATIO: float = 0.95


# ---------------------------------------------------------------------------
# Public TypedDict for the feature report
# ---------------------------------------------------------------------------


class FeatureReportEntry(TypedDict):
    """One entry in the feature_report list returned by preprocess()."""

    original_column: str
    action: str          # "kept", "ip_encoded", "datetime_encoded",
                         # "freq_encoded", "dropped"
    produced_features: list[str]   # empty list when action == "dropped"
    drop_reason: str     # empty string when not dropped


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class PreprocessingError(Exception):
    """Raised when the preprocessing pipeline cannot produce usable features."""


class CSVParseError(Exception):
    """Raised when raw CSV bytes cannot be parsed into a DataFrame."""


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def csv_bytes_to_dataframe(content: bytes) -> pd.DataFrame:
    """Parse raw CSV bytes into a DataFrame.

    Raises:
        CSVParseError: if the bytes cannot be decoded or parsed as CSV.
    """
    try:
        return pd.read_csv(io.BytesIO(content))
    except Exception as exc:  # noqa: BLE001
        raise CSVParseError(f"Could not parse CSV content: {exc}") from exc


# ---------------------------------------------------------------------------
# Column-type detection helpers
# ---------------------------------------------------------------------------


def _is_ip_like(series: pd.Series) -> bool:
    """Return True if > 80 % of a sample of non-null values parse as IP addresses."""
    non_null = series.dropna()
    if non_null.empty:
        return False
    sample = non_null.head(_SAMPLE_SIZE)
    hits = 0
    for val in sample:
        try:
            ipaddress.ip_address(str(val).strip())
            hits += 1
        except ValueError:
            pass
    return (hits / len(sample)) > _PARSE_THRESHOLD


def _is_datetime_like(series: pd.Series) -> bool:
    """Return True if > 80 % of non-null values parse as datetimes (utc=True)."""
    non_null = series.dropna()
    if non_null.empty:
        return False
    # format="mixed" lets pandas infer per-value (handles ISO-8601, DD/MM/YYYY,
    # MM/DD/YYYY, etc.) without raising the dayfirst-ambiguity UserWarning.
    parsed = pd.to_datetime(non_null, errors="coerce", utc=True, format="mixed")
    success_rate = parsed.notna().sum() / len(non_null)
    return success_rate > _PARSE_THRESHOLD


# ---------------------------------------------------------------------------
# Feature-engineering helpers
# ---------------------------------------------------------------------------


def _freq_encode(series: pd.Series) -> pd.Series:
    """Replace each value with its normalised frequency (0..1) in the series."""
    counts = series.value_counts(normalize=True, dropna=False)
    return series.map(counts).astype(float)


def _encode_ip(col: str, series: pd.Series) -> dict[str, pd.Series]:
    """Produce a single frequency-encoded feature for an IP column."""
    return {f"{col}_freq": _freq_encode(series.astype(str))}


def _encode_datetime(col: str, series: pd.Series) -> dict[str, pd.Series]:
    """Produce hour-of-day, day-of-week, and epoch-seconds features."""
    # format="mixed" lets pandas infer per-value and avoids the dayfirst
    # UserWarning for non-ISO formats such as "DD/MM/YYYY HH:MM".
    parsed = pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    n = len(series)
    hour = parsed.dt.hour.astype("float64")
    dow = parsed.dt.dayofweek.astype("float64")
    epoch = parsed.astype("int64") // 10 ** 9  # nanoseconds → seconds
    # For rows where parsing failed, epoch will be a large negative sentinel
    # from NaT; convert those to NaN so they are handled by the median-fill step.
    epoch_float = epoch.astype("float64")
    nat_mask = parsed.isna()
    epoch_float[nat_mask] = float("nan")
    hour[nat_mask] = float("nan")
    dow[nat_mask] = float("nan")
    return {
        f"{col}_hour": hour,
        f"{col}_dow": dow,
        f"{col}_epoch": epoch_float,
    }


# ---------------------------------------------------------------------------
# Main preprocessing pipeline
# ---------------------------------------------------------------------------


def preprocess(
    df: pd.DataFrame,
) -> tuple[np.ndarray, list[str], list[FeatureReportEntry]]:
    """Clean, feature-engineer, and normalise a raw log DataFrame.

    Pipeline steps
    --------------
    FE. Feature engineering (NEW — runs first):
        For each column in the original DataFrame, auto-detect its semantic type
        and expand it into numeric features:

        - Numeric dtype       → kept as-is (no change).
        - IP-like string      → replaced by ``<col>_freq`` (normalised frequency).
        - Datetime-like string → replaced by ``<col>_hour``, ``<col>_dow``,
                                ``<col>_epoch`` (epoch seconds, UTC).
        - Entirely null       → dropped.
        - Constant (1 unique value) → dropped.
        - High-cardinality free text (unique ratio > 0.95, non-numeric) → dropped.
        - Any other string    → replaced by ``<col>_freq`` (frequency encoding).

    1. Raise PreprocessingError immediately if the DataFrame has no rows.
    2–5. NaN fill (median for numeric), coerce to numeric, drop non-convertible.
    6. Drop zero-variance columns (skipped for single-row DataFrames).
    7. Raise PreprocessingError if no columns remain.
    8. MinMax normalisation.

    Returns
    -------
    feature_matrix : np.ndarray
        Shape ``(n_rows, n_retained_cols)``, all values in ``[0.0, 1.0]``.
    retained_columns : list[str]
        Names of the engineered columns that survived the full pipeline.
    feature_report : list[FeatureReportEntry]
        One entry per original column documenting what was done to it.

    Raises
    ------
    PreprocessingError
        If the DataFrame has no rows, or if no usable feature columns remain
        after preprocessing.
    """
    # ------------------------------------------------------------------
    # Step 1 — guard: empty DataFrame
    # ------------------------------------------------------------------
    if df.empty or len(df) == 0:
        raise PreprocessingError("No data rows to process")

    # Work on the raw copy; we will build the engineered frame from scratch.
    raw = df.copy()
    n_rows = len(raw)

    feature_report: list[FeatureReportEntry] = []
    # Ordered dict of engineered column name → Series, assembled column-by-column.
    engineered: dict[str, pd.Series] = {}

    # ------------------------------------------------------------------
    # Step FE — per-column feature engineering
    # ------------------------------------------------------------------
    for col in raw.columns:
        series = raw[col]
        non_null = series.dropna()

        # ── Entirely null ──────────────────────────────────────────────
        if non_null.empty:
            feature_report.append(
                FeatureReportEntry(
                    original_column=col,
                    action="dropped",
                    produced_features=[],
                    drop_reason="entirely null",
                )
            )
            logger.info("Dropping column %r: entirely null", col)
            continue

        # ── Numeric dtype → keep as-is ────────────────────────────────
        if pd.api.types.is_numeric_dtype(series):
            feature_report.append(
                FeatureReportEntry(
                    original_column=col,
                    action="kept",
                    produced_features=[col],
                    drop_reason="",
                )
            )
            engineered[col] = series.copy()
            continue

        # At this point the column has object (or other non-numeric) dtype.
        # Cast non-null values to str for type-detection probes.

        # ── Constant column (1 unique value, already non-null-checked) ─
        if series.nunique(dropna=True) <= 1:
            feature_report.append(
                FeatureReportEntry(
                    original_column=col,
                    action="dropped",
                    produced_features=[],
                    drop_reason="constant (single unique value)",
                )
            )
            logger.info("Dropping column %r: constant", col)
            continue

        # ── IP-like ───────────────────────────────────────────────────
        if _is_ip_like(series):
            new_cols = _encode_ip(col, series)
            feature_report.append(
                FeatureReportEntry(
                    original_column=col,
                    action="ip_encoded",
                    produced_features=list(new_cols.keys()),
                    drop_reason="",
                )
            )
            engineered.update(new_cols)
            logger.info("IP-encoded column %r → %s", col, list(new_cols.keys()))
            continue

        # ── Datetime-like ─────────────────────────────────────────────
        if _is_datetime_like(series):
            new_cols = _encode_datetime(col, series)
            feature_report.append(
                FeatureReportEntry(
                    original_column=col,
                    action="datetime_encoded",
                    produced_features=list(new_cols.keys()),
                    drop_reason="",
                )
            )
            engineered.update(new_cols)
            logger.info(
                "Datetime-encoded column %r → %s", col, list(new_cols.keys())
            )
            continue

        # ── High-cardinality free text ────────────────────────────────
        n_unique = series.nunique(dropna=True)
        unique_ratio = n_unique / n_rows if n_rows > 0 else 0.0
        if unique_ratio > _HIGH_CARDINALITY_RATIO:
            feature_report.append(
                FeatureReportEntry(
                    original_column=col,
                    action="dropped",
                    produced_features=[],
                    drop_reason=(
                        f"high-cardinality free text "
                        f"(unique ratio {unique_ratio:.2f} > {_HIGH_CARDINALITY_RATIO})"
                    ),
                )
            )
            logger.info(
                "Dropping column %r: high-cardinality (unique ratio %.2f)",
                col,
                unique_ratio,
            )
            continue

        # ── Generic string → frequency encoding ───────────────────────
        freq_col = f"{col}_freq"
        new_cols = {freq_col: _freq_encode(series.astype(str))}
        feature_report.append(
            FeatureReportEntry(
                original_column=col,
                action="freq_encoded",
                produced_features=[freq_col],
                drop_reason="",
            )
        )
        engineered.update(new_cols)
        logger.info("Freq-encoded column %r → %r", col, freq_col)

    # ------------------------------------------------------------------
    # Build the engineered DataFrame
    # ------------------------------------------------------------------
    if not engineered:
        raise PreprocessingError(
            "No usable feature columns remain after feature engineering"
        )

    eng_df = pd.DataFrame(engineered, index=raw.index)

    # ------------------------------------------------------------------
    # Step 2 — fill NaN in numeric columns with column median
    # ------------------------------------------------------------------
    for col in eng_df.columns:
        if eng_df[col].isna().any():
            median_val = pd.to_numeric(eng_df[col], errors="coerce").median()
            if not pd.isna(median_val):
                eng_df[col] = eng_df[col].fillna(median_val)

    # ------------------------------------------------------------------
    # Step 3 — coerce to numeric and drop non-convertible columns
    # ------------------------------------------------------------------
    for col in eng_df.columns:
        eng_df[col] = pd.to_numeric(eng_df[col], errors="coerce")

    cols_with_nan = [c for c in eng_df.columns if eng_df[c].isna().any()]
    if cols_with_nan:
        logger.info(
            "Dropping %d engineered column(s) that could not be coerced to numeric: %s",
            len(cols_with_nan),
            cols_with_nan,
        )
        eng_df.drop(columns=cols_with_nan, inplace=True)

    # ------------------------------------------------------------------
    # Step 4 — drop zero-variance columns
    # (skipped for single-row DataFrames — requirement 7.5)
    # ------------------------------------------------------------------
    if len(eng_df) > 1:
        zero_var_cols = [
            c for c in eng_df.columns if eng_df[c].nunique(dropna=False) <= 1
        ]
        if zero_var_cols:
            logger.info(
                "Dropping %d zero-variance engineered column(s): %s",
                len(zero_var_cols),
                zero_var_cols,
            )
            eng_df.drop(columns=zero_var_cols, inplace=True)

    # ------------------------------------------------------------------
    # Step 5 — guard: no usable columns remain
    # ------------------------------------------------------------------
    if eng_df.shape[1] == 0:
        raise PreprocessingError(
            "No usable feature columns remain after preprocessing"
        )

    # ------------------------------------------------------------------
    # Step 6 — min-max normalisation
    # ------------------------------------------------------------------
    retained_columns: list[str] = list(eng_df.columns)
    scaler = MinMaxScaler()
    feature_matrix: np.ndarray = scaler.fit_transform(
        eng_df.values.astype(float)
    )

    return feature_matrix, retained_columns, feature_report
