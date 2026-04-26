"""
Preprocessing pipeline for the AI Log Anomaly Detection Dashboard.

Exposes:
    preprocess(df)              → (np.ndarray, list[str])
    csv_bytes_to_dataframe(b)   → pd.DataFrame
    PreprocessingError          — raised when the pipeline cannot produce usable features
    CSVParseError               — raised when raw CSV bytes cannot be parsed
"""

from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

logger = logging.getLogger(__name__)


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
# Main preprocessing pipeline
# ---------------------------------------------------------------------------


def preprocess(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Clean, encode, and normalise a raw log DataFrame.

    Pipeline steps
    --------------
    1. Raise PreprocessingError immediately if the DataFrame has no rows.
    2. Detect column types: ``dtype == object`` → categorical; everything else → numeric.
    3. Fill NaN in categorical columns with ``"unknown"``; fill NaN in numeric
       columns with the column median.
    4. Label-encode each categorical column independently with
       ``sklearn.preprocessing.LabelEncoder``.
    5. Coerce all columns to numeric via ``pd.to_numeric(errors='coerce')``.
       Drop any column that still contains NaN after coercion (non-convertible).
    6. Drop any column with zero variance (all values identical).
    7. Raise PreprocessingError if no columns remain.
    8. Apply ``sklearn.preprocessing.MinMaxScaler`` to all remaining columns.

    Returns
    -------
    feature_matrix : np.ndarray
        Shape ``(n_rows, n_retained_cols)``, all values in ``[0.0, 1.0]``.
    retained_columns : list[str]
        Names of the columns that survived the pipeline, in order.

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

    # Work on a copy so we never mutate the caller's DataFrame.
    df = df.copy()

    # ------------------------------------------------------------------
    # Step 2 — detect column types
    # ------------------------------------------------------------------
    # A column is categorical if its dtype is object AND it contains at least
    # one actual string value among the non-null entries.  Columns whose dtype
    # is object solely because they hold mixed types or nulls alongside numeric
    # values are treated as numeric so that the median fill path is used.
    def _is_categorical(series: pd.Series) -> bool:
        if series.dtype != object:
            return False
        non_null = series.dropna()
        if non_null.empty:
            return False
        return any(isinstance(v, str) for v in non_null)

    categorical_cols: list[str] = [c for c in df.columns if _is_categorical(df[c])]
    numeric_cols: list[str] = [c for c in df.columns if c not in categorical_cols]

    # ------------------------------------------------------------------
    # Step 3 — fill NaN values
    # ------------------------------------------------------------------
    # Categorical: fill with the sentinel string "unknown"
    if categorical_cols:
        df[categorical_cols] = df[categorical_cols].fillna("unknown")

    # Numeric: fill with the column median.
    # For a single-row DataFrame the median equals the single value, which is
    # the correct behaviour (no special-casing needed).
    for col in numeric_cols:
        # Convert to a numeric series first so that median() works even when
        # the column has object dtype due to mixed types or nulls.
        numeric_series = pd.to_numeric(df[col], errors="coerce")
        median_val = numeric_series.median()
        if pd.isna(median_val):
            # All values are non-numeric; leave as-is and let step 5 drop it.
            pass
        else:
            df[col] = df[col].fillna(median_val)

    # ------------------------------------------------------------------
    # Step 4 — label-encode categorical columns
    # ------------------------------------------------------------------
    # Encoders are local to this call; they are not persisted.
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # ------------------------------------------------------------------
    # Step 5 — coerce to numeric and drop non-convertible columns
    # ------------------------------------------------------------------
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    cols_with_nan = [c for c in df.columns if df[c].isna().any()]
    if cols_with_nan:
        logger.info(
            "Dropping %d column(s) that could not be coerced to numeric: %s",
            len(cols_with_nan),
            cols_with_nan,
        )
        df.drop(columns=cols_with_nan, inplace=True)

    # ------------------------------------------------------------------
    # Step 6 — drop zero-variance columns
    # ------------------------------------------------------------------
    # A column has zero variance when all its values are identical.  For a
    # single-row DataFrame every column trivially has one unique value, so we
    # skip this check to avoid dropping all columns and raising an error for
    # valid single-row inputs (requirement 7.5).
    if len(df) > 1:
        zero_var_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
        if zero_var_cols:
            logger.info(
                "Dropping %d zero-variance column(s): %s",
                len(zero_var_cols),
                zero_var_cols,
            )
            df.drop(columns=zero_var_cols, inplace=True)

    # ------------------------------------------------------------------
    # Step 7 — guard: no usable columns remain
    # ------------------------------------------------------------------
    if df.shape[1] == 0:
        raise PreprocessingError(
            "No usable feature columns remain after preprocessing"
        )

    # ------------------------------------------------------------------
    # Step 8 — min-max normalisation
    # ------------------------------------------------------------------
    retained_columns: list[str] = list(df.columns)
    scaler = MinMaxScaler()
    feature_matrix: np.ndarray = scaler.fit_transform(df.values.astype(float))

    return feature_matrix, retained_columns
