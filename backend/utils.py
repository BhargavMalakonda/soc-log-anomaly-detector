"""
Shared utilities for the AI Log Anomaly Detection Dashboard.

Exposes:
    assign_severities(scores, labels) → list[str]
        Relative severity ("HIGH", "MEDIUM", "LOW") derived from the
        distribution of anomaly scores among the flagged rows only.

    error_response(message) → dict  ({"error": message})

Note: csv_bytes_to_dataframe and CSVParseError live in preprocess.py.
"""

from __future__ import annotations

import numpy as np


def assign_severities(
    scores: np.ndarray,
    labels: np.ndarray,
) -> list[str]:
    """Assign relative severity labels to every row.

    Severity is derived from the *distribution of scores among the flagged
    (anomalous) rows only*, rather than from hardcoded global thresholds.
    This keeps HIGH/MEDIUM/LOW meaningful regardless of the absolute scale
    of the IsolationForest decision function on a given dataset.

    Algorithm
    ---------
    1. Collect the anomaly scores for all rows where ``label == -1``.
    2. Sort those scores ascending (most negative = worst anomaly first).
    3. Split into thirds by index position:
         - bottom third  → HIGH   (most anomalous)
         - middle third  → MEDIUM
         - top third     → LOW    (least anomalous among flagged)
    4. For rows where ``label == 1`` (normal), severity is ``""``.

    Tercile boundaries are computed as:
        low_boundary  = scores_sorted[n // 3]
        high_boundary = scores_sorted[2 * n // 3]
    where n = number of flagged rows.

    Edge cases
    ----------
    0 flagged rows:
        All rows are normal; every entry in the returned list is ``""``.

    1 flagged row:
        That single row receives ``"HIGH"``.

    2 flagged rows:
        The lower score (more anomalous) → ``"HIGH"``; the other → ``"LOW"``.
        ``"MEDIUM"`` is not used when fewer than 3 rows are flagged.

    Parameters
    ----------
    scores : np.ndarray, shape (n_samples,)
        ``decision_function`` output from IsolationForest.
        More negative → more anomalous.
    labels : np.ndarray of int, shape (n_samples,)
        ``-1`` for anomalous rows, ``1`` for normal rows.

    Returns
    -------
    list[str]
        Length ``n_samples``.  Each element is ``"HIGH"``, ``"MEDIUM"``,
        ``"LOW"`` for anomalous rows, or ``""`` for normal rows.
    """
    n = len(scores)
    result: list[str] = [""] * n

    # Indices of flagged rows, sorted by score ascending (worst first).
    flagged_idx: list[int] = [
        i for i in range(n) if int(labels[i]) == -1
    ]
    flagged_idx.sort(key=lambda i: float(scores[i]))

    k = len(flagged_idx)

    if k == 0:
        return result

    if k == 1:
        result[flagged_idx[0]] = "HIGH"
        return result

    if k == 2:
        result[flagged_idx[0]] = "HIGH"
        result[flagged_idx[1]] = "LOW"
        return result

    # General case: split sorted flagged indices into thirds by position.
    third = k // 3
    # remainder rows go to the middle and top thirds so the bottom third
    # never grows unexpectedly large.
    for pos, idx in enumerate(flagged_idx):
        if pos < third:
            result[idx] = "HIGH"
        elif pos < 2 * third:
            result[idx] = "MEDIUM"
        else:
            result[idx] = "LOW"

    return result


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
