"""
Anomaly detection module for the AI Log Anomaly Detection Dashboard.

Exposes:
    CONTAMINATION   — default expected anomaly fraction (0.05)
    RANDOM_STATE    — fixed seed for reproducibility (42)
    run_isolation_forest(X, contamination, random_state) → (labels, scores)
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

CONTAMINATION: float = 0.05
RANDOM_STATE: int = 42


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_isolation_forest(
    X: np.ndarray,
    contamination: float = CONTAMINATION,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit an IsolationForest on *X* and return per-row labels and scores.

    A new model is instantiated and fitted on every call — no state is
    persisted between requests.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape ``(n_samples, n_features)``.
        Must have at least 1 row; 0 rows raises ``ValueError``.
    contamination : float, optional
        Expected proportion of anomalies in the dataset.
        Defaults to :data:`CONTAMINATION` (``0.05``).
    random_state : int, optional
        Random seed for reproducibility.
        Defaults to :data:`RANDOM_STATE` (``42``).

    Returns
    -------
    labels : np.ndarray of int, shape (n_samples,)
        ``-1`` for anomalous entries, ``1`` for normal entries.
    scores : np.ndarray of float, shape (n_samples,)
        ``decision_function`` output — more negative means more anomalous.

    Raises
    ------
    ValueError
        If *X* has 0 rows.
    """
    if X.shape[0] == 0:
        raise ValueError("Cannot run anomaly detection on empty dataset")

    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
    )

    labels: np.ndarray = model.fit_predict(X)
    scores: np.ndarray = model.decision_function(X)

    return labels, scores
