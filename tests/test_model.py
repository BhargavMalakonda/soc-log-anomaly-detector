"""
Unit tests for backend/model.py — run_isolation_forest.

Covers:
  - Labels contain only -1 and 1                          (Req 3.8)
  - Anomaly fraction approximates contamination           (Req 3.1)
  - Reproducibility: same inputs + random_state → same outputs (Req 3.1)
  - Empty dataset raises ValueError                       (edge case)
  - Single-column input works                             (edge case)
  - Single-row input works                                (Req 7.5)
  - Custom contamination is respected                     (Req 3.1)
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.model import CONTAMINATION, RANDOM_STATE, run_isolation_forest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_X(n_rows: int = 200, n_cols: int = 4, seed: int = 0) -> np.ndarray:
    """Return a reproducible float feature matrix."""
    rng = np.random.default_rng(seed)
    return rng.random((n_rows, n_cols)).astype(float)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunIsolationForest:
    """Tests for run_isolation_forest."""

    def test_labels_contain_only_minus_one_and_one(self):
        """Labels must be exactly -1 (anomalous) or 1 (normal)."""
        X = _make_X()
        labels, _ = run_isolation_forest(X)
        unique = set(labels.tolist())
        assert unique <= {-1, 1}, f"Unexpected label values: {unique}"

    def test_scores_are_floats(self):
        """Scores array must be float dtype."""
        X = _make_X()
        _, scores = run_isolation_forest(X)
        assert np.issubdtype(scores.dtype, np.floating)

    def test_output_shapes_match_input(self):
        """Both output arrays must have length equal to number of input rows."""
        X = _make_X(n_rows=150)
        labels, scores = run_isolation_forest(X)
        assert labels.shape == (150,)
        assert scores.shape == (150,)

    def test_anomaly_fraction_approximates_contamination(self):
        """Fraction of -1 labels should be within 5 pp of contamination."""
        n = 500
        X = _make_X(n_rows=n)
        labels, _ = run_isolation_forest(X, contamination=CONTAMINATION)
        fraction = (labels == -1).sum() / n
        assert abs(fraction - CONTAMINATION) <= 0.05, (
            f"Anomaly fraction {fraction:.3f} deviates more than 5 pp "
            f"from contamination {CONTAMINATION}"
        )

    def test_reproducibility_same_random_state(self):
        """Identical inputs with the same random_state must produce identical outputs."""
        X = _make_X(n_rows=200)
        labels_a, scores_a = run_isolation_forest(X, random_state=RANDOM_STATE)
        labels_b, scores_b = run_isolation_forest(X, random_state=RANDOM_STATE)
        np.testing.assert_array_equal(labels_a, labels_b)
        np.testing.assert_array_almost_equal(scores_a, scores_b)

    def test_different_random_states_may_differ(self):
        """Different random states should (in practice) produce different scores."""
        X = _make_X(n_rows=200)
        _, scores_42 = run_isolation_forest(X, random_state=42)
        _, scores_99 = run_isolation_forest(X, random_state=99)
        # It is astronomically unlikely that all scores are identical with
        # different seeds on a 200-row dataset.
        assert not np.allclose(scores_42, scores_99), (
            "Scores were identical for different random states — unexpected"
        )

    def test_empty_dataset_raises_value_error(self):
        """An empty feature matrix must raise ValueError."""
        X = np.empty((0, 4), dtype=float)
        with pytest.raises(ValueError, match="empty dataset"):
            run_isolation_forest(X)

    def test_single_column_input_works(self):
        """IsolationForest should handle a single-feature matrix without error."""
        X = _make_X(n_rows=100, n_cols=1)
        labels, scores = run_isolation_forest(X)
        assert labels.shape == (100,)
        assert scores.shape == (100,)
        assert set(labels.tolist()) <= {-1, 1}

    def test_single_row_input_works(self):
        """A single-row input must not raise an exception (Req 7.5)."""
        X = _make_X(n_rows=1, n_cols=4)
        labels, scores = run_isolation_forest(X)
        assert labels.shape == (1,)
        assert scores.shape == (1,)

    def test_custom_contamination_changes_anomaly_count(self):
        """Higher contamination should yield more -1 labels."""
        X = _make_X(n_rows=300)
        labels_low, _ = run_isolation_forest(X, contamination=0.05)
        labels_high, _ = run_isolation_forest(X, contamination=0.30)
        anomalies_low = (labels_low == -1).sum()
        anomalies_high = (labels_high == -1).sum()
        assert anomalies_high > anomalies_low, (
            f"Expected more anomalies with higher contamination: "
            f"low={anomalies_low}, high={anomalies_high}"
        )

    def test_labels_dtype_is_integer(self):
        """Labels array must have an integer dtype."""
        X = _make_X()
        labels, _ = run_isolation_forest(X)
        assert np.issubdtype(labels.dtype, np.integer)
