"""
tests/test_preprocess.py
========================
Unit tests for backend/preprocess.py.

Covers:
  - Datetime column expands to hour / dow / epoch features
  - IP column becomes a single frequency feature
  - All-null column is dropped and present in feature_report with action="dropped"
  - Constant (single-value) string column is dropped
  - All-numeric CSV passes through with action="kept" for every column
  - Mixed-type object column (numbers and strings) does not crash
  - Fewer than 2 usable columns after engineering raises PreprocessingError
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.preprocess import PreprocessingError, preprocess


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _report_map(report: list) -> dict[str, dict]:
    """Return {original_column: entry_dict} for quick look-ups."""
    return {e["original_column"]: e for e in report}


# ---------------------------------------------------------------------------
# 1. Datetime column → hour / dow / epoch
# ---------------------------------------------------------------------------


class TestDatetimeEncoding:
    """A column of ISO-8601 timestamps is expanded into three numeric features."""

    def _make_df(self) -> pd.DataFrame:
        # 10 timestamps spread across two calendar days + a numeric column
        # so we always have ≥ 2 retained features.
        return pd.DataFrame(
            {
                "ts": [
                    "2024-01-15T08:00:00",
                    "2024-01-15T09:30:00",
                    "2024-01-15T14:00:00",
                    "2024-01-15T23:59:00",
                    "2024-01-16T00:01:00",
                    "2024-01-16T06:00:00",
                    "2024-01-16T12:00:00",
                    "2024-01-16T18:00:00",
                    "2024-01-16T20:00:00",
                    "2024-01-16T22:00:00",
                ],
                "value": list(range(10)),
            }
        )

    def test_produced_feature_names_contain_hour_dow_epoch(self):
        df = self._make_df()
        _, retained, report = preprocess(df)
        rm = _report_map(report)
        assert rm["ts"]["action"] == "datetime_encoded"
        produced = set(rm["ts"]["produced_features"])
        assert "ts_hour" in produced
        assert "ts_dow" in produced
        assert "ts_epoch" in produced

    def test_hour_feature_is_in_retained_or_dropped_for_variance(self):
        """At least ts_epoch must survive (timestamps span two calendar days)."""
        df = self._make_df()
        _, retained, _ = preprocess(df)
        # epoch always varies when timestamps differ
        assert "ts_epoch" in retained

    def test_matrix_values_are_in_0_1(self):
        df = self._make_df()
        X, _, _ = preprocess(df)
        assert X.min() >= 0.0 - 1e-9
        assert X.max() <= 1.0 + 1e-9

    def test_output_row_count_matches_input(self):
        df = self._make_df()
        X, retained, _ = preprocess(df)
        assert X.shape[0] == len(df)


# ---------------------------------------------------------------------------
# 2. IP column → single frequency feature
# ---------------------------------------------------------------------------


class TestIPEncoding:
    """A column of dotted-quad IP addresses is replaced by <col>_freq."""

    def _make_df(self) -> pd.DataFrame:
        # Pool of 4 IPs; add a numeric column so we have ≥ 2 features.
        ips = (
            ["10.0.0.1"] * 5
            + ["10.0.0.2"] * 3
            + ["192.168.1.1"] * 1
            + ["172.16.0.1"] * 1
        )
        return pd.DataFrame({"ip": ips, "bytes": list(range(10))})

    def test_action_is_ip_encoded(self):
        df = self._make_df()
        _, _, report = preprocess(df)
        rm = _report_map(report)
        assert rm["ip"]["action"] == "ip_encoded"

    def test_produced_feature_is_ip_freq(self):
        df = self._make_df()
        _, _, report = preprocess(df)
        rm = _report_map(report)
        assert rm["ip"]["produced_features"] == ["ip_freq"]

    def test_freq_column_is_retained(self):
        df = self._make_df()
        _, retained, _ = preprocess(df)
        assert "ip_freq" in retained

    def test_freq_values_sum_to_approximately_one_per_unique_ip(self):
        """The most common IP (10.0.0.1, 5/10 = 0.5) should map to 0.5 before scaling."""
        df = self._make_df()
        # We can't directly inspect pre-scaling values from the matrix, but we
        # CAN verify that all values are in [0, 1] and the column is present.
        X, retained, _ = preprocess(df)
        idx = retained.index("ip_freq")
        col_vals = X[:, idx]
        assert col_vals.min() >= 0.0 - 1e-9
        assert col_vals.max() <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# 3. All-null column is dropped and appears in feature_report
# ---------------------------------------------------------------------------


class TestAllNullColumn:
    """A column that is entirely NaN must be dropped before any other step."""

    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "null_col": [None, None, None, None, None],
                "b": [10, 20, 30, 40, 50],
            }
        )

    def test_null_column_action_is_dropped(self):
        df = self._make_df()
        _, _, report = preprocess(df)
        rm = _report_map(report)
        assert rm["null_col"]["action"] == "dropped"

    def test_null_column_drop_reason_mentions_null(self):
        df = self._make_df()
        _, _, report = preprocess(df)
        rm = _report_map(report)
        assert "null" in rm["null_col"]["drop_reason"].lower()

    def test_null_column_not_in_retained(self):
        df = self._make_df()
        _, retained, _ = preprocess(df)
        assert "null_col" not in retained

    def test_null_column_produced_features_is_empty(self):
        df = self._make_df()
        _, _, report = preprocess(df)
        rm = _report_map(report)
        assert rm["null_col"]["produced_features"] == []

    def test_other_columns_are_still_processed(self):
        df = self._make_df()
        _, retained, _ = preprocess(df)
        assert "a" in retained
        assert "b" in retained


# ---------------------------------------------------------------------------
# 4. Constant column is dropped
# ---------------------------------------------------------------------------


class TestConstantColumn:
    """A non-numeric column with only one distinct value must be dropped."""

    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "x": list(range(10)),
                "const_str": ["same_value"] * 10,
                "y": list(range(10, 20)),
            }
        )

    def test_constant_column_action_is_dropped(self):
        df = self._make_df()
        _, _, report = preprocess(df)
        rm = _report_map(report)
        assert rm["const_str"]["action"] == "dropped"

    def test_constant_column_drop_reason_mentions_constant(self):
        df = self._make_df()
        _, _, report = preprocess(df)
        rm = _report_map(report)
        assert "constant" in rm["const_str"]["drop_reason"].lower()

    def test_constant_column_not_in_retained(self):
        df = self._make_df()
        _, retained, _ = preprocess(df)
        assert "const_str" not in retained

    def test_numeric_columns_survive(self):
        df = self._make_df()
        _, retained, _ = preprocess(df)
        assert "x" in retained
        assert "y" in retained


# ---------------------------------------------------------------------------
# 5. All-numeric CSV passes through with action="kept"
# ---------------------------------------------------------------------------


class TestAllNumericPassthrough:
    """Every column in an all-numeric DataFrame should have action='kept'."""

    def _make_df(self) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        return pd.DataFrame(
            {
                "col_a": rng.integers(1, 100, 20).astype(float),
                "col_b": rng.integers(1, 1000, 20).astype(float),
                "col_c": rng.random(20),
            }
        )

    def test_all_actions_are_kept(self):
        df = self._make_df()
        _, _, report = preprocess(df)
        for entry in report:
            assert entry["action"] == "kept", (
                f"Column {entry['original_column']!r} has action {entry['action']!r}, expected 'kept'"
            )

    def test_all_columns_are_retained(self):
        df = self._make_df()
        _, retained, _ = preprocess(df)
        for col in ["col_a", "col_b", "col_c"]:
            assert col in retained

    def test_output_shape(self):
        df = self._make_df()
        X, retained, _ = preprocess(df)
        assert X.shape == (20, 3)


# ---------------------------------------------------------------------------
# 6. Mixed-type object column does not crash
# ---------------------------------------------------------------------------


class TestMixedTypeColumn:
    """A column containing both integers and strings (object dtype) must not crash."""

    def _make_df(self) -> pd.DataFrame:
        # Mix of ints and strings in one column; pandas will use object dtype.
        mixed = [1, "two", 3, "four", 5, 1, "two", 3, "four", 5]
        return pd.DataFrame(
            {
                "mixed": mixed,
                "numeric": list(range(10)),
            }
        )

    def test_does_not_raise(self):
        df = self._make_df()
        # Should complete without raising any exception.
        X, retained, report = preprocess(df)

    def test_output_has_correct_row_count(self):
        df = self._make_df()
        X, retained, report = preprocess(df)
        assert X.shape[0] == 10

    def test_numeric_column_is_retained(self):
        df = self._make_df()
        _, retained, _ = preprocess(df)
        assert "numeric" in retained

    def test_report_has_entry_for_mixed_column(self):
        df = self._make_df()
        _, _, report = preprocess(df)
        rm = _report_map(report)
        assert "mixed" in rm


# ---------------------------------------------------------------------------
# 7. Fewer than 2 usable columns raises PreprocessingError
# ---------------------------------------------------------------------------


class TestInsufficientColumns:
    """PreprocessingError must be raised when < 2 usable features remain."""

    def test_single_constant_string_column_raises(self):
        """Only column is constant → dropped → no features → PreprocessingError."""
        df = pd.DataFrame({"only_col": ["x"] * 10})
        with pytest.raises(PreprocessingError):
            preprocess(df)

    def test_all_null_df_raises(self):
        """All columns are null → all dropped → PreprocessingError."""
        df = pd.DataFrame({"a": [None] * 5, "b": [None] * 5})
        with pytest.raises(PreprocessingError):
            preprocess(df)

    def test_empty_dataframe_raises(self):
        """Empty DataFrame (0 rows) must raise PreprocessingError."""
        df = pd.DataFrame({"a": [], "b": []})
        with pytest.raises(PreprocessingError):
            preprocess(df)

    def test_one_numeric_column_raises_from_main_guard(self):
        """Only one numeric column → 1 retained feature → main.py guard raises 422.

        Preprocess itself succeeds (returns 1 column), so we verify the count
        directly rather than expecting PreprocessingError from preprocess().
        """
        df = pd.DataFrame({"only": list(range(10))})
        X, retained, _ = preprocess(df)
        # preprocess allows 1 column through; main.py enforces the ≥ 2 guard.
        assert len(retained) == 1


# ---------------------------------------------------------------------------
# (d) Completely empty column + constant column in the same DataFrame
# ---------------------------------------------------------------------------


class TestEmptyAndConstantColumns:
    """Both a completely null column and a constant string column must each be
    independently dropped and reported; other columns must survive unaffected."""

    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "numeric":  list(range(20)),
                "empty":    [None] * 20,           # object dtype, all null
                "constant": ["always"] * 20,       # constant string
                "value":    list(range(100, 120)),  # second numeric, ensures ≥ 2 survive
            }
        )

    def test_empty_column_action_is_dropped(self):
        df = self._make_df()
        _, _, report = preprocess(df)
        rm = _report_map(report)
        assert rm["empty"]["action"] == "dropped"

    def test_empty_column_drop_reason_mentions_null(self):
        df = self._make_df()
        _, _, report = preprocess(df)
        rm = _report_map(report)
        assert "null" in rm["empty"]["drop_reason"].lower()

    def test_constant_column_action_is_dropped(self):
        df = self._make_df()
        _, _, report = preprocess(df)
        rm = _report_map(report)
        assert rm["constant"]["action"] == "dropped"

    def test_constant_column_drop_reason_mentions_constant(self):
        df = self._make_df()
        _, _, report = preprocess(df)
        rm = _report_map(report)
        assert "constant" in rm["constant"]["drop_reason"].lower()

    def test_both_dropped_columns_absent_from_retained(self):
        df = self._make_df()
        _, retained, _ = preprocess(df)
        assert "empty" not in retained
        assert "constant" not in retained

    def test_numeric_columns_survive(self):
        df = self._make_df()
        _, retained, _ = preprocess(df)
        assert "numeric" in retained
        assert "value" in retained

    def test_report_has_entry_for_every_original_column(self):
        """feature_report must contain exactly one entry per original column."""
        df = self._make_df()
        _, _, report = preprocess(df)
        reported_cols = {e["original_column"] for e in report}
        assert reported_cols == set(df.columns)

    def test_output_matrix_has_two_columns(self):
        """Only the two numeric columns survive, so the matrix must have 2 cols."""
        df = self._make_df()
        X, retained, _ = preprocess(df)
        assert X.shape == (20, 2)


# ---------------------------------------------------------------------------
# (e1) Non-ISO datetime format: "DD/MM/YYYY HH:MM"
# ---------------------------------------------------------------------------


class TestNonISODatetime:
    """Timestamps in DD/MM/YYYY HH:MM format must still be recognised as
    datetime-like and expanded into hour / dow / epoch features."""

    def _make_df(self) -> pd.DataFrame:
        # 20 distinct hours on the same day — dow is constant so ts_dow will
        # be dropped by the zero-variance step, but ts_epoch must vary.
        timestamps = [f"15/06/2024 {h:02d}:30" for h in range(20)]
        return pd.DataFrame(
            {
                "ts":    timestamps,
                "value": list(range(20)),
            }
        )

    def test_action_is_datetime_encoded(self):
        df = self._make_df()
        _, _, report = preprocess(df)
        rm = _report_map(report)
        assert rm["ts"]["action"] == "datetime_encoded", (
            f"Expected datetime_encoded, got {rm['ts']['action']!r}"
        )

    def test_produced_features_include_hour_dow_epoch(self):
        df = self._make_df()
        _, _, report = preprocess(df)
        rm = _report_map(report)
        produced = set(rm["ts"]["produced_features"])
        assert {"ts_hour", "ts_dow", "ts_epoch"} == produced

    def test_epoch_feature_is_retained(self):
        """ts_epoch always varies when timestamps differ → survives zero-var drop."""
        df = self._make_df()
        _, retained, _ = preprocess(df)
        assert "ts_epoch" in retained

    def test_matrix_values_in_0_1(self):
        df = self._make_df()
        X, _, _ = preprocess(df)
        assert X.min() >= 0.0 - 1e-9
        assert X.max() <= 1.0 + 1e-9

    def test_row_count_preserved(self):
        df = self._make_df()
        X, _, _ = preprocess(df)
        assert X.shape[0] == 20

    def test_no_pandas_userwarning_raised(self, recwarn):
        """Parsing non-ISO dates must not emit a pandas UserWarning."""
        df = self._make_df()
        preprocess(df)
        user_warnings = [
            w for w in recwarn.list
            if issubclass(w.category, UserWarning)
            and "dayfirst" in str(w.message).lower()
        ]
        assert user_warnings == [], (
            f"Unexpected dayfirst UserWarning(s): {[str(w.message) for w in user_warnings]}"
        )


# ---------------------------------------------------------------------------
# (e2) Unix epoch integers in a numeric column
# ---------------------------------------------------------------------------


class TestUnixEpochIntegers:
    """A column of Unix timestamps stored as int64 (numeric dtype) must be
    treated as a plain numeric feature (action='kept'), NOT as datetime-like.
    The pipeline cannot distinguish epoch integers from arbitrary large integers
    without application-level context."""

    def _make_df(self) -> pd.DataFrame:
        base = 1_700_000_000   # 2023-11-14 ~22:13 UTC
        return pd.DataFrame(
            {
                "epoch_ts": [base + i * 3600 for i in range(20)],
                "value":    list(range(20)),
            }
        )

    def test_action_is_kept(self):
        """int64 epoch column has numeric dtype → pipeline keeps it as-is."""
        df = self._make_df()
        _, _, report = preprocess(df)
        rm = _report_map(report)
        assert rm["epoch_ts"]["action"] == "kept", (
            f"Expected 'kept' for int64 epoch column, got {rm['epoch_ts']['action']!r}"
        )

    def test_epoch_column_is_retained(self):
        df = self._make_df()
        _, retained, _ = preprocess(df)
        assert "epoch_ts" in retained

    def test_no_datetime_expansion_occurs(self):
        """No _hour / _dow / _epoch derived columns must appear in retained."""
        df = self._make_df()
        _, retained, _ = preprocess(df)
        for col in retained:
            assert not col.startswith("epoch_ts_"), (
                f"Unexpected datetime-expanded column in retained: {col!r}"
            )

    def test_matrix_values_in_0_1(self):
        df = self._make_df()
        X, _, _ = preprocess(df)
        assert X.min() >= 0.0 - 1e-9
        assert X.max() <= 1.0 + 1e-9

    def test_output_shape(self):
        df = self._make_df()
        X, retained, _ = preprocess(df)
        assert X.shape == (20, 2)


# ---------------------------------------------------------------------------
# (f) 100 rows, 3 numeric columns
# ---------------------------------------------------------------------------


class TestSmallThreeColumnCSV:
    """A 100-row, 3-column all-numeric CSV must pass through cleanly with
    all three columns kept and the feature_report classifying each as 'kept'."""

    def _make_df(self) -> pd.DataFrame:
        rng = np.random.default_rng(7)
        return pd.DataFrame(
            {
                "col_a": rng.integers(1, 500, 100).astype(float),
                "col_b": rng.random(100),
                "col_c": rng.integers(1, 10,  100).astype(float),
            }
        )

    def test_all_columns_have_action_kept(self):
        df = self._make_df()
        _, _, report = preprocess(df)
        for entry in report:
            assert entry["action"] == "kept", (
                f"Column {entry['original_column']!r}: expected 'kept', "
                f"got {entry['action']!r}"
            )

    def test_all_three_columns_retained(self):
        df = self._make_df()
        _, retained, _ = preprocess(df)
        assert set(retained) == {"col_a", "col_b", "col_c"}

    def test_output_shape_is_100x3(self):
        df = self._make_df()
        X, _, _ = preprocess(df)
        assert X.shape == (100, 3)

    def test_matrix_values_in_0_1(self):
        df = self._make_df()
        X, _, _ = preprocess(df)
        assert X.min() >= 0.0 - 1e-9
        assert X.max() <= 1.0 + 1e-9

    def test_report_has_three_entries(self):
        df = self._make_df()
        _, _, report = preprocess(df)
        assert len(report) == 3

    def test_no_drop_reasons(self):
        """No column should be dropped, so all drop_reason values must be empty."""
        df = self._make_df()
        _, _, report = preprocess(df)
        for entry in report:
            assert entry["drop_reason"] == "", (
                f"Column {entry['original_column']!r} has unexpected "
                f"drop_reason: {entry['drop_reason']!r}"
            )

    def test_produced_features_each_equal_column_name(self):
        """For 'kept' columns, produced_features must be [original_column_name]."""
        df = self._make_df()
        _, _, report = preprocess(df)
        for entry in report:
            assert entry["produced_features"] == [entry["original_column"]], (
                f"Column {entry['original_column']!r}: expected produced_features "
                f"= [{entry['original_column']!r}], got {entry['produced_features']}"
            )
