"""
tests/test_api.py
=================
Integration tests for the FastAPI endpoints in backend/main.py.

Uses FastAPI's built-in TestClient (backed by httpx) so the full
request/response cycle — validation, preprocessing, model, serialisation —
is exercised without a live server.

Covers:
  - Valid CSV → 200 with required keys (columns, rows, summary, feature_report)
  - Empty file → 400
  - Non-.csv extension → 422
  - anomaly_label == -1 count in rows equals summary.anomalies
  - contamination outside [0.01, 0.30] → 422 with informative message
  - Single outlier row (bytes=9_999_999) in 200-row normal dataset is the
    most anomalous row (most negative score)
  - Windows-1252 encoded CSV → 422 (clean error, not 500)
"""

from __future__ import annotations

import io
import warnings

import numpy as np
import pandas as pd
import pytest

# Suppress the starlette httpx deprecation warning that fires on some
# installed version combinations — it does not affect test correctness.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient

from backend.main import app

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

client = TestClient(app, raise_server_exceptions=False)


def _csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialise a DataFrame to UTF-8 CSV bytes."""
    return df.to_csv(index=False).encode("utf-8")


def _sample_csv() -> bytes:
    """Return the project's real sample_logs.csv as bytes."""
    with open("data/sample_logs.csv", "rb") as f:
        return f.read()


def _post_detect(csv_bytes: bytes, contamination: float | None = None,
                 filename: str = "test.csv") -> "Response":
    """POST to /detect; optionally include a contamination form field."""
    files = {"file": (filename, io.BytesIO(csv_bytes), "text/csv")}
    data = {}
    if contamination is not None:
        data["contamination"] = str(contamination)
    return client.post("/detect", files=files, data=data)


def _post_upload(csv_bytes: bytes, filename: str = "test.csv") -> "Response":
    files = {"file": (filename, io.BytesIO(csv_bytes), "text/csv")}
    return client.post("/upload", files=files)


# ---------------------------------------------------------------------------
# 1. Valid CSV → 200 with required keys
# ---------------------------------------------------------------------------


class TestValidCSV:
    """Happy-path tests against data/sample_logs.csv."""

    def test_status_200(self):
        r = _post_detect(_sample_csv())
        assert r.status_code == 200, r.text

    def test_response_has_columns_key(self):
        r = _post_detect(_sample_csv())
        assert "columns" in r.json()

    def test_response_has_rows_key(self):
        r = _post_detect(_sample_csv())
        assert "rows" in r.json()

    def test_response_has_summary_key(self):
        r = _post_detect(_sample_csv())
        assert "summary" in r.json()

    def test_response_has_feature_report_key(self):
        r = _post_detect(_sample_csv())
        assert "feature_report" in r.json()

    def test_summary_has_expected_sub_keys(self):
        r = _post_detect(_sample_csv())
        summary = r.json()["summary"]
        for key in ("total", "anomalies", "high", "medium", "low"):
            assert key in summary, f"Missing summary key: {key!r}"

    def test_columns_includes_detection_columns(self):
        r = _post_detect(_sample_csv())
        cols = r.json()["columns"]
        for required in ("anomaly_label", "anomaly_score", "severity"):
            assert required in cols, f"Missing result column: {required!r}"

    def test_rows_count_matches_summary_total(self):
        r = _post_detect(_sample_csv())
        data = r.json()
        assert len(data["rows"]) == data["summary"]["total"]

    def test_feature_report_is_non_empty_list(self):
        r = _post_detect(_sample_csv())
        report = r.json()["feature_report"]
        assert isinstance(report, list)
        assert len(report) > 0

    def test_feature_report_entries_have_required_keys(self):
        r = _post_detect(_sample_csv())
        for entry in r.json()["feature_report"]:
            for key in ("original_column", "action", "produced_features", "drop_reason"):
                assert key in entry, f"feature_report entry missing key {key!r}"


# ---------------------------------------------------------------------------
# 2. anomaly_label count equals summary.anomalies
# ---------------------------------------------------------------------------


class TestAnomalyCountConsistency:
    """The count of rows with anomaly_label == -1 must equal summary.anomalies."""

    def test_count_matches_at_default_contamination(self):
        r = _post_detect(_sample_csv())
        data = r.json()
        flagged = sum(1 for row in data["rows"] if row["anomaly_label"] == -1)
        assert flagged == data["summary"]["anomalies"]

    def test_count_matches_at_contamination_015(self):
        r = _post_detect(_sample_csv(), contamination=0.15)
        data = r.json()
        flagged = sum(1 for row in data["rows"] if row["anomaly_label"] == -1)
        assert flagged == data["summary"]["anomalies"]

    def test_severity_counts_sum_to_anomalies(self):
        """high + medium + low must equal total anomalies."""
        r = _post_detect(_sample_csv())
        s = r.json()["summary"]
        assert s["high"] + s["medium"] + s["low"] == s["anomalies"]

    def test_normal_rows_have_empty_severity(self):
        """Rows with anomaly_label == 1 must have severity == ''."""
        r = _post_detect(_sample_csv())
        for row in r.json()["rows"]:
            if row["anomaly_label"] == 1:
                assert row["severity"] == "", (
                    f"Normal row has non-empty severity: {row['severity']!r}"
                )

    def test_anomalous_rows_have_non_empty_severity(self):
        """Rows with anomaly_label == -1 must have a non-empty severity."""
        r = _post_detect(_sample_csv())
        for row in r.json()["rows"]:
            if row["anomaly_label"] == -1:
                assert row["severity"] in ("HIGH", "MEDIUM", "LOW"), (
                    f"Anomalous row has unexpected severity: {row['severity']!r}"
                )


# ---------------------------------------------------------------------------
# 3. Empty file → 400
# ---------------------------------------------------------------------------


class TestEmptyFile:
    def test_empty_csv_returns_400(self):
        r = _post_detect(b"", filename="empty.csv")
        assert r.status_code == 400, r.text

    def test_empty_csv_error_body(self):
        r = _post_detect(b"", filename="empty.csv")
        assert "error" in r.json()


# ---------------------------------------------------------------------------
# 4. Non-.csv extension → 422
# ---------------------------------------------------------------------------


class TestNonCSVExtension:
    def test_txt_extension_returns_422(self):
        r = _post_detect(b"a,b\n1,2\n", filename="data.txt")
        assert r.status_code == 422, r.text

    def test_json_extension_returns_422(self):
        r = _post_detect(b'{"a":1}', filename="data.json")
        assert r.status_code == 422, r.text

    def test_error_message_mentions_csv(self):
        r = _post_detect(b"a,b\n1,2\n", filename="data.txt")
        assert "csv" in r.json()["error"].lower()


# ---------------------------------------------------------------------------
# 5. contamination outside [0.01, 0.30] → 422
# ---------------------------------------------------------------------------


class TestContaminationValidation:
    def test_contamination_too_high_returns_422(self):
        r = _post_detect(_sample_csv(), contamination=0.50)
        assert r.status_code == 422, r.text

    def test_contamination_too_low_returns_422(self):
        r = _post_detect(_sample_csv(), contamination=0.005)
        assert r.status_code == 422, r.text

    def test_contamination_at_lower_bound_is_accepted(self):
        r = _post_detect(_sample_csv(), contamination=0.01)
        assert r.status_code == 200, r.text

    def test_contamination_at_upper_bound_is_accepted(self):
        r = _post_detect(_sample_csv(), contamination=0.30)
        assert r.status_code == 200, r.text

    def test_error_message_mentions_contamination(self):
        r = _post_detect(_sample_csv(), contamination=0.99)
        assert "contamination" in r.json()["error"].lower()

    def test_error_message_includes_value(self):
        r = _post_detect(_sample_csv(), contamination=0.99)
        assert "0.99" in r.json()["error"]


# ---------------------------------------------------------------------------
# 6. Outlier row has the most negative anomaly score
# ---------------------------------------------------------------------------


class TestOutlierDetection:
    """A single row with bytes=9_999_999 among 200 rows of bytes=1000 must
    receive the most negative anomaly_score in the response."""

    @pytest.fixture(scope="class")
    @classmethod
    def detect_response(cls):
        rng = np.random.default_rng(42)
        # 200 normal rows
        normal_bytes = rng.integers(800, 1200, 200).tolist()
        ips = [f"10.0.0.{(i % 20) + 1}" for i in range(200)]
        statuses = [200] * 200
        normal_rows = [
            {"timestamp": f"2024-01-01T{(i % 24):02d}:00:00",
             "ip": ips[i], "status": statuses[i], "bytes": normal_bytes[i]}
            for i in range(200)
        ]
        # 1 outlier row at position 100
        outlier = {
            "timestamp": "2024-01-01T12:00:00",
            "ip": "10.0.0.1",
            "status": 200,
            "bytes": 9_999_999,
        }
        all_rows = normal_rows[:100] + [outlier] + normal_rows[100:]
        df = pd.DataFrame(all_rows)
        csv_bytes = _csv_bytes(df)
        return _post_detect(csv_bytes, contamination=0.10)

    def test_status_200(self, detect_response):
        assert detect_response.status_code == 200, detect_response.text

    def test_outlier_row_is_flagged(self, detect_response):
        """The outlier at index 100 must be labeled -1."""
        rows = detect_response.json()["rows"]
        outlier_row = rows[100]
        assert outlier_row["anomaly_label"] == -1, (
            f"Expected outlier to be labeled anomaly, got {outlier_row['anomaly_label']}"
        )

    def test_outlier_has_most_negative_score(self, detect_response):
        """bytes=9_999_999 must have the minimum (most negative) anomaly_score."""
        rows = detect_response.json()["rows"]
        scores = [row["anomaly_score"] for row in rows]
        min_score = min(scores)
        outlier_score = rows[100]["anomaly_score"]
        assert outlier_score == min_score, (
            f"Outlier score {outlier_score} is not the minimum ({min_score})"
        )

    def test_outlier_has_high_severity(self, detect_response):
        """The most anomalous row must receive HIGH severity."""
        rows = detect_response.json()["rows"]
        assert rows[100]["severity"] == "HIGH"


# ---------------------------------------------------------------------------
# 7. Windows-1252 encoded CSV → 422, not 500
# ---------------------------------------------------------------------------


class TestEncodingError:
    """A Windows-1252 encoded CSV that can't be decoded as UTF-8 must return
    HTTP 422 with an error body, not HTTP 500."""

    @staticmethod
    def _win1252_bytes() -> bytes:
        # Build a CSV that contains a byte sequence valid in Windows-1252
        # but invalid in UTF-8 (byte 0x92 = curly right-single-quote in cp1252).
        header = b"name,value\n"
        row = "caf\u00e9,42\n".encode("windows-1252")  # \xe9 is valid UTF-8 too
        # Use a byte that is illegal in strict UTF-8: 0x81 (reserved in cp1252).
        illegal_utf8_row = b"bad\x81val,1\n"
        return header + row + illegal_utf8_row

    def test_win1252_returns_4xx_not_500(self):
        csv_bytes = self._win1252_bytes()
        r = _post_detect(csv_bytes, filename="logs.csv")
        assert r.status_code in (400, 413, 422), (
            f"Expected 4xx for encoding error, got {r.status_code}: {r.text}"
        )

    def test_win1252_does_not_return_500(self):
        csv_bytes = self._win1252_bytes()
        r = _post_detect(csv_bytes, filename="logs.csv")
        assert r.status_code != 500, (
            f"Got 500 for encoding error — internal details may be leaking: {r.text}"
        )

    def test_win1252_response_has_error_key(self):
        csv_bytes = self._win1252_bytes()
        r = _post_detect(csv_bytes, filename="logs.csv")
        assert "error" in r.json()
