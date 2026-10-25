"""
scripts/smoke_test_unseen.py
============================
Smoke test using three synthetic CSVs that were NOT used to build the project.
Each CSV is sent to POST /detect via FastAPI's TestClient (same code path
as the live API, no logic duplicated).

CSV shapes:
  (a) Apache-style access log: timestamp, host, method, path, status, bytes
  (b) Network flow log:        src_ip, dst_ip, port, protocol, duration, packets
  (c) Numeric-only sensor:    5 columns, one clear outlier row at index 50

For each CSV the script:
  - Prints the HTTP status, the feature_report, and the anomaly count.
  - Asserts HTTP 200 (no 500 / 422 / 413).
  - Verifies that every planted anomaly row is flagged by the detector.
  - For (c), asserts the outlier row has the lowest anomaly_score.

If a planted row is missed, the script reports it with the features that
should have driven the detection, and exits with a non-zero status.

Run from the project root:
    python scripts/smoke_test_unseen.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from io import BytesIO
from typing import NamedTuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient

from backend.main import app

# ---------------------------------------------------------------------------
# Client setup
# ---------------------------------------------------------------------------
client = TestClient(app, raise_server_exceptions=False)

# Higher contamination so outliers are more likely to be flagged on small data.
CONTAMINATION = 0.10


# ---------------------------------------------------------------------------
# Planted anomaly descriptor
# ---------------------------------------------------------------------------


class PlantedAnomaly(NamedTuple):
    """Describes a single planted anomaly row for verification."""
    row_index: int
    description: str
    driving_feature: str   # the original column that should make this anomalous


# ---------------------------------------------------------------------------
# CSV generators — each returns (csv_bytes, [PlantedAnomaly, ...])
# ---------------------------------------------------------------------------


def _apache_csv() -> tuple[bytes, list[PlantedAnomaly]]:
    """Apache-style access log: timestamp, host, method, path, status, bytes."""
    rng = np.random.default_rng(100)
    n = 200
    base_ts = pd.Timestamp("2024-03-10T00:00:00Z")
    methods = ["GET", "POST", "PUT", "DELETE"]
    paths = ["/", "/login", "/api/users", "/api/data", "/static/style.css"]
    hosts = [f"10.0.0.{i % 10 + 1}" for i in range(n)]
    statuses_normal = [200, 200, 200, 301, 404]
    rows = []
    for i in range(n):
        ts = base_ts + pd.Timedelta(minutes=i * 7 + int(rng.integers(-30, 30)))
        rows.append({
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S"),
            "host":      hosts[i],
            "method":    str(rng.choice(methods)),
            "path":      str(rng.choice(paths)),
            "status":    int(rng.choice(statuses_normal)),
            "bytes":     int(rng.integers(500, 50000)),
        })
    # Plant 3 anomalies
    rows[50]["bytes"] = 9_999_999
    rows[50]["host"] = "203.0.113.99"
    rows[100]["status"] = 500
    rows[150]["method"] = "TRACE"
    rows[150]["path"] = "/wp-admin"

    planted = [
        PlantedAnomaly(50,  "high bytes (9.99M) from rare host 203.0.113.99", "bytes"),
        PlantedAnomaly(100, "HTTP 500 status code",                         "status"),
        PlantedAnomaly(150, "TRACE method + unusual /wp-admin path",        "method"),
    ]
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8"), planted


def _network_flow_csv() -> tuple[bytes, list[PlantedAnomaly]]:
    """Network flow log: src_ip, dst_ip, port, protocol, duration, packets."""
    rng = np.random.default_rng(200)
    n = 200
    src_ips = [f"192.168.1.{i % 20 + 1}" for i in range(n)]
    dst_ips = [f"10.0.0.{i % 5 + 1}" for i in range(n)]
    ports = [80, 443, 22, 53, 8080]
    protocols = ["TCP", "TCP", "TCP", "UDP", "ICMP"]
    rows = []
    for i in range(n):
        rows.append({
            "src_ip":   src_ips[i],
            "dst_ip":   dst_ips[i],
            "port":     int(rng.choice(ports)),
            "protocol": str(rng.choice(protocols)),
            "duration": float(rng.integers(10, 5000)),
            "packets":  int(rng.integers(5, 200)),
        })
    # Plant 3 anomalies
    rows[75]["packets"] = 999_999
    rows[125]["port"] = 31337
    rows[175]["protocol"] = "GRE"

    planted = [
        PlantedAnomaly(75,  "huge packet count (999,999)",  "packets"),
        PlantedAnomaly(125, "rare port 31337",               "port"),
        PlantedAnomaly(175, "unusual protocol GRE",          "protocol"),
    ]
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8"), planted


def _sensor_csv() -> tuple[bytes, list[PlantedAnomaly]]:
    """Numeric-only sensor data: 5 columns, one clear outlier at row 50."""
    rng = np.random.default_rng(300)
    n = 100
    data = {
        "temp":     rng.normal(22.0, 2.0, n),
        "pressure": rng.normal(1013.0, 5.0, n),
        "humidity": rng.normal(45.0, 10.0, n),
        "voltage":  rng.normal(3.3, 0.1, n),
        "current":  rng.normal(0.5, 0.05, n),
    }
    df = pd.DataFrame(data)
    df.loc[50, "temp"] = 150.0
    df.loc[50, "pressure"] = 2000.0
    df.loc[50, "humidity"] = 99.0
    df.loc[50, "voltage"] = 12.0
    df.loc[50, "current"] = 50.0

    planted = [
        PlantedAnomaly(50, "extreme outlier on all 5 sensor columns", "all columns"),
    ]
    return df.to_csv(index=False).encode("utf-8"), planted


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _print_feature_report(report: list) -> None:
    print(f"  Feature report ({len(report)} entries):")
    print(f"  {'Column':<20s} {'Action':<22s} {'Produced features':<40s} Drop reason")
    print(f"  {'-'*20} {'-'*22} {'-'*40} {'-'*30}")
    for entry in report:
        produced = ", ".join(entry["produced_features"]) if entry["produced_features"] else "—"
        reason = entry["drop_reason"] or "—"
        print(f"  {entry['original_column']:<20s} {entry['action']:<22s} {produced:<40s} {reason}")


def _run(
    name: str,
    csv_bytes: bytes,
    planted: list[PlantedAnomaly],
    check_lowest_score: bool = False,
) -> bool:
    """POST a CSV to /detect, print results, verify planted anomalies.

    Returns True if all assertions passed, False if any planted anomaly was missed.
    """
    SEP = "=" * 70
    print(f"\n{SEP}")
    print(f"  {name}")
    print(SEP)

    response = client.post(
        "/detect",
        files={"file": ("test.csv", BytesIO(csv_bytes), "text/csv")},
        data={"contamination": str(CONTAMINATION)},
    )

    print(f"  HTTP status: {response.status_code}")

    if response.status_code != 200:
        print(f"  ERROR body: {response.json()}")
        assert response.status_code != 500, f"{name}: got HTTP 500"
        raise AssertionError(f"{name}: expected 200, got {response.status_code}")

    body = response.json()
    rows = body.get("rows", [])
    report = body.get("feature_report", [])
    summary = body.get("summary", {})
    anomalies = summary.get("anomalies", 0)
    total = summary.get("total", 0)

    _print_feature_report(report)

    print(f"\n  Anomalies: {anomalies} / {total} rows flagged")
    print(f"  Severity:  HIGH={summary.get('high',0)}  "
          f"MEDIUM={summary.get('medium',0)}  LOW={summary.get('low',0)}")

    produced_all: list[str] = []
    for e in report:
        produced_all.extend(e["produced_features"])
    print(f"  Retained features: {produced_all}")

    # ── Verify each planted anomaly ─────────────────────────────────────
    print(f"\n  Planted anomaly verification ({len(planted)} planted):")
    missed: list[PlantedAnomaly] = []

    for p in planted:
        row = rows[p.row_index]
        flagged = row.get("anomaly_label") == -1
        score = row.get("anomaly_score", None)
        severity = row.get("severity", "")

        status_icon = "✓" if flagged else "✗"
        print(
            f"  {status_icon} row {p.row_index:>3d}  {p.description:<50s}"
            f"  label={row.get('anomaly_label')}  score={score}  sev={severity}"
        )
        if not flagged:
            missed.append(p)

    caught = len(planted) - len(missed)
    print(f"\n  Planted anomalies caught: {caught} / {len(planted)}")

    # ── Print feature values for missed rows ────────────────────────────
    if missed:
        print(f"\n  ✗ MISSED planted anomaly rows:")
        # Build a map of original column → produced features for lookup
        col_to_features: dict[str, list[str]] = {}
        for e in report:
            col_to_features[e["original_column"]] = e["produced_features"]

        for p in missed:
            row = rows[p.row_index]
            print(f"    row {p.row_index}: {p.description}")
            print(f"      driving feature: {p.driving_feature}")
            print(f"      row values: { {k: v for k, v in row.items() if k not in ('anomaly_label','anomaly_score','severity')} }")
            # Show what the driving feature was expanded to
            feats = col_to_features.get(p.driving_feature, [])
            if feats:
                feat_values = {f: row.get(f, "(not in row — internal feature)") for f in feats}
                print(f"      produced features from {p.driving_feature}: {feats}")
                print(f"      (these are internal to the feature matrix, not in the response row)")

    # ── Outlier lowest-score check (for sensor data) ────────────────────
    if check_lowest_score:
        scores_all = [r["anomaly_score"] for r in rows]
        min_score = min(scores_all)
        min_index = scores_all.index(min_score)
        outlier_index = planted[0].row_index
        outlier_score = rows[outlier_index]["anomaly_score"]

        print(f"\n  Outlier lowest-score check:")
        print(f"    outlier row {outlier_index}: anomaly_score={outlier_score}")
        print(f"    min score in file: {min_score} (at row {min_index})")

        assert outlier_score == min_score, (
            f"Outlier at row {outlier_index} has score {outlier_score}, "
            f"but the minimum score in the file is {min_score} at row {min_index}"
        )
        print(f"    ✓ Outlier has the lowest anomaly_score in the file")

    # ── Final assertions ─────────────────────────────────────────────────
    assert response.status_code != 500, f"{name}: got HTTP 500"
    assert anomalies >= 1, f"{name}: expected at least 1 anomaly, got {anomalies}"

    if missed:
        print(f"\n  ✗ {name}: {len(missed)} planted anomaly row(s) were NOT flagged.")
        return False

    print(f"\n  ✓ All assertions passed (no 500, ≥1 anomaly, all planted rows caught)")
    return True


def main() -> None:
    print("Smoke test: three unseen synthetic CSVs through POST /detect")
    print(f"Contamination: {CONTAMINATION}")

    all_passed = True

    csv_a, planted_a = _apache_csv()
    passed_a = _run("(a) Apache-style access log", csv_a, planted_a)
    all_passed = all_passed and passed_a

    csv_b, planted_b = _network_flow_csv()
    passed_b = _run("(b) Network flow log", csv_b, planted_b)
    all_passed = all_passed and passed_b

    csv_c, planted_c = _sensor_csv()
    passed_c = _run("(c) Numeric-only sensor data", csv_c, planted_c,
                    check_lowest_score=True)
    all_passed = all_passed and passed_c

    print("\n" + "=" * 70)
    if all_passed:
        print("  All three CSVs processed — no 500s, all planted anomalies caught.")
    else:
        print("  ✗ Some planted anomalies were missed. See details above.")
    print("=" * 70)

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
