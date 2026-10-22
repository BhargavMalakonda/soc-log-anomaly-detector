"""
generate_sample_logs.py

Generates two files:

  data/sample_logs.csv
      The file the app ingests: columns timestamp, ip, status, bytes.
      Format is unchanged from the original.

  data/sample_logs_labeled.csv
      Same rows in the same order, plus an is_anomaly column
      (1 for planted-anomaly rows, 0 for normal rows).
      Used by scripts/evaluate.py for reproducible evaluation.

Row counts
----------
The actual output size differs from the value requested in ANOMALY_COUNT because
generate_anomalous_entries() clamps repeated-IP entries to at least 31 to
satisfy the "≥ 30 occurrences" signal requirement.  The constant TOTAL_ROWS
reflects what the script actually writes.

Run with:
    python data/generate_sample_logs.py
"""

import csv
import os
import random
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH         = os.path.join(SCRIPT_DIR, "sample_logs.csv")
OUTPUT_LABELED_PATH = os.path.join(SCRIPT_DIR, "sample_logs_labeled.csv")

# ---------------------------------------------------------------------------
# Generation parameters
# ---------------------------------------------------------------------------

# Normal entries to generate.
NORMAL_COUNT = 202

# Anomaly sub-type counts (see generate_anomalous_entries).
# Type 1 — high-bytes:        5
# Type 2 — rare status codes: 3
# Type 3 — repeated IP:       max(ANOMALY_COUNT - 8, 31) = 31
# Total anomalous rows actually written: 5 + 3 + 31 = 39
_HIGH_BYTES_COUNT  = 5
_RARE_STATUS_COUNT = 3
_MIN_REPEATED_IP   = 31    # must be ≥ 31 to guarantee the signal
ANOMALY_COUNT = _HIGH_BYTES_COUNT + _RARE_STATUS_COUNT + _MIN_REPEATED_IP  # 39

# TOTAL_ROWS reflects the real output (202 normal + 39 anomalous).
TOTAL_ROWS = NORMAL_COUNT + ANOMALY_COUNT  # 241

# ---------------------------------------------------------------------------
# Address / status pools
# ---------------------------------------------------------------------------

NORMAL_IP_POOL = [
    "10.0.0.1",
    "10.0.0.2",
    "10.0.0.3",
    "10.0.0.4",
    "10.0.0.5",
    "192.168.1.10",
    "192.168.1.11",
    "192.168.1.12",
    "192.168.1.13",
    "192.168.1.14",
    "172.16.0.1",
    "172.16.0.2",
    "172.16.0.3",
    "172.16.0.4",
    "172.16.0.5",
    "203.0.113.10",
    "203.0.113.11",
    "203.0.113.12",
    "198.51.100.5",
    "198.51.100.6",
]

NORMAL_STATUS_CODES    = [200, 301, 302, 304, 404]
ANOMALOUS_STATUS_CODES = [500, 503, 429]

# Single IP that appears repeatedly — the frequency-based anomaly signal.
REPEATED_ANOMALOUS_IP = "45.33.32.156"

# ---------------------------------------------------------------------------
# Timestamp helper
# ---------------------------------------------------------------------------

BASE_TIME = datetime(2024, 6, 15, 0, 0, 0)


def random_timestamp(index: int, total: int) -> str:
    """Return an ISO-8601 timestamp spread across a 24-hour window."""
    seconds_offset = int((index / total) * 86400) + random.randint(-30, 30)
    seconds_offset = max(0, seconds_offset)
    return (BASE_TIME + timedelta(seconds=seconds_offset)).strftime(
        "%Y-%m-%dT%H:%M:%S"
    )


# ---------------------------------------------------------------------------
# Row generators
# ---------------------------------------------------------------------------


def generate_normal_entry(index: int, total: int) -> dict:
    """Return a single normal log entry (is_anomaly=0)."""
    return {
        "timestamp": random_timestamp(index, total),
        "ip":        random.choice(NORMAL_IP_POOL),
        "status":    random.choice(NORMAL_STATUS_CODES),
        "bytes":     random.randint(100, 50000),
        "is_anomaly": 0,
    }


def generate_anomalous_entries(total: int) -> list[dict]:
    """Return all planted-anomaly rows (is_anomaly=1).

    Three anomaly types
    -------------------
    Type 1 — high bytes (5 rows)
        bytes in [500_001, 5_000_000]; status 200; IP from normal pool.
        Detected primarily via the `bytes` feature.

    Type 2 — rare status codes (3 rows)
        status in {500, 503, 429}; normal-pool IP; bytes in [100, 5000].
        Detected via the `status` feature.

    Type 3 — repeated source IP (31 rows)
        All from REPEATED_ANOMALOUS_IP.  The frequency encoding used by
        preprocess() assigns this IP a very low frequency score, making
        every row from it an outlier in the ip_freq feature.
    """
    entries: list[dict] = []

    # Type 1 — high bytes
    for i in range(_HIGH_BYTES_COUNT):
        entries.append({
            "timestamp":  random_timestamp(i * 10, total),
            "ip":         random.choice(NORMAL_IP_POOL),
            "status":     200,
            "bytes":      random.randint(500_001, 5_000_000),
            "is_anomaly": 1,
        })

    # Type 2 — rare status codes
    for i in range(_RARE_STATUS_COUNT):
        entries.append({
            "timestamp":  random_timestamp(i * 15 + 5, total),
            "ip":         random.choice(NORMAL_IP_POOL),
            "status":     random.choice(ANOMALOUS_STATUS_CODES),
            "bytes":      random.randint(100, 5000),
            "is_anomaly": 1,
        })

    # Type 3 — repeated IP
    for i in range(_MIN_REPEATED_IP):
        entries.append({
            "timestamp":  random_timestamp(i * 3 + 2, total),
            "ip":         REPEATED_ANOMALOUS_IP,
            "status":     random.choice(ANOMALOUS_STATUS_CODES + [200]),
            "bytes":      random.randint(100, 50000),
            "is_anomaly": 1,
        })

    return entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    rows: list[dict] = []

    # Normal rows
    for i in range(NORMAL_COUNT):
        rows.append(generate_normal_entry(i, TOTAL_ROWS))

    # Anomalous rows
    rows.extend(generate_anomalous_entries(TOTAL_ROWS))

    # Shuffle so anomalies are interspersed; re-assign timestamps to be
    # chronological after the shuffle.
    random.shuffle(rows)
    for i, row in enumerate(rows):
        row["timestamp"] = random_timestamp(i, len(rows))

    # ------------------------------------------------------------------
    # Write app CSV — no is_anomaly column (existing format, unchanged)
    # ------------------------------------------------------------------
    app_fieldnames = ["timestamp", "ip", "status", "bytes"]
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=app_fieldnames,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # ------------------------------------------------------------------
    # Write labeled CSV — same rows + is_anomaly column
    # ------------------------------------------------------------------
    labeled_fieldnames = ["timestamp", "ip", "status", "bytes", "is_anomaly"]
    with open(OUTPUT_LABELED_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=labeled_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_written   = len(rows)
    anomaly_written = sum(1 for r in rows if r["is_anomaly"] == 1)
    normal_written  = total_written - anomaly_written
    repeated_ip_n   = sum(1 for r in rows if r["ip"] == REPEATED_ANOMALOUS_IP)

    print(f"Wrote {OUTPUT_PATH}")
    print(f"  Columns : timestamp, ip, status, bytes")
    print(f"  Rows    : {total_written}")

    print(f"\nWrote {OUTPUT_LABELED_PATH}")
    print(f"  Columns : timestamp, ip, status, bytes, is_anomaly")
    print(f"  Rows    : {total_written}")
    print(f"  Normal  : {normal_written}")
    print(f"  Anomaly : {anomaly_written} ({anomaly_written / total_written * 100:.1f}%)")
    print(f"  Repeated IP ({REPEATED_ANOMALOUS_IP}): {repeated_ip_n} occurrences")


if __name__ == "__main__":
    main()
