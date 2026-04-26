"""
generate_sample_logs.py

Generates data/sample_logs.csv with realistic HTTP server log data.
- At least 200 rows total
- Columns: timestamp, ip, status, bytes
- ~90-95% normal entries, ~5-10% anomalous entries

Run with:
    python data/generate_sample_logs.py
"""

import csv
import os
import random
from datetime import datetime, timedelta

# Fixed seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Output path relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "sample_logs.csv")

# Total rows to generate
TOTAL_ROWS = 220

# Anomaly fraction: ~8% of total rows
ANOMALY_COUNT = 18  # ~8.2% of 220
NORMAL_COUNT = TOTAL_ROWS - ANOMALY_COUNT  # 202

# Pool of ~20 normal IP addresses
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

# Normal HTTP status codes
NORMAL_STATUS_CODES = [200, 301, 302, 304, 404]

# Rare/anomalous status codes
ANOMALOUS_STATUS_CODES = [500, 503, 429]

# A single IP that will appear repeatedly in the anomalous set (> 30 occurrences total)
REPEATED_ANOMALOUS_IP = "45.33.32.156"

# Base timestamp: start 24 hours ago, spread entries across the day
BASE_TIME = datetime(2024, 6, 15, 0, 0, 0)


def random_timestamp(index: int, total: int) -> str:
    """Generate an ISO-8601 timestamp spread across a 24-hour window."""
    seconds_offset = int((index / total) * 86400) + random.randint(-30, 30)
    seconds_offset = max(0, seconds_offset)
    ts = BASE_TIME + timedelta(seconds=seconds_offset)
    return ts.strftime("%Y-%m-%dT%H:%M:%S")


def generate_normal_entry(index: int, total: int) -> dict:
    """Generate a single normal log entry."""
    return {
        "timestamp": random_timestamp(index, total),
        "ip": random.choice(NORMAL_IP_POOL),
        "status": random.choice(NORMAL_STATUS_CODES),
        "bytes": random.randint(100, 50000),
    }


def generate_anomalous_entries(count: int, total: int) -> list[dict]:
    """
    Generate anomalous log entries with three types of anomalies:
    1. Unusually high bytes values (> 500,000)
    2. Rare status codes (500, 503, 429)
    3. Repeated requests from a single IP (> 30 occurrences in the anomalous set)
    """
    entries = []

    # Type 1: High bytes anomalies (5 entries)
    high_bytes_count = 5
    for i in range(high_bytes_count):
        entries.append({
            "timestamp": random_timestamp(i * 10, total),
            "ip": random.choice(NORMAL_IP_POOL),
            "status": 200,
            "bytes": random.randint(500001, 5000000),
        })

    # Type 2: Rare status code anomalies (not from repeated IP, 3 entries)
    rare_status_count = 3
    for i in range(rare_status_count):
        entries.append({
            "timestamp": random_timestamp(i * 15 + 5, total),
            "ip": random.choice(NORMAL_IP_POOL),
            "status": random.choice(ANOMALOUS_STATUS_CODES),
            "bytes": random.randint(100, 5000),
        })

    # Type 3: Repeated requests from a single IP (remaining entries, ensuring > 30 total)
    # We need > 30 occurrences from REPEATED_ANOMALOUS_IP across the whole dataset.
    # We'll add enough here to guarantee that.
    repeated_ip_count = count - high_bytes_count - rare_status_count
    # Ensure at least 31 entries from the repeated IP
    repeated_ip_count = max(repeated_ip_count, 31)
    for i in range(repeated_ip_count):
        entries.append({
            "timestamp": random_timestamp(i * 3 + 2, total),
            "ip": REPEATED_ANOMALOUS_IP,
            "status": random.choice(ANOMALOUS_STATUS_CODES + [200]),
            "bytes": random.randint(100, 50000),
        })

    return entries


def main():
    rows = []

    # Generate normal entries
    for i in range(NORMAL_COUNT):
        rows.append(generate_normal_entry(i, TOTAL_ROWS))

    # Generate anomalous entries
    anomalous = generate_anomalous_entries(ANOMALY_COUNT, TOTAL_ROWS)
    rows.extend(anomalous)

    # Shuffle all rows so anomalies are interspersed
    random.shuffle(rows)

    # Re-assign timestamps in order so the log looks chronological
    for i, row in enumerate(rows):
        row["timestamp"] = random_timestamp(i, len(rows))

    # Write to CSV
    fieldnames = ["timestamp", "ip", "status", "bytes"]
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total_written = len(rows)
    anomaly_written = len(anomalous)
    repeated_ip_count = sum(1 for r in rows if r["ip"] == REPEATED_ANOMALOUS_IP)

    print(f"Generated {OUTPUT_PATH}")
    print(f"  Total rows    : {total_written}")
    print(f"  Normal rows   : {total_written - anomaly_written}")
    print(f"  Anomalous rows: {anomaly_written} ({anomaly_written / total_written * 100:.1f}%)")
    print(f"  Repeated IP ({REPEATED_ANOMALOUS_IP}): {repeated_ip_count} occurrences")


if __name__ == "__main__":
    main()
