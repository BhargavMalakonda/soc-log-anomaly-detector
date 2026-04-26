# AI SOC Log Anomaly Detection Dashboard

A SOC-style system that analyzes system logs to automatically detect suspicious behavior using machine learning. 

Upload logs → detect anomalies → visualize threats with severity scoring in a simple dashboard.

---

## Features

- **CSV Upload** — drag-and-drop or file-picker upload with full validation (type, size ≤ 50 MB, non-empty)
- **Flexible Schema** — dynamic column detection; works with any CSV, not just the sample format
- **ML Anomaly Detection** — Isolation Forest (scikit-learn) with configurable contamination rate
- **Severity Classification** — each anomaly is rated HIGH / MEDIUM / LOW based on its anomaly score
- **Color-Coded Dashboard** — anomalous rows highlighted red, normal rows green, sticky table header
- **Anomaly Filter** — toggle to show only anomalous entries
- **Download Results** — export current view (filtered or full) as `anomaly_results.csv`
- **Summary Stats** — total entries, anomaly count, and per-severity breakdown
- **Error Handling** — all edge cases handled gracefully with user-friendly messages

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI (Python) |
| ML | scikit-learn — IsolationForest |
| Frontend | HTML5 + CSS3 + Vanilla JS (no frameworks) |
| Templates | Jinja2 (served by FastAPI) |
| Data | pandas + numpy |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Browser (Client)                        │
│                                                                 │
│  ┌──────────────┐   ┌──────────────────────────────────────┐    │
│  │  index.html  │   │           script.js                  │    │
│  │  (Jinja2)    │   │  - File upload via Fetch API         │    │
│  └──────────────┘   │  - POST /upload → preview table      │    │
│                     │  - POST /detect → results table      │    │
│                     │  - Filter toggle (client-side)       │    │
│                     │  - CSV export (client-side)          │    │
│                     └──────────────────────────────────────┘    │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP (JSON / multipart)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Application (main.py)                │
│                                                                 │
│  GET  /              → serves index.html (Jinja2)               │
│  POST /upload        → validates + parses CSV → preview JSON    │
│  POST /detect        → preprocess + detect → results JSON       │
│  GET  /static/*      → serves CSS / JS                          │
│                                                                 │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │ preprocess.py│  │    model.py      │  │    utils.py      │   │
│  │              │  │                  │  │                  │   │
│  │ - detect     │  │ - IsolationForest│  │ - severity_label │   │
│  │   col types  │  │   fit_predict    │  │ - error_response │   │
│  │ - label enc  │  │ - score_samples  │  │                  │   │
│  │ - fill NaN   │  │                  │  │                  │   │
│  │ - drop bad   │  └──────────────────┘  └──────────────────┘   │
│  │   cols       │                                               │
│  │ - min-max    │                                               │
│  │   normalize  │                                               │
│  └──────────────┘                                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
ai-log-anomaly-dashboard/
│
├── backend/
│   ├── __init__.py
│   ├── main.py          # FastAPI app, routes, validation
│   ├── model.py         # IsolationForest wrapper
│   ├── preprocess.py    # Preprocessing pipeline
│   └── utils.py         # Shared utilities (severity_label, error_response)
│
├── frontend/
│   ├── templates/
│   │   └── index.html   # Jinja2 template (full SPA)
│   └── static/
│       ├── style.css    # SOC-style dark theme
│       └── script.js    # Vanilla JS client logic
│
├── data/
│   ├── generate_sample_logs.py   # Script to regenerate sample data
│   └── sample_logs.csv           # 241-row realistic log dataset
│
├── tests/
│   ├── __init__.py
│   └── test_model.py    # Unit tests for IsolationForest wrapper
│
├── requirements.txt
└── README.md
```

---

## Setup & Running

### 1. Clone / download the project

```bash
git clone <repo-url>
cd ai-log-anomaly-dashboard
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the server

```bash
uvicorn backend.main:app --reload
```

The app will be available at **http://127.0.0.1:8000**

### 5. Open the dashboard

Navigate to `http://127.0.0.1:8000` in your browser.

### 6. Try it with sample data

Upload `data/sample_logs.csv` using the file picker, then click **Detect Anomalies**.

---

## API Endpoints

### `GET /`
Serves the dashboard HTML page.

---

### `POST /upload`

Validates and previews an uploaded CSV file.

**Request:** `multipart/form-data` with field `file`

**Success response (200):**
```json
{
  "columns": ["timestamp", "ip", "status", "bytes"],
  "preview_rows": [...],
  "total_rows": 241
}
```

**Error responses:**

| Status | Condition |
|---|---|
| 400 | File missing or empty |
| 413 | File exceeds 50 MB |
| 422 | Not a `.csv` file, or CSV cannot be parsed |

---

### `POST /detect`

Runs the full preprocessing + Isolation Forest pipeline.

**Request:** `multipart/form-data` with field `file`

**Success response (200):**
```json
{
  "columns": ["timestamp", "ip", "status", "bytes", "anomaly_label", "anomaly_score", "severity"],
  "rows": [
    {
      "timestamp": "2024-06-15T00:00:00",
      "ip": "10.0.0.1",
      "status": 200,
      "bytes": 22854,
      "anomaly_label": 1,
      "anomaly_score": 0.082341,
      "severity": "LOW"
    }
  ],
  "summary": {
    "total": 241,
    "anomalies": 12,
    "high": 2,
    "medium": 3,
    "low": 7
  }
}
```

**Severity thresholds:**

| Score range | Severity |
|---|---|
| `score < -0.2` | HIGH |
| `-0.2 ≤ score < -0.1` | MEDIUM |
| `score ≥ -0.1` | LOW |

**Error responses:**

| Status | Condition |
|---|---|
| 400 | File missing or empty |
| 413 | File exceeds 50 MB |
| 422 | Not a `.csv` file, CSV cannot be parsed, or fewer than 2 usable feature columns |
| 500 | Internal model error |

---

## Screenshots

_Screenshots will be added after first deployment._

| View | Description |
|---|---|
| Upload | File picker and upload button |
| Preview | Data preview table (up to 10 rows) |
| Results | Color-coded anomaly detection results |
| Summary | Detection statistics cards |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Regenerating Sample Data

```bash
python data/generate_sample_logs.py
```
---

## 👤 Author

**Malakonda Chaitanya Bhargav**  
[LinkedIn](https://www.linkedin.com/in/chaitanya-bhargav-malakonda/)
