"""
scripts/evaluate.py
===================
Reproducible evaluation of the IsolationForest anomaly detector against
the planted ground-truth labels in data/sample_logs_labeled.csv.

What it does
------------
1. Loads data/sample_logs_labeled.csv (produced by data/generate_sample_logs.py).
2. Separates the is_anomaly ground-truth column from the feature columns.
3. Runs the *exact same* pipeline the API uses — preprocess() from
   backend.preprocess, then run_isolation_forest() from backend.model —
   no logic is duplicated or copy-pasted here.
4. Maps IsolationForest labels: -1 → predicted anomaly (1), 1 → normal (0).
5. Evaluates at contamination = 0.05, 0.10, 0.15, 0.20 and prints:
     - True anomaly rate of the labeled file
     - Per-contamination: confusion matrix, precision, recall, F1, FPR
   Saves everything to docs/evaluation.md.

Run from the project root:
    python scripts/evaluate.py

Honest caveat printed with every run:
    The dataset is fully synthetic.  Anomalies were planted deliberately
    to be detectable (high bytes, rare status codes, repeated source IP).
    These results measure the detector's ability to recover known signals,
    not its performance on real-world log data.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so backend imports work when the
# script is run from the project root as:  python scripts/evaluate.py
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.model import RANDOM_STATE, run_isolation_forest  # noqa: E402
from backend.preprocess import csv_bytes_to_dataframe, preprocess  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LABELED_CSV = PROJECT_ROOT / "data" / "sample_logs_labeled.csv"
DOCS_DIR    = PROJECT_ROOT / "docs"
OUTPUT_MD   = DOCS_DIR / "evaluation.md"

# Contamination values to sweep
CONTAMINATION_VALUES: list[float] = [0.05, 0.10, 0.15, 0.20]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CAVEAT = (
    "⚠  NOTE: The dataset is fully synthetic. "
    "Anomalies were planted deliberately to be detectable "
    "(high bytes, rare HTTP status codes, repeated source IP). "
    "These results measure the detector's ability to recover known signals, "
    "NOT its performance on real-world log data."
)

RECALL_CAP_NOTE = (
    "At low contamination the model is only allowed to flag a small fraction "
    "of rows as anomalies. When true anomalies exceed that fraction, "
    "recall is mechanically capped — the model cannot return more positives "
    "than the contamination parameter permits, regardless of how good its "
    "ranking is. The practical implication: setting contamination ≥ true "
    "anomaly rate is a necessary (though not sufficient) condition for "
    "achieving high recall in this unsupervised setup."
)


def _fpr(tn: int, fp: int) -> float:
    """False positive rate = FP / (FP + TN). Returns 0.0 if denominator is 0."""
    denom = fp + tn
    return fp / denom if denom > 0 else 0.0


def _confusion_parts(y_true: list[int], y_pred: list[int]) -> tuple[int, int, int, int]:
    """Return (TP, FP, FN, TN) from binary labels."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return int(tp), int(fp), int(fn), int(tn)


def evaluate_at_contamination(
    df_features: pd.DataFrame,
    y_true: list[int],
    contamination: float,
) -> dict:
    """Run the full pipeline at one contamination value and return metrics."""
    # Read through csv_bytes → preserves identical code path to the API.
    csv_bytes = df_features.to_csv(index=False).encode()
    df = csv_bytes_to_dataframe(csv_bytes)

    X, retained_cols, _ = preprocess(df)
    labels, _ = run_isolation_forest(X, contamination=contamination,
                                     random_state=RANDOM_STATE)

    # Map IsolationForest convention (-1 anomaly, 1 normal) → binary (1/0)
    y_pred = [1 if lbl == -1 else 0 for lbl in labels]

    tp, fp, fn, tn = _confusion_parts(y_true, y_pred)

    return {
        "contamination": contamination,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "fpr":       _fpr(tn, fp),
        "n_flagged": tp + fp,
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _bar(value: float, width: int = 20) -> str:
    """ASCII progress bar for a 0-1 value."""
    filled = round(value * width)
    return "[" + "█" * filled + "·" * (width - filled) + "]"


def _pct(value: float) -> str:
    return f"{value * 100:5.1f}%"


def _md_row(*cells: str) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _md_sep(n: int) -> str:
    return "|" + "|".join([":---:"] * n) + "|"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # Load labeled data
    # ------------------------------------------------------------------
    if not LABELED_CSV.exists():
        sys.exit(
            f"ERROR: {LABELED_CSV} not found.\n"
            "Run  python data/generate_sample_logs.py  first."
        )

    df_full = pd.read_csv(LABELED_CSV)
    y_true: list[int] = df_full["is_anomaly"].tolist()
    df_features = df_full.drop(columns=["is_anomaly"])

    n_total   = len(y_true)
    n_anomaly = sum(y_true)
    n_normal  = n_total - n_anomaly
    true_rate = n_anomaly / n_total

    # ------------------------------------------------------------------
    # Console header
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("  IsolationForest Anomaly Detector — Evaluation")
    print("=" * 70)
    print()
    print(CAVEAT)
    print()
    print(f"  Dataset : {LABELED_CSV.name}")
    print(f"  Rows    : {n_total}  ({n_normal} normal, {n_anomaly} planted anomalies)")
    print(f"  True anomaly rate: {true_rate:.1%}  ({n_anomaly}/{n_total})")
    print()

    # ------------------------------------------------------------------
    # Evaluate at each contamination value
    # ------------------------------------------------------------------
    results: list[dict] = []
    for c in CONTAMINATION_VALUES:
        r = evaluate_at_contamination(df_features, y_true, c)
        results.append(r)

    # ------------------------------------------------------------------
    # Per-contamination detail blocks
    # ------------------------------------------------------------------
    for r in results:
        c = r["contamination"]
        print("-" * 70)
        print(f"  Contamination = {c:.2f}  (model flags ~{round(c * n_total)} of {n_total} rows)")
        print()
        print(f"  Confusion matrix")
        print(f"                      Predicted Normal  Predicted Anomaly")
        print(f"  Actual Normal   :        {r['tn']:>4d}                {r['fp']:>4d}")
        print(f"  Actual Anomaly  :        {r['fn']:>4d}                {r['tp']:>4d}")
        print()
        print(f"  Precision  {_pct(r['precision'])}  {_bar(r['precision'])}")
        print(f"  Recall     {_pct(r['recall'])}  {_bar(r['recall'])}")
        print(f"  F1 score   {_pct(r['f1'])}  {_bar(r['f1'])}")
        print(f"  FPR        {_pct(r['fpr'])}  {_bar(r['fpr'])}")
        print()

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    print("-" * 70)
    print("  Contamination comparison")
    print()
    hdr = f"  {'Contam':>7}  {'Flagged':>7}  {'Precision':>9}  {'Recall':>9}  {'F1':>9}  {'FPR':>9}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in results:
        print(
            f"  {r['contamination']:>7.2f}  "
            f"{r['n_flagged']:>7d}  "
            f"{_pct(r['precision']):>9}  "
            f"{_pct(r['recall']):>9}  "
            f"{_pct(r['f1']):>9}  "
            f"{_pct(r['fpr']):>9}"
        )
    print()
    print("  NOTE:", RECALL_CAP_NOTE)
    print()
    print("=" * 70)

    # ------------------------------------------------------------------
    # Write docs/evaluation.md
    # ------------------------------------------------------------------
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    md_lines: list[str] = []
    a = md_lines.append

    a("# Anomaly Detector — Evaluation Report")
    a("")
    a("> **Honest caveat**")
    a(">")
    a(f"> {CAVEAT.replace('⚠  NOTE: ', '')}")
    a("")
    a("## Dataset")
    a("")
    a(f"| | |")
    a(f"|---|---|")
    a(f"| File | `{LABELED_CSV.name}` |")
    a(f"| Total rows | {n_total} |")
    a(f"| Normal rows | {n_normal} |")
    a(f"| Planted anomaly rows | {n_anomaly} |")
    a(f"| True anomaly rate | {true_rate:.1%} |")
    a("")
    a("Anomaly sub-types planted in the data:")
    a("")
    a("- **Type 1 — high bytes** (5 rows): `bytes` in \\[500 001, 5 000 000\\]")
    a("- **Type 2 — rare status codes** (3 rows): `status` ∈ {500, 503, 429}")
    a("- **Type 3 — repeated source IP** (31 rows): all from `45.33.32.156`")
    a("")
    a("## Pipeline (identical to the API)")
    a("")
    a("```")
    a("csv_bytes_to_dataframe(bytes)")
    a("  → preprocess(df)          # type-detect, freq-encode IPs,")
    a("                            # datetime-expand timestamps, MinMaxScale")
    a("  → run_isolation_forest(X, contamination=c, random_state=42)")
    a("  → map label -1 → 1 (anomaly), 1 → 0 (normal)")
    a("```")
    a("")
    a("## Results by contamination")
    a("")
    a("> **Why recall is capped at low contamination**")
    a(">")
    a(f"> {RECALL_CAP_NOTE}")
    a("")

    for r in results:
        c = r["contamination"]
        a(f"### contamination = {c:.2f}")
        a("")
        a(f"Model flags approximately **{r['n_flagged']}** of {n_total} rows as anomalies.")
        a("")
        a("**Confusion matrix**")
        a("")
        a("| | Predicted Normal | Predicted Anomaly |")
        a("|---|:---:|:---:|")
        a(f"| **Actual Normal** | {r['tn']} | {r['fp']} |")
        a(f"| **Actual Anomaly** | {r['fn']} | {r['tp']} |")
        a("")
        a("| Metric | Value |")
        a("|---|---|")
        a(f"| Precision | {r['precision']:.3f} |")
        a(f"| Recall | {r['recall']:.3f} |")
        a(f"| F1 score | {r['f1']:.3f} |")
        a(f"| False positive rate | {r['fpr']:.3f} |")
        a(f"| True positives | {r['tp']} |")
        a(f"| False positives | {r['fp']} |")
        a(f"| False negatives | {r['fn']} |")
        a(f"| True negatives | {r['tn']} |")
        a("")

    # Comparison table
    a("## Contamination comparison")
    a("")
    cols = ["Contamination", "Rows flagged", "Precision", "Recall", "F1", "FPR"]
    a(_md_row(*cols))
    a(_md_sep(len(cols)))
    for r in results:
        a(_md_row(
            f"{r['contamination']:.2f}",
            str(r["n_flagged"]),
            f"{r['precision']:.3f}",
            f"{r['recall']:.3f}",
            f"{r['f1']:.3f}",
            f"{r['fpr']:.3f}",
        ))
    a("")
    # Assumes: results = list of dicts sorted by contamination, each with keys
    # "contamination", "flagged", "tp", "fp", "precision", "recall", "f1", "fpr"
    first, last = results[0], results[-1]
    best = max(results, key=lambda r: r["f1"])

    a("## Interpretation")
    a("")
    a(
        "At **contamination = {c:.2f}** the model flags {flagged} rows, of which "
        "{tp} are true anomalies (precision {p:.3f}, {fp} false positives). "
        "Recall is {r:.3f} because the true anomaly rate is {rate:.1%} and the "
        "model is only allowed to flag {c:.0%} of rows at this setting.".format(
            c=first["contamination"], flagged=first["n_flagged"], tp=first["tp"],
            p=first["precision"], fp=first["fp"], r=first["recall"], rate=true_rate,
        )
    )
    a("")
    a(
        "Across the sweep, recall goes from {r0:.3f} to {r1:.3f} and precision "
        "goes from {p0:.3f} to {p1:.3f}: a higher contamination lets the model "
        "flag more rows, which finds more anomalies but also pulls in "
        "borderline-normal rows.".format(
            r0=first["recall"], r1=last["recall"],
            p0=first["precision"], p1=last["precision"],
        )
    )
    a("")
    if best is last:
        a(
            "F1 is highest at **{c:.2f}** (F1 = {f1:.3f}), the largest value "
            "tested, and was still rising at the last point, so the true optimum "
            "is not established by this sweep.".format(
                c=best["contamination"], f1=best["f1"]
            )
        )
    else:
        a(
            "F1 is highest at **{c:.2f}** (F1 = {f1:.3f}) among the values "
            "tested.".format(c=best["contamination"], f1=best["f1"])
        )
    a("")
    a(
        "At that setting the model reports {tp} true anomalies with {fp} false "
        "positives (FPR {fpr:.3f}). A real user would not know the true anomaly "
        "rate in advance, so contamination has to be chosen from domain "
        "knowledge or by reviewing the flagged rows.".format(
            tp=best["tp"], fp=best["fp"], fpr=best["fpr"]
        )
    )
    a("")
    a(
        "Because IsolationForest scores every row relative to the current "
        "batch, results will shift on other datasets, especially larger ones or "
        "ones with subtler anomalies. These numbers describe this synthetic "
        "file only."
    )
    a("")
    a("---")
    a("")
    a("*Generated by `scripts/evaluate.py` — re-run to refresh.*")

    OUTPUT_MD.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"  Saved → {OUTPUT_MD}")
    print()


if __name__ == "__main__":
    main()
