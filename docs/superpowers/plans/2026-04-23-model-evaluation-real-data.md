# Model Evaluation on Real Data — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `tests/evaluate_model.py` that evaluates the fraud detection model on all 284 807 Kaggle transactions (Phase 1 — ML metrics + saved files) and sends a representative sample through the live API (Phase 2 — monitoring feed).

**Architecture:** Phase 1 loads the model and scaler directly, preprocesses the full CSV identically to `app/model.py`, computes sklearn metrics, and saves CSV + 2 PNG files. Phase 2 checks API availability, builds a stratified sample (all 492 frauds + 500 random legits), POSTs each transaction, and prints a per-class accuracy summary.

**Tech Stack:** pandas, numpy, scikit-learn, joblib, matplotlib, seaborn, requests, pytest

---

## File Map

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `tests/evaluate_model.py` | Full evaluation script (both phases) |
| Create | `tests/test_evaluate_model.py` | Unit tests for preprocessing and Phase 2 fallback |
| Auto-created | `evaluation_output/` | report.csv, confusion_matrix.png, roc_curve.png |

---

### Task 1: Preprocessing utility — `build_feature_matrix`

**Files:**
- Create: `tests/evaluate_model.py`
- Create: `tests/test_evaluate_model.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_evaluate_model.py`:

```python
import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.evaluate_model import build_feature_matrix


class DummyScaler:
    def transform(self, X):
        return X * 2.0


def make_dummy_df(n=5):
    data = {"Time": np.ones(n), "Amount": np.ones(n) * 3.0, "Class": [0, 1, 0, 1, 0]}
    for i in range(1, 29):
        data[f"V{i}"] = np.ones(n) * float(i)
    return pd.DataFrame(data)


def test_build_feature_matrix_shape():
    df = make_dummy_df()
    X, y = build_feature_matrix(df, DummyScaler())
    assert X.shape == (5, 30), f"Expected (5, 30), got {X.shape}"
    assert y.shape == (5,)


def test_build_feature_matrix_column_order():
    df = make_dummy_df(n=1)
    X, y = build_feature_matrix(df, DummyScaler())
    # scaled_amount = 3.0 * 2 = 6.0, scaled_time = 1.0 * 2 = 2.0
    assert X[0, 0] == pytest.approx(6.0), "Column 0 must be scaled_amount"
    assert X[0, 1] == pytest.approx(2.0), "Column 1 must be scaled_time"
    assert X[0, 2] == pytest.approx(1.0), "Column 2 must be V1 (unscaled)"
    assert X[0, 29] == pytest.approx(28.0), "Column 29 must be V28"


def test_build_feature_matrix_labels():
    df = make_dummy_df()
    _, y = build_feature_matrix(df, DummyScaler())
    np.testing.assert_array_equal(y, [0, 1, 0, 1, 0])
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /Users/vivienbistrel/Desktop/Anomaly
python -m pytest tests/test_evaluate_model.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError` or `ImportError` — `evaluate_model` does not exist yet.

- [ ] **Step 3: Create `tests/evaluate_model.py` with `build_feature_matrix`**

```python
import os
import time
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH    = os.path.join(PROJECT_ROOT, "data", "creditcard.csv")
MODEL_PATH   = os.path.join(PROJECT_ROOT, "app", "random_forest_fraud_model.joblib")
SCALER_PATH  = os.path.join(PROJECT_ROOT, "app", "robust_scaler.joblib")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "evaluation_output")
API_URL      = "http://127.0.0.1:8001/predict"
API_HEALTH   = "http://127.0.0.1:8001/"


def build_feature_matrix(df: pd.DataFrame, scaler) -> tuple:
    """Replicates app/model.py preprocessing exactly: scale Amount and Time, append V1-V28."""
    scaled_amount = scaler.transform(df[["Amount"]].values)[:, 0]
    scaled_time   = scaler.transform(df[["Time"]].values)[:, 0]
    v_cols        = np.column_stack([df[f"V{i}"].values for i in range(1, 29)])
    X = np.column_stack([scaled_amount, scaled_time, v_cols])
    y = df["Class"].values
    return X, y
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_evaluate_model.py::test_build_feature_matrix_shape \
                 tests/test_evaluate_model.py::test_build_feature_matrix_column_order \
                 tests/test_evaluate_model.py::test_build_feature_matrix_labels -v
```

Expected: 3 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add tests/evaluate_model.py tests/test_evaluate_model.py
git commit -m "feat: add evaluate_model scaffold with build_feature_matrix"
```

---

### Task 2: Phase 1 — ML evaluation and output files

**Files:**
- Modify: `tests/evaluate_model.py` — add `run_phase1` and `_save_outputs`
- Modify: `tests/test_evaluate_model.py` — add output file tests

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_evaluate_model.py`:

```python
import tempfile
from unittest.mock import patch


def test_save_outputs_creates_files():
    y_true = np.array([0, 0, 1, 1, 0])
    y_pred = np.array([0, 0, 1, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.9, 0.4, 0.1])
    cm     = np.array([[3, 0], [1, 1]])
    auc    = 0.875
    df     = make_dummy_df()

    from tests.evaluate_model import _save_outputs

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("tests.evaluate_model.OUTPUT_DIR", tmpdir):
            _save_outputs(df, y_true, y_pred, y_prob, cm, auc)
        files = os.listdir(tmpdir)
        assert "report.csv" in files
        assert "confusion_matrix.png" in files
        assert "roc_curve.png" in files


def test_save_outputs_csv_columns():
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    y_prob = np.array([0.05, 0.95])
    cm     = np.array([[1, 0], [0, 1]])
    auc    = 1.0
    df     = make_dummy_df(n=2)

    from tests.evaluate_model import _save_outputs

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("tests.evaluate_model.OUTPUT_DIR", tmpdir):
            _save_outputs(df, y_true, y_pred, y_prob, cm, auc)
        result = pd.read_csv(os.path.join(tmpdir, "report.csv"))
        assert set(["Amount", "true_label", "predicted_label", "fraud_probability"]).issubset(result.columns)
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_evaluate_model.py::test_save_outputs_creates_files \
                 tests/test_evaluate_model.py::test_save_outputs_csv_columns -v
```

Expected: `ImportError` — `_save_outputs` not defined yet.

- [ ] **Step 3: Add `_save_outputs` and `run_phase1` to `tests/evaluate_model.py`**

Append after `build_feature_matrix`:

```python
def _save_outputs(
    df: pd.DataFrame,
    y: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    cm: np.ndarray,
    auc: float,
):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # CSV report
    out = df[["Amount"]].copy()
    out.index.name        = "id"
    out["true_label"]        = y
    out["predicted_label"]   = y_pred
    out["fraud_probability"] = y_prob
    out.to_csv(os.path.join(OUTPUT_DIR, "report.csv"))

    # Confusion matrix heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Normal", "Fraude"],
                yticklabels=["Normal", "Fraude"])
    ax.set_xlabel("Prediction")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plt.close(fig)

    # ROC curve
    fpr, tpr, _ = roc_curve(y, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"))
    plt.close(fig)

    print(f"\nFiles saved to {OUTPUT_DIR}/")
    print("  report.csv | confusion_matrix.png | roc_curve.png")


def run_phase1(model, scaler) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("PHASE 1 - FULL ML EVALUATION")
    print("=" * 60)

    t0 = time.time()
    df = pd.read_csv(DATA_PATH)
    X, y = build_feature_matrix(df, scaler)

    n_fraud = int(y.sum())
    n_legit = len(y) - n_fraud
    print(f"Dataset : {len(y):,} transactions ({n_fraud} frauds, {n_legit:,} legit)")

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    cm     = confusion_matrix(y, y_pred)
    auc    = roc_auc_score(y, y_prob)

    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=["Normal", "Fraud"]))
    print(f"ROC-AUC Score      : {auc:.4f}")
    print(f"Evaluation time    : {time.time() - t0:.1f}s")

    _save_outputs(df, y, y_pred, y_prob, cm, auc)
    return df
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_evaluate_model.py::test_save_outputs_creates_files \
                 tests/test_evaluate_model.py::test_save_outputs_csv_columns -v
```

Expected: 2 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add tests/evaluate_model.py tests/test_evaluate_model.py
git commit -m "feat: add Phase 1 ML evaluation with CSV and plot outputs"
```

---

### Task 3: Phase 2 — API test with graceful fallback

**Files:**
- Modify: `tests/evaluate_model.py` — add `run_phase2`
- Modify: `tests/test_evaluate_model.py` — add API fallback test

- [ ] **Step 1: Write the failing test**

Append to `tests/test_evaluate_model.py`:

```python
import io
from contextlib import redirect_stdout


def test_phase2_skips_gracefully_when_api_down():
    df = make_dummy_df(n=10)
    # Add more legit rows so sample(500) doesn't fail — use the real function with a patched n
    df_large = pd.concat([df] * 100).reset_index(drop=True)
    df_large.loc[0, "Class"] = 1   # ensure at least one fraud

    from tests.evaluate_model import run_phase2
    import unittest.mock as mock

    # Simulate connection refused
    with mock.patch("tests.evaluate_model.requests.get",
                    side_effect=ConnectionError("refused")):
        buf = io.StringIO()
        with redirect_stdout(buf):
            run_phase2(df_large)
        output = buf.getvalue()

    assert "Phase 2" in output or "non accessible" in output or "ignored" in output.lower()
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_evaluate_model.py::test_phase2_skips_gracefully_when_api_down -v
```

Expected: `ImportError` — `run_phase2` not defined yet.

- [ ] **Step 3: Add `run_phase2` to `tests/evaluate_model.py`**

Append after `run_phase1`:

```python
def run_phase2(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("PHASE 2 - API TEST")
    print("=" * 60)

    try:
        requests.get(API_HEALTH, timeout=3).raise_for_status()
    except Exception:
        print("API not reachable at http://127.0.0.1:8001 — Phase 2 ignored.")
        print("Start the API with `docker-compose up` then re-run.")
        return

    frauds = df[df["Class"] == 1]
    legits = df[df["Class"] == 0].sample(n=500, random_state=42)
    sample = pd.concat([frauds, legits]).sample(frac=1, random_state=42).reset_index(drop=True)

    n_fraud = len(frauds)
    n_legit = len(legits)
    total   = len(sample)
    print(f"Sending {total} transactions ({n_fraud} frauds + {n_legit} legit)...")

    v_cols          = [f"V{i}" for i in range(1, 29)]
    correct         = 0
    fraud_correct   = 0
    legit_correct   = 0
    latencies: list = []

    for _, row in sample.iterrows():
        payload = {"Time": float(row["Time"]), "Amount": float(row["Amount"]),
                   **{c: float(row[c]) for c in v_cols}}
        t0 = time.time()
        try:
            resp = requests.post(API_URL, json=payload, timeout=10)
            latencies.append(time.time() - t0)
            if resp.status_code == 200:
                pred  = resp.json()["is_fraud"]
                truth = bool(row["Class"])
                if pred == truth:
                    correct += 1
                    if truth:
                        fraud_correct += 1
                    else:
                        legit_correct += 1
        except Exception as e:
            print(f"  Warning: {e}")

    avg_lat_ms = (sum(latencies) / len(latencies) * 1000) if latencies else 0.0

    print(f"\n=== API SUMMARY ({total} transactions) ===")
    print(f"Frauds correctly detected  : {fraud_correct}/{n_fraud} ({fraud_correct/n_fraud*100:.1f}%)")
    print(f"Legit correctly classified : {legit_correct}/{n_legit} ({legit_correct/n_legit*100:.1f}%)")
    print(f"Overall accuracy           : {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"Average latency            : {avg_lat_ms:.1f}ms")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_evaluate_model.py::test_phase2_skips_gracefully_when_api_down -v
```

Expected: PASSED.

- [ ] **Step 5: Run full test suite**

```bash
python -m pytest tests/test_evaluate_model.py -v
```

Expected: all 5 tests PASSED.

- [ ] **Step 6: Commit**

```bash
git add tests/evaluate_model.py tests/test_evaluate_model.py
git commit -m "feat: add Phase 2 API test with graceful fallback"
```

---

### Task 4: Wire `main()` and run end-to-end

**Files:**
- Modify: `tests/evaluate_model.py` — add `main()` and `if __name__ == "__main__"` block

- [ ] **Step 1: Append `main()` to `tests/evaluate_model.py`**

```python
def main():
    print("=" * 60)
    print("FRAUD DETECTION MODEL — FULL EVALUATION")
    print("=" * 60)

    print("\nLoading model and scaler...")
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and scaler loaded.")

    df = run_phase1(model, scaler)
    run_phase2(df)

    print("\nDone.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full test suite one more time**

```bash
python -m pytest tests/test_evaluate_model.py -v
```

Expected: 5 tests PASSED.

- [ ] **Step 3: Dry-run Phase 1 only (no API needed)**

```bash
cd /Users/vivienbistrel/Desktop/Anomaly
python tests/evaluate_model.py 2>&1 | tail -20
```

Expected output (approximate):
```
PHASE 1 - FULL ML EVALUATION
Dataset : 284,807 transactions (492 frauds, 284315 legit)
...
ROC-AUC Score      : 0.99xx
Evaluation time    : xx.xs
Files saved to .../evaluation_output/
  report.csv | confusion_matrix.png | roc_curve.png
PHASE 2 - API TEST
API not reachable at http://127.0.0.1:8001 — Phase 2 ignored.
```

- [ ] **Step 4: Verify output files exist**

```bash
ls /Users/vivienbistrel/Desktop/Anomaly/evaluation_output/
```

Expected: `confusion_matrix.png  report.csv  roc_curve.png`

- [ ] **Step 5: Commit**

```bash
git add tests/evaluate_model.py evaluation_output/
git commit -m "feat: wire main() for full evaluation — Phase 1 confirmed working"
```

---

### Task 5: Run Phase 2 with live API (optional — requires Docker)

This task requires the full stack to be running.

- [ ] **Step 1: Start the stack**

```bash
docker-compose up -d
```

Wait ~10 seconds for the API and PostgreSQL to be ready, then check:

```bash
curl -s http://127.0.0.1:8001/ | python -m json.tool
```

Expected: `{"message": "✅ Fraud Detection API is UP and Running!"}`

- [ ] **Step 2: Run the full evaluation**

```bash
python tests/evaluate_model.py
```

Expected Phase 2 output (approximate):
```
PHASE 2 - API TEST
Sending 992 transactions (492 frauds + 500 legit)...

=== API SUMMARY (992 transactions) ===
Frauds correctly detected  : 48x/492 (9x.x%)
Legit correctly classified : 49x/500 (9x.x%)
Overall accuracy           : 97x/992 (9x.x%)
Average latency            : xxms
```

- [ ] **Step 3: Check Grafana dashboard**

Open `http://localhost:3000` in a browser. Verify that:
- `fraud_predictions_total` counter increased
- `fraud_probability_score` histogram shows a bimodal distribution (low prob for legit, high for fraud)

- [ ] **Step 4: Final commit**

```bash
git add tests/evaluate_model.py
git commit -m "feat: evaluate_model complete — Phase 1 + Phase 2 verified"
```
