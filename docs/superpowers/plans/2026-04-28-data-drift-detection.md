# Data Drift Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement KS-based data drift detection that compares the last 150 live transactions against the training distribution for 7 key features, and exposes results via Prometheus Gauges and a `GET /drift` endpoint.

**Architecture:** `scripts/generate_baseline.py` is run once to produce `app/drift_baseline.pkl` (numpy arrays of training distributions). `app/drift.py` implements `DriftDetector` which loads that baseline and runs `scipy.stats.ks_2samp` on 7 features. `app/main.py` gains a thread-safe counter that triggers drift detection every 150 predictions, updates Prometheus `Gauge` metrics, and serves a `GET /drift` endpoint returning the latest status.

**Tech Stack:** Python 3.11, scipy (ks_2samp), joblib, numpy, prometheus-client (Gauge), FastAPI, SQLAlchemy, pytest

---

### Task 1: Baseline Generator Script

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/generate_baseline.py`
- Create: `tests/test_drift.py` (first test only)

- [ ] **Step 1: Write the failing test**

Create `tests/test_drift.py`:

```python
# tests/test_drift.py
"""Unit tests for data drift detection — drift.py and generate_baseline.py."""

import os

import joblib
import numpy as np
import pandas as pd
import pytest

from scripts.generate_baseline import generate_baseline

FEATURES = ["V4", "V10", "V11", "V12", "V14", "V16", "Amount"]

_ALL_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time", "Class"]


def _make_dummy_csv(path: str, n: int = 50) -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(0, 1, (n, len(_ALL_COLS))), columns=_ALL_COLS)
    df["Class"] = 0
    df.to_csv(path, index=False)


def test_generate_baseline_creates_file(tmp_path):
    csv_path = str(tmp_path / "creditcard.csv")
    pkl_path = str(tmp_path / "drift_baseline.pkl")
    _make_dummy_csv(csv_path, n=50)

    generate_baseline(csv_path, pkl_path)

    assert os.path.exists(pkl_path)
    loaded = joblib.load(pkl_path)
    for f in FEATURES:
        assert f in loaded
        assert len(loaded[f]) == 50
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
PYTHONPATH=. pytest tests/test_drift.py::test_generate_baseline_creates_file -v
```

Expected: `FAILED` — `ModuleNotFoundError: No module named 'scripts'`

- [ ] **Step 3: Create the scripts package**

```bash
mkdir -p scripts && touch scripts/__init__.py
```

- [ ] **Step 4: Write scripts/generate_baseline.py**

```python
# scripts/generate_baseline.py
"""Generate training baseline distributions for data drift detection.

Reads data/creditcard.csv, extracts the 7 monitored features (V4, V10,
V11, V12, V14, V16, Amount), and saves their raw value arrays to
app/drift_baseline.pkl using joblib.

Run this script once before starting the API.

Usage:
    python scripts/generate_baseline.py
"""

import os

import joblib
import pandas as pd

FEATURES = ["V4", "V10", "V11", "V12", "V14", "V16", "Amount"]

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "creditcard.csv")
_OUTPUT_PATH = os.path.join(_PROJECT_ROOT, "app", "drift_baseline.pkl")


def generate_baseline(data_path: str, output_path: str) -> dict:
    """Extract feature arrays from the training CSV and persist them.

    Args:
        data_path: Path to creditcard.csv.
        output_path: Destination path for drift_baseline.pkl.

    Returns:
        Dict mapping each feature name to its numpy array.
    """
    df = pd.read_csv(data_path)
    baseline = {f: df[f].values for f in FEATURES}
    joblib.dump(baseline, output_path)
    print(f"Baseline saved to {output_path}")
    print(f"  Rows : {len(df):,}")
    for f, arr in baseline.items():
        print(f"  {f:8s}: mean={arr.mean():.4f}  std={arr.std():.4f}")
    return baseline


if __name__ == "__main__":
    generate_baseline(_DATA_PATH, _OUTPUT_PATH)
```

- [ ] **Step 5: Run the test — it should pass**

```bash
PYTHONPATH=. pytest tests/test_drift.py::test_generate_baseline_creates_file -v
```

Expected: `PASSED`

- [ ] **Step 6: Commit**

```bash
git add scripts/__init__.py scripts/generate_baseline.py tests/test_drift.py
git commit -m "feat: add baseline generator script for drift detection"
```

---

### Task 2: DriftDetector class

**Files:**
- Modify: `app/drift.py` (replace placeholder with full implementation)
- Modify: `tests/test_drift.py` (append 4 tests)

- [ ] **Step 1: Append 4 failing tests to tests/test_drift.py**

Add these functions at the end of `tests/test_drift.py`:

```python
from app.drift import DriftDetector


def _make_baseline_pkl(tmp_path, n: int = 500) -> str:
    rng = np.random.default_rng(42)
    baseline = {f: rng.normal(0, 1, n) for f in FEATURES}
    pkl = str(tmp_path / "baseline.pkl")
    joblib.dump(baseline, pkl)
    return pkl


def _make_records(baseline: dict, n: int = 150) -> list[dict]:
    rng = np.random.default_rng(42)
    return [
        {f: float(rng.choice(baseline[f])) for f in FEATURES}
        for _ in range(n)
    ]


def test_is_ready_false_when_no_baseline(tmp_path):
    detector = DriftDetector(baseline_path=str(tmp_path / "nonexistent.pkl"))
    assert detector.is_ready is False
    result = detector.compute_drift([])
    assert result == {"status": "baseline_not_loaded"}


def test_compute_drift_no_drift(tmp_path):
    pkl = _make_baseline_pkl(tmp_path)
    baseline = joblib.load(pkl)
    detector = DriftDetector(baseline_path=pkl)

    records = _make_records(baseline)
    result = detector.compute_drift(records)

    assert result["drift_detected"] is False
    for f in FEATURES:
        assert result["results"][f]["drift_detected"] is False


def test_compute_drift_drift_detected(tmp_path):
    pkl = _make_baseline_pkl(tmp_path)
    detector = DriftDetector(baseline_path=pkl)

    rng = np.random.default_rng(99)
    records = [{f: float(rng.normal(100, 1)) for f in FEATURES} for _ in range(150)]
    result = detector.compute_drift(records)

    assert result["drift_detected"] is True
    assert any(result["results"][f]["drift_detected"] for f in FEATURES)


def test_compute_drift_result_structure(tmp_path):
    pkl = _make_baseline_pkl(tmp_path)
    baseline = joblib.load(pkl)
    detector = DriftDetector(baseline_path=pkl)

    records = _make_records(baseline)
    result = detector.compute_drift(records)

    assert "window" in result
    assert "features_monitored" in result
    assert "results" in result
    assert "drift_detected" in result
    assert "last_checked_at" in result
    for f in FEATURES:
        r = result["results"][f]
        assert "statistic" in r
        assert "p_value" in r
        assert "drift_detected" in r
```

- [ ] **Step 2: Run the 4 new tests to confirm they fail**

```bash
PYTHONPATH=. pytest tests/test_drift.py -k "not generate_baseline" -v
```

Expected: 4 `FAILED` — `ImportError: cannot import name 'DriftDetector' from 'app.drift'`

- [ ] **Step 3: Replace app/drift.py with full implementation**

```python
# app/drift.py
"""Data drift detection using Kolmogorov-Smirnov two-sample test.

Pipeline step: Step 7 - Data drift detection.
"""

import datetime
import logging
import os
from typing import Any

import joblib
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

FEATURES: list[str] = ["V4", "V10", "V11", "V12", "V14", "V16", "Amount"]
DRIFT_WINDOW: int = 150
P_VALUE_THRESHOLD: float = 0.05

_BASELINE_PATH = os.path.join(os.path.dirname(__file__), "drift_baseline.pkl")

DRIFT_STATUS: dict[str, Any] = {}


class DriftDetector:
    """Detects covariate drift via KS test against a pre-computed training baseline.

    Args:
        baseline_path: Path to the .pkl file produced by generate_baseline.py.
    """

    def __init__(self, baseline_path: str = _BASELINE_PATH) -> None:
        self.baseline: dict[str, np.ndarray] = {}
        self._load_baseline(baseline_path)

    def _load_baseline(self, path: str) -> None:
        if not os.path.exists(path):
            logger.warning(
                "drift_baseline.pkl not found at %s — drift detection disabled.", path
            )
            return
        self.baseline = joblib.load(path)
        logger.info("Drift baseline loaded: %d features.", len(self.baseline))

    @property
    def is_ready(self) -> bool:
        return bool(self.baseline)

    def compute_drift(self, records: list[dict]) -> dict[str, Any]:
        """Run KS test for each monitored feature against the training baseline.

        Args:
            records: List of feature dicts from TransactionRecord.features.

        Returns:
            Updated DRIFT_STATUS dict, or {"status": "baseline_not_loaded"}.
        """
        global DRIFT_STATUS  # noqa: PLW0603

        if not self.is_ready:
            return {"status": "baseline_not_loaded"}

        results: dict[str, Any] = {}
        any_drift = False

        for feature in FEATURES:
            baseline_arr = self.baseline.get(feature)
            if baseline_arr is None:
                results[feature] = {"error": f"{feature} missing from baseline"}
                continue

            live_values = [
                float(rec[feature])
                for rec in records
                if rec.get(feature) is not None
            ]
            if len(live_values) < 2:
                results[feature] = {"error": "insufficient live data"}
                continue

            try:
                stat, p_value = stats.ks_2samp(baseline_arr, np.array(live_values))
                drift = p_value < P_VALUE_THRESHOLD
                if drift:
                    any_drift = True
                results[feature] = {
                    "statistic": round(float(stat), 6),
                    "p_value": round(float(p_value), 6),
                    "drift_detected": drift,
                }
            except Exception as exc:  # pragma: no cover
                logger.error("KS test error for %s: %s", feature, exc)
                results[feature] = {"drift_detected": False, "error": str(exc)}

        DRIFT_STATUS.update({
            "window": DRIFT_WINDOW,
            "features_monitored": FEATURES,
            "results": results,
            "drift_detected": any_drift,
            "last_checked_at": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
        })

        return DRIFT_STATUS
```

- [ ] **Step 4: Run the 4 tests — they should pass**

```bash
PYTHONPATH=. pytest tests/test_drift.py -k "not generate_baseline" -v
```

Expected: 4 `PASSED`

- [ ] **Step 5: Run all 5 drift tests**

```bash
PYTHONPATH=. pytest tests/test_drift.py -v
```

Expected: 5 `PASSED`

- [ ] **Step 6: Commit**

```bash
git add app/drift.py tests/test_drift.py
git commit -m "feat: implement DriftDetector with KS test"
```

---

### Task 3: FastAPI integration — counter, Prometheus Gauges, /drift endpoint

**Files:**
- Modify: `app/main.py`
- Modify: `tests/test_drift.py` (append 2 endpoint tests)

- [ ] **Step 1: Append 2 failing endpoint tests to tests/test_drift.py**

Add at the end of `tests/test_drift.py`:

```python
def test_drift_endpoint_not_checked_yet(monkeypatch):
    import app.drift as drift_module
    monkeypatch.setattr(drift_module, "DRIFT_STATUS", {})

    from fastapi.testclient import TestClient
    from app.main import app as fastapi_app

    client = TestClient(fastapi_app)
    response = client.get("/drift")
    assert response.status_code == 200
    assert response.json() == {"status": "not_checked_yet"}


def test_drift_endpoint_returns_full_status(monkeypatch):
    import app.drift as drift_module

    fake_status = {
        "window": 150,
        "features_monitored": FEATURES,
        "results": {
            f: {"statistic": 0.05, "p_value": 0.50, "drift_detected": False}
            for f in FEATURES
        },
        "drift_detected": False,
        "last_checked_at": "2026-04-28T12:00:00+00:00",
    }
    monkeypatch.setattr(drift_module, "DRIFT_STATUS", fake_status)

    from fastapi.testclient import TestClient
    from app.main import app as fastapi_app

    client = TestClient(fastapi_app)
    response = client.get("/drift")
    assert response.status_code == 200
    data = response.json()
    assert data["drift_detected"] is False
    assert "results" in data
    assert "last_checked_at" in data
    for f in FEATURES:
        assert f in data["results"]
```

- [ ] **Step 2: Run the 2 new tests to confirm they fail**

```bash
PYTHONPATH=. pytest tests/test_drift.py::test_drift_endpoint_not_checked_yet tests/test_drift.py::test_drift_endpoint_returns_full_status -v
```

Expected: 2 `FAILED` — `404 Not Found` (endpoint doesn't exist yet)

- [ ] **Step 3: Update imports at the top of app/main.py**

Change line 19:
```python
# Before
from prometheus_client import Counter, Histogram
```
```python
# After
import threading

from prometheus_client import Counter, Gauge, Histogram
```

Add after line 25 (`from app.schemas import PredictionOutput, TransactionInput`):
```python
import app.drift as _drift_module
from app.drift import DriftDetector, DRIFT_WINDOW
```

- [ ] **Step 4: Add Drift Gauge and prediction counter after the existing Prometheus metrics (after line 53)**

Insert after the `PREDICTION_LATENCY = Histogram(...)` block:

```python
DATA_DRIFT_DETECTED = Gauge(
    "data_drift_detected",
    "1 if KS drift detected for a feature in the last window, 0 otherwise.",
    ["feature"],
)

_prediction_count: int = 0
_count_lock: threading.Lock = threading.Lock()
```

- [ ] **Step 5: Add drift_detector global and initialise it in the lifespan**

Change line 60:
```python
# Before
# Holds the ML model in memory for the lifetime of the process.
fraud_detector: FraudDetector | None = None
```
```python
# After
# Holds the ML model and drift detector in memory for the lifetime of the process.
fraud_detector: FraudDetector | None = None
drift_detector: DriftDetector | None = None
```

In the lifespan function, change:
```python
# Before
    global fraud_detector  # noqa: PLW0603
```
```python
# After
    global fraud_detector  # noqa: PLW0603
    global drift_detector  # noqa: PLW0603
```

After the `fraud_detector = FraudDetector()` try/except block, add:
```python
    logger.info("Initialising drift detector…")
    drift_detector = DriftDetector()
    if drift_detector.is_ready:
        logger.info("Drift detector ready.")
    else:
        logger.warning("Drift baseline not found — drift detection disabled.")
```

Change the shutdown section:
```python
# Before
    # Shutdown: release model memory.
    fraud_detector = None
```
```python
# After
    # Shutdown: release model memory.
    fraud_detector = None
    drift_detector = None
```

- [ ] **Step 6: Add drift trigger inside /predict, after db.refresh(db_record)**

After the line `db.refresh(db_record)` and before `return PredictionOutput(...)`, insert:

```python
        # Drift detection trigger: every DRIFT_WINDOW predictions
        global _prediction_count  # noqa: PLW0603
        with _count_lock:
            _prediction_count += 1
            should_check = _prediction_count >= DRIFT_WINDOW
            if should_check:
                _prediction_count = 0

        if should_check and drift_detector is not None and drift_detector.is_ready:
            recent = (
                db.query(TransactionRecord)
                .order_by(TransactionRecord.id.desc())
                .limit(DRIFT_WINDOW)
                .all()
            )
            drift_results = drift_detector.compute_drift(
                [r.features for r in recent]
            )
            if "results" in drift_results:
                for feat, info in drift_results["results"].items():
                    if "drift_detected" in info:
                        DATA_DRIFT_DETECTED.labels(feature=feat).set(
                            1 if info["drift_detected"] else 0
                        )
```

- [ ] **Step 7: Add the GET /drift endpoint after the /predict route**

Add after the closing `except` block of `/predict` and before the `if __name__ == "__main__":` block:

```python
@app.get("/drift")
async def get_drift_status():
    """Return the latest drift detection results.

    Returns:
        dict: Latest DRIFT_STATUS, or {"status": "not_checked_yet"} if
        fewer than 150 predictions have been processed since startup.
    """
    if not _drift_module.DRIFT_STATUS:
        return {"status": "not_checked_yet"}
    return _drift_module.DRIFT_STATUS
```

- [ ] **Step 8: Run the 2 endpoint tests — they should pass**

```bash
PYTHONPATH=. pytest tests/test_drift.py::test_drift_endpoint_not_checked_yet tests/test_drift.py::test_drift_endpoint_returns_full_status -v
```

Expected: 2 `PASSED`

- [ ] **Step 9: Run the full drift test suite**

```bash
PYTHONPATH=. pytest tests/test_drift.py -v
```

Expected: 7 `PASSED`

- [ ] **Step 10: Commit**

```bash
git add app/main.py tests/test_drift.py
git commit -m "feat: integrate drift detection into FastAPI — Gauges, counter, /drift endpoint"
```

---

### Task 4: Generate baseline and smoke-test

**Files:**
- Run: `scripts/generate_baseline.py` (requires `data/creditcard.csv`)
- Verify: `app/drift_baseline.pkl` is created and gitignored

- [ ] **Step 1: Run the baseline generator**

```bash
PYTHONPATH=. python scripts/generate_baseline.py
```

Expected output (numbers will vary):
```
Baseline saved to .../app/drift_baseline.pkl
  Rows : 284,807
  V4      : mean=0.1669  std=1.1530
  V10     : mean=0.0005  std=1.1120
  V11     : mean=0.0010  std=1.0196
  V12     : mean=-0.0001  std=1.0001
  V14     : mean=0.0010  std=1.0002
  V16     : mean=-0.0005  std=1.0000
  Amount  : mean=88.3496  std=250.1201
```

- [ ] **Step 2: Confirm the file exists and is gitignored**

```bash
ls -lh app/drift_baseline.pkl && git status | grep drift_baseline
```

Expected: file listed by `ls`, NOT listed by `git status` (gitignored).

- [ ] **Step 3: Start the API and call /drift before 150 predictions**

```bash
docker-compose up -d && sleep 5 && curl http://127.0.0.1:8001/drift
```

Expected:
```json
{"status": "not_checked_yet"}
```

- [ ] **Step 4: Final commit**

```bash
git add scripts/__init__.py scripts/generate_baseline.py
git commit -m "chore: add scripts package to version control"
```
