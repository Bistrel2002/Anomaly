# Data Drift Detection — Design Spec

**Date:** 2026-04-28
**Pipeline step:** ÉTAPE 7 — Détection de Data Drift

---

## Goal

Detect when the distribution of live transaction features deviates significantly from the training distribution, and expose that signal via Prometheus and a REST endpoint.

## Architecture

Four files are involved:

| File | Role |
|---|---|
| `scripts/generate_baseline.py` | One-shot script: reads `data/creditcard.csv`, extracts the 7 monitored features, saves `app/drift_baseline.pkl` |
| `app/drift_baseline.pkl` | Pre-computed baseline — numpy arrays of training distributions per feature |
| `app/drift.py` | Drift detection logic: loads baseline, runs KS test, maintains in-memory state |
| `app/main.py` | Integrates drift trigger (every 150 predictions), Prometheus Gauges, `/drift` endpoint |

## Features Monitored

V4, V10, V11, V12, V14, V16, Amount — 7 features chosen for their high fraud correlation.

## Statistical Method

**Kolmogorov-Smirnov two-sample test** (`scipy.stats.ks_2samp`).

- Compares the CDF of the live window against the training baseline.
- Threshold: `p_value < 0.05` → drift detected for that feature.
- Returns `statistic` (KS distance) and `p_value` per feature.

## Trigger

Every 150 predictions processed by `/predict`, a drift check is triggered:
1. Query the last 150 `TransactionRecord` rows from PostgreSQL.
2. Extract the 7 feature values from the `features` JSON column.
3. Run KS test for each feature against `drift_baseline.pkl`.
4. Update Prometheus Gauges and in-memory `DRIFT_STATUS`.
5. Reset the prediction counter to 0.

A `threading.Lock` protects the counter increment to avoid race conditions under concurrent requests.

## Components

### `scripts/generate_baseline.py`

- Reads `data/creditcard.csv`.
- Extracts raw arrays for V4, V10, V11, V12, V14, V16, Amount.
- Saves a dict `{feature_name: np.ndarray}` to `app/drift_baseline.pkl` using `joblib.dump`.
- Must be run once before starting the API.

### `app/drift.py`

```
DriftDetector
  baseline: dict[str, np.ndarray]   # loaded from drift_baseline.pkl at init
  DRIFT_STATUS: dict                 # in-memory, updated each check

  load_baseline(path) -> dict
  compute_drift(records: list[dict]) -> dict
    # returns per-feature: {statistic, p_value, drift_detected}
    # sets DRIFT_STATUS and last_checked_at timestamp
```

`compute_drift` is pure: takes a list of feature dicts, returns results. No side effects except updating `DRIFT_STATUS`.

### `app/main.py` additions

```python
# Prometheus Gauges — one per feature
DATA_DRIFT_DETECTED = Gauge(
    "data_drift_detected",
    "1 if KS drift detected for a feature, 0 otherwise",
    ["feature"],
)

# Internal counter
_prediction_count: int = 0
_count_lock: threading.Lock = threading.Lock()
```

In `/predict`, after persisting to PostgreSQL:
```python
with _count_lock:
    _prediction_count += 1
    should_check = (_prediction_count >= 150)
    if should_check:
        _prediction_count = 0

if should_check:
    records = db.query(TransactionRecord).order_by(TransactionRecord.id.desc()).limit(150).all()
    results = drift_detector.compute_drift([r.features for r in records])
    for feature, info in results.items():
        DATA_DRIFT_DETECTED.labels(feature=feature).set(1 if info["drift_detected"] else 0)
```

### `GET /drift` endpoint

Returns current drift status. If no check has run yet, returns `{"status": "not_checked_yet"}`.

Response schema:
```json
{
  "window": 150,
  "features_monitored": ["V4", "V10", "V11", "V12", "V14", "V16", "Amount"],
  "results": {
    "V4":     {"statistic": 0.12, "p_value": 0.03, "drift_detected": true},
    "V10":    {"statistic": 0.04, "p_value": 0.71, "drift_detected": false},
    "V11":    {"statistic": 0.07, "p_value": 0.18, "drift_detected": false},
    "V12":    {"statistic": 0.09, "p_value": 0.08, "drift_detected": false},
    "V14":    {"statistic": 0.15, "p_value": 0.01, "drift_detected": true},
    "V16":    {"statistic": 0.03, "p_value": 0.89, "drift_detected": false},
    "Amount": {"statistic": 0.06, "p_value": 0.42, "drift_detected": false}
  },
  "drift_detected": true,
  "last_checked_at": "2026-04-28T14:32:00Z"
}
```

`drift_detected` at the top level is `true` if **any** feature has drift.

## Error Handling

- If `drift_baseline.pkl` is missing at startup: log a warning, disable drift detection (skip the check silently, `/drift` returns `{"status": "baseline_not_loaded"}`).
- If fewer than 150 records exist in PostgreSQL: skip the check, log a debug message.
- KS test errors per feature: log and mark that feature as `{"drift_detected": false, "error": "..."}`.

## Testing

File: `tests/test_drift.py` — 4 unit tests, no PostgreSQL dependency.

1. **`test_compute_drift_no_drift`** — live data identical to baseline → all p-values > 0.05, all `drift_detected = False`.
2. **`test_compute_drift_drift_detected`** — live data from a completely different distribution (e.g., all zeros vs baseline) → p-value < 0.05 for at least one feature.
3. **`test_drift_endpoint_returns_structure`** — mock `DRIFT_STATUS` → `GET /drift` returns correct JSON fields.
4. **`test_generate_baseline_creates_file`** — call `generate_baseline.py` on a small dummy CSV → `drift_baseline.pkl` exists and contains the 7 features.

## Prometheus Metrics Added

| Metric | Type | Labels | Description |
|---|---|---|---|
| `data_drift_detected` | Gauge | `feature` | 1 if drift detected, 0 otherwise |

These metrics will be visible in Grafana via the existing Prometheus datasource.

## Files Created / Modified

- **Create:** `scripts/generate_baseline.py`
- **Create:** `tests/test_drift.py`
- **Modify:** `app/drift.py` (currently a placeholder)
- **Modify:** `app/main.py` (counter + Gauges + `/drift` endpoint)
- **Create (generated):** `app/drift_baseline.pkl` (not committed to git)
