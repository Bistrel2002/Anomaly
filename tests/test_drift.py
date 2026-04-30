"""Unit tests for data drift detection — drift.py and generate_baseline.py."""

import os

import joblib
import numpy as np
import pandas as pd
import pytest

from app.drift import DriftDetector, FEATURES
from scripts.generate_baseline import generate_baseline
from fastapi.testclient import TestClient
from app.main import app as fastapi_app

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
    rng = np.random.default_rng(42)
    # Use identical arrays for baseline and live data — guarantees p_value = 1.0
    exact = {f: rng.normal(0, 1, 150) for f in FEATURES}
    pkl = str(tmp_path / "baseline.pkl")
    joblib.dump(exact, pkl)
    detector = DriftDetector(baseline_path=pkl)

    records = [{f: float(v) for f, v in zip(FEATURES, [exact[f][i] for f in FEATURES])} for i in range(150)]
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


def test_drift_endpoint_not_checked_yet(monkeypatch):
    import app.drift as drift_module
    monkeypatch.setattr(drift_module, "DRIFT_STATUS", {})

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

    client = TestClient(fastapi_app)
    response = client.get("/drift")
    assert response.status_code == 200
    data = response.json()
    assert data["drift_detected"] is False
    assert "results" in data
    assert "last_checked_at" in data
    for f in FEATURES:
        assert f in data["results"]

