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
