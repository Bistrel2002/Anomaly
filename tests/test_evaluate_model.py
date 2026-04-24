import os
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from tests.evaluate_model import build_feature_matrix


class DummyScaler:
    def transform(self, X):
        return X * 2.0


def make_dummy_df(n=5):
    labels = ([0, 1, 0, 1, 0] * ((n // 5) + 1))[:n]
    data = {"Time": np.ones(n), "Amount": np.ones(n) * 3.0, "Class": labels}
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
        assert {"Amount", "true_label", "predicted_label", "fraud_probability"}.issubset(result.columns)


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
