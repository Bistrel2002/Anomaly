"""Unit tests for ``tests/evaluate_model.py`` helper functions.

Validates:
    - Feature matrix shape and column ordering.
    - Label extraction.
    - Output file creation (CSV, confusion matrix, ROC curve).
    - Graceful Phase 2 skip when the API is unreachable.

Usage:
    pytest tests/test_evaluate_model.py -v
"""

import io
import os
import tempfile
import unittest.mock as mock
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from tests.evaluate_model import build_feature_matrix


class DummyScaler:
    """Minimal scaler stub that doubles every value (for deterministic assertions)."""

    def transform(self, X):
        """Return ``X * 2.0``."""
        return X * 2.0


def _make_dummy_df(n: int = 5) -> pd.DataFrame:
    """Build a small synthetic DataFrame matching the expected schema.

    Args:
        n: Number of rows.

    Returns:
        DataFrame with Time, Amount, V1–V28, and Class columns.
    """
    labels = ([0, 1, 0, 1, 0] * ((n // 5) + 1))[:n]
    data = {"Time": np.ones(n), "Amount": np.ones(n) * 3.0, "Class": labels}
    for i in range(1, 29):
        data[f"V{i}"] = np.ones(n) * float(i)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Feature matrix tests
# ---------------------------------------------------------------------------


def test_build_feature_matrix_shape():
    """Output matrix should have shape (n_rows, 30)."""
    df = _make_dummy_df()
    X, y = build_feature_matrix(df, DummyScaler())
    assert X.shape == (5, 30), f"Expected (5, 30), got {X.shape}"
    assert y.shape == (5,)


def test_build_feature_matrix_column_order():
    """Columns must be [scaled_amount, scaled_time, V1, …, V28]."""
    df = _make_dummy_df(n=1)
    X, _ = build_feature_matrix(df, DummyScaler())
    # scaled_amount = 3.0 * 2 = 6.0, scaled_time = 1.0 * 2 = 2.0
    assert X[0, 0] == pytest.approx(6.0), "Column 0 must be scaled_amount"
    assert X[0, 1] == pytest.approx(2.0), "Column 1 must be scaled_time"
    assert X[0, 2] == pytest.approx(1.0), "Column 2 must be V1 (unscaled)"
    assert X[0, 29] == pytest.approx(28.0), "Column 29 must be V28"


def test_build_feature_matrix_labels():
    """Labels should be extracted verbatim from the Class column."""
    df = _make_dummy_df()
    _, y = build_feature_matrix(df, DummyScaler())
    np.testing.assert_array_equal(y, [0, 1, 0, 1, 0])


# ---------------------------------------------------------------------------
# Output file tests
# ---------------------------------------------------------------------------


def test_save_outputs_creates_files():
    """All three artefact files must be created in the output directory."""
    from tests.evaluate_model import _save_outputs

    y_true = np.array([0, 0, 1, 1, 0])
    y_pred = np.array([0, 0, 1, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.9, 0.4, 0.1])
    cm = np.array([[3, 0], [1, 1]])
    auc = 0.875
    df = _make_dummy_df()

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("tests.evaluate_model.OUTPUT_DIR", tmpdir):
            _save_outputs(df, y_true, y_pred, y_prob, cm, auc)
            files = os.listdir(tmpdir)
            assert "report.csv" in files
            assert "confusion_matrix.png" in files
            assert "roc_curve.png" in files


def test_save_outputs_csv_columns():
    """The CSV report must contain the expected columns."""
    from tests.evaluate_model import _save_outputs

    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    y_prob = np.array([0.05, 0.95])
    cm = np.array([[1, 0], [0, 1]])
    auc = 1.0
    df = _make_dummy_df(n=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("tests.evaluate_model.OUTPUT_DIR", tmpdir):
            _save_outputs(df, y_true, y_pred, y_prob, cm, auc)
        result = pd.read_csv(os.path.join(tmpdir, "report.csv"))
        expected = {"Amount", "true_label", "predicted_label", "fraud_probability"}
        assert expected.issubset(result.columns)


# ---------------------------------------------------------------------------
# Phase 2 graceful degradation
# ---------------------------------------------------------------------------


def test_phase2_skips_gracefully_when_api_down():
    """Phase 2 should print a skip message instead of crashing."""
    from tests.evaluate_model import run_phase2

    df = pd.concat([_make_dummy_df()] * 100).reset_index(drop=True)
    df.loc[0, "Class"] = 1  # ensure at least one fraud row

    with mock.patch(
        "tests.evaluate_model.requests.get",
        side_effect=ConnectionError("refused"),
    ):
        buf = io.StringIO()
        with redirect_stdout(buf):
            run_phase2(df)
        output = buf.getvalue()

    assert "ignored" in output.lower() or "skipped" in output.lower()
