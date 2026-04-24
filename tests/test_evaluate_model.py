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
