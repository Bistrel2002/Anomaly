import os
import time  # noqa: F401
import numpy as np
import pandas as pd
import joblib  # noqa: F401
import matplotlib.pyplot as plt  # noqa: F401
import seaborn as sns  # noqa: F401
import requests  # noqa: F401
from sklearn.metrics import (  # noqa: F401
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


def build_feature_matrix(df: pd.DataFrame, scaler) -> tuple[np.ndarray, np.ndarray]:
    """Replicates app/model.py preprocessing exactly: scale Amount and Time, append V1-V28."""
    scaled_amount = scaler.transform(df[["Amount"]].values)[:, 0]
    scaled_time   = scaler.transform(df[["Time"]].values)[:, 0]
    v_cols        = np.column_stack([df[f"V{i}"].values for i in range(1, 29)])
    X = np.column_stack([scaled_amount, scaled_time, v_cols])
    y = df["Class"].values
    return X, y
