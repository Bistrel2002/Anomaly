import os
import time
import numpy as np
import pandas as pd
import joblib  # noqa: F401
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


def build_feature_matrix(df: pd.DataFrame, scaler) -> tuple[np.ndarray, np.ndarray]:
    """Replicates app/model.py preprocessing exactly: scale Amount and Time, append V1-V28."""
    scaled_amount = scaler.transform(df[["Amount"]].values)[:, 0]
    scaled_time   = scaler.transform(df[["Time"]].values)[:, 0]
    v_cols        = np.column_stack([df[f"V{i}"].values for i in range(1, 29)])
    X = np.column_stack([scaled_amount, scaled_time, v_cols])
    y = df["Class"].values
    return X, y


def _save_outputs(
    df: pd.DataFrame,
    y: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    cm: np.ndarray,
    auc: float,
) -> None:
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
                xticklabels=["Normal", "Fraud"],
                yticklabels=["Normal", "Fraud"])
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
    legits = df[df["Class"] == 0]
    n_sample = min(500, len(legits))
    legits = legits.sample(n=n_sample, random_state=42)
    sample = pd.concat([frauds, legits]).sample(frac=1, random_state=42).reset_index(drop=True)

    n_fraud = len(frauds)
    n_legit = len(legits)
    total   = len(sample)
    print(f"Sending {total} transactions ({n_fraud} frauds + {n_legit} legit)...")

    v_cols          = [f"V{i}" for i in range(1, 29)]
    correct         = 0
    fraud_correct   = 0
    legit_correct   = 0
    latencies: list[float] = []

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

    fraud_pct = f"{fraud_correct/n_fraud*100:.1f}%" if n_fraud > 0 else "N/A"
    legit_pct = f"{legit_correct/n_legit*100:.1f}%" if n_legit > 0 else "N/A"
    total_pct = f"{correct/total*100:.1f}%" if total > 0 else "N/A"

    print(f"\n=== API SUMMARY ({total} transactions) ===")
    print(f"Frauds correctly detected  : {fraud_correct}/{n_fraud} ({fraud_pct})")
    print(f"Legit correctly classified : {legit_correct}/{n_legit} ({legit_pct})")
    print(f"Overall accuracy           : {correct}/{total} ({total_pct})")
    print(f"Average latency            : {avg_lat_ms:.1f}ms")
