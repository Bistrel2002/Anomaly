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
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required feature columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    baseline = {f: df[f].values for f in FEATURES}
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    joblib.dump(baseline, output_path)
    print(f"Baseline saved to {output_path}")
    print(f"  Rows : {len(df):,}")
    for f, arr in baseline.items():
        print(f"  {f:8s}: mean={arr.mean():.4f}  std={arr.std():.4f}")
    return baseline


if __name__ == "__main__":
    generate_baseline(_DATA_PATH, _OUTPUT_PATH)
