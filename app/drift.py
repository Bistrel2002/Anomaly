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
                drift = bool(p_value < P_VALUE_THRESHOLD)
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
