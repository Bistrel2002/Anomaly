"""Fraud detection model wrapper for real-time inference.

This module encapsulates the trained Random Forest classifier and its
associated RobustScaler.  It loads the serialised ``.joblib`` artefacts
from disk and provides a single ``predict()`` method that transforms a
validated Pydantic input into the feature vector expected by the model.

Pipeline step: Step 4 – Model serving / inference.
"""

import os

import joblib
import numpy as np

from app.schemas import TransactionInput


class FraudDetector:
    """Wrapper around the serialised Random Forest fraud-detection model.

    On instantiation the class loads two artefacts from the ``app/``
    directory:
        * ``random_forest_fraud_model.joblib`` – trained classifier.
        * ``robust_scaler.joblib`` – RobustScaler fitted during training.

    Attributes:
        model: The scikit-learn ``RandomForestClassifier`` instance, or
            ``None`` if loading failed.
        scaler: The scikit-learn ``RobustScaler`` instance, or ``None``
            if loading failed.
    """

    def __init__(self) -> None:
        """Load the model and scaler from the ``app/`` directory."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "random_forest_fraud_model.joblib")
        scaler_path = os.path.join(current_dir, "robust_scaler.joblib")

        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
        except FileNotFoundError as exc:
            print(f"ERROR: .joblib artefact not found – {exc}")
            self.model = None
            self.scaler = None

    def predict(self, transaction: TransactionInput) -> dict:
        """Run fraud inference on a single transaction.

        The method replicates the exact preprocessing applied during
        training:
            1. Scale ``Amount`` and ``Time`` with the RobustScaler.
            2. Concatenate the scaled values with the 28 PCA features
               (V1–V28) in the correct column order.
            3. Feed the resulting 30-dimensional vector to the Random
               Forest for classification.

        Args:
            transaction: A validated Pydantic ``TransactionInput``
                object containing Time, V1–V28, and Amount.

        Returns:
            dict: A dictionary with two keys:
                - ``is_fraud`` (bool): ``True`` if the model predicts
                  fraud (class 1).
                - ``fraud_probability`` (float): The model's estimated
                  probability of fraud, rounded to four decimal places.

        Raises:
            RuntimeError: If the model or scaler failed to load during
                initialisation.
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("ML model or scaler is not loaded.")

        transaction_dict = transaction.model_dump()

        amount = transaction_dict["Amount"]
        time_val = transaction_dict["Time"]

        # Apply the same scaling used during training.
        scaled_amount = self.scaler.transform(np.array([[amount]]))[0][0]
        scaled_time = self.scaler.transform(np.array([[time_val]]))[0][0]

        # Build the feature vector: [scaled_amount, scaled_time, V1…V28].
        ordered_features = [scaled_amount, scaled_time]
        for v_num in range(1, 29):
            ordered_features.append(transaction_dict[f"V{v_num}"])

        final_array = np.array([ordered_features])

        is_fraud_pred = self.model.predict(final_array)[0]
        fraud_probability = self.model.predict_proba(final_array)[0][1]

        return {
            "is_fraud": bool(is_fraud_pred),
            "fraud_probability": round(float(fraud_probability), 4),
        }
