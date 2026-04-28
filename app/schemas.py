"""Pydantic schemas for API request and response validation.

These models enforce strict type-checking on every incoming transaction
payload and guarantee a consistent JSON structure in every API response.

Pipeline step: Step 4 – Model serving / data validation.
"""

from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    """Schema for an incoming credit-card transaction.

    Each field maps directly to a column in the original Kaggle
    ``creditcard.csv`` dataset.  The 28 ``V`` features are
    dimensionality-reduced via PCA; their real-world meaning is
    anonymised for confidentiality.

    Attributes:
        Time: Seconds elapsed since the first transaction in the
            dataset.
        V1–V28: PCA-transformed anonymised features.
        Amount: Monetary value of the transaction.
    """

    Time: float = Field(..., description="Seconds elapsed since the first transaction")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., description="Transaction amount in original currency")


class PredictionOutput(BaseModel):
    """Schema for the prediction response returned by ``/predict``.

    Attributes:
        transaction_time: Echo of the input ``Time`` value.
        transaction_amount: Echo of the input ``Amount`` value.
        is_fraud: Binary classification result (``True`` = fraud).
        fraud_probability: Model confidence score in [0, 1].
        message: Human-readable summary of the prediction.
    """

    transaction_time: float
    transaction_amount: float
    is_fraud: bool
    fraud_probability: float
    message: str
