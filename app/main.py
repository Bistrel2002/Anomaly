"""FastAPI application entry point for the Fraud Detection ML pipeline.

This module bootstraps the web server, loads the trained Random Forest model
into memory at startup, and exposes the ``/predict`` endpoint that receives
credit-card transaction data, runs real-time inference, persists results to
PostgreSQL, and feeds business metrics to Prometheus.

Pipeline steps covered:
    - Step 4: Model serving / inference API
    - Step 5: PostgreSQL persistence
    - Step 6: Prometheus + Grafana monitoring
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
import threading

from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from sqlalchemy.orm import Session

from app.database import Base, TransactionRecord, engine, get_db
from app.model import FraudDetector
from app.schemas import PredictionOutput, TransactionInput

import app.drift as _drift_module
from app.drift import DriftDetector, DRIFT_WINDOW

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus – custom business metrics
# ---------------------------------------------------------------------------

FRAUD_PREDICTIONS = Counter(
    "fraud_predictions_total",
    "Total number of predictions broken down by outcome.",
    ["result"],  # label values: "fraud" | "normal"
)

FRAUD_PROBABILITY = Histogram(
    "fraud_probability_score",
    "Distribution of fraud probability scores returned by the model.",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Time spent inside the ML model for a single inference (seconds).",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0],
)

DATA_DRIFT_DETECTED = Gauge(
    "data_drift_detected",
    "1 if KS drift detected for a feature in the last window, 0 otherwise.",
    ["feature"],
)

_prediction_count: int = 0
_count_lock: threading.Lock = threading.Lock()

# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------

# Holds the ML model and drift detector in memory for the lifetime of the process.
fraud_detector: FraudDetector | None = None
drift_detector: DriftDetector | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown resources for the FastAPI application.

    On startup:
        1. Load the serialised Random Forest model into memory.
        2. Ensure all SQLAlchemy ORM tables exist in PostgreSQL.

    On shutdown:
        Release the model reference so the garbage collector can reclaim
        memory.

    Yields:
        Control to the running application between startup and shutdown.
    """
    global fraud_detector  # noqa: PLW0603
    global drift_detector  # noqa: PLW0603
    global _prediction_count  # noqa: PLW0603

    logger.info("Starting server – loading ML model…")
    try:
        fraud_detector = FraudDetector()
    except Exception as exc:
        logger.error("Critical failure while loading the model: %s", exc)

    logger.info("Initialising PostgreSQL tables…")
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully.")
    except Exception as exc:
        logger.error("Database connection error: %s", exc)

    logger.info("Initialising drift detector…")
    try:
        drift_detector = DriftDetector()
        if drift_detector.is_ready:
            logger.info("Drift detector ready.")
        else:
            logger.warning("Drift baseline not found — drift detection disabled.")
    except Exception as exc:
        logger.error("Failed to initialise drift detector: %s", exc)

    yield

    # Shutdown: release resources.
    fraud_detector = None
    drift_detector = None
    _prediction_count = 0


# ---------------------------------------------------------------------------
# FastAPI application instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Fraud Detection ML API",
    description=(
        "MLOps API that receives credit-card transactions in real time, "
        "predicts fraud probability, persists results, and exposes "
        "Prometheus metrics."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Auto-instrument HTTP metrics (requests/sec, latency, status codes).
Instrumentator().instrument(app).expose(app)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
async def root():
    """Health-check endpoint.

    Returns:
        dict: Simple status message confirming the API is operational.
    """
    return {"message": "✅ Fraud Detection API is UP and Running!"}


@app.post("/predict", response_model=PredictionOutput)
async def predict_fraud(
    transaction: TransactionInput,
    db: Session = Depends(get_db),
):
    """Run fraud inference on a single transaction.

    Workflow:
        1. Validate that the ML model is loaded and available.
        2. Run inference and measure latency.
        3. Record Prometheus business metrics (prediction count, score
           distribution, inference latency).
        4. Persist the transaction and its prediction to PostgreSQL.
        5. Return the structured prediction response.

    Args:
        transaction: Validated input containing Time, V1–V28, and Amount.
        db: SQLAlchemy session injected by FastAPI's dependency system.

    Returns:
        PredictionOutput: Prediction result with fraud flag, probability,
        and a human-readable message.

    Raises:
        HTTPException 503: If the ML model is not loaded.
        HTTPException 500: If an unexpected error occurs during inference.
    """
    global fraud_detector  # noqa: PLW0603
    global _prediction_count  # noqa: PLW0603

    if fraud_detector is None or fraud_detector.model is None:
        raise HTTPException(
            status_code=503,
            detail="ML service unavailable – model not loaded.",
        )

    try:
        # --- Inference & latency tracking ---
        start_time = time.time()
        prediction_result = fraud_detector.predict(transaction)
        inference_duration = time.time() - start_time
        PREDICTION_LATENCY.observe(inference_duration)

        is_fraud = prediction_result["is_fraud"]
        prob = prediction_result["fraud_probability"]

        # --- Business metrics ---
        FRAUD_PREDICTIONS.labels(
            result="fraud" if is_fraud else "normal"
        ).inc()
        FRAUD_PROBABILITY.observe(prob)

        message = (
            "ALERT: Suspicious Transaction (Fraud likely)!"
            if is_fraud
            else "Transaction Normal."
        )

        # --- Persist to PostgreSQL ---
        features_dict = {
            f"V{i}": getattr(transaction, f"V{i}") for i in range(1, 29)
        }
        features_dict["Amount"] = transaction.Amount

        db_record = TransactionRecord(
            time=transaction.Time,
            amount=transaction.Amount,
            features=features_dict,
            is_fraud=is_fraud,
            fraud_probability=prob,
            model_version="1.0.0",
        )
        db.add(db_record)
        db.commit()
        db.refresh(db_record)

        # Drift detection trigger: every DRIFT_WINDOW predictions
        should_check = False
        with _count_lock:
            _prediction_count += 1
            if _prediction_count >= DRIFT_WINDOW:
                should_check = True
                _prediction_count = 0

        try:
            if should_check and drift_detector is not None and drift_detector.is_ready:
                recent = (
                    db.query(TransactionRecord)
                    .order_by(TransactionRecord.id.desc())
                    .limit(DRIFT_WINDOW)
                    .all()
                )
                drift_results = drift_detector.compute_drift(
                    [r.features for r in recent]
                )
                if "results" in drift_results:
                    for feat, info in drift_results["results"].items():
                        if "drift_detected" in info:
                            DATA_DRIFT_DETECTED.labels(feature=feat).set(
                                1 if info["drift_detected"] else 0
                            )
        except Exception as drift_exc:
            logger.error("Drift computation error (non-fatal): %s", drift_exc)

        return PredictionOutput(
            transaction_time=transaction.Time,
            transaction_amount=transaction.Amount,
            is_fraud=is_fraud,
            fraud_probability=prob,
            message=message,
        )

    except Exception as exc:
        logger.error("Prediction error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Internal inference error: {exc}",
        ) from exc


@app.get("/drift")
async def get_drift_status():
    """Return the latest drift detection results.

    Returns:
        dict: Latest DRIFT_STATUS, or {"status": "not_checked_yet"} if
        fewer than 150 predictions have been processed since startup.
    """
    if not _drift_module.DRIFT_STATUS:
        return {"status": "not_checked_yet"}
    return _drift_module.DRIFT_STATUS


# ---------------------------------------------------------------------------
# Standalone execution (development only)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
