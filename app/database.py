"""SQLAlchemy database configuration and ORM models for PostgreSQL.

This module sets up the database engine, session factory, declarative
base, and the ``TransactionRecord`` table used to persist every incoming
transaction along with its fraud prediction.

Pipeline step: Step 5 – PostgreSQL storage.
"""

import datetime
import os

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    JSON,
    String,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL environment variable is not set. "
        "Set it to a valid PostgreSQL connection string before starting the server."
    )

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# ---------------------------------------------------------------------------
# ORM model
# ---------------------------------------------------------------------------


class TransactionRecord(Base):
    """Persisted record of a single transaction and its prediction.

    Each row captures the raw input features (as JSON), the model's
    binary decision, its confidence score, and metadata for auditing and
    future model retraining.

    Attributes:
        id: Auto-incrementing primary key.
        time: The ``Time`` feature from the original dataset (seconds
            elapsed since the first transaction).
        amount: Transaction amount in the original currency.
        features: JSON blob storing PCA features V1–V28.
        is_fraud: Binary prediction (``True`` = fraud).
        fraud_probability: Model confidence score in [0, 1].
        model_version: Semantic version tag of the model that produced
            the prediction.
        created_at: UTC timestamp of when the record was inserted.
    """

    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    time = Column(Float, index=True)
    amount = Column(Float)
    features = Column(JSON)
    is_fraud = Column(Boolean)
    fraud_probability = Column(Float)
    model_version = Column(String, default="1.0.0")
    created_at = Column(
        DateTime,
        default=lambda: datetime.datetime.now(datetime.timezone.utc),
    )


# ---------------------------------------------------------------------------
# Dependency injection helper
# ---------------------------------------------------------------------------


def get_db():
    """Yield a SQLAlchemy session and guarantee cleanup after use.

    Intended to be used as a FastAPI dependency via ``Depends(get_db)``.
    The session is always closed in the ``finally`` block, even if the
    request handler raises an exception, preventing connection leaks.

    Yields:
        Session: An active SQLAlchemy database session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
