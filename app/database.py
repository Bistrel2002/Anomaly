"""
# ÉTAPE 5 — Stockage PostgreSQL
# Fichier pour la connexion et les requêtes à la base de données
"""
import os
import datetime
from sqlalchemy import create_engine, Column, Integer, Float, Boolean, String, DateTime, JSON
from sqlalchemy.orm import sessionmaker, declarative_base

# L'URL de connexion à la base de données (PostgreSQL)
# Par défaut, on pointe vers notre base locale via Docker Compose
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://anomaly_user:anomaly_password@127.0.0.1:5433/anomaly_db"
)

# Création de l'engine SQLAlchemy
engine = create_engine(DATABASE_URL)

# Création d'une session (qui sera utilisée à chaque requête)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base de déclaration pour nos modèles
Base = declarative_base()

# Modèle de la table des transactions
class TransactionRecord(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    time = Column(Float, index=True)
    amount = Column(Float)
    
    # On stocke les features (V1 à V28) sous forme de JSON pour plus de flexibilité
    features = Column(JSON)
    
    # Résultats de la prédiction
    is_fraud = Column(Boolean)
    fraud_probability = Column(Float)
    
    # Métadonnées
    model_version = Column(String, default="1.0.0")
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))

# Dépendance pour FastAPI afin d'obtenir une session de Base de Données
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
