from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import logging
from app.schemas import TransactionInput, PredictionOutput
from app.model import FraudDetector

# ÉTAPE 4 — Serveur Backend / Inférence du modèle
# Point d'entrée principal (Serveur Web)

# Configuration du logging pour voir ce qui se passe dans la console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager
from app.database import Base, engine, get_db, TransactionRecord
from sqlalchemy.orm import Session
from fastapi import Depends

# Variable globale pour stocker le modèle (chargé au démarrage)
fraud_detector = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    S'exécute au démarrage et à l'arrêt de l'application.
    C'est ici qu'on charge les gros fichiers (.joblib) en mémoire.
    """
    global fraud_detector
    logger.info("Démarrage du serveur et chargement de l'intelligence artificielle...")
    try:
        fraud_detector = FraudDetector()
    except Exception as e:
        logger.error(f"Échec critique du chargement du modèle: {e}")
        
    logger.info("Initialisation de la base de données PostgreSQL...")
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Tables créées avec succès.")
    except Exception as e:
        logger.error(f"Erreur de connexion à la base de données: {e}")
        
    yield
    # Code exécuté à l'arrêt de l'API (ex: nettoyer la mémoire si nécessaire)
    fraud_detector = None

# Instanciation de l'application FastAPI
app = FastAPI(
    title="Fraud Detection ML API",
    description="API MLOps recevant des transactions en temps réel pour détecter la fraude à la carte bancaire.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    """Route de santé (Health Check)"""
    return {"message": "✅ Fraud Detection API is UP and Running!"}

@app.post("/predict", response_model=PredictionOutput)
async def predict_fraud(transaction: TransactionInput, db: Session = Depends(get_db)):
    """
    Endpoint principal qui reçoit une transaction et retourne la prédiction.
    Maintenant, il sauvegarde aussi le résultat dans PostgreSQL.
    """
    global fraud_detector
    
    if fraud_detector is None or fraud_detector.model is None:
        raise HTTPException(
            status_code=503, 
            detail="Le service ML est indisponible ou le modèle n'est pas chargé correctement."
        )

    try:
        # On passe la transaction validée par Pydantic à notre classe métier FraudDetector
        prediction_result = fraud_detector.predict(transaction)
        
        # On formate la réponse pour qu'elle corresponde au schéma PredictionOutput
        is_fraud = prediction_result["is_fraud"]
        prob = prediction_result["fraud_probability"]
        
        message = "ALERTE: Transaction Suspecte (Fraude probable)!" if is_fraud else "Transaction Normale."
        
        # --- NOUVEAU: Sauvegarde en base de données ---
        # On extrait les features pertinentes (V1 à V28)
        features_dict = {f"V{i}": getattr(transaction, f"V{i}") for i in range(1, 29)}
        
        db_record = TransactionRecord(
            time=transaction.Time,
            amount=transaction.Amount,
            features=features_dict,
            is_fraud=is_fraud,
            fraud_probability=prob,
            model_version="1.0.0"
        )
        
        db.add(db_record)
        db.commit()
        db.refresh(db_record)
        # ---------------------------------------------
        
        return PredictionOutput(
            transaction_time=transaction.Time,
            transaction_amount=transaction.Amount,
            is_fraud=is_fraud,
            fraud_probability=prob,
            message=message
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne lors de l'inférence: {str(e)}")
