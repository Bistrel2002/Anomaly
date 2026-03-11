from pydantic import BaseModel, Field

# ÉTAPE 4 — Mise en Production (Serving) / Validation de données
# Fichier pour les schémas de validation (ex: Pydantic)

class TransactionInput(BaseModel):
    """
    Spécifie la forme exacte de la donnée que l'API s'attend à recevoir 
    lorsqu'un client demande une prédiction de fraude.
    """
    Time: float
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
    Amount: float = Field(..., description="Le montant de la transaction")

class PredictionOutput(BaseModel):
    """
    Spécifie la forme exacte de la réponse renvoyée par l'API.
    """
    transaction_time: float
    transaction_amount: float
    is_fraud: bool
    fraud_probability: float
    message: str
