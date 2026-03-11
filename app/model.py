import os
import joblib
import pandas as pd
import numpy as np
from app.schemas import TransactionInput

# ÉTAPE 4 — Serveur Backend / Inférence du modèle
# Fichier qui gère l'intelligence artificielle proprement dite (machine learning logic)

class FraudDetector:
    def __init__(self):
        # Trouve le chemin vers le dossier actuel 'app/' pour éviter les bugs de chemin relatif
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Construit les chemins complets vers les fichiers d'intelligence artificielle
        model_path = os.path.join(current_dir, 'random_forest_fraud_model.joblib')
        scaler_path = os.path.join(current_dir, 'robust_scaler.joblib')
        
        try:
            # Réveille le modèle Random Forest et le RobustScaler
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
        except FileNotFoundError as e:
            print(f"❌ ERREUR: Fichier .joblib introuvable. Détails: {e}")
            self.model = None
            self.scaler = None

    def predict(self, transaction: TransactionInput) -> dict:
        # Sécurité: empêche de prédire si l'initialisation a échoué
        if self.model is None or self.scaler is None:
            raise RuntimeError("Le modèle ML ou le Scaler n'est pas chargé.")

        # Transforme l'objet Pydantic sécurisé en dictionnaire Python classique
        transaction_dict = transaction.model_dump()
        
        # Extrait les valeurs de Montant et de Temps
        amount = transaction_dict['Amount']
        time = transaction_dict['Time']
        
        # Applique l'échelle mathématique (Scaler) sur le Montant et le Temps, 
        # comme on l'a fait pendant l'entraînement, pour que le modèle puisse les comprendre
        scaled_amount = self.scaler.transform(np.array([[amount]]))[0][0]
        scaled_time = self.scaler.transform(np.array([[time]]))[0][0]
        
        # Construit le vecteur de features dans l'ordre exact attendu par le modèle
        # (D'abord Amount et Time scalés)
        ordered_features = [scaled_amount, scaled_time]
        
        # (Ensuite les 28 variables V1 à V28)
        for v_num in range(1, 29):
            ordered_features.append(transaction_dict[f'V{v_num}'])
            
        # Convertit la liste en un vecteur NumPy ultra-rapide
        final_array = np.array([ordered_features])
        
        # Le modèle prend sa décision (0 = Normal, 1 = Fraude)
        is_fraud_pred = self.model.predict(final_array)[0]
        
        # Le modèle calcule son propre pourcentage de certitude (ex: 0.9812 pour Fraude)
        fraud_probability = self.model.predict_proba(final_array)[0][1]
        
        # Formate la réponse propre (Vrai/Faux) et probabilité arrondie à renvoyer à l'API
        return {
            "is_fraud": bool(is_fraud_pred),
            "fraud_probability": round(float(fraud_probability), 4)
        }
