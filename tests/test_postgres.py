import requests
import time
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.database import TransactionRecord

# Définition des données de test
test_data = {
    "Time": 0.0,
    "V1": -1.3598071336738,
    "V2": -0.0727811733098497,
    "V3": 2.53634673796914,
    "V4": 1.37815522427443,
    "V5": -0.338320769942518,
    "V6": 0.462387777762292,
    "V7": 0.239598554061257,
    "V8": 0.0986979012610507,
    "V9": 0.363786969611213,
    "V10": 0.0907941719789316,
    "V11": -0.551599533260813,
    "V12": -0.617800855762348,
    "V13": -0.991389847235408,
    "V14": -0.311169353699879,
    "V15": 1.46817697209427,
    "V16": -0.470400525259478,
    "V17": 0.207971241929242,
    "V18": 0.0257905801985591,
    "V19": 0.403992960255733,
    "V20": 0.251412098239705,
    "V21": -0.018306777944153,
    "V22": 0.277837575558899,
    "V23": -0.110473910188767,
    "V24": 0.0669280749146731,
    "V25": 0.128539358273528,
    "V26": -0.189114843888824,
    "V27": 0.133558376740387,
    "V28": -0.0210530534538215,
    "Amount": 149.62
}

def run_test():
    print("Envoi d'une requête POST à /predict...")
    response = requests.post("http://127.0.0.1:8000/predict", json=test_data)
    
    if response.status_code == 200:
        print("✅ API a répondu avec succès :")
        print(response.json())
    else:
        print(f"❌ Erreur de l'API ({response.status_code}): {response.text}")
        return

    print("\nVérification dans la base de données...")
    time.sleep(1) # Attendre un peu que la base soit bien mise à jour

    DATABASE_URL = "postgresql://anomaly_user:anomaly_password@localhost:5432/anomaly_db"
    try:
        engine = create_engine(DATABASE_URL)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        session = SessionLocal()
        
        # Récupérer la dernière transaction
        latest_record = session.query(TransactionRecord).order_by(TransactionRecord.id.desc()).first()
        
        if latest_record:
            print("✅ Enregistrement trouvé dans PostgreSQL :")
            print(f"- ID: {latest_record.id}")
            print(f"- Montant: {latest_record.amount}")
            print(f"- Est Fraude: {latest_record.is_fraud}")
            print(f"- Probabilité: {latest_record.fraud_probability}")
            print(f"- Version Modèle: {latest_record.model_version}")
        else:
            print("❌ Aucun enregistrement trouvé dans la base de données.")
            
    except Exception as e:
        print(f"❌ Erreur lors de la connexion à la base de données : {e}")

if __name__ == "__main__":
    run_test()
