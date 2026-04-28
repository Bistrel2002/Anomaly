# ÉTAPE 9 — Containerisation de l'API FastAPI
# Image Docker pour le service d'inférence ML

FROM python:3.11-slim

# Définir le répertoire de travail dans le container
WORKDIR /code

# Copier et installer les dépendances en premier (cache Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'application
COPY app/ ./app/
COPY scripts/ ./scripts/

# Exposer le port de l'API
EXPOSE 8000

# Lancer le serveur uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
