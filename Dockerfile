# ÉTAPE 9 — Containerisation de l'API FastAPI
# Image Docker pour le service d'inférence ML

FROM python:3.11-slim

# Définir le répertoire de travail dans le container
WORKDIR /code

# Install postgresql-client for pg_dump / psql used by backup scripts
RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client \
    cron \
    && rm -rf /var/lib/apt/lists/*

COPY cron/crontab /etc/cron.d/backup_secure
RUN chmod 0644 /etc/cron.d/backup_secure && crontab /etc/cron.d/backup_secure

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
