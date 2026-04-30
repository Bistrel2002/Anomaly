# Fraud Detection — MLOps Pipeline

A production-grade MLOps pipeline that receives credit-card transactions in real time, predicts fraud probability using a trained Random Forest classifier, persists results to PostgreSQL, and exposes Prometheus metrics accompanied by a graphical dashboard via Grafana. The stack includes automated encrypted backups to S3 via HashiCorp Vault, data drift detection, and a full CI/CD pipeline.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT / TEST                            │
│              POST /predict  |  POST /predict/batch              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   FastAPI API   │  
                    │  (inference +   │
                    │   REST layer)   │
                    └──┬──────────┬───┘
                       │          │
              ┌────────▼──┐  ┌────▼────────┐
              │ PostgreSQL│  │  Prometheus │  
              │           │  │  (metrics)  │
              └───────────┘  └──────┬──────┘
                                    │
                             ┌──────▼──────┐
                             │   Grafana   │  
                             │ (dashboards)│
                             └─────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     SECURITY LAYER                              │
│                                                                 │
│   HashiCorp Vault      ←──     All secrets (DB, MinIO, keys)    │
│          │                                                      │
│   ┌──────▼──────┐    ┌─────────────────────────────────────┐    │
│   │  backup     │    │  MinIO (S3) :                       │    │
│   │  container  │───►│  Encrypted .enc + metadata .json    │    │
│   │  (cron/2min)│    │  backups/YYYY/MM/DD/                │    │
│   └─────────────┘    └─────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | Random Forest (scikit-learn 1.7.2) |
| API | FastAPI + Uvicorn |
| Database | PostgreSQL 15 |
| Secrets | HashiCorp Vault (KV v2) |
| Object Storage | MinIO (S3-compatible) |
| Encryption | Fernet / AES-128-CBC |
| Integrity | SHA-256 (before + after encryption) |
| Monitoring | Prometheus + Grafana + cAdvisor |
| Drift Detection | Kolmogorov-Smirnov test |
| Containerisation | Docker + Docker Compose |
| CI/CD | GitHub Actions + GHCR |

---

## Project Structure

```
.
├── app/
│   ├── main.py                        # FastAPI app, routes, Prometheus metrics
│   ├── model.py                       # Random Forest wrapper
│   ├── database.py                    # SQLAlchemy ORM, Vault-based URL resolution
│   ├── drift.py                       # KS drift detector
│   ├── schemas.py                     # Pydantic input/output schemas
│   ├── random_forest_fraud_model.joblib
│   └── robust_scaler.joblib
├── scripts/
│   ├── backup_secure.py               # Encrypted backup to S3
│   ├── restore_secure.py              # Controlled restore with 3 integrity checks
│   ├── generate_baseline.py           # Generates drift detection baseline
│   └── visualize_drift.py             # Drift visualisation plots
├── training/
│   └── train.py                       # Model training pipeline
├── tests/
│   ├── test_api.py                    # API endpoint tests
│   ├── test_drift.py                  # Drift detector tests
│   ├── test_evaluate_model.py         # Model evaluation tests
│   └── test_postgres.py               # Database tests
├── monitoring/
│   ├── prometheus.yml                 # Prometheus scrape config
│   └── grafana/
│       ├── dashboards/                # Pre-built Grafana dashboards
│       └── provisioning/              # Auto-provisioning config
├── cron/
│   └── crontab                        # Backup cron schedule (every 2 min)
├── data/
│   └── data.py                        # Data loading utilities
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Prerequisites

- Docker Desktop (with Compose v2)
- Git
- `curl` or any REST client (for testing)

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Bistrel2002/anomaly.git
cd anomaly
```

### 2. Create your environment file

```bash
cp .env.example .env
```

Edit `.env` with your values. The only secrets you need to set are:

```env
POSTGRES_PASSWORD=your_password
VAULT_TOKEN=your_vault_token
AWS_ACCESS_KEY_ID=your_minio_user
AWS_SECRET_ACCESS_KEY=your_minio_password
GF_ADMIN_PASSWORD=your_grafana_password
```

Everything else (DATABASE_URL, MinIO credentials) will be automatically stored in Vault at first boot and retrieved from there at runtime.

### 3. Start the full stack

```bash
docker-compose up -d
```

On first boot, two init containers run automatically:
- **vault-init** — stores the database URL and MinIO credentials in Vault
- **minio-init** — creates the backup bucket in MinIO

### 4. Verify everything is running

```bash
docker ps
```

Expected containers: `anomaly_api`, `anomaly_postgres`, `anomaly_vault`, `anomaly_minio`, `anomaly_backup`, `anomaly_prometheus`, `anomaly_grafana`, `anomaly_cadvisor`

---

## Services & Ports

| Service | URL | Description |
|---|---|---|
| Fraud Detection API | http://localhost:8001 | Main inference API |
| API Docs (Swagger) | http://localhost:8001/docs | Interactive API documentation |
| Grafana | http://localhost:3000 | Monitoring dashboards |
| Prometheus | http://localhost:9090 | Metrics collection |
| HashiCorp Vault | http://localhost:8201 | Secrets management |
| MinIO Console | http://localhost:9003 | Backup file browser |
| cAdvisor | http://localhost:8083 | Container resource metrics |

---

## API Endpoints

### Health check

```
GET /
```

### Single transaction prediction

```
POST /predict
Content-Type: application/json

{
  "Time": 406.0,
  "V1": -2.31, "V2": 1.95, "V3": -1.60, "V4": 3.99,
  "V5": -0.52, "V6": -1.42, "V7": -2.53, "V8": 1.39,
  "V9": -2.77, "V10": -2.77, "V11": 3.20, "V12": -2.89,
  "V13": -0.59, "V14": -4.28, "V15": 0.38, "V16": -1.14,
  "V17": -2.83, "V18": -0.01, "V19": 0.41, "V20": 0.12,
  "V21": 0.51, "V22": -0.03, "V23": -0.46, "V24": 0.32,
  "V25": 0.04, "V26": 0.17, "V27": 0.26, "V28": -0.14,
  "Amount": 0.0
}
```

Response:
```json
{
  "transaction_time": 406.0,
  "transaction_amount": 0.0,
  "is_fraud": true,
  "fraud_probability": 0.8835,
  "message": "ALERT: Suspicious Transaction (Fraud likely)!"
}
```

### Batch prediction

```
POST /predict/batch
Content-Type: application/json

[
  { "Time": 0.0, "V1": -1.35, ..., "Amount": 149.62 },
  { "Time": 406.0, "V1": -2.31, ..., "Amount": 0.0 }
]
```

Returns an array of prediction results in the same order.

### Drift status

```
GET /drift
```

Returns the latest KS-test drift results across all features. Drift is computed every 150 predictions.

---

## Security Architecture

All secrets are managed exclusively by HashiCorp Vault. Services only need `VAULT_ADDR` and `VAULT_TOKEN` to bootstrap — everything else is fetched from Vault at runtime.

### Secrets stored in Vault

| Path | Contents |
|---|---|
| `secret/database/prod` | PostgreSQL password + connection URL |
| `secret/minio/prod` | MinIO access key + secret key |
| `secret/backup/symmetric_key` | Fernet AES-128-CBC encryption key |

### Verify Vault contents

```bash
docker exec anomaly_vault sh -c \
  "VAULT_ADDR=http://127.0.0.1:8201 VAULT_TOKEN=root vault kv list secret/"
```

---

## Automated Backups

The backup container runs `backup_secure.py` every 2 minutes via cron.

### Backup pipeline

1. Fetch DB password from Vault (`secret/database/prod`)
2. Fetch MinIO credentials from Vault (`secret/minio/prod`)
3. Get or create Fernet symmetric key from Vault (`secret/backup/symmetric_key`)
4. `pg_dump` → `.sql` (plaintext, temporary)
5. SHA-256 hash before encryption
6. Fernet encrypt → `.enc`
7. SHA-256 hash after encryption
8. Generate metadata JSON (15 fields including both hashes, row count, key reference)
9. Upload `.enc` + `.json` to MinIO at `backups/YYYY/MM/DD/`
10. Delete plaintext dump from disk

### Run a backup manually

```bash
docker exec anomaly_backup sh -c "cd /code && python scripts/backup_secure.py"
```

### Check backup files in MinIO

```bash
docker run --rm --network anomaly_default --entrypoint sh minio/mc \
  -c "mc alias set local http://minio:9000 minioadmin minioadmin 2>/dev/null \
      && mc ls --recursive local/anomaly-backups/"
```

### Restore a backup

```bash
docker exec -it anomaly_backup sh -c "cd /code && python scripts/restore_secure.py"
```

The restore script performs 3 integrity checks:
1. Hash of downloaded `.enc` must match `hash_apres_chiffrement` from JSON
2. Hash of decrypted `.sql` must match `hash_avant_chiffrement` from JSON
3. Row count after restore must match the row count snapshot from backup time

---

## Monitoring

### Prometheus metrics exposed by the API

| Metric | Type | Description |
|---|---|---|
| `fraud_predictions_total` | Counter | Total predictions, labelled `fraud` / `normal` |
| `fraud_probability_score` | Histogram | Distribution of fraud probability scores |
| `prediction_latency_seconds` | Histogram | Inference time per request |
| `data_drift_detected` | Gauge | 1 if KS drift detected for a feature, 0 otherwise |
| `http_requests_total` | Counter | All HTTP requests (auto-instrumented) |

### Grafana

Open http://localhost:3000 (default: `admin` / your `GF_ADMIN_PASSWORD`).

The Fraud Detection dashboard is pre-provisioned and shows:
- Prediction rate (fraud vs normal)
- Fraud probability distribution
- API latency
- Container CPU and memory via cAdvisor

---

## Data Drift Detection

The API runs a Kolmogorov-Smirnov test every 150 predictions, comparing the live distribution of each feature against the training baseline.

### Generate the baseline

```bash
docker exec anomaly_api python scripts/generate_baseline.py
```

### Check drift status

```bash
curl http://localhost:8001/drift
```

If drift is detected on a feature, the `data_drift_detected{feature="Vn"}` Prometheus gauge is set to 1, which triggers a Grafana alert.

---

## Running Tests

```bash
# From the project root (requires local Python env with requirements installed)
pytest tests/ -v
```

Or inside the API container:

```bash
docker exec anomaly_api pytest tests/ -v
```

---

## CI/CD

GitHub Actions runs on every push to `main`:

1. **Test** — runs `pytest` against the full test suite
2. **Build & Push** — builds the Docker image and pushes to GitHub Container Registry (GHCR) as `ghcr.io/<username>/anomaly:latest`

The workflow file is at `.github/workflows/ci.yml`.

---

## Environment Variables Reference

| Variable | Used by | Description |
|---|---|---|
| `POSTGRES_USER` | PostgreSQL container | Database user |
| `POSTGRES_PASSWORD` | PostgreSQL + vault-init | Database password (also stored in Vault) |
| `POSTGRES_DB` | PostgreSQL container | Database name |
| `DATABASE_URL` | vault-init only | Full connection URL (stored in Vault, not used by API directly) |
| `VAULT_ADDR` | All services | Vault server address |
| `VAULT_TOKEN` | All services | Vault authentication token |
| `AWS_ACCESS_KEY_ID` | MinIO + vault-init | MinIO root user (also stored in Vault) |
| `AWS_SECRET_ACCESS_KEY` | MinIO + vault-init | MinIO root password (also stored in Vault) |
| `AWS_DEFAULT_REGION` | backup/restore | S3 region (use `us-east-1` for MinIO) |
| `AWS_ENDPOINT_URL` | backup/restore | MinIO internal endpoint (`http://minio:9000`) |
| `S3_BUCKET_NAME` | backup/restore | Target bucket name |
| `GF_ADMIN_USER` | Grafana | Admin username |
| `GF_ADMIN_PASSWORD` | Grafana | Admin password |

---

## Model

- **Algorithm:** Random Forest Classifier
- **Dataset:** Kaggle Credit Card Fraud Detection (284,807 transactions, 492 fraud)
- **Features:** Time, V1–V28 (PCA-anonymised), Amount (RobustScaler-normalised)
- **Evaluation:** ROC-AUC, Precision, Recall, Confusion Matrix (see `evaluation_output/`)
- **Artefacts:** `app/random_forest_fraud_model.joblib`, `app/robust_scaler.joblib`

---

## License

This project is for personal and demonstration purposes.

