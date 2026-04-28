"""Encrypted, integrity-verified database backup to S3.

Pipeline:
    1. Fetch DB password from HashiCorp Vault
    2. Get/create Fernet symmetric key in Vault
    3. pg_dump → .sql
    4. SHA-256 hash before encryption
    5. Fernet encrypt → .enc
    6. SHA-256 hash after encryption
    7. Write metadata JSON
    8. Upload .enc + .json to S3 (backups/YYYY/MM/DD/)
    9. Delete plaintext .sql only after confirmed upload

Required environment variables:
    VAULT_ADDR            e.g. http://127.0.0.1:8200
    VAULT_TOKEN           Vault authentication token
    AWS_ACCESS_KEY_ID     AWS credentials
    AWS_SECRET_ACCESS_KEY AWS credentials
    AWS_DEFAULT_REGION    e.g. eu-west-1
    S3_BUCKET_NAME        Target S3 bucket
    DATABASE_URL          PostgreSQL connection string (used to parse host/user/db)

Usage:
    python scripts/backup_secure.py
"""

import hashlib
import json
import logging
import os
import subprocess
from datetime import datetime

import boto3
import hvac
from cryptography.fernet import Fernet

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    filename="/var/log/secure_backup.log",
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def log(level: str, message: str) -> None:
    getattr(logging, level.lower())(message)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {level} - {message}")


# ---------------------------------------------------------------------------
# Vault helpers
# ---------------------------------------------------------------------------

def init_vault_client() -> hvac.Client:
    client = hvac.Client(
        url=os.environ["VAULT_ADDR"],
        token=os.environ["VAULT_TOKEN"],
    )
    if not client.is_authenticated():
        raise RuntimeError("Vault: authentication failed")
    log("INFO", "[OK] Vault connection successful — secret retrieved")
    return client


def get_db_password_from_vault(vault_client: hvac.Client) -> str:
    secret = vault_client.secrets.kv.read_secret_version(path="database/prod")
    return secret["data"]["data"]["password"]


def get_or_create_symmetric_key(vault_client: hvac.Client) -> bytes:
    path = "backup/symmetric_key"
    try:
        secret = vault_client.secrets.kv.read_secret_version(path=path)
        key = secret["data"]["data"]["key"].encode()
        log("INFO", "Symmetric key retrieved from Vault")
    except Exception:
        key = Fernet.generate_key()
        vault_client.secrets.kv.create_or_update_secret(
            path=path,
            secret={"key": key.decode()},
        )
        log("INFO", "Symmetric key generated and stored in Vault")
    return key


# ---------------------------------------------------------------------------
# DB config
# ---------------------------------------------------------------------------

def get_db_config(db_password: str) -> dict:
    """Parse DATABASE_URL into a dict for pg_dump."""
    url = os.environ["DATABASE_URL"]
    # postgresql://user:password@host:port/dbname
    without_scheme = url.split("://", 1)[1]
    userinfo, hostinfo = without_scheme.split("@", 1)
    user = userinfo.split(":")[0]
    hostport, dbname = hostinfo.rsplit("/", 1)
    host = hostport.split(":")[0]
    return {"type": "postgresql", "user": user, "host": host,
            "name": dbname, "password": db_password}


# ---------------------------------------------------------------------------
# Core steps
# ---------------------------------------------------------------------------

def generate_dump(db_config: dict) -> tuple[str, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"db_backup_{timestamp}.sql"

    env = os.environ.copy()
    env["PGPASSWORD"] = db_config["password"]

    cmd = ["pg_dump", "-U", db_config["user"], "-h", db_config["host"], db_config["name"]]
    with open(filename, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, env=env)

    if result.returncode != 0:
        raise RuntimeError(f"pg_dump failed: {result.stderr.decode()}")

    size = os.path.getsize(filename)
    log("INFO", f"Dump generated: {filename} ({size} bytes)")
    return filename, timestamp


def compute_hash(filepath: str) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def encrypt_file(input_path: str, key: bytes) -> str:
    fernet = Fernet(key)
    output_path = input_path.replace(".sql", ".enc")
    with open(input_path, "rb") as f:
        data = f.read()
    with open(output_path, "wb") as f:
        f.write(fernet.encrypt(data))
    log("INFO", f"Encrypted file produced: {output_path}")
    return output_path


def generate_metadata_json(timestamp: str, dump_file: str, enc_file: str,
                            hash_before: str, hash_after: str,
                            key_vault_ref: str, db_config: dict) -> str:
    metadata = {
        "date": timestamp[:8],
        "heure": timestamp[9:],
        "nom_dump": dump_file,
        "nom_chiffre": enc_file,
        "type_bdd": db_config["type"],
        "nom_base": db_config["name"],
        "algorithme_hash": "sha256",
        "hash_avant_chiffrement": hash_before,
        "hash_apres_chiffrement": hash_after,
        "algorithme_chiffrement": "Fernet/AES-128-CBC",
        "taille_dump": os.path.getsize(dump_file),
        "taille_enc": os.path.getsize(enc_file),
        "cle_vault_ref": key_vault_ref,
        "statut": "success",
        "erreur": None,
    }
    json_file = dump_file.replace(".sql", ".json")
    with open(json_file, "w") as f:
        json.dump(metadata, f, indent=2)
    log("INFO", f"Metadata JSON generated: {json_file}")
    return json_file


def upload_to_s3(enc_file: str, json_file: str, timestamp: str) -> None:
    endpoint = os.environ.get("AWS_ENDPOINT_URL")
    s3 = boto3.client("s3", endpoint_url=endpoint)
    bucket = os.environ["S3_BUCKET_NAME"]
    date_path = f"{timestamp[:4]}/{timestamp[4:6]}/{timestamp[6:8]}"

    for local_file in [enc_file, json_file]:
        s3_key = f"backups/{date_path}/{os.path.basename(local_file)}"
        s3.upload_file(local_file, bucket, s3_key)
        log("INFO", f"Uploaded to S3: s3://{bucket}/{s3_key}")


def delete_plaintext_dump(dump_file: str) -> None:
    os.remove(dump_file)
    if os.path.exists(dump_file):
        raise RuntimeError(f"ALERT: {dump_file} was not deleted!")
    log("INFO", f"Plaintext dump deleted: {dump_file}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        log("INFO", "=== BACKUP START ===")

        vault_client = init_vault_client()
        db_password = get_db_password_from_vault(vault_client)
        key = get_or_create_symmetric_key(vault_client)
        key_vault_ref = "backup/symmetric_key"

        db_config = get_db_config(db_password)
        dump_file, timestamp = generate_dump(db_config)

        hash_before = compute_hash(dump_file)
        log("INFO", f"Hash before encryption (SHA-256): {hash_before}")

        enc_file = encrypt_file(dump_file, key)

        hash_after = compute_hash(enc_file)
        log("INFO", f"Hash after encryption (SHA-256): {hash_after}")

        json_file = generate_metadata_json(
            timestamp, dump_file, enc_file,
            hash_before, hash_after, key_vault_ref, db_config,
        )

        upload_to_s3(enc_file, json_file, timestamp)
        delete_plaintext_dump(dump_file)

        log("INFO", "=== BACKUP COMPLETED SUCCESSFULLY ===")

    except Exception as e:
        log("ERROR", f"Backup failed: {e}")
        raise


if __name__ == "__main__":
    main()
