"""Controlled, integrity-verified database restore from S3.

Three-stage integrity check:
    1. Hash of downloaded .enc matches hash recorded at backup time
    2. Hash of decrypted .sql matches hash recorded before encryption
    3. Hash of post-restore pg_dump matches original hash (round-trip proof)

Required environment variables:
    VAULT_ADDR            e.g. http://127.0.0.1:8200
    VAULT_TOKEN           Vault authentication token
    AWS_ACCESS_KEY_ID     AWS credentials
    AWS_SECRET_ACCESS_KEY AWS credentials
    AWS_DEFAULT_REGION    e.g. eu-west-1
    S3_BUCKET_NAME        Source S3 bucket
    DATABASE_URL          PostgreSQL connection string

Usage:
    python scripts/restore_secure.py
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
# Shared helpers (duplicated intentionally — restore is standalone)
# ---------------------------------------------------------------------------

def compute_hash(filepath: str) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def init_vault_client() -> hvac.Client:
    client = hvac.Client(
        url=os.environ["VAULT_ADDR"],
        token=os.environ["VAULT_TOKEN"],
    )
    if not client.is_authenticated():
        raise RuntimeError("Vault: authentication failed")
    log("INFO", "[OK] Vault connection successful")
    return client


def get_db_config() -> dict:
    url = os.environ["DATABASE_URL"]
    without_scheme = url.split("://", 1)[1]
    userinfo, hostinfo = without_scheme.split("@", 1)
    user = userinfo.split(":")[0]
    password = userinfo.split(":")[1] if ":" in userinfo else ""
    hostport, dbname = hostinfo.rsplit("/", 1)
    host = hostport.split(":")[0]
    return {"user": user, "host": host, "name": dbname, "password": password}


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def list_available_backups() -> list[dict]:
    endpoint = os.environ.get("AWS_ENDPOINT_URL")
    s3 = boto3.client("s3", endpoint_url=endpoint)
    bucket = os.environ["S3_BUCKET_NAME"]
    response = s3.list_objects_v2(Bucket=bucket, Prefix="backups/")

    json_keys = sorted(
        obj["Key"] for obj in response.get("Contents", [])
        if obj["Key"].endswith(".json")
    )

    backups = []
    for jk in json_keys:
        obj = s3.get_object(Bucket=bucket, Key=jk)
        meta = json.loads(obj["Body"].read())
        backups.append({"s3_key_json": jk, "meta": meta})

    print("\nAvailable backups:")
    for i, b in enumerate(backups, 1):
        m = b["meta"]
        print(f"  [{i}] {m['nom_chiffre']}  —  {m['date']} {m['heure']}"
              f"  —  {m['taille_enc']} bytes  —  status: {m['statut']}")

    return backups


def download_backup(selected: dict) -> tuple[str, str]:
    endpoint = os.environ.get("AWS_ENDPOINT_URL")
    s3 = boto3.client("s3", endpoint_url=endpoint)
    bucket = os.environ["S3_BUCKET_NAME"]
    meta = selected["meta"]

    s3_key_enc = selected["s3_key_json"].replace(".json", ".enc")
    local_enc = meta["nom_chiffre"]
    local_json = "restore_meta.json"

    s3.download_file(bucket, s3_key_enc, local_enc)
    s3.download_file(bucket, selected["s3_key_json"], local_json)
    log("INFO", f"Archive downloaded: {local_enc}")
    return local_enc, local_json


# ---------------------------------------------------------------------------
# Integrity checks
# ---------------------------------------------------------------------------

def verify_enc_integrity(enc_file: str, metadata: dict) -> None:
    actual = compute_hash(enc_file)
    expected = metadata["hash_apres_chiffrement"]
    if actual != expected:
        raise ValueError(
            f"INTEGRITY COMPROMISED: encrypted hash expected={expected}, got={actual}"
        )
    log("INFO", "✅ Check 1 OK: encrypted archive is intact")


def verify_dump_integrity(dump_file: str, metadata: dict) -> None:
    actual = compute_hash(dump_file)
    expected = metadata["hash_avant_chiffrement"]
    if actual != expected:
        raise ValueError(
            f"DECRYPTION INCORRECT: dump hash expected={expected}, got={actual}"
        )
    log("INFO", "✅ Check 2 OK: decrypted dump matches original")


def verify_post_restore(metadata: dict, db_config: dict) -> str:
    control_dump = "control_dump.sql"
    env = os.environ.copy()
    env["PGPASSWORD"] = db_config["password"]

    cmd = ["pg_dump", "-U", db_config["user"], "-h", db_config["host"], db_config["name"]]
    with open(control_dump, "w") as f:
        subprocess.run(cmd, stdout=f, check=True, env=env)

    actual = compute_hash(control_dump)
    expected = metadata["hash_avant_chiffrement"]
    if actual != expected:
        log("ERROR", "❌ Check 3 FAILED: restored database differs from backup")
        raise ValueError("Restore invalid or incomplete")

    log("INFO", "✅ Check 3 OK: restored database matches original backup")
    return control_dump


# ---------------------------------------------------------------------------
# Decrypt
# ---------------------------------------------------------------------------

def get_key_from_vault(vault_client: hvac.Client, key_ref: str) -> bytes:
    secret = vault_client.secrets.kv.read_secret_version(path=key_ref)
    key = secret["data"]["data"]["key"].encode()
    log("INFO", f"Key retrieved from Vault: {key_ref}")
    return key


def decrypt_file(enc_file: str, key: bytes) -> str:
    fernet = Fernet(key)
    dump_file = enc_file.replace(".enc", ".sql")
    with open(enc_file, "rb") as f:
        encrypted = f.read()
    with open(dump_file, "wb") as f:
        f.write(fernet.decrypt(encrypted))
    log("INFO", f"Dump decrypted: {dump_file}")
    return dump_file


# ---------------------------------------------------------------------------
# Restore
# ---------------------------------------------------------------------------

def restore_database(dump_file: str, db_config: dict) -> None:
    env = os.environ.copy()
    env["PGPASSWORD"] = db_config["password"]

    cmd = ["psql", "-U", db_config["user"], "-h", db_config["host"], db_config["name"]]
    with open(dump_file, "r") as f:
        result = subprocess.run(cmd, stdin=f, stderr=subprocess.PIPE, env=env)

    if result.returncode != 0:
        raise RuntimeError(f"Restore failed: {result.stderr.decode()}")
    log("INFO", "Database restored successfully")


def cleanup_local_files(*files: str) -> None:
    for f in files:
        if f and os.path.exists(f):
            os.remove(f)
            log("INFO", f"Temporary file deleted: {f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    enc_file = dump_file = control_dump = json_file = None
    try:
        log("INFO", "=== RESTORE START ===")

        backups = list_available_backups()
        if not backups:
            raise RuntimeError("No backups found in S3.")
        choice = int(input("\nYour choice (number): ")) - 1
        selected = backups[choice]

        enc_file, json_file = download_backup(selected)
        with open(json_file) as f:
            metadata = json.load(f)

        # Check 1 — encrypted file integrity
        verify_enc_integrity(enc_file, metadata)

        # Decrypt
        vault_client = init_vault_client()
        key = get_key_from_vault(vault_client, metadata["cle_vault_ref"])
        dump_file = decrypt_file(enc_file, key)

        # Check 2 — decrypted dump integrity
        verify_dump_integrity(dump_file, metadata)

        # Restore
        db_config = get_db_config()
        restore_database(dump_file, db_config)

        # Check 3 — post-restore integrity
        control_dump = verify_post_restore(metadata, db_config)

        cleanup_local_files(enc_file, dump_file, control_dump, json_file)

        log("INFO", "=== RESTORE COMPLETED SUCCESSFULLY ===")

    except Exception as e:
        log("ERROR", f"Restore failed: {e}")
        raise


if __name__ == "__main__":
    main()
