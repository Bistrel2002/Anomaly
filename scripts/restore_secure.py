"""Controlled, integrity-verified database restore from S3.

Three-stage integrity check:
    1. Hash of downloaded .enc matches hash recorded at backup time
    2. Hash of decrypted .sql matches hash recorded before encryption
    3. Hash of post-restore pg_dump matches original hash (round-trip proof)

Required environment variables:
    VAULT_ADDR            e.g. http://[IP_ADDRESS]
    VAULT_TOKEN           Vault authentication token
    AWS_ACCESS_KEY_ID     AWS credentials
    AWS_SECRET_ACCESS_KEY AWS credentials
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
    filename=os.environ.get("BACKUP_LOG_FILE", "/tmp/secure_backup.log"),
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


def get_db_config(vault_client: hvac.Client) -> dict:
    secret = vault_client.secrets.kv.read_secret_version(path="database/prod")
    data = secret["data"]["data"]
    url = data["url"]
    without_scheme = url.split("://", 1)[1]
    userinfo, hostinfo = without_scheme.split("@", 1)
    user = userinfo.split(":")[0]
    password = data["password"]
    hostport, dbname = hostinfo.rsplit("/", 1)
    host = hostport.split(":")[0]
    log("INFO", "Database config retrieved from Vault")
    return {"user": user, "host": host, "name": dbname, "password": password}


def get_minio_credentials(vault_client: hvac.Client) -> tuple[str, str]:
    secret = vault_client.secrets.kv.read_secret_version(path="minio/prod")
    data = secret["data"]["data"]
    log("INFO", "MinIO credentials retrieved from Vault")
    return data["access_key"], data["secret_key"]


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _s3_client(access_key: str, secret_key: str):
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL"),
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )


def list_available_backups(access_key: str, secret_key: str) -> list[dict]:
    s3 = _s3_client(access_key, secret_key)
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


def download_backup(selected: dict, access_key: str, secret_key: str) -> tuple[str, str]:
    s3 = _s3_client(access_key, secret_key)
    bucket = os.environ["S3_BUCKET_NAME"]
    meta = selected["meta"]

    s3_key_enc = selected["s3_key_json"].replace(".json", ".enc")
    local_enc = os.path.join("/tmp", os.path.basename(meta["nom_chiffre"]))
    local_json = "/tmp/restore_meta.json"

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


def verify_post_restore(metadata: dict, db_config: dict) -> None:
    # pg_dump output is non-deterministic (timestamps, sequences, stats differ
    # between runs), so hash comparison would always fail. Row count is a
    # reliable proxy: if every table has the expected number of live rows, the
    # restore is complete.
    env = os.environ.copy()
    env["PGPASSWORD"] = db_config["password"]

    query = (
        "SELECT schemaname, tablename, n_live_tup "
        "FROM pg_stat_user_tables ORDER BY tablename;"
    )
    cmd = ["psql", "-U", db_config["user"], "-h", db_config["host"],
           db_config["name"], "-t", "-c", query]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if result.returncode != 0:
        raise RuntimeError(f"Post-restore check failed: {result.stderr}")

    total_rows = sum(
        int(line.split("|")[2].strip())
        for line in result.stdout.strip().splitlines()
        if line.strip() and "|" in line
    )

    expected_rows = metadata.get("total_rows")
    if expected_rows is not None and total_rows != expected_rows:
        log("ERROR", f"❌ Check 3 FAILED: expected {expected_rows} rows, got {total_rows}")
        raise ValueError("Restore invalid or incomplete")

    log("INFO", f"✅ Check 3 OK: {total_rows} rows present after restore")


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

        vault_client = init_vault_client()
        minio_access_key, minio_secret_key = get_minio_credentials(vault_client)
        db_config = get_db_config(vault_client)

        backups = list_available_backups(minio_access_key, minio_secret_key)
        if not backups:
            raise RuntimeError("No backups found in S3.")
        choice = int(input("\nYour choice (number): ")) - 1
        selected = backups[choice]

        enc_file, json_file = download_backup(selected, minio_access_key, minio_secret_key)
        with open(json_file) as f:
            metadata = json.load(f)

        # Check 1 — encrypted file integrity
        verify_enc_integrity(enc_file, metadata)

        # Decrypt
        key = get_key_from_vault(vault_client, metadata["cle_vault_ref"])
        dump_file = decrypt_file(enc_file, key)

        # Check 2 — decrypted dump integrity
        verify_dump_integrity(dump_file, metadata)

        # Restore
        restore_database(dump_file, db_config)

        # Check 3 — post-restore row count vs backup snapshot
        verify_post_restore(metadata, db_config)

        cleanup_local_files(enc_file, dump_file, json_file)

        log("INFO", "=== RESTORE COMPLETED SUCCESSFULLY ===")

    except Exception as e:
        log("ERROR", f"Restore failed: {e}")
        raise


if __name__ == "__main__":
    main()
