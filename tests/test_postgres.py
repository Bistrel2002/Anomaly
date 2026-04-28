"""Integration test: API prediction → PostgreSQL persistence.

This script sends a mix of legitimate and fraudulent transactions to
the running API, then queries PostgreSQL to verify that every prediction
was correctly persisted.

Prerequisites:
    - The API must be running on ``http://127.0.0.1:8001``.
    - PostgreSQL must be accessible on ``127.0.0.1:5433``.

Usage:
    PYTHONPATH=. python tests/test_postgres.py
"""

import time

import requests
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import TransactionRecord

DATABASE_URL = "postgresql://anomaly_user:anomaly_password@127.0.0.1:5433/anomaly_db"
API_URL = "http://127.0.0.1:8001/predict"

# ---------------------------------------------------------------------------
# Fixture data – legitimate transactions
# ---------------------------------------------------------------------------

LEGIT_1 = {
    "Time": 0.0,
    "V1": -1.3598071336738, "V2": -0.0727811733098497,
    "V3": 2.53634673796914, "V4": 1.37815522427443,
    "V5": -0.338320769942518, "V6": 0.462387777762292,
    "V7": 0.239598554061257, "V8": 0.0986979012610507,
    "V9": 0.363786969611213, "V10": 0.0907941719789316,
    "V11": -0.551599533260813, "V12": -0.617800855762348,
    "V13": -0.991389847235408, "V14": -0.311169353699879,
    "V15": 1.46817697209427, "V16": -0.470400525259478,
    "V17": 0.207971241929242, "V18": 0.0257905801985591,
    "V19": 0.403992960255733, "V20": 0.251412098239705,
    "V21": -0.018306777944153, "V22": 0.277837575558899,
    "V23": -0.110473910188767, "V24": 0.0669280749146731,
    "V25": 0.128539358273528, "V26": -0.189114843888824,
    "V27": 0.133558376740387, "V28": -0.0210530534538215,
    "Amount": 149.62,
    "_label": "LEGIT — everyday purchase (149 EUR)",
}

LEGIT_2 = {
    "Time": 3600.0,
    "V1": 1.19185711131486, "V2": 0.266150712616991,
    "V3": 0.166480113814521, "V4": 0.448154078460911,
    "V5": 0.0600176492822243, "V6": -0.0823608088155687,
    "V7": -0.0788029833562369, "V8": 0.0851016549108859,
    "V9": -0.255425128109186, "V10": -0.166974414004614,
    "V11": 1.61272666105479, "V12": 1.06523531137287,
    "V13": 0.489095437869789, "V14": -0.143772296441519,
    "V15": 0.635558093258208, "V16": 0.463917041022677,
    "V17": -0.114804663102346, "V18": -0.183361270123994,
    "V19": -0.145783041325259, "V20": -0.069083135467068,
    "V21": -0.225775248033138, "V22": -0.638671952771851,
    "V23": 0.101288021253234, "V24": -0.339846475529127,
    "V25": 0.167170404418143, "V26": 0.125894532368176,
    "V27": -0.00898309914322813, "V28": 0.0147241691924927,
    "Amount": 2.69,
    "_label": "LEGIT — small transaction (2.69 EUR)",
}

LEGIT_3 = {
    "Time": 7200.0,
    "V1": -0.966271711572060, "V2": -0.185226008082898,
    "V3": 1.79299333957872, "V4": -0.863291275036453,
    "V5": -0.0103088796030823, "V6": 1.24720316752486,
    "V7": 0.237608940919978, "V8": 0.377435874652262,
    "V9": -1.38702406270197, "V10": -0.0549519224713749,
    "V11": -0.226487263835401, "V12": 0.178228225877303,
    "V13": 0.507756869957169, "V14": -0.287923521596745,
    "V15": -0.631418117537453, "V16": -1.05964725027099,
    "V17": -0.684092786345479, "V18": 1.96577500353669,
    "V19": -1.23262197788648, "V20": -0.208037781160366,
    "V21": -0.108300452035598, "V22": 0.00527359351820244,
    "V23": -0.190320518742841, "V24": 0.703337945499006,
    "V25": -0.506271393938518, "V26": -0.012545523166931,
    "V27": -0.14939648941954, "V28": -0.313628642682921,
    "Amount": 378.66,
    "_label": "LEGIT — online purchase (378 EUR)",
}

LEGIT_4 = {
    "Time": 14400.0,
    "V1": 2.15634735177897, "V2": 0.425412452765629,
    "V3": 1.03048825428584, "V4": 1.38927297312997,
    "V5": 0.310849680285451, "V6": 0.461613695817647,
    "V7": 0.476241900380573, "V8": 0.155903607834524,
    "V9": 0.201401592659097, "V10": 0.347697748748148,
    "V11": 0.828452498851688, "V12": 0.612571244819475,
    "V13": -0.279830019936453, "V14": 0.436580082787988,
    "V15": 0.523083477432808, "V16": -0.195839026765555,
    "V17": 0.0714823783481961, "V18": 0.381888282396955,
    "V19": 0.193540474769041, "V20": 0.131958659443939,
    "V21": -0.0629581170050766, "V22": 0.0962984965296567,
    "V23": -0.0815695769504126, "V24": 0.155325282019285,
    "V25": 0.0513659044029264, "V26": 0.277227499474972,
    "V27": 0.0153785799990718, "V28": 0.00981266413563684,
    "Amount": 29.95,
    "_label": "LEGIT — monthly subscription (29.95 EUR)",
}

# ---------------------------------------------------------------------------
# Fixture data – fraudulent transactions
# ---------------------------------------------------------------------------
# Fraud samples exhibit extreme values on V10, V12, V14, V17
# (features with strong fraud correlation).

FRAUD_1 = {
    "Time": 406.0,
    "V1": -2.3122265423263, "V2": 1.95199201064158,
    "V3": -1.60985073229769, "V4": 3.9979055875468,
    "V5": -0.522187864667764, "V6": -1.42654531920595,
    "V7": -2.53738730624579, "V8": 1.39165724829804,
    "V9": -2.77008927719433, "V10": -2.77227214465915,
    "V11": 3.20203320709635, "V12": -2.89990738849473,
    "V13": -0.595221881324605, "V14": -4.28925378244217,
    "V15": 0.389724120274487, "V16": -1.14074717980657,
    "V17": -2.83005567450437, "V18": -0.0168224681808257,
    "V19": 0.416955705037907, "V20": 0.126910559061074,
    "V21": 0.517232370861764, "V22": -0.0350493686052974,
    "V23": -0.465211076998498, "V24": 0.320198198514526,
    "V25": 0.0445191674731724, "V26": 0.177839798284401,
    "V27": 0.261145002567677, "V28": -0.143275874698919,
    "Amount": 239.93,
    "_label": "FRAUD — fake payment (239 EUR)",
}

FRAUD_2 = {
    "Time": 472.0,
    "V1": -3.0435406239976, "V2": -3.15730712090173,
    "V3": 1.08846277937187, "V4": 2.2886436183814,
    "V5": 1.35980512966016, "V6": -1.06482252961849,
    "V7": 0.325574266158614, "V8": -0.0677936531906209,
    "V9": -0.270952836733825, "V10": -9.09837246118459,
    "V11": 9.71432781280393, "V12": -12.5614879849717,
    "V13": -7.44516834530605, "V14": -16.6496281595399,
    "V15": 4.43797780032895, "V16": -6.40402438219738,
    "V17": -8.23615490684877, "V18": -0.431061697957323,
    "V19": 0.131542494062139, "V20": 1.21704637936765,
    "V21": 0.76443547706533, "V22": 0.331544530942613,
    "V23": 0.310694770803726, "V24": 0.154699833625849,
    "V25": -0.0315448826699452, "V26": 0.0889682611640618,
    "V27": -0.0549062512036694, "V28": 0.0146803521843559,
    "Amount": 529.00,
    "_label": "FRAUD — atypical high-value transaction (529 EUR)",
}

FRAUD_3 = {
    "Time": 6100.0,
    "V1": 1.23659501624986, "V2": 3.01888247879069,
    "V3": -4.30496924408079, "V4": 4.73279709935952,
    "V5": -2.29767965793998, "V6": -3.56757566025648,
    "V7": -0.137458079999604, "V8": -2.18755296049043,
    "V9": 0.509815823399366, "V10": -8.97407453580576,
    "V11": 3.15803578380879, "V12": -10.7783491165272,
    "V13": 0.538818614048857, "V14": -7.08498289748149,
    "V15": -0.255614892213393, "V16": -3.78325127924936,
    "V17": -4.52491639577651, "V18": -0.00695819264173003,
    "V19": -0.258905404743073, "V20": -0.455808801754767,
    "V21": 0.44622001011891, "V22": 0.0988700441091394,
    "V23": 0.00753199052249437, "V24": 0.234527098695616,
    "V25": 0.183283993208424, "V26": -0.183754236154697,
    "V27": 0.0124476025984426, "V28": 0.0286866745847199,
    "Amount": 1.00,
    "_label": "FRAUD — 1 EUR card-test before attack",
}

ALL_TRANSACTIONS = [LEGIT_1, LEGIT_2, LEGIT_3, LEGIT_4, FRAUD_1, FRAUD_2, FRAUD_3]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def send_transaction(tx: dict) -> dict | None:
    """POST a transaction to the API and return the JSON response.

    Internal ``_label`` keys (prefixed with ``_``) are stripped before
    sending.

    Args:
        tx: Transaction dictionary including optional ``_label`` metadata.

    Returns:
        Parsed JSON response on success, or ``None`` on error.
    """
    payload = {k: v for k, v in tx.items() if not k.startswith("_")}
    response = requests.post(API_URL, json=payload, timeout=10)
    if response.status_code == 200:
        return response.json()
    print(f"  ERROR: API responded with {response.status_code}: {response.text}")
    return None


def check_db_record(session, label: str) -> None:
    """Query the latest record in PostgreSQL and print its summary.

    Args:
        session: Active SQLAlchemy session.
        label: Human-readable label for console output.
    """
    record = (
        session.query(TransactionRecord)
        .order_by(TransactionRecord.id.desc())
        .first()
    )
    if record:
        flag = "FRAUD" if record.is_fraud else "LEGIT"
        print(
            f"  [{flag}] prob={record.fraud_probability:.4f}  "
            f"amount={record.amount} EUR  — {label}"
        )
    else:
        print("  ERROR: No record found in the database.")


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------


def run_test() -> None:
    """Send all fixture transactions and verify database persistence."""
    print("=" * 60)
    print("POSTGRES INTEGRATION TEST — sending multiple transactions")
    print("=" * 60)

    engine = create_engine(DATABASE_URL)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = session_factory()

    for tx in ALL_TRANSACTIONS:
        label = tx.get("_label", "unlabelled")
        print(f"\n-> Sending: {label}")
        result = send_transaction(tx)
        if result is None:
            continue
        time.sleep(0.5)
        check_db_record(session, label)

    session.close()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY — latest records in database")
    print("=" * 60)

    session2 = session_factory()
    records = (
        session2.query(TransactionRecord)
        .order_by(TransactionRecord.id.desc())
        .limit(len(ALL_TRANSACTIONS))
        .all()
    )
    frauds = sum(1 for r in records if r.is_fraud)
    legits = len(records) - frauds
    print(f"  Records retrieved : {len(records)}")
    print(f"  Frauds detected   : {frauds}")
    print(f"  Legitimate        : {legits}")
    session2.close()


if __name__ == "__main__":
    run_test()