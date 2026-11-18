
import csv
import random
import hashlib
from datetime import datetime, timedelta, timezone

random.seed(42)

OUTPUT_FILE = "data/raw/bitcoin_transactions.csv"
TOTAL_RECORDS = 1000

script_types = ["P2PKH", "P2SH", "P2WPKH", "P2TR"]

def generate_txid():
    return hashlib.sha256(
        str(random.random()).encode()
    ).hexdigest()

def generate_wallet():
    return "wallet_" + str(random.randint(1, 500))

def generate_ip():
    return ".".join(
        str(random.randint(1, 254)) for _ in range(4)
    )

start_time = datetime(2026, 1, 1, tzinfo=timezone.utc)

with open(OUTPUT_FILE, "w", newline="") as file:

    writer = csv.DictWriter(file, fieldnames=[
        "timestamp",
        "src_ip",
        "dst_ip",
        "src_port",
        "dst_port",
        "txid",
        "input_addresses",
        "output_addresses",
        "input_amounts",
        "output_amounts",
        "fee_sats",
        "script_type",
        "scenario_label"
    ])

    writer.writeheader()

    for i in range(TOTAL_RECORDS):

        timestamp = start_time + timedelta(
            seconds=random.randint(0, 86400 * 30)
        )

        input_wallets = [
            generate_wallet()
            for _ in range(random.randint(1, 4))
        ]

        output_wallets = [
            generate_wallet()
            for _ in range(random.randint(1, 4))
        ]

        input_amounts = [
            random.randint(10000, 1000000)
            for _ in input_wallets
        ]

        output_amounts = [
            random.randint(5000, 900000)
            for _ in output_wallets
        ]

        fee = random.randint(100, 5000)

        # Inject a synthetic unusual-fee scenario
        is_anomaly = random.random() < 0.05

        if is_anomaly:
            fee = random.randint(100000, 500000)
            src_ip = "203.0.113.10"
        else:
            src_ip = generate_ip()

        writer.writerow({
            "timestamp": timestamp.isoformat(),
            "src_ip": src_ip,
            "dst_ip": generate_ip(),
            "src_port": random.randint(1000, 65535),
            "dst_port": 8333,
            "txid": generate_txid(),
            "input_addresses": "|".join(input_wallets),
            "output_addresses": "|".join(output_wallets),
            "input_amounts": "|".join(map(str, input_amounts)),
            "output_amounts": "|".join(map(str, output_amounts)),
            "fee_sats": fee,
            "script_type": random.choice(script_types),
            "scenario_label": int(is_anomaly)
        })

print(f"Generated {TOTAL_RECORDS} synthetic transactions.")
print(f"Saved to {OUTPUT_FILE}")
