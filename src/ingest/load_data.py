
import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import duckdb

REQUIRED_COLUMNS = [
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
]

ARRAY_COLUMNS = [
    "input_addresses",
    "output_addresses",
    "input_amounts",
    "output_amounts",
]


def normalize_record(record):
    """Convert list-valued fields into pipe-separated strings."""
    for column in ARRAY_COLUMNS:
        value = record.get(column)

        if isinstance(value, list):
            record[column] = "|".join(map(str, value))

    return record


def load_csv(path):
    return pd.read_csv(path)


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        content = file.read().strip()

    try:
        records = json.loads(content)
        if isinstance(records, dict):
            records = records.get("transactions", [records])
    except json.JSONDecodeError:
        # Support JSON Lines format
        records = [
            json.loads(line)
            for line in content.splitlines()
            if line.strip()
        ]

    return pd.DataFrame(
        [normalize_record(dict(record)) for record in records]
    )


def load_xml(path):
    tree = ET.parse(path)
    root = tree.getroot()

    records = []

    for transaction in root.findall(".//transaction"):
        record = {}

        for child in transaction:
            if len(child):
                record[child.tag] = [
                    item.text for item in child
                ]
            else:
                record[child.tag] = child.text

        records.append(normalize_record(record))

    return pd.DataFrame(records)


def validate_data(df):
    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df.empty:
        raise ValueError("Dataset is empty")

    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], utc=True, errors="coerce"
    )

    numeric_columns = [
        "src_port", "dst_port", "fee_sats"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    if "scenario_label" not in df.columns:
        df["scenario_label"] = None

    df["scenario_label"] = pd.to_numeric(
        df["scenario_label"], errors="coerce"
    ).astype("Int64")

    df["src_ip"] = df["src_ip"].astype("string")
    df["dst_ip"] = df["dst_ip"].astype("string")
    df["txid"] = df["txid"].astype("string")

    invalid_mask = (
        df[REQUIRED_COLUMNS].isna().any(axis=1)
        | ~df["txid"].str.fullmatch(r"[0-9a-fA-F]{64}", na=False)
        | ~df["src_port"].between(1, 65535)
        | ~df["dst_port"].between(1, 65535)
        | (df["fee_sats"] < 0)
    )

    rejected = df[invalid_mask].copy()
    valid = df[~invalid_mask].copy()

    return valid, rejected


def ingest(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    extension = path.suffix.lower()

    if extension == ".csv":
        df = load_csv(path)
    elif extension == ".json":
        df = load_json(path)
    elif extension == ".xml":
        df = load_xml(path)
    else:
        raise ValueError("Supported formats: CSV, JSON, XML")

    valid, rejected = validate_data(df)

    Path("data/processed").mkdir(
        parents=True, exist_ok=True
    )

    database = "data/processed/bitcoin.duckdb"

    with duckdb.connect(database) as connection:
        connection.register("valid_data", valid)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS transactions AS
            SELECT * FROM valid_data WHERE 1 = 0
        """)
        connection.execute("""
            INSERT INTO transactions
            SELECT * FROM valid_data
        """)
        connection.unregister("valid_data")

    rejected.to_csv(
        "data/processed/rejected_records.csv",
        index=False
    )

    print("\nINGESTION REPORT")
    print("-------------------------")
    print(f"Input file:       {path}")
    print(f"Total records:    {len(df)}")
    print(f"Valid records:    {len(valid)}")
    print(f"Rejected records: {len(rejected)}")
    print(f"Database:         {database}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Input CSV, JSON or XML file")
    args = parser.parse_args()

    ingest(args.file)
